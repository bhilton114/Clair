# FILE: test_learning_promote.py
# Clair v2 – Learning + Promotion Harness (COUNT(*) FIX)
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from typing import Any, Dict, List, Tuple, Optional

import config
from memory.working_memory import WorkingMemory


def _count_wm(wm: WorkingMemory) -> Dict[str, Any]:
    return {
        "wm_buffer": len(getattr(wm, "buffer", []) or []),
        "wm_types": len(getattr(wm, "type_index", {}) or {}),
    }


def _guess_ltm_db_path(wm: WorkingMemory) -> Optional[str]:
    lt = getattr(wm, "long_term", None)
    if lt is None:
        return None

    # Common attribute names
    for attr in ("db_path", "path", "sqlite_path", "db_file"):
        p = getattr(lt, attr, None)
        if isinstance(p, str) and p.strip() and os.path.exists(p):
            return p

    # Common config names (best effort)
    for k in (
        "LONG_TERM_DB_PATH",
        "LTM_DB_PATH",
        "LTM_PATH",
        "LONG_TERM_PATH",
        "MEMORY_DB_PATH",
        "DB_PATH",
    ):
        p = getattr(config, k, None)
        if isinstance(p, str) and p.strip() and os.path.exists(p):
            return p

    return None


def _count_ltm_total_sqlite(db_path: str) -> Optional[int]:
    """
    Count all rows in the LTM sqlite table.
    We don't assume table name; we sniff.
    """
    try:
        con = sqlite3.connect(db_path)
        cur = con.cursor()

        # Find a likely memory table
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall() if r and isinstance(r[0], str)]

        # common guesses first
        candidates = [t for t in tables if t.lower() in ("memories", "memory", "ltm", "long_term_memory")]
        if not candidates:
            # fallback: anything that looks like a memory table
            for t in tables:
                tl = t.lower()
                if "memor" in tl or "ltm" in tl:
                    candidates.append(t)

        if not candidates:
            return None

        # Use first candidate
        table = candidates[0]
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        n = cur.fetchone()[0]
        con.close()
        return int(n)
    except Exception:
        try:
            con.close()
        except Exception:
            pass
        return None


def _count_ltm(wm: WorkingMemory) -> int:
    lt = getattr(wm, "long_term", None)
    if lt is None:
        return 0

    db_path = _guess_ltm_db_path(wm)
    if db_path:
        n = _count_ltm_total_sqlite(db_path)
        if isinstance(n, int):
            return n

    # Fallback: try retrieve with a big limit if supported.
    # (Your retrieve() seems to default to 50.)
    try:
        # try common "limit" params without knowing signature
        for kwargs in ({"limit": 100000}, {"count": 100000}, {"n": 100000}, {}):
            try:
                items = lt.retrieve(**kwargs)  # type: ignore
                if items is not None:
                    return len(items or [])
            except TypeError:
                continue
    except Exception:
        pass

    return 0


def _find_wm_entries_by_content(wm: WorkingMemory, content_exact: str) -> List[Dict[str, Any]]:
    needle = (content_exact or "").strip()
    if not needle:
        return []
    out = []
    for m in getattr(wm, "buffer", []) or []:
        if isinstance(m, dict) and (m.get("content") or "").strip() == needle:
            out.append(m)
    return out


def _force_eligible_for_promotion(wm: WorkingMemory, entry: Dict[str, Any], *, force_age_min: float) -> None:
    now = time.time()
    conf_thresh = float(getattr(wm, "PROMOTION_CONFIDENCE", 0.95))
    reinf_thresh = int(getattr(wm, "PROMOTION_REINFORCEMENTS", 3))

    entry["timestamp"] = now - float(force_age_min) * 60.0 - 10.0
    entry["confidence"] = max(float(entry.get("confidence", 0.9)), conf_thresh)
    entry["reinforcement_count"] = max(int(entry.get("reinforcement_count", 0)), reinf_thresh)
    entry["persisted"] = False
    entry["age"] = float(force_age_min) + 0.2


def _safe_learning_items(n: int) -> List[Tuple[str, Dict[str, Any]]]:
    base = [
        ("When uncertain, prefer deferring a claim over guessing, and seek verification.", {"type": "policy", "domain": "general", "tags": ["calibration", "epistemic"]}),
        ("Timeboxing helps prevent runaway loops: set a short timer and stop when it ends.", {"type": "lesson", "domain": "general", "tags": ["planning", "timeboxing"]}),
        ("If a task feels too large, shrink it into the smallest next physical step.", {"type": "lesson", "domain": "general", "tags": ["planning", "execution"]}),
        ("Write decisions down immediately or they will evaporate later.", {"type": "lesson", "domain": "general", "tags": ["working_memory", "process"]}),
        ("In emergencies, prioritize: safety first, then communication, then recovery.", {"type": "lesson", "domain": "survival", "tags": ["survival", "priority"]}),
        ("If two memories conflict, keep both but lower confidence until verified.", {"type": "policy", "domain": "general", "tags": ["memory", "conflict"]}),
    ]
    out: List[Tuple[str, Dict[str, Any]]] = []
    i = 0
    while len(out) < max(1, n):
        c, meta = base[i % len(base)]
        if i >= len(base):
            c = f"{c} (variant {i+1})"
        out.append((c, dict(meta)))
        i += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_items", type=int, default=3)
    ap.add_argument("--reinforce", type=int, default=3)
    ap.add_argument("--force_age_min", type=float, default=None)
    ap.add_argument("--out", type=str, default="learning_promotion_report.json")
    args = ap.parse_args()

    t0 = time.time()
    wm = WorkingMemory(preload_long_term=True)

    counts_pre = {"wm": _count_wm(wm), "ltm_total": _count_ltm(wm)}

    configured_min_age = float(getattr(config, "WM_PROMOTION_MIN_AGE", 2))
    force_age_min = float(args.force_age_min) if args.force_age_min is not None else (configured_min_age + 0.1)

    n_items = max(1, int(args.n_items))
    reinforce_passes = max(1, int(args.reinforce))

    injected: List[str] = []
    for content, meta in _safe_learning_items(n_items):
        injected.append(content)
        msg = {
            "type": meta.get("type", "lesson"),
            "content": content,
            "confidence": 0.90,
            "weight": 1.0,
            "domain": meta.get("domain", "general"),
            "tags": meta.get("tags", []),
            "details": {"source": "learning_harness", "harness": "promotion"},
        }

        for _ in range(reinforce_passes):
            wm.store([msg])

        matches = _find_wm_entries_by_content(wm, content)
        for m in matches:
            _force_eligible_for_promotion(wm, m, force_age_min=force_age_min)

    wm.reflect()
    wm.reflect()

    counts_post = {"wm": _count_wm(wm), "ltm_total": _count_ltm(wm)}
    promoted_delta = int(counts_post["ltm_total"]) - int(counts_pre["ltm_total"])

    report: Dict[str, Any] = {
        "seconds": round(time.time() - t0, 3),
        "counts_pre": counts_pre,
        "counts_post": counts_post,
        "promoted_delta": promoted_delta,
        "expected_promotions_upper_bound": min(n_items, int(getattr(config, "WM_MAX_PROMOTIONS_PER_CYCLE", 2))),
        "test_params": {
            "n_items": n_items,
            "reinforce_passes": reinforce_passes,
            "force_age_min_minutes": force_age_min,
            "configured_min_age_minutes": configured_min_age,
        },
        "injected_contents": injected,
        "ltm_db_path_guess": _guess_ltm_db_path(wm),
    }

    out_path = os.path.join(os.getcwd(), args.out)
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    print("\n=== LEARNING + PROMOTION HARNESS: SUMMARY ===")
    print(f"Total seconds: {report['seconds']}")
    print(f"Counts pre:  {counts_pre}")
    print(f"Counts post: {counts_post}")
    print(f"Promoted delta (LTM): {promoted_delta}")
    print(f"Expected upper bound (WM_MAX_PROMOTIONS_PER_CYCLE): {report['expected_promotions_upper_bound']}")
    print(f"LTM db path guess: {report['ltm_db_path_guess']}")
    print(f"Wrote report: {out_path}")
    print("=== DONE ===\n")

    # Now the pass/fail makes sense.
    return 0 if promoted_delta > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
