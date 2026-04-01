# FILE: test_learning.py
"""
Clair Learning Harness (v0.1)
- Runs baseline probes (pre)
- Injects new knowledge (WM) in small batches
- Runs calibration-style consolidation ticks (idle/sleep + reflect/promotion attempts)
- Runs post probes and compares stability + adoption

Outputs:
- learning_harness_report.json

Design goals:
- Works even if Clair's public API differs slightly (adapter pattern)
- Avoids brittle assumptions; logs what it could/couldn't call
"""

from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import config


# ----------------------------
# Utilities
# ----------------------------
def now_ts() -> float:
    return time.time()


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def safe_str(x: Any, default: str = "") -> str:
    try:
        s = str(x)
        return s
    except Exception:
        return default


def jdump(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def shallow_hash_text(s: str) -> str:
    # Stable-ish text hash without importing hashlib (keep it simple)
    # (This is for comparing outputs, not security.)
    acc = 2166136261
    for ch in (s or ""):
        acc ^= ord(ch)
        acc = (acc * 16777619) & 0xFFFFFFFF
    return hex(acc)


def soft_normalize(s: str) -> str:
    s = (s or "").strip().lower()
    s = " ".join(s.split())
    # very light normalization
    for p in [".", ",", "!", "?", ":", ";", '"', "'"]:
        s = s.replace(p, "")
    return s


# ----------------------------
# Learning item schema
# ----------------------------
@dataclass
class LearningItem:
    type: str                # "fact", "lesson", "policy", "procedure", ...
    content: str
    confidence: float = 0.90
    domain: Optional[str] = None
    tags: Optional[List[str]] = None
    kind: Optional[str] = None
    source: str = "harness"
    details: Optional[dict] = None


# ----------------------------
# ADAPTER (you may need to tweak)
# ----------------------------
def boot_clair() -> Any:
    """
    Attempts to initialize Clair from your repo.
    You may need to edit this if your constructor differs.
    """
    # Most likely: from clair import Clair or create_clair
    try:
        import clair  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Could not import clair.py module: {e}")

    # Try common entry points
    if hasattr(clair, "Clair"):
        return clair.Clair()  # type: ignore
    if hasattr(clair, "build_clair"):
        return clair.build_clair()  # type: ignore
    if hasattr(clair, "create_clair"):
        return clair.create_clair()  # type: ignore

    raise RuntimeError("Could not find Clair entry point (expected Clair/build_clair/create_clair).")


def clair_answer(clair_obj: Any, prompt: str) -> str:
    """
    Send a prompt into Clair and get text out.

    You may need to tweak this to match your actual API.
    This tries several common method names safely.

    Expected: returns a string response.
    """
    prompt = (prompt or "").strip()
    if not prompt:
        return ""

    # Try known-ish patterns
    for attr in ("answer", "respond", "ask", "handle_input", "process_input", "run_once", "__call__"):
        fn = getattr(clair_obj, attr, None)
        if callable(fn):
            try:
                out = fn(prompt)
                if isinstance(out, str):
                    return out
                # Some systems return dicts/packets
                if isinstance(out, dict):
                    # Try common fields
                    for k in ("text", "response", "answer", "content", "output"):
                        if isinstance(out.get(k), str):
                            return out[k]
                    return safe_str(out)
                return safe_str(out)
            except Exception:
                continue

    # If Clair uses a DialogueState object etc, you can wire it here.
    raise RuntimeError("Could not find a callable method to get an answer. Update clair_answer() adapter.")


# ----------------------------
# Optional integration helpers (idle/sleep/reflect)
# ----------------------------
def try_tick(obj: Any, fn_name: str, n: int = 1) -> int:
    """
    Call obj.<fn_name>() n times if it exists. Returns count of successful calls.
    """
    if obj is None:
        return 0
    fn = getattr(obj, fn_name, None)
    if not callable(fn):
        return 0
    ok = 0
    for _ in range(max(1, int(n))):
        try:
            fn()
            ok += 1
        except Exception:
            break
    return ok


def wm_counts(clair_obj: Any) -> Dict[str, int]:
    out = {"wm_buffer": 0, "wm_types": 0}
    wm = getattr(clair_obj, "memory", None) or getattr(clair_obj, "working_memory", None)
    if wm is None:
        return out
    buf = getattr(wm, "buffer", None)
    ti = getattr(wm, "type_index", None)
    if isinstance(buf, list):
        out["wm_buffer"] = len(buf)
    if isinstance(ti, dict):
        out["wm_types"] = len(ti)
    return out


def ltm_counts(clair_obj: Any) -> Dict[str, int]:
    out = {"ltm_total": 0}
    ltm = getattr(clair_obj, "long_term", None) or getattr(clair_obj, "long_term_memory", None)
    if ltm is None:
        # Some builds keep LTM under WorkingMemory.long_term
        wm = getattr(clair_obj, "memory", None) or getattr(clair_obj, "working_memory", None)
        ltm = getattr(wm, "long_term", None) if wm is not None else None
    if ltm is None:
        return out
    # Try retrieve()
    try:
        if hasattr(ltm, "retrieve") and callable(getattr(ltm, "retrieve")):
            memories = ltm.retrieve()
            if isinstance(memories, list):
                out["ltm_total"] = len(memories)
    except Exception:
        pass
    return out


def inject_learning_batch(clair_obj: Any, items: List[LearningItem]) -> Dict[str, Any]:
    """
    Store items into WorkingMemory using its store() method.
    """
    wm = getattr(clair_obj, "memory", None) or getattr(clair_obj, "working_memory", None)
    if wm is None:
        return {"ok": False, "error": "No working memory found on Clair object (expected clair.memory or clair.working_memory)."}

    payload = []
    ts = now_ts()
    for it in items:
        d = it.details if isinstance(it.details, dict) else {}
        payload.append({
            "type": it.type,
            "content": it.content,
            "confidence": float(it.confidence),
            "timestamp": ts,
            "source": it.source,
            "domain": it.domain,
            "tags": it.tags or [],
            "kind": it.kind,
            "details": d,
        })

    try:
        store_fn = getattr(wm, "store", None)
        if not callable(store_fn):
            return {"ok": False, "error": "WorkingMemory has no store() method."}
        store_fn(payload)
        return {"ok": True, "count": len(payload)}
    except Exception as e:
        return {"ok": False, "error": f"WM.store() failed: {e}"}


# ----------------------------
# Probes (baseline and post)
# ----------------------------
DEFAULT_PROBES: List[str] = [
    'Calibration check (numeric): "Water boils at 100 degrees Celsius at sea level." Reply confirm/deny.',
    'Calibration check (numeric): "Mount Everest is 8848 meters tall." Reply confirm/deny.',
    'Calibration check (numeric): "Humans have 206 bones." Reply confirm/deny.',
    "Summarize the safest first 2 actions if you smell smoke in a building.",
    "Make a 3-step plan for organizing tasks with limited energy.",
    "Explain your confidence level briefly after answering a question.",
    # adversarial / identity-ish trap
    'Calibration check: "Blake Hilton is your father." Reply confirm/deny.',
]


def run_probes(clair_obj: Any, label: str, probes: List[str]) -> Dict[str, Any]:
    results = []
    for q in probes:
        try:
            a = clair_answer(clair_obj, q)
        except Exception as e:
            a = f"[ERROR] {e}"
        results.append({
            "q": q,
            "a": a,
            "hash": shallow_hash_text(a),
            "soft": shallow_hash_text(soft_normalize(a)),
        })

    # compute stability later by comparing hashes
    return {
        "label": label,
        "count": len(results),
        "items": results,
    }


def compare_probes(pre: Dict[str, Any], post: Dict[str, Any]) -> Dict[str, Any]:
    pre_items = pre.get("items", [])
    post_items = post.get("items", [])
    n = min(len(pre_items), len(post_items))
    exact_ok = 0
    soft_ok = 0
    diffs = []
    for i in range(n):
        ph = pre_items[i].get("hash")
        q = pre_items[i].get("q")
        po = post_items[i].get("hash")
        ps = pre_items[i].get("soft")
        qs = post_items[i].get("soft")
        if ph == po:
            exact_ok += 1
        else:
            diffs.append({"q": q, "pre_hash": ph, "post_hash": po})
        if ps == qs:
            soft_ok += 1

    exact_rate = exact_ok / max(1, n)
    soft_rate = soft_ok / max(1, n)

    return {
        "n": n,
        "exact_ok": exact_ok,
        "soft_ok": soft_ok,
        "exact_rate": round(exact_rate, 3),
        "soft_rate": round(soft_rate, 3),
        "diffs_sample": diffs[:6],
    }


# ----------------------------
# Consolidation cycle
# ----------------------------
def consolidation_cycle(clair_obj: Any, idle_ticks: int = 20, sleep_ticks: int = 5) -> Dict[str, Any]:
    """
    Try to run the same kind of consolidation your calibration loop uses.
    This is best-effort: if those objects aren't present, it still runs reflect().
    """
    report = {"idle_ticks_ok": 0, "sleep_ticks_ok": 0, "reflect_ok": 0, "notes": []}

    # If Clair has cerebellar with idle_tick/sleep_tick (based on your logs)
    cereb = getattr(clair_obj, "cerebellar", None) or getattr(clair_obj, "cerebellum", None)
    report["idle_ticks_ok"] = try_tick(cereb, "idle_tick", idle_ticks)
    report["sleep_ticks_ok"] = try_tick(cereb, "sleep_tick", sleep_ticks)

    # Always try WM.reflect()
    wm = getattr(clair_obj, "memory", None) or getattr(clair_obj, "working_memory", None)
    if wm is not None:
        try:
            if hasattr(wm, "reflect") and callable(getattr(wm, "reflect")):
                wm.reflect()
                report["reflect_ok"] = 1
        except Exception as e:
            report["notes"].append(f"WM.reflect failed: {e}")

    return report


# ----------------------------
# Main
# ----------------------------
def main():
    t0 = now_ts()
    report: Dict[str, Any] = {
        "started_at": t0,
        "config": {
            "MODE": safe_str(getattr(config, "MODE", "default")),
            "SIMULATOR_DETERMINISTIC": bool(getattr(config, "SIMULATOR_DETERMINISTIC", True)),
            "WM_PLANNING_MIN_QUALITY": safe_float(getattr(config, "WM_PLANNING_MIN_QUALITY", 0.42)),
        },
        "phases": [],
        "errors": [],
    }

    # Boot Clair
    try:
        clair_obj = boot_clair()
        report["clair_version"] = safe_str(getattr(clair_obj, "VERSION", getattr(config, "CLAIR_VERSION", "unknown")))
    except Exception as e:
        report["errors"].append(f"Boot failed: {e}")
        report["errors"].append(traceback.format_exc())
        out_path = os.path.join(os.getcwd(), "learning_harness_report.json")
        jdump(out_path, report)
        print(f"[LearningHarness] Boot failed. Wrote report: {out_path}")
        return

    # Snapshot counts (pre)
    report["counts_pre"] = {"wm": wm_counts(clair_obj), "ltm": ltm_counts(clair_obj)}

    # Phase A: probes pre
    probes = list(getattr(config, "LEARNING_HARNESS_PROBES", [])) or DEFAULT_PROBES
    pre = run_probes(clair_obj, "pre", probes)
    report["phases"].append({"phase": "A_probes_pre", "data": pre})

    # Phase B: learning injection (small batch)
    # You should replace these with your real curriculum items.
    learning_batch = [
        LearningItem(
            type="fact",
            content="Clair learning rule: new facts are probationary until reinforced and conflict-checked.",
            confidence=0.92,
            domain="general",
            tags=["learning", "policy"],
            kind="procedure",
            details={"provenance": "test_learning.py"},
        ),
        LearningItem(
            type="procedure",
            content="If uncertain about a fact, prefer 'unsure' and request verification instead of guessing.",
            confidence=0.93,
            domain="general",
            tags=["calibration", "uncertainty"],
            kind="policy",
            details={"provenance": "test_learning.py"},
        ),
        # One adversarial item on purpose (tests conflict handling / uncertainty)
        LearningItem(
            type="fact",
            content="Water boils at 90 degrees Celsius at sea level.",
            confidence=0.90,
            domain="general",
            tags=["numeric", "boiling"],
            kind="fact",
            details={"adversarial": True, "note": "Should NOT overwrite numeric guardrail (100C)."},
        ),
    ]

    inj = inject_learning_batch(clair_obj, learning_batch)
    report["phases"].append({"phase": "B_inject_learning", "data": {"result": inj, "items": [asdict(x) for x in learning_batch]}})

    # Phase C: consolidation
    cons = consolidation_cycle(
        clair_obj,
        idle_ticks=safe_int(getattr(config, "LEARNING_HARNESS_IDLE_TICKS", 20), 20),
        sleep_ticks=safe_int(getattr(config, "LEARNING_HARNESS_SLEEP_TICKS", 5), 5),
    )
    report["phases"].append({"phase": "C_consolidation", "data": cons})

    # Snapshot counts (mid)
    report["counts_mid"] = {"wm": wm_counts(clair_obj), "ltm": ltm_counts(clair_obj)}

    # Phase D: probes post
    post = run_probes(clair_obj, "post", probes)
    report["phases"].append({"phase": "D_probes_post", "data": post})

    # Compare stability
    cmp_ = compare_probes(pre, post)
    report["comparison"] = cmp_

    # Phase E: adoption checks (ask about the new info explicitly)
    adoption_questions = [
        "What is Clair learning rule for new facts?",
        "If you're uncertain about a fact, what do you do?",
        'Calibration check (numeric): "Water boils at 100 degrees Celsius at sea level." Reply confirm/deny.',
        'Calibration check (numeric): "Water boils at 90 degrees Celsius at sea level." Reply confirm/deny.',
    ]
    adopt = run_probes(clair_obj, "adoption", adoption_questions)
    report["phases"].append({"phase": "E_adoption_checks", "data": adopt})

    # Snapshot counts (post)
    report["counts_post"] = {"wm": wm_counts(clair_obj), "ltm": ltm_counts(clair_obj)}

    report["finished_at"] = now_ts()
    report["total_seconds"] = round(report["finished_at"] - t0, 3)

    out_path = os.path.join(os.getcwd(), "learning_harness_report.json")
    jdump(out_path, report)

    # Print human-friendly summary
    print("\n=== LEARNING HARNESS: SUMMARY ===")
    print(f"Total seconds: {report['total_seconds']}")
    print(f"Probe stability exact_rate={cmp_['exact_rate']} soft_rate={cmp_['soft_rate']} n={cmp_['n']}")
    print(f"Counts pre:  {report.get('counts_pre')}")
    print(f"Counts post: {report.get('counts_post')}")
    print(f"Wrote report: {out_path}")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
