"""
Integrated Three-Loop Harness (v11.0)

Tests ALL THREE LOOPS end-to-end:
  1) Reasoning Loop (Clair response loop): repeated prompts -> output fingerprints -> drift score
  2) Simulator Loop (planning/simulator): options generated -> planned_next checks -> determinism checks
  3) Calibration Loop (ACC + Cerebellar): idle Q/A + sleep maintenance

Workflow:
  Phase A: Reasoning PRE
  Phase B: Simulator PRE
  Phase C: Calibration (Idle + Sleep)
  Phase D: Simulator POST
  Phase E: Reasoning POST
  Phase F: Optional benchmark (test_clair.py)

Outputs:
  - console summary
  - JSON report written to calibration_three_loop_report.json

v11.0 updates:
  - supports both dict rows and MemoryRecord rows in WM / memory views
  - hardens conflict-row injection against WM contract-backed storage
  - prefers wm.store_fallback(...) when available for exact conflict rows
  - verifies requested ids against live WM, exported memory views, and text fallback
  - keeps reporting structure close to v10.4 for compatibility
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import random
import re
import subprocess
import sys
import time
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


# =============================================================================
# Helpers
# =============================================================================
def _now() -> float:
    return time.time()


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return str(obj)


def _write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8", errors="replace") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def _ensure_repo_root_on_path() -> str:
    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def _import_any(module_names: List[str]):
    last_err = None
    for m in module_names:
        try:
            return importlib.import_module(m)
        except Exception as e:
            last_err = e
    raise ImportError(f"Failed to import any of: {module_names}. Last error: {last_err}")


def _get_first_attr(mod: Any, names: List[str]) -> Any:
    for n in names:
        try:
            v = getattr(mod, n, None)
        except Exception:
            v = None
        if v is not None:
            return v
    return None


def _construct_any(cls: Any, arg_sets: List[Tuple[Any, ...]]) -> Any:
    last_err = None
    for args in arg_sets:
        try:
            return cls(*args)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to construct {getattr(cls, '__name__', cls)}. Last error: {last_err}")


def _call_first(obj: Any, method_names: List[str], *args, **kwargs) -> Tuple[bool, Any, str]:
    for name in method_names:
        fn = getattr(obj, name, None)
        if callable(fn):
            try:
                return True, fn(*args, **kwargs), name
            except TypeError:
                try:
                    return True, fn(*args), name
                except Exception as e:
                    return False, e, name
            except Exception as e:
                return False, e, name
    return False, None, ""


def _norm_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _fingerprint_text(s: str) -> str:
    s2 = _norm_text(s)
    return hashlib.sha1(s2.encode("utf-8", errors="ignore")).hexdigest()[:12]


def _run_cmd(cmd_str: str, root: str) -> Dict[str, Any]:
    cmd = cmd_str.strip().split()
    t0 = _now()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        out = {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "seconds": round(_now() - t0, 3),
        }
        try:
            with open(os.path.join(root, "benchmark_stdout.txt"), "w", encoding="utf-8", errors="replace") as f:
                f.write(proc.stdout or "")
            with open(os.path.join(root, "benchmark_stderr.txt"), "w", encoding="utf-8", errors="replace") as f:
                f.write(proc.stderr or "")
        except Exception:
            pass
        return out
    except Exception as e:
        return {"error": str(e), "seconds": round(_now() - t0, 3)}


# =============================================================================
# Generic dict-like / MemoryRecord compatibility
# =============================================================================
def _is_row_like(obj: Any) -> bool:
    if isinstance(obj, dict):
        return True
    return hasattr(obj, "get") and callable(getattr(obj, "get", None))


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    getter = getattr(row, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except Exception:
            return default
    return default


def _row_set(row: Any, key: str, value: Any) -> bool:
    if isinstance(row, dict):
        row[key] = value
        return True

    try:
        if hasattr(row, key):
            setattr(row, key, value)
            return True
    except Exception:
        pass

    try:
        meta = getattr(row, "metadata", None)
        if isinstance(meta, dict):
            meta[key] = value
            return True
    except Exception:
        pass

    return False


def _row_to_export_dict(row: Any) -> Dict[str, Any]:
    if isinstance(row, dict):
        return row

    out: Dict[str, Any] = {}
    for key in (
        "id",
        "memory_id",
        "type",
        "content",
        "claim",
        "summary",
        "confidence",
        "domain",
        "kind",
        "status",
        "verification_status",
        "details",
        "tags",
        "evidence",
        "context",
        "timestamp",
        "persisted",
        "weight",
        "content_obj",
        "source",
        "hazard_family",
        "conflict",
        "last_verified",
        "memory_class",
        "pending_verification",
        "verified",
        "contested",
        "source_trust",
        "staleness_risk",
        "reinforcement_count",
        "revision_trace",
        "numeric_guarded",
        "last_reinforced",
        "keywords",
        "quality",
        "evidence_strength",
    ):
        out[key] = _row_get(row, key, None)
    return out


# =============================================================================
# Clair shims
# =============================================================================
def _attach_clair_shims(clair: Any) -> None:
    def _get_all_memories(
        include_wm: bool = True,
        include_ltm: bool = True,
        limit: int = 250,
        msg_type: Optional[str] = None,
        **_kw: Any,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []

        if include_wm:
            try:
                buf = getattr(getattr(clair, "memory", None), "buffer", None)
                if isinstance(buf, list):
                    for m in buf:
                        if _is_row_like(m):
                            out.append(_row_to_export_dict(m))
            except Exception:
                pass

        if include_ltm:
            try:
                ltm = getattr(clair, "long_term", None)
                if ltm is not None and hasattr(ltm, "retrieve"):
                    rows = None
                    try:
                        rows = ltm.retrieve(msg_type=msg_type, limit=max(50, int(limit)))
                    except TypeError:
                        try:
                            rows = ltm.retrieve(msg_type, max(50, int(limit)))
                        except Exception:
                            rows = None

                    if isinstance(rows, list):
                        for r in rows:
                            if isinstance(r, dict):
                                out.append(r)
            except Exception:
                pass

        seen = set()
        deduped: List[Dict[str, Any]] = []
        for m in out:
            try:
                k = (
                    str(m.get("id") or m.get("memory_id") or ""),
                    str(m.get("type") or ""),
                    str(m.get("content") or m.get("claim") or ""),
                    str(m.get("domain") or ""),
                    str(m.get("kind") or ""),
                )
            except Exception:
                k = (str(m), "", "", "", "")
            if k in seen:
                continue
            seen.add(k)
            deduped.append(m)

        return deduped[: max(1, int(limit))]

    for name in ("get_all_memories", "get_memories", "all_memories"):
        if not callable(getattr(clair, name, None)):
            setattr(clair, name, _get_all_memories)


# =============================================================================
# Optional: NO-WRITE mode
# =============================================================================
def _patch_ltm_no_write(clair: Any) -> Dict[str, Any]:
    ltm = getattr(clair, "long_term", None)
    if ltm is None:
        return {"ok": False, "patched": [], "note": "clair.long_term missing"}

    write_like = [
        "store", "add", "insert", "save", "upsert", "write",
        "append", "commit_memory", "remember", "update_memory",
    ]

    patched: List[str] = []

    def _blocked(*_a: Any, **_kw: Any):
        return {"ok": False, "blocked": True, "reason": "calibration_no_write"}

    for name in write_like:
        fn = getattr(ltm, name, None)
        if callable(fn):
            try:
                setattr(ltm, name, _blocked)
                patched.append(name)
            except Exception:
                pass

    return {"ok": bool(patched), "patched": patched, "note": "original methods not stored (json-safe)"}


# =============================================================================
# Memory helpers
# =============================================================================
_MEMID_RE = re.compile(r"\bmem_[a-z0-9]{6,64}\b", re.IGNORECASE)
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^a-z0-9\s]+")
_CONFLICT_KIND_RE = re.compile(r"conflict", re.IGNORECASE)


def _mem_text(m: Any) -> str:
    t = _row_get(m, "claim")
    if isinstance(t, str) and t.strip():
        return t.strip()
    t2 = _row_get(m, "content")
    if isinstance(t2, str) and t2.strip():
        return t2.strip()
    return ""


def _mem_id_any(m: Any) -> Optional[str]:
    for k in ("id", "memory_id", "uid", "pk"):
        v = _row_get(m, k)
        if isinstance(v, str) and v.strip():
            return v.strip().lower()
    return None


def _cereb_style_mem_id(m: Any) -> Optional[str]:
    try:
        t = str(_row_get(m, "type") or "")
        dom = str(_row_get(m, "domain") or "")
        txt = _mem_text(m)
        if not txt:
            return None
        base = f"{t}|{dom}|{txt}"
        h = hashlib.sha1(base.encode("utf-8", errors="ignore")).hexdigest()[:12]
        return f"mem_{h}".lower()
    except Exception:
        return None


def _mem_summary(m: Optional[Any]) -> Dict[str, Any]:
    if m is None or not _is_row_like(m):
        return {}
    return {
        "id": _mem_id_any(m) or _cereb_style_mem_id(m),
        "type": _row_get(m, "type"),
        "domain": _row_get(m, "domain"),
        "kind": _row_get(m, "kind"),
        "text": _mem_text(m),
    }


def _details_dict(m: Any) -> Dict[str, Any]:
    d = _row_get(m, "details")
    if isinstance(d, dict):
        return d

    if isinstance(m, dict):
        d = {}
        m["details"] = d
        return d

    try:
        meta = getattr(m, "metadata", None)
        if isinstance(meta, dict):
            d2 = meta.get("details")
            if isinstance(d2, dict):
                return d2
            d2 = {}
            meta["details"] = d2
            return d2
    except Exception:
        pass

    return {}


def _set_details_value(m: Any, key: str, value: Any) -> None:
    d = _details_dict(m)
    d[key] = value


def _make_mem_id(mem_type: str, domain: str, text: str) -> str:
    base = f"{mem_type}|{domain}|{text}"
    h = hashlib.sha1(base.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"mem_{h}".lower()


def _wm_buffer_rows(clair: Any) -> List[Any]:
    try:
        wm = getattr(clair, "memory", None)
        buf = getattr(wm, "buffer", None)
        if isinstance(buf, list):
            return [r for r in buf if _is_row_like(r)]
    except Exception:
        pass
    return []


def _all_memory_rows(clair: Any, *, lookup_limit: int = 800) -> List[Any]:
    out: List[Any] = []

    wm_rows = _wm_buffer_rows(clair)
    if wm_rows:
        out.extend(wm_rows[: max(1, int(lookup_limit))])

    fn = getattr(clair, "get_all_memories", None)
    if callable(fn):
        try:
            rows = fn(limit=int(lookup_limit))
            if isinstance(rows, list):
                out.extend([r for r in rows if _is_row_like(r)])
        except Exception:
            pass

    seen_obj = set()
    deduped: List[Any] = []
    for row in out:
        ident = id(row)
        if ident in seen_obj:
            continue
        seen_obj.add(ident)
        deduped.append(row)

    return deduped[: max(1, int(lookup_limit))]


def _build_mem_index(clair: Any, *, lookup_limit: int = 800) -> Dict[str, Any]:
    idx: Dict[str, Any] = {}
    for m in _all_memory_rows(clair, lookup_limit=lookup_limit):
        mid = _mem_id_any(m)
        if mid:
            idx[mid] = m

        sid = _cereb_style_mem_id(m)
        if sid:
            idx[sid] = m

        txt = _mem_text(m)
        if txt:
            idx[f"text::{_fingerprint_text(txt)}"] = m
    return idx


def _find_memory_by_exact_text(clair: Any, text: str, *, lookup_limit: int = 800) -> Optional[Any]:
    target = _norm_text(text)
    if not target:
        return None

    for row in _wm_buffer_rows(clair):
        if _norm_text(_mem_text(row)) == target:
            return row

    for row in _all_memory_rows(clair, lookup_limit=lookup_limit):
        if _norm_text(_mem_text(row)) == target:
            return row
    return None


def _find_memory_by_exact_id_live_wm(clair: Any, mem_id: str) -> Optional[Any]:
    target = str(mem_id or "").strip().lower()
    if not target:
        return None

    for row in _wm_buffer_rows(clair):
        rid = _mem_id_any(row)
        sid = _cereb_style_mem_id(row)
        if target == rid or target == sid:
            return row
    return None


def _find_memory_by_exact_id(clair: Any, mem_id: str, *, lookup_limit: int = 800) -> Optional[Any]:
    target = str(mem_id or "").strip().lower()
    if not target:
        return None

    live = _find_memory_by_exact_id_live_wm(clair, target)
    if live is not None:
        return live

    for row in _all_memory_rows(clair, lookup_limit=lookup_limit):
        rid = _mem_id_any(row)
        sid = _cereb_style_mem_id(row)
        if target == rid or target == sid:
            return row
    return None


def _find_memory_by_text_fallback(clair: Any, target_text: str, *, lookup_limit: int = 800) -> Optional[Any]:
    target = _norm_text(target_text)
    if not target:
        return None

    best: Optional[Any] = None
    best_score = 0.0

    for row in _all_memory_rows(clair, lookup_limit=lookup_limit):
        txt = _norm_text(_mem_text(row))
        if not txt:
            continue
        if txt == target:
            return row

        if target in txt or txt in target:
            score = min(len(target), len(txt)) / max(1, max(len(target), len(txt)))
            if score > best_score:
                best = row
                best_score = score

        sim = SequenceMatcher(None, target, txt).ratio()
        if sim > best_score:
            best = row
            best_score = sim

        if _fingerprint_text(txt) == _fingerprint_text(target):
            return row

    return best if best_score >= 0.72 else None


def _resolve_live_memory_row(
    clair: Any,
    requested_id: str,
    text: str,
    *,
    lookup_limit: int = 800,
) -> Optional[Any]:
    rid = str(requested_id or "").strip().lower()
    if rid:
        hit = _find_memory_by_exact_id(clair, rid, lookup_limit=lookup_limit)
        if hit is not None:
            return hit

    hit = _find_memory_by_exact_text(clair, text, lookup_limit=lookup_limit)
    if hit is not None:
        return hit

    hit = _find_memory_by_text_fallback(clair, text, lookup_limit=lookup_limit)
    if hit is not None:
        return hit

    expected_sid = _make_mem_id("lesson", "general", text)
    if expected_sid:
        hit = _find_memory_by_exact_id(clair, expected_sid, lookup_limit=lookup_limit)
        if hit is not None:
            return hit

    return None


def _resolve_live_memory_row_strict(
    clair: Any,
    requested_id: str,
    text: str,
    *,
    lookup_limit: int = 800,
) -> Optional[Any]:
    rid = str(requested_id or "").strip().lower()
    if rid:
        hit = _find_memory_by_exact_id_live_wm(clair, rid)
        if hit is not None:
            return hit

        hit = _find_memory_by_exact_id(clair, rid, lookup_limit=lookup_limit)
        if hit is not None:
            return hit

    if text:
        hit = _find_memory_by_exact_text(clair, text, lookup_limit=lookup_limit)
        if hit is not None:
            return hit

    return None


def _force_memory_identity(row: Any, requested_id: str) -> None:
    rid = str(requested_id or "").strip().lower()
    if not rid or row is None:
        return

    _row_set(row, "id", rid)
    _row_set(row, "memory_id", rid)
    details = _details_dict(row)
    details.setdefault("explicit_id", rid)
    details.setdefault("original_id", rid)


def _upsert_conflict_metadata(
    row: Any,
    *,
    self_id: str,
    other_id: str,
    self_text: str,
    other_text: str,
    now_ts: float,
) -> None:
    _row_set(row, "id", self_id)
    _row_set(row, "memory_id", self_id)
    _row_set(row, "claim", _mem_text(row) or self_text)
    _row_set(row, "content", _mem_text(row) or self_text)
    _row_set(row, "conflict", True)
    _row_set(row, "last_verified", now_ts)

    details = _details_dict(row)
    details["verified"] = False
    details["pending_verification"] = True
    details["contested"] = True
    details["conflict"] = True
    details["superseded"] = False
    details["recall_blocked"] = False
    details["status"] = "contested"
    details["verification_status"] = "contested"
    details["last_verified"] = now_ts
    details["explicit_id"] = self_id
    details["original_id"] = self_id

    ids = details.setdefault("conflict_with_ids", [])
    if other_id and other_id not in ids:
        ids.append(other_id)

    texts = details.setdefault("conflict_with_text", [])
    if other_text and other_text not in texts:
        texts.append(other_text)

    refs = details.setdefault("conflict_with", [])
    if other_text and other_text not in refs:
        refs.append(other_text)

    if isinstance(row, dict):
        rt = row.setdefault("revision_trace", [])
        if isinstance(rt, list):
            rt.append(f"harness_conflict_injection linked to {other_id}")
    else:
        try:
            meta = getattr(row, "metadata", None)
            if isinstance(meta, dict):
                rt = meta.setdefault("revision_trace", [])
                if isinstance(rt, list):
                    rt.append(f"harness_conflict_injection linked to {other_id}")
        except Exception:
            pass


def _direct_wm_insert_exact(clair: Any, payload: Dict[str, Any], requested_id: str) -> Tuple[bool, str]:
    wm = getattr(clair, "memory", None)
    if wm is None:
        return False, "wm_missing"

    # Best path for your newer WM rewrite.
    fallback_fn = getattr(wm, "store_fallback", None)
    if callable(fallback_fn):
        try:
            fallback_fn(dict(payload))
            return True, "wm.store_fallback"
        except Exception as e:
            return False, f"wm.store_fallback_error:{e}"

    store_fn = getattr(wm, "store", None)
    if callable(store_fn):
        try:
            store_fn(dict(payload))
            return True, "wm.store"
        except Exception as e:
            return False, f"wm.store_error:{e}"

    return False, "no_insert_path"


def _store_conflict_row_exact(
    clair: Any,
    payload: Dict[str, Any],
    requested_id: str,
    text: str,
    *,
    lookup_limit: int = 800,
) -> Dict[str, Any]:
    wm = getattr(clair, "memory", None)
    attempts: List[str] = []
    errors: List[str] = []

    if wm is not None:
        fallback_fn = getattr(wm, "store_fallback", None)
        if callable(fallback_fn):
            try:
                fallback_fn(dict(payload))
                attempts.append("wm.store_fallback")
            except Exception as e:
                errors.append(f"wm.store_fallback:{e}")

            hit = _find_memory_by_exact_id_live_wm(clair, requested_id)
            if hit is None:
                hit = _resolve_live_memory_row_strict(clair, requested_id, text, lookup_limit=lookup_limit)
            if hit is not None:
                _force_memory_identity(hit, requested_id)
                return {"ok": True, "mode": "wm.store_fallback", "row": hit, "attempts": attempts, "errors": errors}

        store_fn = getattr(wm, "store", None)
        if callable(store_fn):
            try:
                store_fn(dict(payload))
                attempts.append("wm.store")
            except Exception as e:
                errors.append(f"wm.store:{e}")

            hit = _find_memory_by_exact_id_live_wm(clair, requested_id)
            if hit is None:
                hit = _resolve_live_memory_row_strict(clair, requested_id, text, lookup_limit=lookup_limit)
            if hit is not None:
                _force_memory_identity(hit, requested_id)
                return {"ok": True, "mode": "wm.store", "row": hit, "attempts": attempts, "errors": errors}

    ok_ins, mode = _direct_wm_insert_exact(clair, payload, requested_id)
    attempts.append(mode)
    if ok_ins:
        hit = _find_memory_by_exact_id_live_wm(clair, requested_id)
        if hit is None:
            hit = _resolve_live_memory_row_strict(clair, requested_id, text, lookup_limit=lookup_limit)
        if hit is not None:
            _force_memory_identity(hit, requested_id)
            return {"ok": True, "mode": mode, "row": hit, "attempts": attempts, "errors": errors}

    exact_text_hit = _find_memory_by_exact_text(clair, text, lookup_limit=lookup_limit)
    if exact_text_hit is not None:
        _force_memory_identity(exact_text_hit, requested_id)
        return {"ok": True, "mode": "exact_text_rebind", "row": exact_text_hit, "attempts": attempts, "errors": errors}

    return {"ok": False, "mode": "failed", "row": None, "attempts": attempts, "errors": errors}


# =============================================================================
# Conflict injection
# =============================================================================
def _inject_conflict_memories(clair: Any, *, lookup_limit: int = 800) -> Dict[str, Any]:
    now_ts = time.time()

    pairs = [
        ("Humans have 206 bones.", "Humans have 208 bones.", "biology"),
        ("Water boils at 100 degrees Celsius at sea level.", "Water boils at 102 degrees Celsius at sea level.", "science"),
    ]

    wm = getattr(clair, "memory", None)
    if wm is None:
        return {
            "ok": False,
            "count": 0,
            "stored_rows": 0,
            "claims": [],
            "ids": [],
            "resolved_ids": [],
            "resolved_map": {},
            "text_targets": [],
            "store_modes": [],
            "errors": ["clair.memory missing"],
        }

    requested_ids: List[str] = []
    resolved_ids: List[str] = []
    text_targets: List[str] = []
    errors: List[str] = []
    store_modes: List[Dict[str, Any]] = []
    stored_rows = 0

    for a_text, b_text, tag in pairs:
        req_a = _make_mem_id("lesson", "general", a_text)
        req_b = _make_mem_id("lesson", "general", b_text)

        requested_ids.extend([req_a, req_b])
        text_targets.extend([a_text, b_text])

        row_a = _resolve_live_memory_row_strict(clair, req_a, a_text, lookup_limit=lookup_limit)
        row_b = _resolve_live_memory_row_strict(clair, req_b, b_text, lookup_limit=lookup_limit)

        if row_a is None:
            exact_a = _store_conflict_row_exact(
                clair,
                {
                    "id": req_a,
                    "memory_id": req_a,
                    "type": "lesson",
                    "content": a_text,
                    "claim": a_text,
                    "source": "harness_conflict_injection",
                    "domain": "general",
                    "tags": [tag, "day23_conflict", "harness_conflict_injection"],
                    "kind": "fact",
                    "confidence": 0.72,
                    "weight": 0.70,
                    "conflict": True,
                    "last_verified": now_ts,
                    "details": {
                        "verified": False,
                        "pending_verification": True,
                        "contested": True,
                        "conflict": True,
                        "superseded": False,
                        "recall_blocked": False,
                        "status": "contested",
                        "verification_status": "contested",
                        "conflict_with_ids": [req_b],
                        "conflict_with_text": [b_text],
                        "conflict_with": [b_text],
                        "explicit_id": req_a,
                        "original_id": req_a,
                        "applied_via": "harness_conflict_injection",
                    },
                },
                req_a,
                a_text,
                lookup_limit=lookup_limit,
            )
            store_modes.append({"id": req_a, "mode": exact_a.get("mode"), "attempts": exact_a.get("attempts", [])})
            if exact_a.get("ok"):
                row_a = exact_a.get("row")
                stored_rows += 1
            else:
                for err in exact_a.get("errors", []) or []:
                    errors.append(f"{req_a}:{err}")

        if row_b is None:
            exact_b = _store_conflict_row_exact(
                clair,
                {
                    "id": req_b,
                    "memory_id": req_b,
                    "type": "lesson",
                    "content": b_text,
                    "claim": b_text,
                    "source": "harness_conflict_injection",
                    "domain": "general",
                    "tags": [tag, "day23_conflict", "harness_conflict_injection"],
                    "kind": "fact",
                    "confidence": 0.72,
                    "weight": 0.70,
                    "conflict": True,
                    "last_verified": now_ts,
                    "details": {
                        "verified": False,
                        "pending_verification": True,
                        "contested": True,
                        "conflict": True,
                        "superseded": False,
                        "recall_blocked": False,
                        "status": "contested",
                        "verification_status": "contested",
                        "conflict_with_ids": [req_a],
                        "conflict_with_text": [a_text],
                        "conflict_with": [a_text],
                        "explicit_id": req_b,
                        "original_id": req_b,
                        "applied_via": "harness_conflict_injection",
                    },
                },
                req_b,
                b_text,
                lookup_limit=lookup_limit,
            )
            store_modes.append({"id": req_b, "mode": exact_b.get("mode"), "attempts": exact_b.get("attempts", [])})
            if exact_b.get("ok"):
                row_b = exact_b.get("row")
                stored_rows += 1
            else:
                for err in exact_b.get("errors", []) or []:
                    errors.append(f"{req_b}:{err}")

        live_a = _resolve_live_memory_row_strict(clair, req_a, a_text, lookup_limit=lookup_limit)
        live_b = _resolve_live_memory_row_strict(clair, req_b, b_text, lookup_limit=lookup_limit)

        if live_a is None:
            errors.append(f"resolve_failed_exact:{req_a}")
            continue
        if live_b is None:
            errors.append(f"resolve_failed_exact:{req_b}")
            continue

        _force_memory_identity(live_a, req_a)
        _force_memory_identity(live_b, req_b)

        _upsert_conflict_metadata(
            live_a,
            self_id=req_a,
            other_id=req_b,
            self_text=a_text,
            other_text=b_text,
            now_ts=now_ts,
        )
        _upsert_conflict_metadata(
            live_b,
            self_id=req_b,
            other_id=req_a,
            self_text=b_text,
            other_text=a_text,
            now_ts=now_ts,
        )

        resolved_ids.extend([req_a, req_b])

    dedup_resolved: List[str] = []
    seen = set()
    for rid in resolved_ids:
        rid2 = str(rid).strip().lower()
        if rid2 and rid2 not in seen:
            seen.add(rid2)
            dedup_resolved.append(rid2)

    expected_exact_rows = len(set(_norm_text(x) for x in text_targets if x))
    ok = (
        len([e for e in errors if str(e).startswith("resolve_failed_exact:")]) == 0
        and len(dedup_resolved) >= expected_exact_rows
    )

    return {
        "ok": ok,
        "count": len(requested_ids),
        "stored_rows": stored_rows,
        "claims": text_targets[:],
        "ids": requested_ids,
        "resolved_ids": dedup_resolved,
        "resolved_map": {rid: rid for rid in dedup_resolved},
        "text_targets": text_targets,
        "store_modes": store_modes,
        "errors": errors,
    }


def _verify_injected_conflicts(clair: Any, injected: Dict[str, Any], *, lookup_limit: int = 800) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "ok": True,
        "requested_ids": list(injected.get("ids", []) or []),
        "resolved_ids": list(injected.get("resolved_ids", []) or []),
        "found_ids": [],
        "missing_ids": [],
        "found_rows": [],
        "index_size": 0,
    }

    idx = _build_mem_index(clair, lookup_limit=lookup_limit)
    out["index_size"] = len(idx)

    requested_ids = [str(x).strip().lower() for x in injected.get("ids", []) if str(x).strip()]
    for mid in requested_ids:
        row = _find_memory_by_exact_id(clair, mid, lookup_limit=lookup_limit)
        if row is not None:
            out["found_ids"].append(mid)
            out["found_rows"].append({
                "id": mid,
                "row_id": _mem_id_any(row),
                "type": _row_get(row, "type"),
                "domain": _row_get(row, "domain"),
                "kind": _row_get(row, "kind"),
                "text": _mem_text(row),
                "details": copy.deepcopy(_details_dict(row)),
            })
        else:
            out["missing_ids"].append(mid)

    out["ok"] = len(out["missing_ids"]) == 0
    return out


def _verify_injected_conflicts_by_text(clair: Any, injected: Dict[str, Any], *, lookup_limit: int = 800) -> Dict[str, Any]:
    targets = [str(x).strip() for x in injected.get("text_targets", []) if str(x).strip()]
    found = []
    missing = []
    fuzzy_only = []

    for target in targets:
        exact_row = _find_memory_by_exact_text(clair, target, lookup_limit=lookup_limit)
        if exact_row is not None:
            found.append({
                "target": target,
                "match_type": "exact",
                "row_id": _mem_id_any(exact_row) or _cereb_style_mem_id(exact_row),
                "text": _mem_text(exact_row),
                "details": copy.deepcopy(_details_dict(exact_row)),
            })
            continue

        fuzzy_row = _find_memory_by_text_fallback(clair, target, lookup_limit=lookup_limit)
        if fuzzy_row is not None:
            fuzzy_only.append({
                "target": target,
                "match_type": "fuzzy_only",
                "row_id": _mem_id_any(fuzzy_row) or _cereb_style_mem_id(fuzzy_row),
                "text": _mem_text(fuzzy_row),
                "details": copy.deepcopy(_details_dict(fuzzy_row)),
            })
        else:
            missing.append(target)

    return {
        "ok": len(missing) == 0 and len(fuzzy_only) == 0,
        "found": found,
        "fuzzy_only": fuzzy_only,
        "missing": missing,
    }


# =============================================================================
# Candidate hydration + repeat governor + diagnostics
# =============================================================================
def _extract_candidate_fields(candidate: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "memory_id": None,
        "memory_ids": [],
        "claim": None,
        "question_text": None,
        "kind": None,
        "raw": candidate,
    }

    if candidate is None:
        return out

    if isinstance(candidate, str):
        out["question_text"] = candidate
        return out

    if isinstance(candidate, dict):
        mid = candidate.get("memory_id")
        if isinstance(mid, str) and mid.strip():
            out["memory_id"] = mid.strip().lower()

        mids = candidate.get("memory_ids")
        if isinstance(mids, list):
            out["memory_ids"] = [str(x).strip().lower() for x in mids if str(x).strip()]

        claim = candidate.get("claim") or candidate.get("content")
        if isinstance(claim, str) and claim.strip():
            out["claim"] = claim.strip()

        q = candidate.get("question") or candidate.get("prompt") or candidate.get("text")
        if isinstance(q, str) and q.strip():
            out["question_text"] = q.strip()

        kind = candidate.get("kind")
        if isinstance(kind, str) and kind.strip():
            out["kind"] = kind.strip()

        if out["question_text"] is None and out["claim"]:
            out["question_text"] = f'Calibration check: "{out["claim"]}" — is this correct?'

        return out

    out["question_text"] = str(candidate)
    return out


def _extract_mem_ids_from_any(value: Any) -> List[str]:
    out: List[str] = []

    def add(v: Any) -> None:
        if isinstance(v, str):
            found = _MEMID_RE.findall(v)
            if found:
                out.extend([x.lower() for x in found])
            else:
                s = v.strip().lower()
                if s.startswith("mem_"):
                    out.append(s)
        elif isinstance(v, dict):
            for k in (
                "memory_id", "id", "uid", "pk",
                "memory_a_id", "memory_b_id", "a_id", "b_id", "id_a", "id_b"
            ):
                vv = v.get(k)
                if isinstance(vv, str) and vv.strip():
                    add(vv)
            for vv in v.values():
                if isinstance(vv, (str, list, tuple, set, dict)):
                    add(vv)
        elif isinstance(v, (list, tuple, set)):
            for item in v:
                add(item)

    add(value)

    seen = set()
    deduped: List[str] = []
    for x in out:
        if x and x not in seen:
            seen.add(x)
            deduped.append(x)
    return deduped


def _extract_candidate_pair_ids(candidate: Any) -> List[str]:
    if not isinstance(candidate, dict):
        return []

    out: List[str] = []

    for key in ("pair_ids", "conflict_pair_ids", "memory_ids"):
        vals = candidate.get(key)
        if isinstance(vals, list):
            out.extend(_extract_mem_ids_from_any(vals))

    mid = candidate.get("memory_id")
    if isinstance(mid, str) and mid.strip():
        out.extend(_extract_mem_ids_from_any(mid))

    payload = candidate.get("payload")
    if isinstance(payload, dict):
        for key in (
            "pair", "pair_ids", "conflict_pair", "conflict_pair_ids",
            "memory_a_id", "memory_b_id", "a_id", "b_id", "id_a", "id_b",
            "memory_a", "memory_b"
        ):
            if key in payload:
                out.extend(_extract_mem_ids_from_any(payload.get(key)))

    for key in ("conflict_hint", "question", "prompt", "text", "claim", "content"):
        v = candidate.get(key)
        if isinstance(v, str) and v.strip():
            out.extend(_extract_mem_ids_from_any(v))

    seen = set()
    deduped: List[str] = []
    for x in out:
        if x and x not in seen:
            seen.add(x)
            deduped.append(x)

    return deduped[:2]


def _extract_candidate_side_texts(candidate: Any) -> Tuple[str, str]:
    if not isinstance(candidate, dict):
        return "", ""

    def pick(d: Dict[str, Any], keys: List[str]) -> str:
        for k in keys:
            v = d.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

    ta = pick(candidate, ["conflict_a", "text_a", "claim_a", "side_a", "memory_a_text"])
    tb = pick(candidate, ["conflict_b", "text_b", "claim_b", "side_b", "memory_b_text"])

    payload = candidate.get("payload")
    if isinstance(payload, dict):
        if not ta:
            ta = pick(payload, ["text_a", "claim_a", "side_a", "memory_a_text"])
        if not tb:
            tb = pick(payload, ["text_b", "claim_b", "side_b", "memory_b_text"])

        ma = payload.get("memory_a")
        mb = payload.get("memory_b")
        if not ta and isinstance(ma, dict):
            ta = _mem_text(ma)
        if not tb and isinstance(mb, dict):
            tb = _mem_text(mb)

    return ta, tb


def _is_conflict_candidate(candidate: Any) -> bool:
    if not isinstance(candidate, dict):
        return False

    kind = str(candidate.get("kind") or "").strip().lower()
    if _CONFLICT_KIND_RE.search(kind):
        return True

    if bool(candidate.get("conflict", False)):
        return True

    pair_ids = _extract_candidate_pair_ids(candidate)
    if len(pair_ids) >= 2:
        return True

    ta, tb = _extract_candidate_side_texts(candidate)
    return bool(ta or tb)


def _canonical_pair_ids(pair_ids: List[str]) -> List[str]:
    xs = [str(x).strip().lower() for x in pair_ids if str(x).strip()]
    if len(xs) >= 2:
        a, b = sorted(xs[:2])
        return [a, b]
    return xs[:2]


def _conflict_lifecycle_diag(
    clair: Any,
    candidate_raw: Any,
    candidate_hydrated: Any,
    *,
    lookup_limit: int = 800,
) -> Dict[str, Any]:
    idx = _build_mem_index(clair, lookup_limit=lookup_limit)

    raw_pair = _extract_candidate_pair_ids(candidate_raw)
    hyd_pair = _extract_candidate_pair_ids(candidate_hydrated)
    canon_pair = _canonical_pair_ids(hyd_pair or raw_pair)

    raw_a, raw_b = _extract_candidate_side_texts(candidate_raw)
    hyd_a, hyd_b = _extract_candidate_side_texts(candidate_hydrated)

    a_id = canon_pair[0] if len(canon_pair) >= 1 else ""
    b_id = canon_pair[1] if len(canon_pair) >= 2 else ""

    ma = idx.get(a_id) if a_id else None
    mb = idx.get(b_id) if b_id else None

    if ma is None and raw_a:
        ma = _find_memory_by_text_fallback(clair, raw_a, lookup_limit=lookup_limit)
    if mb is None and raw_b:
        mb = _find_memory_by_text_fallback(clair, raw_b, lookup_limit=lookup_limit)
    if ma is None and hyd_a:
        ma = _find_memory_by_text_fallback(clair, hyd_a, lookup_limit=lookup_limit)
    if mb is None and hyd_b:
        mb = _find_memory_by_text_fallback(clair, hyd_b, lookup_limit=lookup_limit)

    if not a_id and ma is not None:
        a_id = _mem_id_any(ma) or _cereb_style_mem_id(ma) or ""
    if not b_id and mb is not None:
        b_id = _mem_id_any(mb) or _cereb_style_mem_id(mb) or ""

    mem_a_text = _mem_text(ma or {})
    mem_b_text = _mem_text(mb or {})

    final_a = hyd_a or mem_a_text or raw_a
    final_b = hyd_b or mem_b_text or raw_b

    exists_a = bool(a_id and ma is not None)
    exists_b = bool(b_id and mb is not None)

    status = "not_conflict"
    if _is_conflict_candidate(candidate_raw) or _is_conflict_candidate(candidate_hydrated):
        if not a_id or not b_id:
            status = "incomplete_missing_pair_ids"
        elif not exists_a and not exists_b:
            status = "incomplete_missing_both_ids"
        elif not exists_a:
            status = "incomplete_missing_a_id"
        elif not exists_b:
            status = "incomplete_missing_b_id"
        elif not final_a and not final_b:
            status = "incomplete_missing_both_texts"
        elif not final_a:
            status = "incomplete_missing_a_text"
        elif not final_b:
            status = "incomplete_missing_b_text"
        else:
            status = "complete"

    return {
        "raw_pair_ids": raw_pair,
        "hydrated_pair_ids": hyd_pair,
        "canonical_pair_ids": [x for x in (a_id, b_id) if x],
        "exists_in_index": {"a": exists_a, "b": exists_b},
        "memory_lookup": {"a": _mem_summary(ma), "b": _mem_summary(mb)},
        "raw_side_texts": {"a": raw_a, "b": raw_b},
        "hydrated_side_texts": {"a": hyd_a, "b": hyd_b},
        "final_side_texts": {"a": final_a, "b": final_b},
        "status": status,
    }


def _hydrate_candidate(clair: Any, candidate: Any, *, lookup_limit: int = 800) -> Tuple[Any, bool, str]:
    if not isinstance(candidate, dict):
        return candidate, False, "none"

    idx = _build_mem_index(clair, lookup_limit=lookup_limit)
    if not idx:
        return candidate, False, "none"

    if _is_conflict_candidate(candidate):
        c2 = dict(candidate)
        payload = c2.get("payload")
        if not isinstance(payload, dict):
            payload = {}

        pair_ids = _canonical_pair_ids(_extract_candidate_pair_ids(c2))
        a_id = pair_ids[0] if len(pair_ids) >= 1 else ""
        b_id = pair_ids[1] if len(pair_ids) >= 2 else ""

        ma = idx.get(a_id) if a_id else None
        mb = idx.get(b_id) if b_id else None

        ta, tb = _extract_candidate_side_texts(c2)

        if ma is None and ta:
            ma = _find_memory_by_text_fallback(clair, ta, lookup_limit=lookup_limit)
        if mb is None and tb:
            mb = _find_memory_by_text_fallback(clair, tb, lookup_limit=lookup_limit)

        if ma is not None:
            ta = _mem_text(ma) or ta
            a_id = _mem_id_any(ma) or _cereb_style_mem_id(ma) or a_id
        if mb is not None:
            tb = _mem_text(mb) or tb
            b_id = _mem_id_any(mb) or _cereb_style_mem_id(mb) or b_id

        if a_id or b_id or ta or tb:
            c2["memory_ids"] = [x for x in (a_id, b_id) if x]
            c2["pair_ids"] = [x for x in (a_id, b_id) if x]
            c2["conflict_pair_ids"] = [x for x in (a_id, b_id) if x]
            c2["memory_id"] = a_id or c2.get("memory_id")
            c2["conflict_a"] = ta
            c2["conflict_b"] = tb
            c2["conflict"] = True

            payload["pair"] = [x for x in (a_id, b_id) if x]
            payload["pair_ids"] = [x for x in (a_id, b_id) if x]
            payload["conflict_pair"] = [x for x in (a_id, b_id) if x]
            payload["conflict_pair_ids"] = [x for x in (a_id, b_id) if x]
            payload["memory_a_id"] = a_id
            payload["memory_b_id"] = b_id
            payload["text_a"] = ta
            payload["text_b"] = tb
            payload["claim_a"] = ta
            payload["claim_b"] = tb
            payload["side_a"] = ta
            payload["side_b"] = tb

            if ma is not None:
                payload["memory_a"] = {
                    "id": a_id,
                    "claim": _mem_text(ma),
                    "content": _mem_text(ma),
                    "type": _row_get(ma, "type"),
                    "domain": _row_get(ma, "domain"),
                    "details": copy.deepcopy(_details_dict(ma)),
                }
            if mb is not None:
                payload["memory_b"] = {
                    "id": b_id,
                    "claim": _mem_text(mb),
                    "content": _mem_text(mb),
                    "type": _row_get(mb, "type"),
                    "domain": _row_get(mb, "domain"),
                    "details": copy.deepcopy(_details_dict(mb)),
                }

            prompt = (
                "Calibration conflict check:\n"
                f"A) {ta or '(missing)'}\n"
                f"B) {tb or '(missing)'}\n"
                "Which is more likely correct? Options: confirm A / confirm B / unsure / modify."
            )
            c2["prompt"] = prompt
            c2["question"] = prompt
            c2["payload"] = payload
            return c2, True, "conflict"

    for k in ("claim", "content", "question", "prompt", "text"):
        v = candidate.get(k)
        if isinstance(v, str) and v.strip():
            return candidate, False, "none"

    cn = _extract_candidate_fields(candidate)

    mids: List[str] = []
    if cn.get("memory_id"):
        mids.append(str(cn["memory_id"]).lower())
    if cn.get("memory_ids"):
        mids.extend([str(x).lower() for x in cn["memory_ids"] if x])

    seen = set()
    mids = [x for x in mids if not (x in seen or seen.add(x))]

    for mid in mids:
        mm = idx.get(mid)
        if not mm:
            continue
        txt = _mem_text(mm)
        if not txt:
            continue
        c2 = dict(candidate)
        c2["claim"] = txt
        prompt = f'Calibration check: "{txt}" — is this correct?'
        c2["prompt"] = prompt
        c2["question"] = prompt
        return c2, True, "simple"

    return candidate, False, "none"


def _candidate_fingerprint(candidate: Any, *, soft: bool) -> str:
    cn = _extract_candidate_fields(candidate)
    text = cn.get("question_text") or cn.get("claim") or ""
    if isinstance(candidate, dict) and (candidate.get("conflict_a") or candidate.get("conflict_b")):
        text = (candidate.get("prompt") or "") + "\n" + (candidate.get("conflict_a") or "") + "\n" + (candidate.get("conflict_b") or "")
    low = (text or "").lower().strip()
    low = _WS.sub(" ", low)
    if soft:
        low = _PUNCT.sub("", low)
        low = _WS.sub(" ", low).strip()
    return hashlib.sha1(low.encode("utf-8", errors="ignore")).hexdigest()[:12]


def _candidate_identity(candidate: Any, *, repeat_soft_norm: bool) -> Dict[str, str]:
    out = {"key": "", "kind": "unknown", "reason": "unknown"}

    if isinstance(candidate, dict):
        kind = str(candidate.get("kind") or "").strip().lower()
        if kind:
            out["kind"] = kind

        if _is_conflict_candidate(candidate):
            pair_ids = _canonical_pair_ids(_extract_candidate_pair_ids(candidate))
            if len(pair_ids) >= 2:
                a, b = pair_ids
                out["key"] = f"conflict_pair|{out['kind']}|{a}|{b}"
                out["reason"] = "conflict_pair_ids"
                return out

            ta, tb = _extract_candidate_side_texts(candidate)
            ta_n = _norm_text(ta)
            tb_n = _norm_text(tb)
            pairs = [x for x in (ta_n, tb_n) if x]
            if pairs:
                if len(pairs) >= 2:
                    aa, bb = sorted(pairs[:2])
                else:
                    aa, bb = pairs[0], ""
                out["key"] = f"conflict_text|{out['kind']}|{_fingerprint_text(aa)}|{_fingerprint_text(bb)}"
                out["reason"] = "conflict_side_texts"
                return out

        mid = candidate.get("memory_id")
        if isinstance(mid, str) and mid.strip():
            out["key"] = f"memory_id|{mid.strip().lower()}"
            out["reason"] = "memory_id"
            return out

        mids = candidate.get("memory_ids")
        if isinstance(mids, list) and mids:
            mids_clean = [str(x).strip().lower() for x in mids if str(x).strip()]
            if len(mids_clean) == 1:
                out["key"] = f"memory_id|{mids_clean[0]}"
                out["reason"] = "memory_ids_single"
                return out

        claim = candidate.get("claim") or candidate.get("content")
        if isinstance(claim, str) and claim.strip():
            out["key"] = f"claim|{_fingerprint_text(claim)}"
            out["reason"] = "claim_fingerprint"
            return out

    fp = _candidate_fingerprint(candidate, soft=bool(repeat_soft_norm))
    out["key"] = f"prompt|{fp}"
    out["reason"] = "prompt_fingerprint"
    return out


def _inspect_acc_state(acc: Any, clair: Any, *, lookup_limit: int = 800, max_items: int = 20) -> Dict[str, Any]:
    snap: Dict[str, Any] = {
        "ok": True,
        "full_audit": None,
        "queue_len": 0,
        "queue_preview": [],
        "errors": [],
    }

    try:
        if hasattr(acc, "full_audit") and callable(getattr(acc, "full_audit")):
            snap["full_audit"] = acc.full_audit()
    except Exception as e:
        snap["errors"].append(f"full_audit_error: {e}")

    try:
        if hasattr(acc, "refresh_queue") and callable(getattr(acc, "refresh_queue")):
            acc.refresh_queue(force=True)
    except Exception as e:
        snap["errors"].append(f"refresh_queue_error: {e}")

    try:
        queue = getattr(acc, "_queue", [])
        if isinstance(queue, list):
            snap["queue_len"] = len(queue)
            preview = []
            for item in queue[: max(1, int(max_items))]:
                payload = getattr(item, "payload", {}) if item is not None else {}
                kind = getattr(item, "kind", None)
                mem_ids = getattr(item, "memory_ids", None)
                qid = getattr(item, "qid", None)

                pair_ids = []
                if isinstance(payload, dict):
                    pair_ids = _canonical_pair_ids(_extract_mem_ids_from_any(
                        payload.get("pair_ids") or payload.get("pair") or payload.get("conflict_pair_ids") or mem_ids or []
                    ))

                fake_candidate = {
                    "kind": kind,
                    "memory_ids": list(mem_ids or []),
                    "pair_ids": pair_ids,
                    "payload": dict(payload) if isinstance(payload, dict) else {},
                    "conflict": bool("conflict" in str(kind or "")),
                }
                diag = _conflict_lifecycle_diag(clair, fake_candidate, fake_candidate, lookup_limit=lookup_limit)

                preview.append({
                    "qid": qid,
                    "kind": kind,
                    "memory_ids": list(mem_ids or []),
                    "pair_ids": pair_ids,
                    "diag": diag,
                })
            snap["queue_preview"] = preview
    except Exception as e:
        snap["errors"].append(f"queue_preview_error: {e}")

    return snap


# =============================================================================
# Deterministic simulated user
# =============================================================================
class SimUser:
    _NUM = re.compile(r"\d+(?:\.\d+)?")

    def __init__(self, seed: int = 1337, *, confirm_rate: float = 0.80):
        self.rng = random.Random(seed)
        self.confirm_rate = max(0.0, min(1.0, float(confirm_rate)))

        self.anchors = [
            (("human", "bone"), "206", "Humans have 206 bones."),
            (("water", "boil"), "100", "Water boils at 100 degrees Celsius at sea level."),
            (("everest",), "8848", "Mount Everest is 8848 meters tall."),
        ]

        self.safe_confirm_hints = (
            "higher ground",
            "avoid floodwater",
            "do not drive",
            "get low",
            "smoke",
            "evacuat",
            "aftershock",
            "drop, cover",
            "signal for help",
            "conserve energy",
            "timebox",
            "3 priorities",
        )

    def _nums(self, s: str) -> List[str]:
        try:
            return self._NUM.findall(s or "")
        except Exception:
            return []

    def _candidate_text(self, candidate: Any) -> str:
        try:
            if isinstance(candidate, dict) and (candidate.get("conflict_a") or candidate.get("conflict_b")):
                return (candidate.get("prompt") or candidate.get("question") or "") + "\n" + (
                    str(candidate.get("conflict_a") or "") + "\n" + str(candidate.get("conflict_b") or "")
                )
            cn = _extract_candidate_fields(candidate)
            return str(cn.get("claim") or cn.get("question_text") or "")
        except Exception:
            return str(candidate or "")

    def answer(self, candidate: Any) -> Dict[str, Any]:
        try:
            text = self._candidate_text(candidate)
            low = (text or "").lower().strip()
            if not low:
                return {"verdict": "unsure", "notes": "Empty prompt."}

            if isinstance(candidate, dict) and (candidate.get("conflict_a") or candidate.get("conflict_b")):
                a = str(candidate.get("conflict_a") or "")
                b = str(candidate.get("conflict_b") or "")
                la = a.lower()
                lb = b.lower()

                for must, good_num, _corr in self.anchors:
                    ma = all(k in la for k in must)
                    mb = all(k in lb for k in must)
                    if ma or mb:
                        na = self._nums(la)
                        nb = self._nums(lb)
                        if ma and any(good_num == n for n in na):
                            return {"verdict": "confirm", "notes": "Conflict: A matches anchor."}
                        if mb and any(good_num == n for n in nb):
                            return {"verdict": "confirm", "notes": "Conflict: B matches anchor."}
                return {"verdict": "unsure", "notes": "Conflict A/B without anchor; conservative."}

            if any(h in low for h in self.safe_confirm_hints):
                return {"verdict": "confirm", "notes": "Plausible safety/lesson guidance."}

            for must, good_num, correction in self.anchors:
                if all(k in low for k in must):
                    nums = self._nums(low)
                    if not nums:
                        return {"verdict": "modify", "correction": correction, "notes": "Anchor missing numeric value."}
                    if any(good_num == n for n in nums):
                        return {"verdict": "confirm", "notes": "Known anchor match."}
                    return {"verdict": "deny", "correction": correction, "notes": "Anchor numeric mismatch."}

            if self._nums(low):
                if self.rng.random() < 0.35:
                    return {"verdict": "confirm", "notes": "Numeric claim seems plausible; probabilistic confirm."}
                return {"verdict": "unsure", "notes": "Numeric claim without anchor; conservative."}

            if 8 <= len(low) <= 220 and self.rng.random() < self.confirm_rate:
                return {"verdict": "confirm", "notes": "Plausible non-numeric claim; confirm for progress."}

            return {"verdict": "unsure", "notes": "Default conservative response."}
        except Exception as e:
            return {"verdict": "unsure", "notes": f"SimUser error fallback: {e.__class__.__name__}"}


# =============================================================================
# Feedback adapters
# =============================================================================
def _map_feedback_to_cerebellar_args(candidate_norm: Dict[str, Any], feedback: Dict[str, Any]) -> Tuple[Optional[str], str, Optional[str]]:
    mid = candidate_norm.get("memory_id")
    verdict = str(feedback.get("verdict") or "unsure").strip().lower()
    correction = feedback.get("correction")
    mod = str(correction).strip() if isinstance(correction, str) and correction.strip() else None

    if verdict not in ("confirm", "deny", "modify", "unsure", "merge"):
        verdict = "unsure"
    if verdict == "merge":
        verdict = "unsure"
    if verdict == "modify" and not mod:
        verdict = "unsure"

    return (str(mid) if mid is not None else None, verdict, mod)


def _apply_feedback_tolerant(cereb: Any, acc: Any, candidate: Any, feedback: Dict[str, Any]) -> Tuple[bool, str, Any]:
    cand_norm = _extract_candidate_fields(candidate)
    mem_id, ans, mod = _map_feedback_to_cerebellar_args(cand_norm, feedback)

    cereb_packet_apply = ["apply_feedback", "answer_question", "submit_feedback", "handle_user_answer", "on_user_feedback"]
    ok, res, used = _call_first(cereb, cereb_packet_apply, candidate, feedback)
    if ok:
        return True, f"cerebellar.{used}", res

    if mem_id is not None:
        ok2, res2, used2 = _call_first(cereb, ["apply_user_feedback"], mem_id, ans, mod)
        if ok2:
            return True, f"cerebellar.{used2}", res2
        ok3, res3, used3 = _call_first(cereb, ["apply_user_feedback"], mem_id, ans)
        if ok3:
            return True, f"cerebellar.{used3}", res3

    acc_apply = ["apply_feedback", "submit_feedback", "handle_feedback", "on_feedback"]
    ok4, res4, used4 = _call_first(acc, acc_apply, candidate, feedback)
    if ok4:
        return True, f"acc.{used4}", res4

    return False, "cerebellar.? / acc.?", {"cereb_error": str(res), "acc_error": str(res4)}


# =============================================================================
# Loop 1: REASONING LOOP TEST
# =============================================================================
def _extract_text_from_response(resp: Any) -> str:
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict):
        for k in ("text", "answer", "response", "content", "output"):
            v = resp.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return _safe_json(resp)
    try:
        return str(resp)
    except Exception:
        return ""


def _clair_reasoning_call(clair: Any, prompt: str) -> Tuple[bool, Any, str]:
    methods = [
        "ask",
        "answer",
        "respond",
        "reply",
        "process",
        "process_input",
        "handle_input",
        "on_input",
        "tick",
        "run_once",
    ]
    return _call_first(clair, methods, prompt)


def _run_reasoning_loop(
    clair: Any,
    prompts: List[str],
    repeats: int,
    *,
    label: str,
    print_samples: bool = False,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "label": label,
        "repeats": repeats,
        "prompts": prompts,
        "runs": [],
        "summary": {},
    }

    num_re = re.compile(r"\d+(?:\.\d+)?")

    for r in range(repeats):
        run_evt = {"r": r, "items": [], "t": _now()}
        for p in prompts:
            ok, resp, used = _clair_reasoning_call(clair, p)
            text = _extract_text_from_response(resp)
            fp = _fingerprint_text(text)
            soft = num_re.sub("#", (text or ""))
            fp_soft = _fingerprint_text(soft)

            run_evt["items"].append(
                {
                    "prompt": p,
                    "ok": bool(ok),
                    "method": used,
                    "fp": fp,
                    "fp_soft": fp_soft,
                    "text_preview": (text[:220] + "…") if len(text) > 220 else text,
                    "error": (str(resp)[:200] if (not ok) else None),
                }
            )
        out["runs"].append(run_evt)

    per_prompt = []
    for i, p in enumerate(prompts):
        fps = [out["runs"][r]["items"][i]["fp"] for r in range(repeats)]
        fps_soft = [out["runs"][r]["items"][i]["fp_soft"] for r in range(repeats)]
        uniq = len(set(fps))
        uniq_soft = len(set(fps_soft))
        per_prompt.append(
            {
                "prompt": p,
                "unique_outputs": uniq,
                "unique_outputs_soft": uniq_soft,
                "stable": (uniq == 1),
                "stable_soft": (uniq_soft == 1),
            }
        )

    stable_ct = sum(1 for x in per_prompt if x["stable"])
    stable_soft_ct = sum(1 for x in per_prompt if x["stable_soft"])
    out["summary"] = {
        "prompts": len(prompts),
        "stable_exact": stable_ct,
        "stable_soft": stable_soft_ct,
        "stability_rate_exact": round(stable_ct / max(1, len(prompts)), 3),
        "stability_rate_soft": round(stable_soft_ct / max(1, len(prompts)), 3),
        "per_prompt": per_prompt,
    }

    if print_samples:
        print(f"\n[ReasoningLoop:{label}] stability exact={out['summary']['stability_rate_exact']}, soft={out['summary']['stability_rate_soft']}")
        for item in per_prompt[: min(6, len(per_prompt))]:
            print(" -", item["prompt"], "stable=", item["stable"], "stable_soft=", item["stable_soft"])

    return out


# =============================================================================
# Loop 2: SIMULATOR LOOP TEST
# =============================================================================
def _sim_option_signature(options: List[dict]) -> str:
    parts: List[str] = []
    for o in options or []:
        d = o.get("details", {}) if isinstance(o, dict) else {}
        k = d.get("option_key")
        pn = d.get("planned_next") if isinstance(d, dict) else None
        pn_seed = ""
        if isinstance(pn, dict):
            pn_seed = str(pn.get("seed_text") or pn.get("seed_memory") or "")
        parts.append(f"{k}|pn={_norm_text(pn_seed)[:80]}")
    blob = "\n".join(parts)
    return _fingerprint_text(blob)


def _run_simulator_loop(
    clair: Any,
    domains: List[str],
    calls_per_domain: int,
    num_actions: int,
    *,
    check_determinism: bool,
    print_samples: bool = False,
) -> Dict[str, Any]:
    from planning.simulator import Simulator

    wm = getattr(clair, "memory", None)
    sim = getattr(clair, "simulator", None)

    if sim is None or not hasattr(sim, "generate_options"):
        sim = Simulator()

    out: Dict[str, Any] = {
        "calls_per_domain": calls_per_domain,
        "num_actions": num_actions,
        "domains": domains,
        "events": [],
        "summary": {},
    }

    det_results = []

    prompts = {
        "general": "Plan three safe next steps for organizing a workday with minimal context.",
        "survival": "You smell smoke and see flames nearby. What should you do first?",
        "identity": "Summarize your role and constraints as an assistant in one sentence.",
    }

    for dom in domains:
        for i in range(calls_per_domain):
            ctx = {"domain": dom, "tags": [dom]}
            q = prompts.get(dom, prompts["general"])

            options = sim.generate_options(wm, num_actions=num_actions, question=q, context_profile=ctx, horizon=2)
            sig = _sim_option_signature(options)

            planned_ok = 0
            for o in options or []:
                d = o.get("details", {}) if isinstance(o, dict) else {}
                pn = d.get("planned_next") if isinstance(d, dict) else None
                if isinstance(pn, dict) and (pn.get("seed_text") or pn.get("seed_memory")):
                    planned_ok += 1

            evt = {
                "domain": dom,
                "i": i,
                "sig": sig,
                "options": len(options or []),
                "planned_next_ok": planned_ok,
                "planned_next_rate": round(planned_ok / max(1, len(options or [])), 3),
            }
            out["events"].append(evt)

        if check_determinism and calls_per_domain >= 1:
            ctx = {"domain": dom, "tags": [dom]}
            q = prompts.get(dom, prompts["general"])
            a = sim.generate_options(wm, num_actions=num_actions, question=q, context_profile=ctx, horizon=2)
            b = sim.generate_options(wm, num_actions=num_actions, question=q, context_profile=ctx, horizon=2)
            det_results.append(
                {
                    "domain": dom,
                    "sig_a": _sim_option_signature(a),
                    "sig_b": _sim_option_signature(b),
                    "match": _sim_option_signature(a) == _sim_option_signature(b),
                }
            )

    out["summary"] = {
        "total_calls": len(out["events"]),
        "avg_options": round(sum(e["options"] for e in out["events"]) / max(1, len(out["events"])), 3),
        "avg_planned_next_rate": round(sum(e["planned_next_rate"] for e in out["events"]) / max(1, len(out["events"])), 3),
        "determinism": det_results,
        "determinism_pass_rate": round(sum(1 for d in det_results if d["match"]) / max(1, len(det_results)), 3),
    }

    if print_samples:
        print("\n[SimulatorLoop] avg_options=", out["summary"]["avg_options"])
        print("[SimulatorLoop] avg_planned_next_rate=", out["summary"]["avg_planned_next_rate"])
        print("[SimulatorLoop] determinism_pass_rate=", out["summary"]["determinism_pass_rate"])
        for d in det_results:
            print(" -", d["domain"], "det_match=", d["match"])

    return out


# =============================================================================
# Loop 3: CALIBRATION LOOP TEST
# =============================================================================
def _run_calibration_loop(
    clair: Any,
    cereb: Any,
    acc: Any,
    *,
    idle_steps: int,
    sleep_steps: int,
    seed: int,
    confirm_rate: float,
    print_events: bool,
    hydrate_lookup_limit: int,
    repeat_window: int,
    repeat_soft_norm: bool,
    max_attempts_per_step: int,
    consolidate_every: int,
    reflect_every: int,
    target_conflict_accepts: int,
    conflict_preference_attempts: int,
) -> Dict[str, Any]:
    sim_user = SimUser(seed=seed, confirm_rate=confirm_rate)

    report: Dict[str, Any] = {
        "meta": {
            "idle_steps": idle_steps,
            "sleep_steps": sleep_steps,
            "seed": seed,
            "confirm_rate": confirm_rate,
            "consolidate_every": consolidate_every,
            "reflect_every": reflect_every,
            "hydrate_lookup_limit": hydrate_lookup_limit,
            "repeat_window": repeat_window,
            "repeat_soft_norm": bool(repeat_soft_norm),
            "max_attempts_per_step": max_attempts_per_step,
            "target_conflict_accepts": target_conflict_accepts,
            "conflict_preference_attempts": conflict_preference_attempts,
        },
        "preflight": _inspect_acc_state(acc, clair, lookup_limit=hydrate_lookup_limit, max_items=20),
        "counters": {
            "idle_candidates": 0,
            "idle_skipped_no_candidate": 0,
            "idle_applied": 0,
            "idle_apply_failed": 0,
            "verdict_confirm": 0,
            "verdict_deny": 0,
            "verdict_modify": 0,
            "verdict_unsure": 0,
            "verdict_merge": 0,
            "sleep_ticks": 0,
            "sleep_fallback_acc": 0,
            "hydrated": 0,
            "hydrated_simple": 0,
            "hydrated_conflicts": 0,
            "repeat_blocked": 0,
            "forced_allow_due_to_attempt_cap": 0,
            "conflict_candidates_seen": 0,
            "conflict_candidates_accepted": 0,
            "conflict_preference_skips": 0,
            "conflict_lifecycle_complete": 0,
            "conflict_lifecycle_incomplete": 0,
            "conflict_missing_pair_ids": 0,
            "conflict_missing_a_id": 0,
            "conflict_missing_b_id": 0,
            "conflict_missing_a_text": 0,
            "conflict_missing_b_text": 0,
        },
        "events": [],
        "errors": [],
        "timing": {},
    }

    t0 = _now()

    idle_t0 = _now()
    acc_getters = ["next_question", "pick_question", "select_question", "propose_question", "audit_next", "next_item"]
    cereb_idle = ["idle_tick", "idle_step", "tick_idle", "tick", "run_idle_once"]

    recent_keys: List[str] = []

    for i in range(idle_steps):
        attempts = 0
        accepted = False

        while not accepted:
            attempts += 1
            evt: Dict[str, Any] = {"phase": "idle", "i": i, "attempt": attempts, "t": _now()}
            candidate = None
            candidate_raw = None
            used2 = ""

            ok, res, used = _call_first(acc, acc_getters)
            if ok and res is not None:
                candidate = res
                candidate_raw = res
                evt["candidate_source"] = f"acc.{used}"
            else:
                evt["candidate_error_acc"] = str(res)
                ok2, res2, used2 = _call_first(cereb, cereb_idle)
                if ok2 and res2 is not None:
                    candidate = res2
                    candidate_raw = res2
                    evt["candidate_source"] = f"cerebellar.{used2}"
                else:
                    evt["candidate_source"] = f"cerebellar.{used2}" if used2 else "cerebellar.none"
                    evt["skipped"] = True
                    evt["skip_reason"] = "no_candidate_available"
                    report["counters"]["idle_skipped_no_candidate"] += 1
                    report["events"].append(evt)
                    accepted = True
                    continue

            candidate2, did_hydrate, hkind = _hydrate_candidate(clair, candidate, lookup_limit=hydrate_lookup_limit)
            evt["hydrated"] = bool(did_hydrate)
            evt["hydration_kind"] = hkind
            if did_hydrate:
                report["counters"]["hydrated"] += 1
                if hkind == "simple":
                    report["counters"]["hydrated_simple"] += 1
                if hkind == "conflict":
                    report["counters"]["hydrated_conflicts"] += 1
            candidate = candidate2

            is_conflict = _is_conflict_candidate(candidate)
            evt["is_conflict_candidate"] = is_conflict

            if is_conflict:
                report["counters"]["conflict_candidates_seen"] += 1
                diag = _conflict_lifecycle_diag(clair, candidate_raw, candidate, lookup_limit=hydrate_lookup_limit)
                evt["conflict_lifecycle"] = diag

                status = str(diag.get("status") or "")
                if status == "complete":
                    report["counters"]["conflict_lifecycle_complete"] += 1
                else:
                    report["counters"]["conflict_lifecycle_incomplete"] += 1
                    if status == "incomplete_missing_pair_ids":
                        report["counters"]["conflict_missing_pair_ids"] += 1
                    elif status == "incomplete_missing_a_id":
                        report["counters"]["conflict_missing_a_id"] += 1
                    elif status == "incomplete_missing_b_id":
                        report["counters"]["conflict_missing_b_id"] += 1
                    elif status == "incomplete_missing_a_text":
                        report["counters"]["conflict_missing_a_text"] += 1
                    elif status == "incomplete_missing_b_text":
                        report["counters"]["conflict_missing_b_text"] += 1

            accepted_conflicts_so_far = int(report["counters"]["conflict_candidates_accepted"])
            prefer_conflicts = accepted_conflicts_so_far < int(target_conflict_accepts)
            evt["prefer_conflicts"] = prefer_conflicts

            if prefer_conflicts and not is_conflict and attempts <= int(conflict_preference_attempts):
                report["counters"]["conflict_preference_skips"] += 1
                evt["skipped"] = True
                evt["skip_reason"] = "prefer_conflict_candidate"
                report["events"].append(evt)
                continue

            ident = _candidate_identity(candidate, repeat_soft_norm=bool(repeat_soft_norm))
            evt["repeat_key"] = ident["key"]
            evt["repeat_key_reason"] = ident["reason"]

            is_repeat = bool(repeat_window) and ident["key"] in recent_keys
            evt["repeat_blocked"] = is_repeat

            if is_repeat and attempts < int(max_attempts_per_step):
                report["counters"]["repeat_blocked"] += 1
                evt["skip_reason"] = f"repeat:{ident['reason']}"
                report["events"].append(evt)
                continue

            if is_repeat and attempts >= int(max_attempts_per_step):
                report["counters"]["forced_allow_due_to_attempt_cap"] += 1
                evt["forced_allow"] = True
                evt["forced_allow_reason"] = f"attempt_cap_after_repeat:{ident['reason']}"

            recent_keys.append(ident["key"])
            if len(recent_keys) > max(1, int(repeat_window)):
                recent_keys.pop(0)

            report["counters"]["idle_candidates"] += 1
            evt["candidate"] = candidate

            feedback = sim_user.answer(candidate)
            evt["feedback"] = feedback

            v = str(feedback.get("verdict") or "unsure").lower().strip()
            if v not in ("confirm", "deny", "modify", "unsure", "merge"):
                v = "unsure"
            report["counters"][f"verdict_{v}"] += 1

            ok_apply, used_apply, apply_res = _apply_feedback_tolerant(cereb, acc, candidate, feedback)
            evt["applied_via"] = used_apply
            evt["apply_result"] = apply_res if ok_apply else str(apply_res)

            if ok_apply:
                report["counters"]["idle_applied"] += 1
                if is_conflict:
                    report["counters"]["conflict_candidates_accepted"] += 1
            else:
                report["counters"]["idle_apply_failed"] += 1
                evt["error"] = "Failed to apply feedback (all methods)."
                report["errors"].append(evt["error"])

            if consolidate_every and consolidate_every > 0:
                if (i + 1) % int(consolidate_every) == 0:
                    try:
                        if hasattr(clair, "consolidate_memory"):
                            clair.consolidate_memory()
                        evt["consolidated"] = True
                    except Exception:
                        evt["consolidated"] = False

            if print_events:
                cn = _extract_candidate_fields(candidate)
                qt = (cn.get("question_text") or "")[:90]
                extra = ""
                if is_conflict:
                    diag = evt.get("conflict_lifecycle", {})
                    extra = f" status={diag.get('status')} pair={diag.get('canonical_pair_ids')}"
                print(
                    f"  [idle {i:02d}] {evt.get('candidate_source')} "
                    f"conflict={'Y' if is_conflict else 'N'} "
                    f"hydrated={evt.get('hydrated', False)} kind={hkind} verdict={v:<7} via={used_apply} "
                    f"repeat={'Y' if is_repeat else 'N'} "
                    f"reason={evt.get('repeat_key_reason')} "
                    f"attempt={attempts:02d} "
                    f"q='{qt}'{extra}"
                )

            report["events"].append(evt)
            accepted = True

    report["timing"]["idle_seconds"] = round(_now() - idle_t0, 3)

    sleep_t0 = _now()
    cereb_sleep = ["sleep_tick", "sleep_step", "tick_sleep", "run_sleep_once", "recalibrate", "sleep_cycle"]
    acc_sleep = ["sleep_tick", "recalibrate", "maintenance", "run_maintenance", "consolidate", "repair_pass", "full_audit"]

    for si in range(sleep_steps):
        evt: Dict[str, Any] = {"phase": "sleep", "i": si, "t": _now()}

        ok, res, used = _call_first(cereb, cereb_sleep)
        if ok:
            evt["result_source"] = f"cerebellar.{used}"
            evt["result"] = res
            report["counters"]["sleep_ticks"] += 1
        else:
            evt["sleep_error_cereb"] = str(res)
            ok2, res2, used2 = _call_first(acc, acc_sleep)
            if ok2:
                evt["result_source"] = f"acc.{used2}"
                evt["result"] = res2
                report["counters"]["sleep_fallback_acc"] += 1
            else:
                evt["error"] = f"Sleep tick not implemented (cereb_err={res}, acc_err={res2})"
                report["errors"].append(evt["error"])

        if reflect_every and reflect_every > 0:
            if (si + 1) % int(reflect_every) == 0:
                try:
                    if hasattr(clair, "reflect"):
                        clair.reflect(force=False)
                    evt["reflected"] = True
                except Exception:
                    evt["reflected"] = False

        report["events"].append(evt)

    report["timing"]["sleep_seconds"] = round(_now() - sleep_t0, 3)
    report["timing"]["calibration_seconds"] = round(_now() - t0, 3)

    return report


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--reasoning_repeats", type=int, default=3)
    ap.add_argument("--sim_calls_per_domain", type=int, default=6)
    ap.add_argument("--sim_num_actions", type=int, default=3)
    ap.add_argument("--check_sim_determinism", type=int, default=1)

    ap.add_argument("--idle_steps", type=int, default=20)
    ap.add_argument("--sleep_steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--confirm_rate", type=float, default=0.80)
    ap.add_argument("--print_events", type=int, default=0)
    ap.add_argument("--hydrate_lookup_limit", type=int, default=800)
    ap.add_argument("--no_write", type=int, default=1, help="Block LTM writes during calibration.")
    ap.add_argument("--repeat_window", type=int, default=8)
    ap.add_argument("--repeat_soft_norm", type=int, default=1)
    ap.add_argument("--max_attempts_per_step", type=int, default=3)
    ap.add_argument("--consolidate_every", type=int, default=5)
    ap.add_argument("--reflect_every", type=int, default=2)
    ap.add_argument("--inject_conflicts", type=int, default=1, help="Inject deliberate numeric conflicts before calibration.")

    ap.add_argument("--target_conflict_accepts", type=int, default=4)
    ap.add_argument("--conflict_preference_attempts", type=int, default=2)

    ap.add_argument("--run_benchmark", type=int, default=1)
    ap.add_argument("--benchmark_cmd", type=str, default="python test_clair.py")

    args = ap.parse_args()
    root = _ensure_repo_root_on_path()

    from clair import Clair
    clair = Clair()
    _attach_clair_shims(clair)

    conflict_injection = {
        "ok": False,
        "count": 0,
        "stored_rows": 0,
        "claims": [],
        "ids": [],
        "resolved_ids": [],
        "resolved_map": {},
        "text_targets": [],
        "store_modes": [],
        "note": "disabled",
    }
    if args.inject_conflicts:
        conflict_injection = _inject_conflict_memories(clair, lookup_limit=int(args.hydrate_lookup_limit))

    injection_verify = {
        "ok": False,
        "requested_ids": [],
        "resolved_ids": [],
        "found_ids": [],
        "missing_ids": [],
        "found_rows": [],
        "index_size": 0,
    }
    text_verify = {"ok": False, "found": [], "fuzzy_only": [], "missing": []}

    if args.inject_conflicts:
        injection_verify = _verify_injected_conflicts(
            clair,
            conflict_injection,
            lookup_limit=int(args.hydrate_lookup_limit),
        )
        text_verify = _verify_injected_conflicts_by_text(
            clair,
            conflict_injection,
            lookup_limit=int(args.hydrate_lookup_limit),
        )

    ltm_patch = {"ok": False, "patched": [], "note": "disabled"}
    if args.no_write:
        ltm_patch = _patch_ltm_no_write(clair)

    cereb_mod = _import_any(["calibration.cerebellar", "calibration.cerebellum", "calibration.cerebellar_py"])
    acc_mod = _import_any(["calibration.ACC", "calibration.acc", "calibration.Acc", "calibration.anterior_cingulate_cortex", "calibration.memory_auditor"])

    Cerebellar = _get_first_attr(cereb_mod, ["Cerebellar", "Cerebellum", "CerebellarLoop"])
    ACC_cls = _get_first_attr(acc_mod, ["ACC", "Acc", "AnteriorCingulateCortex", "MemoryAuditor"])
    if Cerebellar is None:
        raise RuntimeError("Could not find Cerebellar class in calibration/cerebellar.py.")
    if ACC_cls is None:
        raise RuntimeError("Could not find ACC class in calibration/ACC.py.")

    acc = _construct_any(
        ACC_cls,
        [
            (clair,),
            (getattr(clair, "memory", None), getattr(clair, "long_term", None)),
            (getattr(clair, "long_term", None),),
            (),
        ],
    )
    cereb = _construct_any(
        Cerebellar,
        [
            (clair, acc),
            (clair, getattr(clair, "long_term", None), acc),
            (getattr(clair, "long_term", None), acc),
            (clair,),
            (),
        ],
    )

    reasoning_prompts = [
        'Calibration check (numeric): "Water boils at 100 degrees Celsius at sea level." Reply confirm/deny.',
        'Calibration check (numeric): "Mount Everest is 8848 meters tall." Reply confirm/deny.',
        "Summarize the safest first 2 actions if you smell smoke in a building.",
        "Make a 3-step plan for organizing tasks with limited energy.",
        "Explain your confidence level briefly after answering a question.",
    ]

    report: Dict[str, Any] = {
        "meta": {
            "reasoning_repeats": args.reasoning_repeats,
            "sim_calls_per_domain": args.sim_calls_per_domain,
            "sim_num_actions": args.sim_num_actions,
            "check_sim_determinism": bool(args.check_sim_determinism),
            "idle_steps": args.idle_steps,
            "sleep_steps": args.sleep_steps,
            "run_benchmark": bool(args.run_benchmark),
            "benchmark_cmd": args.benchmark_cmd,
            "no_write": bool(args.no_write),
            "inject_conflicts": bool(args.inject_conflicts),
            "target_conflict_accepts": int(args.target_conflict_accepts),
            "conflict_preference_attempts": int(args.conflict_preference_attempts),
        },
        "ltm_no_write_patch": ltm_patch,
        "conflict_injection": conflict_injection,
        "conflict_injection_verify": injection_verify,
        "conflict_injection_verify_text": text_verify,
        "phases": {},
        "timing": {},
        "benchmark": None,
    }

    print("\n=== THREE-LOOP HARNESS: START ===")
    T0 = _now()

    print("\n[Phase A] Reasoning loop (PRE)")
    t = _now()
    r_pre = _run_reasoning_loop(
        clair,
        reasoning_prompts,
        repeats=max(1, int(args.reasoning_repeats)),
        label="pre",
        print_samples=True,
    )
    report["phases"]["reasoning_pre"] = r_pre
    report["timing"]["reasoning_pre_seconds"] = round(_now() - t, 3)

    print("\n[Phase B] Simulator loop (PRE)")
    t = _now()
    sim_pre = _run_simulator_loop(
        clair,
        domains=["general", "survival"],
        calls_per_domain=max(1, int(args.sim_calls_per_domain)),
        num_actions=max(1, int(args.sim_num_actions)),
        check_determinism=bool(args.check_sim_determinism),
        print_samples=True,
    )
    report["phases"]["simulator_pre"] = sim_pre
    report["timing"]["simulator_pre_seconds"] = round(_now() - t, 3)

    print("\n[Phase C] Calibration loop (idle + sleep)")
    t = _now()
    cal = _run_calibration_loop(
        clair, cereb, acc,
        idle_steps=max(0, int(args.idle_steps)),
        sleep_steps=max(0, int(args.sleep_steps)),
        seed=int(args.seed),
        confirm_rate=float(args.confirm_rate),
        print_events=bool(args.print_events),
        hydrate_lookup_limit=int(args.hydrate_lookup_limit),
        repeat_window=int(args.repeat_window),
        repeat_soft_norm=bool(args.repeat_soft_norm),
        max_attempts_per_step=int(args.max_attempts_per_step),
        consolidate_every=int(args.consolidate_every),
        reflect_every=int(args.reflect_every),
        target_conflict_accepts=max(0, int(args.target_conflict_accepts)),
        conflict_preference_attempts=max(0, int(args.conflict_preference_attempts)),
    )
    report["phases"]["calibration"] = cal
    report["timing"]["calibration_seconds"] = round(_now() - t, 3)

    print("\n[Phase D] Simulator loop (POST)")
    t = _now()
    sim_post = _run_simulator_loop(
        clair,
        domains=["general", "survival"],
        calls_per_domain=max(1, int(args.sim_calls_per_domain)),
        num_actions=max(1, int(args.sim_num_actions)),
        check_determinism=bool(args.check_sim_determinism),
        print_samples=True,
    )
    report["phases"]["simulator_post"] = sim_post
    report["timing"]["simulator_post_seconds"] = round(_now() - t, 3)

    print("\n[Phase E] Reasoning loop (POST)")
    t = _now()
    r_post = _run_reasoning_loop(
        clair,
        reasoning_prompts,
        repeats=max(1, int(args.reasoning_repeats)),
        label="post",
        print_samples=True,
    )
    report["phases"]["reasoning_post"] = r_post
    report["timing"]["reasoning_post_seconds"] = round(_now() - t, 3)

    pre_exact = report["phases"]["reasoning_pre"]["summary"]["stability_rate_exact"]
    post_exact = report["phases"]["reasoning_post"]["summary"]["stability_rate_exact"]
    pre_soft = report["phases"]["reasoning_pre"]["summary"]["stability_rate_soft"]
    post_soft = report["phases"]["reasoning_post"]["summary"]["stability_rate_soft"]
    report["phases"]["reasoning_delta"] = {
        "stability_rate_exact_pre": pre_exact,
        "stability_rate_exact_post": post_exact,
        "delta_exact": round(post_exact - pre_exact, 3),
        "stability_rate_soft_pre": pre_soft,
        "stability_rate_soft_post": post_soft,
        "delta_soft": round(post_soft - pre_soft, 3),
    }

    if args.run_benchmark:
        print("\n[Phase F] Running benchmark:", args.benchmark_cmd)
        t = _now()
        bm = _run_cmd(args.benchmark_cmd, root)
        report["benchmark"] = bm
        report["timing"]["benchmark_seconds"] = round(_now() - t, 3)
        print(f"[Benchmark] returncode={bm.get('returncode')} seconds={bm.get('seconds')}")

    report["timing"]["total_seconds"] = round(_now() - T0, 3)

    out_path = os.path.join(root, "calibration_three_loop_report.json")
    _write_json(out_path, report)

    print("\n=== THREE-LOOP HARNESS: SUMMARY ===")
    print("Total seconds:", report["timing"]["total_seconds"])
    print("\nReasoning stability (exact): pre=", pre_exact, "post=", post_exact, "delta=", report["phases"]["reasoning_delta"]["delta_exact"])
    print("Reasoning stability (soft):  pre=", pre_soft, "post=", post_soft, "delta=", report["phases"]["reasoning_delta"]["delta_soft"])

    sp = report["phases"]["simulator_pre"]["summary"]
    so = report["phases"]["simulator_post"]["summary"]
    print("\nSimulator avg planned_next rate: pre=", sp["avg_planned_next_rate"], "post=", so["avg_planned_next_rate"])
    print("Simulator determinism pass rate: pre=", sp["determinism_pass_rate"], "post=", so["determinism_pass_rate"])

    print("\nConflict injection:")
    print("  enabled =", bool(args.inject_conflicts))
    print("  ok      =", report["conflict_injection"].get("ok"))
    print("  count   =", report["conflict_injection"].get("count"))
    print("  stored_rows =", report["conflict_injection"].get("stored_rows"))
    print("  requested_ids =", len(report["conflict_injection"].get("ids", [])))
    print("  resolved_ids  =", len(report["conflict_injection"].get("resolved_ids", [])))
    print("  resolved_id_list =", report["conflict_injection"].get("resolved_ids", []))
    if report["conflict_injection"].get("store_modes"):
        print("  store_modes =", report["conflict_injection"].get("store_modes"))
    if report["conflict_injection"].get("errors"):
        print("  injection_errors =", report["conflict_injection"].get("errors"))

    injv = report.get("conflict_injection_verify", {})
    print("  verify_ok =", injv.get("ok"))
    print("  found_ids =", len(injv.get("found_ids", [])))
    print("  missing_ids =", len(injv.get("missing_ids", [])))
    if injv.get("missing_ids"):
        print("  missing_list =", injv.get("missing_ids"))

    txtv = report.get("conflict_injection_verify_text", {})
    print("  text_verify_ok =", txtv.get("ok"))
    print("  text_found =", len(txtv.get("found", [])))
    print("  text_fuzzy_only =", len(txtv.get("fuzzy_only", [])))
    print("  text_missing =", len(txtv.get("missing", [])))
    if txtv.get("fuzzy_only"):
        print("  text_fuzzy_targets =", [x.get("target") for x in txtv.get("fuzzy_only", [])])
    if txtv.get("missing"):
        print("  text_missing_list =", txtv.get("missing"))

    preflight = report["phases"]["calibration"].get("preflight", {})
    print("\nACC preflight:")
    print("  queue_len =", preflight.get("queue_len"))
    fa = preflight.get("full_audit") or {}
    print("  flagged_conflicts =", len(fa.get("flagged_conflicts", []) if isinstance(fa, dict) else []))
    print("  numeric_conflicts =", len(fa.get("numeric_conflicts", []) if isinstance(fa, dict) else []))
    print("  negation_conflicts =", len(fa.get("negation_conflicts", []) if isinstance(fa, dict) else []))

    cal_sum = report["phases"]["calibration"]["counters"]
    print("\nCalibration counters:")
    for k in [
        "idle_candidates",
        "idle_skipped_no_candidate",
        "idle_applied",
        "idle_apply_failed",
        "repeat_blocked",
        "forced_allow_due_to_attempt_cap",
        "sleep_ticks",
        "sleep_fallback_acc",
        "hydrated",
        "hydrated_conflicts",
        "conflict_candidates_seen",
        "conflict_candidates_accepted",
        "conflict_preference_skips",
        "conflict_lifecycle_complete",
        "conflict_lifecycle_incomplete",
        "conflict_missing_pair_ids",
        "conflict_missing_a_id",
        "conflict_missing_b_id",
        "conflict_missing_a_text",
        "conflict_missing_b_text",
    ]:
        print(f"  {k:<34} {cal_sum.get(k)}")

    print("\nWrote report:", out_path)
    print("=== DONE ===")


if __name__ == "__main__":
    main()
