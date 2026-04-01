# calibration/cerebellar.py
# Clair Cerebellar Calibration System
# Handles memory calibration, drift control, confidence rescoring, and sleep maintenance.
#
# v2.60 rewrite (Day 21/21.5 conflict hydration hardening):
# - Preserves v2.59 behavior and backward compatibility
# - Reads/writes both top-level and details-based truth metadata
# - Aligns with Day 21 fields:
#     * source
#     * evidence
#     * last_verified
#     * conflict
#     * verification_status
#     * conflict_with_ids / conflict_with_text
# - Improves question priority for explicit WM conflicts
# - Builds richer verification packets for clair.py route_verify_request(...)
# - Keeps bounded sleep maintenance and harness aliases
# - NEW:
#     * robust conflict packet hydration from ACC payloads
#     * symmetrical A/B resolution
#     * fallback extraction from conflict_hint / payload / nested memory objects
#     * conflict-aware feedback application for injected conflict packets

from __future__ import annotations

import hashlib
import random
import re
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple


class Cerebellar:
    VERSION = "2.60-day21-hydration"

    def __init__(self, *args: Any, config: Any = None):
        self.clair = None
        self.wm = None
        self.ltm = None
        self.acc = None
        self.config = config

        # ------------------------------------------------------------
        # Parse args loosely (harness tolerant)
        # Common patterns:
        #   (clair, acc)
        #   (clair, ltm, acc)
        #   (wm, ltm, acc)
        #   (ltm, acc)
        # ------------------------------------------------------------
        if args:
            a0 = args[0]
            if hasattr(a0, "memory") or hasattr(a0, "long_term"):
                self.clair = a0
                self.wm = getattr(a0, "memory", None)
                self.ltm = getattr(a0, "long_term", None)

                if len(args) >= 2:
                    cand = args[1]
                    if hasattr(cand, "retrieve") and len(args) >= 3:
                        self.ltm = cand
                        self.acc = args[2]
                    else:
                        self.acc = cand
            else:
                if len(args) >= 2 and hasattr(args[0], "buffer"):
                    self.wm = args[0]
                    self.ltm = args[1]
                    if len(args) >= 3:
                        self.acc = args[2]
                else:
                    self.ltm = args[0]
                    if len(args) >= 2:
                        self.acc = args[1]

        cfg = config

        # -----------------------------
        # Question policy
        # -----------------------------
        self.question_limit = int(getattr(cfg, "CAL_Q_LIMIT", 2)) if cfg else 2
        self.cooldown_sec = float(getattr(cfg, "CAL_Q_COOLDOWN_SEC", 0.0)) if cfg else 0.0

        # Diversity + repeat controls
        self.mem_cooldown_sec = float(getattr(cfg, "CAL_MEM_COOLDOWN_SEC", 30.0)) if cfg else 30.0
        self.topk_pick = int(getattr(cfg, "CAL_Q_TOPK_PICK", 8)) if cfg else 8
        self.recent_fingerprint_window = int(getattr(cfg, "CAL_RECENT_FP_WINDOW", 64)) if cfg else 64
        self.recent_fp_cooldown_sec = float(getattr(cfg, "CAL_RECENT_FP_COOLDOWN_SEC", 45.0)) if cfg else 45.0
        self.seed = int(getattr(cfg, "CAL_SEED", 1337)) if cfg else 1337
        self.rng = random.Random(self.seed)

        # -----------------------------
        # Confidence adjustment policy
        # -----------------------------
        self.promotion_step = float(getattr(cfg, "CAL_PROMOTION_STEP", 0.08)) if cfg else 0.08
        self.demotion_step = float(getattr(cfg, "CAL_DEMOTION_STEP", 0.12)) if cfg else 0.12
        self.confidence_cap_no_web = float(getattr(cfg, "CAL_MAX_CONFIDENCE_NO_WEB", 0.85)) if cfg else 0.85
        self.confidence_cap_with_external = float(getattr(cfg, "CAL_MAX_CONFIDENCE_WITH_EXTERNAL", 0.97)) if cfg else 0.97

        # -----------------------------
        # Sleep maintenance policy
        # -----------------------------
        self.stale_days = int(getattr(cfg, "CAL_STALE_DAYS", 30)) if cfg else 30
        self.decay_multiplier = float(getattr(cfg, "CAL_DECAY_MULT", 0.98)) if cfg else 0.98

        # Hard safety rails for sleep tick cost
        self.sleep_time_budget_sec = float(getattr(cfg, "CAL_SLEEP_TIME_BUDGET_SEC", 0.60)) if cfg else 0.60
        self.max_pairs = int(getattr(cfg, "CAL_MAX_PAIR_CHECKS", 1200)) if cfg else 1200
        self.max_numeric_items = int(getattr(cfg, "CAL_MAX_NUMERIC_ITEMS", 120)) if cfg else 120
        self.semantic_overlap_min = int(getattr(cfg, "CAL_SEM_OVERLAP_MIN", 3)) if cfg else 3

        # Metadata caps
        self.max_source_events = int(getattr(cfg, "CAL_MAX_SOURCE_EVENTS", 16)) if cfg else 16
        self.max_history_items = int(getattr(cfg, "CAL_MAX_HISTORY_ITEMS", 12)) if cfg else 12
        self.max_conflict_links = int(getattr(cfg, "CAL_MAX_CONFLICT_LINKS", 16)) if cfg else 16
        self.max_conflict_text_links = int(getattr(cfg, "CAL_MAX_CONFLICT_TEXT_LINKS", 16)) if cfg else 16
        self.max_evidence_items = int(getattr(cfg, "CAL_MAX_EVIDENCE_ITEMS", 16)) if cfg else 16

        # External verification heuristics
        self.external_verify_min_staleness = float(
            getattr(cfg, "CAL_EXT_VERIFY_MIN_STALENESS", 0.50)
        ) if cfg else 0.50
        self.external_verify_min_confidence = float(
            getattr(cfg, "CAL_EXT_VERIFY_MIN_CONFIDENCE", 0.70)
        ) if cfg else 0.70

        # Episode persistence
        self.persist_episodes = bool(getattr(cfg, "CAL_PERSIST_EPISODES", 0)) if cfg else False

        self.last_question_time = 0.0

        # Regex helpers
        self._number_pattern = re.compile(r"\d+(?:\.\d+)?")
        self._ws = re.compile(r"\s+")
        self._nonword = re.compile(r"[^a-z0-9\s]+")
        self._memid_pat = re.compile(r"\bmem_[a-z0-9]{6,64}\b", re.IGNORECASE)

        # Known calibration/meta prefixes we never want to scan as factual claims
        self._cal_log_prefixes = (
            "sleep calibration ran:",
            "sleep calibration report=",
            "calibration feedback verdict=",
            "calibration check:",
            "calibration check (numeric):",
            "calibration conflict check:",
        )

        # Cheap stopword set for topic key bucketing
        self._stop = {
            "the", "a", "an", "and", "or", "to", "of", "in", "on", "at", "for", "is", "are", "was", "were",
            "be", "been", "being", "this", "that", "these", "those", "it", "as", "by", "with", "from",
            "what", "which", "who", "whom", "where", "when", "why", "how",
        }

        self.audit_log: List[Dict[str, Any]] = []

        # Recent asked tracking
        self._mem_last_asked: Dict[str, float] = {}
        self._recent_fp: Deque[Tuple[str, float]] = deque(maxlen=max(8, self.recent_fingerprint_window))

    # ---------------------------------------------------------------------
    # Memory adapters
    # ---------------------------------------------------------------------
    def _mem_text(self, m: Dict[str, Any]) -> str:
        t = m.get("claim")
        if isinstance(t, str) and t.strip():
            return t.strip()
        t2 = m.get("content")
        if isinstance(t2, str) and t2.strip():
            return t2.strip()
        return str(t or t2 or "").strip()

    def _mem_details(self, m: Dict[str, Any]) -> Dict[str, Any]:
        d = m.get("details")
        if isinstance(d, dict):
            return d
        d = {}
        m["details"] = d
        return d

    def _mem_id(self, m: Dict[str, Any]) -> str:
        for k in ("id", "memory_id", "uid", "pk"):
            v = m.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        base = f"{m.get('type', '')}|{m.get('domain', '')}|{self._mem_text(m)}"
        h = hashlib.sha1(base.encode("utf-8", errors="ignore")).hexdigest()[:12]
        return f"mem_{h}"

    def _clamp01(self, x: Any) -> float:
        try:
            v = float(x)
        except Exception:
            return 0.0
        return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

    def _mem_conf(self, m: Dict[str, Any]) -> float:
        try:
            return float(m.get("confidence", self._mem_details(m).get("confidence", 0.5)))
        except Exception:
            return 0.5

    def _mem_source(self, m: Dict[str, Any]) -> str:
        d = self._mem_details(m)
        return str(m.get("source") or d.get("source") or "").strip().lower()

    def _mem_verification_status(self, m: Dict[str, Any]) -> str:
        d = self._mem_details(m)

        v = str(d.get("verification_status") or m.get("verification_status") or "").strip().lower()
        if v:
            return v

        old = str(d.get("status") or m.get("status") or "").strip().lower()
        if old in {"verified", "contested", "unverified", "revised", "deprecated", "pending_external", "pending", "hypothesis"}:
            return old

        if bool(d.get("verified", False)):
            return "verified"
        if bool(d.get("contested", False)):
            return "contested"
        if bool(d.get("pending_verification", False)):
            return "pending"

        return "unverified"

    def _mem_last_verified(self, m: Dict[str, Any]) -> float:
        d = self._mem_details(m)
        for k in ("last_verified", "verified_at", "updated_at", "last_used", "last_access", "created_at", "timestamp"):
            try:
                v = float(m.get(k, d.get(k, 0.0)) or 0.0)
                if v > 0.0:
                    return v
            except Exception:
                continue
        return 0.0

    def _mem_last_used(self, m: Dict[str, Any]) -> float:
        now = time.time()
        d = self._mem_details(m)
        for k in ("last_used", "last_access", "updated_at", "created_at", "timestamp"):
            try:
                v = float(m.get(k, d.get(k, 0.0)) or 0.0)
                if v > 0.0:
                    return v
            except Exception:
                continue
        return now

    def _mem_staleness_risk(self, m: Dict[str, Any]) -> float:
        d = self._mem_details(m)
        sr = d.get("staleness_risk", m.get("staleness_risk"))
        if isinstance(sr, str):
            low = sr.strip().lower()
            if low == "low":
                return 0.15
            if low == "medium":
                return 0.55
            if low == "high":
                return 0.90
        try:
            if sr is not None:
                return self._clamp01(sr)
        except Exception:
            pass

        last_v = self._mem_last_verified(m)
        if last_v <= 0.0:
            return 0.65

        age_days = max(0.0, (time.time() - last_v) / 86400.0)

        if age_days >= 180:
            return 0.95
        if age_days >= 90:
            return 0.80
        if age_days >= 30:
            return 0.55
        if age_days >= 7:
            return 0.25
        return 0.10

    def _mem_memory_class(self, m: Dict[str, Any]) -> str:
        d = self._mem_details(m)
        return str(d.get("memory_class") or m.get("memory_class") or m.get("kind") or m.get("type") or "").strip().lower()

    def _mem_conflict_ids(self, m: Dict[str, Any]) -> List[str]:
        d = self._mem_details(m)
        vals = d.get("conflict_with_ids", m.get("conflict_with_ids"))
        if isinstance(vals, list):
            return [str(x) for x in vals if x]

        vals2 = d.get("conflicts", m.get("conflicts"))
        if isinstance(vals2, list):
            return [str(x) for x in vals2 if x]

        return []

    def _mem_conflict_text(self, m: Dict[str, Any]) -> List[str]:
        d = self._mem_details(m)
        vals = d.get("conflict_with_text", m.get("conflict_with_text"))
        if isinstance(vals, list):
            return [str(x) for x in vals if str(x).strip()]
        return []

    def _mem_conflict_flag(self, m: Dict[str, Any]) -> bool:
        d = self._mem_details(m)
        if bool(m.get("conflict", False)):
            return True
        if bool(d.get("conflict", False)):
            return True
        if bool(d.get("contested", False)):
            return True
        if self._mem_conflict_ids(m):
            return True
        if self._mem_conflict_text(m):
            return True
        if self._mem_verification_status(m) == "contested":
            return True
        return False

    def _mem_is_hypothesis(self, m: Dict[str, Any]) -> bool:
        d = self._mem_details(m)
        h = d.get("hypothesis", m.get("hypothesis", None))
        if isinstance(h, bool):
            return h
        if isinstance(h, str):
            return h.strip().lower() in {"1", "true", "yes", "y"}
        return False

    def _mem_source_count(self, m: Dict[str, Any]) -> int:
        d = self._mem_details(m)

        ev = m.get("evidence", d.get("evidence", None))
        if isinstance(ev, list):
            return len(ev)

        src = m.get("sources", d.get("sources", None))
        if isinstance(src, list):
            return len(src)

        return 0

    def _mem_evidence(self, m: Dict[str, Any]) -> List[str]:
        d = self._mem_details(m)
        ev = m.get("evidence", d.get("evidence", []))
        if not isinstance(ev, list):
            return []
        out: List[str] = []
        for item in ev:
            s = str(item).strip()
            if s:
                out.append(s)
        return out[: self.max_evidence_items]

    def _verified_age_days(self, m: Dict[str, Any]) -> float:
        lv = self._mem_last_verified(m)
        if lv <= 0.0:
            return 9999.0
        return max(0.0, (time.time() - lv) / 86400.0)

    def _class_decay_factor(self, m: Dict[str, Any]) -> float:
        mem_class = self._mem_memory_class(m)

        if mem_class in {"fact", "numeric_fact", "rule", "semantic"}:
            return 1.00
        if mem_class in {"claim", "literary_frame", "concept_frame"}:
            return 0.90
        if mem_class in {"episode"}:
            return 0.70
        if mem_class in {"meta_log", "operations"}:
            return 0.50
        return 0.85

    # ---------------------------------------------------------------------
    # Conflict hydration helpers
    # ---------------------------------------------------------------------
    def _norm_text(self, s: Any) -> str:
        return self._ws.sub(" ", str(s or "").strip())

    def _extract_mem_ids_from_any(self, value: Any) -> List[str]:
        out: List[str] = []

        def add(v: Any) -> None:
            if isinstance(v, str):
                hits = self._memid_pat.findall(v)
                if hits:
                    out.extend(hits)
                else:
                    s = v.strip()
                    if s.startswith("mem_"):
                        out.append(s)
            elif isinstance(v, dict):
                for key in ("id", "memory_id", "uid", "pk", "a_id", "b_id", "id_a", "id_b", "memory_a_id", "memory_b_id"):
                    vv = v.get(key)
                    if isinstance(vv, str) and vv.strip():
                        add(vv)
                for vv in v.values():
                    if isinstance(vv, (str, list, tuple, set, dict)):
                        add(vv)
            elif isinstance(v, (list, tuple, set)):
                for item in v:
                    add(item)

        add(value)

        deduped: List[str] = []
        seen = set()
        for x in out:
            s = str(x).strip()
            if s and s not in seen:
                seen.add(s)
                deduped.append(s)
        return deduped

    def _extract_texts_from_any(self, value: Any) -> List[str]:
        out: List[str] = []

        def add(v: Any) -> None:
            if isinstance(v, str):
                s = self._norm_text(v)
                if s:
                    out.append(s)
            elif isinstance(v, dict):
                for key in (
                    "claim", "content", "text", "text_a", "text_b", "claim_a", "claim_b",
                    "side_a", "side_b", "memory_a_text", "memory_b_text"
                ):
                    vv = v.get(key)
                    if isinstance(vv, str) and vv.strip():
                        out.append(self._norm_text(vv))
            elif isinstance(v, (list, tuple, set)):
                for item in v:
                    add(item)

        add(value)

        deduped: List[str] = []
        seen = set()
        for x in out:
            if x and x not in seen:
                seen.add(x)
                deduped.append(x)
        return deduped

    def _candidate_mem_ids(self, c: Dict[str, Any]) -> List[str]:
        out: List[str] = []

        if isinstance(c.get("memory_ids"), list):
            out.extend([str(x).strip() for x in c.get("memory_ids", []) if str(x).strip()])

        mid = c.get("memory_id")
        if isinstance(mid, str) and mid.strip():
            out.append(mid.strip())

        for key in ("pair_ids", "conflict_pair_ids"):
            vals = c.get(key)
            if isinstance(vals, list):
                out.extend([str(x).strip() for x in vals if str(x).strip()])

        payload = c.get("payload", {})
        if isinstance(payload, dict):
            for key in (
                "pair", "pair_ids", "conflict_pair", "conflict_pair_ids",
                "memory_a_id", "memory_b_id", "a_id", "b_id", "id_a", "id_b"
            ):
                out.extend(self._extract_mem_ids_from_any(payload.get(key)))

            for key in ("memory_a", "memory_b"):
                if isinstance(payload.get(key), dict):
                    out.extend(self._extract_mem_ids_from_any(payload.get(key)))

        hint = c.get("conflict_hint")
        if isinstance(hint, str) and hint.strip():
            out.extend(self._extract_mem_ids_from_any(hint))

        deduped: List[str] = []
        seen = set()
        for x in out:
            s = str(x).strip()
            if s and s not in seen:
                seen.add(s)
                deduped.append(s)
        return deduped

    def _is_conflict_candidate(self, c: Dict[str, Any]) -> bool:
        kind = str(c.get("kind") or "").strip().lower()
        if "conflict" in kind:
            return True
        if bool(c.get("conflict", False)):
            return True

        payload = c.get("payload", {})
        if isinstance(payload, dict):
            if payload.get("pair") or payload.get("pair_ids") or payload.get("conflict_pair") or payload.get("conflict_pair_ids"):
                return True

        hint = str(c.get("conflict_hint") or "").strip().lower()
        return "between mem_" in hint

    def _hydrate_conflict_candidate(self, c: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enriches an ACC conflict candidate so both sides A/B resolve consistently.
        Does not break the harness trick of top-level memory_ids remaining optional.
        """
        out = dict(c)
        payload = out.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}

        mem_ids = self._candidate_mem_ids(out)
        if len(mem_ids) > 2:
            mem_ids = mem_ids[:2]

        if len(mem_ids) >= 2:
            a_id, b_id = mem_ids[0], mem_ids[1]
        elif len(mem_ids) == 1:
            a_id, b_id = mem_ids[0], ""
        else:
            a_id, b_id = "", ""

        ma = self._find_memory_by_id(a_id) if a_id else None
        mb = self._find_memory_by_id(b_id) if b_id else None

        # Fallback text sources from payload
        text_a = ""
        text_b = ""

        if isinstance(payload.get("text_a"), str):
            text_a = self._norm_text(payload.get("text_a"))
        if isinstance(payload.get("text_b"), str):
            text_b = self._norm_text(payload.get("text_b"))

        if not text_a and isinstance(payload.get("claim_a"), str):
            text_a = self._norm_text(payload.get("claim_a"))
        if not text_b and isinstance(payload.get("claim_b"), str):
            text_b = self._norm_text(payload.get("claim_b"))

        if not text_a and isinstance(payload.get("side_a"), str):
            text_a = self._norm_text(payload.get("side_a"))
        if not text_b and isinstance(payload.get("side_b"), str):
            text_b = self._norm_text(payload.get("side_b"))

        mem_a_obj = payload.get("memory_a") if isinstance(payload.get("memory_a"), dict) else None
        mem_b_obj = payload.get("memory_b") if isinstance(payload.get("memory_b"), dict) else None

        if not text_a and isinstance(mem_a_obj, dict):
            text_a = self._mem_text(mem_a_obj)
        if not text_b and isinstance(mem_b_obj, dict):
            text_b = self._mem_text(mem_b_obj)

        if isinstance(ma, dict):
            text_a = self._mem_text(ma) or text_a
        if isinstance(mb, dict):
            text_b = self._mem_text(mb) or text_b

        # Final fallback: scavenge textual payloads if one side is still missing
        if not text_a or not text_b:
            scavenged = self._extract_texts_from_any(payload)
            for s in scavenged:
                if not text_a:
                    text_a = s
                    continue
                if not text_b and s != text_a:
                    text_b = s
                    break

        payload["pair"] = [x for x in (a_id, b_id) if x]
        payload["pair_ids"] = [x for x in (a_id, b_id) if x]
        payload["conflict_pair"] = [x for x in (a_id, b_id) if x]
        payload["conflict_pair_ids"] = [x for x in (a_id, b_id) if x]
        payload["memory_a_id"] = a_id
        payload["memory_b_id"] = b_id
        payload["a_id"] = a_id
        payload["b_id"] = b_id
        payload["id_a"] = a_id
        payload["id_b"] = b_id
        payload["text_a"] = text_a
        payload["text_b"] = text_b
        payload["claim_a"] = text_a
        payload["claim_b"] = text_b
        payload["side_a"] = text_a
        payload["side_b"] = text_b
        payload["memory_a_text"] = text_a
        payload["memory_b_text"] = text_b

        if isinstance(ma, dict):
            payload["memory_a"] = {
                "id": a_id,
                "claim": self._mem_text(ma),
                "content": self._mem_text(ma),
                "type": ma.get("type"),
                "domain": ma.get("domain"),
                "details": self._mem_details(ma),
            }
        elif a_id and text_a:
            payload["memory_a"] = {
                "id": a_id,
                "claim": text_a,
                "content": text_a,
            }

        if isinstance(mb, dict):
            payload["memory_b"] = {
                "id": b_id,
                "claim": self._mem_text(mb),
                "content": self._mem_text(mb),
                "type": mb.get("type"),
                "domain": mb.get("domain"),
                "details": self._mem_details(mb),
            }
        elif b_id and text_b:
            payload["memory_b"] = {
                "id": b_id,
                "claim": text_b,
                "content": text_b,
            }

        out["payload"] = payload
        out["conflict"] = True
        if a_id and b_id:
            out["pair_ids"] = [a_id, b_id]
            out["conflict_pair_ids"] = [a_id, b_id]
        if not out.get("conflict_hint") and a_id and b_id:
            out["conflict_hint"] = f"{out.get('kind') or 'conflict'} between {a_id} and {b_id}"

        # Preserve harness behavior: top-level memory_ids can remain empty.
        # But when missing entirely, we still expose them internally for apply_feedback.
        if not isinstance(out.get("memory_ids"), list):
            out["memory_ids"] = []
        if out.get("memory_id") is None:
            out["memory_id"] = None

        return out

    def _resolved_candidate_mem_ids(self, c: Dict[str, Any]) -> List[str]:
        mem_ids = self._candidate_mem_ids(c)
        if mem_ids:
            return mem_ids[:2]
        return []

    # ---------------------------------------------------------------------
    # Truth / writeback helpers
    # ---------------------------------------------------------------------
    def _set_verification_status(self, m: Dict[str, Any], value: str) -> None:
        value = str(value).strip().lower()
        d = self._mem_details(m)

        d["verification_status"] = value
        d["status"] = value
        m["verification_status"] = value
        m["status"] = value

        if value == "verified":
            d["verified"] = True
            d["pending_verification"] = False
            d["contested"] = False
            d["recall_blocked"] = False
            d["conflict"] = False
            m["conflict"] = False
        elif value in {"unverified", "pending", "pending_external"}:
            d["verified"] = False
            d["pending_verification"] = True
        elif value == "contested":
            d["verified"] = False
            d["pending_verification"] = True
            d["contested"] = True
            d["recall_blocked"] = True
            d["conflict"] = True
            m["conflict"] = True
        elif value == "deprecated":
            d["superseded"] = True
            d["recall_blocked"] = True

    def _touch_verified(self, m: Dict[str, Any], *, when: Optional[float] = None) -> None:
        ts = float(when if when is not None else time.time())
        d = self._mem_details(m)
        old_risk = self._mem_staleness_risk(m)

        d["last_verified"] = ts
        d["staleness_risk"] = min(old_risk, 0.10)
        d["verified"] = True
        d["pending_verification"] = False
        d["contested"] = False
        d["recall_blocked"] = False
        d["conflict"] = False

        m["last_verified"] = ts
        m["staleness_risk"] = min(old_risk, 0.10)
        m["conflict"] = False

    def _set_conflict_link(self, a: Dict[str, Any], b_id: str, b_text: Optional[str] = None) -> None:
        d = self._mem_details(a)

        ids = self._mem_conflict_ids(a)
        if b_id and b_id not in ids:
            ids.append(b_id)
        ids = ids[: self.max_conflict_links]

        txts = self._mem_conflict_text(a)
        if isinstance(b_text, str) and b_text.strip() and b_text.strip() not in txts:
            txts.append(b_text.strip())
        txts = txts[: self.max_conflict_text_links]

        d["conflict_with_ids"] = ids
        d["conflicts"] = ids
        d["conflict_with_text"] = txts
        d["contested"] = True
        d["recall_blocked"] = True
        d["verification_status"] = "contested"
        d["status"] = "contested"
        d["conflict"] = True

        a["conflict_with_ids"] = ids
        a["conflicts"] = ids
        a["conflict_with_text"] = txts
        a["conflict"] = True
        a["verification_status"] = "contested"
        a["status"] = "contested"

    def _build_conflict_degree_map(self, mems: List[Dict[str, Any]]) -> Dict[str, int]:
        degree: Dict[str, int] = {}

        for m in mems:
            mid = self._mem_id(m)
            ids = self._mem_conflict_ids(m)
            degree[mid] = degree.get(mid, 0) + len(ids)
            for other in ids:
                degree[other] = degree.get(other, 0)

        return degree

    def _append_history(self, m: Dict[str, Any], value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            return
        d = self._mem_details(m)
        hist = d.get("history", m.get("history"))
        if not isinstance(hist, list):
            hist = []
        if value not in hist:
            hist.append(value)
        hist = hist[-self.max_history_items :]
        d["history"] = hist
        m["history"] = hist

    def _append_source_event(
        self,
        m: Dict[str, Any],
        event_type: str,
        *,
        origin: Optional[str] = None,
        detail: Optional[str] = None,
        timestamp: Optional[float] = None,
        external: Optional[bool] = None,
    ) -> None:
        ts = float(timestamp if timestamp is not None else time.time())
        d = self._mem_details(m)

        src = d.get("sources", m.get("sources"))
        if not isinstance(src, list):
            src = []

        evt: Dict[str, Any] = {"type": str(event_type), "timestamp": ts}
        if origin:
            evt["origin"] = str(origin)
        if detail:
            evt["detail"] = str(detail)
        if external is not None:
            evt["external"] = bool(external)

        signature = (
            evt.get("type"),
            evt.get("origin"),
            evt.get("detail"),
            evt.get("external"),
        )

        existing = set()
        compacted: List[Dict[str, Any]] = []
        for item in src:
            if not isinstance(item, dict):
                continue
            sig = (
                item.get("type"),
                item.get("origin"),
                item.get("detail"),
                item.get("external"),
            )
            if sig in existing:
                continue
            existing.add(sig)
            compacted.append(item)

        if signature not in existing:
            compacted.append(evt)

        compacted = compacted[-self.max_source_events :]
        d["sources"] = compacted
        m["sources"] = compacted

    def _merge_list_unique(self, a: Any, b: Any, *, cap: int) -> List[Any]:
        out: List[Any] = []
        seen = set()

        for seq in (a, b):
            if not isinstance(seq, list):
                continue
            for item in seq:
                try:
                    sig = repr(item)
                except Exception:
                    sig = str(item)
                if sig in seen:
                    continue
                seen.add(sig)
                out.append(item)

        return out[-cap:]

    def _merge_memory_metadata(self, keep: Dict[str, Any], other: Dict[str, Any]) -> None:
        kd = self._mem_details(keep)
        od = self._mem_details(other)

        merged_sources = self._merge_list_unique(
            kd.get("sources", keep.get("sources")),
            od.get("sources", other.get("sources")),
            cap=self.max_source_events,
        )
        kd["sources"] = merged_sources
        keep["sources"] = merged_sources

        merged_history = self._merge_list_unique(
            kd.get("history", keep.get("history")),
            od.get("history", other.get("history")),
            cap=self.max_history_items,
        )
        kd["history"] = merged_history
        keep["history"] = merged_history

        merged_conflicts = self._merge_list_unique(
            self._mem_conflict_ids(keep),
            [x for x in self._mem_conflict_ids(other) if x != self._mem_id(keep)],
            cap=self.max_conflict_links,
        )
        kd["conflict_with_ids"] = merged_conflicts
        kd["conflicts"] = merged_conflicts
        keep["conflict_with_ids"] = merged_conflicts
        keep["conflicts"] = merged_conflicts

        merged_conflict_text = self._merge_list_unique(
            self._mem_conflict_text(keep),
            self._mem_conflict_text(other),
            cap=self.max_conflict_text_links,
        )
        kd["conflict_with_text"] = merged_conflict_text
        keep["conflict_with_text"] = merged_conflict_text

        merged_evidence = self._merge_list_unique(
            self._mem_evidence(keep),
            self._mem_evidence(other),
            cap=self.max_evidence_items,
        )
        kd["evidence"] = merged_evidence
        keep["evidence"] = merged_evidence

        best_risk = min(self._mem_staleness_risk(keep), self._mem_staleness_risk(other))
        newest_verified = max(self._mem_last_verified(keep), self._mem_last_verified(other))

        kd["staleness_risk"] = best_risk
        keep["staleness_risk"] = best_risk

        if newest_verified > 0.0:
            kd["last_verified"] = newest_verified
            keep["last_verified"] = newest_verified

    # ---------------------------------------------------------------------
    # External verification scaffolding
    # ---------------------------------------------------------------------
    def needs_external_verification(self, m: Dict[str, Any]) -> bool:
        txt = self._mem_text(m)
        if not txt or self._is_calibration_log(m):
            return False

        vstat = self._mem_verification_status(m)
        stale = self._mem_staleness_risk(m)
        conf = self._mem_conf(m)
        conflicts = len(self._mem_conflict_ids(m))
        is_hyp = self._mem_is_hypothesis(m)
        source_count = self._mem_source_count(m)
        has_num = bool(self._number_pattern.search(txt))
        source = self._mem_source(m)

        if vstat in {"contested", "revised", "pending_external"}:
            return True
        if is_hyp and (stale >= 0.40 or conflicts > 0):
            return True
        if conflicts > 0 and has_num:
            return True
        if source in {"user_input", "reading"} and source_count == 0 and (has_num or conf >= self.external_verify_min_confidence):
            return True
        if source_count == 0 and has_num:
            return True
        if stale >= self.external_verify_min_staleness and vstat not in {"deprecated", "verified"}:
            return True

        return False

    def build_verification_packet(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        mid = self._mem_id(memory)
        txt = self._mem_text(memory)
        return {
            "type": "verification_request",
            "kind": "external_fact_check",
            "memory_id": mid,
            "claim": txt,
            "memory_class": self._mem_memory_class(memory),
            "verification_status": self._mem_verification_status(memory),
            "staleness_risk": self._mem_staleness_risk(memory),
            "conflict_with_ids": self._mem_conflict_ids(memory),
            "conflict_with_text": self._mem_conflict_text(memory),
            "hypothesis": self._mem_is_hypothesis(memory),
            "source_count": self._mem_source_count(memory),
            "source": self._mem_source(memory),
            "evidence": self._mem_evidence(memory),
            "last_verified": self._mem_last_verified(memory),
            "priority_hint": "high" if self.needs_external_verification(memory) else "normal",
            "ts": time.time(),
        }

    # ---------------------------------------------------------------------
    # Filters
    # ---------------------------------------------------------------------
    def _is_calibration_log(self, m: Dict[str, Any]) -> bool:
        d = self._mem_details(m)

        t = str(m.get("type") or "").lower()
        dom = str(m.get("domain") or "").lower()
        mem_class = self._mem_memory_class(m)

        tags = m.get("tags", d.get("tags", [])) or []
        tags_l = [str(x).lower() for x in tags] if isinstance(tags, list) else []

        if dom in {"operations", "calibration", "meta", "debug"}:
            return True

        if mem_class in {"meta_log", "operations"}:
            return True

        if "calibration" in tags_l or "sleep" in tags_l or "meta_log" in tags_l:
            return True

        txt = (self._mem_text(m) or "").strip().lower()
        if not txt:
            return True

        for p in self._cal_log_prefixes:
            if txt.startswith(p):
                return True

        if "report={" in txt or ("verdict=" in txt and "mem_ids=" in txt):
            return True

        if t in {"committed_action"}:
            return True

        return False

    # ---------------------------------------------------------------------
    # Memory fetching
    # ---------------------------------------------------------------------
    def _get_memories(self, limit: int = 250) -> List[Dict[str, Any]]:
        limit = max(1, int(limit))
        out: List[Dict[str, Any]] = []

        if self.clair is not None:
            fn = getattr(self.clair, "get_all_memories", None)
            if callable(fn):
                try:
                    rows = fn(limit=limit)
                    if isinstance(rows, list):
                        out = [r for r in rows if isinstance(r, dict)]
                except Exception:
                    pass

        if not out:
            try:
                buf = getattr(self.wm, "buffer", None)
                if isinstance(buf, list):
                    out.extend([m for m in buf if isinstance(m, dict)])
            except Exception:
                pass

            try:
                if self.ltm is not None and hasattr(self.ltm, "retrieve"):
                    try:
                        rows = self.ltm.retrieve(msg_type=None, limit=max(50, limit))
                    except TypeError:
                        rows = self.ltm.retrieve(None, max(50, limit))
                    if isinstance(rows, list):
                        out.extend([r for r in rows if isinstance(r, dict)])
            except Exception:
                pass

        seen = set()
        deduped: List[Dict[str, Any]] = []
        for m in out:
            try:
                key = ("id", self._mem_id(m))
            except Exception:
                key = (
                    "fallback",
                    m.get("type"),
                    self._mem_text(m),
                    m.get("domain"),
                    m.get("kind"),
                    m.get("memory_class"),
                    m.get("timestamp"),
                )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(m)

        return deduped[:limit]

    # ---------------------------------------------------------------------
    # Episode logging
    # ---------------------------------------------------------------------
    def _store_episode(
        self,
        content: str,
        tags: Optional[List[str]] = None,
        weight: float = 0.8,
        *,
        persist_to_ltm: Optional[bool] = None,
        domain: str = "operations",
        kind: str = "meta_log",
    ) -> None:
        if persist_to_ltm is None:
            persist_to_ltm = bool(self.persist_episodes)

        pkt = {
            "type": "observe",
            "content": str(content),
            "confidence": 0.70,
            "weight": float(weight),
            "context": [],
            "source": "calibration",
            "domain": domain,
            "tags": tags or ["calibration"],
            "kind": kind,
            "memory_class": "meta_log",
            "verification_status": "verified",
            "status": "verified",
            "hypothesis": False,
            "persisted": False,
            "timestamp": time.time(),
            "details": {
                "source": "calibration",
                "verified": True,
                "pending_verification": False,
                "verification_status": "verified",
                "status": "verified",
                "memory_class": "meta_log",
                "staleness_risk": "low",
            },
        }

        self.audit_log.append(pkt)

        try:
            if self.wm is not None and hasattr(self.wm, "store"):
                self.wm.store([pkt])
        except Exception:
            pass

        if persist_to_ltm:
            try:
                if self.ltm is not None and hasattr(self.ltm, "store"):
                    self.ltm.store([pkt])
                    pkt["persisted"] = True
            except Exception:
                pass

    # ---------------------------------------------------------------------
    # Repeat control helpers
    # ---------------------------------------------------------------------
    def _fingerprint_question(self, prompt: str) -> str:
        p = (prompt or "").strip().lower()
        p = self._ws.sub(" ", p)
        return hashlib.sha1(p.encode("utf-8", errors="ignore")).hexdigest()[:16]

    def _fp_is_recent(self, fp: str, now: float) -> bool:
        if not fp:
            return False

        pruned: Deque[Tuple[str, float]] = deque(maxlen=self._recent_fp.maxlen)
        for fpi, ts in self._recent_fp:
            if (now - ts) <= float(self.recent_fp_cooldown_sec):
                pruned.append((fpi, ts))
        self._recent_fp = pruned

        return any(fpi == fp for fpi, _ in self._recent_fp)

    def _mark_asked(self, mem_id: Optional[str], fp: Optional[str]) -> None:
        now = time.time()
        if mem_id:
            self._mem_last_asked[str(mem_id)] = now
        if fp:
            self._recent_fp.append((str(fp), now))

    def _mem_on_cooldown(self, mem_id: str, now: float) -> bool:
        if not mem_id:
            return False
        last = float(self._mem_last_asked.get(mem_id, 0.0) or 0.0)
        return (now - last) < float(self.mem_cooldown_sec)

    # ---------------------------------------------------------------------
    # Idle questions
    # ---------------------------------------------------------------------
    def _tokenize_topic(self, text: str) -> List[str]:
        t = (text or "").lower()
        t = self._number_pattern.sub(" ", t)
        t = self._nonword.sub(" ", t)
        toks = [x for x in self._ws.split(t.strip()) if x and x not in self._stop]
        return toks

    def _semantic_overlap(self, a: str, b: str) -> bool:
        ta = set(self._tokenize_topic(a))
        tb = set(self._tokenize_topic(b))
        if not ta or not tb:
            return False
        return len(ta.intersection(tb)) >= self.semantic_overlap_min

    def _priority_score(self, m: Dict[str, Any], conflict_degree_map: Dict[str, int]) -> float:
        txt = self._mem_text(m)
        conf = self._clamp01(self._mem_conf(m))
        usage = float(m.get("usage_count", 1) or 1)
        vstat = self._mem_verification_status(m)
        stale = self._mem_staleness_risk(m)
        direct_conflicts = len(self._mem_conflict_ids(m))
        conflict_degree = float(conflict_degree_map.get(self._mem_id(m), direct_conflicts))
        mem_class = self._mem_memory_class(m)
        is_hyp = self._mem_is_hypothesis(m)
        source_count = self._mem_source_count(m)
        verified_age_days = self._verified_age_days(m)
        explicit_conflict = self._mem_conflict_flag(m)

        uncertainty = 1.0 - conf
        numeric_bonus = 1.25 if self._number_pattern.search(txt) else 0.0

        verification_bonus = 0.0
        if vstat in {"unverified", "pending"}:
            verification_bonus += 1.25
        elif vstat == "contested":
            verification_bonus += 2.00
        elif vstat == "revised":
            verification_bonus += 0.90
        elif vstat == "deprecated":
            verification_bonus -= 2.0
        elif vstat == "verified":
            verification_bonus += 0.10 * stale
        elif vstat == "pending_external":
            verification_bonus += 1.10
        elif vstat == "hypothesis":
            verification_bonus += 1.30

        class_bonus = 0.0
        if mem_class in {"fact", "semantic", "rule", "numeric_fact", "claim"}:
            class_bonus += 0.75
        elif mem_class in {"episode", "meta_log", "operations"}:
            class_bonus -= 0.75

        weak_support_bonus = 0.0
        if source_count == 0:
            weak_support_bonus += 1.10
        elif source_count == 1:
            weak_support_bonus += 0.45

        hypothesis_bonus = 1.10 if is_hyp else 0.0
        if is_hyp and stale >= 0.50:
            hypothesis_bonus += 0.50
        if is_hyp and direct_conflicts > 0:
            hypothesis_bonus += 0.65

        verified_refresh_bonus = 0.0
        if vstat == "verified":
            if verified_age_days >= 90:
                verified_refresh_bonus += 0.60
            elif verified_age_days >= 30:
                verified_refresh_bonus += 0.30

        ext_bonus = 0.65 if self.needs_external_verification(m) else 0.0
        usage_bonus = min(1.25, 0.18 * max(0.0, usage - 1.0))
        explicit_conflict_bonus = 1.40 if explicit_conflict else 0.0

        return (
            (1.35 * uncertainty)
            + (1.10 * stale)
            + numeric_bonus
            + verification_bonus
            + class_bonus
            + weak_support_bonus
            + hypothesis_bonus
            + verified_refresh_bonus
            + ext_bonus
            + usage_bonus
            + explicit_conflict_bonus
            + (0.55 * direct_conflicts)
            + (0.35 * conflict_degree)
        )

    def select_question_candidates(self) -> List[Dict[str, Any]]:
        mems = self._get_memories(limit=250)
        conflict_degree_map = self._build_conflict_degree_map(mems)
        scored: List[Tuple[float, Dict[str, Any]]] = []

        for m in mems:
            if self._is_calibration_log(m):
                continue

            txt = self._mem_text(m)
            if not txt:
                continue

            priority = self._priority_score(m, conflict_degree_map)
            scored.append((priority, m))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[: max(1, self.question_limit * max(2, self.topk_pick))]]

    def build_question(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        mid = self._mem_id(memory)
        txt = self._mem_text(memory) or "Unknown"

        has_num = bool(self._number_pattern.search(txt))
        vstat = self._mem_verification_status(memory)
        stale = self._mem_staleness_risk(memory)
        mem_class = self._mem_memory_class(memory)
        is_hyp = self._mem_is_hypothesis(memory)
        conflict_ids = self._mem_conflict_ids(memory)
        conflict_text = self._mem_conflict_text(memory)
        source_count = self._mem_source_count(memory)
        priority = self._priority_score(memory, {mid: len(conflict_ids)})
        source = self._mem_source(memory)
        evidence = self._mem_evidence(memory)

        reason_bits: List[str] = []
        if vstat in {"unverified", "pending", "contested", "revised", "pending_external", "hypothesis"}:
            reason_bits.append(f"status={vstat}")
        if stale >= 0.50:
            reason_bits.append(f"stale={stale:.2f}")
        if conflict_ids:
            reason_bits.append(f"conflicts={len(conflict_ids)}")
        if source_count <= 1:
            reason_bits.append(f"sources={source_count}")
        if is_hyp:
            reason_bits.append("hypothesis")
        if mem_class:
            reason_bits.append(f"class={mem_class}")
        if source:
            reason_bits.append(f"source={source}")
        if self.needs_external_verification(memory):
            reason_bits.append("needs_external_verification")

        reason = " | ".join(reason_bits)

        if has_num:
            prompt = (
                f'Calibration check (numeric): "{txt}"\n'
                f"Reply confirm/deny, or provide correction with the correct value."
            )
        elif conflict_ids:
            prompt = (
                f'Calibration conflict check: "{txt}"\n'
                f"Reply confirm/deny/modify. This memory conflicts with other stored material."
            )
        else:
            prompt = f'Calibration check: "{txt}" — is this correct?'

        if reason:
            prompt += f"\nReason: {reason}"

        return {
            "id": f"q_{hashlib.sha1((mid + str(time.time())).encode()).hexdigest()[:10]}",
            "type": "calibration_question",
            "kind": "verify_memory",
            "memory_id": mid,
            "memory_ids": [mid],
            "claim": txt,
            "question": prompt,
            "prompt": prompt,
            "options": ["confirm", "deny", "modify", "unsure", "merge"],
            "verification_status": vstat,
            "staleness_risk": stale,
            "memory_class": mem_class,
            "conflict_with_ids": conflict_ids,
            "conflict_with_text": conflict_text,
            "hypothesis": is_hyp,
            "source_count": source_count,
            "source": source,
            "evidence": evidence,
            "priority_score": round(float(priority), 4),
            "needs_external_verification": self.needs_external_verification(memory),
            "verification_packet": self.build_verification_packet(memory),
            "ts": time.time(),
        }

    def idle_tick(self) -> Optional[Dict[str, Any]]:
        now = time.time()

        if self.cooldown_sec > 0.0 and (now - self.last_question_time) < self.cooldown_sec:
            return None

        # Prefer ACC if it can produce a question
        if self.acc is not None:
            fn = getattr(self.acc, "next_question", None)
            if callable(fn):
                try:
                    q = fn()
                    if isinstance(q, dict):
                        if self._is_conflict_candidate(q):
                            q = self._hydrate_conflict_candidate(q)

                        qt = str(q.get("prompt") or q.get("question") or q.get("conflict_hint") or "")
                        if qt:
                            fp = self._fingerprint_question(qt)
                            if self._fp_is_recent(fp, now):
                                return None

                            # mark both sides if possible
                            resolved = self._resolved_candidate_mem_ids(q)
                            if resolved:
                                for mid in resolved[:2]:
                                    self._mark_asked(mid, None)
                                self._mark_asked(resolved[0], fp)
                            else:
                                self._mark_asked(q.get("memory_id"), fp)

                            self.last_question_time = now
                            return q
                except Exception:
                    pass

        cands = self.select_question_candidates()
        if not cands:
            return None

        top = cands[: max(1, int(self.topk_pick))]
        viable: List[Tuple[float, Dict[str, Any]]] = []

        for idx, m in enumerate(top):
            mid = self._mem_id(m)
            if self._mem_on_cooldown(mid, now):
                continue

            q = self.build_question(m)
            qt = str(q.get("prompt") or q.get("question") or "")
            fp = self._fingerprint_question(qt)

            if self._fp_is_recent(fp, now):
                continue

            w = max(0.10, 1.0 - (idx / max(1.0, float(len(top)))))
            viable.append((w, q))

        if not viable:
            return None

        total = sum(w for w, _ in viable)
        r = self.rng.random() * total
        accw = 0.0
        picked = viable[0][1]

        for w, q in viable:
            accw += w
            if accw >= r:
                picked = q
                break

        fp = self._fingerprint_question(str(picked.get("prompt") or picked.get("question") or ""))
        self._mark_asked(picked.get("memory_id"), fp)

        self.last_question_time = now
        return picked

    # harness aliases
    def idle_step(self) -> Optional[Dict[str, Any]]:
        return self.idle_tick()

    def tick_idle(self) -> Optional[Dict[str, Any]]:
        return self.idle_tick()

    def tick(self) -> Optional[Dict[str, Any]]:
        return self.idle_tick()

    def run_idle_once(self) -> Optional[Dict[str, Any]]:
        return self.idle_tick()

    # ---------------------------------------------------------------------
    # Feedback apply
    # ---------------------------------------------------------------------
    def _find_memory_by_id(self, mem_id: str) -> Optional[Dict[str, Any]]:
        mems = self._get_memories(limit=400)
        for m in mems:
            if self._mem_id(m) == mem_id:
                return m
        return None

    def _write_back_best_effort(self, m: Dict[str, Any]) -> bool:
        if self.ltm is None:
            return False

        fn = getattr(self.ltm, "update_memory", None)
        if callable(fn):
            try:
                fn(m)
                return True
            except Exception:
                pass

        fn2 = getattr(self.ltm, "store", None)
        if callable(fn2):
            try:
                m2 = dict(m)
                m2["persisted"] = False
                m2["source"] = m2.get("source") or "calibration"
                fn2([m2])
                return True
            except Exception:
                pass

        return False

    def _apply_single_memory_feedback(
        self,
        m: Dict[str, Any],
        *,
        verdict: str,
        correction: Any,
        external_verified: bool,
        external_origin: Optional[str],
        external_detail: Any,
    ) -> Dict[str, Any]:
        d = self._mem_details(m)
        target_id = self._mem_id(m)
        conf = self._clamp01(self._mem_conf(m))
        conf_cap = self.confidence_cap_with_external if external_verified else self.confidence_cap_no_web

        if verdict == "confirm":
            conf = min(conf_cap, conf + self.promotion_step)
            m["confidence"] = conf
            d["confidence"] = conf
            self._set_verification_status(m, "verified")
            self._touch_verified(m)
            d["hypothesis"] = False
            m["hypothesis"] = False
            self._append_source_event(
                m,
                "external_confirm" if external_verified else "user_confirm",
                origin=external_origin or "user",
                detail=external_detail,
                external=external_verified,
            )
            self._write_back_best_effort(m)
            return {
                "id": target_id,
                "confidence": conf,
                "verification_status": self._mem_verification_status(m),
                "last_verified": m.get("last_verified", d.get("last_verified")),
                "staleness_risk": m.get("staleness_risk", d.get("staleness_risk")),
            }

        if verdict == "deny":
            conf = max(0.0, conf - self.demotion_step)
            m["confidence"] = conf
            d["confidence"] = conf
            self._set_verification_status(m, "contested")
            d["staleness_risk"] = max(self._mem_staleness_risk(m), 0.85)
            m["staleness_risk"] = d["staleness_risk"]
            self._append_source_event(
                m,
                "external_deny" if external_verified else "user_deny",
                origin=external_origin or "user",
                detail=external_detail,
                external=external_verified,
            )
            self._write_back_best_effort(m)
            return {
                "id": target_id,
                "confidence": conf,
                "verification_status": "contested",
                "staleness_risk": m.get("staleness_risk", d.get("staleness_risk")),
            }

        if verdict == "modify":
            if isinstance(correction, str) and correction.strip():
                old = self._mem_text(m)
                self._append_history(m, old)

                if "claim" in m:
                    m["claim"] = correction.strip()
                else:
                    m["content"] = correction.strip()

                self._set_verification_status(m, "revised")
                m["confidence"] = min(conf_cap, max(0.35, conf))
                d["confidence"] = m["confidence"]
                d["hypothesis"] = False
                m["hypothesis"] = False
                self._touch_verified(m)
                self._append_source_event(
                    m,
                    "external_modify" if external_verified else "user_modify",
                    origin=external_origin or "user",
                    detail=external_detail or correction.strip(),
                    external=external_verified,
                )
                self._write_back_best_effort(m)
                return {
                    "id": target_id,
                    "verification_status": "revised",
                    "last_verified": m.get("last_verified", d.get("last_verified")),
                }
            return {"id": target_id, "error": "modify verdict requires correction text."}

        # unsure
        self._set_verification_status(m, "unverified")
        m["confidence"] = max(0.0, conf - (0.25 * self.demotion_step))
        d["confidence"] = m["confidence"]
        d["staleness_risk"] = max(self._mem_staleness_risk(m), 0.60)
        m["staleness_risk"] = d["staleness_risk"]
        self._append_source_event(
            m,
            "external_unsure" if external_verified else "user_unsure",
            origin=external_origin or "user",
            detail=external_detail,
            external=external_verified,
        )
        self._write_back_best_effort(m)
        return {
            "id": target_id,
            "verification_status": "unverified",
            "staleness_risk": m.get("staleness_risk", d.get("staleness_risk")),
        }

    def apply_feedback(self, candidate: Any, feedback: Dict[str, Any]) -> Dict[str, Any]:
        c = candidate if isinstance(candidate, dict) else {"prompt": str(candidate)}
        f = feedback if isinstance(feedback, dict) else {"verdict": str(feedback)}

        if self._is_conflict_candidate(c):
            c = self._hydrate_conflict_candidate(c)

        verdict = str(f.get("verdict") or "unsure").strip().lower()
        if verdict not in {"confirm", "deny", "unsure", "modify", "merge"}:
            verdict = "unsure"

        mem_ids = self._resolved_candidate_mem_ids(c)
        correction = f.get("correction")
        notes = f.get("notes")
        external_verified = bool(f.get("external_verified", False))
        external_origin = (f.get("external_origin") or "external_check") if external_verified else None
        external_detail = f.get("external_detail")

        self._store_episode(
            content=f"Calibration feedback verdict={verdict} mem_ids={mem_ids} notes={notes} correction={correction}",
            tags=["calibration", "feedback", verdict],
            weight=0.55,
            persist_to_ltm=False,
            domain="operations",
            kind="meta_log",
        )

        applied = {
            "ok": True,
            "verdict": verdict,
            "memory_ids": mem_ids,
            "changed": [],
            "timestamp": time.time(),
        }

        # Merge still requires actual resolved memories
        if verdict == "merge":
            mems = [self._find_memory_by_id(mid) for mid in mem_ids]
            mems = [m for m in mems if isinstance(m, dict)]
            if len(mems) >= 2:
                mems_sorted = sorted(mems, key=lambda m: self._mem_conf(m), reverse=True)
                keep = mems_sorted[0]

                for other in mems_sorted[1:]:
                    self._merge_memory_metadata(keep, other)

                self._set_verification_status(keep, "verified")
                self._touch_verified(keep)
                self._mem_details(keep)["hypothesis"] = False
                keep["hypothesis"] = False
                self._append_source_event(
                    keep,
                    "merge_keep",
                    origin="calibration",
                    detail=f"merged={len(mems_sorted)-1}",
                    external=external_verified,
                )
                self._write_back_best_effort(keep)

                keep_id = self._mem_id(keep)
                keep_text = self._mem_text(keep)
                for drop in mems_sorted[1:]:
                    self._set_verification_status(drop, "deprecated")
                    drop["confidence"] = self._clamp01(self._mem_conf(drop) - self.demotion_step)
                    self._set_conflict_link(drop, keep_id, keep_text)
                    self._mem_details(drop)["hypothesis"] = False
                    drop["hypothesis"] = False
                    self._append_source_event(
                        drop,
                        "merge_drop",
                        origin="calibration",
                        detail=f"kept={keep_id}",
                        external=external_verified,
                    )
                    self._write_back_best_effort(drop)
                    applied["changed"].append({
                        "id": self._mem_id(drop),
                        "verification_status": "deprecated",
                    })

                applied["primary"] = keep_id
            else:
                applied["ok"] = False
                applied["error"] = "merge verdict requires at least two resolved memories."
            return applied

        if not mem_ids:
            applied["ok"] = False
            applied["error"] = "No memory_id provided in candidate."
            return applied

        # Conflict packets can act on both sides, especially for deny/confirm
        is_conflict = self._is_conflict_candidate(c)

        if is_conflict and len(mem_ids) >= 2:
            a = self._find_memory_by_id(mem_ids[0])
            b = self._find_memory_by_id(mem_ids[1])

            if not isinstance(a, dict) or not isinstance(b, dict):
                applied["ok"] = False
                applied["error"] = f"Conflict hydration failed for ids={mem_ids}"
                return applied

            a_id = self._mem_id(a)
            b_id = self._mem_id(b)
            a_text = self._mem_text(a)
            b_text = self._mem_text(b)

            # Keep link symmetry intact
            self._set_conflict_link(a, b_id, b_text)
            self._set_conflict_link(b, a_id, a_text)

            if verdict == "confirm":
                # confirming a conflict packet means "yes, this conflict is real"
                # so both remain contested and linked
                for m, other_id, other_text in ((a, b_id, b_text), (b, a_id, a_text)):
                    self._set_verification_status(m, "contested")
                    self._set_conflict_link(m, other_id, other_text)
                    self._append_source_event(
                        m,
                        "conflict_confirm",
                        origin=external_origin or "user",
                        detail=external_detail or f"linked={other_id}",
                        external=external_verified,
                    )
                    self._write_back_best_effort(m)
                    applied["changed"].append({
                        "id": self._mem_id(m),
                        "verification_status": "contested",
                        "conflict_with_ids": self._mem_conflict_ids(m),
                    })
                return applied

            if verdict == "deny":
                # denying a conflict packet means conflict should be cleared as best we can
                for m in (a, b):
                    d = self._mem_details(m)
                    ids = [x for x in self._mem_conflict_ids(m) if x not in {a_id, b_id}]
                    txts = [x for x in self._mem_conflict_text(m) if x not in {a_text, b_text}]
                    d["conflict_with_ids"] = ids
                    d["conflicts"] = ids
                    d["conflict_with_text"] = txts
                    m["conflict_with_ids"] = ids
                    m["conflicts"] = ids
                    m["conflict_with_text"] = txts
                    if not ids and not txts:
                        self._set_verification_status(m, "verified")
                        self._touch_verified(m)
                    self._append_source_event(
                        m,
                        "conflict_deny",
                        origin=external_origin or "user",
                        detail=external_detail,
                        external=external_verified,
                    )
                    self._write_back_best_effort(m)
                    applied["changed"].append({
                        "id": self._mem_id(m),
                        "verification_status": self._mem_verification_status(m),
                        "conflict_with_ids": self._mem_conflict_ids(m),
                    })
                return applied

            if verdict == "modify":
                # modify on a conflict packet applies to side A by convention
                # unless caller later adds explicit side targeting
                res = self._apply_single_memory_feedback(
                    a,
                    verdict=verdict,
                    correction=correction,
                    external_verified=external_verified,
                    external_origin=external_origin,
                    external_detail=external_detail,
                )
                if res.get("error"):
                    applied["ok"] = False
                    applied["error"] = res["error"]
                else:
                    applied["changed"].append(res)
                    # keep relationship visible until explicitly cleared
                    self._set_conflict_link(a, b_id, b_text)
                    self._set_conflict_link(b, a_id, self._mem_text(a))
                    self._write_back_best_effort(a)
                    self._write_back_best_effort(b)
                return applied

            if verdict == "unsure":
                for m, other_id, other_text in ((a, b_id, b_text), (b, a_id, a_text)):
                    self._set_verification_status(m, "pending_external")
                    self._set_conflict_link(m, other_id, other_text)
                    self._append_source_event(
                        m,
                        "conflict_unsure",
                        origin=external_origin or "user",
                        detail=external_detail,
                        external=external_verified,
                    )
                    self._write_back_best_effort(m)
                    applied["changed"].append({
                        "id": self._mem_id(m),
                        "verification_status": "pending_external",
                        "conflict_with_ids": self._mem_conflict_ids(m),
                    })
                return applied

        # Single-memory fallback path
        target_id = mem_ids[0] if mem_ids else None
        if not target_id:
            applied["ok"] = False
            applied["error"] = "No memory_id provided in candidate."
            return applied

        m = self._find_memory_by_id(target_id)
        if not isinstance(m, dict):
            applied["ok"] = False
            applied["error"] = f"Memory not found for id={target_id}"
            return applied

        result = self._apply_single_memory_feedback(
            m,
            verdict=verdict,
            correction=correction,
            external_verified=external_verified,
            external_origin=external_origin,
            external_detail=external_detail,
        )
        if result.get("error"):
            applied["ok"] = False
            applied["error"] = result["error"]
        else:
            applied["changed"].append(result)

        return applied

    # harness aliases
    def answer_question(self, candidate: Any, feedback: Dict[str, Any]) -> Dict[str, Any]:
        return self.apply_feedback(candidate, feedback)

    def submit_feedback(self, candidate: Any, feedback: Dict[str, Any]) -> Dict[str, Any]:
        return self.apply_feedback(candidate, feedback)

    def handle_user_answer(self, candidate: Any, feedback: Dict[str, Any]) -> Dict[str, Any]:
        return self.apply_feedback(candidate, feedback)

    def on_user_feedback(self, candidate: Any, feedback: Dict[str, Any]) -> Dict[str, Any]:
        return self.apply_feedback(candidate, feedback)

    # ---------------------------------------------------------------------
    # Sleep calibration
    # ---------------------------------------------------------------------
    def _topic_key(self, text: str, max_len: int = 80) -> str:
        toks = self._tokenize_topic(text)
        if not toks:
            return ""
        k = " ".join(toks[:8])
        return k[:max_len]

    def _detect_duplicates(self, mems: List[Dict[str, Any]], *, deadline: float) -> List[List[str]]:
        seen: Dict[str, List[str]] = {}
        for m in mems:
            if time.time() > deadline:
                break
            if self._is_calibration_log(m):
                continue
            txt = self._mem_text(m).strip().lower()
            if not txt:
                continue
            seen.setdefault(txt, []).append(self._mem_id(m))
        return [ids for ids in seen.values() if len(ids) > 1]

    def _detect_numeric_conflicts(self, mems: List[Dict[str, Any]], *, deadline: float) -> List[Tuple[str, str]]:
        buckets: Dict[str, List[Tuple[str, str, List[str], str]]] = {}
        numeric_items = 0

        for m in mems:
            if time.time() > deadline:
                break
            if self._is_calibration_log(m):
                continue

            txt = self._mem_text(m)
            if not txt:
                continue

            nums = self._number_pattern.findall(txt)
            if not nums:
                continue

            numeric_items += 1
            if numeric_items > self.max_numeric_items:
                break

            key = self._topic_key(txt)
            if not key:
                continue

            buckets.setdefault(key, []).append((self._mem_id(m), txt, nums, self._mem_text(m)))

        conflicts: List[Tuple[str, str]] = []
        pairs = 0

        for _, items in buckets.items():
            if time.time() > deadline:
                break
            if len(items) < 2:
                continue

            items = items[:25]

            for i in range(len(items)):
                if time.time() > deadline:
                    return conflicts
                id1, t1, n1, _raw1 = items[i]
                for j in range(i + 1, len(items)):
                    pairs += 1
                    if pairs > self.max_pairs:
                        return conflicts
                    id2, t2, n2, _raw2 = items[j]
                    if n1 == n2:
                        continue
                    if self._semantic_overlap(t1, t2):
                        conflicts.append((id1, id2))

        return conflicts

    def _decay_stale(self, mems: List[Dict[str, Any]], *, deadline: float) -> int:
        now = time.time()
        stale_sec = 60 * 60 * 24 * int(self.stale_days)
        decayed = 0

        for m in mems:
            if time.time() > deadline:
                break
            if self._is_calibration_log(m):
                continue

            vstat = self._mem_verification_status(m)
            if vstat == "deprecated":
                continue

            last_verified = self._mem_last_verified(m)
            current_risk = self._mem_staleness_risk(m)
            d = self._mem_details(m)

            should_decay = False
            if last_verified <= 0.0:
                should_decay = True
            elif (now - last_verified) > stale_sec:
                should_decay = True
            elif current_risk >= 0.85:
                should_decay = True

            if should_decay:
                conf = self._clamp01(self._mem_conf(m))
                decay_factor = self._class_decay_factor(m)

                if vstat == "verified":
                    decay_factor *= 0.60

                effective_decay = 1.0 - ((1.0 - self.decay_multiplier) * decay_factor)
                new_conf = self._clamp01(conf * effective_decay)
                if new_conf < conf:
                    m["confidence"] = new_conf
                    d["confidence"] = new_conf
                    d["staleness_risk"] = min(1.0, max(current_risk, 0.75))
                    m["staleness_risk"] = d["staleness_risk"]
                    self._append_source_event(m, "sleep_decay", origin="sleep", external=False, timestamp=now)
                    if self._write_back_best_effort(m):
                        decayed += 1

        return decayed

    def sleep_cycle(self) -> Dict[str, int]:
        t0 = time.time()
        deadline = t0 + float(self.sleep_time_budget_sec)

        mems = self._get_memories(limit=400)
        report: Dict[str, int] = {"conflicts": 0, "merged": 0, "decayed": 0}

        dups = self._detect_duplicates(mems, deadline=deadline)
        merged = 0
        for ids in dups:
            if time.time() > deadline:
                break

            mm = [self._find_memory_by_id(mid) for mid in ids]
            mm = [m for m in mm if isinstance(m, dict)]
            if len(mm) < 2:
                continue

            mm_sorted = sorted(mm, key=lambda m: self._mem_conf(m), reverse=True)
            keep = mm_sorted[0]
            keep_id = self._mem_id(keep)

            for other in mm_sorted[1:]:
                self._merge_memory_metadata(keep, other)

            self._set_verification_status(keep, "verified")
            self._touch_verified(keep)
            self._mem_details(keep)["hypothesis"] = False
            keep["hypothesis"] = False
            self._append_source_event(keep, "sleep_merge_keep", origin="sleep", detail=f"merged={len(mm_sorted)-1}")
            self._write_back_best_effort(keep)

            for drop in mm_sorted[1:]:
                self._set_verification_status(drop, "deprecated")
                drop["confidence"] = self._clamp01(self._mem_conf(drop) - 0.5 * self.demotion_step)
                self._set_conflict_link(drop, keep_id, self._mem_text(keep))
                self._mem_details(drop)["hypothesis"] = False
                drop["hypothesis"] = False
                self._append_source_event(drop, "sleep_merge_drop", origin="sleep", detail=f"kept={keep_id}")
                if self._write_back_best_effort(drop):
                    merged += 1

        report["merged"] = merged

        conflicts = self._detect_numeric_conflicts(mems, deadline=deadline)
        report["conflicts"] = len(conflicts)

        for a, b in conflicts:
            if time.time() > deadline:
                break

            ma = self._find_memory_by_id(a)
            mb = self._find_memory_by_id(b)
            if not (isinstance(ma, dict) and isinstance(mb, dict)):
                continue

            weak = mb if self._mem_conf(ma) >= self._mem_conf(mb) else ma
            other = ma if weak is mb else mb
            other_id = self._mem_id(other)
            other_text = self._mem_text(other)

            self._set_verification_status(weak, "contested")
            self._set_conflict_link(weak, other_id, other_text)
            weak["confidence"] = self._clamp01(self._mem_conf(weak) - self.demotion_step)
            self._mem_details(weak)["confidence"] = weak["confidence"]
            self._mem_details(weak)["staleness_risk"] = max(self._mem_staleness_risk(weak), 0.80)
            weak["staleness_risk"] = self._mem_details(weak)["staleness_risk"]
            self._append_source_event(weak, "sleep_conflict", origin="sleep", detail=f"other={other_id}")
            self._write_back_best_effort(weak)

        report["decayed"] = self._decay_stale(mems, deadline=deadline)

        dt = time.time() - t0
        self._store_episode(
            content=(
                f"Sleep calibration ran: merged={report['merged']} "
                f"conflicts={report['conflicts']} decayed={report['decayed']} dt={dt:.3f}s"
            ),
            tags=["calibration", "sleep", "meta_log"],
            weight=0.25,
            persist_to_ltm=False,
            domain="operations",
            kind="meta_log",
        )

        return report

    def sleep_tick(self) -> Dict[str, Any]:
        return {"ok": True, "report": self.sleep_cycle(), "timestamp": time.time()}

    # harness aliases
    def sleep_step(self) -> Dict[str, Any]:
        return self.sleep_tick()

    def tick_sleep(self) -> Dict[str, Any]:
        return self.sleep_tick()

    def run_sleep_once(self) -> Dict[str, Any]:
        return self.sleep_tick()

    def recalibrate(self) -> Dict[str, Any]:
        return self.sleep_tick()