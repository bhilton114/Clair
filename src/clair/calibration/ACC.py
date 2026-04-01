# FILE: calibration/ACC.py
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class AuditItem:
    qid: str
    kind: str
    memory_ids: List[str]
    payload: Dict[str, Any]


class ACC:
    VERSION = "3.6"

    def __init__(self, *args: Any):
        self.clair = None
        self.wm = None
        self.ltm = None

        if len(args) == 1:
            a0 = args[0]

            if self._looks_like_working_memory(a0):
                self.wm = a0
            elif self._looks_like_long_term_memory(a0):
                self.ltm = a0
            elif self._looks_like_clair(a0):
                self.clair = a0
                self.wm = getattr(a0, "memory", None)
                self.ltm = getattr(a0, "long_term", None)

        elif len(args) >= 2:
            a0, a1 = args[0], args[1]

            if self._looks_like_working_memory(a0):
                self.wm = a0
            elif self._looks_like_long_term_memory(a0):
                self.ltm = a0
            elif self._looks_like_clair(a0):
                self.clair = a0
                self.wm = getattr(a0, "memory", None)
                self.ltm = getattr(a0, "long_term", None)

            if self._looks_like_working_memory(a1) and self.wm is None:
                self.wm = a1
            elif self._looks_like_long_term_memory(a1) and self.ltm is None:
                self.ltm = a1

        self._num_pat = re.compile(r"\d+(?:\.\d+)?")
        self._memid_pat = re.compile(r"\bmem_[a-z0-9]{6,128}\b", re.IGNORECASE)
        self._ws = re.compile(r"\s+")
        self._tok_pat = re.compile(r"[a-z0-9']+")
        self._neg_pat = re.compile(
            r"\b(no|not|never|cannot|can't|isn't|aren't|doesn't|without|false|incorrect)\b",
            re.IGNORECASE,
        )

        self._queue: List[AuditItem] = []
        self._cursor = 0
        self._last_refresh = 0.0
        self.feedback_log: List[Dict[str, Any]] = []

        self.refresh_interval_sec = 2.0
        self.max_pairs = 6000
        self.semantic_overlap_min_tokens = 3
        self.stale_days = 60

        self._asked: Dict[str, float] = {}
        self._asked_ttl_sec = 120.0

        self.max_conflict_items_per_refresh = 12
        self.max_duplicate_items_per_refresh = 6
        self.include_drift_item = True

        self.max_text_len = 600
        self.min_text_len = 6

        self._ban_phrases = (
            "sleep calibration report",
            "calibration feedback",
            "calibration report",
            "verdict=",
            "mem_ids=",
            "applied_via",
            "hydration_kind",
            "idle_candidates",
            "idle_applied",
        )
        self._ban_type_tokens = (
            "calibration", "audit", "report", "telemetry", "debug", "trace"
        )

        self._procedural_markers = (
            "minute", "minutes", "hour", "hours",
            "step", "steps", "phase", "day",
            "v2", "version", "rollout", "rollouts", "scale", "horizon",
            "rule of", "options", "priorities", "timebox",
            "confirm_rate", "confirm rate",
            "benchmark",
        )

        self._stop = {
            "a", "an", "the", "and", "or", "to", "of", "in", "on", "at", "for",
            "if", "then", "than", "is", "are", "was", "were", "be", "being", "been",
            "it", "this", "that", "these", "those", "with", "as", "by", "from",
            "into", "over", "under", "do", "does", "did", "not", "no", "never",
            "only", "can", "could", "should", "would", "will", "may", "might",
            "you", "your", "we", "our", "i", "me", "my", "they", "them", "their",
            "have", "has", "had",
        }

        self._topics = {
            "fire", "flood", "earthquake", "aftershock", "smoke",
            "boil", "boils", "boiling", "celsius", "c",
            "bones", "bone",
            "everest", "meters", "meter",
            "lost", "shelter",
            "human", "humans",
            "water", "sea", "level",
        }

    # ------------------------------------------------------------------
    # Constructor helpers
    # ------------------------------------------------------------------
    def _looks_like_clair(self, value: Any) -> bool:
        return (
            value is not None
            and hasattr(value, "memory")
            and hasattr(value, "long_term")
            and not hasattr(value, "buffer")
        )

    def _looks_like_working_memory(self, value: Any) -> bool:
        return (
            value is not None
            and hasattr(value, "buffer")
            and hasattr(value, "retrieve")
            and hasattr(value, "store")
        )

    def _looks_like_long_term_memory(self, value: Any) -> bool:
        return (
            value is not None
            and hasattr(value, "retrieve")
            and hasattr(value, "store")
            and not hasattr(value, "buffer")
        )

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------
    def _safe_str(self, value: Any) -> str:
        if value is None:
            return ""
        try:
            return str(value).strip()
        except Exception:
            return ""

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    def _enum_value(self, value: Any) -> str:
        if isinstance(value, Enum):
            try:
                return str(value.value).strip().lower()
            except Exception:
                return str(value).strip().lower()
        return self._safe_str(value).lower()

    def _record_metadata(self, record: Any) -> Dict[str, Any]:
        if isinstance(record, dict):
            return record
        meta = getattr(record, "metadata", None)
        return meta if isinstance(meta, dict) else {}

    def _record_details(self, record: Any) -> Dict[str, Any]:
        if isinstance(record, dict):
            details = record.get("details")
            return details if isinstance(details, dict) else {}
        meta = self._record_metadata(record)
        details = meta.get("details")
        return details if isinstance(details, dict) else {}

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------
    def _coerce_to_legacy_dict(self, record: Any) -> Optional[Dict[str, Any]]:
        if isinstance(record, dict):
            return dict(record)

        if self.wm is not None:
            converter = getattr(self.wm, "_record_to_legacy_dict", None)
            if callable(converter):
                try:
                    out = converter(record)
                    if isinstance(out, dict):
                        return out
                except Exception:
                    pass

        meta = self._record_metadata(record)
        details = self._record_details(record)

        text = ""
        for attr in ("text", "content", "summary", "claim"):
            s = self._safe_str(getattr(record, attr, None))
            if s:
                text = s
                break
        if not text:
            for key in ("claim", "content", "text"):
                s = self._safe_str(meta.get(key))
                if s:
                    text = s
                    break

        if not text:
            return None

        memory_id = ""
        for attr in ("memory_id", "id"):
            s = self._safe_str(getattr(record, attr, None)).lower()
            if s and s != "none":
                memory_id = s
                break
        if not memory_id:
            for key in ("memory_id", "id", "ltm_id"):
                s = self._safe_str(meta.get(key)).lower()
                if s and s != "none":
                    memory_id = s
                    break
        if not memory_id:
            h = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:12]
            memory_id = f"mem_{h}"

        mtype = self._safe_str(getattr(record, "type", None) or meta.get("type")).lower()
        domain = self._safe_str(getattr(record, "domain", None) or meta.get("domain")).lower()

        kind_raw = getattr(record, "kind", None)
        kind = self._enum_value(kind_raw) or self._safe_str(meta.get("kind")).lower()

        conf = getattr(record, "confidence", None)
        if conf is None:
            conf = meta.get("confidence", details.get("confidence", 0.5))

        vstat = (
            getattr(record, "verification_status", None)
            or meta.get("verification_status")
            or details.get("verification_status")
            or details.get("status")
            or meta.get("status")
        )
        status = details.get("status") or meta.get("status") or vstat

        tags = getattr(record, "tags", None)
        if tags is None:
            tags = meta.get("tags", [])

        evidence = getattr(record, "evidence", None)
        if evidence is None:
            evidence = meta.get("evidence", [])

        out = {
            "id": memory_id,
            "memory_id": memory_id,
            "type": mtype,
            "content": text,
            "claim": text,
            "domain": domain,
            "kind": kind,
            "confidence": self._safe_float(conf, 0.5),
            "verification_status": self._safe_str(vstat).lower(),
            "status": self._safe_str(status).lower(),
            "details": details,
            "timestamp": meta.get("timestamp"),
            "updated_at": meta.get("updated_at"),
            "last_used": meta.get("last_used"),
            "tags": list(tags or []),
            "evidence": evidence if evidence is not None else [],
            "conflict": bool(meta.get("conflict", False) or details.get("conflict", False)),
        }

        for key in (
            "source", "source_ref", "persisted", "layer", "store", "memory_store",
            "source_layer", "doc_id", "document_id", "origin", "path", "file_path"
        ):
            if key in meta and key not in out:
                out[key] = meta.get(key)

        return out

    # ------------------------------------------------------------------
    # Memory adapters
    # ------------------------------------------------------------------
    def _mem_text(self, m: Any) -> str:
        if isinstance(m, dict):
            for key in ("claim", "content", "text"):
                v = m.get(key)
                if isinstance(v, str) and v.strip():
                    return v.strip()
        return ""

    def _mem_details(self, m: Any) -> Dict[str, Any]:
        details = m.get("details")
        return details if isinstance(details, dict) else {}

    def _mem_type(self, m: Any) -> str:
        return self._safe_str(m.get("type")).lower()

    def _mem_domain(self, m: Any) -> str:
        return self._safe_str(m.get("domain")).lower()

    def _mem_kind(self, m: Any) -> str:
        return self._safe_str(m.get("kind")).lower()

    def _mem_id(self, m: Any) -> str:
        for k in ("memory_id", "id", "uid", "pk"):
            s = self._safe_str(m.get(k)).lower()
            if s and s != "none":
                return s
        base = f"{self._mem_type(m)}|{self._mem_domain(m)}|{self._mem_text(m)}"
        h = hashlib.sha1(base.encode("utf-8", errors="ignore")).hexdigest()[:12]
        return f"mem_{h}".lower()

    def _mem_confidence(self, m: Any) -> float:
        if m.get("confidence") is not None:
            return self._safe_float(m.get("confidence"), 0.5)
        return self._safe_float(self._mem_details(m).get("confidence"), 0.5)

    def _mem_status(self, m: Any) -> str:
        d = self._mem_details(m)
        return self._safe_str(d.get("status") or m.get("status")).lower()

    def _mem_verification_status(self, m: Any) -> str:
        d = self._mem_details(m)
        return self._safe_str(
            m.get("verification_status")
            or d.get("verification_status")
            or d.get("status")
            or m.get("status")
        ).lower()

    def _mem_last_used(self, m: Any) -> float:
        now = time.time()
        d = self._mem_details(m)

        for k in ("last_verified",):
            ts = self._safe_float(m.get(k, d.get(k, 0.0)), 0.0)
            if ts > 0.0:
                return ts

        for k in ("last_used", "last_access", "updated_at", "created_at", "timestamp"):
            ts = self._safe_float(m.get(k, 0.0), 0.0)
            if ts > 0.0:
                return ts

        return now

    def _mem_evidence_count(self, m: Any) -> int:
        d = self._mem_details(m)
        ev = m.get("evidence", d.get("evidence", []))
        if isinstance(ev, list):
            return len(ev)
        if isinstance(ev, dict):
            items = ev.get("items", [])
            if isinstance(items, list):
                return len(items)
        return 0

    def _mem_conflict_flag(self, m: Dict[str, Any]) -> bool:
        d = self._mem_details(m)
        if bool(m.get("conflict", False)):
            return True
        if bool(d.get("conflict", False)):
            return True
        if bool(d.get("contested", False)):
            return True
        if d.get("conflict_with_ids") or d.get("conflict_with_text"):
            return True
        return self._mem_verification_status(m) in {"contested", "disputed"}

    def _norm_text(self, s: str) -> str:
        return self._ws.sub(" ", str(s or "").strip())

    def _norm_text_low(self, s: str) -> str:
        return self._norm_text(s).lower()

    def _fp_text(self, s: str) -> str:
        blob = self._norm_text_low(s)
        return hashlib.sha1(blob.encode("utf-8", errors="ignore")).hexdigest()[:12]

    def _pair_key(self, a: str, b: str) -> str:
        x = str(a or "").strip().lower()
        y = str(b or "").strip().lower()
        return f"{x}|{y}" if x <= y else f"{y}|{x}"

    def _extract_mem_ids_from_any(self, value: Any) -> List[str]:
        out: List[str] = []

        def add(v: Any) -> None:
            if isinstance(v, str):
                hits = self._memid_pat.findall(v)
                if hits:
                    out.extend([str(x).strip().lower() for x in hits])
                else:
                    s = v.strip().lower()
                    if s.startswith("mem_"):
                        out.append(s)
            elif isinstance(v, dict):
                for key in (
                    "id", "memory_id", "uid", "pk",
                    "memory_a_id", "memory_b_id", "a_id", "b_id", "id_a", "id_b",
                    "ltm_id",
                ):
                    vv = v.get(key)
                    if vv is not None:
                        add(str(vv))
                for vv in v.values():
                    if isinstance(vv, (str, list, tuple, dict, set)):
                        add(vv)
            elif isinstance(v, (list, tuple, set)):
                for item in v:
                    add(item)

        add(value)

        deduped: List[str] = []
        seen: Set[str] = set()
        for x in out:
            s = str(x).strip().lower()
            if s and s not in seen:
                seen.add(s)
                deduped.append(s)
        return deduped

    def _extract_texts_from_any(self, value: Any) -> List[str]:
        out: List[str] = []

        def add(v: Any) -> None:
            if isinstance(v, str):
                s = v.strip()
                if s:
                    out.append(s)
            elif isinstance(v, dict):
                for key in (
                    "text", "content", "claim",
                    "text_a", "text_b", "claim_a", "claim_b",
                    "side_a", "side_b", "memory_a_text", "memory_b_text",
                ):
                    vv = v.get(key)
                    if isinstance(vv, str) and vv.strip():
                        out.append(vv.strip())
                for vv in v.values():
                    if isinstance(vv, (str, list, tuple, dict, set)):
                        add(vv)
            elif isinstance(v, (list, tuple, set)):
                for item in v:
                    add(item)

        add(value)

        deduped: List[str] = []
        seen: Set[str] = set()
        for x in out:
            s = self._norm_text(x)
            if s and s.lower() not in seen:
                seen.add(s.lower())
                deduped.append(s)
        return deduped

    def _mem_conflict_with_ids(self, m: Dict[str, Any]) -> List[str]:
        d = self._mem_details(m)
        out: List[str] = []

        candidates = [
            d.get("conflict_with_ids", []),
            d.get("conflict_with_id"),
            d.get("conflict_ids"),
            d.get("linked_ids"),
            m.get("conflict_with_ids", []),
            m.get("conflict_with_id"),
            m.get("conflict_ids"),
        ]

        for cand in candidates:
            out.extend(self._extract_mem_ids_from_any(cand))

        deduped: List[str] = []
        seen: Set[str] = set()
        for x in out:
            s = str(x).strip().lower()
            if s and s not in seen:
                seen.add(s)
                deduped.append(s)
        return deduped

    def _mem_conflict_with_texts(self, m: Dict[str, Any]) -> List[str]:
        d = self._mem_details(m)
        out: List[str] = []

        candidates = [
            d.get("conflict_with_text", []),
            d.get("conflict_with"),
            m.get("conflict_with_text", []),
            m.get("conflict_with"),
        ]

        for cand in candidates:
            out.extend(self._extract_texts_from_any(cand))

        deduped: List[str] = []
        seen: Set[str] = set()
        for x in out:
            s = self._norm_text(x)
            if s and s.lower() not in seen:
                seen.add(s.lower())
                deduped.append(s)
        return deduped

    def _is_system_memory(self, m: Dict[str, Any]) -> bool:
        try:
            combo = f"{self._mem_type(m)} {self._mem_kind(m)} {self._mem_domain(m)}"
            if any(tok in combo for tok in self._ban_type_tokens):
                return True
        except Exception:
            pass

        low = (self._mem_text(m) or "").lower()
        if not low:
            return True
        if len(low) < self.min_text_len:
            return True
        if len(low) > self.max_text_len:
            return True
        if any(p in low for p in self._ban_phrases):
            return True
        if ("{" in low and "}" in low and ":" in low and "report" in low):
            return True
        return False

    def _infer_layer(self, m: Dict[str, Any]) -> str:
        # Prefer explicitly stamped ACC provenance first.
        stamped = self._safe_str(m.get("_acc_layer")).lower()
        if stamped in {"wm", "ltm", "clair", "unknown"}:
            return stamped

        mt = self._mem_type(m)
        mk = self._mem_kind(m)
        md = self._mem_domain(m)
        details = self._mem_details(m)

        for key in ("layer", "store", "memory_store", "source_layer"):
            value = self._safe_str(m.get(key) or details.get(key)).lower()
            if value in {"wm", "working", "working_memory"}:
                return "wm"
            if value in {"ltm", "long_term", "long_term_memory"}:
                return "ltm"

        persisted = m.get("persisted", details.get("persisted"))
        if persisted is True:
            return "ltm"
        if persisted is False:
            return "wm"

        combo = f"{mt} {mk} {md}".lower()
        if "working" in combo or "wm" in combo:
            return "wm"
        if "long_term" in combo or "ltm" in combo:
            return "ltm"

        return "unknown"

    def _source_key(self, m: Dict[str, Any]) -> str:
        details = self._mem_details(m)
        for key in (
            "source_id", "doc_id", "document_id", "source", "origin",
            "book", "chapter", "url", "file_path", "path"
        ):
            value = self._safe_str(m.get(key) or details.get(key)).lower()
            if value:
                return value
        return ""

    def _stamp_layer(self, m: Dict[str, Any], layer: str) -> Dict[str, Any]:
        out = dict(m)
        current = self._safe_str(out.get("_acc_layer")).lower()
        layer = self._safe_str(layer).lower()

        if current in {"wm", "ltm"}:
            return out
        if layer in {"wm", "ltm", "clair", "unknown"}:
            out["_acc_layer"] = layer
        return out

    def _collect_clair_rows(self, limit: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if self.clair is None:
            return out

        fn = getattr(self.clair, "get_all_memories", None)
        if not callable(fn):
            return out

        try:
            rows = fn(limit=limit)
            if not isinstance(rows, list):
                return out
        except Exception:
            return out

        for raw in rows:
            m = self._coerce_to_legacy_dict(raw)
            if not isinstance(m, dict):
                continue
            out.append(self._stamp_layer(m, "clair"))

        return out

    def _collect_wm_rows(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        try:
            buf = getattr(self.wm, "buffer", None)
            if not isinstance(buf, list):
                return out
        except Exception:
            return out

        for raw in buf:
            m = self._coerce_to_legacy_dict(raw)
            if not isinstance(m, dict):
                continue
            out.append(self._stamp_layer(m, "wm"))

        return out

    def _collect_ltm_rows(self, limit: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        try:
            if self.ltm is None or not hasattr(self.ltm, "retrieve"):
                return out

            try:
                rows = self.ltm.retrieve(msg_type=None, limit=max(50, limit))
            except TypeError:
                rows = self.ltm.retrieve(None, max(50, limit))

            if not isinstance(rows, list):
                return out
        except Exception:
            return out

        for raw in rows:
            m = self._coerce_to_legacy_dict(raw)
            if not isinstance(m, dict):
                continue
            out.append(self._stamp_layer(m, "ltm"))

        return out

    def get_memories(self, limit: int = 250) -> List[Dict[str, Any]]:
        limit = max(1, int(limit))

        if self.wm is None and self.clair is not None:
            self.wm = getattr(self.clair, "memory", None)
        if self.ltm is None and self.clair is not None:
            self.ltm = getattr(self.clair, "long_term", None)

        raw_items: List[Dict[str, Any]] = []
        raw_items.extend(self._collect_clair_rows(limit))
        raw_items.extend(self._collect_wm_rows())
        raw_items.extend(self._collect_ltm_rows(limit))

        cleaned: List[Dict[str, Any]] = []
        seen: Set[Tuple[str, str, str, str, str, str]] = set()

        for m in raw_items:
            if not isinstance(m, dict):
                continue
            if self._is_system_memory(m):
                continue

            layer = self._infer_layer(m)
            key = (
                self._mem_id(m),
                self._mem_type(m),
                self._mem_domain(m),
                self._mem_text(m),
                self._mem_kind(m),
                layer,
            )
            if key in seen:
                continue
            seen.add(key)

            m["_acc_layer"] = layer
            cleaned.append(m)

        return cleaned[:limit]

    # ------------------------------------------------------------------
    # Canonical claim helpers
    # ------------------------------------------------------------------
    def _numeric_signature(self, text: str) -> Tuple[str, ...]:
        return tuple(self._num_pat.findall(text or ""))

    def _canonical_claim_key(self, m: Dict[str, Any]) -> str:
        text = self._norm_text_low(self._mem_text(m))
        mtype = self._mem_type(m)
        domain = self._mem_domain(m)
        kind = self._mem_kind(m)
        source = self._source_key(m)
        nums = "|".join(self._numeric_signature(text))
        blob = f"{text}|{mtype}|{domain}|{kind}|{source}|{nums}"
        return hashlib.sha1(blob.encode("utf-8", errors="ignore")).hexdigest()[:16]

    def build_canonical_claims(
        self,
        memories: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        memories = memories if isinstance(memories, list) else self.get_memories(limit=500)

        buckets: Dict[str, Dict[str, Any]] = {}

        for m in memories:
            text = self._mem_text(m)
            if not text:
                continue

            ckey = self._canonical_claim_key(m)
            mid = self._mem_id(m)
            layer = self._safe_str(m.get("_acc_layer") or self._infer_layer(m)).lower() or "unknown"

            bucket = buckets.get(ckey)
            if bucket is None:
                bucket = {
                    "claim_key": ckey,
                    "canonical_text": text,
                    "normalized_text": self._norm_text_low(text),
                    "memory_ids": [],
                    "layers": [],
                    "row_count": 0,
                    "texts": [],
                    "records": [],
                    "type": self._mem_type(m),
                    "domain": self._mem_domain(m),
                    "kind": self._mem_kind(m),
                    "source_key": self._source_key(m),
                    "numeric_signature": list(self._numeric_signature(text)),
                    "topic_set": sorted(self._topic_set(text)),
                    "confidence_values": [],
                    "verification_statuses": [],
                    "statuses": [],
                    "conflict_flags": [],
                }
                buckets[ckey] = bucket

            bucket["row_count"] += 1
            bucket["memory_ids"].append(mid)
            bucket["records"].append(m)
            bucket["confidence_values"].append(self._mem_confidence(m))
            bucket["verification_statuses"].append(self._mem_verification_status(m))
            bucket["statuses"].append(self._mem_status(m))
            bucket["conflict_flags"].append(self._mem_conflict_flag(m))

            if layer and layer not in bucket["layers"]:
                bucket["layers"].append(layer)

            existing_texts = {self._norm_text_low(x) for x in bucket["texts"]}
            if text and self._norm_text_low(text) not in existing_texts:
                bucket["texts"].append(text)

            cur = self._safe_str(bucket.get("canonical_text"))
            if not cur or (len(text.strip()) < len(cur.strip()) and text.strip()):
                bucket["canonical_text"] = text

        out: List[Dict[str, Any]] = []
        for bucket in buckets.values():
            mids = sorted(
                [str(x).strip().lower() for x in bucket["memory_ids"] if str(x).strip()]
            )
            layers = sorted(
                [str(x).strip().lower() for x in bucket["layers"] if str(x).strip()]
            )

            normalized_layers = [x for x in layers if x in {"wm", "ltm", "clair", "unknown"}]
            bucket["memory_ids"] = mids
            bucket["layers"] = normalized_layers
            bucket["mirror_only"] = (
                bucket["row_count"] > 1
                and "wm" in normalized_layers
                and "ltm" in normalized_layers
            )
            bucket["avg_confidence"] = (
                sum(float(x) for x in bucket["confidence_values"]) / max(1, len(bucket["confidence_values"]))
            )
            bucket["has_conflict_flag"] = any(bool(x) for x in bucket["conflict_flags"])
            bucket["epistemic_state"] = self._collapse_epistemic_state(
                bucket["verification_statuses"],
                bucket["statuses"],
                bucket["has_conflict_flag"],
            )
            out.append(bucket)

        out.sort(key=lambda x: (x["normalized_text"], x["claim_key"]))
        return out

    def _collapse_epistemic_state(
        self,
        verification_statuses: List[str],
        statuses: List[str],
        has_conflict_flag: bool,
    ) -> str:
        vals = {
            self._safe_str(x).lower()
            for x in list(verification_statuses or []) + list(statuses or [])
            if self._safe_str(x)
        }

        if has_conflict_flag:
            return "contested"
        if "contradicted" in vals or "rejected" in vals:
            return "contradicted"
        if "contested" in vals or "disputed" in vals:
            return "contested"
        if "supported" in vals or "verified" in vals or "confirm" in vals:
            return "supported"
        if "document_scoped" in vals:
            return "document_scoped"
        return "unverified"

    def _canonical_index_by_memory_id(
        self,
        canonical_items: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for item in canonical_items:
            for mid in item.get("memory_ids", []):
                out[str(mid).strip().lower()] = item
        return out

    # ------------------------------------------------------------------
    # Text helpers
    # ------------------------------------------------------------------
    def _tokens(self, s: str) -> List[str]:
        low = (s or "").lower()
        return [t for t in self._tok_pat.findall(low) if t and t not in self._stop]

    def _topic_set(self, s: str) -> Set[str]:
        return set(self._tokens(s)).intersection(self._topics)

    def _semantic_overlap_count(self, a: str, b: str) -> int:
        ta = set(self._tokens(a))
        tb = set(self._tokens(b))
        return len(ta.intersection(tb))

    def _semantic_overlap(self, a: str, b: str) -> bool:
        return self._semantic_overlap_count(a, b) >= int(self.semantic_overlap_min_tokens)

    def _looks_procedural_numeric(self, text: str) -> bool:
        low = (text or "").lower()
        if not self._num_pat.search(low):
            return False
        if any(marker in low for marker in self._procedural_markers):
            return True
        return any(x in low for x in ("priority", "timebox", "minute", "rule of"))

    def _find_memory_by_id(self, memories: List[Dict[str, Any]], mem_id: str) -> Optional[Dict[str, Any]]:
        target = str(mem_id or "").strip().lower()
        if not target:
            return None
        for m in memories:
            if self._mem_id(m) == target:
                return m
        return None

    def _find_memory_by_text(self, memories: List[Dict[str, Any]], text: str) -> Optional[Dict[str, Any]]:
        target = self._norm_text_low(text)
        if not target:
            return None

        for m in memories:
            mt = self._norm_text_low(self._mem_text(m))
            if mt and mt == target:
                return m

        target_fp = self._fp_text(target)
        for m in memories:
            mt = self._mem_text(m)
            if mt and self._fp_text(mt) == target_fp:
                return m

        best: Optional[Dict[str, Any]] = None
        best_score = 0.0
        for m in memories:
            mt = self._norm_text_low(self._mem_text(m))
            if not mt:
                continue
            if target in mt or mt in target:
                score = min(len(target), len(mt)) / max(1, max(len(target), len(mt)))
                if score > best_score:
                    best = m
                    best_score = score

        return best if best_score >= 0.70 else None

    def _find_canonical_by_claim_key(
        self,
        canonical_items: List[Dict[str, Any]],
        claim_key: str,
    ) -> Optional[Dict[str, Any]]:
        target = self._safe_str(claim_key)
        if not target:
            return None
        for item in canonical_items:
            if self._safe_str(item.get("claim_key")) == target:
                return item
        return None

    # ------------------------------------------------------------------
    # Detectors on canonical claims
    # ------------------------------------------------------------------
    def detect_numeric_conflicts(
        self,
        memories: Optional[List[Dict[str, Any]]] = None,
        canonical_items: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Tuple[str, str]]:
        memories = memories if isinstance(memories, list) else self.get_memories(limit=500)
        canonical_items = (
            canonical_items
            if isinstance(canonical_items, list)
            else self.build_canonical_claims(memories)
        )

        items: List[Tuple[str, str, Tuple[str, ...], Set[str]]] = []
        for item in canonical_items:
            txt = self._safe_str(item.get("canonical_text"))
            if not txt:
                continue
            if not self._num_pat.search(txt):
                continue
            if self._looks_procedural_numeric(txt):
                continue

            nums = tuple(self._numeric_signature(txt))
            if not nums:
                continue

            topics = self._topic_set(txt)
            items.append((self._safe_str(item.get("claim_key")), txt, nums, topics))

        conflicts: List[Tuple[str, str]] = []
        seen = set()
        pairs = 0

        for i in range(len(items)):
            id1, t1, n1, topic1 = items[i]
            for j in range(i + 1, len(items)):
                pairs += 1
                if pairs > self.max_pairs:
                    return conflicts

                id2, t2, n2, topic2 = items[j]
                if n1 == n2:
                    continue

                shared_topics = topic1.intersection(topic2)
                if shared_topics:
                    key = self._pair_key(id1, id2)
                    if key not in seen:
                        seen.add(key)
                        conflicts.append((id1, id2))
                    continue

                if not topic1 and not topic2 and self._semantic_overlap(t1, t2):
                    key = self._pair_key(id1, id2)
                    if key not in seen:
                        seen.add(key)
                        conflicts.append((id1, id2))

        return conflicts

    def detect_negation_conflicts(
        self,
        memories: Optional[List[Dict[str, Any]]] = None,
        canonical_items: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Tuple[str, str]]:
        memories = memories if isinstance(memories, list) else self.get_memories(limit=500)
        canonical_items = (
            canonical_items
            if isinstance(canonical_items, list)
            else self.build_canonical_claims(memories)
        )

        texts: List[Tuple[str, str, bool, Set[str]]] = []
        for item in canonical_items:
            txt = self._safe_str(item.get("canonical_text"))
            if not txt:
                continue
            neg = bool(self._neg_pat.search(txt))
            topics = self._topic_set(txt)
            texts.append((self._safe_str(item.get("claim_key")), txt, neg, topics))

        conflicts: List[Tuple[str, str]] = []
        seen = set()
        pairs = 0

        for i in range(len(texts)):
            id1, t1, neg1, topic1 = texts[i]
            for j in range(i + 1, len(texts)):
                pairs += 1
                if pairs > self.max_pairs:
                    return conflicts

                id2, t2, neg2, topic2 = texts[j]
                if neg1 == neg2:
                    continue
                if t1.strip().lower() == t2.strip().lower():
                    continue
                if (topic1 or topic2) and not topic1.intersection(topic2):
                    continue
                if self._semantic_overlap(t1, t2):
                    key = self._pair_key(id1, id2)
                    if key not in seen:
                        seen.add(key)
                        conflicts.append((id1, id2))

        return conflicts

    def detect_flagged_conflicts(
        self,
        memories: Optional[List[Dict[str, Any]]] = None,
        canonical_items: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Tuple[str, str]]:
        memories = memories if isinstance(memories, list) else self.get_memories(limit=500)
        canonical_items = (
            canonical_items
            if isinstance(canonical_items, list)
            else self.build_canonical_claims(memories)
        )

        canonical_by_mem_id = self._canonical_index_by_memory_id(canonical_items)

        conflicts: List[Tuple[str, str]] = []
        seen: Set[str] = set()

        contested_rows: List[Dict[str, Any]] = []
        for m in memories:
            if self._mem_conflict_flag(m):
                contested_rows.append(m)

        for m in contested_rows:
            my_item = canonical_by_mem_id.get(self._mem_id(m))
            if my_item is None:
                continue
            my_key = self._safe_str(my_item.get("claim_key"))

            linked_ids = self._mem_conflict_with_ids(m)
            for other_id in linked_ids:
                other_item = canonical_by_mem_id.get(other_id)
                if other_item is None:
                    continue
                other_key = self._safe_str(other_item.get("claim_key"))
                if not other_key or other_key == my_key:
                    continue
                pair_key = self._pair_key(my_key, other_key)
                if pair_key not in seen:
                    seen.add(pair_key)
                    conflicts.append((my_key, other_key))

        for m in contested_rows:
            my_item = canonical_by_mem_id.get(self._mem_id(m))
            if my_item is None:
                continue
            my_key = self._safe_str(my_item.get("claim_key"))

            linked_texts = self._mem_conflict_with_texts(m)
            for other_text in linked_texts:
                other_raw = self._find_memory_by_text(memories, other_text)
                if other_raw is None:
                    continue
                other_item = canonical_by_mem_id.get(self._mem_id(other_raw))
                if other_item is None:
                    continue
                other_key = self._safe_str(other_item.get("claim_key"))
                if not other_key or other_key == my_key:
                    continue
                pair_key = self._pair_key(my_key, other_key)
                if pair_key not in seen:
                    seen.add(pair_key)
                    conflicts.append((my_key, other_key))

        contested_claims: List[Dict[str, Any]] = [
            item for item in canonical_items if bool(item.get("has_conflict_flag"))
        ]

        pairs = 0
        for i in range(len(contested_claims)):
            a = contested_claims[i]
            ida = self._safe_str(a.get("claim_key"))
            ta = self._safe_str(a.get("canonical_text"))
            if not ida or not ta:
                continue

            for j in range(i + 1, len(contested_claims)):
                pairs += 1
                if pairs > self.max_pairs:
                    return conflicts

                b = contested_claims[j]
                idb = self._safe_str(b.get("claim_key"))
                tb = self._safe_str(b.get("canonical_text"))
                if not idb or not tb or ida == idb:
                    continue

                key = self._pair_key(ida, idb)
                if key in seen:
                    continue
                if self._norm_text_low(ta) == self._norm_text_low(tb):
                    continue

                overlap = self._semantic_overlap_count(ta, tb)
                nums_a = tuple(self._num_pat.findall(ta))
                nums_b = tuple(self._num_pat.findall(tb))
                neg_a = bool(self._neg_pat.search(ta))
                neg_b = bool(self._neg_pat.search(tb))
                shared_topics = self._topic_set(ta).intersection(self._topic_set(tb))

                looks_conflicting = False
                if nums_a and nums_b and nums_a != nums_b and shared_topics:
                    looks_conflicting = True
                elif neg_a != neg_b and (shared_topics or overlap >= int(self.semantic_overlap_min_tokens)):
                    looks_conflicting = True
                elif shared_topics and overlap >= max(1, int(self.semantic_overlap_min_tokens) - 1):
                    looks_conflicting = True

                if looks_conflicting:
                    seen.add(key)
                    conflicts.append((ida, idb))

        return conflicts

    def detect_duplicates(
        self,
        memories: Optional[List[Dict[str, Any]]] = None,
        canonical_items: Optional[List[Dict[str, Any]]] = None,
    ) -> List[List[str]]:
        memories = memories if isinstance(memories, list) else self.get_memories(limit=500)
        canonical_items = (
            canonical_items
            if isinstance(canonical_items, list)
            else self.build_canonical_claims(memories)
        )

        duplicate_groups: List[List[str]] = []
        for item in canonical_items:
            mids = list(item.get("memory_ids", []))
            if len(mids) < 2:
                continue

            layers = set(item.get("layers", []))
            mirror_only = bool(item.get("mirror_only", False))

            if mirror_only and "wm" in layers and "ltm" in layers:
                continue

            duplicate_groups.append(mids)

        return duplicate_groups

    def detect_drift_signals(self, memories: Optional[List[Dict[str, Any]]] = None) -> Dict[str, int]:
        memories = memories if isinstance(memories, list) else self.get_memories(limit=500)
        now = time.time()
        stale_sec = 60 * 60 * 24 * int(self.stale_days)

        signals = {"low_confidence": 0, "contested": 0, "stale": 0, "low_evidence": 0}
        for m in memories:
            if self._mem_confidence(m) < 0.4:
                signals["low_confidence"] += 1
            if self._mem_conflict_flag(m):
                signals["contested"] += 1
            if (now - self._mem_last_used(m)) > stale_sec:
                signals["stale"] += 1
            if self._mem_evidence_count(m) <= 0:
                signals["low_evidence"] += 1

        return signals

    # ------------------------------------------------------------------
    # Packet builders
    # ------------------------------------------------------------------
    def _qid(self, prefix: str, *parts: str) -> str:
        blob = "|".join([prefix] + [str(p or "") for p in parts])
        h = hashlib.sha1(blob.encode("utf-8", errors="ignore")).hexdigest()[:12]
        return f"{prefix}_{h}"

    def _build_conflict_item(
        self,
        canonical_items: List[Dict[str, Any]],
        a_key: str,
        b_key: str,
        *,
        kind: str = "conflict",
    ) -> Optional[AuditItem]:
        a = self._find_canonical_by_claim_key(canonical_items, a_key)
        b = self._find_canonical_by_claim_key(canonical_items, b_key)
        if a is None or b is None:
            return None

        ta = self._safe_str(a.get("canonical_text"))
        tb = self._safe_str(b.get("canonical_text"))
        if not ta or not tb:
            return None

        x, y = sorted([self._safe_str(a.get("claim_key")), self._safe_str(b.get("claim_key"))])
        qid = self._qid("conflict", x, y)

        prompt = (
            "Calibration conflict check:\n"
            f"A) {ta}\n"
            f"B) {tb}\n"
            "Which is more likely correct? Options: confirm A / confirm B / unsure / modify."
        )

        payload = {
            "claim_a_key": x,
            "claim_b_key": y,
            "pair": [x, y],
            "pair_ids": [x, y],
            "conflict_pair": [x, y],
            "conflict_pair_ids": [x, y],
            "text_a": ta,
            "text_b": tb,
            "claim_a": ta,
            "claim_b": tb,
            "side_a": ta,
            "side_b": tb,
            "memory_a_ids": list(a.get("memory_ids", [])),
            "memory_b_ids": list(b.get("memory_ids", [])),
            "claim_a_record": {
                "claim_key": x,
                "claim": ta,
                "content": ta,
                "type": a.get("type"),
                "domain": a.get("domain"),
                "layers": list(a.get("layers", [])),
                "epistemic_state": a.get("epistemic_state"),
            },
            "claim_b_record": {
                "claim_key": y,
                "claim": tb,
                "content": tb,
                "type": b.get("type"),
                "domain": b.get("domain"),
                "layers": list(b.get("layers", [])),
                "epistemic_state": b.get("epistemic_state"),
            },
        }

        memory_ids = sorted(set(list(a.get("memory_ids", [])) + list(b.get("memory_ids", []))))

        return AuditItem(
            qid=qid,
            kind=kind,
            memory_ids=memory_ids,
            payload={
                "memory_id": (memory_ids[0] if memory_ids else None),
                "memory_ids": memory_ids,
                "pair_ids": [x, y],
                "conflict_pair_ids": [x, y],
                "conflict": True,
                "conflict_a": ta,
                "conflict_b": tb,
                "prompt": prompt,
                "question": prompt,
                "payload": payload,
            },
        )

    def _build_duplicate_item(
        self,
        canonical_items: List[Dict[str, Any]],
        ids: List[str],
    ) -> Optional[AuditItem]:
        if not ids:
            return None

        target_set = {str(x).strip().lower() for x in ids if str(x).strip()}
        rows: List[Dict[str, Any]] = []

        for item in canonical_items:
            mids = {str(x).strip().lower() for x in item.get("memory_ids", [])}
            if mids == target_set:
                rows.append(item)

        if not rows:
            return None

        row = rows[0]
        mids = list(row.get("memory_ids", []))
        texts = list(row.get("texts", [])) or [self._safe_str(row.get("canonical_text"))]
        claim = self._safe_str(row.get("canonical_text"))
        if not claim:
            return None

        qid = self._qid("duplicate", *sorted(mids))
        prompt = (
            "Duplicate memory check:\n"
            f'Claim: "{claim}"\n'
            "These appear duplicated. Keep strongest, merge, or ignore?"
        )

        return AuditItem(
            qid=qid,
            kind="duplicate",
            memory_ids=mids,
            payload={
                "memory_id": mids[0] if mids else None,
                "memory_ids": mids,
                "claim": claim,
                "content": claim,
                "texts": texts,
                "layers": list(row.get("layers", [])),
                "row_count": int(row.get("row_count", 0)),
                "prompt": prompt,
                "question": prompt,
                "duplicate": True,
            },
        )

    def _build_drift_item(self, signals: Dict[str, int]) -> Optional[AuditItem]:
        if not signals or sum(int(v) for v in signals.values()) <= 0:
            return None

        qid = self._qid(
            "drift",
            str(signals.get("low_confidence", 0)),
            str(signals.get("contested", 0)),
            str(signals.get("stale", 0)),
            str(signals.get("low_evidence", 0)),
        )
        prompt = (
            "Memory drift check:\n"
            f"low_confidence={signals.get('low_confidence', 0)}, "
            f"contested={signals.get('contested', 0)}, "
            f"stale={signals.get('stale', 0)}, "
            f"low_evidence={signals.get('low_evidence', 0)}.\n"
            "Should calibration prioritize a maintenance pass?"
        )

        return AuditItem(
            qid=qid,
            kind="drift",
            memory_ids=[],
            payload={
                "prompt": prompt,
                "question": prompt,
                "signals": dict(signals),
            },
        )

    # ------------------------------------------------------------------
    # Queue helpers
    # ------------------------------------------------------------------
    def _prune_asked(self) -> None:
        now = time.time()
        stale = [k for k, ts in self._asked.items() if (now - ts) > self._asked_ttl_sec]
        for k in stale:
            self._asked.pop(k, None)

    def _queue_key(self, item: AuditItem) -> str:
        mids = sorted([str(x).strip().lower() for x in (item.memory_ids or []) if str(x).strip()])
        return f"{item.kind}|{'|'.join(mids)}|{item.qid}"

    def refresh_queue(self, force: bool = False) -> List[AuditItem]:
        now = time.time()
        if not force and (now - self._last_refresh) < self.refresh_interval_sec:
            return self._queue

        self._prune_asked()
        memories = self.get_memories(limit=500)
        canonical_items = self.build_canonical_claims(memories)

        conflict_pairs: List[Tuple[str, str]] = []
        seen_pairs: Set[str] = set()

        for detector in (
            self.detect_flagged_conflicts,
            self.detect_numeric_conflicts,
            self.detect_negation_conflicts,
        ):
            for pair in detector(memories, canonical_items):
                key = self._pair_key(pair[0], pair[1])
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    conflict_pairs.append(pair)

        new_queue: List[AuditItem] = []

        for a_id, b_id in conflict_pairs[: max(1, int(self.max_conflict_items_per_refresh))]:
            item = self._build_conflict_item(canonical_items, a_id, b_id, kind="conflict")
            if item is not None and self._queue_key(item) not in self._asked:
                new_queue.append(item)

        duplicate_groups = self.detect_duplicates(memories, canonical_items)
        for ids in duplicate_groups[: max(0, int(self.max_duplicate_items_per_refresh))]:
            item = self._build_duplicate_item(canonical_items, ids)
            if item is not None and self._queue_key(item) not in self._asked:
                new_queue.append(item)

        if self.include_drift_item:
            drift_item = self._build_drift_item(self.detect_drift_signals(memories))
            if drift_item is not None and self._queue_key(drift_item) not in self._asked:
                new_queue.append(drift_item)

        self._queue = new_queue
        self._cursor = 0
        self._last_refresh = now
        return self._queue

    def _audit_item_to_dict(self, item: AuditItem) -> Dict[str, Any]:
        payload = dict(item.payload or {})
        out = {
            "qid": item.qid,
            "kind": item.kind,
            "memory_ids": list(item.memory_ids or []),
        }
        out.update(payload)
        if "memory_id" not in out:
            out["memory_id"] = (out["memory_ids"][0] if out["memory_ids"] else None)
        return out

    def next_question(self) -> Optional[Dict[str, Any]]:
        self.refresh_queue(force=False)
        if self._cursor >= len(self._queue):
            return None

        item = self._queue[self._cursor]
        self._cursor += 1
        self._asked[self._queue_key(item)] = time.time()
        return self._audit_item_to_dict(item)

    def pick_question(self) -> Optional[Dict[str, Any]]:
        return self.next_question()

    def select_question(self) -> Optional[Dict[str, Any]]:
        return self.next_question()

    def propose_question(self) -> Optional[Dict[str, Any]]:
        return self.next_question()

    def audit_next(self) -> Optional[Dict[str, Any]]:
        return self.next_question()

    def next_item(self) -> Optional[Dict[str, Any]]:
        return self.next_question()

    # ------------------------------------------------------------------
    # Feedback handling
    # ------------------------------------------------------------------
    def apply_feedback(self, candidate: Any, feedback: Dict[str, Any]) -> Dict[str, Any]:
        self.feedback_log.append({
            "t": time.time(),
            "candidate": candidate,
            "feedback": dict(feedback or {}),
        })

        verdict = str((feedback or {}).get("verdict") or "unsure").strip().lower()
        if verdict not in {"confirm", "deny", "modify", "unsure", "merge"}:
            verdict = "unsure"

        return {
            "ok": True,
            "applied": True,
            "verdict": verdict,
            "candidate_kind": str((candidate or {}).get("kind") if isinstance(candidate, dict) else ""),
        }

    def submit_feedback(self, candidate: Any, feedback: Dict[str, Any]) -> Dict[str, Any]:
        return self.apply_feedback(candidate, feedback)

    def handle_feedback(self, candidate: Any, feedback: Dict[str, Any]) -> Dict[str, Any]:
        return self.apply_feedback(candidate, feedback)

    def on_feedback(self, candidate: Any, feedback: Dict[str, Any]) -> Dict[str, Any]:
        return self.apply_feedback(candidate, feedback)

    # ------------------------------------------------------------------
    # Maintenance / diagnostics
    # ------------------------------------------------------------------
    def full_audit(self) -> Dict[str, Any]:
        memories = self.get_memories(limit=500)
        canonical_items = self.build_canonical_claims(memories)

        flagged = self.detect_flagged_conflicts(memories, canonical_items)
        numeric = self.detect_numeric_conflicts(memories, canonical_items)
        negation = self.detect_negation_conflicts(memories, canonical_items)
        duplicates = self.detect_duplicates(memories, canonical_items)
        drift = self.detect_drift_signals(memories)

        mirror_groups = [
            {
                "claim_key": item.get("claim_key"),
                "claim": item.get("canonical_text"),
                "memory_ids": list(item.get("memory_ids", [])),
                "layers": list(item.get("layers", [])),
                "row_count": int(item.get("row_count", 0)),
            }
            for item in canonical_items
            if bool(item.get("mirror_only"))
        ]

        return {
            "version": self.VERSION,
            "raw_memory_count": len(memories),
            "memory_count": len(canonical_items),
            "canonical_claim_count": len(canonical_items),
            "mirror_row_groups": mirror_groups,
            "flagged_conflicts": [
                {"a_id": a, "b_id": b, "pair_key": self._pair_key(a, b)}
                for a, b in flagged
            ],
            "numeric_conflicts": [
                {"a_id": a, "b_id": b, "pair_key": self._pair_key(a, b)}
                for a, b in numeric
            ],
            "negation_conflicts": [
                {"a_id": a, "b_id": b, "pair_key": self._pair_key(a, b)}
                for a, b in negation
            ],
            "duplicates": [list(ids) for ids in duplicates],
            "drift_signals": dict(drift),
            "queue_len": len(self._queue),
            "feedback_log_len": len(self.feedback_log),
        }

    def maintenance(self) -> Dict[str, Any]:
        self.refresh_queue(force=True)
        return {"ok": True, "queue_len": len(self._queue)}

    def run_maintenance(self) -> Dict[str, Any]:
        return self.maintenance()

    def recalibrate(self) -> Dict[str, Any]:
        return self.maintenance()

    def repair_pass(self) -> Dict[str, Any]:
        return self.maintenance()

    def consolidate(self) -> Dict[str, Any]:
        return self.maintenance()

    def sleep_tick(self) -> Dict[str, Any]:
        return self.maintenance()

    def debug_snapshot(self) -> Dict[str, Any]:
        memories = self.get_memories(limit=500)
        canonical_items = self.build_canonical_claims(memories)
        self.refresh_queue(force=False)

        return {
            "version": self.VERSION,
            "raw_memory_count": len(memories),
            "canonical_claim_count": len(canonical_items),
            "mirror_group_count": sum(1 for x in canonical_items if bool(x.get("mirror_only"))),
            "queue_len": len(self._queue),
            "cursor": self._cursor,
            "queue_preview": [self._audit_item_to_dict(x) for x in self._queue[:10]],
            "asked_count": len(self._asked),
            "feedback_log_len": len(self.feedback_log),
            "last_refresh": self._last_refresh,
        }