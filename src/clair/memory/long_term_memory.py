# FILE: memory/long_term_memory.py
# Clair Long-Term Memory (v3.2-LTM)
#
# Contract-backed rewrite:
# - canonical MemoryRecord bridge
# - SQLite durability
# - duplicate reinforcement
# - semantic revision merge
# - contested conflict storage
# - backwards-compatible dict APIs for current Clair modules
# - stronger short-run benchmark behavior
# - safer row normalization / update paths
# - better retrieval ordering and promotion visibility
# - conflict-pair preservation for calibration / contradiction harnesses
# - exact memory_id aware updates
# - conflict-linked rows protected from duplicate collapse

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from dataclasses import replace
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import config
from memory.contracts import (
    EvidencePacket,
    MemoryKind,
    MemoryRecord,
    MemorySignals,
    MemoryTier,
    SourceType,
    VerificationStatus,
    make_memory_record,
)


class LongTermMemory:
    VERSION = "3.2-LTM"

    NUMERIC_GUARDRAILS = {
        "water boiling": 100,
        "human bones": 206,
        "everest height": 8848,
    }

    STOP_TOKENS = {
        "a", "an", "the", "and", "or", "to", "of", "in", "on", "at", "for", "from", "by",
        "is", "are", "was", "were", "be", "been", "being",
        "if", "then", "after", "before", "during", "when", "while",
        "do", "dont", "don't", "not", "no", "yes",
        "with", "as", "it", "this", "that", "these", "those",
        "you", "your", "me", "my", "we", "our", "they", "their",
        "should", "would", "could", "can", "may", "might",
        "help", "stay", "move", "moving", "get", "make", "avoid", "safe", "possible",
    }

    VALID_TYPES = {
        "lesson", "observe", "fact", "policy", "feedback",
        "identity", "reasoning_action", "committed_action",
        "claim", "chapter_summary", "section_summary", "summary",
    }

    VALID_STATUS = {
        "unverified",
        "verified",
        "contested",
        "deprecated",
        "revised",
        "provisional",
        "pending",
        "hypothesis",
    }

    BAD_MEMORY_IDS = {"", "none", "null", "nan"}

    NEAR_DUPLICATE_SIM = float(getattr(config, "LTM_NEAR_DUPLICATE_SIM", 0.92))
    CORE_JACCARD_DUP = float(getattr(config, "LTM_CORE_JACCARD_DUP", 0.88))
    DOMAIN_MISMATCH_DUP_SIM = float(getattr(config, "LTM_DOMAIN_MISMATCH_DUP_SIM", 0.97))
    KIND_MISMATCH_DUP_SIM = float(getattr(config, "LTM_KIND_MISMATCH_DUP_SIM", 0.97))
    REVISION_MERGE_SIM = float(getattr(config, "LTM_REVISION_MERGE_SIM", 0.86))
    TOPIC_OVERLAP_MIN = int(getattr(config, "LTM_TOPIC_OVERLAP_MIN", 3))
    NUMERIC_REL_DIFF = float(getattr(config, "LTM_NUMERIC_REL_DIFF", 0.15))

    REINFORCE_CONFIDENCE_BUMP = float(getattr(config, "LTM_REINFORCE_CONFIDENCE_BUMP", 0.01))
    VERIFIED_REINFORCE_CONFIDENCE_BUMP = float(
        getattr(config, "LTM_VERIFIED_REINFORCE_CONFIDENCE_BUMP", 0.03)
    )
    REINFORCE_CONFIDENCE_CAP = float(getattr(config, "LTM_REINFORCE_CONFIDENCE_CAP", 1.0))
    REVISION_REPLACE_MARGIN = float(getattr(config, "LTM_REVISION_REPLACE_MARGIN", 0.10))

    SEARCH_SIM_WEIGHT = float(getattr(config, "LTM_SEARCH_SIM_WEIGHT", 0.50))
    SEARCH_OVERLAP_WEIGHT = float(getattr(config, "LTM_SEARCH_OVERLAP_WEIGHT", 0.30))
    SEARCH_CONF_WEIGHT = float(getattr(config, "LTM_SEARCH_CONF_WEIGHT", 0.10))
    SEARCH_DOMAIN_WEIGHT = float(getattr(config, "LTM_SEARCH_DOMAIN_WEIGHT", 0.10))

    ENABLE_POV_SAFE_IDENTITY = bool(getattr(config, "LTM_POV_SAFE_IDENTITY", True))

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or getattr(config, "LONG_TERM_DB_PATH", None)
        if not self.db_path:
            raise ValueError("config.LONG_TERM_DB_PATH is not set.")

        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA foreign_keys=OFF;")
        self._create_or_migrate_schema()

    # -------------------------------------------------------------------------
    # Time / debugging
    # -------------------------------------------------------------------------
    def _now(self) -> float:
        return time.time()

    def _utcnow(self) -> datetime:
        return datetime.now(timezone.utc)

    def _debug_reject(self, reason: str, msg_type: str, content: str) -> None:
        if getattr(config, "VERBOSE", False):
            print(f"[LongTermMemory] Reject ({reason}): type={msg_type} content={content[:90]!r}")

    def _debug_info(self, msg: str) -> None:
        if getattr(config, "VERBOSE", False):
            print(f"[LongTermMemory] {msg}")

    # -------------------------------------------------------------------------
    # Schema
    # -------------------------------------------------------------------------
    def _create_or_migrate_schema(self) -> None:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT,
                type TEXT NOT NULL,
                content TEXT NOT NULL,
                content_obj TEXT,
                context TEXT,
                confidence REAL DEFAULT 0.0,
                timestamp REAL DEFAULT (strftime('%s','now')),
                source TEXT DEFAULT 'system',
                revision_trace TEXT,
                numeric_guarded INTEGER DEFAULT 0,
                domain TEXT,
                tags TEXT,
                kind TEXT,
                details TEXT,

                status TEXT DEFAULT 'unverified',
                sources TEXT,
                evidence TEXT,
                history TEXT,
                usage_count INTEGER DEFAULT 0,
                last_used REAL,
                updated_at REAL,

                stability REAL DEFAULT 0.0,
                decay_score REAL DEFAULT 0.0,
                priority REAL DEFAULT 0.5,
                contradiction_count INTEGER DEFAULT 0,
                retrieval_hits INTEGER DEFAULT 0,
                retrieval_misses INTEGER DEFAULT 0,
                related_ids TEXT,
                source_ref TEXT,
                signals TEXT,
                metadata TEXT
            )
            """
        )
        self.conn.commit()

        cursor.execute("PRAGMA table_info(memory)")
        existing_cols = {row[1] for row in cursor.fetchall()}

        def add_col(name: str, ddl: str) -> None:
            if name not in existing_cols:
                cursor.execute(f"ALTER TABLE memory ADD COLUMN {ddl}")

        add_col("memory_id", "memory_id TEXT")
        add_col("content_obj", "content_obj TEXT")
        add_col("numeric_guarded", "numeric_guarded INTEGER DEFAULT 0")
        add_col("domain", "domain TEXT")
        add_col("tags", "tags TEXT")
        add_col("kind", "kind TEXT")
        add_col("details", "details TEXT")
        add_col("status", "status TEXT DEFAULT 'unverified'")
        add_col("sources", "sources TEXT")
        add_col("evidence", "evidence TEXT")
        add_col("history", "history TEXT")
        add_col("usage_count", "usage_count INTEGER DEFAULT 0")
        add_col("last_used", "last_used REAL")
        add_col("updated_at", "updated_at REAL")
        add_col("stability", "stability REAL DEFAULT 0.0")
        add_col("decay_score", "decay_score REAL DEFAULT 0.0")
        add_col("priority", "priority REAL DEFAULT 0.5")
        add_col("contradiction_count", "contradiction_count INTEGER DEFAULT 0")
        add_col("retrieval_hits", "retrieval_hits INTEGER DEFAULT 0")
        add_col("retrieval_misses", "retrieval_misses INTEGER DEFAULT 0")
        add_col("related_ids", "related_ids TEXT")
        add_col("source_ref", "source_ref TEXT")
        add_col("signals", "signals TEXT")
        add_col("metadata", "metadata TEXT")

        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_type_time ON memory(type, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_type_content ON memory(type, content)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_domain_kind ON memory(domain, kind)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_status ON memory(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_last_used ON memory(last_used)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_memory_id ON memory(memory_id)")
        except Exception:
            pass

        self.conn.commit()

    # -------------------------------------------------------------------------
    # Normalization helpers
    # -------------------------------------------------------------------------
    def _norm_str(self, x: Any) -> str:
        if x is None:
            return ""
        try:
            return str(x).strip()
        except Exception:
            return ""

    def _norm_lower(self, x: Any) -> str:
        s = self._norm_str(x)
        return s.lower() if s else ""

    def _norm_domain(self, d: Any) -> Optional[str]:
        d2 = self._norm_lower(d)
        return d2 or None

    def _norm_kind(self, k: Any) -> Optional[str]:
        k2 = self._norm_lower(k)
        return k2 or None

    def _normalize_content(self, content: str) -> str:
        return " ".join((content or "").split()).strip()

    def _normalize_memory_id(self, value: Any) -> str:
        s = self._norm_lower(value)
        return "" if s in self.BAD_MEMORY_IDS else s

    def _ensure_memory_id(self, value: Any = None) -> str:
        mid = self._normalize_memory_id(value)
        if mid:
            return mid
        return make_memory_record("ltm").memory_id.lower()

    def _norm_tags(self, tags: Any) -> List[str]:
        if tags is None:
            return []
        if isinstance(tags, str):
            tags = [tags]
        if not isinstance(tags, (list, set, tuple)):
            return []
        out: List[str] = []
        seen = set()
        for t in tags:
            s = self._norm_lower(t)
            if s and s not in seen:
                out.append(s)
                seen.add(s)
        out.sort()
        return out

    def _ensure_list(self, x: Any) -> List[Any]:
        if x is None:
            return []
        if isinstance(x, list):
            return list(x)
        if isinstance(x, (set, tuple)):
            return list(x)
        return [x]

    def _normalize_context(self, context: Any) -> List[Any]:
        if context is None:
            return []
        if isinstance(context, list):
            return list(context)
        return [context]

    def _normalize_revision_trace(self, rt: Any) -> List[str]:
        items = self._ensure_list(rt)
        out: List[str] = []
        for item in items:
            s = self._norm_str(item)
            if s:
                out.append(s)
        return out

    def _normalize_sources(self, sources: Any, source: Optional[str] = None) -> List[str]:
        items = self._ensure_list(sources)
        out: List[str] = []
        seen = set()
        if source:
            s0 = self._norm_lower(source)
            if s0:
                out.append(s0)
                seen.add(s0)
        for item in items:
            s = self._norm_lower(item)
            if s and s not in seen:
                out.append(s)
                seen.add(s)
        return out

    def _normalize_evidence_dict(self, evidence: Any) -> Dict[str, Any]:
        if evidence is None:
            return {}
        if isinstance(evidence, dict):
            return dict(evidence)
        if isinstance(evidence, list):
            text_items = []
            for item in evidence:
                s = self._norm_str(item)
                if s:
                    text_items.append(s)
            return {"items": text_items} if text_items else {}
        s = self._norm_str(evidence)
        return {"value": s} if s else {}

    def _normalize_history(self, history: Any) -> List[Dict[str, Any]]:
        items = self._ensure_list(history)
        out: List[Dict[str, Any]] = []
        for item in items:
            if isinstance(item, dict):
                out.append(dict(item))
            else:
                s = self._norm_str(item)
                if s:
                    out.append({"ts": self._now(), "event": s})
        return out

    def _normalize_details(self, details: Any) -> Dict[str, Any]:
        if isinstance(details, dict):
            return dict(details)
        s = self._norm_str(details)
        return {"value": s} if s else {}

    def _json_dumps(self, obj: Any, default: Optional[str] = None) -> Optional[str]:
        try:
            return json.dumps(obj, ensure_ascii=False, sort_keys=True)
        except Exception:
            return default

    def _json_loads(self, s: Any, default: Any) -> Any:
        if not s or not isinstance(s, str):
            return default
        try:
            return json.loads(s)
        except Exception:
            return default

    # -------------------------------------------------------------------------
    # Conflict-pair helpers
    # -------------------------------------------------------------------------
    def _canonicalize_text(self, text: Any) -> str:
        s = self._norm_str(text).lower()
        s = s.replace("’", "'").replace("–", "-").replace("—", "-")
        s = re.sub(r"[^a-z0-9\s\.\-]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _topic_signature_from_text(self, text: str) -> str:
        toks = list(self._tokenize_core(text or ""))
        if not toks:
            return ""
        return " ".join(sorted(toks)[:6])

    def _numeric_signature_from_text(self, text: str) -> Optional[str]:
        nums = re.findall(r"(?<!\w)(?:\d+(?:\.\d+)?)(?!\w)", text or "")
        if not nums:
            return None
        return "|".join(nums[:4])

    def _make_conflict_pair_id(self) -> str:
        return make_memory_record("ltm_conflict").memory_id.lower()

    def _ensure_pair_fields(self, details: Dict[str, Any], text: str) -> Dict[str, Any]:
        d = dict(details or {})
        d.setdefault("canonical_text", self._canonicalize_text(text))
        d.setdefault("topic_signature", self._topic_signature_from_text(text))
        d.setdefault("numeric_signature", self._numeric_signature_from_text(text))
        d.setdefault("conflict_pair_id", None)
        d.setdefault("conflict_role", None)
        d.setdefault("paired_memory_id", None)
        d.setdefault("pair_status", None)
        return d

    def _is_conflict_linked_details(self, details: Dict[str, Any]) -> bool:
        return bool(
            details.get("conflict_pair_id")
            or details.get("paired_memory_id")
            or details.get("pair_status") in {"pending", "complete", "orphaned"}
        )

    def _same_conflict_pair_details(self, a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        ap = self._norm_str(a.get("conflict_pair_id"))
        bp = self._norm_str(b.get("conflict_pair_id"))
        return bool(ap and bp and ap == bp)

    # -------------------------------------------------------------------------
    # Verification / source / kind mapping
    # -------------------------------------------------------------------------
    def _status_from_verification(self, status: VerificationStatus) -> str:
        mapping = {
            VerificationStatus.UNVERIFIED: "unverified",
            VerificationStatus.PROVISIONAL: "provisional",
            VerificationStatus.VERIFIED: "verified",
            VerificationStatus.DISPUTED: "contested",
            VerificationStatus.REJECTED: "deprecated",
        }
        return mapping.get(status, "unverified")

    def _verification_from_status(self, status: Any) -> VerificationStatus:
        s = self._norm_lower(status)
        mapping = {
            "unverified": VerificationStatus.UNVERIFIED,
            "verified": VerificationStatus.VERIFIED,
            "contested": VerificationStatus.DISPUTED,
            "disputed": VerificationStatus.DISPUTED,
            "deprecated": VerificationStatus.REJECTED,
            "rejected": VerificationStatus.REJECTED,
            "revised": VerificationStatus.VERIFIED,
            "provisional": VerificationStatus.PROVISIONAL,
            "pending": VerificationStatus.PROVISIONAL,
            "hypothesis": VerificationStatus.PROVISIONAL,
        }
        return mapping.get(s, VerificationStatus.UNVERIFIED)

    def _source_type_from_string(self, source: str) -> SourceType:
        s = self._norm_lower(source)
        mapping = {
            "user_input": SourceType.USER_INPUT,
            "user": SourceType.USER_INPUT,
            "reading": SourceType.DOCUMENT,
            "document": SourceType.DOCUMENT,
            "verification": SourceType.VERIFICATION,
            "verified": SourceType.VERIFICATION,
            "seed_verified": SourceType.IMPORTED_MEMORY,
            "system": SourceType.SYSTEM,
            "reflection": SourceType.REFLECTION,
            "simulation": SourceType.SIMULATION,
            "sensor": SourceType.SENSOR,
            "harness_conflict_injection": SourceType.SYSTEM,
        }
        return mapping.get(s, SourceType.UNKNOWN)

    def _kind_from_strings(self, kind: Any, msg_type: Any) -> MemoryKind:
        k = self._norm_lower(kind)
        t = self._norm_lower(msg_type)

        mapping = {
            "fact": MemoryKind.FACT,
            "summary": MemoryKind.SUMMARY,
            "procedure": MemoryKind.PROCEDURE,
            "episode": MemoryKind.EPISODE,
            "hazard": MemoryKind.HAZARD,
            "goal": MemoryKind.GOAL,
            "reflection": MemoryKind.REFLECTION,
            "unresolved": MemoryKind.UNRESOLVED,
            "rule": MemoryKind.RULE,
            "user_preference": MemoryKind.USER_PREFERENCE,
            "scenario": MemoryKind.SCENARIO,
            "revision": MemoryKind.FACT,
            "troubleshooting": MemoryKind.FACT,
            "policy": MemoryKind.RULE,
        }
        if k in mapping:
            return mapping[k]

        if t in {"chapter_summary", "section_summary", "summary"}:
            return MemoryKind.SUMMARY
        if t in {"committed_action"}:
            return MemoryKind.GOAL
        if t in {"identity", "fact", "claim", "lesson", "policy"}:
            return MemoryKind.FACT
        if t in {"feedback"}:
            return MemoryKind.REFLECTION
        if t in {"reasoning_action"}:
            return MemoryKind.PROCEDURE
        return MemoryKind.UNRESOLVED

    # -------------------------------------------------------------------------
    # POV-safe identity normalization
    # -------------------------------------------------------------------------
    def _pov_safe_identity(self, content: str) -> str:
        if not content:
            return content
        raw = content.strip()
        low = raw.lower().strip()

        if "i am your father" in low:
            idx = low.find("i am your father")
            tail = raw[idx + len("i am your father"):].strip().strip(" .,!?:;\"'")
            if tail:
                return f"{tail} is my father."
            return content

        if low.startswith("my name is "):
            name = raw.split("is", 1)[1].strip().strip(" .,!?:;\"'")
            if name:
                return f"The user's name is {name}."
            return content

        return content

    # -------------------------------------------------------------------------
    # Guardrails / similarity
    # -------------------------------------------------------------------------
    def _numeric_guardrail_ok(self, content: str) -> bool:
        if not content:
            return False
        content_lower = content.lower()
        numbers = list(map(int, re.findall(r"\d+", content_lower)))
        for key, canonical in self.NUMERIC_GUARDRAILS.items():
            if key in content_lower:
                if (not numbers) or (numbers[0] != canonical):
                    self._debug_info(
                        f"Blocked store: numeric mismatch for '{key}' (found {numbers}, expected {canonical})"
                    )
                    return False
                break
        return True

    def _tokenize_core(self, text: str) -> Set[str]:
        t = (text or "").lower()
        toks = re.findall(r"[a-z0-9]+", t)
        return {w for w in toks if w and w not in self.STOP_TOKENS}

    def _jaccard(self, a: Set[str], b: Set[str]) -> float:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return float(inter) / float(union) if union else 0.0

    def _semantic_similarity(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        ac = self._canonicalize_text(a)
        bc = self._canonicalize_text(b)
        ta = self._tokenize_core(ac)
        tb = self._tokenize_core(bc)
        jac = self._jaccard(ta, tb)
        seq = SequenceMatcher(None, ac, bc).ratio()
        return 0.65 * jac + 0.35 * seq

    def _extract_numbers(self, text: str) -> List[float]:
        if not text:
            return []
        nums = re.findall(r"(?<!\w)(?:\d+(?:\.\d+)?)(?!\w)", text)
        out: List[float] = []
        for n in nums:
            try:
                out.append(float(n))
            except Exception:
                continue
        return out

    def _topic_overlap(self, a: str, b: str) -> int:
        return len(self._tokenize_core(a) & self._tokenize_core(b))

    def _topic_overlap_ok(self, a: str, b: str) -> bool:
        return self._topic_overlap(a, b) >= self.TOPIC_OVERLAP_MIN

    def _numeric_conflict(self, new_text: str, old_text: str) -> bool:
        new_nums = self._extract_numbers(new_text)
        old_nums = self._extract_numbers(old_text)
        if not new_nums or not old_nums:
            return False
        if not self._topic_overlap_ok(new_text, old_text):
            return False
        n1 = new_nums[0]
        n2 = old_nums[0]
        denom = max(1.0, abs(n2))
        rel_diff = abs(n1 - n2) / denom
        return rel_diff >= self.NUMERIC_REL_DIFF

    def _negation_conflict(self, new_text: str, old_text: str) -> bool:
        if not self._topic_overlap_ok(new_text, old_text):
            return False
        neg_pat = re.compile(r"\b(no|not|never|cannot|can't|isn't|aren't|doesn't|without|false|incorrect)\b", re.I)
        new_neg = bool(neg_pat.search(new_text or ""))
        old_neg = bool(neg_pat.search(old_text or ""))
        return new_neg != old_neg

    def _near_duplicate(
        self,
        new_text: str,
        old_text: str,
        new_domain: Optional[str],
        old_domain: Optional[str],
        new_kind: Optional[str],
        old_kind: Optional[str],
        new_details: Optional[Dict[str, Any]] = None,
        old_details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        nd = dict(new_details or {})
        od = dict(old_details or {})

        if self._same_conflict_pair_details(nd, od):
            return False
        if self._is_conflict_linked_details(nd) or self._is_conflict_linked_details(od):
            return False

        sim = self._semantic_similarity(new_text, old_text)
        core_j = self._jaccard(self._tokenize_core(new_text), self._tokenize_core(old_text))
        near = (sim >= self.NEAR_DUPLICATE_SIM) or (core_j >= self.CORE_JACCARD_DUP)

        if old_domain and new_domain and old_domain != new_domain:
            near = sim >= self.DOMAIN_MISMATCH_DUP_SIM
        if old_kind and new_kind and old_kind != new_kind:
            near = sim >= self.KIND_MISMATCH_DUP_SIM

        if self._numeric_conflict(new_text, old_text):
            return False

        return near

    def _revision_candidate(
        self,
        new_text: str,
        old_text: str,
        new_domain: Optional[str],
        old_domain: Optional[str],
        new_kind: Optional[str],
        old_kind: Optional[str],
        new_details: Optional[Dict[str, Any]] = None,
        old_details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if not new_text or not old_text:
            return False

        nd = dict(new_details or {})
        od = dict(old_details or {})
        if self._same_conflict_pair_details(nd, od):
            return False
        if self._is_conflict_linked_details(nd) or self._is_conflict_linked_details(od):
            return False

        if self._numeric_conflict(new_text, old_text):
            return False
        if self._negation_conflict(new_text, old_text):
            return False
        if old_domain and new_domain and old_domain != new_domain:
            return False
        if old_kind and new_kind and old_kind != new_kind:
            return False
        sim = self._semantic_similarity(new_text, old_text)
        overlap = self._topic_overlap(new_text, old_text)
        return sim >= self.REVISION_MERGE_SIM and overlap >= self.TOPIC_OVERLAP_MIN

    def _mem_strength(
        self,
        confidence: float,
        context: list,
        timestamp: float,
        usage_count: int = 0,
        status: str = "unverified",
    ) -> float:
        conf = float(confidence or 0.0)
        has_ctx = 1.0 if (isinstance(context, list) and len(context) > 0) else 0.0

        ts = float(timestamp or 0.0)
        age = max(0.0, self._now() - ts) if ts > 0 else 999999.0
        recency = 1.0 / (1.0 + (age / 3600.0))

        usage_bonus = min(1.0, float(max(0, int(usage_count or 0))) / 10.0)
        status_bonus = 0.0
        if status == "verified":
            status_bonus = 0.10
        elif status == "contested":
            status_bonus = -0.20
        elif status == "deprecated":
            status_bonus = -0.15
        elif status == "provisional":
            status_bonus = 0.03

        return (0.60 * conf) + (0.15 * has_ctx) + (0.10 * recency) + (0.15 * usage_bonus) + status_bonus

    # -------------------------------------------------------------------------
    # Merge helpers
    # -------------------------------------------------------------------------
    def _merge_lists_unique(self, a: Any, b: Any, norm_lower: bool = False) -> List[Any]:
        out: List[Any] = []
        seen = set()

        for seq in (self._ensure_list(a), self._ensure_list(b)):
            for item in seq:
                key = (
                    self._norm_lower(item)
                    if norm_lower
                    else json.dumps(item, sort_keys=True, ensure_ascii=False, default=str)
                )
                if key not in seen:
                    out.append(item)
                    seen.add(key)
        return out

    def _merge_details(self, old_details: Dict[str, Any], new_details: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(old_details or {})
        for k, v in (new_details or {}).items():
            if k not in merged:
                merged[k] = v
                continue

            ov = merged.get(k)
            if isinstance(ov, list) or isinstance(v, list):
                merged[k] = self._merge_lists_unique(ov, v, norm_lower=False)
            elif isinstance(ov, dict) and isinstance(v, dict):
                tmp = dict(ov)
                for kk, vv in v.items():
                    if kk not in tmp:
                        tmp[kk] = vv
                    elif isinstance(tmp[kk], list) or isinstance(vv, list):
                        tmp[kk] = self._merge_lists_unique(tmp[kk], vv, norm_lower=False)
                    else:
                        tmp[kk] = vv if vv not in (None, "", [], {}) else tmp[kk]
                merged[k] = tmp
            else:
                merged[k] = v if v not in (None, "", [], {}) else ov
        return merged

    def _append_history_event(
        self,
        history: List[Dict[str, Any]],
        event: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        out = list(history or [])
        payload = {"ts": self._now(), "event": event}
        if isinstance(extra, dict):
            payload.update(extra)
        out.append(payload)
        return out

    # -------------------------------------------------------------------------
    # Contract bridge
    # -------------------------------------------------------------------------
    def _record_from_legacy_dict(self, msg: Dict[str, Any]) -> Optional[MemoryRecord]:
        if not isinstance(msg, dict):
            return None

        source = self._norm_lower(msg.get("source") or "unknown")
        text = self._normalize_content(self._norm_str(msg.get("content") or msg.get("claim")))
        if not text:
            return None

        if (
            self.ENABLE_POV_SAFE_IDENTITY
            and source in {"user", "user_input"}
            and self._norm_lower(msg.get("type")) in {"lesson", "identity"}
        ):
            text = self._pov_safe_identity(text)

        if not self._numeric_guardrail_ok(text):
            return None

        details = self._normalize_details(msg.get("details", {}))
        details = self._ensure_pair_fields(details, text)

        tags = self._norm_tags(msg.get("tags"))
        domain = self._norm_domain(msg.get("domain"))

        raw_evidence = msg.get("evidence")
        evidence_packets: List[EvidencePacket] = []
        if isinstance(raw_evidence, dict):
            for item in raw_evidence.get("items", []) or []:
                s = self._norm_str(item)
                if s:
                    evidence_packets.append(
                        EvidencePacket(
                            source_type=self._source_type_from_string(source),
                            snippet=s,
                            stance="support",
                            confidence=max(0.0, min(1.0, float(msg.get("confidence", 0.0) or 0.0))),
                        )
                    )
        elif isinstance(raw_evidence, list):
            for item in raw_evidence:
                s = self._norm_str(item)
                if s:
                    evidence_packets.append(
                        EvidencePacket(
                            source_type=self._source_type_from_string(source),
                            snippet=s,
                            stance="support",
                            confidence=max(0.0, min(1.0, float(msg.get("confidence", 0.0) or 0.0))),
                        )
                    )

        signals_data = msg.get("signals", {})
        if not isinstance(signals_data, dict):
            signals_data = {}

        signals = MemorySignals(
            domain=self._norm_domain(signals_data.get("domain") or domain),
            hazard_family=self._norm_domain(signals_data.get("hazard_family")),
            temporal_scope=self._norm_lower(signals_data.get("temporal_scope")) or None,
            novelty=float(signals_data.get("novelty", 0.0) or 0.0),
            urgency=float(signals_data.get("urgency", 0.0) or 0.0),
            usefulness=float(signals_data.get("usefulness", 0.0) or 0.0),
            metadata=signals_data.get("metadata", {}) if isinstance(signals_data.get("metadata"), dict) else {},
        )

        memory_id = self._ensure_memory_id(msg.get("memory_id") or msg.get("id"))

        record = MemoryRecord(
            memory_id=memory_id,
            text=text,
            summary=self._norm_str(msg.get("summary")) or None,
            kind=self._kind_from_strings(msg.get("kind"), msg.get("type")),
            tier=MemoryTier.LONG_TERM,
            source_type=self._source_type_from_string(source),
            source_ref=self._norm_str(msg.get("source_ref")) or None,
            confidence=max(0.0, min(1.0, float(msg.get("confidence", 0.0) or 0.0))),
            stability=max(0.0, min(1.0, float(msg.get("stability", 0.0) or 0.0))),
            decay_score=max(0.0, min(1.0, float(msg.get("decay_score", 0.0) or 0.0))),
            priority=max(0.0, min(1.0, float(msg.get("priority", 0.5) or 0.5))),
            verification_status=self._verification_from_status(
                msg.get("status") or msg.get("verification_status") or details.get("verification_status")
            ),
            access_count=max(0, int(msg.get("access_count", 0) or 0)),
            contradiction_count=max(0, int(msg.get("contradiction_count", 0) or 0)),
            retrieval_hits=max(0, int(msg.get("retrieval_hits", 0) or 0)),
            retrieval_misses=max(0, int(msg.get("retrieval_misses", 0) or 0)),
            tags=tags,
            related_ids=list(msg.get("related_ids", []) or []),
            evidence=evidence_packets,
            signals=signals,
            metadata={
                "ltm_id": msg.get("ltm_id"),
                "type": self._norm_lower(msg.get("type")) or "lesson",
                "domain": domain,
                "context": self._normalize_context(msg.get("context", [])),
                "timestamp": float(msg.get("timestamp", self._now()) or self._now()),
                "source": source or "unknown",
                "revision_trace": self._normalize_revision_trace(msg.get("revision_trace", [])),
                "numeric_guarded": bool(msg.get("numeric_guarded", False)),
                "details": details,
                "status": self._norm_lower(msg.get("status") or "unverified"),
                "sources": self._normalize_sources(msg.get("sources", None), source=source),
                "history": self._normalize_history(msg.get("history", None)),
                "usage_count": int(msg.get("usage_count", 0) or 0),
                "last_used": msg.get("last_used", None),
                "updated_at": float(msg.get("updated_at", self._now()) or self._now()),
                "content_obj": msg.get("content_obj"),
                "weight": float(msg.get("weight", 1.0) or 1.0),
                "summary": self._norm_str(msg.get("summary")) or None,
            },
        )

        details = record.metadata["details"]
        details.setdefault("source", source)
        details.setdefault("confidence", record.confidence)
        details.setdefault("status", self._status_from_verification(record.verification_status))
        details.setdefault("verification_status", self._status_from_verification(record.verification_status))
        details.setdefault("contested", record.verification_status == VerificationStatus.DISPUTED)
        details.setdefault("conflict", record.verification_status == VerificationStatus.DISPUTED)

        return record

    def _record_to_legacy_dict(self, record: MemoryRecord) -> Dict[str, Any]:
        details = dict(record.metadata.get("details", {}) or {})
        details = self._ensure_pair_fields(details, record.text)
        status = self._status_from_verification(record.verification_status)

        ltm_id = int(record.metadata["ltm_id"]) if record.metadata.get("ltm_id") is not None else None
        ts = float(record.metadata.get("timestamp", self._now()) or self._now())

        return {
            "id": ltm_id,
            "memory_id": record.memory_id,
            "type": self._norm_lower(record.metadata.get("type")) or "lesson",
            "content": record.text,
            "claim": record.text,
            "summary": record.summary,
            "content_obj": record.metadata.get("content_obj"),
            "context": list(record.metadata.get("context", []) or []),
            "confidence": float(record.confidence),
            "timestamp": ts,
            "age": max(0.0, self._now() - ts),
            "persisted": True,
            "source": self._norm_lower(record.metadata.get("source") or record.source_type.value or "system"),
            "revision_trace": list(record.metadata.get("revision_trace", []) or []),
            "numeric_guarded": bool(record.metadata.get("numeric_guarded", False)),
            "domain": self._norm_domain(record.metadata.get("domain")),
            "tags": set(record.tags),
            "kind": record.kind.value,
            "details": details,
            "status": status,
            "verification_status": status,
            "sources": list(record.metadata.get("sources", []) or []),
            "evidence": {"items": [ev.snippet for ev in record.evidence if ev.snippet]},
            "history": list(record.metadata.get("history", []) or []),
            "usage_count": int(record.metadata.get("usage_count", 0) or 0),
            "last_used": record.metadata.get("last_used"),
            "updated_at": record.metadata.get("updated_at"),
            "stability": record.stability,
            "decay_score": record.decay_score,
            "priority": record.priority,
            "contradiction_count": record.contradiction_count,
            "retrieval_hits": record.retrieval_hits,
            "retrieval_misses": record.retrieval_misses,
            "related_ids": list(record.related_ids),
            "source_ref": record.source_ref,
            "signals": record.signals.to_dict(),
        }

    def _row_to_record(self, row: Tuple[Any, ...]) -> MemoryRecord:
        (
            ltm_id, memory_id, t, content, content_obj_raw, ctx_raw, conf, ts, source, rev_raw,
            numeric_guarded, domain, tags_raw, kind, details_raw,
            status, sources_raw, evidence_raw, history_raw, usage_count, last_used, updated_at,
            stability, decay_score, priority, contradiction_count, retrieval_hits,
            retrieval_misses, related_ids_raw, source_ref, signals_raw, metadata_raw
        ) = row

        context = self._json_loads(ctx_raw, default=[])
        if not isinstance(context, list):
            context = []

        revision_trace = self._json_loads(rev_raw, default=[])
        if not isinstance(revision_trace, list):
            revision_trace = []

        content_obj = self._json_loads(content_obj_raw, default=None)

        tags_list = self._json_loads(tags_raw, default=[])
        if not isinstance(tags_list, list):
            tags_list = []

        details = self._json_loads(details_raw, default={})
        if not isinstance(details, dict):
            details = {}
        details = self._ensure_pair_fields(details, content or "")

        sources = self._json_loads(sources_raw, default=[])
        if not isinstance(sources, list):
            sources = []

        evidence = self._json_loads(evidence_raw, default={})
        if not isinstance(evidence, dict):
            evidence = {}

        history = self._json_loads(history_raw, default=[])
        if not isinstance(history, list):
            history = []

        related_ids = self._json_loads(related_ids_raw, default=[])
        if not isinstance(related_ids, list):
            related_ids = []

        signals_data = self._json_loads(signals_raw, default={})
        if not isinstance(signals_data, dict):
            signals_data = {}

        metadata = self._json_loads(metadata_raw, default={})
        if not isinstance(metadata, dict):
            metadata = {}

        evidence_packets: List[EvidencePacket] = []
        for item in evidence.get("items", []) or []:
            s = self._norm_str(item)
            if s:
                evidence_packets.append(
                    EvidencePacket(
                        source_type=self._source_type_from_string(source),
                        source_ref=self._norm_str(source_ref) or None,
                        snippet=s,
                        stance="support",
                        confidence=max(0.0, min(1.0, float(conf or 0.0))),
                    )
                )

        record = MemoryRecord(
            memory_id=self._ensure_memory_id(memory_id or f"ltm_{ltm_id}"),
            text=content or "",
            summary=metadata.get("summary"),
            kind=self._kind_from_strings(kind, t),
            tier=MemoryTier.LONG_TERM,
            source_type=self._source_type_from_string(source or "system"),
            source_ref=self._norm_str(source_ref) or None,
            confidence=float(conf or 0.0),
            stability=float(stability or 0.0),
            decay_score=float(decay_score or 0.0),
            priority=float(priority or 0.5),
            verification_status=self._verification_from_status(status),
            access_count=int(usage_count or 0),
            contradiction_count=int(contradiction_count or 0),
            retrieval_hits=int(retrieval_hits or 0),
            retrieval_misses=int(retrieval_misses or 0),
            tags=self._norm_tags(tags_list),
            related_ids=related_ids,
            evidence=evidence_packets,
            signals=MemorySignals(
                domain=self._norm_domain(signals_data.get("domain") or domain),
                hazard_family=self._norm_domain(signals_data.get("hazard_family")),
                temporal_scope=self._norm_lower(signals_data.get("temporal_scope")) or None,
                novelty=float(signals_data.get("novelty", 0.0) or 0.0),
                urgency=float(signals_data.get("urgency", 0.0) or 0.0),
                usefulness=float(signals_data.get("usefulness", 0.0) or 0.0),
                metadata=signals_data.get("metadata", {}) if isinstance(signals_data.get("metadata"), dict) else {},
            ),
            metadata={
                **metadata,
                "ltm_id": int(ltm_id),
                "type": self._norm_lower(t),
                "context": context,
                "timestamp": float(ts or self._now()),
                "source": self._norm_lower(source or "system"),
                "revision_trace": revision_trace,
                "numeric_guarded": bool(int(numeric_guarded or 0)),
                "domain": self._norm_domain(domain),
                "details": details,
                "status": self._norm_lower(status or "unverified"),
                "sources": sources,
                "history": history,
                "usage_count": int(usage_count or 0),
                "last_used": float(last_used) if last_used is not None else None,
                "updated_at": float(updated_at) if updated_at is not None else None,
                "content_obj": content_obj,
                "summary": metadata.get("summary"),
            },
        )
        return record

    # -------------------------------------------------------------------------
    # SQLite row helpers
    # -------------------------------------------------------------------------
    def _fetch_row_by_id(self, ltm_id: int) -> Optional[Tuple[Any, ...]]:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT id, memory_id, type, content, content_obj, context, confidence, timestamp, source, revision_trace,
                   numeric_guarded, domain, tags, kind, details,
                   status, sources, evidence, history, usage_count, last_used, updated_at,
                   stability, decay_score, priority, contradiction_count, retrieval_hits,
                   retrieval_misses, related_ids, source_ref, signals, metadata
            FROM memory
            WHERE id=?
            LIMIT 1
            """,
            (int(ltm_id),),
        )
        return cursor.fetchone()

    def _fetch_row_by_memory_id(self, memory_id: str) -> Optional[Tuple[Any, ...]]:
        mid = self._ensure_memory_id(memory_id)
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT id, memory_id, type, content, content_obj, context, confidence, timestamp, source, revision_trace,
                   numeric_guarded, domain, tags, kind, details,
                   status, sources, evidence, history, usage_count, last_used, updated_at,
                   stability, decay_score, priority, contradiction_count, retrieval_hits,
                   retrieval_misses, related_ids, source_ref, signals, metadata
            FROM memory
            WHERE LOWER(COALESCE(memory_id,''))=?
            ORDER BY updated_at DESC, timestamp DESC, id DESC
            LIMIT 1
            """,
            (mid,),
        )
        return cursor.fetchone()

    def _find_exact_row(
        self,
        msg_type: str,
        content: str,
        domain: Optional[str],
        kind: Optional[str],
    ) -> Optional[Tuple[Any, ...]]:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT id, memory_id, type, content, content_obj, context, confidence, timestamp, source, revision_trace,
                   numeric_guarded, domain, tags, kind, details,
                   status, sources, evidence, history, usage_count, last_used, updated_at,
                   stability, decay_score, priority, contradiction_count, retrieval_hits,
                   retrieval_misses, related_ids, source_ref, signals, metadata
            FROM memory
            WHERE type=? AND content=? AND COALESCE(domain,'')=? AND COALESCE(kind,'')=?
            ORDER BY updated_at DESC, timestamp DESC, id DESC
            LIMIT 1
            """,
            (msg_type, content, domain or "", kind or ""),
        )
        return cursor.fetchone()

    def _candidate_rows(
        self,
        msg_type: str,
        domain: Optional[str],
        kind: Optional[str],
        limit: int = 250,
    ) -> List[Tuple[Any, ...]]:
        cursor = self.conn.cursor()

        if domain and kind:
            cursor.execute(
                """
                SELECT id, memory_id, type, content, content_obj, context, confidence, timestamp, source, revision_trace,
                       numeric_guarded, domain, tags, kind, details,
                       status, sources, evidence, history, usage_count, last_used, updated_at,
                       stability, decay_score, priority, contradiction_count, retrieval_hits,
                       retrieval_misses, related_ids, source_ref, signals, metadata
                FROM memory
                WHERE type=? AND (domain=? OR domain IS NULL OR domain='') AND (kind=? OR kind IS NULL OR kind='')
                ORDER BY updated_at DESC, timestamp DESC, id DESC
                LIMIT ?
                """,
                (msg_type, domain, kind, limit),
            )
            rows = cursor.fetchall()
            if rows:
                return rows

        if domain:
            cursor.execute(
                """
                SELECT id, memory_id, type, content, content_obj, context, confidence, timestamp, source, revision_trace,
                       numeric_guarded, domain, tags, kind, details,
                       status, sources, evidence, history, usage_count, last_used, updated_at,
                       stability, decay_score, priority, contradiction_count, retrieval_hits,
                       retrieval_misses, related_ids, source_ref, signals, metadata
                FROM memory
                WHERE type=? AND (domain=? OR domain IS NULL OR domain='')
                ORDER BY updated_at DESC, timestamp DESC, id DESC
                LIMIT ?
                """,
                (msg_type, domain, limit),
            )
            rows = cursor.fetchall()
            if rows:
                return rows

        cursor.execute(
            """
            SELECT id, memory_id, type, content, content_obj, context, confidence, timestamp, source, revision_trace,
                   numeric_guarded, domain, tags, kind, details,
                   status, sources, evidence, history, usage_count, last_used, updated_at,
                   stability, decay_score, priority, contradiction_count, retrieval_hits,
                   retrieval_misses, related_ids, source_ref, signals, metadata
            FROM memory
            WHERE type=?
            ORDER BY updated_at DESC, timestamp DESC, id DESC
            LIMIT ?
            """,
            (msg_type, limit),
        )
        return cursor.fetchall()

    # -------------------------------------------------------------------------
    # Record mutation helpers
    # -------------------------------------------------------------------------
    def _reinforce_existing(
        self,
        existing: MemoryRecord,
        incoming: MemoryRecord,
    ) -> MemoryRecord:
        record = replace(existing)

        existing_status = self._status_from_verification(existing.verification_status)
        incoming_status = self._status_from_verification(incoming.verification_status)

        conf_bump = self.VERIFIED_REINFORCE_CONFIDENCE_BUMP if (
            existing_status == "verified" or incoming_status == "verified"
        ) else self.REINFORCE_CONFIDENCE_BUMP

        record.confidence = min(
            self.REINFORCE_CONFIDENCE_CAP,
            max(existing.confidence, incoming.confidence) + conf_bump,
        )

        record.tags = self._norm_tags(self._merge_lists_unique(existing.tags, incoming.tags, norm_lower=True))
        record.related_ids = self._merge_lists_unique(existing.related_ids, incoming.related_ids, norm_lower=False)

        merged_evidence: List[EvidencePacket] = list(existing.evidence)
        existing_snips = {self._norm_str(ev.snippet) for ev in existing.evidence if ev.snippet}
        for ev in incoming.evidence:
            s = self._norm_str(ev.snippet)
            if s and s not in existing_snips:
                merged_evidence.append(ev)
                existing_snips.add(s)
        record.evidence = merged_evidence

        record.access_count = int(existing.access_count) + max(1, int(incoming.access_count or 0) or 1)
        record.priority = max(existing.priority, incoming.priority)
        record.stability = max(existing.stability, incoming.stability)

        details = self._merge_details(
            existing.metadata.get("details", {}),
            incoming.metadata.get("details", {}),
        )
        details = self._ensure_pair_fields(details, record.text)

        revision_trace = self._merge_lists_unique(
            existing.metadata.get("revision_trace", []),
            incoming.metadata.get("revision_trace", []),
            norm_lower=False,
        )
        revision_trace.append("Reinforced via repeated exposure in LTM")

        history = self._merge_lists_unique(
            existing.metadata.get("history", []),
            incoming.metadata.get("history", []),
            norm_lower=False,
        )
        history = self._append_history_event(
            history,
            "reinforced_existing",
            {
                "incoming_source": self._norm_lower(incoming.metadata.get("source") or incoming.source_type.value),
                "incoming_confidence": float(incoming.confidence),
            },
        )

        sources = self._normalize_sources(
            self._merge_lists_unique(
                existing.metadata.get("sources", []),
                incoming.metadata.get("sources", []),
                norm_lower=True,
            ),
            source=incoming.metadata.get("source"),
        )

        record.metadata = {
            **existing.metadata,
            "details": details,
            "revision_trace": revision_trace,
            "history": history,
            "sources": sources,
            "last_used": self._now(),
            "updated_at": self._now(),
            "usage_count": int(existing.metadata.get("usage_count", 0) or 0)
            + max(1, int(incoming.metadata.get("usage_count", 0) or 0) or 1),
        }

        if incoming.verification_status == VerificationStatus.VERIFIED:
            record.verification_status = VerificationStatus.VERIFIED
        elif existing.verification_status == VerificationStatus.DISPUTED and incoming.verification_status in {
            VerificationStatus.VERIFIED,
            VerificationStatus.PROVISIONAL,
        }:
            record.verification_status = incoming.verification_status

        return record

    def _merge_revision(
        self,
        existing: MemoryRecord,
        incoming: MemoryRecord,
    ) -> MemoryRecord:
        record = replace(existing)

        old_strength = self._mem_strength(
            confidence=existing.confidence,
            context=existing.metadata.get("context", []),
            timestamp=float(existing.metadata.get("timestamp", self._now()) or self._now()),
            usage_count=int(existing.metadata.get("usage_count", 0) or 0),
            status=self._status_from_verification(existing.verification_status),
        )
        new_strength = self._mem_strength(
            confidence=incoming.confidence,
            context=incoming.metadata.get("context", []),
            timestamp=float(incoming.metadata.get("timestamp", self._now()) or self._now()),
            usage_count=int(incoming.metadata.get("usage_count", 0) or 0),
            status=self._status_from_verification(incoming.verification_status),
        )

        replace_content = new_strength >= (old_strength + self.REVISION_REPLACE_MARGIN)

        if replace_content:
            record.text = incoming.text
            record.summary = incoming.summary or existing.summary
            record.source_type = incoming.source_type
            record.source_ref = incoming.source_ref or existing.source_ref
            record.confidence = max(existing.confidence, incoming.confidence)
            record.metadata["timestamp"] = float(incoming.metadata.get("timestamp", self._now()) or self._now())
        else:
            record.confidence = max(existing.confidence, incoming.confidence)

        record.tags = self._norm_tags(self._merge_lists_unique(existing.tags, incoming.tags, norm_lower=True))
        record.related_ids = self._merge_lists_unique(existing.related_ids, incoming.related_ids, norm_lower=False)
        record.priority = max(existing.priority, incoming.priority)
        record.stability = max(existing.stability, incoming.stability)

        merged_evidence: List[EvidencePacket] = list(existing.evidence)
        existing_snips = {self._norm_str(ev.snippet) for ev in existing.evidence if ev.snippet}
        for ev in incoming.evidence:
            s = self._norm_str(ev.snippet)
            if s and s not in existing_snips:
                merged_evidence.append(ev)
                existing_snips.add(s)
        record.evidence = merged_evidence

        revision_trace = self._merge_lists_unique(
            existing.metadata.get("revision_trace", []),
            incoming.metadata.get("revision_trace", []),
            norm_lower=False,
        )
        revision_trace.append(
            "Merged semantic revision in LTM"
            + (" (incoming content adopted)" if replace_content else " (metadata/evidence absorbed)")
        )

        history = self._merge_lists_unique(
            existing.metadata.get("history", []),
            incoming.metadata.get("history", []),
            norm_lower=False,
        )
        history = self._append_history_event(
            history,
            "merged_revision",
            {
                "replaced_content": bool(replace_content),
                "incoming_source": self._norm_lower(incoming.metadata.get("source") or incoming.source_type.value),
                "incoming_confidence": float(incoming.confidence),
            },
        )

        details = self._merge_details(
            existing.metadata.get("details", {}),
            incoming.metadata.get("details", {}),
        )
        details = self._ensure_pair_fields(details, record.text)

        sources = self._normalize_sources(
            self._merge_lists_unique(
                existing.metadata.get("sources", []),
                incoming.metadata.get("sources", []),
                norm_lower=True,
            ),
            source=incoming.metadata.get("source"),
        )

        record.metadata = {
            **existing.metadata,
            "details": details,
            "revision_trace": revision_trace,
            "history": history,
            "sources": sources,
            "last_used": self._now(),
            "updated_at": self._now(),
            "usage_count": int(existing.metadata.get("usage_count", 0) or 0)
            + max(1, int(incoming.metadata.get("usage_count", 0) or 0) or 1),
            "summary": record.summary,
        }

        if incoming.verification_status == VerificationStatus.VERIFIED:
            record.verification_status = VerificationStatus.VERIFIED
        elif existing.verification_status == VerificationStatus.VERIFIED:
            record.verification_status = VerificationStatus.VERIFIED
        elif incoming.verification_status == VerificationStatus.PROVISIONAL:
            record.verification_status = VerificationStatus.PROVISIONAL

        return record

    def _link_contested_variant(
        self,
        existing: MemoryRecord,
        incoming: MemoryRecord,
    ) -> Tuple[MemoryRecord, MemoryRecord]:
        existing_rec = replace(existing)
        incoming_rec = replace(incoming)

        existing_id = int(existing.metadata.get("ltm_id"))
        existing_mem_id = self._ensure_memory_id(existing.memory_id)
        incoming_mem_id = self._ensure_memory_id(incoming.memory_id)

        existing_ref = existing.text[:160]
        incoming_ref = incoming.text[:160]

        reason = "material_conflict"
        if self._numeric_conflict(incoming.text, existing.text):
            reason = "numeric_conflict"
        elif self._negation_conflict(incoming.text, existing.text):
            reason = "negation_conflict"

        existing_details = self._ensure_pair_fields(dict(existing_rec.metadata.get("details", {}) or {}), existing_rec.text)
        incoming_details = self._ensure_pair_fields(dict(incoming_rec.metadata.get("details", {}) or {}), incoming_rec.text)

        pair_id = self._norm_str(existing_details.get("conflict_pair_id")) or self._norm_str(incoming_details.get("conflict_pair_id"))
        if not pair_id:
            pair_id = self._make_conflict_pair_id()

        existing_details.setdefault("conflict_with_ids", [])
        existing_details.setdefault("conflict_with_text", [])
        existing_details.setdefault("conflict_with", [])

        if incoming_mem_id and incoming_mem_id not in existing_details["conflict_with_ids"]:
            existing_details["conflict_with_ids"].append(incoming_mem_id)
        if incoming_ref and incoming_ref not in existing_details["conflict_with_text"]:
            existing_details["conflict_with_text"].append(incoming_ref)
        if incoming_ref and incoming_ref not in existing_details["conflict_with"]:
            existing_details["conflict_with"].append(incoming_ref)

        existing_details["contested"] = True
        existing_details["conflict"] = True
        existing_details["verification_status"] = "contested"
        existing_details["conflict_reason"] = reason
        existing_details["conflict_pair_id"] = pair_id
        existing_details["conflict_role"] = "a"
        existing_details["paired_memory_id"] = incoming_mem_id
        existing_details["pair_status"] = "complete"

        existing_rec.verification_status = VerificationStatus.DISPUTED
        existing_rec.contradiction_count += 1
        existing_rec.metadata["details"] = existing_details
        existing_rec.metadata["revision_trace"] = list(existing_rec.metadata.get("revision_trace", []) or [])
        existing_rec.metadata["revision_trace"].append(
            f"Marked contested in LTM conflict_pair_id={pair_id} against incoming conflicting claim: '{incoming_ref}'"
        )
        existing_rec.metadata["history"] = self._append_history_event(
            list(existing_rec.metadata.get("history", []) or []),
            "marked_contested",
            {"reason": reason, "conflict_pair_id": pair_id},
        )
        existing_rec.metadata["updated_at"] = self._now()
        existing_rec.metadata["last_used"] = self._now()

        incoming_details.setdefault("conflict_with_ids", [])
        incoming_details.setdefault("conflict_with_text", [])
        incoming_details.setdefault("conflict_with", [])

        if existing_mem_id and existing_mem_id not in incoming_details["conflict_with_ids"]:
            incoming_details["conflict_with_ids"].append(existing_mem_id)
        if existing_ref and existing_ref not in incoming_details["conflict_with_text"]:
            incoming_details["conflict_with_text"].append(existing_ref)
        if existing_ref and existing_ref not in incoming_details["conflict_with"]:
            incoming_details["conflict_with"].append(existing_ref)

        incoming_details["contested"] = True
        incoming_details["conflict"] = True
        incoming_details["verification_status"] = "contested"
        incoming_details["conflict_reason"] = reason
        incoming_details["conflict_pair_id"] = pair_id
        incoming_details["conflict_role"] = "b"
        incoming_details["paired_memory_id"] = existing_mem_id
        incoming_details["pair_status"] = "complete"

        incoming_rec.verification_status = VerificationStatus.DISPUTED
        incoming_rec.contradiction_count += 1
        incoming_rec.metadata["details"] = incoming_details
        incoming_rec.metadata["revision_trace"] = list(incoming_rec.metadata.get("revision_trace", []) or [])
        incoming_rec.metadata["revision_trace"].append(
            f"Stored as contested conflicting variant in LTM conflict_pair_id={pair_id} against existing id={existing_id}"
        )
        incoming_rec.metadata["history"] = self._append_history_event(
            list(incoming_rec.metadata.get("history", []) or []),
            "stored_contested_variant",
            {"reason": reason, "conflict_with_id": existing_id, "conflict_pair_id": pair_id},
        )
        incoming_rec.metadata["updated_at"] = self._now()
        incoming_rec.metadata["last_used"] = self._now()

        return existing_rec, incoming_rec

    # -------------------------------------------------------------------------
    # Row persistence
    # -------------------------------------------------------------------------
    def _update_row(self, record: MemoryRecord) -> bool:
        ltm_id = record.metadata.get("ltm_id")
        if ltm_id is None:
            return False

        details = self._ensure_pair_fields(
            self._normalize_details(record.metadata.get("details", {})),
            record.text,
        )

        cursor = self.conn.cursor()
        cursor.execute(
            """
            UPDATE memory
            SET memory_id=?,
                type=?,
                content=?,
                content_obj=?,
                context=?,
                confidence=?,
                timestamp=?,
                source=?,
                revision_trace=?,
                numeric_guarded=?,
                domain=?,
                tags=?,
                kind=?,
                details=?,
                status=?,
                sources=?,
                evidence=?,
                history=?,
                usage_count=?,
                last_used=?,
                updated_at=?,
                stability=?,
                decay_score=?,
                priority=?,
                contradiction_count=?,
                retrieval_hits=?,
                retrieval_misses=?,
                related_ids=?,
                source_ref=?,
                signals=?,
                metadata=?
            WHERE id=?
            """,
            (
                self._ensure_memory_id(record.memory_id),
                self._norm_lower(record.metadata.get("type")) or "lesson",
                record.text,
                self._json_dumps(record.metadata.get("content_obj"), default=None),
                self._json_dumps(self._normalize_context(record.metadata.get("context", [])), default="[]"),
                float(record.confidence),
                float(record.metadata.get("timestamp", self._now()) or self._now()),
                self._norm_lower(record.metadata.get("source") or record.source_type.value or "system"),
                self._json_dumps(self._normalize_revision_trace(record.metadata.get("revision_trace", [])), default="[]"),
                1 if bool(record.metadata.get("numeric_guarded", False)) else 0,
                self._norm_domain(record.metadata.get("domain")),
                self._json_dumps(self._norm_tags(record.tags), default="[]"),
                record.kind.value,
                self._json_dumps(details, default="{}"),
                self._status_from_verification(record.verification_status),
                self._json_dumps(
                    self._normalize_sources(record.metadata.get("sources", []), source=record.metadata.get("source")),
                    default="[]",
                ),
                self._json_dumps({"items": [ev.snippet for ev in record.evidence if ev.snippet]}, default="{}"),
                self._json_dumps(self._normalize_history(record.metadata.get("history", [])), default="[]"),
                int(record.metadata.get("usage_count", record.access_count) or 0),
                float(record.metadata.get("last_used")) if record.metadata.get("last_used") is not None else None,
                float(record.metadata.get("updated_at", self._now()) or self._now()),
                float(record.stability),
                float(record.decay_score),
                float(record.priority),
                int(record.contradiction_count),
                int(record.retrieval_hits),
                int(record.retrieval_misses),
                self._json_dumps(list(record.related_ids), default="[]"),
                record.source_ref,
                self._json_dumps(record.signals.to_dict(), default="{}"),
                self._json_dumps({**record.metadata, "details": details}, default="{}"),
                int(ltm_id),
            ),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def _insert_record(self, record: MemoryRecord) -> Optional[int]:
        details = self._ensure_pair_fields(
            self._normalize_details(record.metadata.get("details", {})),
            record.text,
        )

        record.memory_id = self._ensure_memory_id(record.memory_id)

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO memory (
                memory_id, type, content, content_obj, context, confidence, timestamp, source, revision_trace,
                numeric_guarded, domain, tags, kind, details,
                status, sources, evidence, history, usage_count, last_used, updated_at,
                stability, decay_score, priority, contradiction_count, retrieval_hits,
                retrieval_misses, related_ids, source_ref, signals, metadata
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.memory_id,
                self._norm_lower(record.metadata.get("type")) or "lesson",
                record.text,
                self._json_dumps(record.metadata.get("content_obj"), default=None),
                self._json_dumps(self._normalize_context(record.metadata.get("context", [])), default="[]"),
                float(record.confidence),
                float(record.metadata.get("timestamp", self._now()) or self._now()),
                self._norm_lower(record.metadata.get("source") or record.source_type.value or "system"),
                self._json_dumps(self._normalize_revision_trace(record.metadata.get("revision_trace", [])), default="[]"),
                1 if bool(record.metadata.get("numeric_guarded", False)) else 0,
                self._norm_domain(record.metadata.get("domain")),
                self._json_dumps(self._norm_tags(record.tags), default="[]"),
                record.kind.value,
                self._json_dumps(details, default="{}"),
                self._status_from_verification(record.verification_status),
                self._json_dumps(
                    self._normalize_sources(record.metadata.get("sources", []), source=record.metadata.get("source")),
                    default="[]",
                ),
                self._json_dumps({"items": [ev.snippet for ev in record.evidence if ev.snippet]}, default="{}"),
                self._json_dumps(self._normalize_history(record.metadata.get("history", [])), default="[]"),
                int(record.metadata.get("usage_count", record.access_count) or 0),
                float(record.metadata.get("last_used")) if record.metadata.get("last_used") is not None else None,
                float(record.metadata.get("updated_at", self._now()) or self._now()),
                float(record.stability),
                float(record.decay_score),
                float(record.priority),
                int(record.contradiction_count),
                int(record.retrieval_hits),
                int(record.retrieval_misses),
                self._json_dumps(list(record.related_ids), default="[]"),
                record.source_ref,
                self._json_dumps(record.signals.to_dict(), default="{}"),
                self._json_dumps({**record.metadata, "details": details}, default="{}"),
            ),
        )
        self.conn.commit()
        new_id = int(cursor.lastrowid)
        record.metadata["ltm_id"] = new_id
        return new_id

    # -------------------------------------------------------------------------
    # Legacy weak-row pruning
    # -------------------------------------------------------------------------
    def _prune_conflicts(
        self,
        msg_type: str,
        new_content: str,
        new_confidence: float,
        new_context: list,
        new_domain: Optional[str],
        new_kind: Optional[str],
        new_timestamp: float,
        new_details: Optional[Dict[str, Any]] = None,
    ) -> None:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT id, memory_id, type, content, content_obj, context, confidence, timestamp, source, revision_trace,
                   numeric_guarded, domain, tags, kind, details,
                   status, sources, evidence, history, usage_count, last_used, updated_at,
                   stability, decay_score, priority, contradiction_count, retrieval_hits,
                   retrieval_misses, related_ids, source_ref, signals, metadata
            FROM memory
            WHERE type=?
            """,
            (msg_type,),
        )

        new_text = new_content or ""
        new_ts = float(new_timestamp or self._now())
        new_dom = self._norm_domain(new_domain)
        new_k = self._norm_kind(new_kind)
        nd = self._ensure_pair_fields(dict(new_details or {}), new_text)

        new_strength = self._mem_strength(
            float(new_confidence or 0.0),
            new_context or [],
            new_ts,
            usage_count=0,
            status="unverified",
        )

        for row in cursor.fetchall():
            existing = self._row_to_record(row)
            old_text = existing.text or ""
            old_conf = float(existing.confidence)
            old_ts = float(existing.metadata.get("timestamp", 0.0) or 0.0)
            old_dom = self._norm_domain(existing.metadata.get("domain"))
            old_k = existing.kind.value if existing.kind else None
            old_status = self._status_from_verification(existing.verification_status)
            old_ctx = self._normalize_context(existing.metadata.get("context", []))
            od = self._ensure_pair_fields(dict(existing.metadata.get("details", {}) or {}), old_text)

            if self._same_conflict_pair_details(nd, od):
                continue
            if self._is_conflict_linked_details(nd) or self._is_conflict_linked_details(od):
                continue

            if old_conf >= 0.9 or len(old_ctx) > 0 or old_status == "verified" or int(existing.metadata.get("usage_count", 0) or 0) >= 2:
                continue

            is_num_conflict = self._numeric_conflict(new_text, old_text)
            is_dup = self._near_duplicate(new_text, old_text, new_dom, old_dom, new_k, old_k, nd, od)

            if not (is_num_conflict or is_dup):
                continue

            old_strength = self._mem_strength(
                old_conf,
                old_ctx,
                old_ts,
                usage_count=int(existing.metadata.get("usage_count", 0) or 0),
                status=old_status,
            )
            margin = new_strength - old_strength

            delete_old = False
            if is_dup:
                delete_old = margin >= 0.12
            elif is_num_conflict:
                delete_old = margin >= 0.20

            if delete_old:
                cursor.execute("DELETE FROM memory WHERE id=?", (int(existing.metadata["ltm_id"]),))
                self._debug_info(
                    f"Pruned weak legacy row id={existing.metadata['ltm_id']} "
                    f"(numeric_conflict={is_num_conflict}, near_dup={is_dup}, margin={margin:.2f})"
                )

        self.conn.commit()

    # -------------------------------------------------------------------------
    # Intake normalization
    # -------------------------------------------------------------------------
    def _normalize_incoming_message(self, msg: Dict[str, Any], now: float) -> Optional[MemoryRecord]:
        record = self._record_from_legacy_dict(msg)
        if record is None:
            self._debug_reject("invalid_or_guardrailed", self._norm_lower(msg.get("type")), self._norm_str(msg.get("content")))
            return None

        msg_type = self._norm_lower(record.metadata.get("type"))
        if msg_type not in self.VALID_TYPES:
            self._debug_reject("invalid_type", msg_type, record.text)
            return None

        record.memory_id = self._ensure_memory_id(record.memory_id)

        details = self._ensure_pair_fields(
            self._normalize_details(record.metadata.get("details", {})),
            record.text,
        )
        record.metadata["details"] = details

        record.metadata.setdefault("timestamp", float(now))
        record.metadata.setdefault("updated_at", float(now))
        record.metadata.setdefault("usage_count", 0)
        record.metadata.setdefault("history", [])
        record.metadata.setdefault("sources", self._normalize_sources(None, source=record.metadata.get("source")))
        record.metadata.setdefault("revision_trace", [])
        record.tier = MemoryTier.LONG_TERM

        return record

    # -------------------------------------------------------------------------
    # Store APIs
    # -------------------------------------------------------------------------
    def store(
        self,
        messages: Union[Dict[str, Any], Sequence[Dict[str, Any]]],
        include_human: bool = True,
        include_system: bool = True,
    ) -> int:
        detailed = self.store_detailed(
            messages=messages,
            include_human=include_human,
            include_system=include_system,
        )
        return int(detailed.get("stored", 0) or 0)

    def store_detailed(
        self,
        messages: Union[Dict[str, Any], Sequence[Dict[str, Any]]],
        include_human: bool = True,
        include_system: bool = True,
    ) -> Dict[str, Any]:
        if not messages:
            return {"stored": 0, "results": []}

        if isinstance(messages, dict):
            messages = [messages]
        elif not isinstance(messages, (list, tuple)):
            return {"stored": 0, "results": []}

        results: List[Dict[str, Any]] = []
        stored = 0
        now = self._now()
        seen = set()

        for msg in messages:
            if not isinstance(msg, dict):
                continue

            source = self._norm_lower(msg.get("source") or "system")
            msg_type = self._norm_lower(msg.get("type"))

            if source == "user" and not include_human:
                self._debug_reject("human_excluded", msg_type, self._norm_str(msg.get("content")))
                results.append({"action": "rejected", "reason": "human_excluded"})
                continue

            if source == "system" and not include_system:
                self._debug_reject("system_excluded", msg_type, self._norm_str(msg.get("content")))
                results.append({"action": "rejected", "reason": "system_excluded"})
                continue

            incoming = self._normalize_incoming_message(msg, now)
            if incoming is None:
                results.append({"action": "rejected", "reason": "invalid_or_guardrailed"})
                continue

            incoming_details = dict(incoming.metadata.get("details", {}) or {})
            incoming_details = self._ensure_pair_fields(incoming_details, incoming.text)
            incoming.metadata["details"] = incoming_details

            dedupe_key = (
                self._norm_lower(incoming.metadata.get("type")),
                incoming.text.lower(),
                self._norm_domain(incoming.metadata.get("domain")) or "",
                incoming.kind.value or "",
                self._norm_str(incoming.memory_id),
            )
            if dedupe_key in seen:
                self._debug_reject("call_level_duplicate", self._norm_lower(incoming.metadata.get("type")), incoming.text)
                results.append({"action": "rejected", "reason": "call_level_duplicate"})
                continue
            seen.add(dedupe_key)

            # Prefer exact memory_id update if it already exists
            if incoming.memory_id:
                by_mid = self._fetch_row_by_memory_id(incoming.memory_id)
                if by_mid:
                    existing_mid = self._row_to_record(by_mid)
                    merged_mid = self._reinforce_existing(existing_mid, incoming)
                    ok_mid = self._update_row(merged_mid)
                    if ok_mid:
                        stored += 1
                        results.append(
                            {
                                "action": "reinforced_existing",
                                "ltm_id": merged_mid.metadata["ltm_id"],
                                "memory_id": merged_mid.memory_id,
                                "content": merged_mid.text,
                            }
                        )
                        continue

            self._prune_conflicts(
                msg_type=self._norm_lower(incoming.metadata.get("type")),
                new_content=incoming.text,
                new_confidence=incoming.confidence,
                new_context=self._normalize_context(incoming.metadata.get("context", [])),
                new_domain=self._norm_domain(incoming.metadata.get("domain")),
                new_kind=incoming.kind.value,
                new_timestamp=float(incoming.metadata.get("timestamp", now) or now),
                new_details=incoming_details,
            )

            exact_row = self._find_exact_row(
                msg_type=self._norm_lower(incoming.metadata.get("type")),
                content=incoming.text,
                domain=self._norm_domain(incoming.metadata.get("domain")),
                kind=incoming.kind.value,
            )
            if exact_row:
                existing = self._row_to_record(exact_row)
                existing_details = self._ensure_pair_fields(
                    dict(existing.metadata.get("details", {}) or {}),
                    existing.text,
                )

                if self._same_conflict_pair_details(incoming_details, existing_details):
                    incoming.metadata["ltm_id"] = existing.metadata["ltm_id"]
                    ok_exact = self._update_row(incoming)
                    if ok_exact:
                        stored += 1
                        results.append(
                            {
                                "action": "reinforced_existing",
                                "ltm_id": incoming.metadata["ltm_id"],
                                "memory_id": incoming.memory_id,
                                "content": incoming.text,
                            }
                        )
                        continue
                    results.append({"action": "rejected", "reason": "exact_pair_update_failed"})
                    continue

                if not self._is_conflict_linked_details(incoming_details) and not self._is_conflict_linked_details(existing_details):
                    reinforced = self._reinforce_existing(existing, incoming)
                    ok = self._update_row(reinforced)
                    if ok:
                        stored += 1
                        results.append(
                            {
                                "action": "reinforced_existing",
                                "ltm_id": reinforced.metadata["ltm_id"],
                                "memory_id": reinforced.memory_id,
                                "content": reinforced.text,
                            }
                        )
                        continue
                    results.append({"action": "rejected", "reason": "update_failed"})
                    continue

            candidates = self._candidate_rows(
                msg_type=self._norm_lower(incoming.metadata.get("type")),
                domain=self._norm_domain(incoming.metadata.get("domain")),
                kind=incoming.kind.value,
                limit=250,
            )

            best_revision: Optional[MemoryRecord] = None
            best_revision_sim = 0.0
            best_conflict: Optional[MemoryRecord] = None
            best_conflict_strength = -999.0

            for row in candidates:
                existing = self._row_to_record(row)
                existing_content = existing.text
                if not existing_content:
                    continue

                existing_details = self._ensure_pair_fields(
                    dict(existing.metadata.get("details", {}) or {}),
                    existing_content,
                )

                if self._same_conflict_pair_details(incoming_details, existing_details):
                    continue

                if self._numeric_conflict(incoming.text, existing_content) or self._negation_conflict(incoming.text, existing_content):
                    strength = self._mem_strength(
                        confidence=existing.confidence,
                        context=self._normalize_context(existing.metadata.get("context", [])),
                        timestamp=float(existing.metadata.get("timestamp", now) or now),
                        usage_count=int(existing.metadata.get("usage_count", 0) or 0),
                        status=self._status_from_verification(existing.verification_status),
                    )
                    if strength > best_conflict_strength:
                        best_conflict_strength = strength
                        best_conflict = existing
                    continue

                if self._revision_candidate(
                    incoming.text,
                    existing_content,
                    self._norm_domain(incoming.metadata.get("domain")),
                    self._norm_domain(existing.metadata.get("domain")),
                    incoming.kind.value,
                    existing.kind.value,
                    incoming_details,
                    existing_details,
                ):
                    sim = self._semantic_similarity(incoming.text, existing_content)
                    if sim > best_revision_sim:
                        best_revision_sim = sim
                        best_revision = existing

            if best_conflict is not None:
                updated_existing, contested_incoming = self._link_contested_variant(best_conflict, incoming)
                ok_existing = self._update_row(updated_existing)
                new_id = None
                if ok_existing:
                    new_id = self._insert_record(contested_incoming)
                if ok_existing and new_id is not None:
                    stored += 1
                    results.append(
                        {
                            "action": "stored_conflict_variant",
                            "ltm_id": int(new_id),
                            "memory_id": contested_incoming.memory_id,
                            "conflict_with_id": int(best_conflict.metadata["ltm_id"]),
                            "conflict_pair_id": self._norm_str(contested_incoming.metadata.get("details", {}).get("conflict_pair_id")),
                            "content": contested_incoming.text,
                        }
                    )
                    continue
                results.append({"action": "rejected", "reason": "conflict_store_failed"})
                continue

            if best_revision is not None:
                merged = self._merge_revision(best_revision, incoming)
                ok = self._update_row(merged)
                if ok:
                    stored += 1
                    results.append(
                        {
                            "action": "merged_revision",
                            "ltm_id": int(best_revision.metadata["ltm_id"]),
                            "memory_id": merged.memory_id,
                            "content": merged.text,
                        }
                    )
                    continue
                results.append({"action": "rejected", "reason": "revision_merge_failed"})
                continue

            new_id = self._insert_record(incoming)
            if new_id is not None:
                stored += 1
                results.append(
                    {
                        "action": "inserted_new",
                        "ltm_id": int(new_id),
                        "memory_id": incoming.memory_id,
                        "content": incoming.text,
                    }
                )
            else:
                results.append({"action": "rejected", "reason": "insert_failed"})

        if getattr(config, "VERBOSE", False):
            self._debug_info(f"Stored/consolidated {stored} entries.")
        return {"stored": stored, "results": results}

    # -------------------------------------------------------------------------
    # Retrieve / search
    # -------------------------------------------------------------------------
    def retrieve(self, msg_type: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()

        if msg_type:
            msg_type = self._norm_lower(msg_type)
            cursor.execute(
                """
                SELECT id, memory_id, type, content, content_obj, context, confidence, timestamp, source, revision_trace,
                       numeric_guarded, domain, tags, kind, details,
                       status, sources, evidence, history, usage_count, last_used, updated_at,
                       stability, decay_score, priority, contradiction_count, retrieval_hits,
                       retrieval_misses, related_ids, source_ref, signals, metadata
                FROM memory
                WHERE type=?
                ORDER BY
                    CASE status
                        WHEN 'verified' THEN 0
                        WHEN 'provisional' THEN 1
                        WHEN 'unverified' THEN 2
                        WHEN 'contested' THEN 3
                        WHEN 'deprecated' THEN 4
                        ELSE 5
                    END,
                    confidence DESC,
                    usage_count DESC,
                    updated_at DESC,
                    timestamp DESC,
                    id DESC
                LIMIT ?
                """,
                (msg_type, limit),
            )
        else:
            cursor.execute(
                """
                SELECT id, memory_id, type, content, content_obj, context, confidence, timestamp, source, revision_trace,
                       numeric_guarded, domain, tags, kind, details,
                       status, sources, evidence, history, usage_count, last_used, updated_at,
                       stability, decay_score, priority, contradiction_count, retrieval_hits,
                       retrieval_misses, related_ids, source_ref, signals, metadata
                FROM memory
                ORDER BY
                    CASE status
                        WHEN 'verified' THEN 0
                        WHEN 'provisional' THEN 1
                        WHEN 'unverified' THEN 2
                        WHEN 'contested' THEN 3
                        WHEN 'deprecated' THEN 4
                        ELSE 5
                    END,
                    confidence DESC,
                    usage_count DESC,
                    updated_at DESC,
                    timestamp DESC,
                    id DESC
                LIMIT ?
                """,
                (limit,),
            )

        rows = cursor.fetchall()
        return [self._record_to_legacy_dict(self._row_to_record(row)) for row in rows]

    def search(
        self,
        query: str,
        limit: int = 25,
        domain: Optional[str] = None,
        msg_type: Optional[str] = None,
        require_tags: Optional[List[str]] = None,
        exclude_status: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []

        q_dom = self._norm_domain(domain)
        q_type = self._norm_lower(msg_type) if msg_type else None
        req_tags = set(self._norm_tags(require_tags))
        excl = set(self._norm_tags(exclude_status))

        window = max(200, limit * 20)
        rows = self.retrieve(msg_type=q_type, limit=window) if q_type else self.retrieve(limit=window)

        q_tokens = self._tokenize_core(query)
        out: List[Tuple[float, Dict[str, Any]]] = []

        for r in rows:
            if q_dom and (r.get("domain") or "") != q_dom:
                continue
            if excl and (r.get("status") in excl):
                continue

            r_tags = {str(t).lower() for t in (r.get("tags") or set())}
            if req_tags and not req_tags.issubset(r_tags):
                continue

            text = r.get("content") or ""
            tag_blob = " ".join(sorted(list(r.get("tags") or []))) if r.get("tags") else ""
            searchable = f"{text} {tag_blob}".strip()
            toks = self._tokenize_core(searchable)

            overlap = len(q_tokens & toks)
            sim = self._semantic_similarity(query, searchable)

            if overlap <= 0 and sim < 0.18:
                continue

            domain_bonus = 0.0
            if q_dom and (r.get("domain") or "") == q_dom:
                domain_bonus = 1.0

            usage_bonus = min(1.0, float(int(r.get("usage_count", 0) or 0)) / 10.0) * 0.10

            score = (
                self.SEARCH_SIM_WEIGHT * sim
                + self.SEARCH_OVERLAP_WEIGHT * min(1.0, overlap / 5.0)
                + self.SEARCH_CONF_WEIGHT * float(r.get("confidence") or 0.0)
                + self.SEARCH_DOMAIN_WEIGHT * domain_bonus
                + usage_bonus
            )
            out.append((score, r))

        out.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in out[:limit]]

    # -------------------------------------------------------------------------
    # Calibration / utility API
    # -------------------------------------------------------------------------
    def get_all_memories(self, limit: int = 5000) -> List[Dict[str, Any]]:
        return self.retrieve(limit=limit)

    def get_memory(self, mem_id: Any) -> Optional[Dict[str, Any]]:
        try:
            mem_id_int = int(mem_id)
        except Exception:
            return None
        row = self._fetch_row_by_id(mem_id_int)
        if not row:
            return None
        return self._record_to_legacy_dict(self._row_to_record(row))

    def get_memory_by_memory_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        row = self._fetch_row_by_memory_id(memory_id)
        if not row:
            return None
        return self._record_to_legacy_dict(self._row_to_record(row))

    def get_memories_by_memory_ids(self, memory_ids: Sequence[str]) -> List[Dict[str, Any]]:
        wanted = {
            self._ensure_memory_id(x)
            for x in (memory_ids or [])
            if self._ensure_memory_id(x)
        }
        if not wanted:
            return []

        cursor = self.conn.cursor()
        placeholders = ",".join("?" for _ in wanted)
        cursor.execute(
            f"""
            SELECT id, memory_id, type, content, content_obj, context, confidence, timestamp, source, revision_trace,
                   numeric_guarded, domain, tags, kind, details,
                   status, sources, evidence, history, usage_count, last_used, updated_at,
                   stability, decay_score, priority, contradiction_count, retrieval_hits,
                   retrieval_misses, related_ids, source_ref, signals, metadata
            FROM memory
            WHERE LOWER(COALESCE(memory_id,'')) IN ({placeholders})
            ORDER BY updated_at DESC, timestamp DESC, id DESC
            """,
            list(wanted),
        )
        rows = cursor.fetchall()
        return [self._record_to_legacy_dict(self._row_to_record(r)) for r in rows]

    def missing_memory_ids(self, memory_ids: Sequence[str]) -> List[str]:
        wanted = [self._ensure_memory_id(x) for x in (memory_ids or []) if self._ensure_memory_id(x)]
        found = {
            self._ensure_memory_id(r.get("memory_id"))
            for r in self.get_memories_by_memory_ids(wanted)
            if self._ensure_memory_id(r.get("memory_id"))
        }
        return [mid for mid in wanted if mid and mid not in found]

    def get_conflict_pair(self, pair_id: str) -> List[Dict[str, Any]]:
        pid = self._norm_str(pair_id)
        if not pid:
            return []

        rows = self.retrieve(limit=5000)
        out: List[Dict[str, Any]] = []
        for row in rows:
            details = row.get("details", {}) if isinstance(row.get("details"), dict) else {}
            if self._norm_str(details.get("conflict_pair_id")) == pid:
                out.append(row)

        out.sort(key=lambda r: (self._norm_str(r.get("details", {}).get("conflict_role")), self._norm_str(r.get("memory_id"))))
        return out

    def verify_conflict_pair_integrity(self, pair_id: str) -> Dict[str, Any]:
        rows = self.get_conflict_pair(pair_id)
        if not rows:
            return {
                "exists": False,
                "pair_id": pair_id,
                "count": 0,
                "side_a_ok": False,
                "side_b_ok": False,
                "cross_link_ok": False,
                "text_ok": False,
                "pair_status_ok": False,
            }

        roles = {self._norm_str(r.get("details", {}).get("conflict_role")) for r in rows}
        mids = {self._ensure_memory_id(r.get("memory_id")) for r in rows if self._ensure_memory_id(r.get("memory_id"))}
        pair_status_ok = all(
            self._norm_str(r.get("details", {}).get("pair_status")) == "complete"
            for r in rows
        )
        text_ok = all(
            self._norm_str(r.get("details", {}).get("canonical_text"))
            for r in rows
        )

        cross_link_ok = True
        for r in rows:
            paired = self._ensure_memory_id(r.get("details", {}).get("paired_memory_id"))
            if paired and paired not in mids:
                cross_link_ok = False
                break

        return {
            "exists": True,
            "pair_id": pair_id,
            "count": len(rows),
            "side_a_ok": "a" in roles,
            "side_b_ok": "b" in roles,
            "cross_link_ok": cross_link_ok,
            "text_ok": text_ok,
            "pair_status_ok": pair_status_ok,
        }

    def update_memory(self, memory: Dict[str, Any]) -> bool:
        if not isinstance(memory, dict):
            return False
        if "id" not in memory:
            return False

        try:
            mem_id = int(memory["id"])
        except Exception:
            return False

        current = self.get_memory(mem_id)
        if not current:
            return False

        merged = dict(current)
        merged.update(memory)

        record = self._record_from_legacy_dict(merged)
        if record is None:
            return False

        record.metadata["ltm_id"] = mem_id
        record.metadata["updated_at"] = float(memory.get("updated_at", self._now()) or self._now())

        if memory.get("_append_history_event", True):
            hist = list(record.metadata.get("history", []) or [])
            record.metadata["history"] = self._append_history_event(
                hist,
                "update_memory",
                {
                    "status": self._status_from_verification(record.verification_status),
                    "confidence": record.confidence,
                },
            )

        return self._update_row(record)

    def increment_usage(self, mem_id: Any, bump_last_used: bool = True) -> bool:
        try:
            mem_id_int = int(mem_id)
        except Exception:
            return False
        cursor = self.conn.cursor()
        now = self._now()
        if bump_last_used:
            cursor.execute(
                """
                UPDATE memory
                SET usage_count = COALESCE(usage_count, 0) + 1,
                    retrieval_hits = COALESCE(retrieval_hits, 0) + 1,
                    last_used = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (now, now, mem_id_int),
            )
        else:
            cursor.execute(
                """
                UPDATE memory
                SET usage_count = COALESCE(usage_count, 0) + 1,
                    retrieval_hits = COALESCE(retrieval_hits, 0) + 1,
                    updated_at = ?
                WHERE id = ?
                """,
                (now, mem_id_int),
            )
        self.conn.commit()
        return cursor.rowcount > 0

    def touch(self, mem_id: Any) -> bool:
        return self.increment_usage(mem_id, bump_last_used=True)

    def count(self, msg_type: Optional[str] = None, exclude_status: Optional[List[str]] = None) -> int:
        cursor = self.conn.cursor()
        excl = [self._norm_lower(x) for x in (exclude_status or []) if self._norm_lower(x)]

        if msg_type:
            msg_type = self._norm_lower(msg_type)
            if excl:
                placeholders = ",".join("?" for _ in excl)
                cursor.execute(
                    f"SELECT COUNT(*) FROM memory WHERE type=? AND COALESCE(status,'unverified') NOT IN ({placeholders})",
                    [msg_type, *excl],
                )
            else:
                cursor.execute("SELECT COUNT(*) FROM memory WHERE type=?", (msg_type,))
        else:
            if excl:
                placeholders = ",".join("?" for _ in excl)
                cursor.execute(
                    f"SELECT COUNT(*) FROM memory WHERE COALESCE(status,'unverified') NOT IN ({placeholders})",
                    excl,
                )
            else:
                cursor.execute("SELECT COUNT(*) FROM memory")

        row = cursor.fetchone()
        return int(row[0] or 0) if row else 0

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    def sanitize_identity_poison(self) -> int:
        cursor = self.conn.cursor()
        patterns = [
            r"^\s*i am your father\b",
            r"^\s*my name is\b",
        ]
        deleted = 0

        cursor.execute("SELECT id, content FROM memory WHERE type IN ('lesson','identity')")
        for mem_id, content in cursor.fetchall():
            c = (content or "").strip()
            low = c.lower()
            if any(re.search(p, low) for p in patterns):
                cursor.execute("DELETE FROM memory WHERE id=?", (mem_id,))
                deleted += 1

        self.conn.commit()
        if getattr(config, "VERBOSE", False):
            print(f"[LongTermMemory] sanitize_identity_poison deleted {deleted} rows.")
        return deleted

    def close(self) -> None:
        try:
            if self.conn:
                self.conn.close()
        finally:
            self.conn = None
            if getattr(config, "VERBOSE", False):
                print("[LongTermMemory] Connection closed.")