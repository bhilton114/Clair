# FILE: memory/working_memory.py
# Working Memory (v3.9-part1)
#
# Contract-backed rewrite:
# - canonical MemoryRecord storage
# - episodic bridge
# - legacy dict compatibility
# - promotion / conflict / retrieval / reflection support
# - hardened against poisoned IDs like "none"/"null"
# - stronger survival-aware recall path
# - improved literature / planning ranking
# - more practical LTM promotion rules
# - immediate store-time promotion pass for short benchmark runs
# - shared-LTM injection support to avoid duplicate LongTermMemory instances
# - conflict-pair preservation for contradiction / calibration harnesses
# - revision-aware helper scaffolding for precision retrieval mode

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

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
from memory.episodic_memory import EpisodicMemory
from memory.long_term_memory import LongTermMemory


# -----------------------------------------------------------------------------
# Compatibility shim
# -----------------------------------------------------------------------------
def _mr_legacy_status(self: MemoryRecord) -> str:
    status_map = {
        VerificationStatus.UNVERIFIED: "unverified",
        VerificationStatus.PROVISIONAL: "pending",
        VerificationStatus.VERIFIED: "verified",
        VerificationStatus.DISPUTED: "contested",
        VerificationStatus.REJECTED: "deprecated",
    }
    try:
        return status_map.get(
            getattr(self, "verification_status", VerificationStatus.UNVERIFIED),
            "unverified",
        )
    except Exception:
        return "unverified"


def _mr_legacy_details(self: MemoryRecord) -> Dict[str, Any]:
    try:
        details = self.metadata.get("details", {}) if isinstance(self.metadata, dict) else {}
        return details if isinstance(details, dict) else {}
    except Exception:
        return {}


def _mr_legacy_get(self: MemoryRecord, key: str, default: Any = None) -> Any:
    try:
        details = _mr_legacy_details(self)
        status = _mr_legacy_status(self)
        evidence = [
            ev.snippet
            for ev in getattr(self, "evidence", [])
            if getattr(ev, "snippet", None)
        ]
        mapping = {
            "id": getattr(self, "memory_id", None),
            "memory_id": getattr(self, "memory_id", None),
            "type": self.metadata.get("type") if isinstance(self.metadata, dict) else None,
            "content": getattr(self, "text", None),
            "claim": getattr(self, "text", None),
            "summary": getattr(self, "summary", None),
            "confidence": getattr(self, "confidence", None),
            "domain": self.metadata.get("domain") if isinstance(self.metadata, dict) else None,
            "kind": self.kind.value if getattr(self, "kind", None) else None,
            "status": status,
            "verification_status": status,
            "details": details,
            "tags": list(getattr(self, "tags", []) or []),
            "evidence": evidence,
            "context": self.metadata.get("context", []) if isinstance(self.metadata, dict) else [],
            "timestamp": self.metadata.get("timestamp") if isinstance(self.metadata, dict) else None,
            "persisted": self.metadata.get("persisted", False) if isinstance(self.metadata, dict) else False,
            "weight": self.metadata.get("weight", 1.0) if isinstance(self.metadata, dict) else 1.0,
            "content_obj": self.metadata.get("content_obj", {}) if isinstance(self.metadata, dict) else {},
            "source": self.metadata.get("source", "unknown") if isinstance(self.metadata, dict) else "unknown",
            "hazard_family": self.metadata.get("hazard_family") if isinstance(self.metadata, dict) else None,
            "last_verified": details.get("last_verified"),
            "memory_class": details.get("memory_class"),
            "staleness_risk": details.get("staleness_risk"),
            "source_trust": details.get("source_trust"),
            "verified": bool(details.get("verified", False)),
            "pending_verification": bool(details.get("pending_verification", False)),
            "contested": bool(details.get("contested", False)),
            "conflict": bool(details.get("conflict", False) or details.get("contested", False)),
            "evidence_strength": details.get("evidence_strength"),
            "quality": self.metadata.get("quality") if isinstance(self.metadata, dict) else None,
            "reinforcement_count": self.metadata.get("reinforcement_count", 0) if isinstance(self.metadata, dict) else 0,
            "last_reinforced": self.metadata.get("last_reinforced", 0.0) if isinstance(self.metadata, dict) else 0.0,
            "numeric_guarded": self.metadata.get("numeric_guarded", False) if isinstance(self.metadata, dict) else False,
        }
        return mapping.get(key, default)
    except Exception:
        return default


def _mr_legacy_keys(self: MemoryRecord):
    return [
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
        "last_verified",
        "memory_class",
        "staleness_risk",
        "source_trust",
        "verified",
        "pending_verification",
        "contested",
        "conflict",
        "evidence_strength",
        "quality",
        "reinforcement_count",
        "last_reinforced",
        "numeric_guarded",
    ]


def _mr_legacy_items(self: MemoryRecord):
    return [(k, _mr_legacy_get(self, k, None)) for k in _mr_legacy_keys(self)]


def _mr_legacy_contains(self: MemoryRecord, key: object) -> bool:
    return isinstance(key, str) and key in set(_mr_legacy_keys(self))


if not hasattr(MemoryRecord, "get"):
    setattr(MemoryRecord, "get", _mr_legacy_get)
if not hasattr(MemoryRecord, "keys"):
    setattr(MemoryRecord, "keys", _mr_legacy_keys)
if not hasattr(MemoryRecord, "items"):
    setattr(MemoryRecord, "items", _mr_legacy_items)
if not hasattr(MemoryRecord, "__contains__"):
    setattr(MemoryRecord, "__contains__", _mr_legacy_contains)


class WorkingMemory:
    VERSION = "3.9"

    BAD_MEMORY_IDS = {"", "none", "null", "nan"}

    # -------------------------------------------------------------------------
    # Promotion controls
    # -------------------------------------------------------------------------
    PROMOTION_CONFIDENCE = float(getattr(config, "WM_PROMOTION_CONFIDENCE", 0.88))
    PROMOTION_REINFORCEMENTS = int(getattr(config, "WM_PROMOTION_REINFORCEMENTS", 2))
    PROMOTION_MIN_AGE_MINUTES = float(getattr(config, "WM_PROMOTION_MIN_AGE", 0.25))
    MAX_PROMOTIONS_PER_CYCLE = int(getattr(config, "WM_MAX_PROMOTIONS_PER_CYCLE", 4))
    PROMOTION_REQUIRES_VERIFICATION = bool(
        getattr(config, "WM_PROMOTION_REQUIRES_VERIFICATION", True)
    )

    FAST_PROMOTION_CONFIDENCE = float(getattr(config, "WM_FAST_PROMOTION_CONFIDENCE", 0.78))
    FAST_PROMOTION_REINFORCEMENTS = int(getattr(config, "WM_FAST_PROMOTION_REINFORCEMENTS", 0))
    FAST_PROMOTION_MIN_AGE_MINUTES = float(getattr(config, "WM_FAST_PROMOTION_MIN_AGE", 0.0))
    FAST_PROMOTION_QUALITY_FLOOR = float(getattr(config, "WM_FAST_PROMOTION_QUALITY_FLOOR", 0.45))
    FAST_PROMOTION_EVIDENCE_FLOOR = float(getattr(config, "WM_FAST_PROMOTION_EVIDENCE_FLOOR", 0.40))

    # -------------------------------------------------------------------------
    # Retrieval / trust controls
    # -------------------------------------------------------------------------
    RECALL_CONFIDENCE_FLOOR = float(getattr(config, "WM_RECALL_CONFIDENCE_FLOOR", 0.45))
    DEFAULT_MIN_RELEVANCE = float(getattr(config, "WM_MIN_RELEVANCE", 1.5))
    DUPLICATE_SIM_THRESHOLD = float(getattr(config, "WM_DUPLICATE_SIM_THRESHOLD", 0.92))
    CONFLICT_OVERLAP_MIN = int(getattr(config, "WM_CONFLICT_OVERLAP_MIN", 2))
    CONFLICT_CONFIDENCE_DROP = float(getattr(config, "WM_CONFLICT_CONFIDENCE_DROP", 0.15))
    DUPLICATE_REINFORCE_CAP = float(getattr(config, "WM_DUPLICATE_REINFORCE_CAP", 1.0))
    REFLECT_DUPLICATE_SCAN_LIMIT = int(getattr(config, "WM_REFLECT_DUPLICATE_SCAN_LIMIT", 80))

    SURVIVAL_PROVISIONAL_RECALL_FLOOR = float(
        getattr(config, "WM_SURVIVAL_PROVISIONAL_RECALL_FLOOR", 0.58)
    )

    # -------------------------------------------------------------------------
    # Evidence controls
    # -------------------------------------------------------------------------
    EVIDENCE_STRENGTH_WEIGHT = float(getattr(config, "WM_EVIDENCE_STRENGTH_WEIGHT", 1.25))
    EVIDENCE_PROMOTION_FLOOR = float(getattr(config, "WM_EVIDENCE_PROMOTION_FLOOR", 0.45))
    CALIBRATION_WEAK_EVIDENCE_TARGET = float(
        getattr(config, "WM_CAL_WEAK_EVIDENCE_TARGET", 0.80)
    )

    # -------------------------------------------------------------------------
    # Quality gate controls
    # -------------------------------------------------------------------------
    WM_PLANNING_MIN_QUALITY = float(getattr(config, "WM_PLANNING_MIN_QUALITY", 0.42))
    WM_MIN_CONTENT_CHARS = int(getattr(config, "WM_MIN_CONTENT_CHARS", 12))
    WM_MAX_REPEAT_RATIO = float(getattr(config, "WM_MAX_REPEAT_RATIO", 0.45))

    # -------------------------------------------------------------------------
    # Domain retention minima
    # -------------------------------------------------------------------------
    DOMAIN_RETENTION_MINIMA: Dict[str, int] = {
        "clubhouse_build": int(getattr(config, "WM_DOMAIN_KEEP_CLUBHOUSE_BUILD", 8)),
        "planning": int(getattr(config, "WM_DOMAIN_KEEP_PLANNING", 5)),
        "safety": int(getattr(config, "WM_DOMAIN_KEEP_SAFETY", 6)),
        "garden": int(getattr(config, "WM_DOMAIN_KEEP_GARDEN", 6)),
        "electrical": int(getattr(config, "WM_DOMAIN_KEEP_ELECTRICAL", 6)),
        "survival": int(getattr(config, "WM_DOMAIN_KEEP_SURVIVAL", 8)),
        "literature": int(getattr(config, "WM_DOMAIN_KEEP_LITERATURE", 6)),
        "identity": int(getattr(config, "WM_DOMAIN_KEEP_IDENTITY", 2)),
    }

    # -------------------------------------------------------------------------
    # Default confidence by source
    # -------------------------------------------------------------------------
    DEFAULT_CONFIDENCE_BY_SOURCE: Dict[str, float] = {
        "reading": float(getattr(config, "WM_DEFAULT_CONF_READING", 0.70)),
        "user_input": float(getattr(config, "WM_DEFAULT_CONF_USER_INPUT", 0.65)),
        "calibration": float(getattr(config, "WM_DEFAULT_CONF_CALIBRATION", 0.90)),
        "verification": float(getattr(config, "WM_DEFAULT_CONF_VERIFICATION", 0.95)),
        "verified": float(getattr(config, "WM_DEFAULT_CONF_VERIFIED", 0.95)),
        "seed_verified": float(getattr(config, "WM_DEFAULT_CONF_SEED_VERIFIED", 0.98)),
        "system": float(getattr(config, "WM_DEFAULT_CONF_SYSTEM", 0.95)),
        "inferred": float(getattr(config, "WM_DEFAULT_CONF_INFERRED", 0.55)),
        "reflection": float(getattr(config, "WM_DEFAULT_CONF_REFLECTION", 0.60)),
        "unknown": float(getattr(config, "WM_DEFAULT_CONF_UNKNOWN", 0.60)),
        "harness_conflict_injection": float(
            getattr(config, "WM_DEFAULT_CONF_HARNESS_CONFLICT", 0.72)
        ),
    }

    # -------------------------------------------------------------------------
    # Guardrails / keywords / parsing
    # -------------------------------------------------------------------------
    NUMERIC_GUARDRAILS = {
        "water boiling": 100,
        "human bones": 206,
        "everest height": 8848,
    }

    TOKEN_ALIASES = {
        "boiling": "boil",
        "boils": "boil",
        "boiled": "boil",
        "bones": "bone",
        "humans": "human",
        "meters": "meter",
        "degrees": "degree",
        "floods": "flood",
        "flooding": "flood",
        "fires": "fire",
        "earthquakes": "earthquake",
        "aftershocks": "aftershock",
        "roads": "road",
        "chapters": "chapter",
        "chatper": "chapter",
        "chaper": "chapter",
        "summarise": "summarize",
        "themes": "theme",
        "readings": "reading",
        "arguments": "argument",
        "lessons": "lesson",
        "books": "book",
        "sections": "section",
        "revised": "revision",
        "updates": "update",
        "updated": "update",
        "posts": "post",
        "supports": "support",
        "supporting": "support",
    }

    STOPWORDS = {
        "is", "are", "was", "were", "the", "a", "an", "of", "and", "or",
        "to", "in", "on", "for", "my", "your", "you", "i", "we", "it",
        "that", "this", "with", "as", "at", "by", "from", "what", "why",
        "how", "do", "does", "did", "should", "would", "can", "could",
        "tell", "me", "about", "during", "after", "before", "while",
        "now", "please", "today", "tonight",
    }

    COMMON_TOKENS = set(getattr(config, "WM_COMMON_TOKENS", [])) or {
        "human", "humans", "body", "water", "people", "person", "thing", "things",
        "world", "day", "time", "help", "question", "answer",
    }

    HAZARD_FAMILIES: Dict[str, Set[str]] = {
        "fire": {"fire", "smoke", "burn", "burning", "flame", "exit", "heat", "low"},
        "flood": {
            "flood", "flooding", "floodwater", "water", "submerged", "evacuate",
            "higher", "ground", "flash", "road", "roads", "fast", "rising",
        },
        "earthquake": {"earthquake", "aftershock", "shaking", "collapse", "quake", "tremor"},
        "lost": {"lost", "night", "shelter", "signal", "conserve", "stay", "stop", "moving", "woods", "put"},
    }

    SURVIVAL_HINT_TAGS = {
        "survival", "fire", "flood", "earthquake", "lost", "emergency",
        "smoke", "evacuate", "aftershock", "floodwater", "submerged",
    }

    LITERATURE_QUERY_TOKENS = {
        "chapter", "book", "reading", "lesson", "argument", "theme",
        "summary", "summarize", "section", "literature", "novel",
        "poem", "poetry", "story", "appearance", "reality",
        "perception", "truth", "sense", "russell",
    }

    IDENTITY_HINTS = {
        "who are you", "what are you", "your name", "who is clair", "what is clair",
        "identity", "your identity", "introduce yourself", "name",
    }

    REVISION_QUERY_MARKERS = {
        "revision",
        "revised",
        "updated plan",
        "new plan",
        "what changed",
        "latest revision",
        "for this project",
        "update for this project",
        "changed plan",
        "support posts",
        "soft soil",
    }

    _ACTIONISH_RE = re.compile(
        r"\baction[_\s]*\d+|follow_up_from:action_|^\s*action\s+action_",
        re.IGNORECASE,
    )
    _CODEISH_RE = re.compile(
        r"\b(traceback|file \"|line \d+|exception|pip install|import \w+)\b",
        re.IGNORECASE,
    )
    _NUM_PAT = re.compile(r"\d+(?:\.\d+)?")
    _NEGATION_PAT = re.compile(
        r"\b(no|not|never|cannot|can't|isn't|aren't|doesn't|without|false|incorrect)\b",
        re.IGNORECASE,
    )
    _CHAPTER_NUM_RE = re.compile(r"\bchapter[_\s]*(\d+)\b", re.IGNORECASE)
    _SECTION_NUM_RE = re.compile(r"\bsection[_\s]*(\d+)\b", re.IGNORECASE)

    _BANNED_SUBSTRINGS = {
        "chatgpt is your mother",
        "your mother ai",
        "i am your mother ai",
        "i'm your mother ai",
        "you are my mother ai",
    }

    # -------------------------------------------------------------------------
    # Init
    # -------------------------------------------------------------------------
    def __init__(
        self,
        max_history: Optional[int] = None,
        decay_rate: Optional[float] = None,
        preload_long_term: bool = True,
        episodic_memory: Optional[EpisodicMemory] = None,
        long_term_memory: Optional[LongTermMemory] = None,
    ):
        self.max_history = int(max_history or getattr(config, "WORKING_MEMORY_MAX", 200))
        self.decay_rate = float(decay_rate or getattr(config, "MEMORY_DECAY_RATE", 0.98))
        self.max_context = int(getattr(config, "WM_MAX_CONTEXT", 20))

        self.buffer: List[MemoryRecord] = []
        self.type_index: Dict[str, List[MemoryRecord]] = {}
        self.context_buffer: List[MemoryRecord] = []

        self.episodic = episodic_memory or EpisodicMemory()
        self.long_term = long_term_memory or (LongTermMemory() if preload_long_term else None)
        self._promotions_this_cycle = 0

        if self.long_term and preload_long_term:
            self._load_long_term()

    # -------------------------------------------------------------------------
    # Basic helpers
    # -------------------------------------------------------------------------
    def _utcnow(self) -> datetime:
        return datetime.now(timezone.utc)

    def _now_ts(self) -> float:
        return time.time()

    def _clamp01(self, x: Any, default: float = 0.0) -> float:
        try:
            v = float(x)
        except Exception:
            v = float(default)
        return max(0.0, min(1.0, v))

    def _stringify_content(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, (dict, list)):
            try:
                return json.dumps(content, ensure_ascii=False, sort_keys=True)
            except Exception:
                return str(content)
        try:
            return str(content)
        except Exception:
            return ""

    def _safe_str(self, value: Any) -> str:
        if value is None:
            return ""
        try:
            return str(value).strip()
        except Exception:
            return ""

    def _safe_domain(self, domain: Any) -> Optional[str]:
        s = self._safe_str(domain).lower()
        if s in self.BAD_MEMORY_IDS:
            return None
        return s or None

    def _normalize_tags_list(self, tags: Any) -> List[str]:
        if tags is None:
            return []
        if isinstance(tags, (list, tuple, set)):
            out: List[str] = []
            seen: Set[str] = set()
            for tag in tags:
                s = self._safe_str(tag).lower()
                if s and s not in seen:
                    out.append(s)
                    seen.add(s)
            return out
        s = self._safe_str(tags).lower()
        return [s] if s else []

    def _tags_set(self, tags: Any) -> Set[str]:
        return set(self._normalize_tags_list(tags))

    def _details(self, record: MemoryRecord) -> Dict[str, Any]:
        details = record.metadata.setdefault("details", {})
        if not isinstance(details, dict):
            details = {}
            record.metadata["details"] = details
        return details

    def _default_confidence_for_source(self, source: str) -> float:
        src = self._safe_str(source).lower() or "unknown"
        return float(
            self.DEFAULT_CONFIDENCE_BY_SOURCE.get(
                src,
                self.DEFAULT_CONFIDENCE_BY_SOURCE["unknown"],
            )
        )

    def _source_type_from_string(self, source: str) -> SourceType:
        src = self._safe_str(source).lower()
        mapping = {
            "user_input": SourceType.USER_INPUT,
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
        return mapping.get(src, SourceType.UNKNOWN)

    def _source_string(self, record: MemoryRecord) -> str:
        return self._safe_str(
            record.metadata.get("source") or record.source_type.value or "unknown"
        ).lower() or "unknown"

    def _normalize_memory_id_candidate(self, value: Any) -> str:
        s = self._safe_str(value).lower()
        return "" if s in self.BAD_MEMORY_IDS else s

    def _make_memory_id(self, incoming_id: Any = None, incoming_memory_id: Any = None) -> str:
        for candidate in (incoming_id, incoming_memory_id):
            s = self._normalize_memory_id_candidate(candidate)
            if s:
                return s
        return make_memory_record("wm").memory_id.lower()

    def _entry_id(self, item: Union[MemoryRecord, Dict[str, Any], None]) -> str:
        if isinstance(item, MemoryRecord):
            return self._normalize_memory_id_candidate(item.memory_id)
        if isinstance(item, dict):
            return self._make_memory_id(item.get("id"), item.get("memory_id"))
        return ""

    def _record_age_minutes(self, record: MemoryRecord) -> float:
        try:
            ts = float(record.metadata.get("timestamp", 0.0) or 0.0)
            if ts > 0:
                return max(0.0, (self._now_ts() - ts) / 60.0)
            return max(0.0, (self._utcnow() - record.created_at).total_seconds() / 60.0)
        except Exception:
            return 0.0

    # -------------------------------------------------------------------------
    # Conflict pair helpers
    # -------------------------------------------------------------------------
    def _canonicalize_text(self, text: Any) -> str:
        s = self._stringify_content(text).lower()
        s = s.replace("’", "'").replace("–", "-").replace("—", "-")
        s = re.sub(r"[^a-z0-9\s\.\-]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _topic_signature_from_text(self, text: str) -> str:
        toks = [
            t for t in self._token_list_from_text(text or "")
            if t and t not in self.COMMON_TOKENS
        ]
        if not toks:
            return ""
        return " ".join(toks[:6])

    def _numeric_signature_from_text(self, text: str) -> Optional[str]:
        nums = self._NUM_PAT.findall(text or "")
        if not nums:
            return None
        return "|".join(nums[:4])

    def _make_conflict_pair_id(self) -> str:
        return make_memory_record("wm_conflict").memory_id.lower()

    def _ensure_pair_fields(self, record: MemoryRecord) -> None:
        details = self._details(record)
        details.setdefault("canonical_text", self._canonicalize_text(record.text or record.summary or ""))
        details.setdefault("topic_signature", self._topic_signature_from_text(record.text or ""))
        details.setdefault("numeric_signature", self._numeric_signature_from_text(record.text or ""))
        details.setdefault("conflict_pair_id", None)
        details.setdefault("conflict_role", None)
        details.setdefault("paired_memory_id", None)
        details.setdefault("pair_status", None)
        if details.get("conflict_pair_id") and not details.get("pair_status"):
            details["pair_status"] = "pending"

    def _same_conflict_pair(self, a: MemoryRecord, b: MemoryRecord) -> bool:
        ad = self._details(a)
        bd = self._details(b)
        ap = self._safe_str(ad.get("conflict_pair_id"))
        bp = self._safe_str(bd.get("conflict_pair_id"))
        return bool(ap and bp and ap == bp)

    def _is_conflict_linked(self, record: MemoryRecord) -> bool:
        details = self._details(record)
        return bool(
            details.get("conflict_pair_id")
            or details.get("paired_memory_id")
            or details.get("pair_status") in {"pending", "complete", "orphaned"}
        )

    # -------------------------------------------------------------------------
    # Revision helpers
    # -------------------------------------------------------------------------
    def _is_revision_entry(self, record: Optional[MemoryRecord]) -> bool:
        if record is None:
            return False

        metadata = record.metadata if isinstance(record.metadata, dict) else {}
        details = self._details(record)
        tags = set(record.tags or [])
        text_blob = " ".join(
            x for x in [record.text or "", record.summary or ""] if x
        ).lower()

        mtype = self._safe_str(metadata.get("type")).lower()
        domain = self._safe_domain(metadata.get("domain"))
        revision_trace = metadata.get("revision_trace", [])

        if bool(metadata.get("is_revision")):
            return True
        if bool(details.get("is_revision")):
            return True
        if mtype == "revision":
            return True
        if "revision" in tags:
            return True
        if "updated_plan" in tags or "revised_plan" in tags:
            return True
        if "revision" in text_blob or "revised" in text_blob:
            return True
        if domain == "clubhouse_build":
            if "for this project" in text_blob or "what changed" in text_blob:
                return True
        if isinstance(revision_trace, list) and revision_trace:
            if any("revision" in self._safe_str(x).lower() for x in revision_trace):
                return True

        return False

    def _is_revision_query(self, context_profile: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(context_profile, dict):
            return False

        q = self._safe_str(context_profile.get("query_text")).lower()
        if not q:
            return False

        if any(marker in q for marker in self.REVISION_QUERY_MARKERS):
            return True

        q_tags = self._tags_set(context_profile.get("tags"))
        if "revision" in q_tags:
            return True

        return False

    def _tag_revision_record(self, record: MemoryRecord) -> None:
        metadata = record.metadata if isinstance(record.metadata, dict) else {}
        details = self._details(record)
        tags = set(record.tags or [])

        metadata["is_revision"] = True
        details["is_revision"] = True
        tags.add("revision")

        if self._safe_str(metadata.get("type")).lower() == "revision":
            record.kind = MemoryKind.FACT

        record.tags = sorted(tags)

    # -------------------------------------------------------------------------
    # Coercion helpers
    # -------------------------------------------------------------------------
    def _coerce_incoming_record(self, item: Any) -> Optional[MemoryRecord]:
        if isinstance(item, MemoryRecord):
            record = item
            if not self._normalize_memory_id_candidate(record.memory_id):
                record.memory_id = self._make_memory_id()
            record.set_tier(MemoryTier.WORKING)
            self._record_quality(record)
            self._ensure_truth_fields(record)
            self._ensure_pair_fields(record)
            if self._is_revision_entry(record):
                self._tag_revision_record(record)
            return record

        if isinstance(item, dict):
            return self._record_from_legacy_dict(item)

        if is_dataclass(item):
            try:
                return self._record_from_legacy_dict(asdict(item))
            except Exception:
                return None

        try:
            raw: Dict[str, Any] = {}
            for attr in (
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
                "last_verified",
                "memory_class",
                "staleness_risk",
                "conflict",
            ):
                if hasattr(item, attr):
                    raw[attr] = getattr(item, attr)
            if raw:
                return self._record_from_legacy_dict(raw)
        except Exception:
            return None

        return None

    # -------------------------------------------------------------------------
    # Tokenization
    # -------------------------------------------------------------------------
    def _stem(self, w: str) -> str:
        w = (w or "").strip().lower()
        if not w:
            return ""
        w = self.TOKEN_ALIASES.get(w, w)

        if len(w) <= 3:
            return w
        for suf in ("ing", "ed"):
            if w.endswith(suf) and len(w) > len(suf) + 2:
                w = w[:-len(suf)]
                break
        if w.endswith("s") and len(w) > 4 and not w.endswith("ss"):
            w = w[:-1]
        return w

    def _normalize_token_set(self, toks: Iterable[str]) -> Set[str]:
        out: Set[str] = set()
        for t in toks or []:
            nt = self._stem(str(t))
            if nt:
                out.add(nt)
        return out

    def _token_list_from_text(self, text: str) -> List[str]:
        if not text:
            return []
        clean = re.sub(r"[^a-z0-9\s]", " ", text.lower()).strip()
        raw = clean.split()
        toks: List[str] = []
        for w in raw:
            if not w or w in self.STOPWORDS:
                continue
            sw = self._stem(w)
            if sw:
                toks.append(sw)
        return toks

    def _tokens_from_text(self, text: str) -> Set[str]:
        return set(self._token_list_from_text(text))

    def _phrase_pairs_from_text(self, text: str) -> Set[Tuple[str, str]]:
        toks = self._token_list_from_text(text)
        if len(toks) < 2:
            return set()
        return {(toks[i], toks[i + 1]) for i in range(len(toks) - 1)}

    def _normalize_for_phrase_match(self, text: str) -> str:
        s = str(text or "").lower()
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _extract_query_anchor_phrases(self, query_text: str) -> Set[str]:
        toks = self._token_list_from_text(query_text or "")
        phrases: Set[str] = set()
        if len(toks) < 2:
            return phrases

        for i in range(len(toks) - 1):
            a = toks[i]
            b = toks[i + 1]
            if a in self.COMMON_TOKENS or b in self.COMMON_TOKENS:
                continue
            if len(a) < 4 or len(b) < 4:
                continue
            phrases.add(f"{a} {b}")
        return phrases

    def _record_text_blob(self, record: MemoryRecord) -> str:
        tags = " ".join(record.tags)
        return " ".join(x for x in (record.text, record.summary or "", tags) if x).strip()

    def _record_token_set(self, record: MemoryRecord) -> Set[str]:
        cached = record.metadata.get("_semantic_token_cache")
        if isinstance(cached, set) and cached:
            return cached

        merged: Set[str] = set()
        merged |= self._tokens_from_text(record.text)
        merged |= self._tokens_from_text(record.summary or "")
        for tag in record.tags:
            merged |= self._tokens_from_text(tag)

        keywords = record.metadata.get("keywords")
        if isinstance(keywords, (set, list, tuple)):
            merged |= self._normalize_token_set(keywords)

        record.metadata["_semantic_token_cache"] = set(merged)
        return merged

    def _record_phrase_pairs(self, record: MemoryRecord) -> Set[Tuple[str, str]]:
        cached = record.metadata.get("_phrase_pair_cache")
        if isinstance(cached, set):
            return cached

        blob = self._record_text_blob(record)
        pairs = self._phrase_pairs_from_text(blob)
        record.metadata["_phrase_pair_cache"] = set(pairs)
        return pairs

    # -------------------------------------------------------------------------
    # Domain / hazard inference
    # -------------------------------------------------------------------------
    def _extract_chapter_hint(self, text: str) -> Optional[str]:
        m = self._CHAPTER_NUM_RE.search(text or "")
        return m.group(1) if m else None

    def _extract_section_hint(self, text: str) -> Optional[str]:
        m = self._SECTION_NUM_RE.search(text or "")
        return m.group(1) if m else None

    def _hazard_family_from_tokens(self, tokens: Set[str]) -> Optional[str]:
        if not tokens:
            return None
        for fam, vocab in self.HAZARD_FAMILIES.items():
            stems = {self._stem(v) for v in vocab}
            if tokens & stems:
                return fam
        return None

    def _is_identity_query(self, text: str) -> bool:
        q = (text or "").strip().lower()
        return bool(q) and any(h in q for h in self.IDENTITY_HINTS)

    def _is_literature_query(
        self,
        query_text: str,
        context_profile: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if isinstance(context_profile, dict):
            dom = self._safe_domain(context_profile.get("domain"))
            tags = self._tags_set(context_profile.get("tags"))
            if dom == "literature":
                return True
            if {"literature", "reading"} & tags:
                return True

        toks = self._tokens_from_text(query_text or "")
        return bool(toks & self.LITERATURE_QUERY_TOKENS)

    def _is_garden_query(
        self,
        query_text: str,
        context_profile: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if isinstance(context_profile, dict):
            if self._safe_domain(context_profile.get("domain")) == "garden":
                return True
            if "garden" in self._tags_set(context_profile.get("tags")):
                return True

        qk = self._tokens_from_text(query_text or "")
        garden_markers = {
            "garden", "raised", "bed", "beds", "vegetable", "vegetables",
            "soil", "mulch", "watering", "seed", "seedling", "compost",
            "plant", "plants", "tomato", "pepper", "weed", "weeds",
            "drainage", "root", "roots",
        }
        return bool(qk & garden_markers)

    def _garden_query_subtype(self, query_text: str) -> Optional[str]:
        qk = self._tokens_from_text(query_text or "")
        raised_bed_markers = {
            "raised", "bed", "beds", "vegetable", "vegetables", "drainage", "soil", "root", "roots"
        }
        watering_markers = {"water", "watering", "seed", "seedling", "stress", "moisture"}

        if len(qk & raised_bed_markers) >= 2:
            return "raised_beds"
        if len(qk & watering_markers) >= 2:
            return "watering"
        return None

    def _garden_record_subtype(self, record: MemoryRecord) -> Optional[str]:
        toks = self._record_token_set(record) | set(record.tags)
        raised_bed_markers = {
            "raised", "bed", "beds", "vegetable", "vegetables", "drainage", "soil", "root", "roots"
        }
        watering_markers = {"water", "watering", "seed", "seedling", "stress", "moisture"}

        if len(toks & raised_bed_markers) >= 2:
            return "raised_beds"
        if len(toks & watering_markers) >= 2:
            return "watering"
        return None

    def _infer_query_context(self, query_text: str) -> Dict[str, Any]:
        q_tokens = self._tokens_from_text(query_text)
        q_fam = self._hazard_family_from_tokens(q_tokens)

        domain = "general"
        tags: List[str] = []

        garden_markers = {
            "garden", "raised", "bed", "beds", "vegetable", "vegetables",
            "soil", "mulch", "watering", "seed", "seedling", "compost",
            "plant", "plants", "tomato", "pepper", "weed", "weeds",
            "drainage", "root", "roots",
        }
        electrical_markers = {
            "electrical", "wire", "wiring", "breaker", "outlet", "voltage",
            "circuit", "panel", "grounding", "gfi", "gfci", "amp", "amps",
        }
        planning_markers = {
            "plan", "planning", "schedule", "steps", "organize", "organizing",
            "task", "tasks", "priority", "priorities", "sequence",
        }
        safety_markers = {"safety", "safe", "hazard", "risk", "injury", "protective", "ppe"}
        clubhouse_markers = {
            "clubhouse", "framing", "roof", "shed", "build", "building",
            "lumber", "joist", "beam", "foundation", "support", "post", "soil",
        }

        if q_fam:
            domain = "survival"
            tags.extend(["survival", q_fam])
        elif self._is_literature_query(query_text):
            domain = "literature"
            tags.extend(["literature", "reading"])
        elif q_tokens & garden_markers:
            domain = "garden"
            tags.append("garden")
            if "raised" in q_tokens and ("bed" in q_tokens or "beds" in q_tokens):
                tags.append("raised_bed")
            if "vegetable" in q_tokens or "vegetables" in q_tokens:
                tags.append("vegetable")
        elif q_tokens & electrical_markers:
            domain = "electrical"
            tags.append("electrical")
        elif q_tokens & clubhouse_markers:
            domain = "clubhouse_build"
            tags.append("clubhouse_build")
        elif q_tokens & safety_markers:
            domain = "safety"
            tags.append("safety")
        elif q_tokens & planning_markers:
            domain = "planning"
            tags.append("planning")

        if self._is_revision_query({"query_text": query_text, "tags": tags, "domain": domain}):
            tags.append("revision")

        ch = self._extract_chapter_hint(query_text)
        if ch:
            tags.append(f"chapter_{ch}")
        sec = self._extract_section_hint(query_text)
        if sec:
            tags.append(f"section_{sec}")

        return {
            "domain": domain,
            "tags": tags,
            "hazard_family": q_fam,
            "chapter_hint": ch,
            "section_hint": sec,
            "query_text": query_text,
        }

    # -------------------------------------------------------------------------
    # Quality helpers
    # -------------------------------------------------------------------------
    def memory_quality_score(self, content: Any) -> float:
        s = self._stringify_content(content).strip()
        if not s:
            return 0.0

        sl = s.lower()
        if any(b in sl for b in self._BANNED_SUBSTRINGS):
            return 0.0
        if len(s) < self.WM_MIN_CONTENT_CHARS:
            return 0.08
        if self._ACTIONISH_RE.search(sl):
            return 0.12
        if self._CODEISH_RE.search(sl):
            return 0.20

        toks = list(self._tokens_from_text(s))
        if not toks:
            return 0.18

        freq: Dict[str, int] = {}
        for t in toks:
            freq[t] = freq.get(t, 0) + 1
        top = max(freq.values()) if freq else 0
        rep = top / max(1, len(toks))
        if rep >= self.WM_MAX_REPEAT_RATIO and len(toks) >= 6:
            return 0.22

        uniq = len(set(toks))
        density = uniq / max(1, len(toks))
        letters = sum(1 for ch in s if ch.isalpha())
        if letters < 6:
            return 0.18

        length_bonus = min(0.25, len(s) / 400.0)
        density_bonus = min(0.35, density)
        token_bonus = min(0.25, uniq / 20.0)

        score = 0.20 + length_bonus + density_bonus + token_bonus
        return max(0.0, min(1.0, float(score)))

    def _record_quality(self, record: MemoryRecord) -> float:
        q = self.memory_quality_score(record.text or record.summary or "")
        record.metadata["quality"] = q
        self._details(record)["quality"] = q
        return q

    def is_trash_memory(self, item: Union[MemoryRecord, Dict[str, Any]]) -> bool:
        record = item if isinstance(item, MemoryRecord) else self._record_from_legacy_dict(item)
        if record is None:
            return True

        if self._is_committed_action(record):
            return False

        s = (record.text or "").strip()
        if not s:
            return True

        q = float(record.metadata.get("quality", 0.0) or 0.0)
        if q <= 0.0:
            q = self._record_quality(record)

        if q <= 0.15:
            return True
        if self._ACTIONISH_RE.search(s.lower()):
            return True
        return False

    # -------------------------------------------------------------------------
    # Record classification / truth-state helpers
    # -------------------------------------------------------------------------
    def _kind_from_legacy(self, msg: Dict[str, Any]) -> MemoryKind:
        kind = self._safe_str(msg.get("kind")).lower()
        mtype = self._safe_str(msg.get("type")).lower()

        mapping = {
            "fact": MemoryKind.FACT,
            "summary": MemoryKind.SUMMARY,
            "procedure": MemoryKind.PROCEDURE,
            "episode": MemoryKind.EPISODE,
            "hazard": MemoryKind.HAZARD,
            "goal": MemoryKind.GOAL,
            "reflection": MemoryKind.REFLECTION,
            "rule": MemoryKind.RULE,
            "scenario": MemoryKind.SCENARIO,
            "user_preference": MemoryKind.USER_PREFERENCE,
            "revision": MemoryKind.FACT,
            "troubleshooting": MemoryKind.FACT,
        }

        if kind in mapping:
            return mapping[kind]

        if mtype in {"chapter_summary", "section_summary", "summary"}:
            return MemoryKind.SUMMARY
        if mtype in {"committed_action"}:
            return MemoryKind.GOAL
        if mtype in {"identity", "fact", "claim", "lesson", "policy", "revision"}:
            return MemoryKind.FACT
        if mtype in {"feedback", "reflection"}:
            return MemoryKind.REFLECTION
        if mtype in {"reasoning_action", "procedure"}:
            return MemoryKind.PROCEDURE

        return MemoryKind.UNRESOLVED

    def _verification_from_legacy(self, msg: Dict[str, Any]) -> VerificationStatus:
        details = msg.get("details", {}) if isinstance(msg.get("details"), dict) else {}
        status = self._safe_str(
            msg.get("verification_status")
            or msg.get("status")
            or details.get("verification_status")
            or details.get("status")
        ).lower()

        mapping = {
            "verified": VerificationStatus.VERIFIED,
            "unverified": VerificationStatus.UNVERIFIED,
            "pending": VerificationStatus.PROVISIONAL,
            "provisional": VerificationStatus.PROVISIONAL,
            "contested": VerificationStatus.DISPUTED,
            "disputed": VerificationStatus.DISPUTED,
            "deprecated": VerificationStatus.REJECTED,
            "rejected": VerificationStatus.REJECTED,
            "hypothesis": VerificationStatus.PROVISIONAL,
        }
        return mapping.get(status, VerificationStatus.UNVERIFIED)

    def _infer_memory_class(self, record: MemoryRecord) -> str:
        mtype = self._safe_str(record.metadata.get("type")).lower()
        kind = record.kind.value
        details = self._details(record)

        if mtype in {"feedback", "committed_action"}:
            return "episodic"
        if kind in {"procedure"}:
            return "procedural"
        if bool(details.get("is_revision")) or mtype == "revision" or "revision" in set(record.tags):
            return "semantic"
        if kind in {"unresolved"} and bool(details.get("hypothesis", False)):
            return "hypothesis"
        if kind in {"fact", "summary", "rule", "user_preference"}:
            return "semantic"
        if kind in {"episode", "scenario", "reflection"}:
            return "episodic"
        return "semantic"

    def _infer_staleness_risk(self, record: MemoryRecord) -> str:
        content = (record.text or "").lower()
        domain = self._safe_domain(record.metadata.get("domain"))
        tags = set(record.tags)

        if domain in {"operations"}:
            return "high"
        if {"current", "today", "recent", "latest", "dynamic"} & tags:
            return "high"
        if any(x in content for x in ("current", "today", "latest", "right now", "now")):
            return "high"

        guarded_numeric = any(key in content for key in self.NUMERIC_GUARDRAILS)
        if guarded_numeric:
            return "low"
        if domain in {"literature", "garden", "electrical"}:
            return "low"
        if domain in {"survival", "clubhouse_build", "planning", "safety"}:
            return "medium"
        return "medium"

    def _is_committed_action(self, record: MemoryRecord) -> bool:
        return self._safe_str(record.metadata.get("type")).lower() == "committed_action"

    def _is_identity_entry(self, record: MemoryRecord) -> bool:
        if self._safe_str(record.metadata.get("type")).lower() == "identity":
            return True
        tags = set(record.tags)
        content = (record.text or "").lower()
        if "identity" in tags or "self" in tags:
            return True
        if "my name is clair" in content or "i am clair" in content:
            return True
        return False

    def _is_feedbackish_entry(self, record: MemoryRecord) -> bool:
        etype = self._safe_str(record.metadata.get("type")).lower()
        domain = self._safe_domain(record.metadata.get("domain"))
        tags = set(record.tags)
        content = (record.text or "").lower()

        if etype == "feedback":
            return True
        if domain == "operations":
            return True
        if "feedback" in tags:
            return True
        if "calibration feedback" in content or "verdict=confirm" in content or "mem_ids=" in content:
            return True
        return False

    def _trusted_source(self, record: MemoryRecord) -> bool:
        source = self._source_string(record)
        details = self._details(record)
        return (
            source in {"seed_verified", "verified", "system", "verification", "reading"}
            or record.verification_status == VerificationStatus.VERIFIED
            or bool(details.get("verified", False))
            or self._safe_str(details.get("source_trust")).lower() == "trusted"
        )

    def _is_harness_injection(self, record: MemoryRecord) -> bool:
        source = self._source_string(record)
        tags = set(record.tags)
        details = self._details(record)
        return (
            source == "harness_conflict_injection"
            or "harness_conflict_injection" in tags
            or self._safe_str(details.get("applied_via")).lower() == "harness_conflict_injection"
        )

    def _hazard_family_for_record(self, record: MemoryRecord) -> Optional[str]:
        details = self._details(record)
        stored = self._safe_domain(
            record.metadata.get("hazard_family") or details.get("hazard_family")
        )
        if stored:
            return stored

        toks = self._record_token_set(record) | set(record.tags)
        fam = self._hazard_family_from_tokens(toks)
        if fam:
            record.metadata["hazard_family"] = fam
            details["hazard_family"] = fam
        return fam

    def _is_action_guidance_text(self, text: str) -> bool:
        s = (text or "").lower().strip()
        if not s:
            return False

        guidance_patterns = (
            "get low",
            "stay low",
            "crawl low",
            "move to higher ground",
            "do not drive through floodwater",
            "leave low areas",
            "safe exit",
            "nearest safe exit",
            "after the shaking stops",
            "protect your head and neck",
            "drop cover and hold on",
            "stay sheltered",
            "avoid smoke inhalation",
            "watch for aftershocks",
            "signal for help",
            "stay put",
            "conserve energy",
            "leave the fire area",
            "rising water",
            "flood zone",
            "floodwater",
            "evacuate",
            "shut off power",
            "verify the circuit is safe",
        )
        return any(p in s for p in guidance_patterns)

    def _is_survival_memory(self, record: MemoryRecord) -> bool:
        if not isinstance(record, MemoryRecord):
            return False

        content = str(record.text or "").strip().lower()
        if not content:
            return False

        rec_domain = self._safe_domain(record.metadata.get("domain"))
        rec_tags = set(record.tags)
        rec_type = self._safe_str(record.metadata.get("type")).lower()
        rec_kind = record.kind.value

        if rec_domain == "survival":
            return True
        if rec_tags & self.SURVIVAL_HINT_TAGS:
            return True
        if self._hazard_family_for_record(record):
            return True
        if rec_type in {"policy", "lesson", "procedure", "reasoning_action", "committed_action"} and self._is_action_guidance_text(content):
            return True
        if rec_kind in {"procedure", "goal"} and self._is_action_guidance_text(content):
            return True
        return False

    def _survival_recall_allowed(
        self,
        record: MemoryRecord,
        context_profile: Optional[Dict[str, Any]],
    ) -> bool:
        if not self._is_survival_memory(record):
            return False

        details = self._details(record)
        if record.verification_status in {VerificationStatus.DISPUTED, VerificationStatus.REJECTED}:
            return False
        if bool(details.get("superseded", False)):
            return False
        if bool(details.get("recall_blocked", False)) and not (
            record.verification_status == VerificationStatus.PROVISIONAL
            and self._is_action_guidance_text(record.text or "")
        ):
            return False

        if self._trusted_source(record):
            return True

        if record.verification_status == VerificationStatus.PROVISIONAL:
            if record.confidence < float(self.SURVIVAL_PROVISIONAL_RECALL_FLOOR):
                return False
            if not self._is_action_guidance_text(record.text or ""):
                return False

            q_fam = None
            if isinstance(context_profile, dict):
                q_fam = self._safe_domain(context_profile.get("hazard_family"))
            r_fam = self._hazard_family_for_record(record)
            if q_fam and r_fam and q_fam != r_fam:
                return False
            return True

        return record.confidence >= float(self.RECALL_CONFIDENCE_FLOOR)

    def _is_recall_blocked(
        self,
        record: MemoryRecord,
        context_profile: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if self._is_committed_action(record):
            return False

        details = self._details(record)
        ctx_domain = (
            self._safe_domain((context_profile or {}).get("domain"))
            if isinstance(context_profile, dict)
            else None
        )

        if ctx_domain == "survival":
            return not self._survival_recall_allowed(record, context_profile)

        if bool(details.get("recall_blocked", False)):
            return True
        if record.verification_status in {VerificationStatus.DISPUTED, VerificationStatus.REJECTED}:
            return True
        if bool(details.get("superseded", False)):
            return True
        if record.confidence < float(self.RECALL_CONFIDENCE_FLOOR):
            return True
        if (
            record.verification_status == VerificationStatus.PROVISIONAL
            and self._safe_str(record.metadata.get("type")).lower() != "identity"
            and bool(details.get("pending_verification", True))
        ):
            return True
        return False

    def _planning_allowed(self, record: MemoryRecord) -> bool:
        if self._is_committed_action(record):
            details = self._details(record)
            return not (
                bool(details.get("contested", False))
                or bool(details.get("superseded", False))
                or bool(details.get("recall_blocked", False))
            )

        if self.is_trash_memory(record):
            return False

        q = float(record.metadata.get("quality", 0.0) or 0.0)
        if q < float(self.WM_PLANNING_MIN_QUALITY):
            return False

        details = self._details(record)
        if details.get("exclude_from_planning") is True:
            return False
        if record.verification_status in {VerificationStatus.DISPUTED, VerificationStatus.REJECTED}:
            return False
        if bool(details.get("superseded", False)):
            return False
        if bool(details.get("recall_blocked", False)):
            return False
        if (
            record.verification_status == VerificationStatus.PROVISIONAL
            and self._safe_str(record.metadata.get("type")).lower() not in {"identity", "policy"}
            and not self._is_survival_memory(record)
        ):
            return False
        return True

    def _evidence_strength(self, record: MemoryRecord) -> float:
        details = self._details(record)
        source = self._source_string(record)
        evidence = record.evidence
        verified = record.verification_status == VerificationStatus.VERIFIED
        pending = record.verification_status == VerificationStatus.PROVISIONAL
        contested = record.verification_status == VerificationStatus.DISPUTED
        hypothesis = bool(details.get("hypothesis", False))
        last_verified = details.get("last_verified")

        score = 0.0
        if source != "unknown":
            score += 0.20
        if len(evidence) >= 1:
            score += 0.20
        if len(evidence) >= 2:
            score += 0.15
        if len(evidence) >= 4:
            score += 0.05
        if source in {"verification", "verified", "seed_verified", "system", "harness_conflict_injection", "reading"}:
            score += 0.15
        if self._safe_str(details.get("source_trust")).lower() == "trusted":
            score += 0.10
        if verified:
            score += 0.15
        if last_verified is not None:
            score += 0.10
        if pending:
            score -= 0.05
        if hypothesis:
            score -= 0.10
        if contested:
            score -= 0.20

        sr = self._safe_str(details.get("staleness_risk") or "medium").lower()
        if sr == "high":
            score -= 0.10
        elif sr == "low":
            score += 0.05

        if self._is_survival_memory(record):
            score += 0.06

        return max(0.0, min(1.0, float(score)))

    def _ensure_truth_fields(self, record: MemoryRecord) -> None:
        details = self._details(record)

        details.setdefault("superseded", False)
        details.setdefault("recall_blocked", False)
        details.setdefault("source_trust", "normal")
        details.setdefault("conflict_with", [])
        details.setdefault("conflict_with_ids", [])
        details.setdefault("conflict_with_text", [])
        details.setdefault("conflict_reason", None)
        details.setdefault("superseded_by", None)
        details.setdefault("times_retrieved", 0)
        details.setdefault("times_helpful", 0)
        details.setdefault("times_corrected", 0)
        details.setdefault("hypothesis", False)
        details.setdefault("is_revision", bool(record.metadata.get("is_revision", False)))

        self._ensure_pair_fields(record)

        source = self._source_string(record)
        source_trust = self._safe_str(details.get("source_trust")).lower()

        incoming_verified = bool(details.get("verified", False))
        incoming_pending = bool(details.get("pending_verification", False))
        incoming_contested = bool(details.get("contested", False))
        incoming_conflict = bool(details.get("conflict", False))
        incoming_status = self._safe_str(
            details.get("verification_status") or details.get("status")
        ).lower()

        trusted_source = (
            source in {"seed_verified", "verified", "system", "verification", "reading"}
            or source_trust == "trusted"
        )

        if incoming_contested or incoming_conflict or incoming_status in {"contested", "disputed"}:
            record.verification_status = VerificationStatus.DISPUTED
        elif incoming_verified or incoming_status == "verified" or trusted_source:
            record.verification_status = VerificationStatus.VERIFIED
        elif bool(details.get("hypothesis", False)) or incoming_status == "hypothesis":
            record.verification_status = VerificationStatus.PROVISIONAL
        elif incoming_pending or incoming_status in {"pending", "provisional"}:
            record.verification_status = VerificationStatus.PROVISIONAL
        elif incoming_status in {"deprecated", "rejected"}:
            record.verification_status = VerificationStatus.REJECTED
        else:
            record.verification_status = VerificationStatus.UNVERIFIED

        details["verified"] = record.verification_status == VerificationStatus.VERIFIED
        details["pending_verification"] = record.verification_status == VerificationStatus.PROVISIONAL
        details["contested"] = record.verification_status == VerificationStatus.DISPUTED
        details["conflict"] = bool(details.get("conflict", False) or details["contested"])

        if trusted_source:
            details["source_trust"] = "trusted"

        if self._is_revision_entry(record):
            self._tag_revision_record(record)

        if not details.get("hazard_family"):
            fam = self._hazard_family_for_record(record)
            if fam:
                details["hazard_family"] = fam

        details["staleness_risk"] = details.get("staleness_risk") or self._infer_staleness_risk(record)
        details["memory_class"] = details.get("memory_class") or self._infer_memory_class(record)

        if details["contested"]:
            details["recall_blocked"] = True
        if bool(details.get("superseded", False)):
            details["recall_blocked"] = True

        if details["verified"] and not details.get("last_verified"):
            details["last_verified"] = float(record.metadata.get("timestamp", self._now_ts()))

        details["evidence_strength"] = self._evidence_strength(record)

        record.metadata["evidence_strength"] = details["evidence_strength"]
        record.metadata["quality"] = record.metadata.get(
            "quality",
            self.memory_quality_score(record.text),
        )

    # -------------------------------------------------------------------------
    # Legacy dict bridge
    # -------------------------------------------------------------------------
    def _record_from_legacy_dict(self, msg: Optional[Dict[str, Any]]) -> Optional[MemoryRecord]:
        if not isinstance(msg, dict):
            return None

        content_any = msg.get("content", msg.get("claim", ""))
        text = self._stringify_content(content_any).strip()
        summary = self._safe_str(msg.get("summary")) or None
        if not text and summary:
            text = summary

        details = dict(msg.get("details", {})) if isinstance(msg.get("details"), dict) else {}

        for key in (
            "last_verified",
            "memory_class",
            "staleness_risk",
            "source_trust",
            "verified",
            "pending_verification",
            "contested",
            "conflict",
            "status",
            "verification_status",
            "conflict_reason",
            "applied_via",
            "hazard_family",
            "conflict_pair_id",
            "conflict_role",
            "paired_memory_id",
            "pair_status",
            "canonical_text",
            "topic_signature",
            "numeric_signature",
            "is_revision",
        ):
            if key in msg and key not in details:
                details[key] = msg.get(key)

        source = self._safe_str(msg.get("source") or details.get("source") or "unknown").lower()
        confidence = (
            self._clamp01(
                msg.get("confidence"),
                default=self._default_confidence_for_source(source),
            )
            if msg.get("confidence") is not None
            else self._default_confidence_for_source(source)
        )

        raw_tags = self._normalize_tags_list(msg.get("tags"))
        if source and source != "unknown" and source not in raw_tags:
            if source == "harness_conflict_injection":
                raw_tags.append(source)

        if (
            bool(msg.get("is_revision"))
            or bool(details.get("is_revision"))
            or self._safe_str(msg.get("type")).lower() == "revision"
            or "revision" in raw_tags
            or "revised" in (text or "").lower()
        ):
            if "revision" not in raw_tags:
                raw_tags.append("revision")
            details["is_revision"] = True

        domain = self._safe_domain(msg.get("domain"))
        hazard_family = self._safe_domain(msg.get("hazard_family") or details.get("hazard_family"))

        signals = MemorySignals(
            domain=domain,
            hazard_family=hazard_family,
            metadata={},
        )

        evidence_items: List[EvidencePacket] = []
        raw_evidence = msg.get("evidence", details.get("evidence", []))
        if isinstance(raw_evidence, dict):
            raw_evidence = raw_evidence.get("items", [])
        if isinstance(raw_evidence, (list, tuple, set)):
            for ev in raw_evidence:
                if isinstance(ev, dict):
                    ev_text = self._safe_str(
                        ev.get("snippet") or ev.get("text") or ev.get("content")
                    )
                else:
                    ev_text = self._safe_str(ev)
                if ev_text:
                    evidence_items.append(
                        EvidencePacket(
                            source_type=self._source_type_from_string(source),
                            source_ref=None,
                            snippet=ev_text,
                            stance="support",
                            confidence=min(1.0, confidence),
                        )
                    )

        rec_id = self._make_memory_id(msg.get("id"), msg.get("memory_id"))

        record = MemoryRecord(
            memory_id=rec_id,
            text=text,
            summary=summary,
            kind=self._kind_from_legacy(msg),
            tier=MemoryTier.WORKING,
            source_type=self._source_type_from_string(source),
            source_ref=self._safe_str(msg.get("source_ref")) or None,
            confidence=confidence,
            stability=self._clamp01(msg.get("stability"), default=0.0),
            decay_score=self._clamp01(msg.get("decay_score"), default=0.0),
            priority=self._clamp01(msg.get("priority"), default=0.5),
            verification_status=self._verification_from_legacy(msg),
            access_count=max(0, int(msg.get("access_count", 0) or 0)),
            contradiction_count=max(0, int(msg.get("contradiction_count", 0) or 0)),
            retrieval_hits=max(0, int(msg.get("retrieval_hits", 0) or 0)),
            retrieval_misses=max(0, int(msg.get("retrieval_misses", 0) or 0)),
            tags=raw_tags,
            related_ids=[
                self._normalize_memory_id_candidate(x)
                for x in list(msg.get("related_ids", []) or [])
                if self._normalize_memory_id_candidate(x)
            ],
            evidence=evidence_items,
            signals=signals,
            metadata={
                "type": self._safe_str(msg.get("type") or "general").lower() or "general",
                "domain": domain,
                "hazard_family": hazard_family,
                "weight": float(msg.get("weight", 1.0) or 1.0),
                "timestamp": float(msg.get("timestamp", self._now_ts())),
                "context": list(msg.get("context", []) or []),
                "persisted": bool(msg.get("persisted", False)),
                "reinforcement_count": int(msg.get("reinforcement_count", 0) or 0),
                "last_reinforced": float(msg.get("last_reinforced", 0.0) or 0.0),
                "numeric_guarded": bool(msg.get("numeric_guarded", False)),
                "revision_trace": list(msg.get("revision_trace", []) or []),
                "source": source,
                "details": details,
                "content_obj": msg.get("content_obj") if isinstance(msg.get("content_obj"), dict) else {},
                "is_revision": bool(details.get("is_revision", False)),
            },
        )

        record.memory_id = rec_id
        self._record_quality(record)
        self._ensure_truth_fields(record)
        self._ensure_pair_fields(record)
        if self._is_revision_entry(record):
            self._tag_revision_record(record)
        return record

    def _record_to_legacy_dict(self, record: MemoryRecord) -> Dict[str, Any]:
        details = self._details(record)
        source = self._source_string(record)
        evidence = [ev.snippet for ev in record.evidence if ev.snippet]

        status_map = {
            VerificationStatus.UNVERIFIED: "unverified",
            VerificationStatus.PROVISIONAL: "pending",
            VerificationStatus.VERIFIED: "verified",
            VerificationStatus.DISPUTED: "contested",
            VerificationStatus.REJECTED: "deprecated",
        }
        status = status_map.get(record.verification_status, "unverified")

        return {
            "id": record.memory_id,
            "memory_id": record.memory_id,
            "type": self._safe_str(record.metadata.get("type")) or "general",
            "content": record.text,
            "claim": record.text,
            "content_obj": record.metadata.get("content_obj", {}),
            "confidence": record.confidence,
            "weight": float(record.metadata.get("weight", 1.0) or 1.0),
            "timestamp": float(record.metadata.get("timestamp", self._now_ts())),
            "age": self._record_age_minutes(record),
            "context": list(record.metadata.get("context", []) or []),
            "keywords": set(self._record_token_set(record)),
            "revision_trace": list(record.metadata.get("revision_trace", []) or []),
            "reinforcement_count": int(record.metadata.get("reinforcement_count", 0) or 0),
            "last_reinforced": float(record.metadata.get("last_reinforced", 0.0) or 0.0),
            "persisted": bool(record.metadata.get("persisted", False)),
            "numeric_guarded": bool(record.metadata.get("numeric_guarded", False)),
            "source": source,
            "details": details,
            "domain": self._safe_domain(record.metadata.get("domain")),
            "tags": list(record.tags),
            "kind": record.kind.value,
            "status": status,
            "verification_status": status,
            "evidence": evidence,
            "last_verified": details.get("last_verified"),
            "conflict": bool(
                record.verification_status == VerificationStatus.DISPUTED
                or details.get("conflict", False)
            ),
            "contested": bool(details.get("contested", False)),
            "verified": bool(details.get("verified", False)),
            "pending_verification": bool(details.get("pending_verification", False)),
            "memory_class": details.get("memory_class"),
            "staleness_risk": details.get("staleness_risk"),
            "source_trust": details.get("source_trust"),
            "evidence_strength": self._evidence_strength(record),
            "quality": float(record.metadata.get("quality", 0.0) or 0.0),
            "hazard_family": self._safe_domain(
                record.metadata.get("hazard_family") or details.get("hazard_family")
            ),
            "is_revision": bool(record.metadata.get("is_revision") or details.get("is_revision")),
        }

    # -------------------------------------------------------------------------
    # Public keyword extraction
    # -------------------------------------------------------------------------
    def extract_keywords(self, text: str) -> Set[str]:
        return self._tokens_from_text(text or "")

    # -------------------------------------------------------------------------
    # LTM preload bridge
    # -------------------------------------------------------------------------
    def _load_long_term(self) -> None:
        memories = self.long_term.retrieve() if self.long_term else []
        loaded = 0

        for raw in memories or []:
            record = self._record_from_legacy_dict(raw)
            if record is None:
                continue

            record.memory_id = self._make_memory_id(raw.get("id"), raw.get("memory_id"))
            record.set_tier(MemoryTier.WORKING)
            record.metadata["persisted"] = True
            record.metadata["reinforcement_count"] = max(
                int(record.metadata.get("reinforcement_count", 0) or 0),
                999,
            )
            self._insert(record, replace_on_id=True)
            loaded += 1

        if getattr(config, "VERBOSE", False):
            print(f"[WorkingMemory] Preloaded {loaded} memories from LTM.")

    # -------------------------------------------------------------------------
    # Public lookup helpers
    # -------------------------------------------------------------------------
    def get_memory_by_id(self, mem_id: str) -> Optional[Dict[str, Any]]:
        target = self._normalize_memory_id_candidate(mem_id)
        if not target:
            return None

        for record in self.buffer:
            rid = self._normalize_memory_id_candidate(record.memory_id)
            if rid == target:
                return self._record_to_legacy_dict(record)
        return None

    def get_memories_by_ids(self, mem_ids: Iterable[str]) -> List[Dict[str, Any]]:
        wanted = {
            self._normalize_memory_id_candidate(x)
            for x in (mem_ids or [])
            if self._normalize_memory_id_candidate(x)
        }
        if not wanted:
            return []

        out: List[Dict[str, Any]] = []
        for record in self.buffer:
            rid = self._normalize_memory_id_candidate(record.memory_id)
            if rid in wanted:
                out.append(self._record_to_legacy_dict(record))
        return out

    def get_conflict_pair(self, pair_id: str) -> List[Dict[str, Any]]:
        target = self._safe_str(pair_id)
        if not target:
            return []

        out: List[Dict[str, Any]] = []
        for record in self.buffer:
            details = self._details(record)
            if self._safe_str(details.get("conflict_pair_id")) == target:
                out.append(self._record_to_legacy_dict(record))

        out.sort(
            key=lambda r: (
                str(r.get("details", {}).get("conflict_role") or ""),
                str(r.get("id") or ""),
            )
        )
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

        roles = {self._safe_str(r.get("details", {}).get("conflict_role")) for r in rows}
        ids = {self._safe_str(r.get("id")) for r in rows if self._safe_str(r.get("id"))}

        pair_status_ok = all(
            self._safe_str(r.get("details", {}).get("pair_status")) == "complete"
            for r in rows
        )
        text_ok = all(
            self._safe_str(r.get("details", {}).get("canonical_text"))
            for r in rows
        )

        cross_link_ok = True
        for r in rows:
            paired = self._safe_str(r.get("details", {}).get("paired_memory_id"))
            if paired and paired not in ids:
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

    def missing_memory_ids(self, mem_ids: Iterable[str]) -> List[str]:
        wanted = [
            self._normalize_memory_id_candidate(x)
            for x in (mem_ids or [])
            if self._normalize_memory_id_candidate(x)
        ]
        found: Set[str] = set()

        for record in self.buffer:
            rid = self._normalize_memory_id_candidate(record.memory_id)
            if rid:
                found.add(rid)

        return [mid for mid in wanted if mid and mid not in found]

    def mark_memory_helpful(self, mem_id: str, amount: int = 1) -> bool:
        target = self._normalize_memory_id_candidate(mem_id)
        if not target:
            return False
        for record in self.buffer:
            if record.memory_id == target:
                details = self._details(record)
                details["times_helpful"] = int(details.get("times_helpful", 0)) + max(1, int(amount))
                return True
        return False

    def mark_memory_corrected(self, mem_id: str, amount: int = 1) -> bool:
        target = self._normalize_memory_id_candidate(mem_id)
        if not target:
            return False
        for record in self.buffer:
            if record.memory_id == target:
                details = self._details(record)
                details["times_corrected"] = int(details.get("times_corrected", 0)) + max(1, int(amount))
                return True
        return False

    # -------------------------------------------------------------------------
    # Storage / indexing
    # -------------------------------------------------------------------------
    def _rebuild_type_index(self) -> None:
        self.type_index.clear()
        for record in self.buffer:
            mtype = self._safe_str(record.metadata.get("type")) or "general"
            self.type_index.setdefault(mtype, []).append(record)

    def _insert(self, record: MemoryRecord, *, replace_on_id: bool = True) -> None:
        self._ensure_truth_fields(record)
        self._ensure_pair_fields(record)

        if not self._normalize_memory_id_candidate(record.memory_id):
            record.memory_id = self._make_memory_id()

        if replace_on_id and record.memory_id:
            for i, existing in enumerate(self.buffer):
                if existing.memory_id == record.memory_id:
                    self.buffer[i] = record
                    self._rebuild_type_index()
                    self.context_buffer.append(record)
                    if len(self.context_buffer) > self.max_context:
                        self.context_buffer.pop(0)
                    return

        self.buffer.append(record)
        mtype = self._safe_str(record.metadata.get("type")) or "general"
        self.type_index.setdefault(mtype, []).append(record)

        self.context_buffer.append(record)
        if len(self.context_buffer) > self.max_context:
            self.context_buffer.pop(0)

    # -------------------------------------------------------------------------
    # Numeric / duplicate / conflict helpers
    # -------------------------------------------------------------------------

    def _numeric_claim_signature(self, text: str) -> Optional[Tuple[str, str]]:
        low = (text or "").lower().strip()
        nums = self._NUM_PAT.findall(low)
        if not nums:
            return None

        if "water boil" in low or "boiling point of water" in low:
            return ("water boiling", nums[0])
        if "bone" in low and "human" in low:
            return ("human bones", nums[0])
        if "everest" in low:
            return ("everest height", nums[0])

        for key in self.NUMERIC_GUARDRAILS:
            if key in low:
                return (key, nums[0])

        toks = sorted(self._tokens_from_text(low))
        if not toks:
            return None
        topic = " ".join(toks[:6])
        return (topic, nums[0])

    def _numeric_material_difference(self, a: str, b: str) -> bool:
        sig_a = self._numeric_claim_signature(a)
        sig_b = self._numeric_claim_signature(b)
        if not sig_a or not sig_b:
            return False
        return sig_a[0] == sig_b[0] and sig_a[1] != sig_b[1]

    def _near_duplicate(self, a: str, b: str) -> bool:
        a2 = (a or "").strip().lower()
        b2 = (b or "").strip().lower()
        if not a2 or not b2:
            return False
        if a2 == b2:
            return True

        if self._numeric_material_difference(a2, b2):
            return False

        a_can = self._canonicalize_text(a2)
        b_can = self._canonicalize_text(b2)
        if a_can == b_can:
            return True

        ta = self._tokens_from_text(a2)
        tb = self._tokens_from_text(b2)
        if not ta or not tb:
            return False

        jac = len(ta & tb) / max(1, len(ta | tb))
        sim = SequenceMatcher(None, a_can, b_can).ratio()

        if jac >= 0.93 and sim >= float(self.DUPLICATE_SIM_THRESHOLD):
            return True
        if sim >= 0.97:
            return True
        return False

    def _is_material_conflict(self, old: MemoryRecord, new_record: MemoryRecord) -> bool:
        old_text = (old.text or "").strip()
        new_text = (new_record.text or "").strip()
        if not old_text or not new_text:
            return False

        old_kw = self._record_token_set(old)
        new_kw = self._record_token_set(new_record)
        overlap = len(old_kw & new_kw)
        if overlap < int(self.CONFLICT_OVERLAP_MIN):
            return False

        old_d = self._details(old)
        new_d = self._details(new_record)

        old_topic = self._safe_str(old_d.get("topic_signature"))
        new_topic = self._safe_str(new_d.get("topic_signature"))
        old_num = self._safe_str(old_d.get("numeric_signature"))
        new_num = self._safe_str(new_d.get("numeric_signature"))

        if old_topic and new_topic and old_topic == new_topic and old_num and new_num and old_num != new_num:
            new_d["conflict_reason"] = f"numeric mismatch:{old_topic}"
            return True

        old_sig = self._numeric_claim_signature(old_text)
        new_sig = self._numeric_claim_signature(new_text)
        if old_sig and new_sig and old_sig[0] == new_sig[0] and old_sig[1] != new_sig[1]:
            new_d["conflict_reason"] = f"numeric mismatch:{old_sig[0]}"
            return True

        old_neg = bool(self._NEGATION_PAT.search(old_text))
        new_neg = bool(self._NEGATION_PAT.search(new_text))
        if old_neg != new_neg and overlap >= int(self.CONFLICT_OVERLAP_MIN):
            new_d["conflict_reason"] = "negation mismatch"
            return True

        return False

    def _apply_conflict_penalty(self, record: MemoryRecord, drop: Optional[float] = None) -> None:
        details = self._details(record)
        amt = float(self.CONFLICT_CONFIDENCE_DROP if drop is None else drop)
        record.confidence = max(0.0, record.confidence - amt)
        record.contradiction_count += 1
        record.set_verification_status(VerificationStatus.DISPUTED)
        details["conflict"] = True
        details["contested"] = True
        details["status"] = "contested"
        details["verification_status"] = "contested"
        details["pending_verification"] = True
        details["recall_blocked"] = True

    def _mark_conflict_pair(self, old: MemoryRecord, new_record: MemoryRecord) -> None:
        old_d = self._details(old)
        new_d = self._details(new_record)

        self._ensure_pair_fields(old)
        self._ensure_pair_fields(new_record)

        pair_id = self._safe_str(old_d.get("conflict_pair_id")) or self._safe_str(new_d.get("conflict_pair_id"))
        if not pair_id:
            pair_id = self._make_conflict_pair_id()

        old_d["conflict"] = True
        old_d["contested"] = True
        old_d["recall_blocked"] = True
        old_d["verification_status"] = "contested"
        old_d["status"] = "contested"

        new_d["conflict"] = True
        new_d["contested"] = True
        new_d["recall_blocked"] = True
        new_d["pending_verification"] = True
        new_d["verification_status"] = "contested"
        new_d["status"] = "contested"

        reason = (
            self._safe_str(new_d.get("conflict_reason"))
            or self._safe_str(old_d.get("conflict_reason"))
            or "material conflict"
        )
        old_d["conflict_reason"] = reason
        new_d["conflict_reason"] = reason

        old_d["conflict_pair_id"] = pair_id
        new_d["conflict_pair_id"] = pair_id

        old_d["conflict_role"] = "a"
        new_d["conflict_role"] = "b"

        old_d["paired_memory_id"] = new_record.memory_id
        new_d["paired_memory_id"] = old.memory_id

        old_d["pair_status"] = "complete"
        new_d["pair_status"] = "complete"

        self._apply_conflict_penalty(old)
        self._apply_conflict_penalty(new_record)

        old_ref = old.text[:120]
        new_ref = new_record.text[:120]

        old_d.setdefault("conflict_with", [])
        new_d.setdefault("conflict_with", [])
        old_d.setdefault("conflict_with_ids", [])
        new_d.setdefault("conflict_with_ids", [])
        old_d.setdefault("conflict_with_text", [])
        new_d.setdefault("conflict_with_text", [])

        if new_ref and new_ref not in old_d["conflict_with"]:
            old_d["conflict_with"].append(new_ref)
        if old_ref and old_ref not in new_d["conflict_with"]:
            new_d["conflict_with"].append(old_ref)

        if new_record.memory_id and new_record.memory_id not in old_d["conflict_with_ids"]:
            old_d["conflict_with_ids"].append(new_record.memory_id)
        if old.memory_id and old.memory_id not in new_d["conflict_with_ids"]:
            new_d["conflict_with_ids"].append(old.memory_id)

        if new_ref and new_ref not in old_d["conflict_with_text"]:
            old_d["conflict_with_text"].append(new_ref)
        if old_ref and old_ref not in new_d["conflict_with_text"]:
            new_d["conflict_with_text"].append(old_ref)

        old.metadata.setdefault("revision_trace", []).append(
            f"Marked contested in conflict_pair_id={pair_id} against memory id={new_record.memory_id or 'unknown'} reason={reason}: '{new_ref}'"
        )
        new_record.metadata.setdefault("revision_trace", []).append(
            f"Marked contested in conflict_pair_id={pair_id} against memory id={old.memory_id or 'unknown'} reason={reason}: '{old_ref}'"
        )

    def _reinforce(self, record: MemoryRecord, incoming: Optional[MemoryRecord] = None) -> None:
        details = self._details(record)
        now = self._now_ts()

        record.metadata["weight"] = float(record.metadata.get("weight", 0.0) or 0.0) + 0.10
        record.metadata["reinforcement_count"] = int(
            record.metadata.get("reinforcement_count", 0) or 0
        ) + 1
        record.metadata["last_reinforced"] = now

        source = self._source_string(record)
        verified = record.verification_status == VerificationStatus.VERIFIED
        pending = record.verification_status == VerificationStatus.PROVISIONAL
        contested = record.verification_status == VerificationStatus.DISPUTED

        if contested or details.get("superseded", False):
            conf_boost = 0.0
        elif verified or source in {"seed_verified", "verified", "system", "verification", "reading"}:
            conf_boost = 0.05
        elif self._safe_str(record.metadata.get("type")).lower() == "identity":
            conf_boost = 0.03
        elif pending:
            conf_boost = 0.005
        else:
            conf_boost = 0.01

        record.confidence = min(float(self.DUPLICATE_REINFORCE_CAP), record.confidence + conf_boost)
        details["confidence"] = record.confidence
        record.metadata.setdefault("revision_trace", []).append(
            f"Reinforced (count={record.metadata['reinforcement_count']}, conf={record.confidence:.2f}, boost={conf_boost:.3f})"
        )

        if self._eligible_for_promotion(record) and self._numeric_guardrail_pass(record):
            if self._promotions_this_cycle < self.MAX_PROMOTIONS_PER_CYCLE:
                self._promote_to_ltm(record)
                self._promotions_this_cycle += 1

    def _handle_conflicts(self, new_record: MemoryRecord, *, force_insert: bool = False) -> Union[bool, str]:
        if self._is_committed_action(new_record):
            return False

        new_text = (new_record.text or "").strip().lower()
        if not new_text:
            return False

        self._ensure_pair_fields(new_record)

        harness_mode = force_insert or self._is_harness_injection(new_record)
        new_domain = self._safe_domain(new_record.metadata.get("domain"))
        new_kind = new_record.kind
        new_kw = self._record_token_set(new_record)

        for old in self.buffer[-200:]:
            if self._is_committed_action(old):
                continue

            old_text = (old.text or "").strip().lower()
            if not old_text:
                continue

            self._ensure_pair_fields(old)

            if self._same_conflict_pair(old, new_record):
                continue

            old_domain = self._safe_domain(old.metadata.get("domain"))
            if old_domain and new_domain and old_domain != new_domain:
                continue

            if (
                old.kind != new_kind
                and old.kind != MemoryKind.UNRESOLVED
                and new_kind != MemoryKind.UNRESOLVED
            ):
                continue

            old_kw = self._record_token_set(old)
            overlap = len(old_kw & new_kw)
            if overlap <= 0 and old_text != new_text:
                continue

            if self._is_material_conflict(old, new_record):
                self._mark_conflict_pair(old, new_record)
                return "insert"

            if old_text == new_text:
                if self._is_conflict_linked(old) or self._is_conflict_linked(new_record):
                    continue
                if harness_mode and old.memory_id != new_record.memory_id:
                    continue
                self._reinforce(old, incoming=new_record)
                old.metadata.setdefault("revision_trace", []).append("Reinforced via repeated exposure")
                return "absorbed"

            if self._is_conflict_linked(old) or self._is_conflict_linked(new_record):
                continue

            if self._near_duplicate(old_text, new_text):
                if harness_mode and old.memory_id != new_record.memory_id:
                    continue

                old_strength = float(old.confidence) + float(old.metadata.get("weight", 0.0) or 0.0)
                new_strength = float(new_record.confidence) + float(
                    new_record.metadata.get("weight", 0.0) or 0.0
                )

                if old_strength >= new_strength:
                    self._reinforce(old, incoming=new_record)
                    old.metadata.setdefault("revision_trace", []).append("Absorbed near-duplicate memory")
                    return "absorbed"

                old_d = self._details(old)
                old_d["superseded"] = True
                old_d["recall_blocked"] = True
                old_d["superseded_by"] = new_record.text[:120]
                old.metadata.setdefault("revision_trace", []).append(
                    f"Superseded by stronger near-duplicate: '{new_record.text}'"
                )
                return "insert"

        return False

    # -------------------------------------------------------------------------
    # Store
    # -------------------------------------------------------------------------
    def _prepare_record_for_store(self, record: MemoryRecord) -> None:
        if self._is_revision_entry(record):
            self._tag_revision_record(record)

        mtype = self._safe_str(record.metadata.get("type")).lower()
        if (
            mtype in {
                "lesson", "fact", "chapter_summary", "section_summary",
                "summary", "policy", "identity", "revision"
            }
            or self._is_survival_memory(record)
            or self._is_revision_entry(record)
        ):
            record.metadata["reinforcement_count"] = max(
                1,
                int(record.metadata.get("reinforcement_count", 0) or 0),
            )
            record.metadata["last_reinforced"] = float(
                record.metadata.get("last_reinforced", self._now_ts()) or self._now_ts()
            )

    def _store_time_promotion_pass(self, inserted_records: List[MemoryRecord]) -> None:
        if not self.long_term or not inserted_records:
            return

        for record in inserted_records[-24:]:
            if self._promotions_this_cycle >= self.MAX_PROMOTIONS_PER_CYCLE:
                break
            if self._eligible_for_promotion(record) and self._numeric_guardrail_pass(record):
                self._promote_to_ltm(record)
                self._promotions_this_cycle += 1

    def _store_impl(self, messages: Any, *, force_insert: bool = False) -> List[str]:
        stored_ids: List[str] = []
        inserted_records: List[MemoryRecord] = []
        self._promotions_this_cycle = 0

        if not messages:
            return stored_ids

        if isinstance(messages, (MemoryRecord, dict)) or is_dataclass(messages):
            messages = [messages]
        elif not isinstance(messages, (list, tuple)):
            return stored_ids

        for item in messages:
            record = self._coerce_incoming_record(item)
            if record is None:
                continue

            self._ensure_pair_fields(record)
            self._prepare_record_for_store(record)

            if self.is_trash_memory(record) and not bool(record.metadata.get("persisted", False)):
                if not self._is_harness_injection(record):
                    continue

            explicit_id = self._normalize_memory_id_candidate(record.memory_id)

            if explicit_id:
                existing = next((r for r in self.buffer if r.memory_id == explicit_id), None)
                if existing is not None:
                    if existing.text == record.text:
                        self._reinforce(existing, incoming=record)
                        stored_ids.append(explicit_id)
                        continue
                    if force_insert:
                        record.memory_id = self._make_memory_id()
                        explicit_id = self._normalize_memory_id_candidate(record.memory_id)

            outcome = self._handle_conflicts(record, force_insert=force_insert)
            if outcome == "absorbed":
                continue

            self._insert(record, replace_on_id=not force_insert)
            inserted_records.append(record)

            if explicit_id:
                stored_ids.append(explicit_id)

            if record.kind in {MemoryKind.EPISODE, MemoryKind.SCENARIO, MemoryKind.REFLECTION}:
                self.episodic.store(replace(record))

        self._store_time_promotion_pass(inserted_records)
        self._prune()
        self._promotions_this_cycle = 0
        return stored_ids

    def store(
        self,
        messages: Union[
            Dict[str, Any],
            List[Dict[str, Any]],
            Tuple[Dict[str, Any], ...],
            MemoryRecord,
            List[MemoryRecord],
            Tuple[MemoryRecord, ...],
        ],
    ) -> List[str]:
        return self._store_impl(messages, force_insert=False)

    def store_fallback(
        self,
        messages: Union[
            Dict[str, Any],
            List[Dict[str, Any]],
            Tuple[Dict[str, Any], ...],
            MemoryRecord,
            List[MemoryRecord],
            Tuple[MemoryRecord, ...],
        ],
    ) -> List[str]:
        return self._store_impl(messages, force_insert=True)

    # -------------------------------------------------------------------------
    # Promotion / guardrails
    # -------------------------------------------------------------------------
    def _promotion_quality(self, record: MemoryRecord) -> float:
        return float(record.metadata.get("quality", self.memory_quality_score(record.text)) or 0.0)

    def _promotion_evidence(self, record: MemoryRecord) -> float:
        return float(record.metadata.get("evidence_strength", self._evidence_strength(record)) or 0.0)

    def _fast_promotion_allowed(self, record: MemoryRecord) -> bool:
        if bool(record.metadata.get("persisted", False)):
            return False
        if record.verification_status == VerificationStatus.DISPUTED:
            return False
        if bool(self._details(record).get("superseded", False)):
            return False

        mtype = self._safe_str(record.metadata.get("type")).lower()
        trusted = self._trusted_source(record)
        quality = self._promotion_quality(record)
        evidence_strength = self._promotion_evidence(record)
        useful_type = mtype in {
            "identity", "policy", "chapter_summary", "section_summary", "summary",
            "fact", "lesson", "committed_action", "revision",
        }
        survival_useful = self._is_survival_memory(record) and self._is_action_guidance_text(record.text or "")
        revision_useful = self._is_revision_entry(record)

        if mtype == "committed_action":
            return quality >= 0.35
        if survival_useful:
            return quality >= 0.40
        if revision_useful and (trusted or evidence_strength >= float(self.FAST_PROMOTION_EVIDENCE_FLOOR)):
            return quality >= float(self.FAST_PROMOTION_QUALITY_FLOOR)
        if useful_type and trusted:
            return quality >= float(self.FAST_PROMOTION_QUALITY_FLOOR)
        if useful_type and evidence_strength >= float(self.FAST_PROMOTION_EVIDENCE_FLOOR):
            return quality >= float(self.FAST_PROMOTION_QUALITY_FLOOR)
        return False

    def _eligible_for_promotion(self, record: MemoryRecord) -> bool:
        age_minutes = self._record_age_minutes(record)
        details = self._details(record)
        mtype = self._safe_str(record.metadata.get("type")).lower()
        reinforce_count = int(record.metadata.get("reinforcement_count", 0) or 0)
        evidence_strength = self._promotion_evidence(record)

        if bool(record.metadata.get("persisted", False)):
            return False
        if record.verification_status == VerificationStatus.DISPUTED:
            return False
        if bool(details.get("superseded", False)):
            return False
        if bool(details.get("recall_blocked", False)) and not self._is_survival_memory(record):
            return False

        if mtype == "committed_action":
            return (
                float(record.confidence) >= 0.70
                and reinforce_count >= 1
                and age_minutes >= 0.0
            )

        safe_type = mtype in {
            "identity", "policy", "lesson",
            "chapter_summary", "section_summary", "summary", "revision",
        }

        if self._fast_promotion_allowed(record):
            return (
                float(record.confidence) >= self.FAST_PROMOTION_CONFIDENCE
                and reinforce_count >= self.FAST_PROMOTION_REINFORCEMENTS
                and float(age_minutes) >= self.FAST_PROMOTION_MIN_AGE_MINUTES
            )

        if self.PROMOTION_REQUIRES_VERIFICATION and not safe_type:
            if not self._trusted_source(record) and evidence_strength < float(self.EVIDENCE_PROMOTION_FLOOR):
                return False

        if not safe_type and evidence_strength < float(self.EVIDENCE_PROMOTION_FLOOR):
            return False

        return (
            float(record.confidence) >= self.PROMOTION_CONFIDENCE
            and reinforce_count >= self.PROMOTION_REINFORCEMENTS
            and float(age_minutes) >= self.PROMOTION_MIN_AGE_MINUTES
        )

    def _numeric_guardrail_pass(self, record: MemoryRecord) -> bool:
        if self._is_committed_action(record):
            return True

        content = (record.text or "").lower()
        numbers = list(map(int, re.findall(r"\d+", content)))
        for key, canonical in self.NUMERIC_GUARDRAILS.items():
            if key in content:
                if not numbers or numbers[0] != canonical:
                    record.metadata.setdefault("revision_trace", []).append(
                        f"Blocked promotion: numeric mismatch for '{key}' (found {numbers}, expected {canonical})"
                    )
                    return False
        return True

    def _promotion_status_for_record(self, record: MemoryRecord) -> str:
        if self._trusted_source(record) or self._is_committed_action(record):
            return "verified"
        if record.verification_status == VerificationStatus.DISPUTED:
            return "contested"
        if bool(self._details(record).get("hypothesis", False)):
            return "hypothesis"
        if self._is_survival_memory(record):
            return "provisional"
        if self._is_revision_entry(record):
            return "provisional" if not self._trusted_source(record) else "verified"
        if self._safe_str(record.metadata.get("type")).lower() in {
            "lesson", "chapter_summary", "section_summary", "summary",
        }:
            return "provisional"
        if record.verification_status == VerificationStatus.PROVISIONAL:
            return "unverified"
        return record.verification_status.value

    def _promote_to_ltm(self, record: MemoryRecord) -> None:
        if not self.long_term:
            return

        if bool(record.metadata.get("persisted", False)):
            return

        if self.is_trash_memory(record):
            details = self._details(record)
            details["exclude_from_planning"] = True
            record.metadata.setdefault("revision_trace", []).append(
                "Blocked promotion: failed WM trash gate"
            )
            return

        payload = [{
            "id": record.memory_id,
            "memory_id": record.memory_id,
            "type": self._safe_str(record.metadata.get("type")) or "general",
            "content": record.text,
            "claim": record.text,
            "content_obj": record.metadata.get("content_obj", {}),
            "context": list(record.metadata.get("context", []) or []),
            "confidence": record.confidence,
            "source": self._source_string(record),
            "numeric_guarded": bool(record.metadata.get("numeric_guarded", False)),
            "revision_trace": list(record.metadata.get("revision_trace", []) or []),
            "details": self._details(record),
            "domain": self._safe_domain(record.metadata.get("domain")),
            "tags": list(record.tags),
            "kind": record.kind.value,
            "timestamp": float(record.metadata.get("timestamp", self._now_ts())),
            "status": self._promotion_status_for_record(record),
            "evidence": [ev.snippet for ev in record.evidence if ev.snippet],
            "last_verified": self._details(record).get("last_verified"),
            "conflict": record.verification_status == VerificationStatus.DISPUTED,
            "evidence_strength": self._evidence_strength(record),
            "hazard_family": self._safe_domain(
                record.metadata.get("hazard_family") or self._details(record).get("hazard_family")
            ),
            "is_revision": bool(record.metadata.get("is_revision") or self._details(record).get("is_revision")),
        }]

        _ = self.long_term.store(payload)
        record.metadata["persisted"] = True
        record.metadata.setdefault("revision_trace", []).append("Auto-promoted to long-term memory")

        if getattr(config, "VERBOSE", False):
            print(f"[WorkingMemory] Auto-promoted to LTM: '{record.text}'")

    # -------------------------------------------------------------------------
    # Retrieval scoring
    # -------------------------------------------------------------------------
    def _domain_penalty(self, rec_domain: Optional[str], ctx_domain: Optional[str]) -> float:
        if not ctx_domain or ctx_domain == "general":
            return 0.0
        if not rec_domain:
            return 1.8
        if rec_domain == ctx_domain:
            return 0.0
        if ctx_domain == "survival":
            return 5.0
        return 3.5

    def _tag_bonus(self, rec_tags: Set[str], ctx_tags: Set[str]) -> float:
        if not rec_tags or not ctx_tags:
            return 0.0
        return float(len(rec_tags & ctx_tags))

    def _weighted_overlap(self, rec_keywords: Set[str], q_keywords: Set[str]) -> Tuple[float, int]:
        if not rec_keywords or not q_keywords:
            return 0.0, 0

        rk = self._normalize_token_set(rec_keywords)
        qk = self._normalize_token_set(q_keywords)

        overlap = rk & qk
        if not overlap:
            return 0.0, 0

        score = 0.0
        strong = 0
        for token in overlap:
            if token in self.COMMON_TOKENS:
                score += 0.25
            else:
                score += 1.0
                strong += 1
        return score, strong

    def _phrase_pair_bonus(
        self,
        record: MemoryRecord,
        context_profile: Optional[Dict[str, Any]],
    ) -> float:
        if not isinstance(context_profile, dict):
            return 0.0

        query_text = self._safe_str(context_profile.get("query_text"))
        if not query_text:
            return 0.0

        q_pairs = self._phrase_pairs_from_text(query_text)
        if not q_pairs:
            return 0.0

        rec_pairs = self._record_phrase_pairs(record)
        if not rec_pairs:
            return 0.0

        overlap = q_pairs & rec_pairs
        return float(len(overlap)) * 2.4 if overlap else 0.0

    def _exact_phrase_adjustment(
        self,
        record: MemoryRecord,
        context_profile: Optional[Dict[str, Any]],
    ) -> float:
        if not isinstance(context_profile, dict):
            return 0.0

        query_text = self._safe_str(context_profile.get("query_text"))
        if not query_text:
            return 0.0

        anchors = self._extract_query_anchor_phrases(query_text)
        if not anchors:
            return 0.0

        content_blob = self._normalize_for_phrase_match(
            " ".join([record.text, record.summary or ""])
        )
        tag_blob = self._normalize_for_phrase_match(" ".join(record.tags))

        is_revision_mode = self._is_revision_query(context_profile)
        is_revision_record = self._is_revision_entry(record)

        bonus = 0.0
        misses = 0
        for phrase in anchors:
            if phrase in content_blob:
                if is_revision_mode and is_revision_record and len(phrase.split()) >= 2:
                    bonus += 6.0
                else:
                    bonus += 4.0
            elif phrase in tag_blob:
                bonus += 2.0
            else:
                misses += 1

        ctx_domain = self._safe_domain((context_profile or {}).get("domain"))
        rec_domain = self._safe_domain(record.metadata.get("domain"))
        if misses > 0 and ctx_domain and rec_domain == ctx_domain:
            bonus -= min(float(misses) * 3.0, 3.0)

        return bonus

    def _identity_adjustment(
        self,
        record: MemoryRecord,
        q_keywords: Set[str],
        context_profile: Optional[Dict[str, Any]],
    ) -> float:
        if not self._is_identity_entry(record):
            return 0.0

        query_text = " ".join(sorted(q_keywords)) if q_keywords else ""
        if isinstance(context_profile, dict):
            raw_q = context_profile.get("query_text")
            if isinstance(raw_q, str) and raw_q.strip():
                query_text = raw_q.strip().lower()

        return 0.0 if self._is_identity_query(query_text) else -6.0

    def _literature_adjustment(
        self,
        record: MemoryRecord,
        q_keywords: Set[str],
        context_profile: Optional[Dict[str, Any]],
    ) -> float:
        query_text = ""
        if isinstance(context_profile, dict):
            raw_q = context_profile.get("query_text")
            if isinstance(raw_q, str):
                query_text = raw_q.strip().lower()

        if not self._is_literature_query(query_text, context_profile):
            return 0.0

        adj = 0.0
        mtype = self._safe_str(record.metadata.get("type")).lower()
        mdomain = self._safe_domain(record.metadata.get("domain"))
        mtags = set(record.tags)
        content = (record.text or "").lower()

        if self._is_feedbackish_entry(record):
            return -8.0

        if mdomain == "literature":
            adj += 2.5
        if {"reading", "literature"} & mtags:
            adj += 2.5

        if mtype == "chapter_summary":
            adj += 7.5
        elif mtype == "section_summary":
            adj += 6.0
        elif mtype in {"concept_frame", "literary_frame", "summary"}:
            adj += 4.5
        elif mtype == "claim":
            adj += 1.5
        elif mtype in {"fact", "identity", "policy"} and mdomain != "literature":
            adj -= 3.0

        ch_hint = context_profile.get("chapter_hint") if isinstance(context_profile, dict) else None
        sec_hint = context_profile.get("section_hint") if isinstance(context_profile, dict) else None

        if ch_hint:
            if f"chapter_{ch_hint}" in mtags:
                adj += 4.0
            elif mtype == "chapter_summary":
                adj += 2.4

        if sec_hint:
            if f"section_{sec_hint}" in mtags:
                adj += 2.5
            elif mtype == "section_summary":
                adj += 1.5

        if mtype in {"chapter_summary", "section_summary", "concept_frame", "literary_frame", "claim"}:
            if not content.strip():
                adj -= 2.0

        lit_overlap = len(set(record.tags) & self._normalize_token_set(q_keywords))
        if lit_overlap > 0:
            adj += 0.5 * lit_overlap

        return adj

    def _garden_adjustment(
        self,
        record: MemoryRecord,
        q_keywords: Set[str],
        context_profile: Optional[Dict[str, Any]],
    ) -> float:
        query_text = ""
        if isinstance(context_profile, dict):
            raw_q = context_profile.get("query_text")
            if isinstance(raw_q, str):
                query_text = raw_q.strip().lower()

        if not self._is_garden_query(query_text, context_profile):
            return 0.0

        rec_domain = self._safe_domain(record.metadata.get("domain"))
        if rec_domain != "garden":
            return 0.0

        q_sub = self._garden_query_subtype(query_text)
        r_sub = self._garden_record_subtype(record)

        if not q_sub:
            return 0.0

        if q_sub == "raised_beds":
            if r_sub == "raised_beds":
                return 5.0
            if r_sub == "watering":
                return -3.5
            return -1.5

        if q_sub == "watering":
            if r_sub == "watering":
                return 4.0
            if r_sub == "raised_beds":
                return -2.0

        return 0.0

    def _survival_adjustment(
        self,
        record: MemoryRecord,
        q_keywords: Set[str],
        context_profile: Optional[Dict[str, Any]],
    ) -> float:
        ctx_domain = (
            self._safe_domain((context_profile or {}).get("domain"))
            if isinstance(context_profile, dict)
            else None
        )
        if ctx_domain != "survival":
            return 0.0

        rec_domain = self._safe_domain(record.metadata.get("domain"))
        rec_type = self._safe_str(record.metadata.get("type")).lower()
        rec_tags = set(record.tags)
        rec_fam = self._hazard_family_for_record(record)
        q_fam = (
            self._safe_domain((context_profile or {}).get("hazard_family"))
            if isinstance(context_profile, dict)
            else None
        )
        content = (record.text or "").lower()

        adj = 0.0

        if self._is_feedbackish_entry(record):
            return -10.0

        if self._is_survival_memory(record):
            adj += 5.5
        else:
            adj -= 8.0

        if rec_domain == "survival":
            adj += 4.0
        elif rec_domain and rec_domain != "survival":
            adj -= 2.5

        if "survival" in rec_tags:
            adj += 2.0

        if rec_fam and q_fam:
            if rec_fam == q_fam:
                adj += 6.0
            else:
                adj -= 7.5
        elif rec_fam and not q_fam:
            adj += 1.0

        if rec_tags & self.SURVIVAL_HINT_TAGS:
            adj += 1.5

        if self._is_action_guidance_text(content):
            adj += 4.0

        if rec_type in {"policy", "lesson", "procedure", "reasoning_action", "committed_action"}:
            adj += 2.5
        elif rec_type in {"fact", "claim"} and not self._is_action_guidance_text(content):
            adj -= 5.0

        if record.verification_status == VerificationStatus.PROVISIONAL:
            if self._survival_recall_allowed(record, context_profile):
                adj += 1.75
            else:
                adj -= 3.5

        return adj

    def _base_semantic_adjustments(
        self,
        record: MemoryRecord,
        q_keywords: Set[str],
        context_profile: Optional[Dict[str, Any]],
        requested_type: Optional[str],
    ) -> float:
        ctx_domain = None
        ctx_tags: Set[str] = set()
        if isinstance(context_profile, dict):
            ctx_domain = self._safe_domain(context_profile.get("domain"))
            ctx_tags = self._tags_set(context_profile.get("tags"))

        rec_tags = set(record.tags)
        rec_domain = self._safe_domain(record.metadata.get("domain"))

        adj = 0.0
        adj += self._tag_bonus(rec_tags, ctx_tags)
        adj -= self._domain_penalty(rec_domain, ctx_domain)
        adj += self._phrase_pair_bonus(record, context_profile)
        adj += self._exact_phrase_adjustment(record, context_profile)
        adj += self._identity_adjustment(record, q_keywords, context_profile)
        adj += self._literature_adjustment(record, q_keywords, context_profile)
        adj += self._garden_adjustment(record, q_keywords, context_profile)
        adj += self._survival_adjustment(record, q_keywords, context_profile)

        if requested_type and self._safe_str(record.metadata.get("type")).lower() == requested_type.lower():
            adj += 1.25

        details = self._details(record)
        ctx_domain = (
            self._safe_domain((context_profile or {}).get("domain"))
            if isinstance(context_profile, dict)
            else None
        )

        is_revision_mode = self._is_revision_query(context_profile)
        is_revision_record = self._is_revision_entry(record)

        if is_revision_record:
            adj += 6.0

        if is_revision_mode:
            if is_revision_record:
                adj += 8.0
            else:
                adj -= 6.0

        if is_revision_mode and ctx_domain == "clubhouse_build":
            rec_type = self._safe_str(record.metadata.get("type")).lower()
            if rec_type in {"lesson", "procedure"} and not is_revision_record:
                adj -= 2.5

        if record.verification_status == VerificationStatus.PROVISIONAL:
            if ctx_domain == "survival" and self._survival_recall_allowed(record, context_profile):
                adj -= 0.20
            else:
                adj -= 1.5
        if record.verification_status == VerificationStatus.DISPUTED:
            adj -= 3.0
        if bool(details.get("superseded", False)):
            adj -= 3.0

        return adj

    def _relevance_score(
        self,
        record: MemoryRecord,
        q_keywords: Set[str],
        context_profile: Optional[Dict[str, Any]] = None,
        requested_type: Optional[str] = None,
    ) -> Tuple[Optional[float], float, int]:
        rec_keywords = self._record_token_set(record)
        overlap_score, strong_overlap = self._weighted_overlap(rec_keywords, q_keywords)
        evidence_strength = float(
            record.metadata.get("evidence_strength", self._evidence_strength(record)) or 0.0
        )

        semantic_adj = self._base_semantic_adjustments(
            record=record,
            q_keywords=q_keywords,
            context_profile=context_profile,
            requested_type=requested_type,
        )

        if q_keywords and strong_overlap == 0 and overlap_score <= 0.0:
            score = (
                float(record.confidence)
                + float(record.metadata.get("weight", 0.0) or 0.0)
                - 3.0
            )
            score += evidence_strength * float(self.EVIDENCE_STRENGTH_WEIGHT) * 0.50
            score += semantic_adj
            return score, overlap_score, strong_overlap

        base = 0.0
        base += float(overlap_score) * 4.0
        base += float(record.confidence)
        base += float(record.metadata.get("weight", 0.0) or 0.0)
        base += evidence_strength * float(self.EVIDENCE_STRENGTH_WEIGHT)
        base += semantic_adj

        return base, overlap_score, strong_overlap

    def _score_domain_pool(
        self,
        source: List[MemoryRecord],
        qk: Set[str],
        context_profile: Optional[Dict[str, Any]],
        requested_type: Optional[str],
        min_relevance: float,
        count: int,
        planning_only: bool,
    ) -> List[Tuple[float, float, int, float, float, MemoryRecord]]:
        def recall_ok(record: MemoryRecord) -> bool:
            return not self._is_recall_blocked(record, context_profile=context_profile)

        def plan_ok(record: MemoryRecord) -> bool:
            return self._planning_allowed(record) if planning_only else recall_ok(record)

        scored: List[Tuple[float, float, int, float, float, MemoryRecord]] = []

        for record in source:
            if not plan_ok(record):
                continue

            score, overlap_score, strong_overlap = self._relevance_score(
                record,
                qk,
                context_profile=context_profile,
                requested_type=requested_type,
            )
            if score is None:
                continue

            if float(score) >= float(min_relevance):
                scored.append(
                    (
                        float(score),
                        float(overlap_score),
                        int(strong_overlap),
                        float(record.confidence),
                        float(record.metadata.get("weight", 0.0) or 0.0),
                        record,
                    )
                )

        scored.sort(key=lambda x: (x[0], x[2], x[1], x[3], x[4]), reverse=True)
        return scored[:count]

    # -------------------------------------------------------------------------
    # Retrieve
    # -------------------------------------------------------------------------
    def retrieve(
        self,
        msg_type: Optional[str] = None,
        count: int = 5,
        keywords: Optional[Iterable[str]] = None,
        context_profile: Optional[Dict[str, Any]] = None,
        min_relevance: Optional[float] = None,
        planning_only: bool = False,
    ) -> List[Dict[str, Any]]:
        if min_relevance is None:
            min_relevance = float(self.DEFAULT_MIN_RELEVANCE)

        requested_type = (
            msg_type if isinstance(msg_type, str) and msg_type in self.type_index else None
        )
        source = self.type_index.get(requested_type, self.buffer) if requested_type else self.buffer

        if isinstance(msg_type, str) and not requested_type and keywords is None:
            query_text = msg_type.strip()
            if query_text:
                keywords = self._tokens_from_text(query_text)
                inferred = self._infer_query_context(query_text)
                if not isinstance(context_profile, dict):
                    context_profile = inferred
                else:
                    merged = dict(context_profile)
                    for key, value in inferred.items():
                        if merged.get(key) in (None, [], "", {}):
                            merged[key] = value
                    context_profile = merged

        if not source:
            return []

        qk: Set[str] = set()
        if keywords is not None:
            if isinstance(keywords, str):
                qk = self._tokens_from_text(keywords)
            elif isinstance(keywords, (set, list, tuple)):
                qk = self._normalize_token_set(keywords)
            else:
                try:
                    qk = self._normalize_token_set(list(keywords))
                except Exception:
                    qk = set()

        if (not isinstance(context_profile, dict)) and qk:
            inferred_text = (
                msg_type.strip()
                if isinstance(msg_type, str) and msg_type.strip() and msg_type not in self.type_index
                else " ".join(sorted(qk))
            )
            context_profile = self._infer_query_context(inferred_text)

        is_revision_mode = self._is_revision_query(context_profile)
        if is_revision_mode:
            revision_only = [r for r in source if self._is_revision_entry(r)]
            if revision_only:
                source = revision_only

        ctx_domain = (
            self._safe_domain(context_profile.get("domain"))
            if isinstance(context_profile, dict)
            else None
        )

        if qk:
            domain_scored: List[Tuple[float, float, int, float, float, MemoryRecord]] = []
            if ctx_domain and ctx_domain != "general":
                domain_pool = [
                    record
                    for record in source
                    if self._safe_domain(record.metadata.get("domain")) == ctx_domain
                    or (ctx_domain == "survival" and self._is_survival_memory(record))
                    or (is_revision_mode and self._is_revision_entry(record))
                ]
                if domain_pool:
                    domain_scored = self._score_domain_pool(
                        source=domain_pool,
                        qk=qk,
                        context_profile=context_profile,
                        requested_type=requested_type,
                        min_relevance=min_relevance,
                        count=max(count, 8),
                        planning_only=planning_only,
                    )

            full_scored = self._score_domain_pool(
                source=source,
                qk=qk,
                context_profile=context_profile,
                requested_type=requested_type,
                min_relevance=min_relevance,
                count=max(count, 8),
                planning_only=planning_only,
            )

            chosen_scored = full_scored
            if domain_scored:
                if not full_scored:
                    chosen_scored = domain_scored
                else:
                    top_domain = float(domain_scored[0][0])
                    top_full = float(full_scored[0][0])
                    top_full_record = full_scored[0][-1]
                    top_full_domain = self._safe_domain(top_full_record.metadata.get("domain"))
                    if is_revision_mode:
                        chosen_scored = domain_scored
                    elif top_full_domain == ctx_domain or top_domain >= (top_full - 0.75):
                        chosen_scored = domain_scored

            results = [record for *_, record in chosen_scored[:count]]
            for record in results:
                record.register_retrieval_hit()
                details = self._details(record)
                details["times_retrieved"] = int(details.get("times_retrieved", 0)) + 1
            return [self._record_to_legacy_dict(record) for record in results]

        def recall_ok(record: MemoryRecord) -> bool:
            return not self._is_recall_blocked(record, context_profile=context_profile)

        def plan_ok(record: MemoryRecord) -> bool:
            return self._planning_allowed(record) if planning_only else recall_ok(record)

        if ctx_domain and ctx_domain != "general":
            same = [
                record for record in source
                if (
                    self._safe_domain(record.metadata.get("domain")) == ctx_domain
                    or (ctx_domain == "survival" and self._is_survival_memory(record))
                    or (is_revision_mode and self._is_revision_entry(record))
                )
                and plan_ok(record)
            ]
            if same:
                same.sort(
                    key=lambda record: (
                        1.0 if (is_revision_mode and self._is_revision_entry(record)) else 0.0,
                        float(record.metadata.get("evidence_strength", self._evidence_strength(record))),
                        float(record.confidence),
                        float(record.metadata.get("weight", 0.0) or 0.0),
                    ),
                    reverse=True,
                )
                results = same[:count]
                for record in results:
                    record.register_retrieval_hit()
                    details = self._details(record)
                    details["times_retrieved"] = int(details.get("times_retrieved", 0)) + 1
                return [self._record_to_legacy_dict(record) for record in results]

        candidates = [record for record in source if plan_ok(record)]
        results = sorted(
            candidates,
            key=lambda record: (
                1.0 if (is_revision_mode and self._is_revision_entry(record)) else 0.0,
                float(record.metadata.get("evidence_strength", self._evidence_strength(record))),
                float(record.confidence),
                float(record.metadata.get("weight", 0.0) or 0.0),
            ),
            reverse=True,
        )[:count]

        for record in results:
            record.register_retrieval_hit()
            details = self._details(record)
            details["times_retrieved"] = int(details.get("times_retrieved", 0)) + 1

        return [self._record_to_legacy_dict(record) for record in results]

    # -------------------------------------------------------------------------
    # Decay / prune / context
    # -------------------------------------------------------------------------
    def _decay(self) -> None:
        now = self._now_ts()
        for record in self.buffer:
            ts = float(record.metadata.get("timestamp", now))
            age_minutes = max(0.0, (now - ts) / 60.0)
            current_weight = float(record.metadata.get("weight", 0.0) or 0.0)
            record.metadata["weight"] = current_weight * (self.decay_rate ** age_minutes)
            record.decay_score = min(1.0, age_minutes / max(1.0, self.max_history))
            record.metadata["age"] = age_minutes

    def _prune(self) -> None:
        self._decay()

        def sort_key(record: MemoryRecord) -> Tuple[float, float, float, float, float, float, float]:
            committed_bonus = 1.0 if self._is_committed_action(record) else 0.0
            identity_bonus = 0.7 if self._is_identity_entry(record) else 0.0
            survival_bonus = 0.4 if self._is_survival_memory(record) else 0.0
            revision_bonus = 0.45 if self._is_revision_entry(record) else 0.0
            contested_penalty = -1.0 if record.verification_status == VerificationStatus.DISPUTED else 0.0
            return (
                committed_bonus + identity_bonus + survival_bonus + revision_bonus,
                contested_penalty,
                float(record.metadata.get("evidence_strength", self._evidence_strength(record))),
                float(record.confidence),
                float(record.metadata.get("weight", 0.0) or 0.0),
                float(record.metadata.get("timestamp", 0.0) or 0.0),
                float(record.metadata.get("quality", 0.0) or 0.0),
            )

        sorted_all = sorted(self.buffer, key=sort_key, reverse=True)

        kept_ids: Set[str] = set()
        retained: List[MemoryRecord] = []

        for domain_name, keep_n in self.DOMAIN_RETENTION_MINIMA.items():
            if keep_n <= 0:
                continue
            domain_rows = [
                record for record in sorted_all
                if self._safe_domain(record.metadata.get("domain")) == domain_name
            ]
            for record in domain_rows[:keep_n]:
                rid = self._normalize_memory_id_candidate(record.memory_id) or f"_obj_{id(record)}"
                if rid not in kept_ids:
                    retained.append(record)
                    kept_ids.add(rid)
                if len(retained) >= self.max_history:
                    break
            if len(retained) >= self.max_history:
                break

        if len(retained) < self.max_history:
            for record in sorted_all:
                rid = self._normalize_memory_id_candidate(record.memory_id) or f"_obj_{id(record)}"
                if rid in kept_ids:
                    continue
                retained.append(record)
                kept_ids.add(rid)
                if len(retained) >= self.max_history:
                    break

        retained.sort(key=sort_key, reverse=True)
        self.buffer = retained[: self.max_history]
        self._rebuild_type_index()

    def context_snapshot(self) -> List[str]:
        return [record.text or "" for record in self.context_buffer[-self.max_context:]]

    # -------------------------------------------------------------------------
    # Reflection helpers
    # -------------------------------------------------------------------------
    def _canonical_strength(self, record: MemoryRecord) -> float:
        details = self._details(record)
        score = 0.0
        score += float(record.confidence) * 1.8
        score += float(record.metadata.get("weight", 0.0) or 0.0) * 1.0
        score += float(record.metadata.get("evidence_strength", self._evidence_strength(record)) or 0.0) * 0.9
        score += min(1.0, float(record.metadata.get("reinforcement_count", 0) or 0.0) / 5.0) * 0.4
        if self._trusted_source(record):
            score += 1.0
        if self._is_survival_memory(record):
            score += 0.35
        if self._is_revision_entry(record):
            score += 0.65
        if record.verification_status == VerificationStatus.DISPUTED:
            score -= 2.0
        if bool(details.get("superseded", False)):
            score -= 2.0
        return score

    def _compress_duplicates(self) -> None:
        window = [
            record for record in self.buffer[-self.REFLECT_DUPLICATE_SCAN_LIMIT:]
            if isinstance(record, MemoryRecord)
        ]
        if len(window) < 2:
            return

        for i in range(len(window)):
            a = window[i]
            if self._details(a).get("superseded"):
                continue
            if self._is_committed_action(a):
                continue
            if a.verification_status == VerificationStatus.DISPUTED:
                continue
            if self._is_harness_injection(a):
                continue
            if self._is_conflict_linked(a):
                continue

            for j in range(i + 1, len(window)):
                b = window[j]
                if self._details(b).get("superseded"):
                    continue
                if self._is_committed_action(b):
                    continue
                if b.verification_status == VerificationStatus.DISPUTED:
                    continue
                if self._is_harness_injection(b):
                    continue
                if self._is_conflict_linked(b):
                    continue

                if self._safe_domain(a.metadata.get("domain")) != self._safe_domain(b.metadata.get("domain")):
                    continue
                if a.kind != b.kind:
                    continue
                if not a.text or not b.text:
                    continue
                if not self._near_duplicate(a.text, b.text):
                    continue

                sa = self._canonical_strength(a)
                sb = self._canonical_strength(b)
                keep, drop = (a, b) if sa >= sb else (b, a)

                drop_d = self._details(drop)
                drop_d["superseded"] = True
                drop_d["recall_blocked"] = True
                drop_d["superseded_by"] = keep.text[:120]

                keep.metadata["weight"] = float(keep.metadata.get("weight", 0.0) or 0.0) + 0.05
                keep.metadata.setdefault("revision_trace", []).append(
                    "Absorbed near-duplicate during reflection"
                )
                drop.metadata.setdefault("revision_trace", []).append(
                    f"Superseded during reflection by: '{keep.text}'"
                )

    def calibration_candidates(self, limit: int = 10) -> List[Dict[str, Any]]:
        candidates: List[Tuple[float, MemoryRecord]] = []
        now = self._now_ts()

        for record in self.buffer:
            if self._is_committed_action(record):
                continue

            details = self._details(record)
            priority = 0.0

            if record.verification_status == VerificationStatus.DISPUTED:
                priority += 5.0
            if bool(details.get("hypothesis", False)):
                priority += 2.0
            if record.verification_status == VerificationStatus.PROVISIONAL:
                priority += 1.5

            staleness = self._safe_str(details.get("staleness_risk") or "medium").lower()
            if staleness == "high":
                priority += 2.0
            elif staleness == "medium":
                priority += 0.75

            last_verified = details.get("last_verified")
            if last_verified is None:
                priority += 0.75
            else:
                try:
                    age_hours = max(0.0, (now - float(last_verified)) / 3600.0)
                    priority += min(2.0, age_hours / 48.0)
                except Exception:
                    priority += 0.25

            priority += max(0.0, 0.75 - float(record.confidence))
            evs = float(record.metadata.get("evidence_strength", self._evidence_strength(record)) or 0.0)
            priority += max(0.0, float(self.CALIBRATION_WEAK_EVIDENCE_TARGET) - evs)

            if priority > 0.0:
                candidates.append((priority, record))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [self._record_to_legacy_dict(record) for _, record in candidates[: max(1, int(limit))]]

    # -------------------------------------------------------------------------
    # Passive reflection
    # -------------------------------------------------------------------------
    def reflect(self) -> None:
        now = self._now_ts()
        self._promotions_this_cycle = 0

        for record in self.buffer:
            age_minutes = self._record_age_minutes(record)
            record.metadata["age"] = age_minutes

            if self._is_committed_action(record):
                details = self._details(record)
                details["exclude_from_planning"] = False
                continue

            content_lower = (record.text or "").lower()
            is_guarded_numeric = any(key in content_lower for key in self.NUMERIC_GUARDRAILS)
            details = self._details(record)

            if age_minutes > 10 and int(record.metadata.get("reinforcement_count", 0) or 0) < 2 and not is_guarded_numeric:
                old_conf = float(record.confidence)
                new_conf = old_conf * 0.9
                if record.verification_status == VerificationStatus.PROVISIONAL:
                    if self._is_survival_memory(record):
                        new_conf = max(new_conf, old_conf * 0.95)
                    elif self._is_revision_entry(record):
                        new_conf = max(new_conf, old_conf * 0.94)
                    else:
                        new_conf = min(new_conf, old_conf * 0.85)
                record.confidence = new_conf
                details["confidence"] = new_conf
                record.metadata.setdefault("revision_trace", []).append(
                    f"Passive decay applied (age={age_minutes:.1f}min, conf {old_conf:.2f} -> {record.confidence:.2f})"
                )

            elif is_guarded_numeric and int(record.metadata.get("reinforcement_count", 0) or 0) < 2:
                old_conf = float(record.confidence)
                record.confidence = max(0.95, old_conf * 0.99)
                details["confidence"] = record.confidence
                record.metadata.setdefault("revision_trace", []).append(
                    f"Guarded numeric decay applied (age={age_minutes:.1f}min, conf {old_conf:.2f} -> {record.confidence:.2f})"
                )

            if self._trusted_source(record) and not details.get("last_verified"):
                details["last_verified"] = float(record.metadata.get("timestamp", now))

            record.metadata["evidence_strength"] = self._evidence_strength(record)
            self._ensure_truth_fields(record)

        self._compress_duplicates()

        for record in self.buffer:
            if self._promotions_this_cycle >= self.MAX_PROMOTIONS_PER_CYCLE:
                break
            if self._eligible_for_promotion(record) and self._numeric_guardrail_pass(record):
                self._promote_to_ltm(record)
                self._promotions_this_cycle += 1

        self._prune()
        self._promotions_this_cycle = 0

    # -------------------------------------------------------------------------
    # Optional convenience diagnostics
    # -------------------------------------------------------------------------
    def force_promote_candidates(self, limit: int = 10) -> int:
        """
        Utility helper for tests or post-run hooks.
        Attempts an immediate promotion sweep over the strongest eligible items.
        """
        if not self.long_term:
            return 0

        promoted = 0
        self._promotions_this_cycle = 0

        candidates = sorted(
            self.buffer,
            key=lambda record: (
                self._promotion_evidence(record),
                float(record.confidence),
                int(record.metadata.get("reinforcement_count", 0) or 0),
                self._promotion_quality(record),
            ),
            reverse=True,
        )

        for record in candidates[: max(1, int(limit))]:
            if self._promotions_this_cycle >= self.MAX_PROMOTIONS_PER_CYCLE:
                break
            if self._eligible_for_promotion(record) and self._numeric_guardrail_pass(record):
                self._promote_to_ltm(record)
                self._promotions_this_cycle += 1
                promoted += 1

        self._promotions_this_cycle = 0
        return promoted

    def stats(self) -> Dict[str, Any]:
        by_kind: Dict[str, int] = {}
        by_status: Dict[str, int] = {}
        by_domain: Dict[str, int] = {}

        for record in self.buffer:
            by_kind[record.kind.value] = by_kind.get(record.kind.value, 0) + 1
            by_status[record.verification_status.value] = (
                by_status.get(record.verification_status.value, 0) + 1
            )
            dom = self._safe_domain(record.metadata.get("domain")) or "unknown"
            by_domain[dom] = by_domain.get(dom, 0) + 1

        return {
            "version": self.VERSION,
            "count": len(self.buffer),
            "max_history": self.max_history,
            "context_count": len(self.context_buffer),
            "episodic_count": len(self.episodic.records),
            "by_kind": by_kind,
            "by_status": by_status,
            "by_domain": by_domain,
        }