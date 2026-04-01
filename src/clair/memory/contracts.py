# FILE: memory/contracts.py
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


# =============================================================================
# Enums
# =============================================================================


class MemoryKind(str, Enum):
    FACT = "fact"
    SUMMARY = "summary"
    PROCEDURE = "procedure"
    EPISODE = "episode"
    HAZARD = "hazard"
    GOAL = "goal"
    REFLECTION = "reflection"
    UNRESOLVED = "unresolved"
    RULE = "rule"
    USER_PREFERENCE = "user_preference"
    SCENARIO = "scenario"


class MemoryTier(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    LONG_TERM = "long_term"
    QUARANTINE = "quarantine"


class VerificationStatus(str, Enum):
    UNVERIFIED = "unverified"
    PROVISIONAL = "provisional"
    VERIFIED = "verified"
    DISPUTED = "disputed"
    REJECTED = "rejected"


class SourceType(str, Enum):
    USER_INPUT = "user_input"
    DOCUMENT = "document"
    SIMULATION = "simulation"
    REFLECTION = "reflection"
    VERIFICATION = "verification"
    SYSTEM = "system"
    SENSOR = "sensor"
    IMPORTED_MEMORY = "imported_memory"
    UNKNOWN = "unknown"


# =============================================================================
# Helpers
# =============================================================================


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_memory_id() -> str:
    return f"mem_{uuid.uuid4().hex}"


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return lo
    return max(lo, min(hi, value))


def _normalize_str_list(values: Optional[List[Any]]) -> List[str]:
    if not values:
        return []
    out: List[str] = []
    seen = set()
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


# =============================================================================
# Evidence + Signals
# =============================================================================


@dataclass
class EvidencePacket:
    """
    Structured support or contradiction evidence tied to a memory record.
    """

    evidence_id: str = field(default_factory=lambda: f"ev_{uuid.uuid4().hex}")
    source_type: SourceType = SourceType.UNKNOWN
    source_ref: Optional[str] = None
    snippet: Optional[str] = None
    stance: str = "support"  # support | contradict | context | unknown
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = _clamp(self.confidence)
        self.stance = (self.stance or "unknown").strip().lower()
        if not self.stance:
            self.stance = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["source_type"] = self.source_type.value
        return data


@dataclass
class MemorySignals:
    """
    Optional structured annotations extracted during ingest, review, or routing.
    """

    domain: Optional[str] = None
    hazard_family: Optional[str] = None
    temporal_scope: Optional[str] = None
    novelty: float = 0.0
    urgency: float = 0.0
    usefulness: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.novelty = _clamp(self.novelty)
        self.urgency = _clamp(self.urgency)
        self.usefulness = _clamp(self.usefulness)

        if isinstance(self.domain, str):
            self.domain = self.domain.strip().lower() or None
        if isinstance(self.hazard_family, str):
            self.hazard_family = self.hazard_family.strip().lower() or None
        if isinstance(self.temporal_scope, str):
            self.temporal_scope = self.temporal_scope.strip().lower() or None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Main Memory Record
# =============================================================================


@dataclass
class MemoryRecord:
    """
    Canonical memory contract.

    This should become the shared record format used by:
    - working memory
    - episodic memory
    - long-term memory
    - thalamus routing
    - verification
    - calibration
    - reflection/promotions

    Design goals:
    - stable schema across modules
    - explicit memory state
    - evidence-aware
    - promotion/demotion friendly
    - retrieval-score friendly
    """

    # ------------------------
    # Unique identity & timing
    # ------------------------
    memory_id: str = field(default_factory=new_memory_id)
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    last_accessed: Optional[datetime] = None

    # ------------------------
    # Core content
    # ------------------------
    text: str = ""
    summary: Optional[str] = None

    # ------------------------
    # Classification
    # ------------------------
    kind: MemoryKind = MemoryKind.UNRESOLVED
    tier: MemoryTier = MemoryTier.WORKING
    source_type: SourceType = SourceType.UNKNOWN
    source_ref: Optional[str] = None

    # ------------------------
    # Trust / stability / lifecycle
    # ------------------------
    confidence: float = 0.0
    stability: float = 0.0
    decay_score: float = 0.0
    priority: float = 0.5
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED

    # ------------------------
    # Access / contention tracking
    # ------------------------
    access_count: int = 0
    contradiction_count: int = 0
    retrieval_hits: int = 0
    retrieval_misses: int = 0

    # ------------------------
    # Connectivity / structure
    # ------------------------
    tags: List[str] = field(default_factory=list)
    related_ids: List[str] = field(default_factory=list)
    evidence: List[EvidencePacket] = field(default_factory=list)
    signals: MemorySignals = field(default_factory=MemorySignals)

    # ------------------------
    # Free-form extension points
    # ------------------------
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.text = (self.text or "").strip()
        self.summary = (self.summary or "").strip() or None

        self.confidence = _clamp(self.confidence)
        self.stability = _clamp(self.stability)
        self.decay_score = _clamp(self.decay_score)
        self.priority = _clamp(self.priority)

        self.tags = _normalize_str_list(self.tags)
        self.related_ids = _normalize_str_list(self.related_ids)

        self.access_count = max(0, int(self.access_count))
        self.contradiction_count = max(0, int(self.contradiction_count))
        self.retrieval_hits = max(0, int(self.retrieval_hits))
        self.retrieval_misses = max(0, int(self.retrieval_misses))

        if self.last_accessed is None and self.access_count > 0:
            self.last_accessed = self.updated_at

    # ------------------------
    # Utility methods
    # ------------------------
    def is_viable(self) -> bool:
        """
        Structural viability only.
        This does NOT imply truth, reliability, or readiness for promotion.
        """
        return bool(self.text)

    def touch(self) -> None:
        """
        Mark access/update timestamps for retrieval accounting.
        """
        now = utcnow()
        self.last_accessed = now
        self.updated_at = now
        self.access_count += 1

    def register_retrieval_hit(self) -> None:
        self.retrieval_hits += 1
        self.touch()

    def register_retrieval_miss(self) -> None:
        self.retrieval_misses += 1
        self.updated_at = utcnow()

    def add_tag(self, tag: str) -> None:
        if not tag:
            return
        normalized = tag.strip()
        if not normalized:
            return
        if normalized.lower() not in {t.lower() for t in self.tags}:
            self.tags.append(normalized)
            self.updated_at = utcnow()

    def add_related_id(self, memory_id: str) -> None:
        if not memory_id:
            return
        memory_id = memory_id.strip()
        if not memory_id:
            return
        if memory_id.lower() not in {m.lower() for m in self.related_ids}:
            self.related_ids.append(memory_id)
            self.updated_at = utcnow()

    def add_evidence(self, packet: EvidencePacket) -> None:
        self.evidence.append(packet)
        self.updated_at = utcnow()

    def set_verification_status(self, status: VerificationStatus) -> None:
        self.verification_status = status
        self.updated_at = utcnow()

    def bump_contradiction(self, amount: int = 1) -> None:
        self.contradiction_count = max(0, self.contradiction_count + int(amount))
        self.updated_at = utcnow()

    def set_tier(self, tier: MemoryTier) -> None:
        self.tier = tier
        self.updated_at = utcnow()

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        data["tier"] = self.tier.value
        data["source_type"] = self.source_type.value
        data["verification_status"] = self.verification_status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryRecord":
        evidence_items = [
            EvidencePacket(
                evidence_id=item.get("evidence_id", f"ev_{uuid.uuid4().hex}"),
                source_type=SourceType(item.get("source_type", SourceType.UNKNOWN.value)),
                source_ref=item.get("source_ref"),
                snippet=item.get("snippet"),
                stance=item.get("stance", "unknown"),
                confidence=item.get("confidence", 0.0),
                metadata=item.get("metadata", {}) or {},
            )
            for item in (data.get("evidence") or [])
        ]

        signals_data = data.get("signals") or {}
        signals = MemorySignals(
            domain=signals_data.get("domain"),
            hazard_family=signals_data.get("hazard_family"),
            temporal_scope=signals_data.get("temporal_scope"),
            novelty=signals_data.get("novelty", 0.0),
            urgency=signals_data.get("urgency", 0.0),
            usefulness=signals_data.get("usefulness", 0.0),
            metadata=signals_data.get("metadata", {}) or {},
        )

        return cls(
            memory_id=data.get("memory_id", new_memory_id()),
            created_at=data.get("created_at", utcnow()),
            updated_at=data.get("updated_at", utcnow()),
            last_accessed=data.get("last_accessed"),
            text=data.get("text", ""),
            summary=data.get("summary"),
            kind=MemoryKind(data.get("kind", MemoryKind.UNRESOLVED.value)),
            tier=MemoryTier(data.get("tier", MemoryTier.WORKING.value)),
            source_type=SourceType(data.get("source_type", SourceType.UNKNOWN.value)),
            source_ref=data.get("source_ref"),
            confidence=data.get("confidence", 0.0),
            stability=data.get("stability", 0.0),
            decay_score=data.get("decay_score", 0.0),
            priority=data.get("priority", 0.5),
            verification_status=VerificationStatus(
                data.get("verification_status", VerificationStatus.UNVERIFIED.value)
            ),
            access_count=data.get("access_count", 0),
            contradiction_count=data.get("contradiction_count", 0),
            retrieval_hits=data.get("retrieval_hits", 0),
            retrieval_misses=data.get("retrieval_misses", 0),
            tags=data.get("tags", []) or [],
            related_ids=data.get("related_ids", []) or [],
            evidence=evidence_items,
            signals=signals,
            metadata=data.get("metadata", {}) or {},
        )


# =============================================================================
# Convenience Constructors
# =============================================================================


def make_memory_record(
    text: str,
    *,
    kind: MemoryKind = MemoryKind.UNRESOLVED,
    tier: MemoryTier = MemoryTier.WORKING,
    source_type: SourceType = SourceType.UNKNOWN,
    source_ref: Optional[str] = None,
    confidence: float = 0.0,
    stability: float = 0.0,
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> MemoryRecord:
    return MemoryRecord(
        text=text,
        kind=kind,
        tier=tier,
        source_type=source_type,
        source_ref=source_ref,
        confidence=confidence,
        stability=stability,
        verification_status=verification_status,
        tags=tags or [],
        metadata=metadata or {},
    )