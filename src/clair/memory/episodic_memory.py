# FILE: memory/episodic_memory.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from memory.contracts import (
    MemoryKind,
    MemoryRecord,
    MemoryTier,
    VerificationStatus,
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class EpisodicMemory:
    """
    Episodic memory store for recent events, loop traces, and short-lived experiences.

    Purpose:
    - Holds recent episodes that should outlive working memory
    - Keeps simulation/reasoning/reflection traces out of long-term memory
    - Supports promotion candidates for long-term storage
    - Supports decay, pruning, and retrieval by recency/relevance

    Design notes:
    - This is an in-memory store for now
    - Records are canonical MemoryRecord objects
    - All records in this store are forced to tier=EPISODIC
    """

    capacity: int = 500
    default_ttl_hours: float = 48.0
    promote_threshold: float = 0.80
    records: List[MemoryRecord] = field(default_factory=list)

    # -------------------------------------------------------------------------
    # Core storage
    # -------------------------------------------------------------------------

    def store(self, record: MemoryRecord) -> MemoryRecord:
        """
        Store a record in episodic memory.
        Forces the tier to EPISODIC.
        """
        record.set_tier(MemoryTier.EPISODIC)
        record.updated_at = utcnow()
        self.records.append(record)
        self._enforce_capacity()
        return record

    def create_and_store(
        self,
        text: str,
        *,
        kind: MemoryKind = MemoryKind.EPISODE,
        summary: Optional[str] = None,
        confidence: float = 0.0,
        stability: float = 0.0,
        verification_status: VerificationStatus = VerificationStatus.PROVISIONAL,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryRecord:
        """
        Convenience constructor for episodic records.
        """
        record = MemoryRecord(
            text=text,
            summary=summary,
            kind=kind,
            tier=MemoryTier.EPISODIC,
            confidence=confidence,
            stability=stability,
            verification_status=verification_status,
            tags=tags or [],
            metadata=metadata or {},
        )
        return self.store(record)

    def get_by_id(self, memory_id: str) -> Optional[MemoryRecord]:
        for record in self.records:
            if record.memory_id == memory_id:
                record.touch()
                return record
        return None

    def all_records(self) -> List[MemoryRecord]:
        return list(self.records)

    def clear(self) -> None:
        self.records.clear()

    # -------------------------------------------------------------------------
    # Retrieval
    # -------------------------------------------------------------------------

    def retrieve(
        self,
        query: Optional[str] = None,
        *,
        tags: Optional[List[str]] = None,
        kinds: Optional[List[MemoryKind]] = None,
        limit: int = 10,
        include_disputed: bool = True,
        max_age_hours: Optional[float] = None,
    ) -> List[MemoryRecord]:
        """
        Retrieve episodic memories using lightweight scoring.

        Score factors:
        - text relevance to query
        - tag overlap
        - recency
        - confidence
        - stability
        - contradiction penalty
        """
        now = utcnow()
        normalized_query = (query or "").strip().lower()
        normalized_tags = {t.strip().lower() for t in (tags or []) if str(t).strip()}
        allowed_kinds = set(kinds or [])

        candidates: List[tuple[float, MemoryRecord]] = []

        for record in self.records:
            if not record.is_viable():
                continue

            if allowed_kinds and record.kind not in allowed_kinds:
                continue

            if not include_disputed and record.verification_status == VerificationStatus.DISPUTED:
                continue

            if max_age_hours is not None:
                age_hours = self._age_hours(record, now=now)
                if age_hours > max_age_hours:
                    continue

            score = self._score_record(
                record,
                query=normalized_query,
                tag_filter=normalized_tags,
                now=now,
            )
            if score <= 0.0:
                continue

            candidates.append((score, record))

        candidates.sort(key=lambda item: item[0], reverse=True)

        out: List[MemoryRecord] = []
        for _, record in candidates[: max(1, limit)]:
            record.register_retrieval_hit()
            out.append(record)
        return out

    def recent(
        self,
        *,
        limit: int = 10,
        kinds: Optional[List[MemoryKind]] = None,
    ) -> List[MemoryRecord]:
        """
        Return most recent episodic records, newest first.
        """
        allowed_kinds = set(kinds or [])
        filtered = [
            r
            for r in self.records
            if r.is_viable() and (not allowed_kinds or r.kind in allowed_kinds)
        ]
        filtered.sort(key=lambda r: r.created_at, reverse=True)

        out = filtered[: max(1, limit)]
        for record in out:
            record.register_retrieval_hit()
        return out

    def get_promotion_candidates(
        self,
        *,
        min_confidence: Optional[float] = None,
        min_stability: Optional[float] = None,
        exclude_disputed: bool = True,
        limit: int = 25,
    ) -> List[MemoryRecord]:
        """
        Return episodic memories that look mature enough for long-term review.
        """
        min_conf = self.promote_threshold if min_confidence is None else float(min_confidence)
        min_stab = self.promote_threshold if min_stability is None else float(min_stability)

        candidates: List[MemoryRecord] = []
        for record in self.records:
            if not record.is_viable():
                continue
            if record.confidence < min_conf:
                continue
            if record.stability < min_stab:
                continue
            if record.contradiction_count > 0:
                continue
            if exclude_disputed and record.verification_status == VerificationStatus.DISPUTED:
                continue
            candidates.append(record)

        candidates.sort(
            key=lambda r: (r.confidence, r.stability, r.retrieval_hits, r.created_at),
            reverse=True,
        )
        return candidates[: max(1, limit)]

    # -------------------------------------------------------------------------
    # Lifecycle / maintenance
    # -------------------------------------------------------------------------

    def prune_expired(self, ttl_hours: Optional[float] = None) -> int:
        """
        Remove expired episodic memories.
        Returns the number of records removed.
        """
        ttl = self.default_ttl_hours if ttl_hours is None else float(ttl_hours)
        now = utcnow()
        kept: List[MemoryRecord] = []
        removed = 0

        for record in self.records:
            if self._age_hours(record, now=now) > ttl:
                removed += 1
                continue
            kept.append(record)

        self.records = kept
        return removed

    def prune_low_value(
        self,
        *,
        min_confidence: float = 0.15,
        min_priority: float = 0.10,
        max_contradictions: int = 3,
    ) -> int:
        """
        Remove weak episodic records that are low confidence, low priority,
        or repeatedly contradicted.
        """
        kept: List[MemoryRecord] = []
        removed = 0

        for record in self.records:
            drop = False

            if record.confidence < min_confidence and record.priority < min_priority:
                drop = True

            if record.contradiction_count > max_contradictions:
                drop = True

            if drop:
                removed += 1
                continue

            kept.append(record)

        self.records = kept
        return removed

    def remove(self, memory_id: str) -> bool:
        """
        Remove a record by ID.
        """
        before = len(self.records)
        self.records = [r for r in self.records if r.memory_id != memory_id]
        return len(self.records) != before

    def demote_to_quarantine(self, memory_id: str, reason: Optional[str] = None) -> bool:
        """
        Mark an episodic record as quarantined in-place.
        Caller can later move it elsewhere if desired.
        """
        record = self.get_by_id(memory_id)
        if record is None:
            return False

        record.set_tier(MemoryTier.QUARANTINE)
        record.set_verification_status(VerificationStatus.DISPUTED)
        if reason:
            notes = record.metadata.setdefault("quarantine_notes", [])
            if isinstance(notes, list):
                notes.append(reason)
            else:
                record.metadata["quarantine_notes"] = [str(reason)]
        record.updated_at = utcnow()
        return True

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------

    def _enforce_capacity(self) -> None:
        if len(self.records) <= self.capacity:
            return

        self.records.sort(key=self._capacity_sort_key)
        overflow = len(self.records) - self.capacity
        if overflow > 0:
            self.records = self.records[overflow:]

    def _capacity_sort_key(self, record: MemoryRecord) -> tuple:
        """
        Lower-value and older records get discarded first.
        """
        last_seen = record.last_accessed or record.created_at
        return (
            record.priority,
            record.confidence,
            record.stability,
            -record.retrieval_hits,
            last_seen.timestamp(),
        )

    def _score_record(
        self,
        record: MemoryRecord,
        *,
        query: str,
        tag_filter: set[str],
        now: datetime,
    ) -> float:
        score = 0.0

        # Query relevance
        if query:
            text_l = record.text.lower()
            summary_l = (record.summary or "").lower()

            if query in text_l:
                score += 1.0
            elif query in summary_l:
                score += 0.8
            else:
                query_terms = [term for term in query.split() if term]
                term_hits = sum(1 for term in query_terms if term in text_l or term in summary_l)
                if query_terms:
                    score += 0.6 * (term_hits / len(query_terms))
        else:
            score += 0.2

        # Tag overlap
        if tag_filter:
            record_tags = {tag.lower() for tag in record.tags}
            overlap = len(record_tags.intersection(tag_filter))
            score += min(0.8, overlap * 0.25)

        # Recency bonus
        age_hours = self._age_hours(record, now=now)
        if age_hours <= 1:
            score += 0.8
        elif age_hours <= 6:
            score += 0.6
        elif age_hours <= 24:
            score += 0.4
        elif age_hours <= 48:
            score += 0.2

        # Stability / confidence / priority bonuses
        score += record.confidence * 0.7
        score += record.stability * 0.5
        score += record.priority * 0.3

        # Retrieval reinforcement
        score += min(0.5, record.retrieval_hits * 0.05)

        # Penalties
        score -= min(1.0, record.contradiction_count * 0.25)
        score -= record.decay_score * 0.3

        if record.verification_status == VerificationStatus.REJECTED:
            score -= 2.0
        elif record.verification_status == VerificationStatus.DISPUTED:
            score -= 0.5
        elif record.verification_status == VerificationStatus.VERIFIED:
            score += 0.3

        return score

    def _age_hours(self, record: MemoryRecord, *, now: Optional[datetime] = None) -> float:
        ref = now or utcnow()
        age = ref - record.created_at
        return max(0.0, age.total_seconds() / 3600.0)

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        now = utcnow()
        by_kind: Dict[str, int] = {}
        by_status: Dict[str, int] = {}

        ages: List[float] = []
        for record in self.records:
            by_kind[record.kind.value] = by_kind.get(record.kind.value, 0) + 1
            by_status[record.verification_status.value] = (
                by_status.get(record.verification_status.value, 0) + 1
            )
            ages.append(self._age_hours(record, now=now))

        avg_age = sum(ages) / len(ages) if ages else 0.0

        return {
            "count": len(self.records),
            "capacity": self.capacity,
            "default_ttl_hours": self.default_ttl_hours,
            "promote_threshold": self.promote_threshold,
            "avg_age_hours": round(avg_age, 3),
            "by_kind": by_kind,
            "by_status": by_status,
        }