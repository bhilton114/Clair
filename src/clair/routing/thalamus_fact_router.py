# FILE: routing/thalamus_fact_router.py
from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import config
from intake.contracts import InputPacket, UncertaintyFlags


class ThalamusFactRouter:
    """
    Routes packets based on structural viability, uncertainty, and intent.

    Guarantees:
    - reads processor packet_type from signals["packet_type"]
    - canonicalizes packet.raw_input into {"type": ..., "content": ...}
    - supports legacy route_packets() callers that expect forwarded/deferred/rejected
    - also emits route-target buckets for fact-thalamus orchestration
    - optional short-TTL dedupe via signals["content_hash"]
    """

    VERSION = "2.0-router"

    def __init__(self):
        self.deferred_log: List[Dict[str, Any]] = []
        self.rejected_log: List[Dict[str, Any]] = []

        self.max_retries = int(getattr(config, "THALAMUS_MAX_RETRIES", 3))
        self.defer_backoff_sec = float(getattr(config, "THALAMUS_DEFER_BACKOFF_SEC", 2.0))

        self.enable_dedupe = bool(getattr(config, "THALAMUS_ENABLE_DEDUPE", True))
        self.dedupe_ttl_sec = float(getattr(config, "THALAMUS_DEDUPE_TTL_SEC", 30.0))
        self._recent_hashes: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def route_packets(self, packets: List[Any]) -> Dict[str, List[Any]]:
        forwarded: List[Any] = []
        deferred: List[Any] = []
        rejected: List[Any] = []

        fact_recall: List[Any] = []
        learning: List[Any] = []
        verification: List[Any] = []
        calibration: List[Any] = []
        reflection: List[Any] = []
        direct_answer: List[Any] = []

        if not packets:
            return self._empty_result()

        now = time.time()
        self._purge_hash_cache(now)

        for packet in packets:
            decision, reason = self._assess(packet, now=now)

            if decision == "forward":
                forwarded.append(packet)
                target = self._route_target(packet)
                if target == "fact_recall":
                    fact_recall.append(packet)
                elif target == "learning":
                    learning.append(packet)
                elif target == "verification":
                    verification.append(packet)
                elif target == "calibration":
                    calibration.append(packet)
                elif target == "reflection":
                    reflection.append(packet)
                else:
                    direct_answer.append(packet)

            elif decision == "defer":
                deferred.append(packet)
                self._log(packet, reason, deferred=True, now=now)

            else:
                rejected.append(packet)
                self._log(packet, reason, deferred=False, now=now)

        deferred.sort(key=lambda p: self._defer_sort_key(p), reverse=True)

        if getattr(config, "VERBOSE", False):
            print(
                "[Thalamus] "
                f"Forwarded={len(forwarded)} Deferred={len(deferred)} Rejected={len(rejected)} "
                f"Recall={len(fact_recall)} Learn={len(learning)} Verify={len(verification)} "
                f"Calibrate={len(calibration)} Reflect={len(reflection)} Answer={len(direct_answer)}"
            )

        return {
            "forwarded": forwarded,
            "deferred": deferred,
            "rejected": rejected,
            "fact_recall": fact_recall,
            "learning": learning,
            "verification": verification,
            "calibration": calibration,
            "reflection": reflection,
            "direct_answer": direct_answer,
        }

    # ------------------------------------------------------------------
    # Assessment logic
    # ------------------------------------------------------------------
    def _assess(self, packet: Any, now: float) -> Tuple[str, str]:
        packet = self._coerce_packet(packet, now=now)
        if packet is None:
            return "reject", "structurally_invalid"

        is_viable_fn = getattr(packet, "is_viable", None)
        if callable(is_viable_fn) and not bool(is_viable_fn()):
            return "reject", "structurally_invalid"

        u = getattr(packet, "uncertainty", None)
        if u is None or not isinstance(u, UncertaintyFlags):
            u = self._coerce_uncertainty(u)
            packet.uncertainty = u

        sig = getattr(packet, "signals", None)
        if not isinstance(sig, dict):
            sig = {}
            packet.signals = sig

        text = sig.get("normalized_text")
        if not isinstance(text, str):
            raw = getattr(packet, "raw_input", "")
            if isinstance(raw, dict):
                text = str(raw.get("content", "") or "")
            else:
                text = str(raw or "")
            sig["normalized_text"] = text

        text = str(text or "").strip()
        if not text:
            return "reject", "empty_text"

        if self.enable_dedupe:
            ch = sig.get("content_hash") or sig.get("content_hash_short")
            if isinstance(ch, str) and ch:
                if ch in self._recent_hashes:
                    return "reject", "duplicate_recent"
                self._recent_hashes[ch] = now

        extraction_confidence = float(getattr(packet, "extraction_confidence", 0.0) or 0.0)
        if getattr(u, "low_signal_quality", False) and extraction_confidence < 0.3:
            return "reject", "insufficient_signal"

        retry_count = int(getattr(packet, "retry_count", 0) or 0)
        if retry_count >= self.max_retries:
            return "reject", "max_retries_exceeded"

        next_time = float(getattr(packet, "next_eligible_time", 0.0) or 0.0)
        if next_time > now:
            return "defer", "backoff_wait"

        pkt_type = self._get_packet_type(packet)
        self._canonicalize_raw_input(packet, pkt_type)
        self._ensure_route_target(packet)

        flags = self._uncertainty_flags(u)
        flag_count = len(flags)

        if flag_count >= 2:
            return "defer", "high_uncertainty"

        if flag_count == 1:
            self._add_constraint(packet, f"single_uncertainty:{flags[0]}")
            return "forward", "acceptable_signal_single_flag"

        return "forward", "acceptable_signal"

    # ------------------------------------------------------------------
    # Packet type + route target
    # ------------------------------------------------------------------
    def _get_packet_type(self, packet: Any) -> str:
        raw = getattr(packet, "raw_input", None)
        if isinstance(raw, dict):
            t = str(raw.get("type", "") or "").strip().lower()
            if t in {"ask", "lesson", "observe", "feedback"}:
                return t

        sig = getattr(packet, "signals", {}) or {}
        t = str(sig.get("packet_type", "") or "").strip().lower()
        if t in {"ask", "lesson", "observe", "feedback"}:
            return t

        modality = str(getattr(packet, "modality", "") or "").strip().lower()
        if modality == "feedback":
            return "feedback"
        if modality == "file":
            return "observe"
        return "ask"

    def _canonicalize_raw_input(self, packet: Any, pkt_type: str) -> None:
        sig = getattr(packet, "signals", {}) or {}
        text = sig.get("normalized_text", "")
        if not isinstance(text, str):
            text = str(text or "")

        raw = getattr(packet, "raw_input", None)
        if isinstance(raw, dict):
            raw["type"] = raw.get("type") or pkt_type
            raw["content"] = raw.get("content") or text
            raw.setdefault("modality", getattr(packet, "modality", None))
            raw.setdefault("source", getattr(packet, "source", None))
            packet.raw_input = raw
            return

        packet.raw_input = {
            "type": pkt_type,
            "content": text,
            "modality": getattr(packet, "modality", None),
            "source": getattr(packet, "source", None),
        }

    def _ensure_route_target(self, packet: Any) -> None:
        sig = getattr(packet, "signals", None)
        if not isinstance(sig, dict):
            sig = {}
            packet.signals = sig

        existing = str(sig.get("route_target", "") or "").strip().lower()
        if existing in {
            "fact_recall",
            "learning",
            "verification",
            "calibration",
            "reflection",
            "direct_answer",
        }:
            return

        text = str(sig.get("normalized_text", "") or "").strip().lower()
        pkt_type = self._get_packet_type(packet)

        if pkt_type in {"lesson", "observe"}:
            sig["route_target"] = "learning"
            return

        if pkt_type == "feedback":
            sig["route_target"] = "calibration"
            return

        if any(x in text for x in ("verify", "verification", "is this true", "check this fact", "fact check")):
            sig["route_target"] = "verification"
            return

        if any(x in text for x in ("reflect", "review memories", "self review", "maintenance pass")):
            sig["route_target"] = "reflection"
            return

        if any(x in text for x in ("calibrate", "calibration", "audit memory", "memory audit")):
            sig["route_target"] = "calibration"
            return

        if "?" in text or any(x in text for x in ("what", "who", "when", "where", "why", "how", "recall", "remember")):
            sig["route_target"] = "fact_recall"
            return

        sig["route_target"] = "direct_answer"

    def _route_target(self, packet: Any) -> str:
        sig = getattr(packet, "signals", {}) or {}
        target = str(sig.get("route_target", "") or "").strip().lower()
        if target in {
            "fact_recall",
            "learning",
            "verification",
            "calibration",
            "reflection",
            "direct_answer",
        }:
            return target
        return "direct_answer"

    # ------------------------------------------------------------------
    # Coercion helpers
    # ------------------------------------------------------------------
    def _coerce_packet(self, packet: Any, now: float) -> Any:
        if packet is None:
            return None

        if isinstance(packet, InputPacket):
            self._ensure_min_fields(packet, now=now)
            return packet

        if not callable(getattr(packet, "is_viable", None)):
            return None

        raw = getattr(packet, "raw_input", None)
        if raw is None:
            return None

        self._ensure_min_fields(packet, now=now)

        if not getattr(packet, "packet_id", None):
            packet.packet_id = f"pkt_{int(now * 1000)}"

        if not getattr(packet, "modality", None):
            packet.modality = "text"

        if not hasattr(packet, "source"):
            packet.source = "test"

        sig = getattr(packet, "signals", None)
        if not isinstance(sig, dict):
            sig = {}
            packet.signals = sig

        if "normalized_text" not in sig or not isinstance(sig.get("normalized_text"), str):
            if isinstance(raw, dict):
                sig["normalized_text"] = str(raw.get("content", "") or "")
            else:
                sig["normalized_text"] = str(raw or "")

        if not hasattr(packet, "extraction_confidence"):
            packet.extraction_confidence = 0.85

        return packet

    def _ensure_min_fields(self, packet: Any, now: float) -> None:
        if not hasattr(packet, "created_at") or not getattr(packet, "created_at", 0.0):
            packet.created_at = float(now)

        if not hasattr(packet, "retry_count"):
            packet.retry_count = 0
        if not hasattr(packet, "next_eligible_time"):
            packet.next_eligible_time = 0.0

        if not hasattr(packet, "constraints") or not isinstance(getattr(packet, "constraints", None), list):
            packet.constraints = []

        u = getattr(packet, "uncertainty", None)
        if u is None or not isinstance(u, UncertaintyFlags):
            packet.uncertainty = self._coerce_uncertainty(u)

        if not hasattr(packet, "extraction_confidence"):
            packet.extraction_confidence = 0.85

        if not hasattr(packet, "signals") or not isinstance(getattr(packet, "signals", None), dict):
            packet.signals = {}

    def _coerce_uncertainty(self, u: Any) -> UncertaintyFlags:
        if isinstance(u, UncertaintyFlags):
            return u

        flags = UncertaintyFlags()

        if isinstance(u, dict):
            for name in ("metaphor_detected", "missing_references", "conflicting_signals", "low_signal_quality"):
                if name in u:
                    setattr(flags, name, bool(u.get(name)))
            return flags

        if u is not None:
            for name in ("metaphor_detected", "missing_references", "conflicting_signals", "low_signal_quality"):
                if hasattr(u, name):
                    setattr(flags, name, bool(getattr(u, name)))
            return flags

        return flags

    def _uncertainty_flags(self, u: UncertaintyFlags) -> List[str]:
        active: List[str] = []
        for name in ("metaphor_detected", "missing_references", "conflicting_signals", "low_signal_quality"):
            if getattr(u, name, False):
                active.append(name)
        return active

    # ------------------------------------------------------------------
    # Severity + sorting
    # ------------------------------------------------------------------
    def severity(self, packet: Any) -> float:
        u = getattr(packet, "uncertainty", None)
        if not isinstance(u, UncertaintyFlags):
            u = self._coerce_uncertainty(u)

        flags = self._uncertainty_flags(u)
        conf = float(getattr(packet, "extraction_confidence", 0.0) or 0.0)
        retry = int(getattr(packet, "retry_count", 0) or 0)

        score = (1.5 * len(flags)) + (1.0 * conf) - (0.4 * retry)
        if "missing_references" in flags:
            score += 0.5
        return score

    def _defer_sort_key(self, packet: Any):
        sev = self.severity(packet)
        retry = int(getattr(packet, "retry_count", 0) or 0)
        ts = float(getattr(packet, "created_at", 0.0) or 0.0)
        return (sev, -retry, -ts)

    def _add_constraint(self, packet: Any, constraint: str) -> None:
        if not getattr(packet, "constraints", None) or not isinstance(packet.constraints, list):
            packet.constraints = []
        if constraint and constraint not in packet.constraints:
            packet.constraints.append(constraint)

    # ------------------------------------------------------------------
    # Logging + backoff
    # ------------------------------------------------------------------
    def _log(self, packet: Any, reason: str, deferred: bool, now: float) -> None:
        packet_id = getattr(packet, "packet_id", None)
        modality = getattr(packet, "modality", None)
        constraints = list(getattr(packet, "constraints", []) or [])

        uncertainty_obj = getattr(packet, "uncertainty", None)
        if not isinstance(uncertainty_obj, UncertaintyFlags):
            uncertainty_obj = self._coerce_uncertainty(uncertainty_obj)
            packet.uncertainty = uncertainty_obj

        uncertainty = vars(uncertainty_obj) if hasattr(uncertainty_obj, "__dict__") else {}
        extraction_confidence = float(getattr(packet, "extraction_confidence", 0.0) or 0.0)
        retry_count = int(getattr(packet, "retry_count", 0) or 0)

        entry = {
            "packet_id": packet_id,
            "timestamp": now,
            "modality": modality,
            "constraints": constraints,
            "uncertainty": uncertainty,
            "extraction_confidence": extraction_confidence,
            "reason": reason,
            "retry_count": retry_count,
            "severity": self.severity(packet),
        }

        if deferred:
            packet.retry_count = retry_count + 1
            packet.next_eligible_time = now + (self.defer_backoff_sec * packet.retry_count)
            self.deferred_log.append(entry)
        else:
            self.rejected_log.append(entry)

    # ------------------------------------------------------------------
    # Dedupe maintenance
    # ------------------------------------------------------------------
    def _purge_hash_cache(self, now: float) -> None:
        if not self.enable_dedupe:
            return
        ttl = max(1.0, float(self.dedupe_ttl_sec))
        dead = [h for h, ts in self._recent_hashes.items() if (now - ts) > ttl]
        for h in dead:
            self._recent_hashes.pop(h, None)

    def _empty_result(self) -> Dict[str, List[Any]]:
        return {
            "forwarded": [],
            "deferred": [],
            "rejected": [],
            "fact_recall": [],
            "learning": [],
            "verification": [],
            "calibration": [],
            "reflection": [],
            "direct_answer": [],
        }