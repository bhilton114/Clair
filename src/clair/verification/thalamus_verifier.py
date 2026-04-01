# FILE: verification/thalamus_verifier.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .thalamus_sources import ThalamusSources
from .thalamus_evidence import ThalamusEvidence


class ThalamusVerifier:
    """
    Selective outside-intake verifier.

    Responsibilities:
    - decide whether a claim should be externally checked
    - extract outside-source material
    - evaluate support / contradiction / insufficiency
    - return structured verification results
    - optionally build a feedback packet for higher orchestration

    This module NEVER writes memory directly.
    Cerebellar, ACC, or higher orchestration remains the authority for memory updates.
    """

    VERSION = "1.2-selective"

    def __init__(
        self,
        *args: Any,
        config: Any = None,
        sources: Optional[ThalamusSources] = None,
        evidence: Optional[ThalamusEvidence] = None,
    ) -> None:
        self.config = config
        self.sources = sources or ThalamusSources()
        self.evidence = evidence or ThalamusEvidence()

    def should_verify(
        self,
        packet: Dict[str, Any],
        *,
        force: bool = False,
    ) -> bool:
        if force:
            return True

        if not isinstance(packet, dict):
            return False

        normalized = self._normalize_packet(packet)
        claim = normalized["claim"]
        if not claim:
            return False

        if bool(normalized.get("needs_external_verification", False)):
            return True

        vstat = self._safe_lower(
            normalized.get("verification_status")
            or normalized.get("status")
            or "unverified"
        )

        stale = self._staleness_value(normalized)
        source_count = self._source_count(normalized)
        hypothesis = bool(normalized.get("hypothesis", False))
        conflicts = normalized.get("conflict_with_ids", []) or []
        route_target = self._safe_lower(normalized.get("route_target"))
        details = self._details(normalized)

        if route_target == "verification":
            return True
        if vstat in {"contested", "revised", "pending_external", "pending", "provisional", "hypothesis"}:
            return True
        if stale >= 0.50:
            return True
        if source_count <= 0:
            return True
        if hypothesis:
            return True
        if conflicts:
            return True
        if bool(details.get("pending_verification", False)):
            return True

        return False

    def verify(
        self,
        packet: Dict[str, Any],
        *,
        intake_text: str = "",
        source_name: str = "intake_text",
        force: bool = False,
    ) -> Dict[str, Any]:
        if not isinstance(packet, dict):
            return {
                "ok": False,
                "status": "error",
                "verdict": "unsure",
                "external_verified": False,
                "error": "packet must be a dict",
            }

        normalized_packet = self._normalize_packet(packet)
        claim = normalized_packet["claim"]

        if not claim:
            return {
                "ok": False,
                "status": "error",
                "verdict": "unsure",
                "external_verified": False,
                "error": "packet missing claim",
            }

        if not self.should_verify(normalized_packet, force=force):
            return {
                "ok": True,
                "claim": claim,
                "status": "skipped",
                "verdict": "unsure",
                "confidence": 0.0,
                "external_verified": False,
                "external_origin": source_name,
                "external_detail": "Verification not required by current gate.",
                "matches": [],
                "conflicts": [],
                "support_score": 0.0,
                "conflict_score": 0.0,
                "packet": self._packet_summary(normalized_packet),
            }

        try:
            extracted = self.sources.extract(
                normalized_packet,
                intake_text=intake_text,
                source_name=source_name,
            )
        except Exception as exc:
            return {
                "ok": False,
                "claim": claim,
                "status": "error",
                "verdict": "unsure",
                "external_verified": False,
                "external_origin": source_name,
                "error": f"sources.extract failed: {exc}",
                "packet": self._packet_summary(normalized_packet),
            }

        try:
            evaluated = self.evidence.evaluate(normalized_packet, extracted)
        except Exception as exc:
            return {
                "ok": False,
                "claim": claim,
                "status": "error",
                "verdict": "unsure",
                "external_verified": False,
                "external_origin": source_name,
                "error": f"evidence.evaluate failed: {exc}",
                "packet": self._packet_summary(normalized_packet),
            }

        status = self._safe_lower(evaluated.get("status", "insufficient")) or "insufficient"
        verdict = self._normalize_verdict(evaluated.get("verdict", "unsure"))
        confidence = self._clamp01(evaluated.get("confidence", 0.35), default=0.35)

        if status == "supported":
            detail = "Claim matched supporting evidence in outside intake."
        elif status == "contradicted":
            detail = "Claim conflicted with outside intake evidence."
        elif status == "mixed":
            detail = "Outside intake contained mixed evidence."
        elif status == "insufficient":
            detail = "Outside intake did not provide enough evidence."
        else:
            detail = str(evaluated.get("note", "Verification completed.") or "Verification completed.")

        matches = self._normalize_snippet_list(evaluated.get("matches", []) or [])
        conflicts = self._normalize_snippet_list(evaluated.get("conflicts", []) or [])

        return {
            "ok": True,
            "claim": claim,
            "status": status,
            "verdict": verdict,
            "confidence": round(confidence, 4),
            "external_verified": status in {"supported", "contradicted", "mixed", "insufficient"},
            "external_origin": source_name,
            "external_detail": detail,
            "matches": matches,
            "conflicts": conflicts,
            "support_score": float(evaluated.get("support_score", 0.0) or 0.0),
            "conflict_score": float(evaluated.get("conflict_score", 0.0) or 0.0),
            "packet": self._packet_summary(normalized_packet),
        }

    def verify_and_build_feedback(
        self,
        packet: Dict[str, Any],
        *,
        intake_text: str = "",
        source_name: str = "intake_text",
        force: bool = False,
    ) -> Dict[str, Any]:
        result = self.verify(
            packet,
            intake_text=intake_text,
            source_name=source_name,
            force=force,
        )

        if not result.get("ok", False):
            return {
                "ok": False,
                "verification_result": result,
                "feedback": {
                    "verdict": "unsure",
                    "external_verified": False,
                    "external_origin": source_name,
                    "external_detail": str(result.get("error", "verification failed") or "verification failed"),
                },
            }

        verdict = self._normalize_feedback_verdict(result.get("verdict", "unsure"))
        detail = str(result.get("external_detail", "") or "").strip()

        feedback: Dict[str, Any] = {
            "verdict": verdict,
            "external_verified": bool(result.get("external_verified", False)),
            "external_origin": result.get("external_origin", source_name),
            "external_detail": detail,
            "verification_status": result.get("status", "insufficient"),
            "confidence": float(result.get("confidence", 0.0) or 0.0),
        }

        matches = result.get("matches", []) or []
        conflicts = result.get("conflicts", []) or []

        if verdict == "confirm" and matches:
            txt = self._snippet_text(matches[0])
            if txt:
                feedback["notes"] = f"External support evidence: {txt}"
                feedback["support_evidence"] = txt

        if verdict == "deny" and conflicts:
            txt = self._snippet_text(conflicts[0])
            if txt:
                feedback["notes"] = f"External conflict evidence: {txt}"
                feedback["conflict_evidence"] = txt

        if result.get("status") == "mixed":
            feedback["notes"] = feedback.get("notes") or "External intake produced mixed evidence."

        return {
            "ok": True,
            "verification_result": result,
            "feedback": feedback,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _normalize_packet(self, packet: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(packet)
        details = self._details(packet)
        signals = packet.get("signals") if isinstance(packet.get("signals"), dict) else {}
        raw = packet.get("raw_input") if isinstance(packet.get("raw_input"), dict) else {}

        claim = self._extract_claim_text(packet)
        normalized["claim"] = claim
        normalized.setdefault("content", claim)

        if "verification_status" not in normalized or normalized.get("verification_status") in (None, ""):
            normalized["verification_status"] = (
                packet.get("status")
                or details.get("verification_status")
                or details.get("status")
                or "unverified"
            )

        if "memory_class" not in normalized or normalized.get("memory_class") in (None, ""):
            normalized["memory_class"] = details.get("memory_class")

        if "staleness_risk" not in normalized or normalized.get("staleness_risk") in (None, ""):
            normalized["staleness_risk"] = details.get("staleness_risk")

        if "source_count" not in normalized or normalized.get("source_count") is None:
            normalized["source_count"] = self._source_count(packet)

        if "hypothesis" not in normalized:
            normalized["hypothesis"] = bool(details.get("hypothesis", False))

        if "conflict_with_ids" not in normalized:
            normalized["conflict_with_ids"] = details.get("conflict_with_ids", []) or []

        if "memory_id" not in normalized or normalized.get("memory_id") in (None, ""):
            normalized["memory_id"] = packet.get("memory_id") or packet.get("id")

        if "route_target" not in normalized or normalized.get("route_target") in (None, ""):
            normalized["route_target"] = signals.get("route_target") or raw.get("route_target")

        if "source" not in normalized or normalized.get("source") in (None, ""):
            normalized["source"] = packet.get("source") or raw.get("source") or details.get("source")

        if "domain" not in normalized or normalized.get("domain") in (None, ""):
            normalized["domain"] = packet.get("domain") or signals.get("domain") or details.get("domain")

        return normalized

    def _extract_claim_text(self, packet: Dict[str, Any]) -> str:
        claim = str(
            packet.get("claim")
            or packet.get("content")
            or packet.get("text")
            or ""
        ).strip()
        if claim:
            return claim

        raw = packet.get("raw_input")
        if isinstance(raw, dict):
            return str(raw.get("content", "") or "").strip()

        signals = packet.get("signals")
        if isinstance(signals, dict):
            return str(signals.get("normalized_text", "") or "").strip()

        return ""

    def _packet_summary(self, packet: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "memory_id": packet.get("memory_id"),
            "memory_class": packet.get("memory_class"),
            "verification_status": packet.get("verification_status"),
            "staleness_risk": packet.get("staleness_risk"),
            "source_count": packet.get("source_count"),
            "hypothesis": packet.get("hypothesis"),
            "route_target": packet.get("route_target"),
            "domain": packet.get("domain"),
            "source": packet.get("source"),
        }

    def _details(self, packet: Dict[str, Any]) -> Dict[str, Any]:
        details = packet.get("details")
        return details if isinstance(details, dict) else {}

    def _source_count(self, packet: Dict[str, Any]) -> int:
        if packet.get("source_count") is not None:
            try:
                return max(0, int(packet.get("source_count", 0) or 0))
            except Exception:
                return 0

        sources = packet.get("sources")
        if isinstance(sources, (list, tuple, set)):
            return len(sources)

        evidence = packet.get("evidence")
        if isinstance(evidence, dict):
            items = evidence.get("items")
            if isinstance(items, list):
                return len(items)
        elif isinstance(evidence, list):
            return len(evidence)

        details = self._details(packet)
        d_evidence = details.get("evidence")
        if isinstance(d_evidence, list):
            return len(d_evidence)

        return 0

    def _staleness_value(self, packet: Dict[str, Any]) -> float:
        raw = packet.get("staleness_risk", None)
        if raw is None:
            raw = self._details(packet).get("staleness_risk", None)

        if isinstance(raw, (int, float)):
            return float(raw)

        s = self._safe_lower(raw)
        return {
            "low": 0.20,
            "medium": 0.50,
            "high": 0.85,
        }.get(s, 0.0)

    def _normalize_snippet_list(self, items: List[Any]) -> List[Any]:
        out: List[Any] = []
        for item in items:
            if isinstance(item, dict):
                text = self._snippet_text(item)
                if text:
                    row = dict(item)
                    row.setdefault("text", text)
                    out.append(row)
            else:
                s = str(item or "").strip()
                if s:
                    out.append({"text": s})
        return out

    def _snippet_text(self, item: Any) -> str:
        if isinstance(item, dict):
            return str(item.get("text", "") or item.get("snippet", "") or "").strip()
        return str(item or "").strip()

    def _normalize_verdict(self, verdict: Any) -> str:
        v = self._safe_lower(verdict)
        if v in {"confirm", "support", "supported"}:
            return "confirm"
        if v in {"deny", "contradict", "contradicted"}:
            return "deny"
        return "unsure"

    def _normalize_feedback_verdict(self, verdict: Any) -> str:
        v = self._safe_lower(verdict)
        if v in {"confirm", "support", "supported"}:
            return "confirm"
        if v in {"deny", "contradict", "contradicted"}:
            return "deny"
        if v in {"modify", "merge"}:
            return v
        return "unsure"

    def _safe_lower(self, value: Any) -> str:
        try:
            return str(value).strip().lower()
        except Exception:
            return ""

    def _clamp01(self, value: Any, default: float = 0.0) -> float:
        try:
            v = float(value)
        except Exception:
            v = float(default)
        return max(0.0, min(1.0, v))