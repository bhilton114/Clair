# comms/broca.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from comms.dialogue_state import DialogueState


@dataclass
class BrocaConfig:
    show_reasoning: bool = False
    bullet_style: str = "-"
    low_risk: float = 0.25
    med_risk: float = 0.60


class Broca:
    """
    Converts internal decisions into structured, human-readable language.
    Does NOT decide. Only formulates expression.
    """

    def __init__(self, config: Optional[BrocaConfig] = None):
        self.cfg = config or BrocaConfig()

    def formulate(
        self,
        decision: Dict[str, Any],
        dialogue: DialogueState,
        *,
        fallback_text: str = "Insufficient data to provide a safe response."
    ) -> Tuple[str, Dict[str, Any]]:

        chosen = decision.get("chosen")
        if not chosen:
            return fallback_text, {"ok": False, "reason": "no_decision"}

        name = chosen.get("name", "unknown_action")
        opt_type = chosen.get("type", "unknown_type")
        risk = float(chosen.get("risk", chosen.get("details", {}).get("risk", 0.5)))
        details = chosen.get("details", {}) or {}
        context = chosen.get("context", {}) or {}

        p_success = details.get("p_success", None)
        expected_utility = details.get("expected_utility", None)
        worst_case = details.get("worst_case_utility", None)

        verbosity = dialogue.verbosity or "normal"
        emotional = dialogue.emotional_load or "low"

        lines = []

        if emotional == "high":
            lines.append("Clear response based on current evaluation:")

        lines.append(f"Recommended action: {self._pretty(name)}")

        if verbosity == "short":
            reason = self._reason(details, context, opt_type)
            if reason:
                lines.append(f"Reason: {reason}")
            return "\n".join(lines), {
                "ok": True,
                "risk": risk,
                "p_success": p_success
            }

        lines.append("")
        lines.append("Summary:")
        lines.extend(self._summary(risk, p_success))

        reason = self._reason(details, context, opt_type)
        if reason:
            lines.append(f"{self.cfg.bullet_style} Why: {reason}")

        sim_lines = self._sim(expected_utility, worst_case, p_success)
        if sim_lines and verbosity in ("normal", "detailed"):
            lines.append(f"{self.cfg.bullet_style} Simulation:")
            lines.extend([f"  {self.cfg.bullet_style} {x}" for x in sim_lines])

        planned_next = details.get("planned_next")
        if verbosity == "detailed" and planned_next:
            lines.append(f"{self.cfg.bullet_style} Next step: {planned_next}")

        if self.cfg.show_reasoning:
            reasoning = decision.get("reasoning")
            if reasoning:
                lines.append("")
                lines.append("Reasoning (brief):")
                lines.append(reasoning.strip()[:600])

        meta = {
            "ok": True,
            "risk": risk,
            "p_success": p_success,
            "action_name": name,
            "type": opt_type,
        }

        return "\n".join(lines).strip(), meta

    def _pretty(self, name: str) -> str:
        return name.replace("_", " ").strip()

    def _risk_label(self, risk: float) -> str:
        if risk <= self.cfg.low_risk:
            return "low"
        if risk <= self.cfg.med_risk:
            return "medium"
        return "high"

    def _summary(self, risk: float, p_success: Any) -> list[str]:
        label = self._risk_label(risk)
        out = [f"{self.cfg.bullet_style} Risk: {label} ({risk:.2f})"]
        if p_success is not None:
            out.append(f"{self.cfg.bullet_style} Estimated success: {p_success}")
        return out

    def _sim(self, expected_utility: Any, worst_case: Any, p_success: Any) -> list[str]:
        lines = []
        if expected_utility is not None:
            lines.append(f"Expected utility: {expected_utility}")
        if worst_case is not None:
            lines.append(f"Worst-case utility: {worst_case}")
        if p_success is not None:
            lines.append(f"p_success: {p_success}")
        return lines

    def _reason(self, details: Dict[str, Any], context: Dict[str, Any], opt_type: str) -> Optional[str]:
        for k in ("rationale", "because", "why"):
            if k in details and details[k]:
                return str(details[k])

        seed = details.get("seed_memory")
        if seed:
            return f"Relevant memory: {seed}"

        hazard = context.get("hazard")
        if hazard:
            return f"Context hazard: {hazard}"

        if opt_type:
            return f"Action type: {opt_type}"
        return None