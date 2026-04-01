# FILE: decision/validator.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

import config


class DecisionValidator:
    """
    Prefrontal validator:
    - normalizes candidate options
    - optionally asks risk_assessor for missing risk
    - rejects options above configured risk threshold
    - can lightly prefer memory-grounded options when metadata is present
    """

    VERSION = "2.0-validator"

    def validate(
        self,
        options: Any,
        working_memory: Optional[Any] = None,
        risk_assessor: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        if getattr(config, "VERBOSE", False):
            print("[DecisionValidator] Validating options...")

        if not isinstance(options, list):
            return []

        validated: List[Dict[str, Any]] = []
        max_risk = float(getattr(config, "VALIDATOR_MAX_RISK", 0.7))
        prefer_memory_grounding = bool(getattr(config, "VALIDATOR_PREFER_MEMORY_GROUNDED", True))

        for opt in options:
            if not isinstance(opt, dict):
                continue

            candidate = dict(opt)

            if "risk" not in candidate and risk_assessor is not None:
                try:
                    assessed = risk_assessor.assess(candidate)
                    if isinstance(assessed, dict):
                        candidate = assessed
                except Exception:
                    pass

            risk_value = self._safe_float(candidate.get("risk", 1.0), default=1.0)
            if risk_value > max_risk:
                if getattr(config, "VERBOSE", False):
                    name = candidate.get("name", "UNKNOWN")
                    print(f"[DecisionValidator] Option {name} rejected due to high risk ({risk_value:.3f})")
                continue

            if prefer_memory_grounding:
                grounded = bool(candidate.get("memory_grounded", False))
                evidence = candidate.get("evidence") or candidate.get("supporting_memories") or []
                if grounded or (isinstance(evidence, list) and evidence):
                    candidate["validator_bonus"] = 0.10
                else:
                    candidate["validator_bonus"] = 0.0

            candidate["validated"] = True
            candidate["risk"] = risk_value
            validated.append(candidate)

        validated.sort(
            key=lambda x: (
                float(x.get("validator_bonus", 0.0) or 0.0),
                -float(x.get("risk", 1.0) or 1.0),
                float(x.get("score", 0.0) or 0.0),
            ),
            reverse=True,
        )
        return validated

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)