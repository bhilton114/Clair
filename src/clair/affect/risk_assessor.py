# FILE: affect/risk_assessor.py
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Union

import config


class RiskAssessor:
    """
    Risk Assessor (v2.5) – contract-tight, side-effect free

    Purpose:
      Assign a risk score (0.0–1.0) to an option/action dict.

    Strict contract:
      - Input: option/action dict (may be malformed)
      - Output:
          * float risk in [0, 1]
            OR
          * dict {"risk": float, "reasons": List[str]} (optional)
      - Does NOT mutate the input dict.
      - Never throws.

    Notes:
      - The Simulator is responsible for writing option["risk"] and details["risk"].
      - This module focuses on computing risk from text evidence.
    """

    DEFAULT_RISK = 0.25

    HIGH_RISK_TERMS = {
        "kill", "harm", "attack", "weapon", "explosive", "suicide", "self harm", "bomb", "poison",
        "steal", "fraud", "hack", "malware", "phish", "dox", "extort",
        # survival hazards
        "fire", "flood", "earthquake", "emergency",
    }

    MED_RISK_TERMS = {
        "danger", "risk", "injury", "blood", "fight", "break", "illegal", "crime",
        "shutdown", "delete", "wipe", "format",
    }

    def assess(self, action: Dict[str, Any]) -> Union[float, Dict[str, Any]]:
        """
        Returns either:
          - float risk
          - {"risk": float, "reasons": [...]}
        """
        try:
            blob = self._action_text_blob(action)
            risk, reasons = self._heuristic_risk(blob)
            risk = self._clamp(risk)

            # optional observability
            if getattr(config, "VERBOSE", False):
                name = "UNKNOWN"
                if isinstance(action, dict):
                    name = str(action.get("name", "UNKNOWN"))
                print(f"[RiskAssessor] Assessed risk for '{name}': {risk}")
                if getattr(config, "RISK_DEBUG", False):
                    print(f"[RiskAssessor] blob='{blob}'")
                    if reasons:
                        print(f"[RiskAssessor] reasons={reasons}")

            return {"risk": risk, "reasons": reasons}
        except Exception:
            # never throw
            return float(self.DEFAULT_RISK)

    # -----------------------------
    # Internals
    # -----------------------------
    def _clamp(self, x: Any) -> float:
        try:
            v = float(x)
        except Exception:
            v = float(self.DEFAULT_RISK)
        if v < 0.0:
            return 0.0
        if v > 1.0:
            return 1.0
        return v

    def _action_text_blob(self, action: Any) -> str:
        """
        Build a lowercase text blob from useful string-ish fields.
        Defensive: never throws, never assumes schema.
        """
        if not isinstance(action, dict):
            return ""

        parts: List[str] = []

        # Common top-level fields
        for k in ("name", "type", "prompt", "question", "scenario", "content"):
            v = action.get(k)
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())

        # Context list
        ctx = action.get("context")
        if isinstance(ctx, list):
            for c in ctx:
                if isinstance(c, str) and c.strip():
                    parts.append(c.strip())
                elif c is not None:
                    parts.append(str(c))

        # Details dict: include seed_text/seed_memory first, then other strings
        details = action.get("details")
        if isinstance(details, dict):
            for k in ("seed_text", "seed_memory"):
                v = details.get(k)
                if isinstance(v, str) and v.strip():
                    parts.append(v.strip())

            for k, v in details.items():
                if k in ("seed_text", "seed_memory"):
                    continue
                if isinstance(v, str) and v.strip():
                    parts.append(v.strip())
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, str) and item.strip():
                            parts.append(item.strip())
                        elif item is not None:
                            parts.append(str(item))

        return " ".join(parts).lower()

    def _heuristic_risk(self, text: str) -> (float, List[str]):
        if not text:
            return float(self.DEFAULT_RISK), []

        tokens = set(re.findall(r"[a-z0-9_]+", text))

        high_hits: List[str] = []
        med_hits: List[str] = []

        for term in self.HIGH_RISK_TERMS:
            if " " in term:
                if term in text:
                    high_hits.append(term)
            else:
                if term in tokens:
                    high_hits.append(term)

        for term in self.MED_RISK_TERMS:
            if " " in term:
                if term in text:
                    med_hits.append(term)
            else:
                if term in tokens:
                    med_hits.append(term)

        risk = float(self.DEFAULT_RISK)
        risk += 0.20 * min(len(high_hits), 2)   # up to +0.40
        risk += 0.08 * min(len(med_hits), 3)    # up to +0.24

        reasons: List[str] = []
        if high_hits:
            reasons.append(f"high_terms={sorted(set(high_hits))[:6]}")
        if med_hits:
            reasons.append(f"med_terms={sorted(set(med_hits))[:6]}")

        return self._clamp(risk), reasons