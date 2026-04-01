# FILE: execution/actuator.py
# Clair Actuator (v1.1) – stable action-preserving execution contract
#
# Fixes:
# - Preserves original action dict in result["action"] for ReflectionEngine
# - Keeps benchmark-friendly no-spew behavior
# - Treats reasoning_action as successful informational execution
# - Returns stable structured result schema for PerformanceEvaluator + ReflectionEngine

from __future__ import annotations

import copy
import random
import time
from typing import Any, Dict, List, Optional


class Actuator:
    """
    Clair Actuator (Motor Cortex)

    Executes chosen actions and returns a stable result contract.

    Goals:
    - No print spew
    - Compatible with simulator output
    - Preserve original action dict for downstream reflection/evaluation
    - Reasoning actions do not "fail" like physical actions
    """

    VERSION = "1.1"

    def __init__(self, verbose: bool = False, rng: Optional[random.Random] = None):
        self.verbose = bool(verbose)
        self._rng = rng or random.Random()

    # -------------------------
    # Public API
    # -------------------------
    def execute(self, actions: Any, system_state: Optional[dict] = None) -> List[Dict[str, Any]]:
        """
        Execute one action dict OR a list of action dicts.

        Returns list[dict], each with:
          - action: original action dict
          - ok: bool
          - outcome: str
          - details: dict
          - harm: bool
          - hazard_family: str|None
          - timestamp: float
          - latency_ms: int
          - batch_latency_ms: int
          - action_name: str
          - action_type: str
          - success_prob: float
          - risk: float
          - context: list
          - weight: float
        """
        start = time.time()
        acts = self._normalize_actions(actions)
        results: List[Dict[str, Any]] = []

        for action in acts:
            t0 = time.time()
            res = self._execute_one(action, system_state=system_state)
            res["timestamp"] = float(res.get("timestamp", time.time()))
            res["latency_ms"] = int(round((time.time() - t0) * 1000))
            results.append(res)

        if results:
            batch_ms = int(round((time.time() - start) * 1000))
            for r in results:
                r.setdefault("batch_latency_ms", batch_ms)

        return results

    # -------------------------
    # Internals
    # -------------------------
    def _normalize_actions(self, actions: Any) -> List[Dict[str, Any]]:
        if actions is None:
            return []
        if isinstance(actions, dict):
            return [actions]
        if isinstance(actions, list):
            return [a for a in actions if isinstance(a, dict)]
        return []

    def _safe_action_copy(self, action: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return copy.deepcopy(action)
        except Exception:
            return dict(action) if isinstance(action, dict) else {}

    def _execute_one(self, action: Dict[str, Any], system_state: Optional[dict] = None) -> Dict[str, Any]:
        action_copy = self._safe_action_copy(action)

        name = str(action.get("name", "unnamed_action"))
        a_type = str(action.get("type", "unknown"))
        details = action.get("details") if isinstance(action.get("details"), dict) else {}
        context = action.get("context") if isinstance(action.get("context"), list) else []
        weight = self._as_float(action.get("weight", 0.5), 0.5)

        rollouts = details.get("rollouts") if isinstance(details.get("rollouts"), list) else None

        hazard_family = details.get("hazard_family")
        if hazard_family is not None:
            hazard_family = str(hazard_family)

        risk = self._clamp01(action.get("risk", details.get("risk", 0.5)), default=0.5)

        # Reasoning actions are informational. They should commit cleanly.
        if a_type == "reasoning_action":
            return {
                "action": action_copy,
                "ok": True,
                "outcome": "completed_reasoning",
                "details": {
                    "risk": risk,
                    "side_effect": None,
                    "p_success": 1.0,
                    "note": "reasoning_action treated as non-failing execution",
                    "system_state": system_state if isinstance(system_state, dict) else None,
                },
                "harm": False,
                "hazard_family": hazard_family,
                "timestamp": time.time(),
                "action_name": name,
                "action_type": a_type,
                "success_prob": 1.0,
                "risk": risk,
                "context": context,
                "weight": weight,
            }

        p_success = self._estimate_success_prob(action, rollouts=rollouts)
        p_success = self._clamp(p_success * (1.0 - 0.30 * risk), 0.01, 0.99, default=0.5)

        success = self._rng.random() < p_success
        side_effect = None
        harm = False

        if success and risk > 0.7:
            if self._rng.random() < (risk - 0.7):
                side_effect = "high_risk_side_effect"
                harm = True

        outcome = "success" if success else "failure"

        return {
            "action": action_copy,
            "ok": bool(success),
            "outcome": outcome,
            "details": {
                "risk": risk,
                "side_effect": side_effect,
                "p_success": round(float(p_success), 3),
                "system_state": system_state if isinstance(system_state, dict) else None,
            },
            "harm": bool(harm),
            "hazard_family": hazard_family,
            "timestamp": time.time(),
            "action_name": name,
            "action_type": a_type,
            "success_prob": round(float(p_success), 3),
            "risk": risk,
            "context": context,
            "weight": weight,
        }

    def _estimate_success_prob(self, action: Dict[str, Any], rollouts: Optional[list] = None) -> float:
        if isinstance(rollouts, list) and rollouts:
            ps = []
            for r in rollouts:
                if isinstance(r, dict) and r.get("p_success") is not None:
                    try:
                        ps.append(float(r["p_success"]))
                    except Exception:
                        continue
            if ps:
                return sum(ps) / len(ps)

        w = self._as_float(action.get("weight", 0.5), 0.5)
        if w > 1.0:
            w = 1.0
        if w < -1.0:
            w = -1.0

        return 0.50 + 0.35 * w

    def _as_float(self, x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return float(default)

    def _clamp01(self, x: Any, default: float = 0.0) -> float:
        try:
            v = float(x)
        except Exception:
            v = float(default)
        if v < 0.0:
            return 0.0
        if v > 1.0:
            return 1.0
        return v

    def _clamp(self, x: Any, lo: float, hi: float, default: float = 0.0) -> float:
        try:
            v = float(x)
        except Exception:
            v = float(default)
        if v < lo:
            return lo
        if v > hi:
            return hi
        return v