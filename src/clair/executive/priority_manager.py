from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Iterable


@dataclass
class PrioritySignal:
    """
    Dynamic weighting for the current cycle.
    Multipliers are applied to option utility (NOT risk).
    """
    stability: float = 1.0
    learning: float = 1.0
    help_human: float = 1.0
    explore: float = 1.0
    correct: float = 1.0


class PriorityManager:
    """
    Computes per-option priority multipliers.

    Inputs:
    - mode: Clair mode string (e.g., 'survival', 'learning', 'default')
    - system_state: optional dict (confidence, overload, etc.)
    - goal_weights: from GoalManager (slow-changing)

    Output:
    - multiplier float (default ~1.0)
    """

    def __init__(self, clamp_min: float = 0.50, clamp_max: float = 1.50):
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

        # Safety: if someone flips these by mistake, normalize them.
        if self.clamp_max < self.clamp_min:
            self.clamp_min, self.clamp_max = self.clamp_max, self.clamp_min

    # -------------------------
    # tiny helpers (defensive)
    # -------------------------
    def _clamp(self, x: Any) -> float:
        try:
            v = float(x)
        except Exception:
            v = 1.0
        return float(max(self.clamp_min, min(self.clamp_max, v)))

    def _safe_float(self, x: Any, default: float) -> float:
        try:
            return float(x)
        except Exception:
            return float(default)

    def _safe_str(self, x: Any) -> str:
        try:
            return str(x) if x is not None else ""
        except Exception:
            return ""

    def _safe_dict(self, x: Any) -> Dict[str, Any]:
        return x if isinstance(x, dict) else {}

    def _iter_context(self, ctx: Any) -> Iterable[Any]:
        """
        Simulator uses list context; but handle dict/str/None safely.
        """
        if ctx is None:
            return []
        if isinstance(ctx, (list, tuple)):
            return ctx
        # if someone passes a single dict or string, wrap it
        return [ctx]

    def _extract_target(self, option: Dict[str, Any], details: Dict[str, Any]) -> Optional[str]:
        """
        Try several locations without assuming shape.
        Returns a normalized target string or None.
        """
        # 1) details target
        t = details.get("target")
        if isinstance(t, str) and t.strip():
            return t.strip().lower()

        # 2) option-level target (if ever used)
        t = option.get("target")
        if isinstance(t, str) and t.strip():
            return t.strip().lower()

        # 3) context objects
        ctx = option.get("context")
        for item in self._iter_context(ctx):
            if isinstance(item, dict):
                t2 = item.get("target")
                if isinstance(t2, str) and t2.strip():
                    return t2.strip().lower()

        return None

    # -------------------------
    # core logic
    # -------------------------
    def build_signals(
        self,
        mode: str,
        goal_weights: Dict[str, float],
        system_state: Optional[Dict[str, Any]] = None,
    ) -> PrioritySignal:
        system_state = system_state or {}
        g = goal_weights or {}

        # start from goals (slow bias)
        sig = PrioritySignal(
            stability=1.0 + (self._safe_float(g.get("stability", 0.5), 0.5) - 0.5) * 0.8,
            learning=1.0 + (self._safe_float(g.get("learning", 0.5), 0.5) - 0.5) * 0.8,
            help_human=1.0 + (self._safe_float(g.get("help_human", 0.5), 0.5) - 0.5) * 0.8,
        )

        # mode is a fast bias
        m = self._safe_str(mode).lower().strip()
        if m == "survival":
            sig.stability *= 1.25
            sig.explore *= 0.70
            sig.learning *= 0.85
            sig.correct *= 1.10
        elif m == "learning":
            sig.learning *= 1.25
            sig.explore *= 1.15
            sig.stability *= 0.95
        # else: "default" or unknown: no change

        # system state can shift priorities
        overload = self._safe_float(system_state.get("overload", 0.0), 0.0)   # 0..1
        confidence = self._safe_float(system_state.get("confidence", 0.5), 0.5)  # 0..1

        if overload > 0.5:
            sig.stability *= 1.15
            sig.explore *= 0.85

        if confidence < 0.35:
            sig.correct *= 1.10
            sig.explore *= 0.90

        # clamp to sane range
        sig.stability = self._clamp(sig.stability)
        sig.learning = self._clamp(sig.learning)
        sig.help_human = self._clamp(sig.help_human)
        sig.explore = self._clamp(sig.explore)
        sig.correct = self._clamp(sig.correct)
        return sig

    def option_multiplier(
        self,
        option: Dict[str, Any],
        mode: str,
        goal_weights: Dict[str, float],
        system_state: Optional[Dict[str, Any]] = None,
    ) -> float:
        option = option if isinstance(option, dict) else {}
        details = self._safe_dict(option.get("details"))

        sig = self.build_signals(mode=mode, goal_weights=goal_weights, system_state=system_state)

        otype = self._safe_str(option.get("type")).lower()
        name = self._safe_str(option.get("name")).lower()

        mult = 1.0

        # crude-but-effective tagging using existing fields
        if "explore" in name or "explor" in name:
            mult *= sig.explore
        if "fix" in name or "repair" in name or "correct" in name:
            mult *= sig.correct

        # type-based routing
        if "survival" in otype:
            mult *= sig.stability
        if "learning" in otype or "reason" in otype:
            mult *= sig.learning

        # target routing (handles list/dict/None safely)
        target = self._extract_target(option, details)
        if target == "human":
            mult *= sig.help_human

        return self._clamp(mult)