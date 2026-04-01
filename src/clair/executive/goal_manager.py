from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class Goal:
    """
    A durable objective that can bias decision-making over time.
    Weight is 0.0 - 1.0. Higher means more important overall.
    """
    name: str
    weight: float = 0.5
    meta: Dict[str, Any] = field(default_factory=dict)


class GoalManager:
    """
    Stores and manages Clair's current goals.

    Design:
    - Goals are slow-changing (compared to priority).
    - This manager does NOT make decisions. It provides bias signals.
    """

    def __init__(self, initial_goals: Optional[Dict[str, Goal]] = None):
        self._goals: Dict[str, Goal] = initial_goals or {}
        if not self._goals:
            # sensible defaults for Clair v2
            self._goals = {
                "stability": Goal("stability", 0.65, {"desc": "avoid chaos / reduce harmful variance"}),
                "learning": Goal("learning", 0.45, {"desc": "increase knowledge / reduce uncertainty"}),
                "help_human": Goal("help_human", 0.75, {"desc": "assist companion safely and effectively"}),
            }

    def set_goal(self, name: str, weight: float, meta: Optional[Dict[str, Any]] = None) -> None:
        weight = float(max(0.0, min(1.0, weight)))
        if name in self._goals:
            self._goals[name].weight = weight
            if meta:
                self._goals[name].meta.update(meta)
        else:
            self._goals[name] = Goal(name=name, weight=weight, meta=meta or {})

    def bump_goal(self, name: str, delta: float) -> None:
        g = self._goals.get(name)
        if not g:
            self._goals[name] = Goal(name=name, weight=float(max(0.0, min(1.0, delta))))
            return
        g.weight = float(max(0.0, min(1.0, g.weight + delta)))

    def get_weights(self) -> Dict[str, float]:
        return {k: v.weight for k, v in self._goals.items()}

    def get_goal(self, name: str) -> Optional[Goal]:
        return self._goals.get(name)

    def as_dict(self) -> Dict[str, Any]:
        return {k: {"weight": v.weight, "meta": dict(v.meta)} for k, v in self._goals.items()}