# FILE: affect/hypothalamus.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Dict, Any

class Mode(str, Enum):
    CURIOSITY = "curiosity"
    JUDGING = "judging"
    PREDICTING = "predicting"
    IDLING = "idling"
    RUSH = "rush"

def clamp01(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

@dataclass
class HypothalamusConfig:
    cooldown_sec: float = 2.0
    hysteresis_margin: float = 0.08

    rush_urgency: float = 0.75
    rush_risk: float = 0.70

    idle_low_activity: float = 0.20
    idle_fatigue: float = 0.70

    judging_uncertainty: float = 0.65
    judging_risk: float = 0.55

    predicting_goal_pressure: float = 0.55
    predicting_uncertainty: float = 0.35

    curiosity_novelty: float = 0.55
    curiosity_risk_max: float = 0.60

    # Bias profiles: each must include keys used below (see get_simulator_overrides)
    biases: Dict[Mode, Dict[str, float]] = field(default_factory=lambda: {
        Mode.CURIOSITY: {
            "exploration_rate": 0.35,
            "sim_horizon": 1.0,
            "sim_rollouts": 0.50,
            "risk_tolerance": 0.65,
            "learn_threshold": 0.35,
            "evidence_threshold": 0.45,
            "action_bias": 0.45,
            "deterministic_planning": 0.40,
        },
        Mode.JUDGING: {
            "exploration_rate": 0.10,
            "sim_horizon": 1.0,
            "sim_rollouts": 0.60,
            "risk_tolerance": 0.35,
            "learn_threshold": 0.65,
            "evidence_threshold": 0.75,
            "action_bias": 0.35,
            "deterministic_planning": 1.00,
        },
        Mode.PREDICTING: {
            "exploration_rate": 0.15,
            "sim_horizon": 2.0,
            "sim_rollouts": 0.85,
            "risk_tolerance": 0.45,
            "learn_threshold": 0.55,
            "evidence_threshold": 0.60,
            "action_bias": 0.40,
            "deterministic_planning": 1.00,
        },
        Mode.IDLING: {
            "exploration_rate": 0.05,
            "sim_horizon": 1.0,
            "sim_rollouts": 0.20,
            "risk_tolerance": 0.50,
            "learn_threshold": 0.80,
            "evidence_threshold": 0.70,
            "action_bias": 0.20,
            "deterministic_planning": 0.90,
        },
        Mode.RUSH: {
            "exploration_rate": 0.08,
            "sim_horizon": 1.0,
            "sim_rollouts": 0.25,
            "risk_tolerance": 0.55,
            "learn_threshold": 0.70,
            "evidence_threshold": 0.50,
            "action_bias": 0.80,
            "deterministic_planning": 0.95,
        },
    })

class Hypothalamus:
    """
    Mode selector and supplier of simulator overrides.

    Use:
      hyp = Hypothalamus()
      mode = hyp.choose_mode(signals)
      overrides = hyp.get_simulator_overrides()
      hyp.apply_to_simulator(sim)   # convenience
    """

    def __init__(self, config: HypothalamusConfig | None = None):
        self.cfg = config or HypothalamusConfig()
        self.mode: Mode = Mode.IDLING
        self._last_switch_ts: float = 0.0

    def _score_modes(self, s: Dict[str, float]) -> Dict[Mode, float]:
        risk = s["risk"]
        unc = s["uncertainty"]
        nov = s["novelty"]
        urg = s["urgency"]
        goal = s["goal_pressure"]
        fat = s["fatigue"]
        act = s["activity"]

        scores: Dict[Mode, float] = {
            Mode.RUSH: (
                0.55 * urg +
                0.25 * risk +
                0.10 * goal -
                0.15 * unc
            ),
            Mode.JUDGING: (
                0.40 * unc +
                0.30 * risk +
                0.15 * goal -
                0.15 * urg
            ),
            Mode.PREDICTING: (
                0.45 * goal +
                0.25 * (1.0 - urg) +
                0.15 * (1.0 - fat) +
                0.15 * (1.0 - abs(unc - 0.45))
            ),
            Mode.CURIOSITY: (
                0.50 * nov +
                0.20 * (1.0 - risk) +
                0.15 * (1.0 - urg) +
                0.15 * (1.0 - fat)
            ),
            Mode.IDLING: (
                0.45 * (1.0 - act) +
                0.25 * fat +
                0.20 * (1.0 - goal) +
                0.10 * (1.0 - urg)
            ),
        }
        return scores

    def choose_mode(self, signals: Dict[str, Any]) -> Mode:
        now = time.time()

        risk = clamp01(signals.get("risk", 0.0))
        uncertainty = clamp01(signals.get("uncertainty", 0.0))
        novelty = clamp01(signals.get("novelty", 0.0))
        urgency = clamp01(signals.get("urgency", 0.0))
        goal_pressure = clamp01(signals.get("goal_pressure", 0.0))
        fatigue = clamp01(signals.get("fatigue", 0.0))

        activity = clamp01(signals.get("activity", (0.40 * urgency + 0.35 * novelty + 0.25 * goal_pressure)))

        s = {
            "risk": risk,
            "uncertainty": uncertainty,
            "novelty": novelty,
            "urgency": urgency,
            "goal_pressure": goal_pressure,
            "fatigue": fatigue,
            "activity": activity,
        }

        # Hard gates
        if urgency >= self.cfg.rush_urgency and (risk >= self.cfg.rush_risk or goal_pressure >= 0.75):
            candidate = Mode.RUSH
        elif activity <= self.cfg.idle_low_activity or fatigue >= self.cfg.idle_fatigue:
            candidate = Mode.IDLING
        elif uncertainty >= self.cfg.judging_uncertainty and risk >= self.cfg.judging_risk:
            candidate = Mode.JUDGING
        elif goal_pressure >= self.cfg.predicting_goal_pressure and urgency <= 0.60:
            candidate = Mode.PREDICTING
        elif novelty >= self.cfg.curiosity_novelty and risk <= self.cfg.curiosity_risk_max and urgency <= 0.65:
            candidate = Mode.CURIOSITY
        else:
            scores = self._score_modes(s)
            candidate = max(scores, key=scores.get)

        # Cooldown + hysteresis
        if candidate != self.mode:
            if (now - self._last_switch_ts) < self.cfg.cooldown_sec:
                return self.mode

            scores = self._score_modes(s)
            if scores.get(candidate, 0.0) < scores.get(self.mode, 0.0) + self.cfg.hysteresis_margin:
                return self.mode

            self.mode = candidate
            self._last_switch_ts = now

        return self.mode

    def get_biases(self) -> Dict[str, float]:
        """Return the parameter biases for the current mode."""
        return dict(self.cfg.biases[self.mode])

    # ------------------------------
    # Simulator integration helpers
    # ------------------------------
    def get_simulator_overrides(self) -> Dict[str, Any]:
        """
        Map hypothalamus biases to simulator knobs.
        Returns a dict with:
          - 'horizon_hint'     : int (1 or 2)
          - 'rollouts_scale'   : float (multiplier)
          - 'exploration_rate' : float [0..1]
          - 'deterministic'    : bool
          - 'risk_tolerance'   : float  (informational)
        """
        b = self.get_biases()
        # safe reads with defaults
        sim_horizon = int(round(b.get("sim_horizon", 1)))
        sim_horizon = 1 if sim_horizon < 1 else (2 if sim_horizon > 2 else sim_horizon)

        rollouts_scale = float(b.get("sim_rollouts", 0.5))
        # scale mapping: bias is relative; clamp to useful bounds
        rollouts_scale = max(0.2, min(2.0, rollouts_scale * 1.4))  # mild amplification

        exploration_rate = float(b.get("exploration_rate", 0.2))
        exploration_rate = max(0.0, min(0.75, exploration_rate))

        deterministic = float(b.get("deterministic_planning", 0.8)) >= 0.5

        risk_tolerance = float(b.get("risk_tolerance", 0.5))
        risk_tolerance = max(0.0, min(1.0, risk_tolerance))

        # Compose overrides
        return {
            "horizon_hint": sim_horizon,
            "rollouts_scale": round(rollouts_scale, 3),
            "exploration_rate": round(exploration_rate, 3),
            "deterministic": deterministic,
            "risk_tolerance": round(risk_tolerance, 3),
        }

    def apply_to_simulator(self, simulator) -> None:
        """
        Convenience: apply the current hypothalamus-derived overrides to a simulator instance.
        Only writes a small, well-defined set of attributes so it is non-destructive.
        """
        if simulator is None:
            return
        overrides = self.get_simulator_overrides()

        # apply minimally and safely
        if "horizon_hint" in overrides:
            try:
                simulator.horizon_hint = int(overrides["horizon_hint"])
            except Exception:
                pass
        if "rollouts_scale" in overrides:
            try:
                simulator.rollouts_scale = float(overrides["rollouts_scale"])
            except Exception:
                pass
        if "exploration_rate" in overrides:
            try:
                simulator.exploration_rate = float(overrides["exploration_rate"])
            except Exception:
                pass
        if "deterministic" in overrides:
            try:
                # simulator should expose a deterministic toggle (see simulator rewrite)
                if hasattr(simulator, "deterministic"):
                    simulator.deterministic = bool(overrides["deterministic"])
                else:
                    # fallback: set an attribute (harmless) so future code can inspect it
                    setattr(simulator, "deterministic", bool(overrides["deterministic"]))
            except Exception:
                pass

        # optional: expose risk_tolerance as informational variable
        try:
            setattr(simulator, "hypothalamus_risk_tolerance", overrides.get("risk_tolerance"))
        except Exception:
            pass