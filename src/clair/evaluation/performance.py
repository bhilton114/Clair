# evaluation/performance.py
from __future__ import annotations

from typing import Any, Dict, List, Optional


def _clamp01(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        v = float(default)
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _is_success_value(v: Any) -> Optional[bool]:
    """
    Normalize various 'success' representations.
    Returns True/False if recognizable, else None.
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"success", "ok", "passed", "done", "true", "yes"}:
            return True
        if s in {"failure", "fail", "failed", "no", "false"}:
            return False
    return None


class PerformanceEvaluator:
    """
    Evaluates executed actions and returns structured feedback.

    Supports:
      - NEW actuator contract:
        {
          "action": <dict>,
          "action_name": str,
          "action_type": str,
          "ok": bool,
          "outcome": "success"/"failure"/"noop"/...,
          "success_prob": float,
          "risk": float,
          "side_effect": str|None,
          "details": dict
        }

      - Legacy contract:
        {
          "action": <dict>,
          "result": "success"/"failure"|bool,
          "success_prob": float,
          "weight": float,
          "context": list,
          "side_effect": str|None
        }
    """

    def evaluate(self, results: List[dict]) -> List[Dict[str, Any]]:
        evaluations: List[Dict[str, Any]] = []

        for r in (results or []):
            if not isinstance(r, dict):
                continue

            # Action dict (always try to preserve it)
            action = r.get("action") if isinstance(r.get("action"), dict) else {}
            name = (
                str(r.get("action_name") or action.get("name") or "unknown")
                .strip()
            )
            a_type = str(r.get("action_type") or action.get("type") or "unknown").strip()

            # Context (prefer action context, but allow result context)
            context = r.get("context", None)
            if not isinstance(context, list):
                context = action.get("context", [])
            if not isinstance(context, list):
                context = []

            # Weight (prefer explicit weight field, fallback to action weight)
            weight = r.get("weight", action.get("weight", 0.5))
            weight = _as_float(weight, 0.5)

            # Risk (prefer result risk, fallback to action risk)
            risk = r.get("risk", action.get("risk", None))
            risk_f = None if risk is None else _clamp01(risk, 0.5)

            # Success probability (optional)
            success_prob = r.get("success_prob", None)
            sp_f = None if success_prob is None else _clamp01(success_prob, 0.5)

            # Side effect (optional)
            side_effect = r.get("side_effect", action.get("side_effect", None))
            if side_effect is not None:
                side_effect = str(side_effect).strip() or None

            # Determine success/outcome across both schemas
            ok = r.get("ok", None)
            ok_norm = _is_success_value(ok)

            outcome = r.get("outcome", None)
            outcome_str = str(outcome).strip().lower() if isinstance(outcome, str) else None

            legacy_result = r.get("result", None)
            legacy_norm = _is_success_value(legacy_result)

            success = None  # type: Optional[bool]
            if ok_norm is not None:
                success = ok_norm
            elif outcome_str in {"success", "failure"}:
                success = (outcome_str == "success")
            elif legacy_norm is not None:
                success = legacy_norm
            else:
                # If nothing explicit, guess from success_prob if present
                if sp_f is not None:
                    success = (sp_f >= 0.5)
                else:
                    success = False  # conservative default

            # Canonical outcome text
            if outcome_str:
                canonical_outcome = outcome_str
            else:
                canonical_outcome = "success" if success else "failure"

            # --- scoring heuristic ---
            # Base score from weight. If it's a reasoning action, failures matter less.
            base = weight
            failure_mult = 0.5
            if a_type == "reasoning_action":
                failure_mult = 0.8

            score = base if success else base * failure_mult

            # Risk penalty applies either way (you can succeed and still be reckless)
            if risk_f is not None:
                score *= max(0.0, 1.0 - 0.30 * risk_f)

            # Side effect penalty if present
            if side_effect:
                score *= 0.85

            # Slightly reward high explicit success_prob if it exists and succeeded
            if sp_f is not None and success:
                score *= (0.95 + 0.10 * sp_f)  # max 1.05-ish

            # Clamp score to sane bounds
            if score < 0.0:
                score = 0.0
            # (no hard upper clamp; weight can be >1 if you do that later)

            # --- confidence heuristic ---
            # Confidence is about observation quality, not the score itself.
            conf = 0.35
            if ok_norm is not None or outcome_str in {"success", "failure"}:
                conf += 0.25
            if sp_f is not None:
                conf += 0.15
            if risk_f is not None:
                conf += 0.10
            if isinstance(r.get("details"), dict) and r["details"]:
                conf += 0.10
            conf = _clamp01(conf, 0.45)

            notes_bits = []
            notes_bits.append(f"action='{name}' type={a_type}")
            notes_bits.append(f"outcome={canonical_outcome}")
            if sp_f is not None:
                notes_bits.append(f"p={sp_f:.2f}")
            if risk_f is not None:
                notes_bits.append(f"risk={risk_f:.2f}")
            if side_effect:
                notes_bits.append(f"side_effect={side_effect}")

            evaluations.append({
                "action": action,
                "action_name": name,
                "action_type": a_type,
                "score": round(float(score), 3),
                "confidence": round(float(conf), 3),
                "outcome": canonical_outcome,
                "success": bool(success),
                "context": context,
                "notes": " | ".join(notes_bits),
            })

        return evaluations


_global_evaluator = PerformanceEvaluator()


def evaluate_action(action: dict, result: dict) -> Dict[str, Any]:
    """
    Convenience wrapper for single action+result evaluation.
    Accepts either new or old actuator result schema.
    """
    combined = {"action": action if isinstance(action, dict) else {}}
    if isinstance(result, dict):
        combined.update(result)
    out = _global_evaluator.evaluate([combined])
    return out[0] if out else {
        "action": action,
        "action_name": (action or {}).get("name", "unknown"),
        "action_type": (action or {}).get("type", "unknown"),
        "score": 0.0,
        "confidence": 0.0,
        "outcome": "unknown",
        "success": False,
        "context": (action or {}).get("context", []),
        "notes": "Evaluation failed: invalid input",
    }