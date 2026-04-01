# FILE: reflection/review.py
# v2.52 – reflection commit hardening + evaluator-optional commit path

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple


class ReflectionEngine:
    """
    Metacognition – commits executed action outcomes into working memory.

    Goals:
    - Preserve structured action outcome data in content_obj
    - Emit WM-safe human-readable content
    - Keep schema stable for working_memory normalization
    - Support both new and legacy actuator/evaluator contracts
    - Commit even when evaluator output is missing
    """

    VERSION = "2.52"

    DEFAULT_CONFIDENCE = 0.5
    DEFAULT_WEIGHT = 0.5

    # -------------------------
    # helpers
    # -------------------------

    def _clamp01(self, x: Any, default: float = 0.0) -> float:
        try:
            v = float(x)
        except (TypeError, ValueError):
            v = float(default)
        return max(0.0, min(1.0, v))

    def _flatten_context(self, context: Any) -> List[Any]:
        if context is None:
            return []
        if isinstance(context, list):
            out: List[Any] = []
            for item in context:
                out.extend(self._flatten_context(item))
            return out
        if isinstance(context, (str, int, float)):
            return [context]
        return []

    def _ensure_context_list(self, context: Any) -> List[Any]:
        if context is None:
            return []
        if isinstance(context, list):
            return self._flatten_context(context)
        return self._flatten_context([context])

    def _safe_str(self, x: Any, default: str = "") -> str:
        try:
            s = str(x).strip()
            return s if s else default
        except Exception:
            return default

    def _safe_dict(self, x: Any) -> Dict[str, Any]:
        return dict(x) if isinstance(x, dict) else {}

    def _safe_list_str(self, tags: Any) -> List[str]:
        if tags is None:
            return []
        if isinstance(tags, str):
            s = tags.strip().lower()
            return [s] if s else []
        if isinstance(tags, (list, tuple, set)):
            out: List[str] = []
            for item in tags:
                try:
                    s = str(item).strip().lower()
                except Exception:
                    continue
                if s:
                    out.append(s)
            return out
        try:
            s = str(tags).strip().lower()
            return [s] if s else []
        except Exception:
            return []

    def _resolve_weight_factor(
        self,
        context: Any,
        context_weights: Optional[Dict[Any, float]],
    ) -> float:
        if not context_weights:
            return 0.5
        flat = self._flatten_context(context)
        if not flat:
            return 0.5
        vals = [self._clamp01(context_weights.get(c, 0.5), default=0.5) for c in flat]
        return sum(vals) / len(vals) if vals else 0.5

    def _score_to_conf_weight(self, score: Any, outcome: str) -> Tuple[float, float]:
        try:
            s = float(score)
        except (TypeError, ValueError):
            s = 0.0

        if -1.0 <= s <= 1.0:
            norm = (s + 1.0) / 2.0 if s < 0.0 else s
        else:
            norm = 1.0 / (1.0 + abs(s)) if s != 0 else 0.0
            if s > 0:
                norm = 1.0 - norm

        norm = self._clamp01(norm, default=0.0)

        outcome_l = (outcome or "").lower()
        if outcome_l in {"success", "ok", "passed", "true", "completed_reasoning"}:
            conf = self._clamp01(norm + 0.05, default=self.DEFAULT_CONFIDENCE)
            weight = self._clamp01(norm + 0.05, default=self.DEFAULT_WEIGHT)
        elif outcome_l in {"failure", "failed", "false", "error"}:
            conf = self._clamp01(norm - 0.05, default=self.DEFAULT_CONFIDENCE)
            weight = self._clamp01(norm - 0.05, default=self.DEFAULT_WEIGHT)
        else:
            conf = self._clamp01(norm, default=self.DEFAULT_CONFIDENCE)
            weight = self._clamp01(norm, default=self.DEFAULT_WEIGHT)

        return conf, weight

    def _extract_action(self, result: Any, evaluation: Any) -> Dict[str, Any]:
        r = self._safe_dict(result)
        e = self._safe_dict(evaluation)

        a1 = r.get("action")
        if isinstance(a1, dict):
            return dict(a1)

        a2 = e.get("action")
        if isinstance(a2, dict):
            return dict(a2)

        d = e.get("details")
        if isinstance(d, dict) and isinstance(d.get("action"), dict):
            return dict(d["action"])

        # Reconstruct minimal action shape from flat result/eval fields
        name = self._safe_str(r.get("action_name") or e.get("action_name"), "")
        a_type = self._safe_str(r.get("action_type") or e.get("action_type"), "")
        context = r.get("context", e.get("context", []))
        if not isinstance(context, list):
            context = []

        details = r.get("details")
        if not isinstance(details, dict):
            details = {}

        tags = r.get("tags", e.get("tags", []))
        domain = r.get("domain", e.get("domain", None))

        if name or a_type or context or details:
            return {
                "name": name or "unnamed_action",
                "type": a_type or "unknown",
                "context": context,
                "details": details,
                "tags": tags,
                "domain": domain,
            }

        return {}

    def _extract_outcome(self, result: Any) -> str:
        r = self._safe_dict(result)
        raw = r.get("result", r.get("outcome", r.get("status", "unknown")))

        if isinstance(raw, bool):
            return "success" if raw else "failure"
        if isinstance(raw, str):
            s = raw.strip().lower()
            if s in {"success", "ok", "passed", "done", "true", "completed_reasoning"}:
                return s
            if s in {"fail", "failed", "failure", "error", "false"}:
                return "failure"
            return s or "unknown"
        return "unknown"

    def _extract_score(self, evaluation: Any, result: Any = None) -> float:
        e = self._safe_dict(evaluation)
        r = self._safe_dict(result)

        if "score" in e:
            try:
                return float(e.get("score", 0.0))
            except Exception:
                pass

        e_details = e.get("details")
        if isinstance(e_details, dict):
            try:
                return float(e_details.get("score", 0.0))
            except Exception:
                pass

        # Fallback to result details if evaluator gave nothing
        r_details = r.get("details")
        if isinstance(r_details, dict):
            for key in ("score", "expected_utility", "p_success", "success_prob"):
                if key in r_details:
                    try:
                        return float(r_details.get(key, 0.0))
                    except Exception:
                        continue

        if "success_prob" in r:
            try:
                return float(r.get("success_prob", 0.0))
            except Exception:
                pass

        return 0.0

    def _human_commit_content(self, action_name: str, outcome: str, action_type: str) -> str:
        safe_name = self._safe_str(action_name, "unnamed_action")
        safe_type = self._safe_str(action_type, "unknown")
        safe_outcome = self._safe_str(outcome, "unknown")
        return f"Committed action '{safe_name}' of type '{safe_type}' ended with outcome '{safe_outcome}'."

    def _fallback_evaluation(self, result: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a minimal evaluation record when PerformanceEvaluator returns nothing.
        """
        outcome = self._extract_outcome(result)
        success_prob = self._clamp01(result.get("success_prob", 0.5), default=0.5)
        weight = self._clamp01(action.get("weight", 0.5), default=0.5)
        risk = self._clamp01(result.get("risk", self._safe_dict(result.get("details")).get("risk", 0.5)), default=0.5)

        # Conservative fallback score
        score = weight
        if outcome in {"failure", "failed", "error"}:
            score *= 0.5
        elif outcome in {"completed_reasoning"}:
            score = max(score, 0.65)

        score *= max(0.0, 1.0 - 0.25 * risk)
        if outcome in {"success", "completed_reasoning"}:
            score *= (0.95 + 0.10 * success_prob)

        score = self._clamp01(score, default=0.5)

        return {
            "action": action,
            "action_name": self._safe_str(action.get("name") or result.get("action_name"), "unnamed_action"),
            "action_type": self._safe_str(action.get("type") or result.get("action_type"), "unknown"),
            "score": round(float(score), 3),
            "confidence": 0.55,
            "outcome": outcome,
            "success": outcome not in {"failure", "failed", "error"},
            "context": self._ensure_context_list(action.get("context", result.get("context", []))),
            "notes": "fallback evaluation generated by ReflectionEngine",
        }

    # -------------------------
    # public API
    # -------------------------

    def process(self, results, evaluations, working_memory, context_weights=None) -> int:
        """
        Commits committed_action entries into working memory.

        Critical behavior:
        - If results exist but evaluations are missing, commits still happen.
        - Reflection should never silently skip action memory just because the evaluator
          came back empty. Humans love brittle pipelines for some reason.
        """
        results = results or []
        evaluations = evaluations or []

        if not hasattr(working_memory, "store") or not callable(getattr(working_memory, "store")):
            return 0

        n = max(len(results), len(evaluations))
        if n <= 0:
            return 0

        commits = 0

        for i in range(n):
            r = self._safe_dict(results[i]) if i < len(results) else {}
            e = self._safe_dict(evaluations[i]) if i < len(evaluations) else {}

            action = self._extract_action(r, e)
            if not action:
                action = self._extract_action(r, {})

            if not e:
                e = self._fallback_evaluation(r, action)

            name = self._safe_str(
                action.get("name") or r.get("action_name") or e.get("action_name"),
                default="unnamed_action",
            )
            action_type = self._safe_str(
                action.get("type") or r.get("action_type") or e.get("action_type"),
                default="unknown",
            )

            context = action.get("context", None)
            if context is None:
                context = r.get("context", None)
            if context is None:
                context = e.get("context", None)
            context_list = self._ensure_context_list(context)

            outcome = self._extract_outcome(r)
            if not outcome or outcome == "unknown":
                outcome = self._safe_str(e.get("outcome"), "unknown").lower()

            base_score = self._extract_score(e, r)

            weight_factor = self._resolve_weight_factor(context_list, context_weights)
            conf, weight = self._score_to_conf_weight(base_score, outcome)

            weight = self._clamp01(weight + 0.05 * (weight_factor - 0.5), default=self.DEFAULT_WEIGHT)
            conf = self._clamp01(conf + 0.05 * (weight_factor - 0.5), default=self.DEFAULT_CONFIDENCE)

            tags = self._safe_list_str(action.get("tags", []))
            for t in ("reflection", "committed_action", outcome):
                if t not in tags:
                    tags.append(t)

            domain = action.get("domain", r.get("domain", None))
            domain_s = self._safe_str(domain, default="")
            if domain_s and domain_s not in tags:
                tags.append(domain_s)

            action_details = self._safe_dict(action.get("details"))
            result_details = self._safe_dict(r.get("details"))
            eval_details = self._safe_dict(e)

            content_obj: Dict[str, Any] = {
                "action_name": name,
                "action_type": action_type,
                "outcome": outcome,
                "score_in": round(float(base_score), 3),
                "confidence": round(float(conf), 3),
                "weight": round(float(weight), 3),
                "context": context_list,
                "result": result_details,
                "evaluation": eval_details,
            }

            memory_entry = {
                "type": "committed_action",
                "content": self._human_commit_content(name, outcome, action_type),
                "content_obj": content_obj,
                "confidence": conf,
                "weight": weight,
                "context": context_list,
                "source": "reflection",
                "tags": tags,
                "domain": domain_s or None,
                "kind": "episode",
                "details": {
                    "action_name": name,
                    "action_type": action_type,
                    "outcome": outcome,
                    "action_details": action_details,
                    "result_details": result_details,
                    "score": round(float(base_score), 3),
                    "verified": True,
                    "pending_verification": False,
                    "contested": False,
                    "superseded": False,
                    "recall_blocked": False,
                    "status": "verified",
                },
                "name": name,
                "outcome": outcome,
                "timestamp": time.time(),
            }

            try:
                working_memory.store([memory_entry])
                commits += 1
            except Exception:
                continue

        return commits