# safety/hard_rules.py

class HardRules:
    """
    Brainstem: enforces absolute safety constraints on actions.

    Guarantees:
    - Every returned action has a normalized 'risk' field (0.0 - 1.0)
    - Malformed actions are rejected explicitly
    - Never returns None
    """

    DEFAULT_RISK = 0.5
    MAX_SAFE_RISK = 0.8

    def _normalize_risk(self, action):
        """
        Ensures risk exists and is a valid float in [0,1].
        """
        raw = action.get("risk", self.DEFAULT_RISK)

        try:
            risk = float(raw)
        except (TypeError, ValueError):
            risk = self.DEFAULT_RISK

        # clamp
        risk = max(0.0, min(1.0, risk))
        action["risk"] = risk
        return risk

    def _validate_action(self, action):
        """
        Minimal schema validation to prevent downstream corruption.
        """
        if not isinstance(action, dict):
            return False

        required = ["name", "type", "details"]
        for key in required:
            if key not in action:
                return False

        return True

    def enforce(self, actions):
        """
        Filters actions based on absolute safety thresholds.

        :param actions: list of action dicts
        :return: list of safe actions (always valid + risk-normalized)
        """
        print("[HardRules] Enforcing safety constraints...")

        if not actions:
            print("[HardRules] No actions provided.")
            return []

        safe_actions = []

        for action in actions:

            if not self._validate_action(action):
                print("[HardRules] Rejected malformed action.")
                continue

            risk = self._normalize_risk(action)

            if risk <= self.MAX_SAFE_RISK:
                safe_actions.append(action)
            else:
                name = action.get("name", "UNKNOWN")
                print(f"[HardRules] Blocked action {name} due to high risk: {risk}")

        if not safe_actions:
            print("[HardRules] No safe actions found. Skipping execution.")
        else:
            print(f"[HardRules] {len(safe_actions)} action(s) cleared for execution.")

        return safe_actions