# comms/dialogue_state.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import time


@dataclass
class DialogueState:
    """
    Lightweight conversational state.
    Tracks conversational context, not facts.
    """
    topic: Optional[str] = None
    intent: Optional[str] = None
    verbosity: str = "normal"              # short | normal | detailed
    emotional_load: str = "low"            # low | medium | high
    last_user_text: str = ""
    last_response_text: str = ""
    updated_at: float = field(default_factory=lambda: time.time())

    notes: Dict[str, Any] = field(default_factory=dict)

    def update_from_user(self, user_text: str) -> None:
        self.last_user_text = user_text
        self.updated_at = time.time()

        t = user_text.lower().strip()

        if any(k in t for k in ("?", "what", "why", "how", "when", "where")):
            self.intent = "question"
        elif any(k in t for k in ("do", "make", "build", "add", "change", "fix", "rewrite")):
            self.intent = "request"
        elif any(k in t for k in ("plan", "roadmap", "steps", "next")):
            self.intent = "plan"
        else:
            self.intent = "statement"

        if any(k in t for k in ("quick", "short", "tldr")):
            self.verbosity = "short"
        elif any(k in t for k in ("detail", "explain", "deep", "full")):
            self.verbosity = "detailed"

        if any(k in t for k in ("panic", "scared", "stressed", "overwhelmed", "urgent")):
            self.emotional_load = "high"
        elif any(k in t for k in ("annoyed", "frustrated", "mad")):
            self.emotional_load = "medium"