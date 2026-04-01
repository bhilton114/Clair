# thalamus/thalamus_sources.py
from __future__ import annotations

import re
from typing import Any, Dict, List


class ThalamusSources:
    """
    Read-only intake extraction layer.

    For v1, this only handles plain outside text that Clair has already received,
    such as:
    - user-provided source text
    - file contents
    - log text
    - copied external notes

    It does NOT decide truth.
    It only extracts candidate evidence snippets.
    """

    VERSION = "1.0-intake"

    def __init__(self) -> None:
        self._ws = re.compile(r"\s+")

    def _normalize(self, text: str) -> str:
        text = str(text or "").strip()
        return self._ws.sub(" ", text)

    def _split_sentences(self, text: str) -> List[str]:
        text = self._normalize(text)
        if not text:
            return []
        parts = re.split(r"(?<=[\.\!\?])\s+|\n+", text)
        out: List[str] = []
        for part in parts:
            p = self._normalize(part)
            if p:
                out.append(p)
        return out

    def extract(
        self,
        packet: Dict[str, Any],
        *,
        intake_text: str = "",
        source_name: str = "intake_text",
    ) -> Dict[str, Any]:
        """
        Returns extracted candidate evidence snippets from intake text.
        This is intentionally dumb-but-safe for v1.
        """
        claim = self._normalize(packet.get("claim", ""))
        text = self._normalize(intake_text)

        if not claim or not text:
            return {
                "ok": True,
                "origin": source_name,
                "claim": claim,
                "snippets": [],
                "note": "No usable intake text.",
            }

        snippets = self._split_sentences(text)

        return {
            "ok": True,
            "origin": source_name,
            "claim": claim,
            "snippets": snippets,
            "note": f"Extracted {len(snippets)} snippet(s) from intake.",
        }