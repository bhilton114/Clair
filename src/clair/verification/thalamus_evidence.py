# thalamus/thalamus_evidence.py
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


class ThalamusEvidence:
    """
    Evidence arbitration layer.

    v1 logic:
    - Find lexical support or contradiction against intake snippets
    - Score support/conflict
    - Return deterministic structured verdict

    This is intentionally lightweight so it plugs in cleanly.
    """

    VERSION = "1.0-lexical"

    def __init__(self) -> None:
        self._ws = re.compile(r"\s+")
        self._nonword = re.compile(r"[^a-z0-9\s]+")
        self._number_pattern = re.compile(r"\d+(?:\.\d+)?")
        self._stop = {
            "the", "a", "an", "and", "or", "to", "of", "in", "on", "at", "for", "is", "are", "was", "were",
            "be", "been", "being", "this", "that", "these", "those", "it", "as", "by", "with", "from",
            "what", "which", "who", "whom", "where", "when", "why", "how",
            "there", "here", "then", "than", "but", "if", "so",
        }

    def _normalize(self, text: str) -> str:
        text = str(text or "").lower().strip()
        text = self._nonword.sub(" ", text)
        text = self._ws.sub(" ", text)
        return text.strip()

    def _tokens(self, text: str) -> List[str]:
        norm = self._normalize(text)
        if not norm:
            return []
        return [t for t in norm.split() if t and t not in self._stop]

    def _number_list(self, text: str) -> List[str]:
        return self._number_pattern.findall(str(text or ""))

    def _token_overlap_score(self, a: str, b: str) -> float:
        ta = set(self._tokens(a))
        tb = set(self._tokens(b))
        if not ta or not tb:
            return 0.0
        overlap = ta.intersection(tb)
        denom = max(1, min(len(ta), len(tb)))
        return len(overlap) / denom

    def _classify_snippet(self, claim: str, snippet: str) -> Tuple[str, float]:
        """
        Returns (label, score)
        label in {"support", "conflict", "neutral"}
        """
        claim_norm = self._normalize(claim)
        snip_norm = self._normalize(snippet)

        if not claim_norm or not snip_norm:
            return ("neutral", 0.0)

        overlap = self._token_overlap_score(claim, snippet)
        claim_nums = self._number_list(claim)
        snip_nums = self._number_list(snippet)

        # Exact/near-exact support
        if claim_norm == snip_norm:
            return ("support", 1.0)

        # Strong lexical support
        if overlap >= 0.80:
            if claim_nums and snip_nums:
                if claim_nums == snip_nums:
                    return ("support", 0.95)
                else:
                    return ("conflict", 0.90)
            return ("support", 0.85)

        # Moderate lexical support
        if overlap >= 0.55:
            if claim_nums and snip_nums and claim_nums != snip_nums:
                return ("conflict", 0.75)
            return ("support", 0.60)

        # Numeric contradiction on same topic
        if overlap >= 0.35 and claim_nums and snip_nums and claim_nums != snip_nums:
            return ("conflict", 0.65)

        return ("neutral", 0.0)

    def evaluate(
        self,
        packet: Dict[str, Any],
        source_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        claim = str(packet.get("claim", "") or "").strip()
        snippets = source_result.get("snippets", []) or []

        support_hits: List[Dict[str, Any]] = []
        conflict_hits: List[Dict[str, Any]] = []
        neutral_hits: List[Dict[str, Any]] = []

        support_score = 0.0
        conflict_score = 0.0

        for snippet in snippets:
            if not isinstance(snippet, str):
                continue

            label, score = self._classify_snippet(claim, snippet)

            row = {
                "text": snippet,
                "label": label,
                "score": round(float(score), 4),
            }

            if label == "support":
                support_hits.append(row)
                support_score += score
            elif label == "conflict":
                conflict_hits.append(row)
                conflict_score += score
            else:
                neutral_hits.append(row)

        if support_score > 0.0 and conflict_score == 0.0:
            status = "supported"
            verdict = "confirm"
            confidence = min(0.97, 0.60 + min(0.35, 0.12 * len(support_hits)) + min(0.20, support_score * 0.08))
        elif conflict_score > 0.0 and support_score == 0.0:
            status = "contradicted"
            verdict = "deny"
            confidence = min(0.97, 0.60 + min(0.35, 0.12 * len(conflict_hits)) + min(0.20, conflict_score * 0.08))
        elif support_score > 0.0 and conflict_score > 0.0:
            status = "mixed"
            verdict = "unsure"
            confidence = 0.55
        else:
            status = "insufficient"
            verdict = "unsure"
            confidence = 0.35

        return {
            "ok": True,
            "claim": claim,
            "status": status,
            "verdict": verdict,
            "confidence": round(float(confidence), 4),
            "support_score": round(float(support_score), 4),
            "conflict_score": round(float(conflict_score), 4),
            "matches": support_hits[:8],
            "conflicts": conflict_hits[:8],
            "neutrals": neutral_hits[:8],
            "note": f"support={len(support_hits)} conflict={len(conflict_hits)} neutral={len(neutral_hits)}",
        }