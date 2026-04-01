# learning/angular_gyrus.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import re


ClaimLike = Union[str, Dict[str, Any]]


@dataclass
class Extraction:
    """
    claims:
      - list[str] OR list[dict] (recommended dict shape below)
      - may include a special narrative frame dict with {"_type": "narrative_frame", ...}

    keywords:
      - top keywords (lowercase) for retrieval/tagging
    """
    claims: List[ClaimLike]
    keywords: List[str]


class AngularGyrus:
    """
    Lightweight meaning extraction WITHOUT an LLM.

    Goals:
    - Works for nonfiction AND literature/poetry
    - Produces claim-like units PLUS an optional narrative frame (as a special dict in claims)
    - Extracts keywords robustly even from PDFs with line breaks/hyphenation

    Output claim dict schema (recommended):
      {
        "text": <string>,
        "speaker": "author" | "narrator" | "character" | "unknown",
        "modality": "asserted" | "quoted" | "hedged" | "hypothetical",
        "claim_type": "fact_candidate" | "fictional_world" | "opinion" | "metaphor" | "uncertain",
        "confidence_text": float 0..1,
        "evidence_span": short string excerpt (optional)
      }

    NOTE:
    - Truth labeling is done later by EpistemicTagger.
    """

    STOP = {
        "the","a","an","and","or","but","to","of","in","on","for","with","as","is","are","was","were",
        "be","been","being","it","that","this","these","those","i","you","we","they","he","she",
        "at","by","from","not","no","yes","can","could","should","would","will","just","about",
        "into","over","under","then","than","so","if","my","your","our","their","his","her",
        "me","him","them","what","who","whom","which","when","where","why","how"
    }

    _WORD_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)
    _KW_RE = re.compile(r"[A-Za-z]{4,}")
    _QUOTE_RE = re.compile(r"[\"“”'].+?[\"“”']")
    _DIALOGUE_CUE_RE = re.compile(r"\b(he said|she said|they said|replied|whispered|cried|muttered)\b", re.IGNORECASE)

    _UNCERTAINTY_CUES = (
        "maybe", "perhaps", "possibly", "probably", "might", "could", "seems", "appears",
        "suggests", "likely", "unlikely", "i think", "i guess", "it is said", "some say",
    )

    _OPINION_CUES = (
        "beautiful", "ugly", "amazing", "terrible", "best", "worst", "good", "bad",
        "boring", "exciting", "sad", "happy", "meaningful", "pointless", "brilliant",
        "stupid", "i love", "i hate", "i prefer",
    )

    _METAPHOR_CUES = ("as if", "as though", "like a", "like an", "shadow of", "sea of", "storm of", "heart of")

    _FACT_TRIGGERS = (" is ", " are ", " was ", " were ", " causes ", " leads ", " therefore ", " means ", " results ")

    def extract(self, text: str, *, max_claims: int = 8) -> Extraction:
        if not text:
            return Extraction([], [])

        cleaned = self._clean_text(text)

        # Keywords should come from the entire cleaned text
        keywords = self._extract_keywords(cleaned, top_k=12)

        # Split into candidate units:
        # - prose sentences
        # - plus poetry lines (if many line breaks)
        units = self._split_units(cleaned)

        # Build a narrative frame (always helpful for fiction/poetry opinions)
        frame = self._build_narrative_frame(units, keywords)

        # Extract claim-like units (works for nonfiction & lit)
        claim_dicts = self._extract_claims(units, keywords, max_claims=max_claims)

        # Insert narrative frame as a special claim item so Hippocampus can store it
        claims: List[ClaimLike] = []
        if frame:
            claims.append({"_type": "narrative_frame", **frame})
        claims.extend(claim_dicts)

        return Extraction(claims=claims, keywords=keywords)

    # -------------------------
    # Text cleaning
    # -------------------------
    def _clean_text(self, text: str) -> str:
        t = str(text)

        # Fix hyphenation across line breaks: "rav-\nen" -> "raven"
        t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)

        # Normalize newlines (keep them, but tame them)
        t = t.replace("\r\n", "\n").replace("\r", "\n")

        # Remove excessive spaces, keep line breaks
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r"\n{3,}", "\n\n", t)

        return t.strip()

    # -------------------------
    # Unit splitting
    # -------------------------
    def _split_units(self, text: str) -> List[str]:
        if not text:
            return []

        # If it looks like poetry (lots of line breaks), keep meaningful lines as units too
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        poetry_like = len(lines) >= 12 and (sum(1 for ln in lines if len(ln) < 80) / max(1, len(lines))) > 0.55

        units: List[str] = []
        if poetry_like:
            # Use lines, but also merge short adjacent lines to form better “semantic units”
            merged: List[str] = []
            buf = ""
            for ln in lines:
                if not buf:
                    buf = ln
                    continue
                # merge if both short-ish
                if len(buf) < 80 and len(ln) < 80 and not buf.endswith((".", "!", "?", ";", ":")):
                    buf = f"{buf} {ln}"
                else:
                    merged.append(buf.strip())
                    buf = ln
            if buf:
                merged.append(buf.strip())
            units.extend([u for u in merged if len(u) >= 35])

        # Also extract prose-like sentences (good for nonfiction)
        sents = re.split(r"(?<=[.!?])\s+", text.strip())
        sents = [s.strip() for s in sents if len(s.strip()) >= 35]
        units.extend(sents)

        # Deduplicate while preserving order
        seen = set()
        out = []
        for u in units:
            key = u.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(u)

        return out[:200]  # hard cap to prevent runaway PDFs

    # -------------------------
    # Keywords
    # -------------------------
    def _extract_keywords(self, text: str, *, top_k: int = 12) -> List[str]:
        words = self._KW_RE.findall(text.lower())
        freq: Dict[str, int] = {}
        for w in words:
            if w in self.STOP:
                continue
            freq[w] = freq.get(w, 0) + 1

        ranked = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        kws = [w for w, _ in ranked[: max(3, int(top_k))]]

        # small cleanup: drop ultra-generic doc artifacts
        junk = {"chapter", "section", "copyright", "page", "edition"}
        kws = [k for k in kws if k not in junk]
        return kws[:top_k]

    # -------------------------
    # Narrative frame
    # -------------------------
    def _build_narrative_frame(self, units: List[str], keywords: List[str]) -> Dict[str, Any]:
        if not units:
            return {}

        # Entities: capitalize-ish tokens (simple heuristic)
        entities = self._extract_entities(units)

        # Themes: based on keyword buckets (simple heuristics)
        themes = self._infer_themes(keywords)

        # Tone: crude sentiment (no hallucination, just lexical scoring)
        tone = self._infer_tone(units)

        # Summary: take 1–2 representative units
        summary = self._make_summary(units)

        # Events: use verbs cues in a shallow way
        events = self._extract_events(units)

        return {
            "domain_hint": "literature" if themes or tone != "neutral" else "general",
            "entities": entities[:12],
            "themes": themes[:8],
            "tone": tone,
            "events": events[:10],
            "summary": summary[:520],
        }

    def _extract_entities(self, units: List[str]) -> List[str]:
        # Pull sequences of Capitalized words, but avoid sentence-start common words
        cap_seq_re = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")
        bad_starts = {"The", "A", "An", "And", "But", "Or", "In", "On", "At", "To", "Of", "For", "With"}

        cand: Dict[str, int] = {}
        for u in units[:80]:
            for m in cap_seq_re.findall(u):
                s = m.strip()
                if not s or s.split()[0] in bad_starts:
                    continue
                if len(s) < 3:
                    continue
                cand[s] = cand.get(s, 0) + 1

        ranked = sorted(cand.items(), key=lambda x: x[1], reverse=True)
        return [e for e, _ in ranked]

    def _infer_themes(self, keywords: List[str]) -> List[str]:
        kws = set(keywords or [])
        themes = []

        def has_any(*ws: str) -> bool:
            return any(w in kws for w in ws)

        if has_any("death", "grave", "mourning", "lost", "loss", "lament"):
            themes.append("grief/loss")
        if has_any("madness", "insane", "dream", "nightmare", "mind"):
            themes.append("madness/obsession")
        if has_any("fear", "dark", "shadow", "terror", "horror"):
            themes.append("fear/dread")
        if has_any("love", "heart", "desire"):
            themes.append("love/longing")
        if has_any("memory", "remember", "forgotten", "past"):
            themes.append("memory")
        if has_any("religion", "angel", "heaven", "hell", "god"):
            themes.append("spirituality")

        return themes

    def _infer_tone(self, units: List[str]) -> str:
        negative = {"dark", "never", "nothing", "death", "lost", "sorrow", "sad", "fear", "dread", "cry"}
        positive = {"joy", "happy", "hope", "light", "love", "smile"}

        score = 0
        for u in units[:80]:
            toks = [t.lower() for t in self._WORD_RE.findall(u)]
            for t in toks:
                if t in negative:
                    score -= 1
                elif t in positive:
                    score += 1

        if score <= -4:
            return "dark"
        if score >= 4:
            return "uplifting"
        return "neutral"

    def _extract_events(self, units: List[str]) -> List[str]:
        # Very shallow: keep units with action verbs common in narrative
        verbs = {"walk", "enter", "arrive", "leave", "knock", "open", "speak", "say", "reply", "look", "see", "hear", "cry"}
        out = []
        for u in units[:120]:
            toks = set(t.lower() for t in self._WORD_RE.findall(u))
            if toks & verbs:
                out.append(u[:220].strip())
        return out

    def _make_summary(self, units: List[str]) -> str:
        # Pick 2 informative units: prefer mid-length with verbs
        scored: List[Tuple[float, str]] = []
        for u in units[:80]:
            ul = u.lower()
            sc = 0.0
            sc += 1.0 if any(tr in ul for tr in self._FACT_TRIGGERS) else 0.0
            sc += 0.8 if self._DIALOGUE_CUE_RE.search(u) else 0.0
            sc += 0.5 if any(ch.isdigit() for ch in u) else 0.0
            # prefer not-too-long
            if 45 <= len(u) <= 220:
                sc += 0.6
            scored.append((sc, u))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [s for _, s in scored[:2]] or units[:2]
        s = " ".join(top).strip()
        return s

    # -------------------------
    # Claim extraction
    # -------------------------
    def _extract_claims(self, units: List[str], keywords: List[str], *, max_claims: int) -> List[Dict[str, Any]]:
        if not units:
            return []

        claims: List[Dict[str, Any]] = []

        # Score units for "claim-likeness"
        scored: List[Tuple[float, str]] = []
        for u in units:
            ul = u.lower()
            score = 0.0

            score += sum(1.0 for t in self._FACT_TRIGGERS if t in ul)
            score += 0.8 if any(ch.isdigit() for ch in u) else 0.0
            score += 0.7 if self._QUOTE_RE.search(u) else 0.0
            score += 0.6 if self._DIALOGUE_CUE_RE.search(u) else 0.0
            score += 0.5 if any(cue in ul for cue in self._UNCERTAINTY_CUES) else 0.0
            score += 0.5 if any(cue in ul for cue in self._OPINION_CUES) else 0.0
            score += 0.4 if any(cue in ul for cue in self._METAPHOR_CUES) else 0.0

            # length preference
            if 45 <= len(u) <= 240:
                score += 0.6
            elif len(u) > 420:
                score -= 0.6

            # keyword overlap bonus
            kwset = set(keywords or [])
            toks = set(self._KW_RE.findall(ul))
            overlap = len(toks & kwset)
            score += min(1.0, overlap * 0.08)

            scored.append((score, u))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Take top max_claims units (including those with low score if we have no better)
        picked = [u for sc, u in scored[: max_claims * 2] if u]  # take extra then filter
        if not picked:
            return []

        for u in picked:
            # Ignore very short fragments
            if len(u) < 30:
                continue

            speaker = self._infer_speaker(u)
            modality = self._infer_modality(u)
            claim_type = self._infer_claim_type(u, speaker, modality)

            confidence_text = self._infer_confidence_text(u, modality)

            claims.append({
                "text": u.strip(),
                "speaker": speaker,
                "modality": modality,
                "claim_type": claim_type,
                "confidence_text": float(confidence_text),
                "evidence_span": u.strip()[:220],
            })

            if len(claims) >= max_claims:
                break

        return claims

    def _infer_speaker(self, unit: str) -> str:
        ul = (unit or "").lower()
        if self._DIALOGUE_CUE_RE.search(unit) or self._QUOTE_RE.search(unit):
            return "character"
        # If it looks like poetry/narration, default narrator
        if "\n" in unit or len(unit) < 140:
            return "narrator"
        return "author"

    def _infer_modality(self, unit: str) -> str:
        u = (unit or "").strip()
        ul = u.lower()
        if self._QUOTE_RE.search(u):
            return "quoted"
        if any(cue in ul for cue in self._UNCERTAINTY_CUES):
            return "hedged"
        if any(x in ul for x in (" if ", " would ", " could ", " might ")):
            return "hypothetical"
        return "asserted"

    def _infer_claim_type(self, unit: str, speaker: str, modality: str) -> str:
        ul = (unit or "").lower()

        if any(cue in ul for cue in self._OPINION_CUES):
            return "opinion"
        if any(cue in ul for cue in self._METAPHOR_CUES):
            return "metaphor"
        if modality in {"hedged", "hypothetical"}:
            return "uncertain"
        if speaker in {"character", "narrator"} and (self._QUOTE_RE.search(unit) or "\n" in unit):
            # narrative default
            return "fictional_world"
        # otherwise treat as fact candidate
        return "fact_candidate"

    def _infer_confidence_text(self, unit: str, modality: str) -> float:
        # How strongly the text asserts it (NOT truth). Very simple heuristic.
        base = 0.72
        if modality == "quoted":
            base = 0.65
        elif modality == "hedged":
            base = 0.55
        elif modality == "hypothetical":
            base = 0.50

        ul = unit.lower()
        if "!" in unit:
            base += 0.03
        if any(tr in ul for tr in self._FACT_TRIGGERS):
            base += 0.08
        if any(ch.isdigit() for ch in unit):
            base += 0.05

        return max(0.35, min(0.95, base))