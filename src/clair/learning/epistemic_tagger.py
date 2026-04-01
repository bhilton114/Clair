# FILE: learning/epistemic_tagger.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import re


@dataclass
class EpistemicResult:
    """
    Epistemic + intake-shaping output.

    truth_label:
      - fictional
      - likely_true
      - likely_false
      - unknown
      - metaphor
      - opinion
      - instruction
      - question
      - uncertain
      - quote

    memory_kind_hint:
      - fact
      - summary
      - procedure
      - reflection
      - scenario
      - rule
      - unresolved

    verification_status_hint:
      - verified
      - unverified
      - provisional
      - disputed
      - rejected

    should_store_semantic:
      Whether this should normally enter semantic WM/LTM flow.

    should_route_to_verifier:
      Whether downstream verification should be asked to inspect it.

    source_trust_hint:
      - trusted
      - normal
      - low
    """
    truth_label: str
    confidence_truth: float
    reasons: List[str] = field(default_factory=list)

    speaker: Optional[str] = None
    modality: Optional[str] = None
    claim_type: Optional[str] = None

    memory_kind_hint: str = "unresolved"
    verification_status_hint: str = "unverified"
    domain_hint: Optional[str] = None
    source_trust_hint: str = "normal"
    should_store_semantic: bool = True
    should_route_to_verifier: bool = False
    hazard_family: Optional[str] = None
    tags: List[str] = field(default_factory=list)


class EpistemicTagger:
    """
    Deterministic epistemic tagger for Clair's intake path.

    Responsibilities:
    - classify truth style / epistemic status
    - shape memory intake hints for hippocampus_ingest
    - stay conservative when uncertain
    """

    _FICTION_GENRE_TOKENS = {
        "fiction", "novel", "poem", "poetry", "short story", "story",
        "tale", "myth", "legend", "fable", "fantasy", "sci-fi", "science fiction",
        "horror", "gothic", "allegory", "parable", "narrator",
    }

    _NARRATIVE_CUES = {
        "once upon a time", "in a distant", "in the kingdom", "the narrator", "chapter",
        "he said", "she said", "they said", "replied", "whispered", "cried", "muttered",
    }

    _METAPHOR_CUES = {
        "as if", "like a", "like an", "as though", "metaphor", "symbol",
        "shadow of", "sea of", "storm of", "heart of", "darkness of",
    }

    _UNCERTAINTY_CUES = {
        "maybe", "perhaps", "possibly", "probably", "might", "could", "seems", "appears",
        "suggests", "likely", "unlikely", "i think", "i guess", "it is said", "some say",
    }

    _OPINION_CUES = {
        "beautiful", "ugly", "amazing", "terrible", "best", "worst", "good", "bad",
        "boring", "exciting", "sad", "happy", "meaningful", "pointless", "brilliant",
        "stupid", "i love", "i hate", "i prefer",
    }

    _INSTRUCTION_CUES = {
        "do ", "don't ", "never ", "always ", "remember to", "make sure", "step", "instructions",
        "you should", "you must", "avoid", "ensure", "try to",
    }

    _SUPERNATURAL_CUES = {
        "ghost", "spirit", "curse", "magic", "wizard", "dragon", "prophecy",
        "immortal", "teleport", "time travel", "vampire", "werewolf",
        "talking raven", "talking animal", "resurrected",
    }

    _RULE_CUES = {
        "must", "must not", "never", "always", "rule", "policy", "requirement",
        "allowed", "forbidden", "prohibited",
    }

    _SUMMARY_CUES = {
        "argues that", "suggests that", "summarizes", "summary", "chapter", "section",
        "the text shows", "the author argues", "the reading explains",
    }

    _REFLECTION_CUES = {
        "i learned", "i realized", "i noticed", "i found that", "looking back",
        "reflection", "review", "retrospective",
    }

    _SCENARIO_CUES = {
        "if", "suppose", "imagine", "scenario", "what happens when", "in the event of",
    }

    _HAZARD_FAMILIES: Dict[str, Set[str]] = {
        "fire": {"fire", "smoke", "burn", "burning", "flame", "heat"},
        "flood": {"flood", "flooding", "floodwater", "water", "submerged", "evacuate", "flash", "rising"},
        "earthquake": {"earthquake", "aftershock", "shaking", "collapse", "quake", "tremor"},
        "lost": {"lost", "shelter", "signal", "conserve", "woods", "night", "stay put"},
    }

    _QUOTE_RE = re.compile(r"(^|[\s(])['\"“”].+?['\"“”]([\s).,!?:;]|$)")
    _QUESTION_RE = re.compile(r"\?\s*$")
    _WORD_RE = re.compile(r"[a-z0-9']+")

    def __init__(self, strict: bool = False, max_reason_len: int = 120):
        self.strict = bool(strict)
        self.max_reason_len = int(max_reason_len)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def tag(
        self,
        claim_text: str,
        *,
        context_profile: Optional[Dict[str, Any]] = None,
        doc_meta: Optional[Dict[str, Any]] = None,
        speaker: Optional[str] = None,
        modality: Optional[str] = None,
        claim_type: Optional[str] = None,
        verified_facts: Optional[Any] = None,
        verified_fact_keywords: Optional[Any] = None,
    ) -> EpistemicResult:
        text = (claim_text or "").strip()
        if not text:
            return EpistemicResult(
                truth_label="unknown",
                confidence_truth=0.0,
                reasons=["empty claim"],
                memory_kind_hint="unresolved",
                verification_status_hint="unverified",
                should_store_semantic=False,
                source_trust_hint="low",
            )

        cp = context_profile or {}
        dm = doc_meta or {}

        dom = self._safe_lower(cp.get("domain", "general"))
        tags = self._as_lower_set(cp.get("tags"))
        goal = self._safe_lower(cp.get("goal", ""))

        meta_is_fiction = self._truthy(dm.get("is_fiction")) or (
            self._safe_lower(dm.get("genre_hint")) in self._FICTION_GENRE_TOKENS
        )
        meta_is_nonfiction = self._truthy(dm.get("is_nonfiction"))

        sp = self._safe_lower(speaker) or self._infer_speaker(text, dom, tags)
        mod = self._safe_lower(modality) or self._infer_modality(text)

        domain_hint = self._infer_domain_hint(text, dom, tags)
        hazard_family = self._infer_hazard_family(text, domain_hint)
        out_tags = self._collect_tags(text, tags, domain_hint, hazard_family, mod)

        # 1) non-assertions
        if self._QUESTION_RE.search(text):
            return EpistemicResult(
                truth_label="question",
                confidence_truth=0.95,
                reasons=["ends with '?' (interrogative, not asserted)"],
                speaker=sp or None,
                modality=mod or None,
                claim_type=claim_type,
                memory_kind_hint="unresolved",
                verification_status_hint="unverified",
                domain_hint=domain_hint,
                source_trust_hint="normal",
                should_store_semantic=False,
                should_route_to_verifier=False,
                hazard_family=hazard_family,
                tags=out_tags,
            )

        # 2) instruction
        if self._looks_like_instruction(text):
            return EpistemicResult(
                truth_label="instruction",
                confidence_truth=0.85,
                reasons=["directive/instruction phrasing detected"],
                speaker=sp or None,
                modality=mod or None,
                claim_type=claim_type,
                memory_kind_hint=self._infer_memory_kind(text, "instruction", domain_hint),
                verification_status_hint="unverified",
                domain_hint=domain_hint,
                source_trust_hint="normal",
                should_store_semantic=True,
                should_route_to_verifier=False,
                hazard_family=hazard_family,
                tags=out_tags,
            )

        # 3) opinion
        if self._looks_like_opinion(text):
            return EpistemicResult(
                truth_label="opinion",
                confidence_truth=0.85,
                reasons=["value-judgment / subjective language detected"],
                speaker=sp or None,
                modality=mod or None,
                claim_type=claim_type,
                memory_kind_hint="reflection",
                verification_status_hint="unverified",
                domain_hint=domain_hint,
                source_trust_hint="low",
                should_store_semantic=False,
                should_route_to_verifier=False,
                hazard_family=hazard_family,
                tags=out_tags + ["opinion"],
            )

        # 4) metaphor
        if self._looks_like_metaphor(text):
            return EpistemicResult(
                truth_label="metaphor",
                confidence_truth=0.80,
                reasons=["figurative/metaphoric cue detected"],
                speaker=sp or None,
                modality=mod or None,
                claim_type=claim_type,
                memory_kind_hint="reflection",
                verification_status_hint="unverified",
                domain_hint=domain_hint,
                source_trust_hint="low",
                should_store_semantic=False,
                should_route_to_verifier=False,
                hazard_family=hazard_family,
                tags=out_tags + ["metaphor"],
            )

        # 5) uncertainty
        if self._looks_like_uncertain(text) or mod in {"hypothetical", "hedged"}:
            return EpistemicResult(
                truth_label="uncertain",
                confidence_truth=0.75,
                reasons=["hedging/speculation cue detected"],
                speaker=sp or None,
                modality=mod or None,
                claim_type=claim_type,
                memory_kind_hint=self._infer_memory_kind(text, "uncertain", domain_hint),
                verification_status_hint="provisional",
                domain_hint=domain_hint,
                source_trust_hint="normal",
                should_store_semantic=True,
                should_route_to_verifier=True,
                hazard_family=hazard_family,
                tags=out_tags + ["uncertain"],
            )

        # 6) quote
        if self._QUOTE_RE.search(text) and (mod in {"quoted", "hearsay"} or sp in {"character", "narrator"}):
            if meta_is_fiction or dom == "literature" or "poetry" in tags:
                return EpistemicResult(
                    truth_label="fictional",
                    confidence_truth=0.88,
                    reasons=["quoted speech in literature/fiction context"],
                    speaker=sp or None,
                    modality="quoted",
                    claim_type=claim_type,
                    memory_kind_hint="summary",
                    verification_status_hint="unverified",
                    domain_hint=domain_hint,
                    source_trust_hint="normal",
                    should_store_semantic=False,
                    should_route_to_verifier=False,
                    hazard_family=hazard_family,
                    tags=out_tags + ["fictional", "quote"],
                )

        # 7) fiction likelihood
        fiction_score, fiction_reasons = self._fiction_likelihood(
            text, dom, tags, meta_is_fiction, meta_is_nonfiction, sp, mod, goal
        )
        if fiction_score >= 0.75:
            return EpistemicResult(
                truth_label="fictional",
                confidence_truth=min(0.99, fiction_score),
                reasons=fiction_reasons,
                speaker=sp or None,
                modality=mod or None,
                claim_type=claim_type,
                memory_kind_hint=self._infer_memory_kind(text, "fictional", domain_hint),
                verification_status_hint="unverified",
                domain_hint=domain_hint,
                source_trust_hint="normal",
                should_store_semantic=(domain_hint == "literature"),
                should_route_to_verifier=False,
                hazard_family=hazard_family,
                tags=out_tags + ["fictional"],
            )

        vf = self._normalize_verified_facts(verified_facts)
        vfk = self._as_lower_set(verified_fact_keywords)

        # 8) verified direct match
        if vf:
            match, why = self._match_verified_fact(text, vf)
            if match:
                return EpistemicResult(
                    truth_label="likely_true",
                    confidence_truth=0.90 if not self.strict else 0.82,
                    reasons=[why],
                    speaker=sp or None,
                    modality=mod or None,
                    claim_type=claim_type,
                    memory_kind_hint=self._infer_memory_kind(text, "likely_true", domain_hint),
                    verification_status_hint="verified",
                    domain_hint=domain_hint,
                    source_trust_hint="trusted",
                    should_store_semantic=True,
                    should_route_to_verifier=False,
                    hazard_family=hazard_family,
                    tags=out_tags + ["verified_match"],
                )

        # 9) contradiction
        if vf:
            contrad, why = self._obvious_numeric_contradiction(text, vf)
            if contrad:
                return EpistemicResult(
                    truth_label="likely_false",
                    confidence_truth=0.85,
                    reasons=[why],
                    speaker=sp or None,
                    modality=mod or None,
                    claim_type=claim_type,
                    memory_kind_hint="unresolved",
                    verification_status_hint="disputed",
                    domain_hint=domain_hint,
                    source_trust_hint="low",
                    should_store_semantic=True,
                    should_route_to_verifier=True,
                    hazard_family=hazard_family,
                    tags=out_tags + ["contradiction_candidate"],
                )

        # 10) anchor overlap
        reasons: List[str] = []
        if vfk:
            anchor_overlap = self._anchor_overlap(text, vfk)
            if anchor_overlap >= (2 if self.strict else 3):
                reasons.append(f"shares {anchor_overlap} truth-anchor keywords")

        # 11) nonfiction/real-world style
        style_score, style_reasons = self._realworld_style_score(
            text, dom, tags, meta_is_fiction, meta_is_nonfiction
        )
        if style_score >= (0.80 if not self.strict else 0.88):
            reasons.extend(style_reasons)
            return EpistemicResult(
                truth_label="likely_true",
                confidence_truth=min(0.92, style_score),
                reasons=self._trim_reasons(reasons[:6]),
                speaker=sp or None,
                modality=mod or None,
                claim_type=claim_type,
                memory_kind_hint=self._infer_memory_kind(text, "likely_true", domain_hint),
                verification_status_hint="unverified",
                domain_hint=domain_hint,
                source_trust_hint="normal",
                should_store_semantic=True,
                should_route_to_verifier=(style_score < 0.90),
                hazard_family=hazard_family,
                tags=out_tags,
            )

        # default unknown
        if not reasons:
            reasons = ["insufficient evidence to determine truth status"]

        return EpistemicResult(
            truth_label="unknown",
            confidence_truth=0.50,
            reasons=self._trim_reasons(reasons[:6]),
            speaker=sp or None,
            modality=mod or None,
            claim_type=claim_type,
            memory_kind_hint=self._infer_memory_kind(text, "unknown", domain_hint),
            verification_status_hint="unverified",
            domain_hint=domain_hint,
            source_trust_hint="normal",
            should_store_semantic=True,
            should_route_to_verifier=True,
            hazard_family=hazard_family,
            tags=out_tags,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _safe_lower(self, x: Any) -> str:
        if x is None:
            return ""
        try:
            return str(x).strip().lower()
        except Exception:
            return ""

    def _truthy(self, x: Any) -> bool:
        if isinstance(x, bool):
            return x
        s = self._safe_lower(x)
        return s in {"1", "true", "yes", "y", "on"}

    def _as_lower_set(self, x: Any) -> Set[str]:
        if x is None:
            return set()
        if isinstance(x, str):
            return {self._safe_lower(x)} if x.strip() else set()
        if isinstance(x, (list, tuple, set)):
            out = set()
            for it in x:
                s = self._safe_lower(it)
                if s:
                    out.add(s)
            return out
        return set()

    def _tokens(self, text: str) -> List[str]:
        return self._WORD_RE.findall((text or "").lower())

    def _looks_like_instruction(self, text: str) -> bool:
        t = (text or "").strip().lower()
        if any(t.startswith(x) for x in ("do ", "don't ", "never ", "always ", "remember to", "make sure")):
            return True
        return any(cue in t for cue in self._INSTRUCTION_CUES)

    def _looks_like_opinion(self, text: str) -> bool:
        t = (text or "").strip().lower()
        if " i think " in f" {t} ":
            return True
        return any(cue in t for cue in self._OPINION_CUES)

    def _looks_like_metaphor(self, text: str) -> bool:
        t = (text or "").strip().lower()
        return any(cue in t for cue in self._METAPHOR_CUES)

    def _looks_like_uncertain(self, text: str) -> bool:
        t = (text or "").strip().lower()
        return any(cue in t for cue in self._UNCERTAINTY_CUES)

    def _infer_modality(self, text: str) -> str:
        t = (text or "").strip().lower()
        if self._QUESTION_RE.search(text):
            return "question"
        if self._QUOTE_RE.search(text):
            return "quoted"
        if any(cue in t for cue in self._UNCERTAINTY_CUES):
            return "hedged"
        if any(x in t for x in ("if ", "would ", "could ", "might ")):
            return "hypothetical"
        return "asserted"

    def _infer_speaker(self, text: str, domain: str, tags: Set[str]) -> str:
        t = (text or "").lower()
        if domain == "literature" or "poetry" in tags or "fiction" in tags:
            if any(cue in t for cue in (" he said", " she said", " they said", " replied", " whispered", " cried")):
                return "character"
            return "narrator"
        return "author"

    def _infer_domain_hint(self, text: str, domain: str, tags: Set[str]) -> str:
        if domain and domain != "general":
            return domain

        toks = set(self._tokens(text))
        if toks & {"chapter", "reading", "author", "narrator", "poem", "novel", "literature"}:
            return "literature"
        if toks & {"fire", "flood", "earthquake", "evacuate", "smoke", "shelter"}:
            return "survival"
        if toks & {"plan", "schedule", "task", "priority"}:
            return "planning"
        if toks & {"garden", "soil", "watering", "mulch", "plant"}:
            return "garden"
        if toks & {"wire", "breaker", "circuit", "voltage"}:
            return "electrical"
        return domain or "general"

    def _infer_hazard_family(self, text: str, domain_hint: str) -> Optional[str]:
        if domain_hint != "survival":
            return None
        toks = set(self._tokens(text))
        for fam, vocab in self._HAZARD_FAMILIES.items():
            if toks & vocab:
                return fam
        return None

    def _collect_tags(
        self,
        text: str,
        existing_tags: Set[str],
        domain_hint: str,
        hazard_family: Optional[str],
        modality: str,
    ) -> List[str]:
        out: List[str] = []
        seen: Set[str] = set()

        def add(tag: Optional[str]) -> None:
            s = self._safe_lower(tag)
            if s and s not in seen:
                out.append(s)
                seen.add(s)

        for t in sorted(existing_tags):
            add(t)

        add(domain_hint)
        add(hazard_family)
        add(modality)

        txt = (text or "").lower()
        if "chapter" in txt:
            m = re.search(r"\bchapter\s+(\d+)\b", txt)
            if m:
                add(f"chapter_{m.group(1)}")
        if "section" in txt:
            m = re.search(r"\bsection\s+(\d+)\b", txt)
            if m:
                add(f"section_{m.group(1)}")

        return out

    def _infer_memory_kind(self, text: str, truth_label: str, domain_hint: str) -> str:
        t = (text or "").strip().lower()

        if truth_label in {"question"}:
            return "unresolved"
        if truth_label in {"opinion", "metaphor"}:
            return "reflection"
        if truth_label in {"instruction"}:
            if any(cue in t for cue in self._RULE_CUES):
                return "rule"
            return "procedure"
        if truth_label in {"fictional"}:
            if domain_hint == "literature":
                return "summary"
            return "unresolved"
        if any(cue in t for cue in self._SUMMARY_CUES):
            return "summary"
        if any(cue in t for cue in self._REFLECTION_CUES):
            return "reflection"
        if any(cue in t for cue in self._SCENARIO_CUES):
            return "scenario"
        if any(cue in t for cue in self._RULE_CUES):
            return "rule"
        return "fact"

    def _fiction_likelihood(
        self,
        text: str,
        domain: str,
        tags: Set[str],
        meta_is_fiction: bool,
        meta_is_nonfiction: bool,
        speaker: str,
        modality: str,
        goal: str,
    ) -> Tuple[float, List[str]]:
        t = (text or "").strip().lower()
        reasons: List[str] = []
        score = 0.0

        if meta_is_fiction:
            score += 0.55
            reasons.append("doc_meta indicates fiction")

        if domain == "literature" or "poetry" in tags:
            score += 0.30
            reasons.append("context domain/tags suggest literature/poetry")

        if speaker in {"narrator", "character"}:
            score += 0.20
            reasons.append(f"speaker inferred as {speaker}")

        if modality in {"quoted"} and (domain == "literature" or "poetry" in tags or meta_is_fiction):
            score += 0.15
            reasons.append("quoted speech in fiction-like context")

        if any(cue in t for cue in self._NARRATIVE_CUES):
            score += 0.20
            reasons.append("narrative cue detected")

        if any(cue in t for cue in self._SUPERNATURAL_CUES):
            score += 0.20
            reasons.append("supernatural cue detected")

        if meta_is_nonfiction:
            score -= 0.40
            reasons.append("doc_meta indicates nonfiction")

        if goal in {"interpret", "analyze", "literary"}:
            score += 0.10
            reasons.append("goal suggests interpretive reading")

        score = max(0.0, min(0.99, score))
        return score, self._trim_reasons(reasons)

    def _realworld_style_score(
        self,
        text: str,
        domain: str,
        tags: Set[str],
        meta_is_fiction: bool,
        meta_is_nonfiction: bool,
    ) -> Tuple[float, List[str]]:
        t = (text or "").strip()
        tl = t.lower()

        reasons: List[str] = []
        score = 0.0

        if meta_is_nonfiction:
            score += 0.40
            reasons.append("doc_meta indicates nonfiction")

        if meta_is_fiction:
            score -= 0.35
            reasons.append("doc_meta indicates fiction")

        if domain in {"tech", "science", "general", "planning", "electrical", "survival"} and ("poetry" not in tags) and ("literature" not in tags):
            score += 0.10
            reasons.append(f"domain={domain} leans factual")

        if re.search(r"\b\d+(\.\d+)?\b", tl):
            score += 0.15
            reasons.append("contains numeric value")

        if any(x in tl for x in (" is ", " are ", " was ", " were ")):
            score += 0.08
            reasons.append("contains copula pattern (is/are/was)")

        if any(x in tl for x in ("%", "km", "meters", "degrees", "celsius", "kg", "mph", "http", "python", "module", "error", "traceback")):
            score += 0.12
            reasons.append("technical marker present")

        if "\n" in t and sum(1 for ch in t if ch in ",;:") >= 3:
            score -= 0.10
            reasons.append("poetry-like formatting cues")

        score = max(0.0, min(0.99, score))
        return score, self._trim_reasons(reasons)

    def _trim_reasons(self, reasons: List[str]) -> List[str]:
        out: List[str] = []
        for r in reasons:
            r2 = (r or "").strip()
            if not r2:
                continue
            if len(r2) > self.max_reason_len:
                r2 = r2[: self.max_reason_len - 1] + "…"
            out.append(r2)

        seen = set()
        uniq = []
        for r in out:
            if r not in seen:
                uniq.append(r)
                seen.add(r)
        return uniq[:8]

    def _normalize_verified_facts(self, verified_facts: Any) -> List[str]:
        if verified_facts is None:
            return []
        out: List[str] = []
        try:
            if isinstance(verified_facts, (list, tuple, set)):
                for it in verified_facts:
                    if isinstance(it, str):
                        s = it.strip().lower()
                        if s:
                            out.append(s)
                    elif isinstance(it, dict):
                        c = it.get("content")
                        if isinstance(c, str) and c.strip():
                            out.append(c.strip().lower())
            elif isinstance(verified_facts, dict):
                facts = verified_facts.get("facts")
                if isinstance(facts, (list, tuple, set)):
                    return self._normalize_verified_facts(facts)
            elif isinstance(verified_facts, str):
                s = verified_facts.strip().lower()
                if s:
                    out.append(s)
        except Exception:
            return []
        return out[:500]

    def _match_verified_fact(self, claim_text: str, vf: List[str]) -> Tuple[bool, str]:
        c = self._normalize_factish(claim_text)
        if not c:
            return False, "claim normalized empty"

        for f in vf:
            fn = self._normalize_factish(f)
            if fn and fn == c:
                return True, "matches verified fact (normalized equality)"
        return False, "no verified-fact match"

    def _normalize_factish(self, s: str) -> str:
        s = (s or "").strip().lower()
        if not s:
            return ""
        s = re.sub(r"[\s]+", " ", s)
        s = s.strip(" .,!?:;\"'“”()[]{}")
        return s

    def _anchor_overlap(self, claim_text: str, anchors: Set[str]) -> int:
        toks = set(self._tokens(claim_text))
        return len(toks & anchors)

    def _obvious_numeric_contradiction(self, claim_text: str, vf: List[str]) -> Tuple[bool, str]:
        nums_claim = re.findall(r"\b\d+(?:\.\d+)?\b", claim_text.lower())
        if not nums_claim:
            return False, "no numeric content"

        n0 = nums_claim[0]
        claim_tokens = set(self._tokens(claim_text))
        if not claim_tokens:
            return False, "no tokens"

        for f in vf:
            nums_f = re.findall(r"\b\d+(?:\.\d+)?\b", f)
            if not nums_f:
                continue
            if nums_f[0] == n0:
                continue

            f_tokens = set(self._tokens(f))
            overlap = len(claim_tokens & f_tokens)
            if overlap >= 5:
                return True, f"numeric contradiction vs verified fact (claim {n0} vs fact {nums_f[0]})"

        return False, "no clear numeric contradiction"

    # ------------------------------------------------------------------
    # Convenience: batch tag
    # ------------------------------------------------------------------
    def tag_many(
        self,
        claims: List[Any],
        *,
        context_profile: Optional[Dict[str, Any]] = None,
        doc_meta: Optional[Dict[str, Any]] = None,
        verified_facts: Optional[Any] = None,
        verified_fact_keywords: Optional[Any] = None,
    ) -> List[EpistemicResult]:
        out: List[EpistemicResult] = []
        for it in claims or []:
            if isinstance(it, str):
                out.append(self.tag(
                    it,
                    context_profile=context_profile,
                    doc_meta=doc_meta,
                    verified_facts=verified_facts,
                    verified_fact_keywords=verified_fact_keywords,
                ))
                continue

            if isinstance(it, dict):
                txt = it.get("text") or it.get("content") or ""
                out.append(self.tag(
                    str(txt),
                    context_profile=context_profile,
                    doc_meta=doc_meta,
                    speaker=it.get("speaker"),
                    modality=it.get("modality"),
                    claim_type=it.get("claim_type"),
                    verified_facts=verified_facts,
                    verified_fact_keywords=verified_fact_keywords,
                ))
                continue

            txt = getattr(it, "text", None)
            if txt is None:
                txt = getattr(it, "content", "")
            out.append(self.tag(
                str(txt),
                context_profile=context_profile,
                doc_meta=doc_meta,
                speaker=getattr(it, "speaker", None),
                modality=getattr(it, "modality", None),
                claim_type=getattr(it, "claim_type", None),
                verified_facts=verified_facts,
                verified_fact_keywords=verified_fact_keywords,
            ))

        return out