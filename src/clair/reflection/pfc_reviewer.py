# FILE: reflection/pfc_reviewer.py
from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
import random
import re


class PFCReviewer:
    """
    Opinion / synthesis layer with strict anti-junk filtering.

    Guarantees:
    - uses only stored WM/LTM-derived memories already in WM
    - tries retrieve() first with multiple signatures
    - falls back to wm.buffer scan when needed
    - literature queries strongly prefer summaries / frames / literature claims
    - excludes feedback / calibration / operations sludge
    """

    VERSION = "2.60-reviewer"

    LITERATURE_TYPES = {
        "chapter_summary",
        "section_summary",
        "concept_frame",
        "literary_frame",
        "claim",
    }

    SUMMARY_TYPES = {
        "chapter_summary",
        "section_summary",
    }

    FRAME_TYPES = {
        "concept_frame",
        "literary_frame",
    }

    GENERIC_FACT_TYPES = {
        "fact", "lesson", "observe", "identity", "policy",
    }

    LITERATURE_QUERY_TOKENS = {
        "chapter", "section", "reading", "book", "lesson", "summary", "summarize",
        "argument", "theme", "story", "novel", "poem", "poetry", "literature",
        "appearance", "reality", "perception", "truth", "russell",
    }

    _CHAPTER_RE = re.compile(r"\bchap+t?e?r[_\s]*(\d+)\b", re.IGNORECASE)
    _SECTION_RE = re.compile(r"\bsection[_\s]*(\d+)\b", re.IGNORECASE)

    def __init__(self, working_memory):
        self.wm = working_memory

    def form_opinion(self, topic: str, *, domain: str = "general", count: int = 12) -> Dict[str, Any]:
        topic = (topic or "").strip()
        if not topic:
            return self._empty(topic, domain, "No topic provided.")

        dom = str(domain or "general").strip().lower()
        kws = self._keywords(topic)

        if self._is_literature_query(topic, dom):
            dom = "literature"

        matches = self._retrieve_any(topic, kws, domain=dom, count=count)

        if not matches:
            matches = self._scan_buffer(topic, kws, domain_hint=dom, limit=60)

        matches = self._sanitize_matches(matches, topic=topic, domain=dom)

        frames: List[dict] = []
        summaries: List[str] = []
        factual: List[str] = []
        fictional: List[str] = []
        uncertain: List[str] = []
        opinions: List[str] = []
        misc: List[str] = []

        for m in matches or []:
            if not isinstance(m, dict):
                continue

            mtype = str(m.get("type") or "").lower().strip()
            content = str(m.get("content") or "").strip()
            cobj = m.get("content_obj") if isinstance(m.get("content_obj"), dict) else {}

            if not content or self._is_probably_binary(content):
                continue

            if mtype in self.SUMMARY_TYPES:
                summaries.append(content)
                continue

            if mtype in self.FRAME_TYPES:
                frames.append(cobj or {"summary": content})
                continue

            if mtype == "claim":
                truth = str((cobj or {}).get("truth_label") or m.get("details", {}).get("truth_label") or "unknown").lower()
                if truth in {"likely_true", "verified"}:
                    factual.append(content)
                elif truth in {"fictional", "metaphor", "quote"}:
                    fictional.append(content)
                elif truth == "opinion":
                    opinions.append(content)
                else:
                    uncertain.append(content)
                continue

            if mtype in {"lesson", "fact", "observe"}:
                misc.append(content)

        if not (frames or summaries or factual or fictional or opinions or uncertain or misc):
            return self._empty(topic, dom, "I don't have enough learned material to form an opinion yet.")

        stance = self._choose_stance(frames, summaries, factual, fictional, opinions, uncertain, misc)
        summary = self._summarize(dom, topic, frames, summaries, factual, fictional, opinions, uncertain, misc)
        pros, cons = self._pros_cons(frames, summaries, factual, fictional, opinions, uncertain, misc)

        return {
            "type": "opinion",
            "topic": topic,
            "stance": stance,
            "summary": summary,
            "pros": pros,
            "cons": cons,
            "confidence": self._confidence(stance, frames, summaries, factual, fictional, opinions, uncertain, misc),
            "kind": "interpretation" if (dom == "literature" or frames or summaries) else "assessment",
            "source": "pfc_review",
        }

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def _retrieve_any(self, topic: str, keywords: List[str], *, domain: str, count: int):
        cp_dom = {"domain": domain, "query_text": topic}
        cp_gen = {"domain": "general", "query_text": topic}
        cp_lit = {"domain": "literature", "tags": ["literature", "reading"], "query_text": topic}

        ch = self._extract_chapter_hint(topic)
        sec = self._extract_section_hint(topic)
        if ch:
            cp_dom["chapter_hint"] = ch
            cp_lit["chapter_hint"] = ch
            cp_dom["tags"] = list(set(cp_dom.get("tags", []) + [f"chapter_{ch}"]))
            cp_lit["tags"] = list(set(cp_lit.get("tags", []) + [f"chapter_{ch}"]))
        if sec:
            cp_dom["section_hint"] = sec
            cp_lit["section_hint"] = sec
            cp_dom["tags"] = list(set(cp_dom.get("tags", []) + [f"section_{sec}"]))
            cp_lit["tags"] = list(set(cp_lit.get("tags", []) + [f"section_{sec}"]))

        cps = [cp_dom]
        if domain != "literature":
            cps.append(cp_lit)
        if domain != "general":
            cps.append(cp_gen)

        if self._is_literature_query(topic, domain):
            for cp in cps:
                for preferred_type in ("chapter_summary", "section_summary", "concept_frame", "literary_frame"):
                    out = self._try_retrieve_type(preferred_type, keywords, cp, count)
                    if out:
                        return out

        for cp in cps:
            out = self._try_retrieve_keywords(keywords, cp, count)
            if out:
                return out
            out = self._try_retrieve_query(topic, cp, count)
            if out:
                return out

        out = self._try_retrieve_keywords(keywords, None, count)
        if out:
            return out
        out = self._try_retrieve_query(topic, None, count)
        if out:
            return out

        return []

    def _try_retrieve_type(self, msg_type: str, keywords: List[str], cp: dict | None, count: int):
        try:
            if cp is None:
                return self.wm.retrieve(msg_type=msg_type, keywords=keywords, count=count)
            return self.wm.retrieve(msg_type=msg_type, keywords=keywords, context_profile=cp, count=count)
        except Exception:
            return []

    def _try_retrieve_keywords(self, keywords: List[str], cp: dict | None, count: int):
        try:
            if cp is None:
                return self.wm.retrieve(keywords=keywords, count=count)
            return self.wm.retrieve(keywords=keywords, context_profile=cp, count=count)
        except TypeError:
            pass
        except Exception:
            return []

        try:
            if cp is None:
                return self.wm.retrieve(keywords, count=count)
            return self.wm.retrieve(keywords, context_profile=cp, count=count)
        except Exception:
            return []

    def _try_retrieve_query(self, query: str, cp: dict | None, count: int):
        q = (query or "").strip()
        if not q:
            return []

        try:
            if cp is None:
                return self.wm.retrieve(q, count=count)
            return self.wm.retrieve(q, context_profile=cp, count=count)
        except TypeError:
            pass
        except Exception:
            return []

        try:
            if cp is None:
                return self.wm.retrieve(query=q, count=count)
            return self.wm.retrieve(query=q, context_profile=cp, count=count)
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Buffer fallback
    # ------------------------------------------------------------------
    def _scan_buffer(self, topic: str, keywords: List[str], *, domain_hint: str, limit: int = 60) -> List[dict]:
        buf = getattr(self.wm, "buffer", None)
        if not isinstance(buf, list) or not buf:
            return []

        keyset = set(k.lower() for k in keywords if isinstance(k, str))
        t_low = topic.lower().strip()

        ch = self._extract_chapter_hint(topic)
        sec = self._extract_section_hint(topic)
        literature_query = self._is_literature_query(topic, domain_hint)

        scored: List[Tuple[float, dict]] = []

        for entry in buf:
            m = self._coerce_buffer_entry(entry)
            if not m:
                continue

            content = str(m.get("content") or "").strip()
            if not content or self._is_probably_binary(content):
                continue

            if self._is_feedbackish_memory(m):
                continue

            dom = str(m.get("domain") or "").strip().lower()
            mtype = str(m.get("type") or "").strip().lower()
            tags = m.get("tags") or []
            tags_l = set(str(x).lower() for x in tags if str(x).strip())

            if literature_query:
                if dom != "literature" and "literature" not in tags_l and "reading" not in tags_l:
                    continue
                if mtype not in self.LITERATURE_TYPES and mtype not in self.GENERIC_FACT_TYPES:
                    continue

            c_low = content.lower()
            score = 0.0

            if t_low and t_low in c_low:
                score += 4.0

            score += sum(1.0 for k in list(keyset)[:16] if k and k in c_low)

            if domain_hint and dom == domain_hint:
                score += 2.0
            if "literature" in tags_l:
                score += 1.5
            if "reading" in tags_l:
                score += 1.5

            if literature_query:
                if mtype == "chapter_summary":
                    score += 8.0
                elif mtype == "section_summary":
                    score += 6.0
                elif mtype in {"concept_frame", "literary_frame"}:
                    score += 4.0
                elif mtype == "claim":
                    score += 1.5
                elif mtype in {"fact", "lesson", "observe"} and dom != "literature":
                    score -= 4.0

                if ch and f"chapter_{ch}" in tags_l:
                    score += 5.0
                if sec and f"section_{sec}" in tags_l:
                    score += 3.0

            if score > 0:
                scored.append((score, m))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _score, m in scored[:limit]]

    def _coerce_buffer_entry(self, entry: Any) -> Optional[dict]:
        if isinstance(entry, dict):
            return entry

        if hasattr(self.wm, "_record_to_legacy_dict"):
            try:
                return self.wm._record_to_legacy_dict(entry)
            except Exception:
                return None

        return None

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------
    def _sanitize_matches(self, matches: List[dict], *, topic: str, domain: str) -> List[dict]:
        if not matches:
            return []

        literature_query = self._is_literature_query(topic, domain)
        ch = self._extract_chapter_hint(topic)
        sec = self._extract_section_hint(topic)

        kept: List[Tuple[float, dict]] = []

        for m in matches:
            if not isinstance(m, dict):
                continue

            if self._is_feedbackish_memory(m):
                continue

            content = str(m.get("content") or "").strip()
            if not content or self._is_probably_binary(content):
                continue

            mtype = str(m.get("type") or "").strip().lower()
            dom = str(m.get("domain") or "").strip().lower()
            tags = set(str(x).lower() for x in (m.get("tags") or []) if str(x).strip())

            score = self._keyword_overlap_score(topic, content)

            if literature_query:
                if dom == "literature":
                    score += 2.0
                if "literature" in tags:
                    score += 2.0
                if "reading" in tags:
                    score += 2.0

                if mtype == "chapter_summary":
                    score += 8.0
                elif mtype == "section_summary":
                    score += 6.0
                elif mtype in {"concept_frame", "literary_frame"}:
                    score += 4.0
                elif mtype == "claim":
                    score += 1.5
                elif mtype in {"fact", "lesson", "observe"} and dom != "literature":
                    score -= 5.0

                if ch and f"chapter_{ch}" in tags:
                    score += 5.0
                if sec and f"section_{sec}" in tags:
                    score += 3.0

            if score > -1.0:
                kept.append((score, m))

        kept.sort(key=lambda x: x[0], reverse=True)
        cleaned = [m for _score, m in kept]

        if literature_query:
            lit_only = []
            for m in cleaned:
                mtype = str(m.get("type") or "").strip().lower()
                dom = str(m.get("domain") or "").strip().lower()
                tags = set(str(x).lower() for x in (m.get("tags") or []) if str(x).strip())
                if (
                    mtype in self.LITERATURE_TYPES
                    or dom == "literature"
                    or "literature" in tags
                    or "reading" in tags
                ):
                    lit_only.append(m)
            if lit_only:
                return lit_only[:24]

        return cleaned[:24]

    def _is_feedbackish_memory(self, m: dict) -> bool:
        if not isinstance(m, dict):
            return True

        mtype = str(m.get("type") or "").strip().lower()
        dom = str(m.get("domain") or "").strip().lower()
        tags = set(str(x).lower() for x in (m.get("tags") or []) if str(x).strip())
        content = str(m.get("content") or "").lower()

        if mtype == "feedback":
            return True
        if dom == "operations":
            return True
        if "feedback" in tags or "calibration" in tags:
            return True
        if "calibration feedback" in content:
            return True
        if "verdict=confirm" in content or "verdict=deny" in content:
            return True
        if "mem_ids=" in content:
            return True
        return False

    # ------------------------------------------------------------------
    # Keywords / query helpers
    # ------------------------------------------------------------------
    def _keywords(self, topic: str) -> List[str]:
        low = topic.lower().strip()
        kws: List[str] = []

        try:
            k = self.wm.extract_keywords(low)
            if isinstance(k, list):
                kws.extend(str(x).strip() for x in k if str(x).strip())
            elif k:
                kws.extend(str(x).strip() for x in list(k) if str(x).strip())
        except Exception:
            pass

        kws.extend(re.findall(r"[a-zA-Z0-9']+", low))
        kws.append(low)

        ch = self._extract_chapter_hint(topic)
        if ch:
            kws.extend([f"chapter_{ch}", f"chapter {ch}", ch])

        sec = self._extract_section_hint(topic)
        if sec:
            kws.extend([f"section_{sec}", f"section {sec}", sec])

        out, seen = [], set()
        for x in kws:
            x = str(x).strip().lower()
            if not x or x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    def _extract_chapter_hint(self, text: str) -> Optional[str]:
        m = self._CHAPTER_RE.search(text or "")
        return m.group(1) if m else None

    def _extract_section_hint(self, text: str) -> Optional[str]:
        m = self._SECTION_RE.search(text or "")
        return m.group(1) if m else None

    def _is_literature_query(self, topic: str, domain: str) -> bool:
        dom = str(domain or "").strip().lower()
        if dom == "literature":
            return True
        toks = set(re.findall(r"[a-zA-Z0-9']+", (topic or "").lower()))
        return bool(toks & self.LITERATURE_QUERY_TOKENS)

    def _keyword_overlap_score(self, topic: str, content: str) -> float:
        tt = set(re.findall(r"[a-zA-Z0-9']+", (topic or "").lower()))
        cc = set(re.findall(r"[a-zA-Z0-9']+", (content or "").lower()))
        if not tt or not cc:
            return 0.0
        return float(len(tt & cc))

    def _is_probably_binary(self, s: str) -> bool:
        low = s.lower()
        if any(tok in low for tok in ("%pdf", "endobj", "xref", "endstream", "stream")):
            return True
        printable = sum(1 for ch in s if (32 <= ord(ch) <= 126) or ch in "\n\r\t")
        return (printable / max(1, len(s))) < 0.78

    # ------------------------------------------------------------------
    # Opinion building
    # ------------------------------------------------------------------
    def _choose_stance(self, frames, summaries, factual, fictional, opinions, uncertain, misc) -> str:
        evidence_count = len(frames) + len(summaries) + len(factual) + len(fictional) + len(opinions) + len(misc)
        if evidence_count < 2:
            return "mixed"
        if summaries or frames or fictional or opinions:
            return random.choice(["mixed", "agree", "mixed"])
        return "mixed"

    def _summarize(self, domain: str, topic: str, frames, summaries, factual, fictional, opinions, uncertain, misc) -> str:
        literature_mode = (domain == "literature") or bool(frames) or bool(summaries)

        if literature_mode:
            if summaries:
                return self._finish(self._abstractify(summaries[0], max_len=320), 520)

            if frames:
                fr0 = frames[0] if isinstance(frames[0], dict) else {}
                themes = fr0.get("themes") or []
                tone = str(fr0.get("tone") or "").strip()
                events = fr0.get("events") or []
                fsum = str(fr0.get("summary") or "").strip()

                parts = []
                if fsum:
                    parts.append(self._abstractify(fsum, max_len=260))
                if themes:
                    parts.append(f"Themes observed: {', '.join(str(t) for t in themes[:6])}.")
                if tone:
                    parts.append(f"Tone appears: {tone}.")
                if events:
                    ev = str(events[0])[:140].strip()
                    if ev:
                        parts.append(f"Notable event: {self._abstractify(ev, max_len=140)}.")

                return self._finish(" ".join(p for p in parts if p.strip()), 520)

            evidence_pool = []
            evidence_pool.extend(factual[:2])
            evidence_pool.extend(misc[:2])
            evidence_pool.extend(uncertain[:1])
            evidence_pool.extend(opinions[:1])

            if not evidence_pool:
                if fictional:
                    return (
                        "I have text evidence stored, but it is mostly direct lines. "
                        "Ask for quotes if you want exact wording, or ask for a chapter summary."
                    )
                return "I don't have enough learned material to form an opinion yet."

            abstract = " ".join(
                self._abstractify(x, max_len=160)
                for x in evidence_pool
                if str(x).strip()
            )
            return self._finish(abstract, 520)

        evidence = []
        evidence.extend(factual[:2])
        evidence.extend(misc[:2])
        evidence.extend(uncertain[:1])
        base = " ".join(e.strip() for e in evidence if e.strip())
        if not base:
            return "I don't have enough learned material to form an opinion yet."
        return self._finish(base, 520)

    def _abstractify(self, text: str, max_len: int = 200) -> str:
        t = " ".join(str(text).strip().split())
        if not t:
            return ""
        t = t.replace("“", "").replace("”", "").replace('"', "").replace("’", "'")
        if len(t) > max_len:
            t = t[: max_len - 1].rstrip() + "…"
        return t

    def _pros_cons(self, frames, summaries, factual, fictional, opinions, uncertain, misc):
        pros = []
        cons = []

        if summaries:
            pros.append(self._abstractify(summaries[0], max_len=180))
        elif frames:
            fr0 = frames[0] if isinstance(frames[0], dict) else {}
            fsum = str(fr0.get("summary") or "").strip()
            if fsum:
                pros.append(self._abstractify(fsum, max_len=180))

        pros.extend(s[:180].rstrip() for s in factual[:2])
        pros.extend(s[:180].rstrip() for s in misc[:1])

        if fictional and not pros:
            pros.append("Text-world evidence is present (direct lines stored).")

        cons.extend(s[:180].rstrip() for s in uncertain[:2])
        cons.extend(s[:180].rstrip() for s in opinions[:2])

        return pros[:3], cons[:3]

    def _confidence(self, stance: str, frames, summaries, factual, fictional, opinions, uncertain, misc) -> float:
        base = 0.48
        base += 0.14 if summaries else 0.0
        base += 0.12 if frames else 0.0
        base += 0.06 if (factual or misc) else 0.0
        base += 0.03 if fictional else 0.0
        base -= 0.05 if len(uncertain) >= 3 else 0.0
        if stance == "mixed":
            base -= 0.02
        return float(max(0.35, min(0.85, base)))

    def _finish(self, text: str, max_len: int) -> str:
        t = " ".join(str(text).strip().split())
        if not t:
            return "I don't have enough learned material to form an opinion yet."
        if len(t) > max_len:
            t = t[: max_len - 1] + "…"
        if not t.endswith((".", "!", "?")):
            t += "."
        return t

    def _empty(self, topic: str, domain: str, msg: str) -> Dict[str, Any]:
        d = str(domain or "general").strip().lower()
        return {
            "type": "opinion",
            "topic": topic,
            "stance": "mixed",
            "summary": msg,
            "pros": [],
            "cons": [],
            "confidence": 0.35,
            "kind": "interpretation" if d == "literature" else "assessment",
            "source": "pfc_review",
        }