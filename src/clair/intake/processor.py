# FILE: intake/processor.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
from intake.contracts import InputPacket, UncertaintyFlags

import re
import hashlib
import os


class IntakeProcessor:
    """
    Transforms raw InputPackets into structured packets ready for routing.

    Responsibilities:
    - Normalize content
    - Decompose into segments
    - Assess uncertainty
    - Compute content hashes
    - Attach file/folder metadata consistently
    - Infer packet intent/type (ask/lesson/observe/feedback/read_document)
    - Enforce stable packet invariants (signals/constraints/uncertainty)

    Reading-aware additions:
    - Detects likely document/book inputs
    - Extracts claim candidates
    - Extracts conceptual frame candidates
    - Produces section/chapter summary candidates
    - Stores file/source metadata for later retrieval anchoring
    """

    # ------------------------
    # Core split / detection patterns
    # ------------------------
    _SEGMENT_SPLIT_RE = re.compile(r"[.!?\n]+")
    _SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
    _AMBIG_PRONOUN_RE = re.compile(r"\b(it|this|that|they|something|someone|somebody)\b", re.IGNORECASE)
    _METAPHOR_RE = re.compile(r"\b(as if|feels like)\b|\blike\b", re.IGNORECASE)
    _CONTRAST_RE = re.compile(r"\bbut\b|\bhowever\b|\byet\b", re.IGNORECASE)

    # Intent/type heuristics
    _LESSON_PREFIX_RE = re.compile(r"^\s*(lesson|learn|remember)\s*[:\-]\s*", re.IGNORECASE)
    _OBSERVE_PREFIX_RE = re.compile(r"^\s*(observe|observation|noticed)\s*[:\-]\s*", re.IGNORECASE)
    _FEEDBACK_PREFIX_RE = re.compile(r"^\s*(feedback|result|score)\s*[:\-]\s*", re.IGNORECASE)
    _READ_PREFIX_RE = re.compile(r"^\s*(read)\s*[:\-]?\s*", re.IGNORECASE)
    _QUESTION_MARK_RE = re.compile(r"\?\s*$")
    _QUESTION_START_RE = re.compile(
        r"^\s*(who|what|when|where|why|how|can|could|should|would|do|does|did|is|are|was|were)\b",
        re.IGNORECASE,
    )

    # Document / file heuristics
    _DOC_EXT_RE = re.compile(r"\.(txt|md|pdf|text)$", re.IGNORECASE)
    _CHAPTER_RE = re.compile(r"\bchapter\s+([ivxlcdm]+|\d+)\b", re.IGNORECASE)
    _SECTION_RE = re.compile(r"\b(section|lesson|part)\s+([ivxlcdm]+|\d+)\b", re.IGNORECASE)
    _HEADING_LINE_RE = re.compile(r"^(chapter|section|part)\b", re.IGNORECASE)

    # Claim / concept heuristics
    _CLAIM_MARKER_RE = re.compile(
        r"\b(is|are|was|were|means|refers to|can|cannot|should|must|may|because|therefore|thus|if|when)\b",
        re.IGNORECASE,
    )

    _FRAME_PATTERNS = {
        "appearance_vs_reality": re.compile(r"\bappearance\b.*\breality\b|\breality\b.*\bappearance\b", re.IGNORECASE),
        "knowledge_and_doubt": re.compile(r"\bknowledge\b|\bdoubt\b|\bcertain(?:ty)?\b|\buncertain(?:ty)?\b", re.IGNORECASE),
        "perception_and_truth": re.compile(r"\bperception\b|\bseem(?:s|ing)?\b|\btruth\b|\bsense-data\b|\bsenses\b", re.IGNORECASE),
        "cause_and_reason": re.compile(r"\bbecause\b|\btherefore\b|\bthus\b|\breason\b", re.IGNORECASE),
        "ethics_and_action": re.compile(r"\bgood\b|\bbad\b|\bright\b|\bwrong\b|\bought\b|\bshould\b", re.IGNORECASE),
        "strategy_and_conflict": re.compile(r"\bstrategy\b|\bconflict\b|\badvantage\b|\bopponent\b", re.IGNORECASE),
        "survival_and_risk": re.compile(r"\brisk\b|\bdanger\b|\bsafe(?:ty)?\b|\bsurvival\b", re.IGNORECASE),
    }

    # Segment safety knobs
    MAX_SEGMENTS = 24
    MAX_SEGMENT_LEN = 320
    MAX_CLAIM_CANDIDATES = 18
    MAX_FRAME_CANDIDATES = 6
    MAX_SUMMARY_SENTENCES = 4

    def process(self, raw_inputs: List[InputPacket]) -> List[InputPacket]:
        packets: List[InputPacket] = []
        if not raw_inputs:
            return packets

        for packet in raw_inputs:
            if not isinstance(packet, InputPacket):
                continue
            if not packet.is_viable():
                continue

            self._ensure_invariants(packet)

            self._normalize(packet)
            try:
                packet.normalize_file_metadata()
            except Exception:
                pass

            self._infer_packet_type(packet)
            self._derive_document_context(packet)
            self._decompose(packet)
            self._assess_uncertainty(packet)
            self._derive_reading_structures(packet)
            self._compute_hash(packet)

            packets.append(packet)

        return packets

    # ------------------------
    # Invariants / Guardrails
    # ------------------------
    def _ensure_invariants(self, packet: InputPacket) -> None:
        if getattr(packet, "signals", None) is None or not isinstance(packet.signals, dict):
            packet.signals = {}

        if getattr(packet, "constraints", None) is None or not isinstance(packet.constraints, list):
            packet.constraints = []

        if getattr(packet, "uncertainty", None) is None or not isinstance(packet.uncertainty, UncertaintyFlags):
            packet.uncertainty = UncertaintyFlags()

    def _add_constraint(self, packet: InputPacket, constraint: str) -> None:
        if not constraint:
            return
        if constraint not in packet.constraints:
            packet.constraints.append(constraint)

    # ------------------------
    # Internal stages
    # ------------------------
    def _normalize(self, packet: InputPacket) -> None:
        text = packet.signals.get("normalized_text")
        if not text:
            text = packet.signals.get("text")
        if not text:
            text = getattr(packet, "raw_input", "")

        normalized = str(text if text is not None else "")
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        normalized = re.sub(r"[ \t]+", " ", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        normalized = normalized.strip()

        packet.signals["normalized_text"] = normalized
        packet.signals["normalized_text_len"] = len(normalized)

        file_path = packet.signals.get("file_path") or packet.signals.get("path")
        if file_path:
            packet.signals["file_path"] = file_path
            packet.signals["file_name"] = os.path.basename(file_path) or "unknown"
            packet.signals["folder_path"] = os.path.dirname(file_path) or None
            packet.signals["file_ext"] = os.path.splitext(str(file_path))[1].lower()

            try:
                looks_like_fs_path = isinstance(file_path, str) and (
                    ":\\" in file_path or file_path.startswith(("/", "\\"))
                )
                if looks_like_fs_path and os.path.exists(file_path):
                    stat = os.stat(file_path)
                    packet.signals["file_mtime"] = int(stat.st_mtime)
                    packet.signals["file_size"] = int(stat.st_size)
                    packet.signals["source_id"] = (
                        f"file:{file_path}:{packet.signals['file_mtime']}:{packet.signals['file_size']}"
                    )
                else:
                    packet.signals.setdefault("file_mtime", None)
                    packet.signals.setdefault("file_size", None)
                    packet.signals.setdefault("source_id", f"file:{file_path}")
            except Exception:
                packet.signals.setdefault("file_mtime", None)
                packet.signals.setdefault("file_size", None)
                packet.signals.setdefault("source_id", f"file:{file_path}")
        else:
            packet.signals.setdefault("file_path", None)
            packet.signals.setdefault("file_name", "unknown")
            packet.signals.setdefault("folder_path", None)
            packet.signals.setdefault("file_ext", None)
            packet.signals.setdefault("source_id", "input:unknown")
            packet.signals.setdefault("file_mtime", None)
            packet.signals.setdefault("file_size", None)

        if getattr(packet, "extraction_confidence", None) is None:
            length = len(normalized)
            if length == 0:
                packet.extraction_confidence = 0.2
                self._add_constraint(packet, "empty_input")
            elif length < 20:
                packet.extraction_confidence = 0.5
            else:
                packet.extraction_confidence = 0.9
        else:
            try:
                packet.extraction_confidence = float(packet.extraction_confidence)
            except Exception:
                packet.extraction_confidence = 0.5
            packet.extraction_confidence = max(0.0, min(1.0, packet.extraction_confidence))

    def _infer_packet_type(self, packet: InputPacket) -> None:
        text = (packet.signals.get("normalized_text") or "").strip()
        if not text:
            packet.signals["packet_type"] = "unknown"
            return

        raw = getattr(packet, "raw_input", None)
        if isinstance(raw, dict):
            t = (raw.get("type") or "").strip().lower()
            if t in {"ask", "lesson", "observe", "feedback", "read_document"}:
                packet.signals["packet_type"] = t
                return

        file_name = str(packet.signals.get("file_name") or "")
        file_ext = str(packet.signals.get("file_ext") or "")

        if file_ext and self._DOC_EXT_RE.search(file_ext):
            packet.signals["packet_type"] = "read_document"
            return

        if self._READ_PREFIX_RE.search(text) and len(text) > 40:
            packet.signals["packet_type"] = "read_document"
            return

        if self._LESSON_PREFIX_RE.search(text):
            packet.signals["packet_type"] = "lesson"
            return

        if self._OBSERVE_PREFIX_RE.search(text):
            packet.signals["packet_type"] = "observe"
            return

        if self._FEEDBACK_PREFIX_RE.search(text):
            packet.signals["packet_type"] = "feedback"
            return

        if self._QUESTION_MARK_RE.search(text) or self._QUESTION_START_RE.search(text):
            packet.signals["packet_type"] = "ask"
            return

        # Long file-based text defaults to read_document-like observation
        if file_name and len(text) >= 120:
            packet.signals["packet_type"] = "read_document"
            return

        packet.signals["packet_type"] = "observe"

    def _derive_document_context(self, packet: InputPacket) -> None:
        text = packet.signals.get("normalized_text", "") or ""
        file_name = str(packet.signals.get("file_name") or "")
        file_stem = os.path.splitext(file_name)[0] if file_name else ""
        packet_type = str(packet.signals.get("packet_type") or "")

        is_document = packet_type == "read_document" or bool(file_name and self._DOC_EXT_RE.search(file_name))
        packet.signals["is_document_like"] = bool(is_document)

        title = self._guess_title(file_stem, text)
        packet.signals["document_title"] = title

        chapter_hint = self._extract_chapter_hint(file_stem, text)
        section_hint = self._extract_section_hint(file_stem, text)

        if chapter_hint:
            packet.signals["chapter_hint"] = chapter_hint
        if section_hint:
            packet.signals["section_hint"] = section_hint

        book_hint = self._guess_book_title(file_stem, packet.signals.get("folder_path"), text)
        if book_hint:
            packet.signals["book_title_hint"] = book_hint

        packet.signals["document_domain_hint"] = self._infer_domain_from_text(text)

    def _decompose(self, packet: InputPacket) -> None:
        text = packet.signals.get("normalized_text", "")
        if not text:
            packet.signals["segments"] = []
            packet.signals["segment_count"] = 0
            return

        is_document = bool(packet.signals.get("is_document_like"))

        if is_document and len(text) > 600:
            cleaned = self._document_chunks(text)
        else:
            segments = self._SEGMENT_SPLIT_RE.split(text)
            cleaned = [s.strip() for s in segments if s and s.strip()]

        seen = set()
        deduped: List[str] = []
        for s in cleaned:
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)

            if len(s) > self.MAX_SEGMENT_LEN:
                s = s[: self.MAX_SEGMENT_LEN].rstrip() + "…"
            deduped.append(s)

            if len(deduped) >= self.MAX_SEGMENTS:
                break

        packet.signals["segments"] = deduped
        packet.signals["segment_count"] = len(deduped)

    def _assess_uncertainty(self, packet: InputPacket) -> None:
        text = (packet.signals.get("normalized_text", "") or "").strip()
        flags = UncertaintyFlags()

        if len(text) < 5:
            flags.low_signal_quality = True
            self._add_constraint(packet, "input_too_short")

        if len(text) >= 8 and self._AMBIG_PRONOUN_RE.search(text):
            flags.missing_references = True
            self._add_constraint(packet, "unresolved_reference")

        if self._METAPHOR_RE.search(text):
            flags.metaphor_detected = True
            self._add_constraint(packet, "possible_metaphor")

        segments = packet.signals.get("segments") or []
        if isinstance(segments, list) and len(segments) > 1:
            if any(self._CONTRAST_RE.search(s or "") for s in segments):
                flags.conflicting_signals = True
                self._add_constraint(packet, "internal_contradiction")

        packet.uncertainty = flags

    def _derive_reading_structures(self, packet: InputPacket) -> None:
        text = packet.signals.get("normalized_text", "") or ""
        packet_type = str(packet.signals.get("packet_type") or "")
        is_document = bool(packet.signals.get("is_document_like"))

        packet.signals["claim_candidates"] = []
        packet.signals["frame_candidates"] = []
        packet.signals["section_summary_candidate"] = None
        packet.signals["chapter_summary_candidate"] = None

        if not text or not (is_document or packet_type == "read_document"):
            return

        claims = self._extract_claim_candidates(text)
        frames = self._extract_frame_candidates(text)
        section_summary = self._make_section_summary(text, claims, frames, packet)
        chapter_summary = self._make_chapter_summary(section_summary, packet, frames)

        packet.signals["claim_candidates"] = claims[: self.MAX_CLAIM_CANDIDATES]
        packet.signals["frame_candidates"] = frames[: self.MAX_FRAME_CANDIDATES]
        packet.signals["section_summary_candidate"] = section_summary
        packet.signals["chapter_summary_candidate"] = chapter_summary

        # Helpful retrieval anchors
        packet.signals["memory_layers_present"] = {
            "claims": bool(packet.signals["claim_candidates"]),
            "frames": bool(packet.signals["frame_candidates"]),
            "section_summary": bool(section_summary),
            "chapter_summary": bool(chapter_summary),
        }

    def _compute_hash(self, packet: InputPacket) -> None:
        text = packet.signals.get("normalized_text", "") or ""
        source_id = packet.signals.get("source_id", "input:unknown") or "input:unknown"
        blob = f"{source_id}||{text}".encode("utf-8", errors="replace")

        digest = hashlib.sha256(blob).hexdigest()
        packet.signals["content_hash"] = digest
        packet.signals["content_hash_short"] = digest[:12]

    # ------------------------
    # Document helpers
    # ------------------------
    def _document_chunks(self, text: str) -> List[str]:
        """
        Produces chunk-like segments from long reading text without destroying structure.
        """
        paras = [p.strip() for p in text.split("\n\n") if p and p.strip()]
        if not paras:
            paras = [s.strip() for s in self._SENTENCE_SPLIT_RE.split(text) if s and s.strip()]

        chunks: List[str] = []
        current: List[str] = []
        current_len = 0
        target = 260

        for p in paras:
            plen = len(p)
            if current and (current_len + plen) > target:
                chunks.append(" ".join(current).strip())
                current = [p]
                current_len = plen
            else:
                current.append(p)
                current_len += plen + 1

        if current:
            chunks.append(" ".join(current).strip())

        return chunks

    def _extract_claim_candidates(self, text: str) -> List[str]:
        sentences = [s.strip() for s in self._SENTENCE_SPLIT_RE.split(text) if s and s.strip()]
        out: List[str] = []

        for s in sentences:
            if len(s) < 35 or len(s) > 320:
                continue
            if self._CLAIM_MARKER_RE.search(s):
                out.append(s)

        if not out:
            for s in sentences:
                if 40 <= len(s) <= 220:
                    out.append(s)

        return self._dedupe_preserve_order(out)

    def _extract_frame_candidates(self, text: str) -> List[str]:
        frames: List[str] = []

        lines = [ln.strip() for ln in text.splitlines() if ln and ln.strip()]
        heading_like = [
            ln for ln in lines[:10]
            if len(ln) <= 100 and (self._HEADING_LINE_RE.search(ln) or ln.isupper())
        ]
        for h in heading_like[:3]:
            frames.append(f"heading:{h}")

        for name, pat in self._FRAME_PATTERNS.items():
            if pat.search(text):
                frames.append(name)

        return self._dedupe_preserve_order(frames)

    def _make_section_summary(
        self,
        text: str,
        claims: List[str],
        frames: List[str],
        packet: InputPacket,
    ) -> Optional[str]:
        title = packet.signals.get("document_title") or packet.signals.get("file_name") or "document"
        chapter = packet.signals.get("chapter_hint")
        section = packet.signals.get("section_hint")

        summary_bits: List[str] = []
        if chapter:
            summary_bits.append(f"Chapter {chapter}")
        if section:
            summary_bits.append(f"Section {section}")
        summary_bits.append(str(title))

        intro = ": ".join([x for x in summary_bits if x])

        top_claims = claims[: min(self.MAX_SUMMARY_SENTENCES, len(claims))]
        top_frames = [f for f in frames if not str(f).startswith("heading:")][:3]

        body_parts: List[str] = []
        if top_frames:
            body_parts.append("Themes include " + ", ".join(top_frames) + ".")

        if top_claims:
            body_parts.append("Key points: " + " ".join(top_claims))

        if not body_parts and len(text) >= 60:
            snippet = text[:260].strip()
            body_parts.append(snippet + ("…" if len(text) > 260 else ""))

        summary = (intro + ". " + " ".join(body_parts)).strip()
        return summary if summary and len(summary) >= 30 else None

    def _make_chapter_summary(
        self,
        section_summary: Optional[str],
        packet: InputPacket,
        frames: List[str],
    ) -> Optional[str]:
        if not section_summary:
            return None

        book_title = packet.signals.get("book_title_hint")
        chapter = packet.signals.get("chapter_hint")
        domain = packet.signals.get("document_domain_hint")

        label_parts: List[str] = []
        if book_title:
            label_parts.append(str(book_title))
        if chapter:
            label_parts.append(f"chapter {chapter}")

        label = " ".join(label_parts).strip()
        if not label:
            label = str(packet.signals.get("document_title") or "document")

        thematic_frames = [f for f in frames if not str(f).startswith("heading:")][:3]
        if thematic_frames:
            return (
                f"{label} summary: This material is mainly about "
                f"{', '.join(thematic_frames)}. {section_summary}"
            )

        if domain and domain != "general":
            return f"{label} summary: This material is in the {domain} domain. {section_summary}"

        return f"{label} summary: {section_summary}"

    def _guess_title(self, file_stem: str, text: str) -> str:
        if file_stem and file_stem.strip():
            return file_stem.strip()

        for line in text.splitlines()[:8]:
            clean = line.strip()
            if 4 <= len(clean) <= 120:
                return clean

        return "Untitled Document"

    def _extract_chapter_hint(self, file_stem: str, text: str) -> Optional[str]:
        for src in (file_stem, text[:800]):
            m = self._CHAPTER_RE.search(src or "")
            if m:
                return m.group(1)
        return None

    def _extract_section_hint(self, file_stem: str, text: str) -> Optional[str]:
        for src in (file_stem, text[:800]):
            m = self._SECTION_RE.search(src or "")
            if m:
                return m.group(2)
        return None

    def _guess_book_title(self, file_stem: str, folder_path: Any, text: str) -> Optional[str]:
        folder_name = os.path.basename(str(folder_path)) if folder_path else ""
        candidates = [file_stem, folder_name]

        for c in candidates:
            if not c:
                continue
            clean = c.replace("_", " ").replace("-", " ").strip()
            if len(clean) >= 4 and "lesson" not in clean.lower():
                return clean

        for line in text.splitlines()[:6]:
            clean = line.strip()
            if 5 <= len(clean) <= 120 and "chapter" not in clean.lower():
                return clean

        return None

    def _infer_domain_from_text(self, text: str) -> str:
        tl = text.lower()

        if re.search(r"\b(appearance|reality|philosophy|knowledge|doubt|truth|sense-data|perception)\b", tl):
            return "philosophy"
        if re.search(r"\b(flood|fire|earthquake|survival|danger|safety)\b", tl):
            return "survival"
        if re.search(r"\b(task|plan|organize|priority|schedule)\b", tl):
            return "productivity"
        if re.search(r"\b(biology|human|water|science|physics|chemistry)\b", tl):
            return "general"

        return "general"

    def _dedupe_preserve_order(self, items: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for item in items:
            key = re.sub(r"\s+", " ", str(item).strip().lower())
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(str(item).strip())
        return out