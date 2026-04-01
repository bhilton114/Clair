# FILE: intake/sensors.py
from __future__ import annotations

import os
import re
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import config
from intake.contracts import InputPacket, UncertaintyFlags

# Default personal folder
PERSONAL_FOLDER = os.path.expanduser(getattr(config, "PERSONAL_FOLDER", "~\\ClairPersonal"))

# Intake guards
MAX_FILES_PER_TICK = int(getattr(config, "INTAKE_MAX_FILES_PER_TICK", 25))
MAX_FILE_BYTES = int(getattr(config, "INTAKE_MAX_FILE_BYTES", 2_000_000))  # 2 MB default
DELETE_FILES_AFTER_READ = bool(getattr(config, "INTAKE_DELETE_FILES_AFTER_READ", False))

# Supported file types
INTAKE_TEXT_EXTS = set(getattr(config, "INTAKE_TEXT_EXTS", {".txt", ".md", ".text"}))
INTAKE_PDF_EXTS = set(getattr(config, "INTAKE_PDF_EXTS", {".pdf"}))
INTAKE_ALLOWED_EXTS = INTAKE_TEXT_EXTS | INTAKE_PDF_EXTS


class IntakeManager:
    """
    Collects inputs from:
      1) PERSONAL_FOLDER files
      2) queued reasoning actions
      3) queued feedback messages

    Improvements:
    - Prevents re-reading unchanged files with mtime/size signature cache
    - Stable ordering for deterministic behavior
    - Supports .txt/.md and .pdf intake
    - PDF extraction fallback: pypdf -> pdfplumber
    - Readable-text filtering to avoid empty/junk ingestion
    - Optional delete-after-read
    """

    def __init__(self, personal_folder: Optional[str] = None):
        self.personal_folder = personal_folder or PERSONAL_FOLDER

        self._queued_actions: List[dict] = []
        self._feedback_queue: List[dict] = []

        # path -> (mtime, size)
        self._file_sig_cache: Dict[str, Tuple[float, int]] = {}

    # ------------------------
    # Public collection API
    # ------------------------
    def collect(self) -> List[InputPacket]:
        packets: List[InputPacket] = []

        packets.extend(self._collect_personal_folder_files())
        packets.extend(self._drain_queued_actions())
        packets.extend(self._drain_feedback())

        return packets

    # ------------------------
    # File intake
    # ------------------------
    def _collect_personal_folder_files(self) -> List[InputPacket]:
        packets: List[InputPacket] = []

        folder = self.personal_folder
        if not folder or not os.path.exists(folder):
            return packets

        try:
            filenames = sorted(os.listdir(folder))
        except Exception as e:
            print(f"[IntakeManager] Failed to list folder '{folder}': {e}")
            return packets

        taken = 0
        for fname in filenames:
            if taken >= MAX_FILES_PER_TICK:
                break

            fpath = os.path.join(folder, fname)
            if not os.path.isfile(fpath):
                continue

            ext = os.path.splitext(fname)[1].lower().strip()
            if ext not in INTAKE_ALLOWED_EXTS:
                continue

            try:
                st = os.stat(fpath)
            except Exception:
                continue

            if st.st_size <= 0:
                continue
            if st.st_size > MAX_FILE_BYTES:
                continue

            sig = (float(getattr(st, "st_mtime", 0.0)), int(getattr(st, "st_size", 0)))
            prev = self._file_sig_cache.get(fpath)
            if prev == sig:
                continue

            content, extractor, warnings = self._read_file_content(fpath, ext)
            if not content:
                if warnings:
                    for w in warnings[:3]:
                        print(f"[IntakeManager] {fname}: {w}")
                continue

            self._file_sig_cache[fpath] = sig

            packet = InputPacket(
                packet_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                modality="file",
                source="personal_folder",
                raw_input=content,
                signals={
                    "normalized_text": content,
                    "file_name": fname,
                    "file_path": fpath,
                    "folder_path": folder,
                    "file_ext": ext,
                    "file_mtime": sig[0],
                    "file_size": sig[1],
                    "extractor": extractor,
                    "warnings": warnings[:5],
                },
                constraints=[],
                uncertainty=UncertaintyFlags(),
                extraction_confidence=self._confidence_for_extractor(extractor, warnings),
            )
            packets.append(packet)
            taken += 1

            if DELETE_FILES_AFTER_READ:
                try:
                    os.remove(fpath)
                    self._file_sig_cache.pop(fpath, None)
                except Exception as e:
                    print(f"[IntakeManager] Failed to delete {fpath}: {e}")

        return packets

    def _read_file_content(self, fpath: str, ext: str) -> Tuple[str, str, List[str]]:
        if ext in INTAKE_TEXT_EXTS:
            return self._read_text_file(fpath)

        if ext in INTAKE_PDF_EXTS:
            return self._read_pdf_file(fpath)

        return "", "none", [f"Unsupported extension: {ext}"]

    def _read_text_file(self, fpath: str) -> Tuple[str, str, List[str]]:
        warnings: List[str] = []

        for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
            try:
                with open(fpath, "r", encoding=encoding, errors="replace") as f:
                    text = f.read() or ""
                text = self._normalize_text(text)

                if not self._is_readable_text(text):
                    warnings.append(f"text decode succeeded with {encoding}, but content was not readable")
                    continue

                return text, f"text:{encoding}", warnings
            except Exception as e:
                warnings.append(f"text decode failed with {encoding}: {e!r}")

        return "", "text:none", warnings

    def _read_pdf_file(self, fpath: str) -> Tuple[str, str, List[str]]:
        warnings: List[str] = []

        ok, text, err = self._extract_pdf_pypdf(fpath)
        if ok:
            text = self._normalize_text(text)
            if self._is_readable_text(text):
                return text, "pdf:pypdf", warnings
            warnings.append("pypdf returned low-quality or empty text")
        elif err:
            warnings.append(f"pypdf failed: {err}")

        ok, text, err = self._extract_pdf_pdfplumber(fpath)
        if ok:
            text = self._normalize_text(text)
            if self._is_readable_text(text):
                return text, "pdf:pdfplumber", warnings
            warnings.append("pdfplumber returned low-quality or empty text")
        elif err:
            warnings.append(f"pdfplumber failed: {err}")

        return "", "pdf:none", warnings

    def _extract_pdf_pypdf(self, fpath: str) -> Tuple[bool, str, Optional[str]]:
        try:
            from pypdf import PdfReader  # type: ignore
        except Exception as e:
            return False, "", f"import error: {e!r}"

        try:
            reader = PdfReader(fpath)
            parts: List[str] = []
            for i, page in enumerate(reader.pages):
                try:
                    txt = page.extract_text() or ""
                    if txt.strip():
                        parts.append(txt)
                except Exception as e:
                    parts.append(f"\n[PAGE {i + 1} ERROR {e!r}]\n")
            return True, "\n\n".join(parts).strip(), None
        except Exception as e:
            return False, "", repr(e)

    def _extract_pdf_pdfplumber(self, fpath: str) -> Tuple[bool, str, Optional[str]]:
        try:
            import pdfplumber  # type: ignore
        except Exception as e:
            return False, "", f"import error: {e!r}"

        try:
            parts: List[str] = []
            with pdfplumber.open(fpath) as pdf:
                for i, page in enumerate(pdf.pages):
                    try:
                        txt = page.extract_text() or ""
                        if txt.strip():
                            parts.append(txt)
                    except Exception as e:
                        parts.append(f"\n[PAGE {i + 1} ERROR {e!r}]\n")
            return True, "\n\n".join(parts).strip(), None
        except Exception as e:
            return False, "", repr(e)

    def _normalize_text(self, text: str) -> str:
        if not text:
            return ""

        text = text.replace("\x00", " ")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _is_readable_text(self, text: str) -> bool:
        if not text:
            return False

        s = text.strip()
        if len(s) < 120:
            return False

        visible = sum(1 for ch in s if not ch.isspace())
        alpha = sum(1 for ch in s if ch.isalpha())
        digit = sum(1 for ch in s if ch.isdigit())

        if visible < 80:
            return False

        alpha_ratio = alpha / max(1, len(s))
        signal_ratio = (alpha + digit) / max(1, len(s))

        if alpha_ratio < 0.20 and signal_ratio < 0.28:
            return False

        # reject mostly parser rubble
        lowered = s.lower()
        if lowered.count("[page") > 5 and alpha < 200:
            return False

        return True

    def _confidence_for_extractor(self, extractor: str, warnings: List[str]) -> float:
        base = 0.9
        if extractor.startswith("pdf:pdfplumber"):
            base = 0.86
        elif extractor.startswith("pdf:pypdf"):
            base = 0.88
        elif extractor.startswith("text:"):
            base = 0.92
        elif extractor == "none":
            base = 0.25

        if warnings:
            base -= min(0.15, 0.03 * len(warnings))

        return max(0.25, min(0.98, base))

    # ------------------------
    # Queues
    # ------------------------
    def _drain_queued_actions(self) -> List[InputPacket]:
        packets: List[InputPacket] = []
        now = datetime.utcnow()

        queued = self._queued_actions or []
        self._queued_actions = []

        for act in queued:
            if not isinstance(act, dict):
                continue

            content = (act.get("content") or "").strip()
            if not content:
                continue

            packets.append(
                InputPacket(
                    packet_id=str(uuid.uuid4()),
                    timestamp=act.get("timestamp", now),
                    modality=act.get("modality", "reasoning_action"),
                    source=str(act.get("source", "system")),
                    raw_input=content,
                    signals={
                        "normalized_text": content,
                        "queued_kind": "reasoning_action",
                    },
                    constraints=list(act.get("constraints", []) or []),
                    uncertainty=UncertaintyFlags(),
                    extraction_confidence=float(act.get("confidence", 1.0) or 1.0),
                )
            )

        return packets

    def _drain_feedback(self) -> List[InputPacket]:
        packets: List[InputPacket] = []
        now = datetime.utcnow()

        queued = self._feedback_queue or []
        self._feedback_queue = []

        for fb in queued:
            if not isinstance(fb, dict):
                continue

            content = (fb.get("content") or "").strip()
            if not content:
                continue

            packets.append(
                InputPacket(
                    packet_id=str(uuid.uuid4()),
                    timestamp=fb.get("timestamp", now),
                    modality=fb.get("modality", "feedback"),
                    source=str(fb.get("source", "system")),
                    raw_input=content,
                    signals={
                        "normalized_text": content,
                        "queued_kind": "feedback",
                    },
                    constraints=list(fb.get("constraints", []) or []),
                    uncertainty=UncertaintyFlags(),
                    extraction_confidence=float(fb.get("confidence", 1.0) or 1.0),
                )
            )

        return packets

    # ------------------------
    # Queuing APIs
    # ------------------------
    def queue_actions_from_reasoning(self, actions: List[dict]):
        """Queue suggested actions from reasoning engine for next cycle."""
        if not actions:
            return
        if not isinstance(actions, list):
            actions = [actions]

        stamped: List[dict] = []
        now = datetime.utcnow()
        for a in actions:
            if not isinstance(a, dict):
                continue
            item = dict(a)
            item.setdefault("timestamp", now)
            stamped.append(item)

        self._queued_actions.extend(stamped)

    def queue_feedback(self, feedback_messages: List[dict]):
        """Queue feedback messages for processing in next cycle."""
        if not feedback_messages:
            return
        if not isinstance(feedback_messages, list):
            feedback_messages = [feedback_messages]

        stamped: List[dict] = []
        now = datetime.utcnow()
        for m in feedback_messages:
            if not isinstance(m, dict):
                continue
            item = dict(m)
            item.setdefault("timestamp", now)
            stamped.append(item)

        self._feedback_queue.extend(stamped)