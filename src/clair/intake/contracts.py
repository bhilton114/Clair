# FILE: intake/contracts.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid


@dataclass
class UncertaintyFlags:
    ambiguous_language: bool = False
    metaphor_detected: bool = False
    missing_references: bool = False
    conflicting_signals: bool = False
    low_signal_quality: bool = False


@dataclass
class InputPacket:
    """
    Canonical intake contract.
    This is the ONLY object allowed to exit intake.

    Contract upgrades (backwards compatible):
    - domain: optional upstream domain classification ("general", "survival", "identity", etc.)
    - hazard_family: optional upstream hazard family pin ("fire", "flood", "earthquake", "lost")
      If set, downstream modules should prefer this over token inference.

    Note: signals remains the extensible container for model-specific extraction.
    """

    # ------------------------
    # Unique identity & timing
    # ------------------------
    packet_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # ------------------------
    # Source & modality
    # ------------------------
    modality: str = "unknown"  # text | file | sensor | scenario | memory | external
    source: Optional[str] = None

    # ------------------------
    # Raw input
    # ------------------------
    raw_input: Any = None

    # ------------------------
    # Extracted signals (normalized, segmented, etc.)
    # ------------------------
    signals: Dict[str, Any] = field(default_factory=dict)

    # ------------------------
    # Optional file/folder metadata
    # ------------------------
    file_name: Optional[str] = None
    file_path: Optional[str] = None
    folder_path: Optional[str] = None

    # ------------------------
    # Optional semantic routing pins
    # ------------------------
    domain: Optional[str] = None
    hazard_family: Optional[str] = None

    # ------------------------
    # Uncertainty and constraints
    # ------------------------
    uncertainty: UncertaintyFlags = field(default_factory=UncertaintyFlags)
    constraints: List[str] = field(default_factory=list)

    # ------------------------
    # Confidence in extraction quality (0.0–1.0)
    # ------------------------
    extraction_confidence: float = 0.0

    # ------------------------
    # Utility methods
    # ------------------------
    def is_viable(self) -> bool:
        """
        Determines whether this packet is structurally valid enough to proceed.
        Does NOT imply correctness.
        """
        if self.raw_input is None:
            return False
        if not isinstance(self.signals, dict):
            return False
        return True

    def normalize_file_metadata(self) -> None:
        """
        Ensures file/folder signals are mirrored in top-level attributes.
        """
        if "file_name" in self.signals:
            self.file_name = self.signals.get("file_name")
        if "file_path" in self.signals:
            self.file_path = self.signals.get("file_path")
        if "folder_path" in self.signals:
            self.folder_path = self.signals.get("folder_path")

    def normalize_semantic_pins(self) -> None:
        """
        Mirrors optional semantic routing pins from signals to top-level attributes.
        This keeps intake flexible while giving downstream systems stable fields.
        """
        d = self.signals.get("domain")
        if isinstance(d, str) and d.strip():
            self.domain = d.strip().lower()

        hf = self.signals.get("hazard_family")
        if isinstance(hf, str) and hf.strip():
            self.hazard_family = hf.strip().lower()