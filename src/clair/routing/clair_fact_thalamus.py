# FILE: routing/clair_fact_thalamus.py
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import config

from intake.sensors import IntakeManager
from intake.processor import IntakeProcessor
from routing.thalamus_fact_router import ThalamusFactRouter
from memory.working_memory import WorkingMemory
from memory.long_term_memory import LongTermMemory
from planning.simulator import Simulator
from affect.risk_assessor import RiskAssessor
from decision.validator import DecisionValidator
from decision.reasoning import ReasoningEngine
from safety.hard_rules import HardRules
from execution.actuator import Actuator
from evaluation.performance import PerformanceEvaluator
from reflection.review import ReflectionEngine


class ClairFactThalamus:
    """
    Clair fact-thalamus orchestration layer.

    Responsibilities:
    - collect + process intake
    - route packets through thalamus
    - dispatch by route target
    - use WM/LTM as fact substrate
    - escalate verification / calibration / reflection when needed
    """

    VERSION = "2.1-fact-thalamus"

    def __init__(self):
        self.long_term = LongTermMemory()
        self.memory = WorkingMemory(
            max_history=getattr(config, "WORKING_MEMORY_MAX", 200),
            decay_rate=getattr(config, "MEMORY_DECAY_RATE", 0.98),
        )

        self.intake = IntakeManager()
        self.processor = IntakeProcessor()
        self.bus = ThalamusFactRouter()

        self.simulator = Simulator(
            exploration_rate=getattr(config, "SIMULATOR_EXPLORATION_RATE", 0.15),
            history_window=getattr(config, "SIMULATOR_HISTORY_WINDOW", 16),
        )
        self.risk_assessor = RiskAssessor()
        self.validator = DecisionValidator()
        self.reasoner = ReasoningEngine(
            simulator=self.simulator,
            risk_assessor=self.risk_assessor,
        )
        self.hard_rules = HardRules()
        self.actuator = Actuator()
        self.performance = PerformanceEvaluator()
        self.reflector = ReflectionEngine()

        self._bootstrap_identity()

        if getattr(config, "VERBOSE", False):
            print("[ClairFactThalamus] Online. Fact-thalamus orchestration initialized.")

    # ------------------------------------------------------------------
    # Bootstrapping
    # ------------------------------------------------------------------
    def _bootstrap_identity(self) -> None:
        try:
            existing = self.memory.retrieve("identity", count=1)
        except Exception:
            existing = []

        if existing:
            return

        self.memory.store([{
            "type": "identity",
            "content": "I am Clair, which stands for Cognitive Learning and Interactive Reasoner.",
            "confidence": 0.98,
            "weight": 1.0,
            "source": "system",
            "domain": "identity",
            "tags": ["identity", "self", "clair"],
            "kind": "fact",
            "details": {
                "verified": True,
                "pending_verification": False,
                "contested": False,
                "status": "verified",
                "verification_status": "verified",
                "source_trust": "trusted",
                "description": "I am Clair, designed to learn, reason, and support my user.",
                "meaning": "Cognitive Learning and Interactive Reasoner",
            },
        }])

    # ------------------------------------------------------------------
    # Identity query
    # ------------------------------------------------------------------
    def handle_identity_query(self) -> str:
        try:
            identity = self.memory.retrieve("identity", count=1)
        except Exception:
            identity = []

        if identity:
            i = identity[0]
            details = i.get("details", {}) if isinstance(i.get("details"), dict) else {}
            meaning = details.get("meaning", "Cognitive Learning and Interactive Reasoner")
            description = details.get(
                "description",
                "I am Clair, designed to learn, reason, and support my user.",
            )
            return f"My name is Clair ({meaning}). {description}"
        return "I am Clair."

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def main_loop(self) -> None:
        while True:
            raw_inputs = self.intake.collect()
            packets = self.processor.process(raw_inputs, modality="text", source="user")
            routed = self.bus.route_packets(packets)

            if getattr(config, "VERBOSE", False):
                print(
                    f"[ClairLoop] "
                    f"Forwarded={len(routed['forwarded'])} "
                    f"Deferred={len(routed['deferred'])} "
                    f"Rejected={len(routed['rejected'])} "
                    f"Recall={len(routed['fact_recall'])} "
                    f"Learn={len(routed['learning'])} "
                    f"Verify={len(routed['verification'])} "
                    f"Calibrate={len(routed['calibration'])} "
                    f"Reflect={len(routed['reflection'])} "
                    f"Answer={len(routed['direct_answer'])}"
                )

            for packet in routed["fact_recall"]:
                self._handle_fact_recall_packet(packet)

            for packet in routed["learning"]:
                self._handle_learning_packet(packet)

            for packet in routed["verification"]:
                self._handle_verification_packet(packet)

            for packet in routed["calibration"]:
                self._handle_calibration_packet(packet)

            for packet in routed["reflection"]:
                self._handle_reflection_packet(packet)

            for packet in routed["direct_answer"]:
                self._handle_direct_answer_packet(packet)

            self.consolidate_memory()
            time.sleep(getattr(config, "HEARTBEAT_INTERVAL", 0.25))

    # ------------------------------------------------------------------
    # Packet handlers
    # ------------------------------------------------------------------
    def _handle_fact_recall_packet(self, packet: Any) -> None:
        content = self._packet_text(packet)
        if not content:
            return

        if self._is_identity_query(content):
            self._emit_response(self.handle_identity_query())
            return

        wm_hits = self.memory.retrieve(
            msg_type=content,
            count=5,
            planning_only=False,
        )

        if wm_hits:
            response = self._format_memory_answer(content, wm_hits, source_name="working memory")
            self._emit_response(response)
            return

        ltm_hits = self.long_term.search(query=content, limit=5)
        if ltm_hits:
            try:
                self.memory.store(ltm_hits)
            except Exception:
                pass
            response = self._format_memory_answer(content, ltm_hits, source_name="long-term memory")
            self._emit_response(response)
            return

        answer = self.reasoner.answer_question(
            content,
            self.memory,
            max_actions=getattr(config, "REASONING_MAX_ACTIONS", 3),
        )
        self._post_reasoner_actions(content, answer)
        self._emit_response(answer.get("answer", "I could not find a reliable memory match."))

    def _handle_learning_packet(self, packet: Any) -> None:
        content = self._packet_text(packet)
        if not content:
            return

        self.memory.store([{
            "type": "observe",
            "content": content,
            "confidence": float(getattr(packet, "extraction_confidence", 0.75) or 0.75),
            "weight": 0.60,
            "source": getattr(packet, "source", "user") or "user",
            "domain": getattr(packet, "domain", None) or self._packet_signal(packet, "domain"),
            "tags": self._learning_tags_from_packet(packet),
            "kind": "fact",
            "details": {
                "verified": False,
                "pending_verification": True,
                "contested": False,
                "status": "unverified",
                "verification_status": "unverified",
                "source_trust": "normal",
                "route_target": self._packet_signal(packet, "route_target"),
            },
        }])

        if getattr(config, "VERBOSE", False):
            print(f"[Clair] Learned from packet: {content[:120]}")

    def _handle_verification_packet(self, packet: Any) -> None:
        content = self._packet_text(packet)
        if not content:
            return

        try:
            candidates = self.memory.calibration_candidates(limit=5)
        except Exception:
            candidates = []

        if getattr(config, "VERBOSE", False):
            print(f"[Clair] Verification requested. Candidate count={len(candidates)}")

        answer = self.reasoner.answer_question(
            content,
            self.memory,
            max_actions=max(1, int(getattr(config, "REASONING_MAX_ACTIONS", 3))),
        )
        self._emit_response(answer.get("answer", "Verification path triggered."))

    def _handle_calibration_packet(self, packet: Any) -> None:
        try:
            candidates = self.memory.calibration_candidates(limit=10)
        except Exception:
            candidates = []

        if getattr(config, "VERBOSE", False):
            print(f"[Clair] Calibration path triggered. Candidates={len(candidates)}")

        self._emit_response(f"Calibration queue contains {len(candidates)} candidate memories.")

    def _handle_reflection_packet(self, packet: Any) -> None:
        try:
            self.memory.reflect()
        except Exception as exc:
            if getattr(config, "VERBOSE", False):
                print(f"[Clair] Reflection error: {exc}")
            return

        self._emit_response("Reflection cycle completed.")

    def _handle_direct_answer_packet(self, packet: Any) -> None:
        content = self._packet_text(packet)
        if not content:
            return

        if self._is_identity_query(content):
            self._emit_response(self.handle_identity_query())
            return

        answer = self.reasoner.answer_question(
            content,
            self.memory,
            max_actions=getattr(config, "REASONING_MAX_ACTIONS", 3),
        )
        self._post_reasoner_actions(content, answer)
        self._emit_response(answer.get("answer", ""))

    # ------------------------------------------------------------------
    # Consolidation
    # ------------------------------------------------------------------
    def consolidate_memory(self) -> None:
        threshold = float(getattr(config, "LTM_AUTO_SYNC_WEIGHT", 0.75))
        promotable: List[Dict[str, Any]] = []

        for record in getattr(self.memory, "buffer", []) or []:
            try:
                persisted = bool(record.metadata.get("persisted", False))
                weight = float(record.metadata.get("weight", 0.0) or 0.0)
                mtype = str(record.metadata.get("type", "") or "").strip().lower()
            except Exception:
                continue

            if persisted:
                continue
            if weight < threshold:
                continue
            if mtype not in {
                "lesson", "observe", "reasoning_action", "fact",
                "claim", "policy", "chapter_summary", "section_summary"
            }:
                continue

            promotable.append(self.memory._record_to_legacy_dict(record))

        if promotable:
            self.long_term.store(promotable)
            for record in getattr(self.memory, "buffer", []) or []:
                try:
                    mtype = str(record.metadata.get("type", "") or "").strip().lower()
                    weight = float(record.metadata.get("weight", 0.0) or 0.0)
                    if (
                        not bool(record.metadata.get("persisted", False))
                        and weight >= threshold
                        and mtype in {
                            "lesson", "observe", "reasoning_action", "fact",
                            "claim", "policy", "chapter_summary", "section_summary"
                        }
                    ):
                        record.metadata["persisted"] = True
                except Exception:
                    continue

            if getattr(config, "VERBOSE", False):
                print(f"[Clair] Consolidated {len(promotable)} memories to long-term storage.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _packet_text(self, packet: Any) -> str:
        signals = getattr(packet, "signals", {}) or {}
        text = signals.get("normalized_text", "")
        return str(text or "").strip()

    def _packet_signal(self, packet: Any, key: str) -> Any:
        signals = getattr(packet, "signals", {}) or {}
        return signals.get(key)

    def _is_identity_query(self, content: str) -> bool:
        low = str(content or "").lower()
        return any(trigger in low for trigger in ("who are you", "your name", "what is your name"))

    def _learning_tags_from_packet(self, packet: Any) -> List[str]:
        tags: List[str] = ["learning", "intake"]
        domain = getattr(packet, "domain", None) or self._packet_signal(packet, "domain")
        hazard_family = getattr(packet, "hazard_family", None) or self._packet_signal(packet, "hazard_family")
        route_target = self._packet_signal(packet, "route_target")

        for value in (domain, hazard_family, route_target):
            if value:
                s = str(value).strip().lower()
                if s and s not in tags:
                    tags.append(s)
        return tags

    def _format_memory_answer(self, query: str, hits: List[Dict[str, Any]], source_name: str) -> str:
        if not hits:
            return "No relevant memory found."

        best = hits[0]
        content = str(best.get("content", "") or "").strip()
        if not content:
            return f"I found a {source_name} match, but it did not contain usable content."

        return f"From {source_name}: {content}"

    def _post_reasoner_actions(self, content: str, answer: Dict[str, Any]) -> None:
        if not isinstance(answer, dict):
            return

        if answer.get("suggested_actions"):
            updated_options = self.simulator.generate_options(
                self.memory,
                num_actions=getattr(config, "SIMULATOR_DEFAULT_NUM_ACTIONS", 5),
                question=content,
            )
            self.intake.queue_actions_from_reasoning(
                updated_options[: getattr(config, "REASONING_MAX_ACTIONS", 3)]
            )

    def _emit_response(self, response: Optional[str]) -> None:
        if not response:
            return
        if getattr(config, "VERBOSE", False):
            print(f"[Clair] Response: {response}")