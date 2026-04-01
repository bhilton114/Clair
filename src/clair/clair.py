# FILE: clair.py
# PART 1 OF 2
#
# Clair (v2.64 - clubhouse locality hardening rewrite)
#
# This part contains:
# - imports and bootstrapping
# - packet / response helpers
# - class config/constants
# - __init__
# - utility helpers
# - loop balance helpers
# - reading / summary helpers
# - recall / survival helpers
# - lesson / fact helpers
# - mode helpers
# - verification / calibration helpers
#
# Part 2 continues with:
# - seeding / output / context helpers
# - query helpers
# - handle_packet
# - reflection / action cycle
# - document ingest
# - CLI loop

from __future__ import annotations

import copy
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import config

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

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

from comms.dialogue_state import DialogueState
from comms.broca import Broca, BrocaConfig

from intake.document_reader import DocumentReader
from learning.angular_gyrus import AngularGyrus
from learning.hippocampus_ingest import HippocampusIngestor
from reflection.pfc_reviewer import PFCReviewer

try:
    from executive.goal_manager import GoalManager
    from executive.priority_manager import PriorityManager
except Exception:
    GoalManager = None  # type: ignore
    PriorityManager = None  # type: ignore

try:
    from affect.hypothalamus import Hypothalamus, HypothalamusConfig
except Exception:
    Hypothalamus = None  # type: ignore
    HypothalamusConfig = None  # type: ignore

try:
    from verification import ThalamusVerifier
except Exception:
    ThalamusVerifier = None  # type: ignore

try:
    from calibration.cerebellar import Cerebellar
except Exception:
    Cerebellar = None  # type: ignore

AnteriorCingulateCortex = None  # type: ignore
try:
    from calibration.ACC import AnteriorCingulateCortex
except Exception:
    try:
        from calibration.ACC import ACC as AnteriorCingulateCortex
    except Exception:
        try:
            from calibration.Acc import AnteriorCingulateCortex
        except Exception:
            try:
                from calibration.Acc import ACC as AnteriorCingulateCortex
            except Exception:
                AnteriorCingulateCortex = None  # type: ignore


class SimplePacket:
    def __init__(self, text: str, ptype: str = "ask", packet_id: str = "cli") -> None:
        self.packet_id = packet_id
        self.raw_input = {"type": ptype, "content": text}
        self.signals = {"normalized_text": text}
        self.retry_count = 0
        self.next_eligible_time = 0.0

    def is_viable(self) -> bool:
        return True


class ResponseManager:
    @staticmethod
    def clean(response: Any) -> str:
        if response is None:
            return ""
        text = " ".join(str(response).strip().split())
        if not text:
            return ""
        if text[-1] not in ".!?":
            text += "."
        return text

    @staticmethod
    def limit_sentences(text: str, max_sentences: int = 3) -> str:
        text = ResponseManager.clean(text)
        if not text:
            return ""
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        parts = [p.strip() for p in parts if p.strip()]
        return " ".join(parts[: max(1, int(max_sentences))]).strip()


class Clair:
    @staticmethod
    def _cfg(name: str, default: Any) -> Any:
        return getattr(config, name, default)

    MAX_DEFER_RETRIES = int(_cfg.__func__("MAX_DEFER_RETRIES", 3))
    HEARTBEAT_INTERVAL = float(_cfg.__func__("HEARTBEAT_INTERVAL", 1.0))
    REFLECTION_INTERVAL = float(_cfg.__func__("REFLECTION_INTERVAL", HEARTBEAT_INTERVAL))

    ACTION_QUESTION_FRESH_SEC = float(_cfg.__func__("ACTION_QUESTION_FRESH_SEC", 8.0))
    ACTION_URGENCY_THRESHOLD = float(_cfg.__func__("ACTION_URGENCY_THRESHOLD", 0.65))
    ACTION_THREAT_THRESHOLD = float(_cfg.__func__("ACTION_THREAT_THRESHOLD", 0.65))
    ACTION_CYCLE_COOLDOWN_SEC = float(_cfg.__func__("ACTION_CYCLE_COOLDOWN_SEC", 2.0))

    REQUIRE_ACTION_WORTHY_TICK = bool(_cfg.__func__("REQUIRE_ACTION_WORTHY_TICK", True))
    BLOCK_ACTION_CYCLE_AFTER_FACT_RECALL = bool(
        _cfg.__func__("BLOCK_ACTION_CYCLE_AFTER_FACT_RECALL", True)
    )
    QUARANTINE_FEEDBACK = bool(_cfg.__func__("QUARANTINE_FEEDBACK", True))

    SPEAK_ACTION_CYCLE = bool(_cfg.__func__("SPEAK_ACTION_CYCLE", True))
    MAX_RESPONSE_SENTENCES = int(_cfg.__func__("MAX_RESPONSE_SENTENCES", 3))

    DIRECT_RECALL_CONFIDENCE = float(_cfg.__func__("DIRECT_RECALL_CONFIDENCE", 0.85))
    CAUTIOUS_RECALL_CONFIDENCE = float(_cfg.__func__("CAUTIOUS_RECALL_CONFIDENCE", 0.55))
    ALLOW_CAUTIOUS_PENDING_RECALL = bool(
        _cfg.__func__("ALLOW_CAUTIOUS_PENDING_RECALL", True)
    )

    DOC_WORDS_PER_CHUNK = int(_cfg.__func__("DOC_WORDS_PER_CHUNK", 900))
    DOC_MAX_CLAIMS_PER_CHUNK = int(_cfg.__func__("DOC_MAX_CLAIMS_PER_CHUNK", 8))
    DOC_PERSIST_TO_LTM = bool(_cfg.__func__("DOC_PERSIST_TO_LTM", False))
    DOC_DEBUG_PREVIEW = bool(_cfg.__func__("DOC_DEBUG_PREVIEW", False))
    DOC_SECTION_SUMMARY_CLAIMS = int(_cfg.__func__("DOC_SECTION_SUMMARY_CLAIMS", 2))
    DOC_CHAPTER_SUMMARY_CLAIMS = int(_cfg.__func__("DOC_CHAPTER_SUMMARY_CLAIMS", 3))
    DOC_VERIFY_ON_INGEST = bool(_cfg.__func__("DOC_VERIFY_ON_INGEST", True))
    DOC_VERIFY_MAX_PER_CHUNK = int(_cfg.__func__("DOC_VERIFY_MAX_PER_CHUNK", 4))

    VERIFY_ROUTE_LOW_CONFIDENCE = float(_cfg.__func__("VERIFY_ROUTE_LOW_CONFIDENCE", 0.75))
    VERIFY_ROUTE_CONFLICT_ALWAYS_EXTERNAL = bool(
        _cfg.__func__("VERIFY_ROUTE_CONFLICT_ALWAYS_EXTERNAL", True)
    )
    LOOP_IMBALANCE_RATIO = float(_cfg.__func__("LOOP_IMBALANCE_RATIO", 3.0))
    LOOP_CALIBRATION_URGENCY_BOOST = float(
        _cfg.__func__("LOOP_CALIBRATION_URGENCY_BOOST", 0.20)
    )

    ENABLE_HYPOTHALAMUS = bool(_cfg.__func__("ENABLE_HYPOTHALAMUS", True))
    LOG_MODE_TRANSITIONS = bool(_cfg.__func__("LOG_MODE_TRANSITIONS", True))

    PLANNING_TRIGGERS = (
        "plan",
        "steps",
        "strategy",
        "roadmap",
        "schedule",
        "how do i",
        "how should i",
        "what should i do",
        "fix",
        "debug",
        "implement",
        "build",
        "rewrite",
        "next",
        "then",
        "after that",
    )

    SURVIVAL_TOKENS = {
        "fire",
        "flood",
        "earthquake",
        "lost",
        "emergency",
        "danger",
        "risk",
        "smoke",
        "evacuate",
        "aftershock",
        "shelter",
        "submerged",
    }

    SURVIVAL_MEMORY_TAGS = {
        "survival",
        "fire",
        "flood",
        "earthquake",
        "lost",
        "emergency",
        "smoke",
        "evacuate",
        "aftershock",
        "flooding",
        "floodwater",
        "flash",
        "collapse",
        "burning",
        "shelter",
    }

    FILE_EXTS = (".pdf", ".txt", ".md", ".docx", ".rtf")

    LITERATURE_QUOTE_INTENT = (
        "best line",
        "best quote",
        "quote",
        "quotes",
        "line",
        "lines",
        "passage",
        "excerpt",
        "stanza",
        "verse",
        "what did it say",
    )

    LITERATURE_QUERY_TOKENS = (
        "chapter",
        "book",
        "reading",
        "read",
        "lesson",
        "argument",
        "main argument",
        "theme",
        "themes",
        "summary",
        "summarize",
        "what was it about",
        "what is it about",
        "what did it say",
        "appearance",
        "reality",
        "sense-data",
        "perception",
        "russell",
    )

    HAZARD_FAMILIES = {
        "fire": {"fire", "smoke", "burn", "burning", "flame", "exit", "leave", "low"},
        "flood": {
            "flood",
            "flooding",
            "floodwater",
            "flash",
            "submerged",
            "evacuate",
            "higher",
            "ground",
            "rising",
        },
        "earthquake": {"earthquake", "aftershock", "shaking", "collapse", "quake"},
        "lost": {"lost", "night", "shelter", "signal", "conserve", "stay", "put"},
    }

    EMOTION_MODES = {"curiosity", "judging", "predicting", "idling", "rush"}

    def __init__(self) -> None:
        self.long_term = LongTermMemory()
        self.memory = WorkingMemory(
            max_history=int(self._cfg("WORKING_MEMORY_MAX", 200)),
            decay_rate=float(self._cfg("MEMORY_DECAY_RATE", 0.98)),
            preload_long_term=False,
        )
        self.memory.long_term = self.long_term
        self.memory._load_long_term()

        self.intake = IntakeManager()
        self.processor = IntakeProcessor()
        self.thalamus = ThalamusFactRouter()
        self.thalamus_verifier = (
            ThalamusVerifier(config=config) if ThalamusVerifier is not None else None
        )

        self.risk_assessor = RiskAssessor()

        self.goal_manager = GoalManager() if GoalManager is not None else None
        self.priority_manager = PriorityManager() if PriorityManager is not None else None

        self.hypothalamus = None
        self.current_mode: str = "idling"
        self.last_mode: str = "idling"
        self.mode_bias: Dict[str, float] = {}
        self.last_signals: Dict[str, float] = {}

        if self.ENABLE_HYPOTHALAMUS and Hypothalamus is not None:
            try:
                self.hypothalamus = Hypothalamus(HypothalamusConfig())
                if hasattr(self.hypothalamus, "get_biases"):
                    self.mode_bias = self.hypothalamus.get_biases() or {}  # type: ignore
            except Exception:
                self.hypothalamus = None
                self.mode_bias = {}

        self.simulator = Simulator(
            exploration_rate=float(self._cfg("SIMULATOR_EXPLORATION_RATE", 0.2)),
            history_window=int(self._cfg("SIMULATOR_HISTORY_WINDOW", 25)),
            risk_assessor=self.risk_assessor,
            goal_manager=self.goal_manager,
            priority_manager=self.priority_manager,
            mode_getter=self._get_mode,
            system_state_getter=self._get_system_state,
        )

        self.validator = DecisionValidator()
        self.reasoner = ReasoningEngine(simulator=self.simulator, reinforcement_enabled=True)
        self.hard_rules = HardRules()
        self.actuator = Actuator(verbose=bool(self._cfg("ACTUATOR_VERBOSE", False)))
        self.performance = PerformanceEvaluator()
        self.reflector = ReflectionEngine()
        self.response_manager = ResponseManager()

        self.dialogue = DialogueState()
        self.broca = Broca(BrocaConfig(show_reasoning=False))

        self.doc_reader = DocumentReader()
        self.angular_gyrus = AngularGyrus()
        self.hippocampus = HippocampusIngestor(self.memory, self.long_term)
        self.pfc = PFCReviewer(self.memory)

        self.cerebellar = None
        self.acc = None
        self._pending_calibration: Optional[Dict[str, Any]] = None

        if AnteriorCingulateCortex is not None:
            try:
                try:
                    self.acc = AnteriorCingulateCortex(self.memory, self.long_term)  # type: ignore
                except TypeError:
                    try:
                        self.acc = AnteriorCingulateCortex(self.long_term)  # type: ignore
                    except TypeError:
                        self.acc = AnteriorCingulateCortex(self)  # type: ignore
            except Exception:
                self.acc = None

        if Cerebellar is not None:
            try:
                try:
                    self.cerebellar = Cerebellar(
                        self.memory,
                        self.long_term,
                        self.acc,
                        config=config,
                    )  # type: ignore
                except TypeError:
                    try:
                        self.cerebellar = Cerebellar(self, self.acc, config=config)  # type: ignore
                    except TypeError:
                        try:
                            self.cerebellar = Cerebellar(self.long_term, self.acc, config=config)  # type: ignore
                        except TypeError:
                            try:
                                self.cerebellar = Cerebellar(self.long_term, config=config)  # type: ignore
                            except TypeError:
                                self.cerebellar = Cerebellar(self.long_term)  # type: ignore
            except Exception:
                self.cerebellar = None

        self.current_context_profile: Dict[str, Any] = {
            "domain": "general",
            "tags": [],
            "threat": 0.0,
            "urgency": 0.1,
            "goal": "inform",
        }

        self.last_question: Optional[str] = None
        self.last_question_ts = 0.0
        self.last_input_ts = 0.0
        self.last_question_context_profile: Optional[Dict[str, Any]] = None
        self.last_answer_was_recall = False

        self.reasoning_context: Dict[str, List[str]] = {}
        self.last_reasoning_packet_id: Optional[str] = None

        self.deferred_queue: List[Any] = []
        self.last_reflection = 0.0
        self.last_action_cycle_ts = 0.0

        self.learnmode = False

        self.reasoning_cycles = 0
        self.learning_events = 0
        self.calibration_events = 0

        self._seed_identity()
        self.seed_verified_facts()

        if bool(self._cfg("VERBOSE", False)):
            print(
                "Clair_v2.64 online. Clubhouse locality hardening + revision precision hooked."
            )

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _clamp01(x: Any) -> float:
        try:
            v = float(x)
        except Exception:
            return 0.0
        return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

    @staticmethod
    def _tokenize_simple(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9_]+", (text or "").lower()))

    @staticmethod
    def _dedupe_preserve_order(items: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for item in items or []:
            s = str(item or "").strip()
            if not s:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
        return out

    @staticmethod
    def _sentence_case(text: str) -> str:
        s = re.sub(r"\s+", " ", str(text or "").strip())
        if not s:
            return ""
        if len(s) == 1:
            return s.upper()
        return s[0].upper() + s[1:]

    @staticmethod
    def _safe_label_piece(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip())

    @staticmethod
    def _strip_leading_summary_fluff(text: str) -> str:
        s = str(text or "").strip()
        if not s:
            return ""
        fluff_patterns = [
            r"^(this\s+(section|chapter|lesson|material|reading)\s+(is\s+mainly\s+about|is\s+about|covers|discusses|explains)\s+)",
            r"^(the\s+(section|chapter|lesson|material|reading)\s+(is\s+mainly\s+about|is\s+about|covers|discusses|explains)\s+)",
            r"^(themes\s+include\s+)",
            r"^(key\s+points\s*:\s*)",
        ]
        out = s
        for pat in fluff_patterns:
            out = re.sub(pat, "", out, flags=re.IGNORECASE)
        return out.strip(" .:-")

    @classmethod
    def _normalize_factish_sentence(cls, text: str) -> str:
        s = str(text or "").strip()
        if not s:
            return ""
        s = re.sub(r"\s+", " ", s)
        s = s.strip(" \t\r\n-–—:;,.…")
        s = cls._strip_leading_summary_fluff(s)
        s = s.strip(" \t\r\n-–—:;,.…")
        if not s:
            return ""
        s = cls._sentence_case(s)
        if s[-1] not in ".!?":
            s += "."
        return s

    def _compress_claim_text(self, text: str, max_words: int = 24) -> str:
        s = self._normalize_factish_sentence(text)
        if not s:
            return ""
        words = s[:-1].split() if s.endswith((".", "!", "?")) else s.split()
        if len(words) > max_words:
            s = " ".join(words[:max_words]).rstrip(" ,;:") + "."
        return s

    def _extract_claim_texts(self, claims: List[Any]) -> List[str]:
        out: List[str] = []
        for item in claims or []:
            if isinstance(item, str):
                s = item.strip()
            elif isinstance(item, dict):
                s = str(item.get("text") or item.get("content") or "").strip()
            else:
                s = ""
            s = self._compress_claim_text(s)
            if s:
                out.append(s)
        return self._dedupe_preserve_order(out)

    def _mem_details(self, mem: Dict[str, Any]) -> Dict[str, Any]:
        details = mem.get("details")
        if not isinstance(details, dict):
            details = {}
            mem["details"] = details
        return details

    def _mem_truth(self, mem: Dict[str, Any]) -> Dict[str, Any]:
        details = self._mem_details(mem)
        source = str(mem.get("source", "") or "").strip().lower()
        status = str(details.get("status", "") or mem.get("status", "") or "").strip().lower()

        verified = bool(details.get("verified", False))
        if source in {"seed_verified", "verified", "verification"}:
            verified = True
        if status == "verified":
            verified = True

        pending = bool(details.get("pending_verification", False)) or status in {
            "unverified",
            "pending",
            "provisional",
        }
        contested = bool(details.get("contested", False)) or status == "contested" or bool(
            mem.get("conflict", False)
        )
        superseded = bool(details.get("superseded", False)) or status == "deprecated"
        recall_blocked = bool(details.get("recall_blocked", False)) or contested or superseded

        return {
            "verified": verified,
            "pending": pending,
            "contested": contested,
            "superseded": superseded,
            "recall_blocked": recall_blocked,
            "confidence": float(mem.get("confidence", 0.0) or 0.0),
            "source": source,
            "status": status or ("verified" if verified else "unverified" if pending else "normal"),
            "last_verified": mem.get("last_verified", details.get("last_verified")),
            "evidence": mem.get("evidence", details.get("evidence", [])),
        }

    def _is_probably_binary(self, s: str) -> bool:
        if not s:
            return False
        t = str(s)
        if not t.strip():
            return False
        low = t.lower()
        if "%pdf" in low or "endobj" in low or "xref" in low or "endstream" in low or "stream" in low:
            return True
        printable = sum(1 for ch in t if (32 <= ord(ch) <= 126) or ch in "\n\r\t")
        ratio = printable / max(1, len(t))
        if ratio < 0.78:
            return True
        if max((len(w) for w in re.findall(r"\S+", t)), default=0) > 120:
            return True
        return False

    def _safe_output(self, text: str) -> str:
        if self._is_probably_binary(text):
            return "That looks like document bytes, not language. Use `read:` to ingest a file."
        return str(text)

    # ------------------------------------------------------------------
    # Loop balance helpers
    # ------------------------------------------------------------------
    def _record_reasoning_cycle(self, n: int = 1) -> None:
        self.reasoning_cycles += max(1, int(n))

    def _record_learning_event(self, n: int = 1) -> None:
        self.learning_events += max(1, int(n))

    def _record_calibration_event(self, n: int = 1) -> None:
        self.calibration_events += max(1, int(n))

    def _calibration_priority_due_to_imbalance(self) -> bool:
        base = max(1.0, float(self.calibration_events))
        return float(self.learning_events) > (base * float(self.LOOP_IMBALANCE_RATIO))

    def _apply_loop_balance_pressure(self, cp: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        out = dict(cp or {})
        if self._calibration_priority_due_to_imbalance():
            out["needs_calibration_priority"] = True
            out["urgency"] = min(
                1.0,
                float(out.get("urgency", 0.1) or 0.1)
                + float(self.LOOP_CALIBRATION_URGENCY_BOOST),
            )
            tags = list(out.get("tags") or [])
            if "calibration_priority" not in [str(t).strip().lower() for t in tags]:
                tags.append("calibration_priority")
            out["tags"] = tags
        return out

    # ------------------------------------------------------------------
    # Reading helpers
    # ------------------------------------------------------------------
    def _extract_frame_candidates_for_ingest(
        self,
        text: str,
        doc_meta: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []

        frames: List[str] = []
        tl = text.lower()

        heading = None
        if isinstance(doc_meta, dict):
            heading = doc_meta.get("heading_hint")
        if heading:
            frames.append(f"heading:{heading}")

        frame_patterns = {
            "appearance_vs_reality": r"\bappearance\b.*\breality\b|\breality\b.*\bappearance\b",
            "knowledge_and_doubt": r"\bknowledge\b|\bdoubt\b|\bcertain(?:ty)?\b|\buncertain(?:ty)?\b",
            "perception_and_truth": r"\bperception\b|\btruth\b|\bsense-data\b|\bsenses\b|\bseem(?:s|ing)?\b",
            "cause_and_reason": r"\bbecause\b|\btherefore\b|\bthus\b|\breason\b",
            "ethics_and_action": r"\bgood\b|\bbad\b|\bright\b|\bwrong\b|\bought\b|\bshould\b",
            "strategy_and_conflict": r"\bstrategy\b|\bconflict\b|\badvantage\b|\bopponent\b",
            "survival_and_risk": r"\brisk\b|\bdanger\b|\bsafe(?:ty)?\b|\bsurvival\b",
        }

        for name, pat in frame_patterns.items():
            if re.search(pat, tl):
                frames.append(name)

        seen = set()
        out: List[str] = []
        for f in frames:
            key = str(f).strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(str(f).strip())
        return out

    def _make_summary_label(
        self,
        doc_meta: Optional[Dict[str, Any]] = None,
        include_section: bool = True,
    ) -> str:
        meta = doc_meta or {}
        title = self._safe_label_piece(
            meta.get("book_title_hint")
            or meta.get("document_title")
            or meta.get("title")
            or meta.get("filename")
            or "document"
        )
        chapter = self._safe_label_piece(meta.get("chapter_hint"))
        section = self._safe_label_piece(meta.get("section_hint"))

        parts: List[str] = [title]
        if chapter:
            parts.append(f"Chapter {chapter}")
        if include_section and section:
            parts.append(f"Section {section}")
        return " | ".join([p for p in parts if p])

    def _score_summary_candidate(self, text: str, frames: List[str]) -> float:
        s = str(text or "").strip()
        if not s:
            return -999.0

        words = s.split()
        word_count = len(words)
        if word_count < 6:
            return -50.0

        score = 0.0
        score += min(word_count, 24) * 0.20
        score -= max(0, word_count - 28) * 0.35

        lower = s.lower()
        fluff_hits = 0
        for bad in (
            "this section",
            "this chapter",
            "this lesson",
            "this material",
            "themes include",
            "key points",
        ):
            if bad in lower:
                fluff_hits += 1
        score -= fluff_hits * 2.0

        if any(ch.isdigit() for ch in s):
            score += 0.5
        if any(f for f in (frames or []) if not str(f).startswith("heading:")):
            score += 0.5
        if lower.startswith("document"):
            score -= 1.0
        return score

    def _build_dense_summary_body(
        self,
        claim_texts: List[str],
        frames: List[str],
        max_claims: int,
    ) -> Optional[str]:
        claims = self._dedupe_preserve_order(claim_texts)[: max(1, int(max_claims))]
        if claims:
            joined = re.sub(r"\s+", " ", " ".join(claims)).strip()
            if joined:
                return joined

        thematic_frames = [
            str(f).strip().replace("_", " ")
            for f in (frames or [])
            if not str(f).startswith("heading:")
        ][:2]
        thematic_frames = self._dedupe_preserve_order(thematic_frames)

        if thematic_frames:
            if len(thematic_frames) == 1:
                return self._normalize_factish_sentence(f"Focus: {thematic_frames[0]}")
            return self._normalize_factish_sentence(
                f"Focus: {thematic_frames[0]} and {thematic_frames[1]}"
            )

        return None

    def _make_section_summary_for_ingest(
        self,
        *,
        text: str,
        claims: List[Any],
        frames: List[str],
        doc_meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        doc_meta = doc_meta or {}
        claim_texts = self._extract_claim_texts(claims)
        body = self._build_dense_summary_body(
            claim_texts,
            frames,
            self.DOC_SECTION_SUMMARY_CLAIMS,
        )

        if not body:
            snippet = re.sub(r"\s+", " ", (text or "").strip())
            if snippet:
                snippet_words = snippet.split()
                snippet = " ".join(snippet_words[:20]).strip(" ,;:")
                if snippet:
                    body = self._normalize_factish_sentence(snippet)

        if not body:
            return None

        label = self._make_summary_label(doc_meta, include_section=True)
        summary = f"{label}: {body}"
        return summary if len(summary) >= 20 else None

    def _make_chapter_summary_for_ingest(
        self,
        *,
        section_summary: Optional[str],
        frames: List[str],
        doc_meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        doc_meta = doc_meta or {}
        if not section_summary:
            return None

        label = self._make_summary_label(doc_meta, include_section=False)
        section_body = (
            section_summary.split(":", 1)[1].strip()
            if ":" in section_summary
            else section_summary.strip()
        )
        section_body = self._strip_leading_summary_fluff(section_body)

        claims_like = [
            self._normalize_factish_sentence(x)
            for x in re.split(r"(?<=[.!?])\s+", section_body)
            if x.strip()
        ]
        claims_like = [x for x in claims_like if x]
        claims_like = self._dedupe_preserve_order(claims_like)[
            : max(1, self.DOC_CHAPTER_SUMMARY_CLAIMS)
        ]

        thematic_frames = [
            str(f).strip().replace("_", " ")
            for f in (frames or [])
            if not str(f).startswith("heading:")
        ][:2]
        thematic_frames = self._dedupe_preserve_order(thematic_frames)

        body_parts: List[str] = []
        if claims_like:
            body_parts.append(" ".join(claims_like))
        if thematic_frames:
            if len(thematic_frames) == 1:
                body_parts.append(
                    self._normalize_factish_sentence(f"Focus: {thematic_frames[0]}")
                )
            else:
                body_parts.append(
                    self._normalize_factish_sentence(
                        f"Focus: {thematic_frames[0]} and {thematic_frames[1]}"
                    )
                )

        body_parts = [p.strip() for p in body_parts if p.strip()]
        if not body_parts:
            return None

        body = " ".join(self._dedupe_preserve_order(body_parts))
        summary = f"{label} summary: {body}"
        return summary if len(summary) >= 20 else None

    # ------------------------------------------------------------------
    # Recall helpers
    # ------------------------------------------------------------------
    def _mem_kindish(self, mem: Dict[str, Any]) -> str:
        return str(mem.get("kind") or mem.get("type") or "").strip().lower()

    def _mem_tag_set(self, mem: Dict[str, Any]) -> set[str]:
        return {
            str(t).strip().lower()
            for t in (mem.get("tags") or [])
            if str(t).strip()
        }

    def _summary_text_cleanup(self, text: str) -> str:
        s = self.response_manager.clean(text)
        if not s:
            return ""
        s = re.sub(
            r"\bThis material is mainly about\b",
            "Main idea:",
            s,
            flags=re.IGNORECASE,
        )
        s = re.sub(
            r"\bThis section is mainly about\b",
            "Main idea:",
            s,
            flags=re.IGNORECASE,
        )
        s = re.sub(
            r"\bThis chapter is mainly about\b",
            "Main idea:",
            s,
            flags=re.IGNORECASE,
        )
        s = re.sub(r"\bThemes include\b", "Themes:", s, flags=re.IGNORECASE)
        s = re.sub(r"\bKey points:\b", "Key points:", s, flags=re.IGNORECASE)
        s = re.sub(r"\s+", " ", s).strip()
        if s and s[-1] not in ".!?":
            s += "."
        return s

    def _format_memory_recall(self, mem: Dict[str, Any], question: str, domain: str) -> str:
        content = str(mem.get("content") or "").strip()
        if not content:
            return ""
        kind = self._mem_kindish(mem)
        if kind in {"chapter_summary", "section_summary", "summary"}:
            return self._summary_text_cleanup(content)
        return self.response_manager.clean(content)

    def _is_feedback_memory(self, mem: Dict[str, Any]) -> bool:
        if not isinstance(mem, dict):
            return False
        mem_domain = str(mem.get("domain") or "").strip().lower()
        tags = self._mem_tag_set(mem)
        content = str(mem.get("content") or "").lower()

        if mem_domain == "operations":
            return True
        if "feedback" in tags:
            return True
        if content.startswith("executed ") and "score=" in content:
            return True
        if "calibration_feedback_" in content:
            return True
        return False

    def _is_revision_query(self, text: str, domain: str = "") -> bool:
        q = (text or "").strip().lower()
        if not q:
            return False

        markers = (
            "for this project",
            "revision",
            "revised",
            "updated plan",
            "new plan",
            "what changed",
            "latest revision",
            "support posts",
            "soft soil",
        )
        if any(m in q for m in markers):
            return True

        return domain == "clubhouse_build" and (
            "for this project" in q or "what changed" in q
        )

    def _is_revision_memory(self, mem: Dict[str, Any]) -> bool:
        if not isinstance(mem, dict):
            return False

        details = self._mem_details(mem)
        tags = self._mem_tag_set(mem)
        mtype = str(mem.get("type") or mem.get("kind") or "").strip().lower()
        content = str(mem.get("content") or "").strip().lower()

        if bool(mem.get("is_revision")):
            return True
        if bool(details.get("is_revision")):
            return True
        if mtype == "revision":
            return True
        if "revision" in tags:
            return True
        if content.startswith("revision:"):
            return True
        if "for this project" in content:
            return True
        return False

    def _extract_anchor_phrases(self, text: str) -> List[str]:
        q = (text or "").lower()
        anchors: List[str] = []

        candidate_phrases = (
            "for this project",
            "soft soil",
            "support posts",
            "before laying plywood",
            "plywood has not been laid yet",
            "joist spacing",
            "floor frame",
            "vertical alignment",
            "concrete pour",
            "uneven",
            "happen first",
            "what should happen first",
            "checked first",
            "level and joist spacing",
        )

        for phrase in candidate_phrases:
            if phrase in q:
                anchors.append(phrase)

        return anchors

    def _phrase_hits(self, text: str, anchors: List[str]) -> int:
        blob = (text or "").lower()
        if not blob or not anchors:
            return 0
        return sum(1 for a in anchors if a in blob)

    def _clubhouse_query_family(self, question_text: str) -> str:
        q = (question_text or "").lower()

        if any(p in q for p in ("support posts", "post depth", "soft soil", "compacted zones", "24 inches", "22 inches", "concrete pour")):
            return "post_depth"

        if any(p in q for p in ("before laying plywood", "plywood has not been laid yet", "joist spacing", "floor frame", "uneven", "checked first", "happen first")):
            return "floor_frame"

        if any(p in q for p in ("decorative", "trim", "visibility", "children", "decorative upgrades")):
            return "priority"

        if any(p in q for p in ("site", "drainage", "stable site", "good drainage")):
            return "site_selection"

        return "general"

    def _clubhouse_memory_family(self, mem: Dict[str, Any]) -> str:
        content = str(mem.get("content") or "").lower()
        tags = self._mem_tag_set(mem)

        if (
            "support posts" in content
            or "post depth" in content
            or "soft soil" in content
            or "compacted zones" in content
            or "concrete pour" in content
            or {"posts", "depth"} <= tags
        ):
            return "post_depth"

        if (
            "before laying plywood" in content
            or "joist spacing" in content
            or "floor frame" in content
            or "joists spaced 16 inches apart" in content
            or "uneven flooring" in content
            or {"floor", "frame"} <= tags
            or "joists" in tags
        ):
            return "floor_frame"

        if (
            "decorative" in content
            or "visibility" in content
            or "children" in content
        ):
            return "priority"

        if (
            "stable site" in content
            or "good drainage" in content
            or "water accumulation" in content
            or "sun exposure" in content
            or "hoa regulations" in content
        ):
            return "site_selection"

        return "general"

    def _clubhouse_locality_adjustment(self, question_text: str, mem: Dict[str, Any]) -> float:
        q_family = self._clubhouse_query_family(question_text)
        m_family = self._clubhouse_memory_family(mem)
        content = str(mem.get("content") or "").lower()
        score = 0.0

        if q_family == "general":
            return score

        if q_family == m_family:
            score += 8.0
        else:
            if q_family == "floor_frame" and m_family in {"post_depth", "site_selection", "priority"}:
                score -= 8.5
            elif q_family == "post_depth" and m_family in {"floor_frame", "site_selection"}:
                score -= 7.0
            elif q_family == "priority" and m_family in {"site_selection", "post_depth"}:
                score -= 5.0
            else:
                score -= 3.0

        if q_family == "floor_frame":
            if "support posts" in content:
                score -= 6.0
            if "vertical alignment" in content:
                score -= 4.5
            if "stable site" in content or "good drainage" in content:
                score -= 9.0
            if "before laying plywood" in content:
                score += 8.0
            if "joist spacing" in content:
                score += 7.0
            if "confirm the frame is level" in content:
                score += 5.0
            if "joist hangers" in content:
                score += 1.5

        if q_family == "post_depth":
            if "decorative upgrades" in content:
                score -= 9.0
            if "decorative trim" in content:
                score -= 6.0
            if "remain 24 inches deep" in content and "even if decorative upgrades are planned" in content:
                score -= 10.0
            if "dug 24 inches deep in soft soil zones" in content:
                score += 12.0
            if "22 inches deep in compacted zones" in content and "soft soil" in (question_text or "").lower():
                score -= 12.0

        return score

    def _recall_priority_score(self, mem: Dict[str, Any], question_text: str, domain: str) -> float:
        if not isinstance(mem, dict):
            return -9999.0

        content = str(mem.get("content") or "").strip()
        if not content or self._is_probably_binary(content):
            return -9999.0
        if self._is_feedback_memory(mem):
            return -9999.0

        truth = self._mem_truth(mem)
        if truth["recall_blocked"] or truth["contested"] or truth["superseded"]:
            return -9999.0

        kind = self._mem_kindish(mem)
        mem_domain = str(mem.get("domain") or "").strip().lower()
        tags = self._mem_tag_set(mem)

        q_tokens = self._tokenize_simple(question_text)
        c_tokens = self._tokenize_simple(content)
        overlap = len(q_tokens & c_tokens)

        anchors = self._extract_anchor_phrases(question_text)
        phrase_hits = self._phrase_hits(content, anchors)

        is_revision_query = self._is_revision_query(question_text, domain=domain)
        is_revision_memory = self._is_revision_memory(mem)

        score = 0.0

        if kind == "chapter_summary":
            score += 9.0
        elif kind == "section_summary":
            score += 8.0
        elif kind == "summary":
            score += 7.0
        elif kind in {"literary_frame", "concept_frame", "frame"}:
            score += 5.0
        elif kind == "fact":
            score += 3.5
        elif kind == "claim":
            score += 2.0
        elif kind == "episode":
            score += 0.5
        elif kind == "revision":
            score += 4.5
        elif kind in {"procedure", "policy"}:
            score += 4.0

        if domain == "literature" and mem_domain == "literature":
            score += 3.0
        elif domain and mem_domain == domain:
            score += 1.5

        if "reading" in tags:
            score += 0.75
        if "literature" in tags:
            score += 0.75

        score += min(overlap, 10) * 0.40
        score += min(float(truth["confidence"] or 0.0), 1.0) * 1.25

        if truth["verified"]:
            score += 1.0
        if truth["pending"]:
            score -= 0.75

        if kind in {"chapter_summary", "section_summary", "summary"}:
            score += self._score_summary_candidate(content, list(tags)) * 0.20

        if is_revision_query:
            if is_revision_memory:
                score += 10.0
            else:
                score -= 6.0

        if phrase_hits > 0:
            score += float(phrase_hits) * 3.5

        if domain == "clubhouse_build":
            qlow = question_text.lower()
            low = content.lower()

            if "before laying plywood" in qlow or "plywood has not been laid yet" in qlow:
                if "before laying plywood" in low:
                    score += 9.0
                elif "plywood" in low:
                    score += 3.0
                else:
                    score -= 3.0

            if "joist spacing" in qlow:
                if "joist spacing" in low or ("joist" in low and "spacing" in low):
                    score += 8.0
                elif "joist" in low:
                    score += 2.5
                else:
                    score -= 2.0

            if "soft soil" in qlow:
                if "soft soil" in low:
                    score += 8.0
                else:
                    score -= 3.0

            if "support posts" in qlow:
                if "support posts" in low or ("support" in low and "post" in low):
                    score += 4.5

            if "uneven" in qlow:
                if "level" in low and "frame" in low:
                    score += 4.5
                elif "level" in low:
                    score += 1.5

            score += self._clubhouse_locality_adjustment(question_text, mem)

        return score

    def _pick_best_recall_match(
        self,
        matched: List[Dict[str, Any]],
        question_text: str,
        domain: str,
    ) -> Optional[Dict[str, Any]]:
        usable = [m for m in (matched or []) if isinstance(m, dict)]
        if not usable:
            return None
        ranked = sorted(
            usable,
            key=lambda m: self._recall_priority_score(
                m,
                question_text=question_text,
                domain=domain,
            ),
            reverse=True,
        )
        return ranked[0] if ranked else None

    def _can_use_literature_summary(self, mem: Dict[str, Any]) -> bool:
        if not isinstance(mem, dict):
            return False
        if self._is_feedback_memory(mem):
            return False
        kind = self._mem_kindish(mem)
        if kind not in {"chapter_summary", "section_summary", "summary"}:
            return False
        content = str(mem.get("content") or "").strip()
        if not content or self._is_probably_binary(content):
            return False
        truth = self._mem_truth(mem)
        return not (truth["recall_blocked"] or truth["contested"] or truth["superseded"])

    def _can_recall_directly(self, mem: Dict[str, Any]) -> bool:
        if not isinstance(mem, dict):
            return False
        if self._is_feedback_memory(mem):
            return False
        truth = self._mem_truth(mem)
        if truth["recall_blocked"] or truth["contested"] or truth["superseded"]:
            return False
        if truth["verified"]:
            return bool(str(mem.get("content") or "").strip())
        return truth["confidence"] >= self.DIRECT_RECALL_CONFIDENCE and not truth["pending"]

    def _can_recall_cautiously(self, mem: Dict[str, Any]) -> bool:
        if not isinstance(mem, dict):
            return False
        if self._is_feedback_memory(mem):
            return False
        truth = self._mem_truth(mem)
        if truth["recall_blocked"] or truth["contested"] or truth["superseded"]:
            return False
        if truth["verified"]:
            return False
        if truth["confidence"] < self.CAUTIOUS_RECALL_CONFIDENCE:
            return False
        if truth["pending"] and not self.ALLOW_CAUTIOUS_PENDING_RECALL:
            return False
        return bool(str(mem.get("content") or "").strip())

    def _format_cautious_recall(self, mem: Dict[str, Any]) -> str:
        content = str(mem.get("content") or "").strip()
        if not content:
            return "I don’t have enough information to answer that yet."
        truth = self._mem_truth(mem)
        if truth["pending"]:
            return f"My best current read is: {content} This may need verification."
        return f"My current best answer is: {content} I’m not fully certain."

    def _question_hazard_tags(self, question_text: str) -> set[str]:
        q = (question_text or "").lower()
        tags = set()
        for t in self.SURVIVAL_MEMORY_TAGS:
            if t in q:
                tags.add(t)
        return tags

    def _hazard_family(self, tokens: set[str]) -> Optional[str]:
        for fam, vocab in self.HAZARD_FAMILIES.items():
            if tokens & vocab:
                return fam
        return None

    def _is_action_guidance_text(self, text: str) -> bool:
        s = (text or "").lower().strip()
        if not s:
            return False

        guidance_patterns = (
            "get low",
            "stay low",
            "crawl low",
            "move to higher ground",
            "do not drive through floodwater",
            "leave low areas",
            "safe exit",
            "nearest safe exit",
            "after the shaking stops",
            "protect your head and neck",
            "drop cover and hold on",
            "stay sheltered",
            "avoid smoke inhalation",
            "watch for aftershocks",
            "signal for help",
            "stay put",
            "conserve energy",
            "leave the fire area",
            "rising water",
            "flood zone",
            "floodwater",
            "evacuate",
            "shut off power",
            "verify the circuit is safe",
        )
        return any(p in s for p in guidance_patterns)

    def _is_survival_memory(self, mem: Dict[str, Any], question_text: str = "") -> bool:
        if not isinstance(mem, dict):
            return False
        if self._is_feedback_memory(mem):
            return False

        content = str(mem.get("content") or "").strip().lower()
        if not content:
            return False

        mem_domain = str(mem.get("domain") or "").strip().lower()
        tags = self._mem_tag_set(mem)
        kind = self._mem_kindish(mem)

        if mem_domain == "survival":
            return True
        if tags & self.SURVIVAL_MEMORY_TAGS:
            return True
        if kind in {"policy", "reasoning_action", "committed_action", "lesson"} and self._is_action_guidance_text(content):
            return True

        if self._is_action_guidance_text(content):
            q_tags = self._question_hazard_tags(question_text)
            c_tokens = self._tokenize_simple(content)
            c_fam = self._hazard_family(c_tokens)
            if q_tags:
                if q_tags & tags:
                    return True
                if c_fam and (c_fam in q_tags or c_fam == self._hazard_family(self._tokenize_simple(question_text))):
                    return True
            elif c_fam is not None:
                return True

        return False

    def _survival_query_family(self, question_text: str) -> Optional[str]:
        return self._hazard_family(self._tokenize_simple(question_text or ""))

    def _survival_memory_score(self, mem: Dict[str, Any], question_text: str) -> float:
        if not isinstance(mem, dict):
            return -9999.0
        if self._is_feedback_memory(mem):
            return -9999.0

        content = str(mem.get("content") or "").strip()
        if not content or self._is_probably_binary(content):
            return -9999.0

        truth = self._mem_truth(mem)
        if truth["recall_blocked"] or truth["contested"] or truth["superseded"]:
            return -9999.0

        tags = self._mem_tag_set(mem)
        kind = self._mem_kindish(mem)
        mem_domain = str(mem.get("domain") or "").strip().lower()

        q_tokens = self._tokenize_simple(question_text)
        c_tokens = self._tokenize_simple(content)

        q_family = self._hazard_family(q_tokens)
        c_family = self._hazard_family(c_tokens)

        overlap = len(q_tokens & c_tokens)

        score = 0.0
        if mem_domain == "survival":
            score += 4.0
        if tags & self.SURVIVAL_MEMORY_TAGS:
            score += 2.5
        if self._is_action_guidance_text(content):
            score += 4.0

        if kind in {"lesson", "policy", "reasoning_action", "committed_action"}:
            score += 1.5
        elif kind in {"fact", "claim"} and not self._is_action_guidance_text(content):
            score -= 2.5

        if q_family:
            if c_family == q_family:
                score += 6.0
            elif c_family is None:
                score += 0.25
            else:
                score -= 5.0

        q_tags = self._question_hazard_tags(question_text)
        if q_tags:
            score += 2.0 * len(q_tags & tags)

        score += min(overlap, 10) * 0.35
        score += min(float(truth["confidence"] or 0.0), 1.0) * 1.0

        if truth["verified"]:
            score += 0.75
        elif truth["pending"]:
            score -= 0.25

        if not self._is_survival_memory(mem, question_text):
            score -= 8.0

        return score

    def _rank_survival_memories(self, matched: List[Dict[str, Any]], question_text: str) -> List[Dict[str, Any]]:
        usable = [m for m in (matched or []) if isinstance(m, dict)]
        if not usable:
            return []
        return sorted(
            usable,
            key=lambda m: self._survival_memory_score(m, question_text),
            reverse=True,
        )

    def _extract_survival_guidance_lines(
        self,
        memories: List[Dict[str, Any]],
        question_text: str,
        max_lines: int = 3,
    ) -> List[str]:
        q_family = self._survival_query_family(question_text)
        out: List[str] = []
        seen = set()

        for mem in memories or []:
            if not isinstance(mem, dict):
                continue
            if self._is_feedback_memory(mem):
                continue

            text = str(mem.get("content") or "").strip()
            if not text or self._is_probably_binary(text):
                continue
            if not self._is_survival_memory(mem, question_text):
                continue

            toks = self._tokenize_simple(text)
            fam = self._hazard_family(toks)
            if q_family and fam and fam != q_family:
                continue

            cleaned = self.response_manager.clean(text)
            if not cleaned:
                continue

            for part in re.split(r"(?<=[.!?])\s+", cleaned):
                p2 = self.response_manager.clean(part)
                if not p2:
                    continue
                key = p2.lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append(p2)
                if len(out) >= max(1, int(max_lines)):
                    return out

        return out

    def _compose_survival_response(self, question_text: str, matched: List[Dict[str, Any]]) -> str:
        ranked = self._rank_survival_memories(matched, question_text)
        if not ranked:
            return ""

        lines = self._extract_survival_guidance_lines(ranked, question_text, max_lines=3)
        if not lines:
            return ""

        q_family = self._survival_query_family(question_text)
        if q_family == "fire":
            priority_terms = ("leave", "exit", "get low", "stay low", "smoke", "call")
        elif q_family == "flood":
            priority_terms = ("higher ground", "avoid", "floodwater", "do not drive", "evac")
        elif q_family == "earthquake":
            priority_terms = ("drop", "cover", "hold on", "aftershock", "protect")
        elif q_family == "lost":
            priority_terms = ("stay put", "signal", "shelter", "conserve")
        else:
            priority_terms = ()

        if priority_terms:
            lines = sorted(
                lines,
                key=lambda s: 0 if any(t in s.lower() for t in priority_terms) else 1,
            )

        lines = self._dedupe_preserve_order(lines)[:2]
        return self.response_manager.clean(" ".join(lines))

    def _try_survival_memory_answer(
        self,
        question_text: str,
        matched: List[Dict[str, Any]],
    ) -> Optional[str]:
        if not matched:
            return None

        ranked = self._rank_survival_memories(matched, question_text)
        if not ranked:
            return None

        top = ranked[0]
        if self._survival_memory_score(top, question_text) < 2.5:
            return None

        direct = self._compose_survival_response(question_text, ranked[:5])
        if direct.strip():
            return direct

        if self._can_recall_directly(top) or self._can_recall_cautiously(top):
            recalled = self._format_memory_recall(top, question=question_text, domain="survival")
            if recalled.strip():
                return recalled

        return None

    def _rank_survival_matches(self, matched: List[Dict[str, Any]], question_text: str) -> List[Dict[str, Any]]:
        ranked = self._rank_survival_memories(
            [m for m in (matched or []) if isinstance(m, dict)],
            question_text,
        )
        return ranked if ranked else matched

    # ------------------------------------------------------------------
    # Lesson / fact helpers
    # ------------------------------------------------------------------
    def _record_lesson(self, raw_input: Dict[str, Any], cp_for_packet: Dict[str, Any]) -> None:
        store_dict = copy.deepcopy(raw_input)
        store_dict["persisted"] = False
        store_dict.setdefault("confidence", 0.72)
        store_dict.setdefault("weight", 0.70)
        store_dict.setdefault("context", [])
        store_dict.setdefault("source", "user_input")
        store_dict.setdefault("domain", cp_for_packet.get("domain", "general"))
        store_dict.setdefault("tags", cp_for_packet.get("tags", []))
        store_dict.setdefault("kind", "episode")
        store_dict.setdefault("evidence", [])
        store_dict.setdefault("last_verified", None)
        store_dict.setdefault("conflict", False)

        details = dict(store_dict.get("details", {}) or {})
        details.setdefault("verified", False)
        details.setdefault("pending_verification", True)
        details.setdefault("contested", False)
        details.setdefault("superseded", False)
        details.setdefault("recall_blocked", False)
        details.setdefault("status", "unverified")
        details.setdefault("source", "user_input")
        details.setdefault("evidence", [])
        store_dict["details"] = details

        self.memory.store([store_dict])
        self._record_learning_event(1)

    def _is_bones_query(self, q: str) -> bool:
        q = (q or "").lower().strip()
        if "bone" not in q and "bones" not in q:
            return False
        triggers = (
            "how many bones",
            "human bones",
            "bones in the human body",
            "bones are in the human body",
            "adult human",
            "skeleton",
            "skeletal",
            "bone count",
        )
        return any(t in q for t in triggers)

    def _is_boiling_query(self, q: str) -> bool:
        q = (q or "").lower().strip()
        return "boiling point of water" in q or "water boil" in q or "water boils" in q

    def _is_everest_query(self, q: str) -> bool:
        q = (q or "").lower().strip()
        return (("everest" in q and ("tall" in q or "height" in q)) or "tallest mountain" in q)

    def _canonical_fact_fastpath(self, q: str) -> Optional[str]:
        q = (q or "").lower().strip()
        if self._is_bones_query(q):
            return "Humans have 206 bones."
        if self._is_boiling_query(q):
            return "Water boils at 100 degrees Celsius."
        if self._is_everest_query(q):
            return "Mount Everest is 8848 meters tall."
        return None

    # ------------------------------------------------------------------
    # Mode helpers
    # ------------------------------------------------------------------
    def _get_mode(self) -> str:
        cp = self.current_context_profile if isinstance(self.current_context_profile, dict) else {}
        dom = str(cp.get("domain") or "").strip().lower()

        if dom == "survival":
            return "rush"

        if self.hypothalamus is not None:
            m = (self.current_mode or "idling").strip().lower()
            return m if m in self.EMOTION_MODES else "idling"

        if dom in {"tech", "literature", "clubhouse_build"}:
            return "curiosity"
        return "idling"

    def _get_system_state(self) -> Dict[str, Any]:
        cp = self.current_context_profile if isinstance(self.current_context_profile, dict) else {}
        threat = self._clamp01(cp.get("threat", 0.0))
        urgency = self._clamp01(cp.get("urgency", 0.1))
        overload = self._clamp01((threat + urgency) / 2.0)
        confidence = float(self._cfg("CONFIDENCE", 0.5))
        return {
            "overload": overload,
            "confidence": confidence,
            "domain": str(cp.get("domain") or "general").strip().lower(),
            "mode": self._get_mode(),
        }

    def _estimate_goal_pressure(self, question_lower: str, cp: Dict[str, Any]) -> float:
        try:
            if self.goal_manager is not None:
                fn = getattr(self.goal_manager, "current_pressure", None)
                if callable(fn):
                    return self._clamp01(fn())
        except Exception:
            pass

        if any(t in question_lower for t in self.PLANNING_TRIGGERS):
            return 0.65

        dom = str((cp or {}).get("domain") or "general").strip().lower()
        if dom in {"tech", "survival", "clubhouse_build"}:
            return 0.60
        return 0.25

    def _estimate_fatigue(self) -> float:
        dq = len(self.deferred_queue or [])
        base = min(
            1.0,
            dq / max(1.0, float(self._cfg("FATIGUE_DEFER_QUEUE_SCALE", 10.0))),
        )
        now = time.time()
        stale = min(
            1.0,
            max(
                0.0,
                (now - (self.last_reflection or now))
                / max(1.0, float(self._cfg("FATIGUE_REFLECT_SCALE_SEC", 60.0))),
            ),
        )
        return self._clamp01(0.70 * base + 0.30 * stale)

    def _update_mode(self, signals: Dict[str, float]) -> None:
        if self.hypothalamus is None:
            return

        cp = self.current_context_profile or {}
        dom = str(cp.get("domain") or "general").strip().lower()

        if dom == "survival":
            self.last_mode = self.current_mode
            self.current_mode = "rush"
            self.last_signals = dict(signals or {})
            return

        chosen_str = self.current_mode
        try:
            chosen = self.hypothalamus.choose_mode(signals)  # type: ignore
            chosen_str = str(getattr(chosen, "value", chosen)).strip().lower()
        except Exception:
            chosen_str = self.current_mode

        if chosen_str not in self.EMOTION_MODES:
            chosen_str = "idling"

        self.last_mode = self.current_mode
        self.current_mode = chosen_str

        try:
            self.mode_bias = self.hypothalamus.get_biases() or {}  # type: ignore
        except Exception:
            self.mode_bias = {}

        self.last_signals = dict(signals or {})

        if (
            self.LOG_MODE_TRANSITIONS
            and self.last_mode != self.current_mode
            and bool(self._cfg("VERBOSE", False))
        ):
            print(
                f"[Hypothalamus] mode {self.last_mode} -> {self.current_mode} "
                f"signals={self.last_signals} bias={self.mode_bias}"
            )

    def _apply_mode_bias(self) -> None:
        bias = self.mode_bias or {}
        if not bias:
            return

        try:
            if "exploration_rate" in bias:
                setattr(self.simulator, "exploration_rate", float(bias["exploration_rate"]))
        except Exception:
            pass

        try:
            if "sim_rollouts" in bias:
                setattr(self.simulator, "rollouts_scale", float(bias["sim_rollouts"]))
        except Exception:
            pass

        try:
            if "sim_horizon" in bias:
                setattr(self.simulator, "horizon_hint", int(round(float(bias["sim_horizon"]))))
        except Exception:
            pass

        try:
            if "risk_tolerance" in bias:
                fn = getattr(self.risk_assessor, "set_tolerance", None)
                if callable(fn):
                    fn(float(bias["risk_tolerance"]))
                else:
                    setattr(self.risk_assessor, "tolerance", float(bias["risk_tolerance"]))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Verification helpers
    # ------------------------------------------------------------------
    def external_verify(self, claim: str) -> Dict[str, Any]:
        return {
            "status": "not_implemented",
            "verdict": "unknown",
            "confidence": None,
            "source": "external_stub",
            "notes": "external verification not implemented yet",
            "claim": str(claim or "").strip(),
        }

    def _memory_conflicts(self, memory: Dict[str, Any]) -> bool:
        truth = self._mem_truth(memory)
        if truth["contested"]:
            return True
        details = self._mem_details(memory)
        if details.get("conflict_with_ids") or details.get("conflict_with_text"):
            return True
        if bool(memory.get("conflict", False)):
            return True
        return False

    def _build_verify_packet(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        truth = self._mem_truth(memory)
        details = self._mem_details(memory)
        claim = str(memory.get("content") or memory.get("claim") or "").strip()

        return {
            "memory_id": memory.get("id") or memory.get("memory_id"),
            "claim": claim,
            "verification_status": truth["status"],
            "staleness_risk": details.get("staleness_risk", "medium"),
            "memory_class": details.get("memory_class")
            or memory.get("kind")
            or memory.get("type")
            or "fact",
            "source_count": len(truth["evidence"]) if isinstance(truth["evidence"], list) else 0,
            "hypothesis": bool(details.get("hypothesis", False)),
            "conflict_with_ids": list(details.get("conflict_with_ids", []) or []),
            "needs_external_verification": False,
            "confidence": float(truth["confidence"] or 0.0),
            "source": truth["source"],
            "evidence": truth["evidence"],
            "last_verified": truth["last_verified"],
        }

    def _apply_verification_result_to_memory(
        self,
        memory: Dict[str, Any],
        result: Dict[str, Any],
    ) -> bool:
        if not isinstance(memory, dict) or not isinstance(result, dict):
            return False

        vr = result.get("verification_result", {}) or {}
        details = self._mem_details(memory)
        status = str(vr.get("status", "") or "").strip().lower()
        verdict = str(vr.get("verdict", "") or "").strip().lower()
        now = time.time()

        if status == "supported" or verdict in {"confirm", "confirmed", "supported"}:
            details["verified"] = True
            details["pending_verification"] = False
            details["contested"] = False
            details["recall_blocked"] = False
            details["status"] = "verified"
            details["verification_status"] = "verified"
            details["last_verified"] = now
            memory["last_verified"] = now
            memory["conflict"] = False
            details["conflict"] = False
            memory["confidence"] = max(float(memory.get("confidence", 0.0) or 0.0), 0.90)
            details["confidence"] = memory["confidence"]
            return True

        if status == "contradicted" or verdict in {"deny", "denied", "contradicted"}:
            details["verified"] = False
            details["pending_verification"] = True
            details["contested"] = True
            details["recall_blocked"] = True
            details["status"] = "contested"
            details["verification_status"] = "contested"
            memory["conflict"] = True
            details["conflict"] = True
            memory["confidence"] = max(
                0.0,
                float(memory.get("confidence", 0.0) or 0.0) - 0.20,
            )
            details["confidence"] = memory["confidence"]
            return True

        return False

    def verify_memory_against_intake(
        self,
        memory: Dict[str, Any],
        intake_text: str,
        source_name: str = "intake_text",
        apply: bool = False,
    ) -> Dict[str, Any]:
        if self.thalamus_verifier is None:
            return {"ok": False, "error": "Thalamus verifier not loaded."}
        if not isinstance(memory, dict):
            return {"ok": False, "error": "memory must be a dict."}

        claim = str(memory.get("content") or memory.get("claim") or "").strip()
        if not claim:
            return {"ok": False, "error": "memory has no usable claim/content."}

        packet = self._build_verify_packet(memory)
        packet["needs_external_verification"] = False

        result = self.thalamus_verifier.verify_and_build_feedback(
            packet,
            intake_text=intake_text,
            source_name=source_name,
        )

        if apply:
            result["applied"] = bool(self._apply_verification_result_to_memory(memory, result))
        return result

    def route_verify_request(
        self,
        memory: Dict[str, Any],
        intake_text: Optional[str] = None,
        source_name: str = "verify_route",
        apply: bool = False,
    ) -> Dict[str, Any]:
        if not isinstance(memory, dict):
            return {"ok": False, "error": "memory must be a dict."}

        truth = self._mem_truth(memory)
        has_conflict = self._memory_conflicts(memory)
        low_conf = float(truth["confidence"] or 0.0) < float(self.VERIFY_ROUTE_LOW_CONFIDENCE)

        route = {
            "ok": True,
            "claim": str(memory.get("content") or "").strip(),
            "memory_id": memory.get("id"),
            "internal_conflict": has_conflict,
            "best_internal_confidence": float(truth["confidence"] or 0.0),
            "source": truth["source"],
            "evidence": truth["evidence"],
            "last_verified": truth["last_verified"],
        }

        if intake_text and str(intake_text).strip():
            result = self.verify_memory_against_intake(
                memory,
                intake_text=str(intake_text),
                source_name=source_name,
                apply=apply,
            )
            route["route"] = "internal_against_intake"
            route["verification"] = result
            return route

        if has_conflict or low_conf:
            route["route"] = "external_stub"
            route["external"] = self.external_verify(route["claim"])
            route["needs_external_check"] = True
            return route

        route["route"] = "internal_only"
        route["needs_external_check"] = False
        route["status"] = "internally_supported"
        return route

    def find_memory_by_text(self, query: str) -> Optional[Dict[str, Any]]:
        q = str(query or "").strip().lower()
        if not q:
            return None

        for m in getattr(self.memory, "buffer", []) or []:
            if hasattr(m, "get"):
                txt = str(m.get("content") or m.get("claim") or "").strip()
                if txt and q in txt.lower():
                    if isinstance(m, dict):
                        return m
                    try:
                        return self.memory._record_to_legacy_dict(m)  # type: ignore[attr-defined]
                    except Exception:
                        continue

        try:
            rows = self.long_term.retrieve(None, 50)
        except TypeError:
            try:
                rows = self.long_term.retrieve(msg_type=None, limit=50)
            except Exception:
                rows = []
        except Exception:
            rows = []

        for m in rows or []:
            if not isinstance(m, dict):
                continue
            txt = str(m.get("content") or m.get("claim") or "").strip()
            if txt and q in txt.lower():
                return m

        return None

    def _verify_ingested_claims_against_chunk(
        self,
        claim_texts: List[str],
        chunk_text: str,
        *,
        source_name: str = "document_chunk",
        max_verify: Optional[int] = None,
    ) -> Dict[str, int]:
        stats = {"attempted": 0, "supported": 0, "contradicted": 0, "updated": 0}

        if not self.DOC_VERIFY_ON_INGEST or self.thalamus_verifier is None:
            return stats
        if not chunk_text or not claim_texts:
            return stats

        max_n = int(max_verify if max_verify is not None else self.DOC_VERIFY_MAX_PER_CHUNK)
        cleaned_claims = self._dedupe_preserve_order(
            [self._compress_claim_text(c) for c in (claim_texts or []) if str(c or "").strip()]
        )
        cleaned_claims = [c for c in cleaned_claims if c][: max(1, max_n)]

        for claim in cleaned_claims:
            target = self.find_memory_by_text(claim)
            if not target:
                continue

            result = self.route_verify_request(
                target,
                intake_text=chunk_text,
                source_name=source_name,
                apply=True,
            )

            verification = result.get("verification", {}) or {}
            if not verification:
                continue

            stats["attempted"] += 1
            vr = verification.get("verification_result", {}) or {}
            status = str(vr.get("status", "") or "").strip().lower()

            if status == "supported":
                stats["supported"] += 1
            elif status == "contradicted":
                stats["contradicted"] += 1

            if bool(verification.get("applied", False)):
                stats["updated"] += 1

        return stats

    # ------------------------------------------------------------------
    # Calibration helpers
    # ------------------------------------------------------------------
    def _format_calibration_question(self, qpkt: Dict[str, Any]) -> str:
        q = str(qpkt.get("question") or "").strip()
        kind = str(qpkt.get("kind") or "").strip().lower()
        opts = qpkt.get("options") or []

        if not q and qpkt.get("conflict_hint"):
            q = f"Calibration conflict: {qpkt.get('conflict_hint')}"
        if not q:
            q = "Calibration question (missing text)."

        lines = [q]
        if isinstance(opts, list) and opts:
            for o in opts:
                lines.append(f"  {o}")
        else:
            if "conflict" in kind:
                lines.append("  1) confirm  2) deny  3) modify: <new text>  4) unsure  5) merge")
            else:
                lines.append("  1) confirm  2) deny  3) modify: <new text>  4) unsure")

        if "conflict" in kind:
            lines.append("Reply with 1/2/3/4/5 or '3: <new claim>'.")
        else:
            lines.append("Reply with 1/2/3/4 or '3: <new claim>'.")
        return "\n".join(lines)

    def _calibration_idle_tick(self) -> Optional[str]:
        if not self.cerebellar:
            return None

        try:
            qpkt = self.cerebellar.idle_tick()  # type: ignore
        except Exception:
            return None

        if not qpkt or not isinstance(qpkt, dict):
            return None

        kind = str(qpkt.get("kind") or "").strip().lower()
        has_memory_id = bool(qpkt.get("memory_id"))
        has_conflict_hint = bool(str(qpkt.get("conflict_hint") or "").strip())

        if not has_memory_id and not has_conflict_hint:
            return None

        if not has_memory_id and has_conflict_hint:
            hint = str(qpkt.get("conflict_hint") or "").strip()
            payload = qpkt.get("payload") or {}
            pair = payload.get("pair") if isinstance(payload, dict) else None

            if isinstance(pair, list) and len(pair) >= 2:
                qpkt["question"] = (
                    f"Conflict detected between memories {pair[0]} and {pair[1]}. "
                    f"Confirm one, deny one, or modify the stronger claim."
                )
            else:
                qpkt["question"] = f"Conflict detected: {hint}"
            qpkt["options"] = ["confirm", "deny", "modify", "unsure", "merge"]

        self._pending_calibration = qpkt
        self._record_calibration_event(1)
        return self._format_calibration_question(qpkt)

    def _parse_calibration_answer(self, user_text: str) -> Tuple[Optional[str], Optional[str]]:
        t = (user_text or "").strip()
        if not t:
            return None, None
        low = t.lower()

        if low in {"1", "confirm", "yes", "y"}:
            return "confirm", None
        if low in {"2", "deny", "no", "n"}:
            return "deny", None
        if low in {"4", "unsure", "unknown", "idk"}:
            return "unsure", None
        if low in {"5", "merge"}:
            return "merge", None
        if low in {"3", "modify"}:
            return "modify", None
        if low.startswith("3:") or low.startswith("3 "):
            mod = t.split(":", 1)[1].strip() if ":" in t else t.split(" ", 1)[1].strip()
            return ("modify", mod) if mod else ("modify", None)
        if low.startswith("modify:"):
            mod = t.split(":", 1)[1].strip()
            return ("modify", mod) if mod else ("modify", None)

        return None, None

    def _apply_calibration_answer(self, user_text: str) -> Optional[str]:
        if not self.cerebellar or not self._pending_calibration:
            return None

        ans, mod = self._parse_calibration_answer(user_text)
        if not ans:
            return None

        pending = dict(self._pending_calibration)
        mem_id = pending.get("memory_id")
        mem_ids = pending.get("memory_ids") or []

        if ans == "modify" and not mod:
            return "Send the corrected text as: 3: <new claim>"

        has_target = bool(mem_id) or bool(mem_ids) or bool(str(pending.get("conflict_hint") or "").strip())
        if not has_target:
            self._pending_calibration = None
            return "Calibration cleared (missing target memory info)."

        try:
            fn = getattr(self.cerebellar, "apply_user_feedback", None)
            if callable(fn) and mem_id:
                fn(str(mem_id), ans, modification=mod)  # type: ignore
            else:
                candidate = {
                    "id": pending.get("id"),
                    "kind": pending.get("kind"),
                    "memory_id": str(mem_id) if mem_id else None,
                    "memory_ids": [str(x) for x in mem_ids if x],
                    "conflict_hint": pending.get("conflict_hint"),
                    "payload": pending.get("payload"),
                    "question": pending.get("question"),
                    "prompt": pending.get("prompt"),
                }
                feedback = {"verdict": ans, "correction": mod}
                self.cerebellar.apply_feedback(candidate, feedback)  # type: ignore
        except Exception:
            self._pending_calibration = None
            return "Calibration update failed."

        self._record_calibration_event(1)
        self._pending_calibration = None
        return "Calibration recorded."

    def _run_sleep_calibration(self) -> str:
        parts: List[str] = []

        audit = None
        if self.acc is not None:
            try:
                audit = self.acc.full_audit()  # type: ignore
            except Exception:
                audit = None

        report = None
        if self.cerebellar is not None:
            try:
                report = self.cerebellar.sleep_cycle()  # type: ignore
            except Exception:
                report = None

        if audit and isinstance(audit, dict):
            drift = audit.get("drift_signals", {}) or {}
            parts.append(
                f"ACC drift: low_conf={int(drift.get('low_confidence', 0) or 0)} "
                f"contested={int(drift.get('contested', 0) or 0)} "
                f"stale={int(drift.get('stale', 0) or 0)}"
            )
            parts.append(
                f"ACC issues: numeric={len(audit.get('numeric_conflicts', []) or [])} "
                f"negation={len(audit.get('negation_conflicts', []) or [])} "
                f"dups={len(audit.get('duplicates', []) or [])}"
            )

        if report and isinstance(report, dict):
            parts.append(
                f"Cerebellar: conflicts={int(report.get('conflicts', 0) or 0)} "
                f"merged={int(report.get('merged', 0) or 0)} "
                f"decayed={int(report.get('decayed', 0) or 0)}"
            )

        self._record_calibration_event(1)
        return " | ".join(parts) if parts else "Sleep calibration unavailable (modules not loaded)."


    # ------------------------------------------------------------------
    # Seeding / output / context helpers
    # ------------------------------------------------------------------
    def _seed_identity(self) -> None:
        identity_obj = {
            "name": "Clair",
            "meaning": "Cognitive Learning and Interactive Reasoner",
            "description": "I am Clair, designed to learn, reason, and support my user.",
        }
        self.memory.store(
            [
                {
                    "type": "identity",
                    "content": (
                        f"My name is {identity_obj['name']} "
                        f"({identity_obj['meaning']}). "
                        f"{identity_obj['description']}"
                    ),
                    "content_obj": identity_obj,
                    "confidence": 1.0,
                    "weight": 1.0,
                    "context": [],
                    "source": "system",
                    "evidence": [],
                    "last_verified": time.time(),
                    "conflict": False,
                    "persisted": True,
                    "domain": "identity",
                    "kind": "fact",
                    "tags": ["identity"],
                    "details": {
                        "verified": True,
                        "status": "verified",
                        "pending_verification": False,
                        "source": "system",
                        "evidence": [],
                    },
                }
            ]
        )

    def _emit(self, text: str) -> str:
        text = self._safe_output(text)
        cleaned = self.response_manager.clean(text)
        if not cleaned:
            return ""

        cleaned = self.response_manager.limit_sentences(
            cleaned,
            max_sentences=self.MAX_RESPONSE_SENTENCES,
        )

        v = str(getattr(self.dialogue, "verbosity", "normal") or "normal").strip().lower()
        emo = str(getattr(self.dialogue, "emotional_load", "low") or "low").strip().lower()

        if v == "short":
            cleaned = self.response_manager.limit_sentences(cleaned, max_sentences=1)
        elif v == "detailed":
            cleaned = self.response_manager.clean(cleaned)
        else:
            cleaned = self.response_manager.limit_sentences(
                cleaned,
                max_sentences=self.MAX_RESPONSE_SENTENCES,
            )

        if emo == "high":
            cleaned = self.response_manager.clean(f"Clear answer: {cleaned}")
            cleaned = self.response_manager.limit_sentences(
                cleaned,
                max_sentences=self.MAX_RESPONSE_SENTENCES,
            )

        self.dialogue.last_response_text = cleaned
        return cleaned

    def _is_greeting(self, q: str) -> bool:
        q = (q or "").strip().lower()
        return q in {"hi", "hello", "hey", "yo", "sup"} or q.startswith(("hi ", "hello ", "hey "))

    def _is_literature_quote_intent(self, question_lower: str) -> bool:
        q = (question_lower or "").strip().lower()
        return any(t in q for t in self.LITERATURE_QUOTE_INTENT)

    def _is_literature_abstraction_query(self, question_lower: str) -> bool:
        q = (question_lower or "").strip().lower()
        return any(t in q for t in self.LITERATURE_QUERY_TOKENS)

    def _normalize_identity_lesson(self, user_text: str) -> Optional[str]:
        if not user_text:
            return None

        raw = user_text.strip()
        low = raw.lower().strip()

        if "i am your father" in low:
            idx = low.find("i am your father")
            tail_raw = raw[idx + len("i am your father") :].strip()
            name = tail_raw.strip(" .,!?:;\"'")
            if name:
                return f"{name} is my father."
            return None

        if low.startswith("my name is "):
            name = raw.split("is", 1)[1].strip().strip(" .,!?:;\"'")
            if name:
                return f"The user's name is {name}."
            return None

        if low.startswith("i am "):
            val = raw[5:].strip().strip(" .,!?:;\"'")
            if not val:
                return None
            looks_like_name = any(ch.isupper() for ch in val) or (len(val.split()) >= 2)
            bad = {"tired", "hungry", "sad", "angry", "bored", "fine", "okay", "ok"}
            if looks_like_name and val.lower() not in bad:
                return f"The user is {val}."
            return None

        return None

    def _build_context_profile(self, text: str) -> Dict[str, Any]:
        q = (text or "").lower().strip()

        profile: Dict[str, Any] = {
            "domain": "general",
            "tags": [],
            "threat": 0.0,
            "urgency": 0.1,
            "goal": "inform",
        }

        if self.QUARANTINE_FEEDBACK and (q.startswith("action action_") and "scored" in q):
            profile["domain"] = "operations"
            profile["tags"] = ["feedback"]
            profile["goal"] = "log"
            profile["urgency"] = 0.0
            profile["threat"] = 0.0
            return profile

        if any(t in q for t in ("who are you", "your name", "what is your name", "who is clair")):
            profile["domain"] = "identity"
            profile["tags"] = ["identity"]
            profile["goal"] = "identify"
            profile["urgency"] = 0.2
            return profile

        if any(t in q for t in ("python", "code", "coding", "program", "debug", "error", "traceback", "module")):
            profile["domain"] = "tech"
            profile["tags"] = ["tech", "code"]
            profile["goal"] = "solve task"
            profile["urgency"] = 0.2
            return profile

        if (
            any(
                t in q
                for t in (
                    "poem",
                    "poetry",
                    "short story",
                    "novel",
                    "edgar allan poe",
                    "literature",
                    "raven",
                    "philosophy",
                    "argument",
                )
            )
            or self._is_literature_abstraction_query(q)
        ):
            profile["domain"] = "literature"
            profile["tags"] = ["literature", "reading"]
            profile["goal"] = "interpret"
            profile["urgency"] = 0.2
            return profile

        if self._is_situational_question(text):
            profile["domain"] = "survival"
            tags: List[str] = []
            for t in (
                "fire",
                "flood",
                "earthquake",
                "emergency",
                "lost",
                "survive",
                "danger",
                "risk",
                "smoke",
                "submerged",
                "evacuate",
                "aftershock",
                "flooding",
                "floodwater",
                "flash",
            ):
                if t in q:
                    tags.append(t)
            profile["tags"] = tags or ["survival"]
            profile["goal"] = "preserve life"
            profile["threat"] = (
                0.7
                if any(t in q for t in ("fire", "flood", "earthquake", "emergency", "danger", "submerged"))
                else 0.4
            )
            profile["urgency"] = (
                0.7
                if any(t in q for t in ("now", "help", "urgent", "immediately", "asap"))
                else 0.5
            )
            return profile

        clubhouse_markers = {
            "clubhouse",
            "playhouse",
            "support posts",
            "post depth",
            "soft soil",
            "joist spacing",
            "floor frame",
            "plywood",
            "concrete pour",
            "framing",
            "roofing",
            "deck-floor",
        }
        if any(marker in q for marker in clubhouse_markers):
            profile["domain"] = "clubhouse_build"
            profile["tags"] = ["clubhouse_build"]
            q_family = self._clubhouse_query_family(q)
            if q_family != "general":
                profile["tags"].append(f"clubhouse_{q_family}")
            profile["goal"] = "build"
            profile["urgency"] = 0.2
            return profile

        return profile

    def _force_survival_profile(self, base_profile: Dict[str, Any], question_text: str) -> Dict[str, Any]:
        cp = dict(base_profile or {})
        cp["domain"] = "survival"
        tags = set(cp.get("tags") or [])
        tags.add("survival")

        q = (question_text or "").lower()
        for t in (
            "fire",
            "flood",
            "earthquake",
            "lost",
            "smoke",
            "submerged",
            "flooding",
            "floodwater",
            "aftershock",
            "flash",
        ):
            if t in q:
                tags.add(t)

        cp["tags"] = sorted(tags)
        cp["goal"] = "preserve life"
        cp["threat"] = max(float(cp.get("threat", 0.0) or 0.0), 0.6)
        cp["urgency"] = max(float(cp.get("urgency", 0.1) or 0.1), 0.5)
        return cp

    def _should_allow_action_cycle_for_question(self, text: str, cp: Dict[str, Any]) -> bool:
        if not isinstance(cp, dict):
            return False

        dom = str(cp.get("domain") or "general").strip().lower()
        if dom == "operations" and self.QUARANTINE_FEEDBACK:
            return False
        if dom in {"survival", "tech"}:
            return True

        try:
            if float(cp.get("urgency", 0.0) or 0.0) >= self.ACTION_URGENCY_THRESHOLD:
                return True
            if float(cp.get("threat", 0.0) or 0.0) >= self.ACTION_THREAT_THRESHOLD:
                return True
        except Exception:
            pass

        q = (text or "").lower()
        return any(t in q for t in self.PLANNING_TRIGGERS)

    def seed_verified_facts(self) -> None:
        seeds = [
            {
                "type": "fact",
                "content": "Water boils at 100 degrees Celsius.",
                "numeric_guarded": True,
                "domain": "general",
                "tags": ["science"],
            },
            {
                "type": "fact",
                "content": "Humans have 206 bones.",
                "numeric_guarded": True,
                "domain": "general",
                "tags": ["biology"],
            },
            {
                "type": "fact",
                "content": "Mount Everest is 8848 meters tall.",
                "numeric_guarded": True,
                "domain": "general",
                "tags": ["geography"],
            },
            {
                "type": "fact",
                "content": "Rain is caused by condensation of water vapor.",
                "domain": "general",
                "tags": ["weather"],
            },
            {
                "type": "fact",
                "content": "The human heart pumps blood throughout the body.",
                "domain": "general",
                "tags": ["biology"],
            },
            {
                "type": "fact",
                "content": "The Sahara is the largest hot desert.",
                "domain": "general",
                "tags": ["geography"],
            },
            {
                "type": "fact",
                "content": "Sharks have been around for over 400 million years.",
                "domain": "general",
                "tags": ["biology"],
            },
            {
                "type": "fact",
                "content": "Python is a programming language.",
                "domain": "tech",
                "tags": ["tech", "code"],
            },
        ]

        payload = []
        now = time.time()
        for s in seeds:
            payload.append(
                {
                    "type": s.get("type", "fact"),
                    "content": s.get("content", ""),
                    "confidence": 1.0,
                    "weight": 1.0,
                    "context": [],
                    "source": "seed_verified",
                    "evidence": ["seed_memory"],
                    "last_verified": now,
                    "conflict": False,
                    "persisted": True,
                    "numeric_guarded": bool(s.get("numeric_guarded", False)),
                    "domain": s.get("domain", "general"),
                    "tags": s.get("tags", []),
                    "kind": "fact",
                    "details": {
                        "verified": True,
                        "pending_verification": False,
                        "contested": False,
                        "superseded": False,
                        "recall_blocked": False,
                        "status": "verified",
                        "source_trust": "trusted",
                        "source": "seed_verified",
                        "evidence": ["seed_memory"],
                        "last_verified": now,
                    },
                }
            )
        self.memory.store(payload)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def _is_situational_question(self, text: str) -> bool:
        if not text:
            return False

        try:
            fn = getattr(self.reasoner, "_is_situational", None)
            if callable(fn) and fn(text):
                return True
        except Exception:
            pass

        q = (text or "").lower()
        triggers = (
            "what should",
            "how should",
            "what to do",
            "how to",
            "lost",
            "fire",
            "flood",
            "earthquake",
            "emergency",
            "survive",
            "danger",
            "risk",
            "submerged",
        )
        if any(t in q for t in triggers):
            return True

        try:
            toks = self.memory.extract_keywords(q)
            return any(t in toks for t in self.SURVIVAL_TOKENS)
        except Exception:
            return any(t in q for t in self.SURVIVAL_TOKENS)

    def _is_meta_reasoning_query(self, text: str) -> bool:
        q = (text or "").lower().strip()
        meta_triggers = (
            "explain your reasoning",
            "show your reasoning",
            "show your work",
            "how did you decide",
            "how did you come to that",
            "how did you get that",
            "why did you",
            "why would you",
            "explain that answer",
            "explain your answer",
            "why was that your answer",
        )
        if any(t in q for t in meta_triggers):
            return True
        if "why" in q and any(p in q for p in ("you ", "your ", "clair ")):
            return True
        return False

    def handle_identity_query(self) -> str:
        try:
            identity = self.memory.retrieve("identity", count=1, context_profile={"domain": "identity"})
        except Exception:
            identity = []

        if identity and isinstance(identity[0], dict):
            cobj = identity[0].get("content_obj")
            if isinstance(cobj, dict):
                return (
                    f"My name is {cobj.get('name')} "
                    f"({cobj.get('meaning')}). "
                    f"{cobj.get('description')}"
                )
            content = identity[0].get("content")
            if isinstance(content, str) and content.strip():
                return content
        return "I am Clair."

    def handle_why_query(self, packet_id: str) -> str:
        steps = self.reasoning_context.get(packet_id, [])
        if not steps and self.last_reasoning_packet_id:
            steps = self.reasoning_context.get(self.last_reasoning_packet_id, [])
            packet_id = self.last_reasoning_packet_id

        if not steps:
            return "I don't have recent reasoning for this query."

        output = steps[:3]
        self.reasoning_context[packet_id] = steps[3:]
        return self.response_manager.clean(" ".join(output))

    def undo_last_reasoning(self, packet_id: str) -> str:
        target = packet_id
        if not self.reasoning_context.get(target) and self.last_reasoning_packet_id:
            target = self.last_reasoning_packet_id

        steps = self.reasoning_context.get(target, [])
        if steps:
            removed = steps.pop()
            self.reasoning_context[target] = steps
            return self.response_manager.clean(f"Removed last reasoning step: {removed}")
        return "No reasoning steps to undo for this query."

    def _set_trace(self, packet_id: str, trace: List[str]) -> None:
        cleaned = [str(t) for t in (trace or []) if str(t).strip()]
        self.reasoning_context[packet_id] = cleaned
        self.last_reasoning_packet_id = packet_id

    # ------------------------------------------------------------------
    # Main packet handler
    # ------------------------------------------------------------------
    def handle_packet(self, packet: Any):
        packet_id = getattr(packet, "packet_id", "default")
        signals = getattr(packet, "signals", {}) or {}
        content = str(signals.get("normalized_text", "") or "").strip()
        if not content:
            return None

        if self._is_probably_binary(content):
            return "That looks like file bytes. Use `read:` with a path to ingest documents."

        now = time.time()
        self.last_input_ts = now

        raw_input = getattr(packet, "raw_input", None)
        if isinstance(raw_input, str):
            raw_input = {"type": "ask", "content": raw_input}
        elif raw_input is None:
            raw_input = {"type": "ask", "content": content}
        elif not isinstance(raw_input, dict):
            raw_input = {"type": "ask", "content": content}

        raw_type = str(raw_input.get("type") or "ask").strip().lower()

        cp_for_packet = self._build_context_profile(content)
        cp_for_packet = self._apply_loop_balance_pressure(cp_for_packet)
        self.current_context_profile = cp_for_packet

        if raw_type == "ask":
            try:
                self.dialogue.update_from_user(content)
            except Exception:
                pass

            self.last_question = content
            self.last_question_ts = now
            cp_cached = dict(cp_for_packet)
            cp_cached["action_ok"] = self._should_allow_action_cycle_for_question(
                content,
                cp_cached,
            )
            self.last_question_context_profile = cp_cached
            self.last_answer_was_recall = False

        if self.QUARANTINE_FEEDBACK and raw_type == "feedback":
            store_dict = copy.deepcopy(raw_input)
            store_dict.setdefault("persisted", False)
            store_dict.setdefault("confidence", 0.6)
            store_dict.setdefault("weight", 0.4)
            store_dict.setdefault("context", [])
            store_dict.setdefault("source", "system")
            store_dict.setdefault("domain", "operations")
            store_dict.setdefault("tags", ["feedback"])
            store_dict.setdefault("kind", "episode")
            store_dict.setdefault("evidence", [])
            store_dict.setdefault("last_verified", None)
            store_dict.setdefault("conflict", False)
            self.memory.store([store_dict])
            return None

        if raw_type in {"lesson", "observe"}:
            lower_content = content.lower()
            existing: List[str] = []
            for m in getattr(self.memory, "buffer", []) or []:
                try:
                    c = m.get("content", "")
                except Exception:
                    c = ""
                if isinstance(c, str) and c:
                    existing.append(c.lower())

            if lower_content not in existing:
                self._record_lesson(raw_input, cp_for_packet)
            return None

        if raw_type != "ask":
            return None

        self._record_reasoning_cycle(1)
        ql = content.lower()

        if self._is_greeting(content):
            self.last_answer_was_recall = True
            return "Hello. What do you want to work on?"

        if "undo last step" in ql or "remove last reasoning" in ql:
            return self.undo_last_reasoning(packet_id)

        if self._is_meta_reasoning_query(ql):
            return self.handle_why_query(packet_id)

        if any(t in ql for t in ("who are you", "your name", "what is your name", "who is clair")):
            self._set_trace(
                packet_id,
                ["Detected identity query.", "Returned identity profile."],
            )
            return self.response_manager.clean(self.handle_identity_query())

        fast = self._canonical_fact_fastpath(ql)
        if fast:
            self._set_trace(
                packet_id,
                [
                    "Canonical fact fast-path matched.",
                    "Returned trusted seeded fact directly.",
                    f"Mode={self._get_mode()}.",
                ],
            )
            self.last_answer_was_recall = True
            return self.response_manager.clean(fast)

        is_situational = (
            self._is_situational_question(content)
            or cp_for_packet.get("domain") == "survival"
            or any(t in ql for t in self.SURVIVAL_TOKENS)
        )

        try:
            keywords = self.memory.extract_keywords(ql)
            if not isinstance(keywords, list):
                keywords = list(keywords) if keywords else []
        except Exception:
            keywords = []

        if "tallest mountain" in ql and "everest" not in keywords:
            keywords.append("everest")
        if "bones" in ql and "bone" not in keywords:
            keywords.append("bone")
        if "human bones" in ql and "206" not in keywords:
            keywords.append("206")

        if self._is_literature_abstraction_query(ql):
            for k in ("reading", "chapter", "book", "summary", "argument", "theme", "literature"):
                if k not in keywords:
                    keywords.append(k)

        min_rel = float(self._cfg("WM_MIN_RELEVANCE", 1.5))

        cp = dict(cp_for_packet or {})
        if is_situational:
            cp = self._force_survival_profile(cp, question_text=content)

        try:
            matched = self.memory.retrieve(
                keywords=keywords,
                context_profile=cp,
                min_relevance=min_rel,
                count=8,
            )
        except Exception:
            matched = []

        dom = str(cp.get("domain") or "general").strip().lower()
        if matched and dom == "clubhouse_build":
            ranked_matches = sorted(
                [m for m in matched if isinstance(m, dict)],
                key=lambda m: self._recall_priority_score(m, question_text=content, domain=dom),
                reverse=True,
            )
            if ranked_matches:
                matched = ranked_matches

            if self._is_revision_query(content, domain=dom):
                revision_only = [m for m in matched if self._is_revision_memory(m)]
                if revision_only:
                    matched = revision_only

        threat = self._clamp01(cp.get("threat", 0.0))
        urgency = self._clamp01(cp.get("urgency", 0.1))
        novelty = 0.20
        uncertainty = 0.35

        if not matched:
            novelty = 0.85
            uncertainty = 0.70
        else:
            try:
                top = matched[0] if matched else {}
                topc = str(top.get("content") or "") if isinstance(top, dict) else ""
                if not topc.strip() or self._is_probably_binary(topc):
                    novelty = 0.70
                    uncertainty = 0.70
            except Exception:
                pass

        best_match = self._pick_best_recall_match(
            matched,
            question_text=content,
            domain=dom,
        )

        if isinstance(best_match, dict):
            truth = self._mem_truth(best_match)
            if truth["contested"] or truth["superseded"] or truth["recall_blocked"]:
                uncertainty = max(uncertainty, 0.90)
            elif truth["pending"]:
                uncertainty = max(uncertainty, 0.65)
            elif truth["verified"]:
                uncertainty = min(uncertainty, 0.30)

        try:
            sys_conf = float(self._cfg("CONFIDENCE", 0.5))
            uncertainty = self._clamp01(
                0.60 * uncertainty + 0.40 * (1.0 - self._clamp01(sys_conf))
            )
        except Exception:
            pass

        goal_pressure = self._estimate_goal_pressure(ql, cp)
        fatigue = self._estimate_fatigue()

        if self.hypothalamus is not None:
            self._update_mode(
                {
                    "risk": threat,
                    "uncertainty": uncertainty,
                    "novelty": novelty,
                    "urgency": urgency,
                    "goal_pressure": goal_pressure,
                    "fatigue": fatigue,
                    "activity": self._clamp01(
                        0.40 * urgency + 0.35 * novelty + 0.25 * goal_pressure
                    ),
                }
            )
            self._apply_mode_bias()

        if is_situational and dom == "survival":
            survival_answer = self._try_survival_memory_answer(content, matched)

            if survival_answer:
                ranked = self._rank_survival_memories(matched, content)
                top = ranked[0] if ranked else None
                self._set_trace(
                    packet_id,
                    [
                        "Situational survival query.",
                        "Used survival-memory synthesis route before simulator fallback.",
                        f"Top survival match kind={top.get('kind') or top.get('type') if isinstance(top, dict) else 'unknown'}.",
                        f"Top survival match domain={top.get('domain') if isinstance(top, dict) else 'unknown'}.",
                        f"Top survival match tags={list(top.get('tags') or []) if isinstance(top, dict) else []}.",
                        f"Mode={self._get_mode()}.",
                        "Returned actionable survival guidance from stored lessons/memories.",
                    ],
                )
                self.last_answer_was_recall = True
                return self.response_manager.clean(survival_answer)

            self._set_trace(
                packet_id,
                [
                    "Situational survival query.",
                    "No usable survival-memory synthesis available.",
                    f"Matched_count={len(matched or [])}.",
                    f"Mode={self._get_mode()}.",
                    "Fell through to reasoning/planning path.",
                ],
            )

        if best_match and (not is_situational) and dom != "survival":
            recalled = self._format_memory_recall(best_match, question=content, domain=dom)
            if recalled.strip():
                if dom == "literature" and self._can_use_literature_summary(best_match):
                    self._set_trace(
                        packet_id,
                        [
                            "Chose literature summary recall route.",
                            f"Best match kind={best_match.get('kind') or best_match.get('type')} domain={best_match.get('domain')}.",
                            f"Best match tags={list(best_match.get('tags') or [])}.",
                            f"Mode={self._get_mode()}.",
                            "Allowed summary recall for literature without requiring strict fact-grade confidence.",
                        ],
                    )
                    self.last_answer_was_recall = True
                    return self.response_manager.clean(recalled)

                if self._can_recall_directly(best_match):
                    self._set_trace(
                        packet_id,
                        [
                            "Chose memory recall route (non-situational query).",
                            f"Best match kind={best_match.get('kind') or best_match.get('type')} domain={best_match.get('domain')}.",
                            f"Best match tags={list(best_match.get('tags') or [])}.",
                            f"Mode={self._get_mode()}.",
                            "Preferred high-locality recall over broad same-domain recall.",
                            "Returned stored memory content directly.",
                        ],
                    )
                    self.last_answer_was_recall = True
                    return self.response_manager.clean(recalled)

                if self._can_recall_cautiously(best_match):
                    self._set_trace(
                        packet_id,
                        [
                            "Memory match found.",
                            f"Best match kind={best_match.get('kind') or best_match.get('type')} domain={best_match.get('domain')}.",
                            "Memory was usable but not trusted enough for direct factual recall.",
                            f"Mode={self._get_mode()}.",
                            "Returned cautious phrasing.",
                        ],
                    )
                    self.last_answer_was_recall = True
                    return self.response_manager.clean(self._format_cautious_recall(best_match))

        if dom == "literature" and not is_situational and not self._is_literature_quote_intent(ql):
            self._set_trace(
                packet_id,
                [
                    "Literature domain detected.",
                    "No strong enough summary/frame recall surfaced.",
                    "Routed to opinion engine (PFC) only after recall preference pass.",
                    f"Mode={self._get_mode()}.",
                ],
            )
            self.last_answer_was_recall = False
            return self.response_manager.clean(self.generate_opinion(content))

        answer_dict = self.reasoner.answer_question(
            question=content,
            working_memory=self.memory,
            max_chain_steps=int(self._cfg("REASONING_MAX_CHAIN_STEPS", 4)),
            max_actions=int(self._cfg("REASONING_MAX_ACTIONS", 3)),
            context_profile=cp,
        )

        trace = answer_dict.get("reasoning_trace", []) or []
        if not trace:
            trace = [
                "Reasoning engine used (trace fallback).",
                f"Context domain={cp.get('domain')} tags={cp.get('tags', [])}.",
            ]

        if isinstance(best_match, dict):
            truth = self._mem_truth(best_match)
            if truth["recall_blocked"] or truth["contested"] or truth["superseded"]:
                trace = ["Blocked direct recall due to truth-state flags."] + trace
            elif truth["pending"]:
                trace = ["Skipped direct factual recall because memory is pending verification."] + trace

        trace = [f"[Mode={self._get_mode()}]"] + trace
        self._set_trace(packet_id, trace)

        answer = self._safe_output(answer_dict.get("answer"))

        if (not answer or not str(answer).strip()) and is_situational and dom == "survival":
            fallback = self._try_survival_memory_answer(content, matched)
            if fallback:
                self._set_trace(
                    packet_id,
                    trace + ["Reasoning answer empty; recovered using survival-memory fallback."],
                )
                self.last_answer_was_recall = True
                return self.response_manager.clean(fallback)

        if not answer or not str(answer).strip():
            return "I don’t have enough information to answer that yet."

        self.last_answer_was_recall = False
        return self.response_manager.clean(answer)

    # ------------------------------------------------------------------
    # Memory / reflection / action cycle
    # ------------------------------------------------------------------
    def consolidate_memory(self) -> None:
        threshold = float(self._cfg("LTM_AUTO_SYNC_WEIGHT", 0.75))
        promotable: List[Dict[str, Any]] = []

        for m in getattr(self.memory, "buffer", []) or []:
            try:
                md = self.memory._record_to_legacy_dict(m) if not isinstance(m, dict) else m  # type: ignore[attr-defined]
            except Exception:
                continue

            if (
                isinstance(md, dict)
                and not md.get("persisted", False)
                and float(md.get("weight", 0.0) or 0.0) >= threshold
                and md.get("type")
                in {
                    "lesson",
                    "observe",
                    "reasoning_action",
                    "fact",
                    "committed_action",
                    "claim",
                    "literary_frame",
                    "concept_frame",
                    "section_summary",
                    "chapter_summary",
                    "summary",
                }
            ):
                promotable.append(md)

        if promotable:
            try:
                self.long_term.store(promotable)
                for m in getattr(self.memory, "buffer", []) or []:
                    try:
                        if getattr(m, "metadata", None) is not None:
                            mid = getattr(m, "memory_id", None)
                            if any(p.get("memory_id") == mid or p.get("id") == mid for p in promotable):
                                m.metadata["persisted"] = True
                    except Exception:
                        continue
            except Exception:
                pass

    def reflect(self, force: bool = False) -> None:
        now = time.time()
        if force or (now - self.last_reflection) >= self.REFLECTION_INTERVAL:
            try:
                self.memory.reflect()
            except Exception:
                pass
            self.last_reflection = now

    def _choose_action(self, safe_options: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not safe_options:
            return None

        cp = getattr(self, "current_context_profile", {}) or {}
        dom = str(cp.get("domain") or "general").strip().lower()
        q = str(getattr(self, "last_question", "") or "")

        if dom == "clubhouse_build":
            ranked_safe = sorted(
                [o for o in safe_options if isinstance(o, dict)],
                key=lambda o: self._score_clubhouse_option_locality(o, q),
                reverse=True,
            )
            if ranked_safe:
                return ranked_safe[0]

        return safe_options[0] if isinstance(safe_options[0], dict) else None

    def _score_clubhouse_option_locality(self, option: Dict[str, Any], question_text: str) -> float:
        blob = " ".join(
            [
                str(option.get("action_name") or ""),
                str(option.get("action") or ""),
                str(option.get("description") or ""),
                str(option.get("seed") or ""),
                str(option.get("seed_text") or ""),
                str(option.get("reason") or ""),
                str((option.get("details") or {}).get("seed_text") or ""),
                str((option.get("details") or {}).get("seed_memory") or ""),
                str((option.get("details") or {}).get("planned_next") or ""),
            ]
        ).lower()

        q_family = self._clubhouse_query_family(question_text)
        score = 0.0

        if q_family == "floor_frame":
            if "before laying plywood" in blob:
                score += 12.0
            if "joist spacing" in blob:
                score += 10.0
            if "floor frame" in blob:
                score += 9.0
            if "confirm the frame is level" in blob or ("level" in blob and "frame" in blob):
                score += 8.0
            if "joist" in blob:
                score += 4.0

            if "stable site" in blob or "good drainage" in blob:
                score -= 14.0
            if "support posts" in blob:
                score -= 8.0
            if "vertical alignment" in blob:
                score -= 6.0
            if "decorative" in blob:
                score -= 5.0

        elif q_family == "post_depth":
            if "24 inches deep in soft soil zones" in blob:
                score += 14.0
            if "support posts" in blob:
                score += 7.0
            if "soft soil" in blob:
                score += 8.0

            if "22 inches deep in compacted zones" in blob and "soft soil" in question_text.lower():
                score -= 15.0
            if "decorative upgrades" in blob:
                score -= 10.0
            if "stable site" in blob or "good drainage" in blob:
                score -= 8.0

        return score

    def _execute_action_cycle(
        self,
        context_profile: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[List[Dict[str, Any]]], Optional[List[Dict[str, Any]]]]:
        if self.hypothalamus is not None:
            self._apply_mode_bias()

        num_actions = int(self._cfg("SIMULATOR_DEFAULT_NUM_ACTIONS", 2))
        horizon = int(self._cfg("SIMULATOR_HORIZON", 2))

        cp = (
            context_profile
            or getattr(self, "last_question_context_profile", None)
            or getattr(self, "current_context_profile", None)
        )
        cp = self._apply_loop_balance_pressure(cp)
        question = getattr(self, "last_question", None)

        try:
            options = self.simulator.generate_options(
                self.memory,
                num_actions=num_actions,
                question=question,
                context_profile=cp,
                horizon=horizon,
            )
        except TypeError:
            try:
                options = self.simulator.generate_options(
                    self.memory,
                    num_actions=num_actions,
                    question=question,
                    context_profile=cp,
                )
            except TypeError:
                try:
                    options = self.simulator.generate_options(
                        self.memory,
                        num_actions=num_actions,
                        question=question,
                    )
                except TypeError:
                    options = self.simulator.generate_options(self.memory, num_actions=num_actions)
        except Exception:
            options = []

        if not options:
            return None, None

        if bool(self._cfg("DEBUG_SIMULATOR_FIRST_ACTION", False)):
            try:
                print("\n=== DEBUG: FIRST ACTION FROM SIMULATOR ===")
                print(options[0])
                print("=== END DEBUG ===\n")
            except Exception:
                pass

        cp_dom = str((cp or {}).get("domain") or "general").strip().lower()
        ql = (question or "").lower()
        allow_explore = (cp_dom in {"survival", "tech"}) or any(
            t in ql for t in self.PLANNING_TRIGGERS
        )
        if not allow_explore:
            options = [
                o
                for o in options
                if not (isinstance(o, dict) and o.get("details", {}).get("is_explore"))
            ]
            if not options:
                return None, None

        if cp_dom == "clubhouse_build":
            q_family = self._clubhouse_query_family(ql)
            if q_family == "floor_frame":
                filtered = []
                for o in options:
                    if not isinstance(o, dict):
                        continue
                    blob = " ".join(
                        [
                            str(o.get("action_name") or ""),
                            str(o.get("action") or ""),
                            str(o.get("description") or ""),
                            str(o.get("seed") or ""),
                            str(o.get("seed_text") or ""),
                            str((o.get("details") or {}).get("seed_text") or ""),
                            str((o.get("details") or {}).get("seed_memory") or ""),
                        ]
                    ).lower()
                    if "stable site" in blob or "good drainage" in blob:
                        continue
                    filtered.append(o)
                if filtered:
                    options = filtered

        validated = self.validator.validate(options, self.memory, self.risk_assessor)
        safe = self.hard_rules.enforce(validated)
        if not safe:
            return None, None

        chosen = self._choose_action(safe)
        if not chosen:
            return None, None

        if self.SPEAK_ACTION_CYCLE:
            try:
                decision_packet = {"chosen": chosen, "mode": self._get_mode(), "reasoning": None}
                speech, _meta = self.broca.formulate(decision_packet, self.dialogue)
                speech = self._emit(speech)
                if speech and bool(self._cfg("VERBOSE", False)):
                    print(f"[Clair]: {speech}")
            except Exception:
                pass

        results = self.actuator.execute(chosen, system_state=self._get_system_state())
        evaluations = self.performance.evaluate(results)

        feedback_packets: List[Dict[str, Any]] = []
        for r, e in zip(results or [], evaluations or []):
            action_name = r.get("action_name", "unknown")
            outcome = r.get("outcome", "unknown")
            ok = bool(r.get("ok", False))
            score = e.get("score") if isinstance(e, dict) else None

            feedback_packets.append(
                {
                    "type": "feedback",
                    "content": f"Executed {action_name}: ok={ok} outcome={outcome} score={score}",
                    "confidence": 0.7,
                    "weight": 0.7,
                    "context": chosen.get("context", []) or [],
                    "source": "system",
                    "evidence": [],
                    "last_verified": None,
                    "conflict": False,
                    "domain": "operations" if self.QUARANTINE_FEEDBACK else "general",
                    "tags": ["feedback"] if self.QUARANTINE_FEEDBACK else [],
                    "kind": "episode",
                }
            )

        try:
            self.intake.queue_feedback(feedback_packets)
        except Exception:
            pass

        try:
            self.memory.store(feedback_packets)
        except Exception:
            pass

        self.reflect(force=False)

        try:
            commits = self.reflector.process(
                results,
                evaluations,
                self.memory,
                context_weights=self._cfg("DEFAULT_CONTEXT_WEIGHTS", {}),
            )
            if bool(self._cfg("VERBOSE", False)):
                print(f"[Reflection] commits={commits}")
        except Exception as e:
            if bool(self._cfg("VERBOSE", False)):
                print(f"[Reflection ERROR] {e}")

        self.consolidate_memory()
        self.last_action_cycle_ts = time.time()
        return results, evaluations

    def _should_run_action_cycle(self, approved_count: int, action_worthy_tick: bool) -> bool:
        now = time.time()

        if self.last_action_cycle_ts and (
            now - self.last_action_cycle_ts
        ) < self.ACTION_CYCLE_COOLDOWN_SEC:
            return False

        cp = getattr(self, "current_context_profile", {}) or {}
        dom = str(cp.get("domain") or "general").strip().lower()

        if dom in {"survival", "tech"}:
            return True

        if self.REQUIRE_ACTION_WORTHY_TICK and not action_worthy_tick:
            urgency = float(cp.get("urgency", 0.0) or 0.0)
            threat = float(cp.get("threat", 0.0) or 0.0)
            if urgency < self.ACTION_URGENCY_THRESHOLD and threat < self.ACTION_THREAT_THRESHOLD:
                return False

        if self.BLOCK_ACTION_CYCLE_AFTER_FACT_RECALL and self.last_answer_was_recall:
            urgency = float(cp.get("urgency", 0.0) or 0.0)
            threat = float(cp.get("threat", 0.0) or 0.0)
            if urgency < self.ACTION_URGENCY_THRESHOLD and threat < self.ACTION_THREAT_THRESHOLD:
                return False

        if approved_count > 0 and action_worthy_tick:
            return True

        if self.last_question_ts and (now - self.last_question_ts) <= self.ACTION_QUESTION_FRESH_SEC:
            cpq = getattr(self, "last_question_context_profile", {}) or {}
            if bool(cpq.get("action_ok", False)):
                return True

        urgency = float(cp.get("urgency", 0.0) or 0.0)
        threat = float(cp.get("threat", 0.0) or 0.0)
        return (urgency >= self.ACTION_URGENCY_THRESHOLD) or (
            threat >= self.ACTION_THREAT_THRESHOLD
        )

    def step(self) -> Dict[str, Any]:
        responses: List[str] = []
        processed = 0
        ran_action_cycle = False
        now = time.time()

        if self.deferred_queue:
            still_deferred = []
            for pkt in list(self.deferred_queue):
                try:
                    retry_count = int(getattr(pkt, "retry_count", 0) or 0)
                    next_time = float(getattr(pkt, "next_eligible_time", 0.0) or 0.0)
                except Exception:
                    retry_count = 0
                    next_time = 0.0

                if next_time and now < next_time:
                    still_deferred.append(pkt)
                    continue

                resp = None
                try:
                    resp = self.handle_packet(pkt)
                except Exception:
                    resp = None

                processed += 1
                if isinstance(resp, str) and resp.strip():
                    responses.append(self._emit(resp))

                if resp is None and retry_count < self.MAX_DEFER_RETRIES:
                    try:
                        pkt.retry_count = retry_count + 1
                        backoff = float(self._cfg("DEFER_BACKOFF_SEC", 0.4)) * (
                            1 + pkt.retry_count
                        )
                        pkt.next_eligible_time = now + backoff
                        still_deferred.append(pkt)
                    except Exception:
                        pass

            self.deferred_queue = still_deferred

        self.reflect(force=False)

        if self._pending_calibration is not None:
            return {
                "responses": [r for r in responses if r],
                "processed": processed,
                "ran_action_cycle": False,
            }

        cp = getattr(self, "current_context_profile", {}) or {}
        dom = str(cp.get("domain") or "general").lower().strip()
        q = (self.last_question or "").lower()
        action_worthy = (dom in {"survival", "tech"}) or any(
            t in q for t in self.PLANNING_TRIGGERS
        )

        if action_worthy and self._should_run_action_cycle(
            approved_count=1,
            action_worthy_tick=True,
        ):
            self._execute_action_cycle(
                context_profile=getattr(self, "last_question_context_profile", None)
                or getattr(self, "current_context_profile", None)
            )
            ran_action_cycle = True

        return {
            "responses": [r for r in responses if r],
            "processed": processed,
            "ran_action_cycle": ran_action_cycle,
        }

    # ------------------------------------------------------------------
    # Document ingest / opinion
    # ------------------------------------------------------------------
    def ingest_document(self, path: str) -> str:
        path = (path or "").strip().strip('"')
        if not path:
            return "No path provided."
        if not os.path.exists(path):
            return "Path does not exist."

        try:
            chunks = self.doc_reader.make_chunks(path, words_per_chunk=self.DOC_WORDS_PER_CHUNK)
        except Exception as e:
            return f"Failed to read document: {e}"

        if not chunks:
            return "No readable text found."

        self.current_context_profile = {
            "domain": "literature",
            "tags": ["literature", "reading"],
            "threat": 0.0,
            "urgency": 0.2,
            "goal": "interpret",
        }

        base = os.path.basename(path)
        _root, ext = os.path.splitext(base)

        total_claims = 0
        total_frames = 0
        total_summaries = 0
        total_verified_attempted = 0
        total_verified_supported = 0
        total_verified_contradicted = 0
        total_verified_updated = 0

        for i, ch in enumerate(chunks):
            try:
                if self.DOC_DEBUG_PREVIEW and i == 0:
                    preview = (ch.text or "")[:220].replace("\n", "\\n")
                    print(f"[DEBUG] chunk0 chars={len(ch.text or '')} preview={preview}")

                ex = self.angular_gyrus.extract(
                    ch.text,
                    max_claims=self.DOC_MAX_CLAIMS_PER_CHUNK,
                )

                chunk_doc_meta = dict(ch.meta or {})
                chunk_doc_meta.setdefault("filename", base)
                chunk_doc_meta.setdefault("file_name", base)
                chunk_doc_meta.setdefault("ext", ext.lower().lstrip("."))
                chunk_doc_meta.setdefault("title", chunk_doc_meta.get("title") or os.path.splitext(base)[0])
                chunk_doc_meta.setdefault("document_title", chunk_doc_meta.get("title"))
                chunk_doc_meta.setdefault("book_title_hint", chunk_doc_meta.get("book_title_hint"))
                chunk_doc_meta.setdefault("chapter_hint", chunk_doc_meta.get("chapter_hint"))
                chunk_doc_meta.setdefault("section_hint", chunk_doc_meta.get("section_hint"))
                chunk_doc_meta.setdefault("domain_hint", chunk_doc_meta.get("domain_hint") or "literature")
                chunk_doc_meta.setdefault("reader_name", chunk_doc_meta.get("reader_name") or "document_reader")
                chunk_doc_meta.setdefault(
                    "evidence_id",
                    chunk_doc_meta.get("evidence_id") or f"{ch.doc_id}:chunk_{ch.chunk_id}",
                )

                frame_candidates = self._extract_frame_candidates_for_ingest(
                    ch.text,
                    chunk_doc_meta,
                )

                section_summary = self._make_section_summary_for_ingest(
                    text=ch.text,
                    claims=list(ex.claims or []),
                    frames=frame_candidates,
                    doc_meta=chunk_doc_meta,
                )

                chapter_summary = self._make_chapter_summary_for_ingest(
                    section_summary=section_summary,
                    frames=frame_candidates,
                    doc_meta=chunk_doc_meta,
                )

                if section_summary:
                    section_summary = self._summary_text_cleanup(section_summary)
                if chapter_summary:
                    chapter_summary = self._summary_text_cleanup(chapter_summary)

                ingest_tags = list(ex.keywords or [])
                ingest_tags.extend(["reading", "literature"])

                if chunk_doc_meta.get("book_title_hint"):
                    ingest_tags.append(
                        str(chunk_doc_meta["book_title_hint"]).strip().lower().replace(" ", "_")
                    )
                if chunk_doc_meta.get("chapter_hint"):
                    ingest_tags.append(
                        f"chapter_{str(chunk_doc_meta['chapter_hint']).strip().lower()}"
                    )
                if chunk_doc_meta.get("section_hint"):
                    ingest_tags.append(
                        f"section_{str(chunk_doc_meta['section_hint']).strip().lower()}"
                    )

                seen_tags = set()
                deduped_tags: List[str] = []
                for t in ingest_tags:
                    ts = str(t).strip()
                    if not ts:
                        continue
                    key = ts.lower()
                    if key in seen_tags:
                        continue
                    seen_tags.add(key)
                    deduped_tags.append(ts)

                out = self.hippocampus.store_claims(
                    doc_id=ch.doc_id,
                    chunk_id=ch.chunk_id,
                    claims=ex.claims,
                    domain="literature",
                    tags=deduped_tags,
                    persist_to_ltm=self.DOC_PERSIST_TO_LTM,
                    doc_meta=chunk_doc_meta,
                    context_profile=self.current_context_profile,
                    frame_candidates=frame_candidates,
                    section_summary=section_summary,
                    chapter_summary=chapter_summary,
                )

                verify_stats = self._verify_ingested_claims_against_chunk(
                    self._extract_claim_texts(list(ex.claims or [])),
                    ch.text,
                    source_name=f"document_chunk:{os.path.basename(path)}",
                    max_verify=self.DOC_VERIFY_MAX_PER_CHUNK,
                )

                total_verified_attempted += int(verify_stats.get("attempted", 0) or 0)
                total_verified_supported += int(verify_stats.get("supported", 0) or 0)
                total_verified_contradicted += int(
                    verify_stats.get("contradicted", 0) or 0
                )
                total_verified_updated += int(verify_stats.get("updated", 0) or 0)

                if isinstance(out, tuple):
                    if len(out) >= 3:
                        c_cnt, f_cnt, s_cnt = (
                            int(out[0] or 0),
                            int(out[1] or 0),
                            int(out[2] or 0),
                        )
                    elif len(out) >= 2:
                        c_cnt, f_cnt = int(out[0] or 0), int(out[1] or 0)
                        s_cnt = 0
                    else:
                        c_cnt, f_cnt, s_cnt = int(out[0] or 0), 0, 0
                else:
                    c_cnt, f_cnt, s_cnt = int(out or 0), 0, 0

                total_claims += c_cnt
                total_frames += f_cnt
                total_summaries += s_cnt
                self._record_learning_event(max(1, c_cnt + f_cnt + s_cnt))

            except Exception as e:
                if bool(self._cfg("VERBOSE", False)):
                    print(f"[ingest_document] chunk {i} failed: {e}")
                continue

        verify_enabled = bool(self.DOC_VERIFY_ON_INGEST and self.thalamus_verifier is not None)

        if verify_enabled:
            return (
                f"Learned {total_claims} claims + {total_frames} frames + "
                f"{total_summaries} summaries from {len(chunks)} chunks. "
                f"Verified {total_verified_attempted} claims "
                f"(supported={total_verified_supported}, "
                f"contradicted={total_verified_contradicted}, "
                f"updated={total_verified_updated})."
            )

        return (
            f"Learned {total_claims} claims + {total_frames} frames + "
            f"{total_summaries} summaries from {len(chunks)} chunks."
        )

    def generate_opinion(self, topic: str) -> str:
        topic = (topic or "").strip()
        if not topic:
            return "No topic provided."

        dom = str(self.current_context_profile.get("domain") or "general").strip().lower()
        if dom == "general":
            self.current_context_profile["domain"] = "literature"
            tags = set(self.current_context_profile.get("tags") or [])
            tags.add("literature")
            self.current_context_profile["tags"] = sorted(tags)
            self.current_context_profile["goal"] = "interpret"

        op = self.pfc.form_opinion(
            topic,
            domain=self.current_context_profile.get("domain", "general"),
        )
        stance = op.get("stance", "mixed")
        summary = op.get(
            "summary",
            "I don't have enough learned material to form an opinion yet.",
        )
        return f"My stance: {stance}. {summary}"

    # ------------------------------------------------------------------
    # CLI
    # ------------------------------------------------------------------
    def _prompt_line(self, prompt: str) -> Optional[str]:
        try:
            s = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nClair shutting down gracefully.")
            return None
        return s

    def _learnmode_accidental_command(self, lower: str) -> bool:
        accidental = {
            "read",
            "opinion",
            "lesson",
            "learn",
            "teach",
            "short",
            "normal",
            "detailed",
            "exit",
            "quit",
            "learnmode on",
            "learnmode off",
            "learnmode",
            "calibrate",
            "calibration",
            "sleep",
            "sleep mode",
            "recalibrate",
            "verify",
        }
        return lower in accidental

    def _looks_like_file_input(self, s: str) -> bool:
        if not s:
            return False
        t = s.strip().strip('"')
        low = t.lower()
        if low.endswith(self.FILE_EXTS):
            return True
        return os.path.exists(t)

    def _clean_path_guess(self, s: str) -> str:
        return (s or "").strip().strip('"')

    def cli_loop(self) -> None:
        print(
            "Clair CLI online. Commands: lesson:, read:, opinion:, verify:, "
            "short|normal|detailed, learnmode on|off, calibrate, sleep, exit"
        )
        while True:
            try:
                user_text = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nClair shutting down gracefully.")
                return

            if not user_text:
                msg = self._calibration_idle_tick()
                if msg:
                    print(f"[Clair]: {self._emit(msg)}")
                continue

            lower = user_text.lower().strip()

            if lower in {"exit", "quit"}:
                print("Clair shutting down gracefully.")
                return

            if self._pending_calibration is not None:
                applied = self._apply_calibration_answer(user_text)
                if applied:
                    print(f"[Clair]: {self._emit(applied)}")
                    continue

            if lower in {"short", "normal", "detailed"}:
                self.dialogue.verbosity = lower
                print(f"[Clair]: {self._emit(f'Verbosity set to {lower}.')}")
                continue

            if lower in {"calibrate", "calibration"}:
                msg = self._calibration_idle_tick()
                if msg:
                    print(f"[Clair]: {self._emit(msg)}")
                else:
                    print(f"[Clair]: {self._emit('No calibration questions right now.')}")
                continue

            if lower in {"sleep", "sleep mode", "recalibrate"}:
                print(f"[Clair]: {self._emit(self._run_sleep_calibration())}")
                continue

            if lower.startswith("learnmode"):
                if "on" in lower:
                    self.learnmode = True
                    print(
                        f"[Clair]: {self._emit('Learnmode enabled. I will store non-command lines as lessons.')}"
                    )
                elif "off" in lower:
                    self.learnmode = False
                    print(f"[Clair]: {self._emit('Learnmode disabled.')}")
                else:
                    state = "on" if self.learnmode else "off"
                    print(
                        f"[Clair]: {self._emit(f'Learnmode is {state}. Use learnmode on|off.')}"
                    )
                continue

            if self._looks_like_file_input(user_text) and not lower.startswith(
                ("read", "lesson:", "learn:", "teach:", "opinion", "verify:")
            ):
                path_guess = self._clean_path_guess(user_text)
                if os.path.exists(path_guess):
                    msg = self.ingest_document(path_guess)
                    print(f"[Clair]: {self._emit(msg)}")
                    continue
                print(f"[Clair]: {self._emit('That looks like a file. Use: read: <full path>')}")
                continue

            if lower == "read" or lower.startswith("read:") or lower.startswith("read "):
                if lower == "read":
                    path = self._prompt_line("Path: ")
                    if path is None:
                        return
                    path = path.strip().strip('"')
                elif lower.startswith("read "):
                    path = user_text.split(" ", 1)[1].strip().strip('"')
                else:
                    path = user_text.split(":", 1)[1].strip().strip('"')

                if not path:
                    print(f"[Clair]: {self._emit('No path provided.')}")
                    continue

                msg = self.ingest_document(path)
                print(f"[Clair]: {self._emit(msg)}")
                continue

            if lower == "verify":
                print(
                    f"[Clair]: {self._emit('Use: verify: <memory text> || <outside intake text>  or  verify: <memory text>')}"
                )
                continue

            if lower == "opinion" or lower.startswith("opinion:") or lower.startswith("opinion "):
                if lower == "opinion":
                    topic = self._prompt_line("Topic: ")
                    if topic is None:
                        return
                    topic = topic.strip()
                elif lower.startswith("opinion "):
                    topic = user_text.split(" ", 1)[1].strip()
                else:
                    topic = user_text.split(":", 1)[1].strip()

                if not topic:
                    print(f"[Clair]: {self._emit('No topic provided.')}")
                    continue

                msg = self.generate_opinion(topic)
                print(f"[Clair]: {self._emit(msg)}")
                continue

            if lower.startswith("verify:"):
                payload = user_text.split(":", 1)[1].strip()

                if "||" in payload:
                    mem_query, intake_text = [x.strip() for x in payload.split("||", 1)]
                else:
                    mem_query, intake_text = payload.strip(), ""

                target = self.find_memory_by_text(mem_query)
                if not target:
                    print(f"[Clair]: {self._emit('No matching memory found for verification.')}")
                    continue

                routed = self.route_verify_request(
                    target,
                    intake_text=intake_text or None,
                    source_name="cli_verify",
                    apply=True if intake_text else False,
                )

                if not routed.get("ok", False):
                    err = routed.get("error", "Verification failed.")
                    print(f"[Clair]: {self._emit(str(err))}")
                    continue

                route_name = str(routed.get("route", "unknown"))
                if route_name == "external_stub":
                    msg = (
                        f"Internal confidence={routed.get('best_internal_confidence', 0.0):.2f}. "
                        f"Conflict={bool(routed.get('internal_conflict', False))}. "
                        f"External verification stub reached: external verification not implemented yet."
                    )
                    print(f"[Clair]: {self._emit(msg)}")
                    continue

                if route_name == "internal_only":
                    msg = (
                        f"Internal verification route: supported. "
                        f"Confidence={routed.get('best_internal_confidence', 0.0):.2f}. "
                        f"No external check needed."
                    )
                    print(f"[Clair]: {self._emit(msg)}")
                    continue

                verification = routed.get("verification", {}) or {}
                if not verification.get("ok", False):
                    err = verification.get("error", "Verification failed.")
                    print(f"[Clair]: {self._emit(str(err))}")
                    continue

                vr = verification.get("verification_result", {}) or {}
                status = vr.get("status", "unknown")
                verdict = vr.get("verdict", "unsure")
                detail = vr.get("external_detail", "No detail.")
                applied = bool(verification.get("applied", False))

                if applied:
                    msg = (
                        f"Verification route={route_name}. Status={status}. "
                        f"Verdict={verdict}. Memory updated. {detail}"
                    )
                else:
                    msg = (
                        f"Verification route={route_name}. Status={status}. "
                        f"Verdict={verdict}. Verification ran without memory update. {detail}"
                    )

                print(f"[Clair]: {self._emit(msg)}")
                continue

            if lower.startswith(("lesson:", "learn:", "teach:")):
                text = user_text.split(":", 1)[1].strip()
                pkt = SimplePacket(text, ptype="lesson", packet_id=f"cli_{int(time.time()*1000)}")
                _ = self.handle_packet(pkt)
                print(f"[Clair]: {self._emit('Learned.')}")
                continue

            normalized = self._normalize_identity_lesson(user_text)
            if normalized:
                pkt = SimplePacket(
                    normalized,
                    ptype="lesson",
                    packet_id=f"cli_{int(time.time()*1000)}",
                )
                _ = self.handle_packet(pkt)
                print(f"[Clair]: {self._emit('Noted.')}")
                continue

            if self.learnmode and self._learnmode_accidental_command(lower):
                print(
                    f"[Clair]: {self._emit('Command detected. Use read: <path> or opinion: <topic>.')}"
                )
                continue

            if self.learnmode and not lower.endswith("?"):
                pkt = SimplePacket(
                    user_text,
                    ptype="lesson",
                    packet_id=f"cli_{int(time.time()*1000)}",
                )
                _ = self.handle_packet(pkt)
                print(f"[Clair]: {self._emit('Learned.')}")
                continue

            pkt = SimplePacket(user_text, ptype="ask", packet_id=f"cli_{int(time.time()*1000)}")
            response = self.handle_packet(pkt)
            if response:
                print(f"[Clair]: {self._emit(response)}")

            if self._pending_calibration is not None:
                continue

            cp = getattr(self, "current_context_profile", {}) or {}
            dom = str(cp.get("domain") or "general").lower().strip()
            action_worthy = (dom in {"survival", "tech"}) or any(
                t in lower for t in self.PLANNING_TRIGGERS
            )

            if action_worthy and self._should_run_action_cycle(
                approved_count=1,
                action_worthy_tick=True,
            ):
                self._execute_action_cycle(
                    context_profile=getattr(self, "last_question_context_profile", None)
                    or getattr(self, "current_context_profile", None)
                )


if __name__ == "__main__":
    Clair().cli_loop()