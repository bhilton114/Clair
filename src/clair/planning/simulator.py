# FILE: planning/simulator.py
# Clair Simulator (v2.57)
# Rewrite goals:
# - preserve public API used by Clair/tests
# - harden survival planning so hazard-locked scenarios never collapse to zero options
# - demote generic fact seeds during planning
# - strengthen domain-aware procedural seed selection
# - expose internal debug counters for candidate-pool forensics
# - bias seed choice toward top-ranked local candidates
# - penalize non-local revision seeds in stage-specific planning
# - improve non-survival local precision under adversarial pressure

from __future__ import annotations

import hashlib
import math
import random
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set, Tuple

import config

try:
    from executive.goal_manager import GoalManager  # noqa: F401
    from executive.priority_manager import PriorityManager  # noqa: F401
except Exception:
    GoalManager = None  # type: ignore
    PriorityManager = None  # type: ignore


@dataclass(frozen=True)
class HazardGuess:
    hazard: str
    confidence: float
    evidence: List[str]


class Simulator:
    """
    Clair Simulator

    v2.57:
    - preserves v2.56 survival hardening / ranking bias / revision locality handling
    - adds stronger non-survival locality penalties for cross-stage clubhouse seeds
    - adds stronger garden symptom-locality bonuses
    - penalizes broad garden facts when the query is symptom-specific
    - adds final non-survival locality penalty before option ranking
    """

    VERSION = "2.57"
    DEFAULT_RISK = 0.5

    HAZARD_FAMILIES: Set[str] = {"fire", "flood", "earthquake", "lost"}

    HAZARD_PATTERNS: Dict[str, List[Tuple[str, float]]] = {
        "earthquake": [
            (r"\b(aftershock|foreshock|tremor|seismic|epicenter|fault\s*line)\b", 3.0),
            (r"\b(earthquake|quake)\b", 2.8),
            (r"\b(magnitude\s*\d(\.\d)?)\b", 2.0),
            (r"\b(shaking|building\s+swaying|rumbling|ground\s+is\s+shaking|ground\s+shaking)\b", 2.2),
            (r"\b(drop\s*,?\s*cover\s*,?\s*(and\s*)?hold\s+on)\b", 2.2),
            (r"\b(debris\s+is\s+falling|structural\s+damage|building\s+damage)\b", 1.6),
        ],
        "flood": [
            (r"\b(flash\s*flood|flood|flooding|inundat(e|ion))\b", 3.2),
            (r"\b(floodwater|flood\s*water|rising\s+water|water\s+is\s+rising)\b", 2.8),
            (r"\b(roads?\s+(are\s+)?(underwater|submerged)|underwater\s+roads?)\b", 2.8),
            (r"\b(higher\s+ground|move\s+to\s+higher\s+ground|go\s+to\s+higher\s+ground)\b", 2.4),
            (r"\b(evacuat(e|ion)\s+orders?|evacuate\s+low\s+areas?|leave\s+low\s+areas?)\b", 2.4),
            (r"\b(do\s+not\s+drive\s+through\s+floodwater|avoid\s+driving\s+through\s+floodwater)\b", 2.8),
            (r"\b(storm\s+surge|levee|sandbag)\b", 1.8),
            (r"\b(swift\s+water|fast[-\s]+moving\s+water)\b", 2.2),
        ],
        "fire": [
            (r"\b(fire|wildfire|flames?|embers?)\b", 3.2),
            (r"\b(smoke|smoky|smoke\s+everywhere|thick\s+smoke)\b", 2.5),
            (r"\b(smoke\s+inhalation)\b", 2.0),
            (r"\b(get\s+low|crawl\s+low|stay\s+low)\b", 2.0),
            (r"\b(closed\s+room|trapped\s+inside|blocked\s+exit)\b", 1.6),
            (r"\b(call\s+(911|emergency\s+services))\b", 1.2),
        ],
        "lost": [
            (r"\b(lost|missing)\b", 2.6),
            (r"\b(at\s+night|nighttime)\b", 1.4),
            (r"\b(stay\s+put|stop\s+moving)\b", 2.0),
            (r"\b(make\s+shelter|build\s+shelter)\b", 1.6),
            (r"\b(signal\s+for\s+help)\b", 1.6),
        ],
    }

    RAW_SCENARIO_CUES: Dict[str, List[str]] = {
        "flood": [
            r"\bwater\s+is\s+rising\b",
            r"\brising\s+water\b",
            r"\bwater\s+around\s+(the\s+)?car\b",
            r"\bcar\s+is\s+in\s+water\b",
            r"\bvehicle\s+in\s+water\b",
            r"\broad\s+is\s+underwater\b",
            r"\broads?\s+are\s+underwater\b",
            r"\bstreet\s+is\s+flooded\b",
            r"\bstreets?\s+are\s+flooded\b",
            r"\bswift[-\s]+moving\s+water\b",
            r"\blow[-\s]+lying\s+area\b",
            r"\btrapped\s+by\s+water\b",
            r"\bflash\s+flood\b",
            r"\bflood\s+warning\b",
        ],
        "fire": [
            r"\broom\s+is\s+filling\s+with\s+smoke\b",
            r"\bhouse\s+is\s+on\s+fire\b",
            r"\bbuilding\s+is\s+on\s+fire\b",
            r"\bsmoke\s+is\s+coming\s+under\s+the\s+door\b",
            r"\bexit\s+is\s+blocked\s+by\s+smoke\b",
            r"\bflames\s+are\s+spreading\b",
            r"\btrapped\s+in\s+(a\s+)?burning\b",
            r"\bsmell\s+smoke\b",
            r"\bthick\s+smoke\b",
            r"\bsmoke[-\s]*filled\b",
            r"\bwildfire\s+is\s+approaching\b",
        ],
        "earthquake": [
            r"\bground\s+is\s+shaking\b",
            r"\bthe\s+ground\s+is\s+shaking\b",
            r"\bbuilding\s+is\s+shaking\b",
            r"\bbuilding\s+is\s+swaying\b",
            r"\bceiling\s+is\s+falling\b",
            r"\bthings?\s+are\s+falling\b",
            r"\bdebris\s+is\s+falling\b",
            r"\baftershocks?\s+(are\s+)?expected\b",
            r"\bwalls?\s+are\s+cracking\b",
            r"\bunder\s+(a\s+)?desk\b",
            r"\bunder\s+(a\s+)?table\b",
            r"\bstrong\s+shaking\b",
        ],
        "lost": [
            r"\bcannot\s+find\s+the\s+trail\b",
            r"\boff\s+trail\b",
            r"\bstranded\b",
            r"\bcan'?t\s+find\s+my\s+way\b",
            r"\bdisoriented\b",
        ],
    }

    NEGATIVE_HINTS: Dict[str, List[Tuple[str, float]]] = {
        "earthquake": [(r"\b(flood|flooding|flash\s*flood|levee|sandbag|floodwater)\b", 1.2)],
        "flood": [(r"\b(aftershock|fault\s*line|seismic|earthquake|quake)\b", 1.4)],
        "fire": [(r"\b(flood|earthquake|aftershock)\b", 0.8)],
        "lost": [(r"\b(flood|earthquake|fire|wildfire)\b", 0.4)],
    }

    TAG_TO_FAMILY: Dict[str, Optional[str]] = {
        "fire": "fire",
        "smoke": "fire",
        "burning": "fire",
        "heat": "fire",
        "flood": "flood",
        "flooding": "flood",
        "floodwater": "flood",
        "submerged": "flood",
        "underwater": "flood",
        "evacuate": "flood",
        "earthquake": "earthquake",
        "aftershock": "earthquake",
        "tremor": "earthquake",
        "lost": "lost",
        "survival": None,
        "emergency": None,
        "danger": None,
        "risk": None,
    }

    DOMAIN_ALIASES: Dict[str, str] = {
        "planning": "clubhouse_build",
        "foundation": "clubhouse_build",
        "posts": "clubhouse_build",
        "floor": "clubhouse_build",
        "framing": "clubhouse_build",
        "clubhouse": "clubhouse_build",
        "safety": "safety",
    }

    NON_SURVIVAL_HINTS: Dict[str, Dict[str, Any]] = {
        "clubhouse_build": {
            "anchors": {
                "level", "uneven", "frame", "joist", "joists", "spacing", "plywood", "floor",
                "framing", "chalk", "line", "footprint", "support", "posts", "post",
                "concrete", "gravel", "diagonal", "square", "deck", "screws", "nails",
                "visibility", "children", "playhouse", "roof", "wall", "walls", "door",
                "alignment", "bracket", "hangers",
            },
            "domain_penalty_terms": {
                "garden", "watering", "seedlings", "breaker", "wiring", "circuit", "tester",
                "mulch", "transplant", "voltage",
            },
        },
        "garden": {
            "anchors": {
                "garden", "raised", "beds", "watering", "seedlings", "mulch", "soil",
                "drainage", "transplant", "overwatering", "yellow", "yellowing", "leaves",
                "fungal", "moisture", "vegetables", "morning", "evening", "soggy", "wilt",
                "wilting", "hardening", "stress",
            },
            "domain_penalty_terms": {
                "joists", "plywood", "wiring", "breaker", "voltage", "concrete", "posts", "frame",
            },
        },
        "electrical": {
            "anchors": {
                "electrical", "wiring", "breaker", "circuit", "voltage", "tester", "deenergized",
                "de", "energized", "flickering", "lights", "neutral", "ground", "hot",
                "wire", "wires", "short", "overload", "live", "wet", "switch", "power",
                "tripping", "trip", "microwave", "connection", "connections",
            },
            "domain_penalty_terms": {
                "joists", "plywood", "raised", "beds", "garden", "gravel", "post", "mulch",
            },
        },
    }

    FACT_DIM_TERMS: Set[str] = {
        "water", "human", "humans", "everest", "python", "rain", "sahara",
        "heart", "sharks", "bones", "celsius", "meters", "desert",
    }

    PROCEDURAL_VERBS: Set[str] = {
        "check", "verify", "measure", "move", "avoid", "leave", "evacuate", "call",
        "stop", "stay", "signal", "conserve", "plan", "list", "timebox", "review",
        "install", "level", "correct", "repair", "inspect", "shut", "turn", "test",
        "diagnose", "secure", "protect", "drop", "cover", "hold",
    }

    SURVIVAL_TEMPLATES: Dict[str, List[str]] = {
        "fire": [
            "Leave the fire area immediately using the safest available exit.",
            "Stay low to avoid smoke and move away from flames.",
            "Once clear, call emergency services and do not re-enter the building.",
        ],
        "flood": [
            "Move to higher ground immediately and get out of floodwater.",
            "Avoid submerged roads and do not drive through floodwater.",
            "Once safer, follow evacuation guidance and stay out of low areas.",
        ],
        "earthquake": [
            "Drop, cover, and hold on until the shaking stops.",
            "Protect your head and neck and stay away from falling debris.",
            "After the shaking stops, move carefully and expect aftershocks.",
        ],
        "lost": [
            "Stop moving if possible and avoid wandering farther off route.",
            "Signal for help and make yourself easier to locate.",
            "Find shelter, conserve energy, and stabilize your position.",
        ],
        "unknown": [
            "Take the safest immediate action to reduce danger.",
            "Move away from the most immediate hazard and reassess.",
            "Get to a safer position and seek help if needed.",
        ],
    }

    _ACTION_SCORE_SPAM_RE = re.compile(r"\baction[_\s]+action[_\s]*\d+.*\bscored\b", re.IGNORECASE)
    _ACTION_NAME_SCORED_RE = re.compile(r"^\s*action_\d+_.*\bscored\b", re.IGNORECASE)
    _BANNED_SEED_SUBSTRINGS = {
        "chatgpt is your mother",
        "your mother ai",
        "i am your mother ai",
        "i'm your mother ai",
        "you are my mother ai",
    }
    _CODEISH_RE = re.compile(r"\b(traceback|file \"|line \d+|exception|pip install|import \w+)\b", re.IGNORECASE)

    def __init__(
        self,
        exploration_rate=None,
        history_window=None,
        risk_assessor=None,
        num_rollouts=None,
        risk_penalty=None,
        uncertainty_penalty=None,
        horizon=None,
        step_discount=None,
        goal_manager=None,
        priority_manager=None,
        mode_getter=None,
        system_state_getter=None,
        deterministic: Optional[bool] = None,
        rng_salt: Optional[str] = None,
    ):
        self.exploration_rate = float(
            exploration_rate if exploration_rate is not None else getattr(config, "SIMULATOR_EXPLORATION_RATE", 0.2)
        )
        self.history_window = int(
            history_window if history_window is not None else getattr(config, "SIMULATOR_HISTORY_WINDOW", 25)
        )
        self.risk_assessor = risk_assessor
        self.num_rollouts = int(
            num_rollouts if num_rollouts is not None else getattr(config, "SIMULATOR_NUM_ROLLOUTS", 5)
        )
        self.risk_penalty = float(
            risk_penalty if risk_penalty is not None else getattr(config, "SIMULATOR_RISK_PENALTY", 0.0)
        )
        self.uncertainty_penalty = float(
            uncertainty_penalty if uncertainty_penalty is not None
            else getattr(config, "SIMULATOR_UNCERTAINTY_PENALTY", 0.5)
        )
        self.horizon = int(horizon if horizon is not None else getattr(config, "SIMULATOR_HORIZON", 1))
        self.horizon = 1 if self.horizon < 1 else (2 if self.horizon > 2 else self.horizon)
        self.step_discount = float(
            step_discount if step_discount is not None else getattr(config, "SIMULATOR_STEP_DISCOUNT", 0.85)
        )

        self.goal_manager = goal_manager
        self.priority_manager = priority_manager
        self.mode_getter = mode_getter
        self.system_state_getter = system_state_getter

        if deterministic is None:
            deterministic = bool(getattr(config, "SIMULATOR_DETERMINISTIC", True))
        self.deterministic = bool(deterministic)

        if rng_salt is None:
            rng_salt = str(getattr(config, "SIMULATOR_RNG_SALT", ""))
        self.rng_salt = str(rng_salt or "")

        self.rollouts_scale = float(getattr(self, "rollouts_scale", 1.0))
        self.horizon_hint = getattr(self, "horizon_hint", None)

        self.past_actions: List[str] = []
        self._risk_cache: Dict[str, Tuple[float, Optional[list]]] = {}

        self.followup_similarity_max = float(getattr(config, "SIMULATOR_FOLLOWUP_SIMILARITY_MAX", 0.78))
        self.followup_min_delta_len = int(getattr(config, "SIMULATOR_FOLLOWUP_MIN_DELTA_LEN", 10))
        self.followup_candidate_cap = int(getattr(config, "SIMULATOR_FOLLOWUP_CANDIDATE_CAP", 18))
        self.followup_lock_rescue_bias = float(getattr(config, "SIMULATOR_FOLLOWUP_LOCK_RESCUE_BIAS", 0.70))

    # ------------------------------------------------------------------
    # Hypothalamus integration
    # ------------------------------------------------------------------
    def apply_overrides(self, overrides: Dict[str, Any]) -> None:
        if not isinstance(overrides, dict):
            return
        if "horizon_hint" in overrides:
            try:
                self.horizon_hint = int(overrides["horizon_hint"])
            except Exception:
                pass
        if "rollouts_scale" in overrides:
            try:
                self.rollouts_scale = float(overrides["rollouts_scale"])
            except Exception:
                pass
        if "exploration_rate" in overrides:
            try:
                self.exploration_rate = float(overrides["exploration_rate"])
            except Exception:
                pass
        if "deterministic" in overrides:
            try:
                self.deterministic = bool(overrides["deterministic"])
            except Exception:
                pass

    # ------------------------------------------------------------------
    # RNG helpers
    # ------------------------------------------------------------------
    def _stable_seed_u32(self, material: str) -> int:
        h = hashlib.sha256(material.encode("utf-8", errors="ignore")).hexdigest()
        return int(h[:8], 16)

    def _context_seed_override(self, context_profile) -> Optional[Any]:
        if not isinstance(context_profile, dict):
            return None
        return context_profile.get("rng_seed")

    def _make_rng(
        self,
        question: Optional[str],
        domain: str,
        mode: str,
        hazard_hint: Optional[str],
        context_profile=None,
    ) -> random.Random:
        if not self.deterministic:
            return random.Random((time.time_ns() ^ (id(self) << 1)) & 0xFFFFFFFF)

        override = self._context_seed_override(context_profile)
        if isinstance(override, int):
            return random.Random(int(override) & 0xFFFFFFFF)
        if override is not None:
            return random.Random(self._stable_seed_u32(f"clair_rng_override|{override}"))

        q = (question or "").strip().lower()
        mat = (
            f"clair_rng|v={self.VERSION}|salt={self.rng_salt}|"
            f"q={q}|domain={domain}|mode={mode}|hazard={hazard_hint or ''}"
        )
        return random.Random(self._stable_seed_u32(mat))

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------
    def _clamp01(self, x, default=0.0) -> float:
        try:
            v = float(x)
        except Exception:
            v = float(default)
        return max(0.0, min(1.0, v))

    def _clamp(self, x, lo: float, hi: float, default: float = 1.0) -> float:
        try:
            v = float(x)
        except Exception:
            v = float(default)
        return max(lo, min(hi, v))

    def _safe_text(self, x, max_len: int = 240) -> str:
        if x is None:
            return ""
        try:
            s = x.strip() if isinstance(x, str) else str(x).strip()
        except Exception:
            return ""
        if len(s) > max_len:
            s = s[:max_len] + "…"
        return s

    def _ensure_context(self, ctx) -> List[Any]:
        if ctx is None:
            return []
        if isinstance(ctx, list):
            return ctx
        return [ctx]

    def _looks_like_action_name(self, s: str) -> bool:
        return bool(re.match(r"^action_\d+_[a-z0-9_]+$", (s or "").strip().lower()))

    def _strip_action_prefix(self, s: str) -> str:
        s = (s or "").strip()
        m = re.match(r"^action_\d+_(.+)$", s.lower())
        return m.group(1) if m else s

    def _slug(self, text: str, max_tokens: int = 8, max_len: int = 40) -> str:
        toks = re.findall(r"[a-z0-9]+", (text or "").lower())
        slug = "_".join(toks[:max_tokens]) if toks else "explore"
        return (slug[:max_len].strip("_") or "explore")

    def _unique_name(self, base: str, rng: random.Random) -> str:
        if self.deterministic:
            return base
        window = int(self.history_window or 25)
        if base not in self.past_actions[-window:]:
            return base
        return f"{base}_v{rng.randint(1, 99)}"

    def _ctx_domain(self, context_profile) -> str:
        if isinstance(context_profile, dict):
            d = context_profile.get("domain")
            if isinstance(d, str) and d.strip():
                dom = d.strip().lower()
                return self.DOMAIN_ALIASES.get(dom, dom)
        return "general"

    def _ctx_tags(self, context_profile) -> Set[str]:
        if not isinstance(context_profile, dict):
            return set()
        tags = context_profile.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        out: Set[str] = set()
        for t in tags:
            try:
                s = str(t).strip().lower()
                if s:
                    out.add(s)
            except Exception:
                continue
        return out

    def _ctx_hazard_family(self, context_profile) -> Optional[str]:
        if not isinstance(context_profile, dict):
            return None
        hf = context_profile.get("hazard_family")
        if isinstance(hf, str) and hf.strip():
            hf = hf.strip().lower()
            return hf if hf in self.HAZARD_FAMILIES else None
        for t in self._ctx_tags(context_profile):
            fam = self.TAG_TO_FAMILY.get(t)
            if fam in self.HAZARD_FAMILIES:
                return fam
        return None

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------
    def _stem(self, w: str) -> str:
        w = (w or "").strip().lower()
        if len(w) <= 3:
            return w
        for suf in ("ing", "ed"):
            if w.endswith(suf) and len(w) > len(suf) + 2:
                w = w[:-len(suf)]
                break
        if w.endswith("s") and len(w) > 4 and not w.endswith("ss"):
            w = w[:-1]
        return w

    def _tokens(self, text: str) -> Set[str]:
        raw = re.sub(r"[^a-z0-9\-]+", " ", (text or "").lower()).strip()
        if not raw:
            return set()
        toks = {self._stem(t) for t in raw.split() if t}
        out: Set[str] = set()
        for t in toks:
            out.add(t)
            if t in {"de-energized", "deenergized"}:
                out.add("de")
                out.add("energized")
        return out

    def _normalized_overlap(self, a: Set[str], b: Set[str]) -> int:
        if not a or not b:
            return 0
        return len(a & b)

    def _active_domain_anchors(self, q_words: Set[str], domain: str) -> Set[str]:
        hint = self.NON_SURVIVAL_HINTS.get(domain) or {}
        anchors = set(hint.get("anchors") or set())
        if not anchors:
            return set()
        active = set(q_words & anchors)
        if active:
            return active
        return set(list(anchors)[:6])

    def _question_focus_terms(self, q_words: Set[str], domain: str) -> Set[str]:
        focus = set(self._active_domain_anchors(q_words, domain))

        if domain == "clubhouse_build":
            if {"floor", "frame"} & q_words:
                focus.update({"floor", "frame", "level"})
            if {"joist", "joists", "spacing"} & q_words:
                focus.update({"joist", "joists", "spacing"})
            if "plywood" in q_words:
                focus.add("plywood")
            if "uneven" in q_words:
                focus.update({"uneven", "level"})
        elif domain == "garden":
            if {"yellow", "yellowing", "leaves"} & q_words:
                focus.update({"yellow", "yellowing", "leaves", "overwatering", "soggy"})
            if {"soggy", "soil"} & q_words:
                focus.update({"soggy", "soil", "overwatering"})
            if {"evening", "morning", "watering"} & q_words:
                focus.update({"watering", "morning", "evening", "fungal", "overwatering"})
            if {"seedlings", "transplant", "wilting", "wilt"} & q_words:
                focus.update({"seedlings", "transplant", "wilting", "hardening", "stress"})
        elif domain == "electrical":
            if {"breaker", "tripping", "trip"} & q_words:
                focus.update({"breaker", "tripping", "overload", "short"})
            if {"wire", "wires", "tester", "voltage"} & q_words:
                focus.update({"wire", "wires", "tester", "voltage", "de", "energized"})
            if {"flickering", "lights", "connection", "connections"} & q_words:
                focus.update({"flickering", "lights", "connection", "connections"})

        return focus

    def _local_anchor_overlap(self, seed_toks: Set[str], q_words: Set[str], domain: str) -> Tuple[int, int]:
        active = self._question_focus_terms(q_words, domain)
        if not active:
            return 0, 0
        hits = len(seed_toks & active)
        return hits, len(active)

    def _clubhouse_stage_bonus(self, seed_toks: Set[str], q_words: Set[str]) -> float:
        bonus = 0.0
        if {"floor", "frame"} & q_words:
            if {"level", "frame"} & seed_toks:
                bonus += 1.4
            if {"joist", "joists", "spacing"} & seed_toks:
                bonus += 1.2
            if "plywood" in q_words and "plywood" in seed_toks:
                bonus += 0.8
        if "uneven" in q_words and {"level", "alignment"} & seed_toks:
            bonus += 0.8
        return bonus

    def _garden_symptom_bonus(self, seed_toks: Set[str], q_words: Set[str], seed_text: str) -> float:
        bonus = 0.0
        s = (seed_text or "").lower()

        yellow_query = bool({"yellow", "yellowing", "leaves"} & q_words)
        soggy_query = bool({"soggy", "soil"} & q_words)
        watering_query = bool({"watering", "evening", "morning"} & q_words)
        transplant_query = bool({"seedlings", "seedling", "transplant", "wilting", "wilt"} & q_words)

        # strongest signal: yellow + soggy => overwatering
        if yellow_query and soggy_query:
            if {"overwatering", "yellow", "soggy"} & seed_toks:
                bonus += 2.6
            if "overwatering" in s:
                bonus += 1.0

        # watering timing / fungal risk
        if watering_query:
            if {"morning", "fungal", "watering"} & seed_toks:
                bonus += 1.2
            if "fungal risk" in s:
                bonus += 0.5

        # transplant-only cues should matter less unless transplant is actually in the prompt
        if transplant_query:
            if {"seedlings", "transplant", "hardening", "stress"} & seed_toks:
                bonus += 0.9
        else:
            if {"seedlings", "transplant", "hardening", "stress"} & seed_toks:
                bonus -= 1.4

        return bonus

    def _electrical_symptom_bonus(self, seed_toks: Set[str], q_words: Set[str], seed_text: str) -> float:
        bonus = 0.0
        s = (seed_text or "").lower()

        if {"breaker", "tripping", "trip"} & q_words:
            if {"breaker", "overload", "short"} & seed_toks:
                bonus += 1.8

        if {"flickering", "lights"} & q_words:
            if {"flickering", "lights", "connection", "connections"} & seed_toks:
                bonus += 1.6
            if "loose wire" in s:
                bonus += 0.7

        if {"tester", "voltage", "wire", "wires"} & q_words:
            if {"tester", "voltage", "wire", "wires", "de", "energized"} & seed_toks:
                bonus += 1.4

        return bonus

    def _clubhouse_forbidden_cross_stage_hits(self, seed_toks: Set[str], q_words: Set[str]) -> int:
        """
        Penalize post/foundation-stage seeds when the question is specifically
        about floor-frame / joist / plywood stage.
        """
        frame_stage = bool({"floor", "frame", "joist", "joists", "plywood", "uneven", "spacing"} & q_words)
        post_stage_terms = {"post", "posts", "depth", "hole", "holes", "concrete", "gravel", "pour"}

        if not frame_stage:
            return 0
        if {"post", "posts"} & q_words:
            return 0

        return len(seed_toks & post_stage_terms)

    def _garden_symptom_locality_hits(self, seed_toks: Set[str], q_words: Set[str]) -> int:
        """
        Count direct symptom-local troubleshooting hits for garden prompts.
        """
        active: Set[str] = set()

        if {"yellow", "yellowing", "leaves"} & q_words:
            active.update({"yellow", "yellowing", "leaves", "overwatering", "soggy"})
        if {"soggy", "soil"} & q_words:
            active.update({"soggy", "soil", "overwatering"})
        if {"evening", "watering"} & q_words:
            active.update({"evening", "morning", "watering", "fungal"})
        if {"wilting", "wilt", "seedlings", "transplant"} & q_words:
            active.update({"wilting", "wilt", "seedlings", "transplant", "hardening", "stress"})

        return len(seed_toks & active)

    def _general_garden_fact_penalty(self, seed_toks: Set[str], q_words: Set[str], seed_text: str) -> float:
        """
        Penalize broad plant-care facts when the query is clearly about a local symptom pattern.
        """
        low = (seed_text or "").lower()
        symptom_query = bool({"yellow", "yellowing", "soggy", "wilting", "wilt", "evening"} & q_words)

        if not symptom_query:
            return 0.0

        generic_markers = 0
        if {"tomatoes", "watering", "soil"} <= seed_toks or "well-drained soil" in low:
            generic_markers += 1
        if "consistent watering" in low:
            generic_markers += 1

        symptom_hits = self._garden_symptom_locality_hits(seed_toks, q_words)
        if generic_markers > 0 and symptom_hits <= 1:
            return 2.4
        return 0.0

    def _non_survival_option_locality_penalty(
        self,
        seed_text: str,
        seed_ref: Optional[dict],
        domain: str,
        query_words: Optional[Set[str]],
    ) -> float:
        q_words = set(query_words or set())
        seed_toks = self._tokens(seed_text)

        if domain == "clubhouse_build":
            hits = self._clubhouse_forbidden_cross_stage_hits(seed_toks, q_words)
            if hits > 0:
                return min(3.0, 1.2 * hits)

        if domain == "garden":
            return self._general_garden_fact_penalty(seed_toks, q_words, seed_text)

        return 0.0

    # ------------------------------------------------------------------
    # Feedback / junk filters
    # ------------------------------------------------------------------
    def _is_feedbackish_text(self, text: Optional[str]) -> bool:
        q = (text or "").strip()
        if not q:
            return False
        ql = q.lower()
        if "follow_up_from:action_" in ql:
            return True
        if self._ACTION_SCORE_SPAM_RE.search(q):
            return True
        if self._ACTION_NAME_SCORED_RE.search(q):
            return True
        if ql.startswith("action action_") and "scored" in ql:
            return True
        return False

    def _seed_quality_ok(self, seed: str) -> bool:
        s = (seed or "").strip()
        if not s:
            return False
        sl = s.lower()
        if any(b in sl for b in self._BANNED_SEED_SUBSTRINGS):
            return False
        if self._is_feedbackish_text(s):
            return False
        if self._CODEISH_RE.search(sl):
            return False
        if len(s) < 6:
            return False
        if self._looks_like_action_name(sl) or re.match(r"^explore_\d+$", sl):
            return False
        return True

    def _is_unusable_seed(self, seed: str) -> bool:
        return not self._seed_quality_ok(seed)

    def _is_generic_planning_seed(self, seed_text: str) -> bool:
        t = (seed_text or "").strip().lower()
        if not t:
            return False
        generic_patterns = [
            r"\bclarify\s+goals\b",
            r"\bclarify\s+the\s+situation\b",
            r"\bgoals?\s*,\s*constraints?\b",
            r"\bavailable\s+resources\b",
            r"\bgather\s+more\s+information\b",
            r"\blist\s+immediate\s+next\s+steps\b",
            r"\bexplore\s+to\s+gather\b",
        ]
        return any(re.search(p, t) for p in generic_patterns)

    def _procedural_verb_count(self, text: str) -> int:
        toks = self._tokens(text)
        return len(toks & self.PROCEDURAL_VERBS)

    def _looks_like_fact_seed(self, seed: str) -> bool:
        low = (seed or "").strip().lower()
        if not low:
            return False
        toks = self._tokens(low)

        has_procedural = self._procedural_verb_count(low) > 0
        if has_procedural:
            return False

        if any(x in low for x in (
            " is ", " are ", " was ", " were ",
            "degrees celsius", "meters tall", "programming language",
            "largest hot desert", "pumps blood", "have 206 bones"
        )):
            return True

        if len(toks & self.FACT_DIM_TERMS) >= 2:
            return True

        return False

    def _normalize_seed_for_key(self, seed_text: str) -> str:
        s = (seed_text or "").strip().lower()
        return re.sub(r"\s+", " ", s)

    # ------------------------------------------------------------------
    # Mode/system/exec helpers
    # ------------------------------------------------------------------
    def _get_mode(self) -> str:
        if callable(self.mode_getter):
            try:
                m = self.mode_getter()
                return str(m).strip().lower() if m else "default"
            except Exception:
                return "default"
        return str(getattr(config, "MODE", "default")).strip().lower()

    def _get_system_state(self) -> dict:
        if callable(self.system_state_getter):
            try:
                s = self.system_state_getter()
                return s if isinstance(s, dict) else {}
            except Exception:
                return {}
        return {
            "confidence": float(getattr(config, "CONFIDENCE", 0.5)),
            "overload": float(getattr(config, "OVERLOAD", 0.0)),
        }

    def _goal_weights(self) -> dict:
        gm = self.goal_manager
        if gm is None:
            return {}
        try:
            if hasattr(gm, "get_weights"):
                w = gm.get_weights()
                return w if isinstance(w, dict) else {}
        except Exception:
            return {}
        return {}

    def _priority_multiplier(self, option: dict) -> float:
        pm = self.priority_manager
        if pm is None:
            return 1.0
        mode = self._get_mode()
        system_state = self._get_system_state()
        goal_weights = self._goal_weights()
        try:
            if hasattr(pm, "option_multiplier"):
                mult = pm.option_multiplier(
                    option=option,
                    mode=mode,
                    goal_weights=goal_weights,
                    system_state=system_state,
                )
                return self._clamp(mult, 0.50, 1.50, default=1.0)
        except Exception:
            return 1.0
        return 1.0

    # ------------------------------------------------------------------
    # Hazard inference
    # ------------------------------------------------------------------
    def _seed_ref_hazard_bonus(self, seed_ref: Optional[dict], hazard: str) -> Tuple[float, List[str]]:
        if not isinstance(seed_ref, dict):
            return 0.0, []

        bonus = 0.0
        ev: List[str] = []

        tags = seed_ref.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        if isinstance(tags, (list, tuple)):
            for t in tags:
                ts = str(t).strip().lower()
                fam = self.TAG_TO_FAMILY.get(ts)
                if fam == hazard:
                    bonus += 1.3
                    ev.append(f"tag:{ts}")

        dom = str(seed_ref.get("domain") or "").strip().lower()
        if dom == "survival":
            bonus += 0.2

        content = str(seed_ref.get("content") or "").strip().lower()
        if content:
            if hazard == "flood" and re.search(r"\b(flood|floodwater|underwater|submerged|higher\s+ground|evacuat|rising\s+water)\b", content):
                bonus += 1.0
                ev.append("seed_ref_content:flood")
            elif hazard == "earthquake" and re.search(r"\b(earthquake|aftershock|quake|ground\s+shaking|tremor|drop\s+cover)\b", content):
                bonus += 1.0
                ev.append("seed_ref_content:earthquake")
            elif hazard == "fire" and re.search(r"\b(fire|smoke|flame|burning|get\s+low)\b", content):
                bonus += 1.0
                ev.append("seed_ref_content:fire")
            elif hazard == "lost" and re.search(r"\b(lost|stay\s+put|signal\s+for\s+help|shelter)\b", content):
                bonus += 1.0
                ev.append("seed_ref_content:lost")

        return bonus, ev

    def _infer_hazard(self, text: Optional[str], seed_ref: Optional[dict] = None) -> HazardGuess:
        t = (text or "").lower().strip()
        if not t:
            return HazardGuess("unknown", 0.0, [])

        if isinstance(seed_ref, dict):
            tags = seed_ref.get("tags") or []
            if isinstance(tags, str):
                tags = [tags]
            if isinstance(tags, (list, tuple)):
                tag_blob = " ".join(str(x).lower() for x in tags if str(x).strip())
                if tag_blob:
                    t = t + " " + tag_blob

        scores: Dict[str, float] = {h: 0.0 for h in self.HAZARD_FAMILIES}
        evidence: Dict[str, List[str]] = {h: [] for h in self.HAZARD_FAMILIES}

        for hazard, pats in self.HAZARD_PATTERNS.items():
            for pat, w in pats:
                m = re.search(pat, t)
                if m:
                    scores[hazard] += w
                    evidence[hazard].append(m.group(0))

        for hazard in self.HAZARD_FAMILIES:
            bonus, ev = self._seed_ref_hazard_bonus(seed_ref, hazard)
            if bonus > 0.0:
                scores[hazard] += bonus
                evidence[hazard].extend(ev)

        for hazard, negs in self.NEGATIVE_HINTS.items():
            for pat, w in negs:
                if re.search(pat, t):
                    scores[hazard] -= w

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_h, top_s = ranked[0]
        runner_s = ranked[1][1] if len(ranked) > 1 else 0.0

        if top_s <= 0.8:
            return HazardGuess("unknown", 0.25 if top_s > 0 else 0.0, [])
        if (top_s - runner_s) < 0.75:
            return HazardGuess("unknown", 0.35, evidence[top_h])

        conf = min(0.95, 0.45 + 0.15 * top_s)
        return HazardGuess(top_h, conf, evidence[top_h][:8])

    def _broad_raw_hazard_signal(self, question: Optional[str], context_profile=None) -> HazardGuess:
        text = (question or "").strip().lower()
        if not text:
            ctx_fam = self._ctx_hazard_family(context_profile)
            if ctx_fam in self.HAZARD_FAMILIES:
                return HazardGuess(ctx_fam, 0.75, [f"context:{ctx_fam}"])
            return HazardGuess("unknown", 0.0, [])

        ctx_fam = self._ctx_hazard_family(context_profile)
        scores: Dict[str, float] = {h: 0.0 for h in self.HAZARD_FAMILIES}
        evidence: Dict[str, List[str]] = {h: [] for h in self.HAZARD_FAMILIES}

        for fam, pats in self.RAW_SCENARIO_CUES.items():
            for pat in pats:
                m = re.search(pat, text)
                if m:
                    scores[fam] += 1.7
                    evidence[fam].append(m.group(0))

        hg = self._infer_hazard(text, None)
        if hg.hazard in self.HAZARD_FAMILIES:
            scores[hg.hazard] += max(0.0, float(hg.confidence) * 2.2)
            evidence[hg.hazard].extend(hg.evidence[:4])

        if ctx_fam in self.HAZARD_FAMILIES:
            scores[ctx_fam] += 0.8
            evidence[ctx_fam].append(f"context:{ctx_fam}")

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_h, top_s = ranked[0]
        runner_s = ranked[1][1] if len(ranked) > 1 else 0.0

        if top_s < 1.2:
            return HazardGuess("unknown", 0.0, [])

        conf = min(0.96, 0.50 + 0.14 * top_s - 0.05 * runner_s)
        if (top_s - runner_s) < 0.35:
            conf = min(conf, 0.62)
        return HazardGuess(top_h, conf, evidence[top_h][:8])

    def _raw_survival_domain(self, question: Optional[str], context_profile=None) -> str:
        ctx_dom = self._ctx_domain(context_profile)
        if ctx_dom == "survival":
            return "survival"

        broad = self._broad_raw_hazard_signal(question, context_profile=context_profile)
        if broad.hazard in self.HAZARD_FAMILIES and broad.confidence >= 0.55:
            return "survival"

        hg = self._infer_hazard(question or "", None)
        if hg.hazard in self.HAZARD_FAMILIES and hg.confidence >= 0.55:
            return "survival"

        return ctx_dom

    def _raw_hazard_pin(self, question: Optional[str], context_profile=None) -> Optional[str]:
        ctx_h = self._ctx_hazard_family(context_profile)
        if ctx_h in self.HAZARD_FAMILIES:
            return ctx_h

        broad = self._broad_raw_hazard_signal(question, context_profile=context_profile)
        if broad.hazard in self.HAZARD_FAMILIES and broad.confidence >= 0.55:
            return broad.hazard

        hg = self._infer_hazard(question or "", None)
        if hg.hazard in self.HAZARD_FAMILIES and hg.confidence >= 0.55:
            return hg.hazard

        return None

    def _question_hazard_pin(self, question: Optional[str], domain: str) -> HazardGuess:
        if domain != "survival":
            return HazardGuess("unknown", 0.0, [])
        return self._infer_hazard(question or "", None)

    # ------------------------------------------------------------------
    # Rollout / risk helpers
    # ------------------------------------------------------------------
    def _effective_horizon(self, horizon: Optional[int]) -> int:
        if horizon is not None:
            h = int(horizon)
        else:
            hint = getattr(self, "horizon_hint", None)
            h = int(hint) if hint is not None else int(self.horizon)
        return 1 if h < 1 else (2 if h > 2 else h)

    def _effective_rollouts(self, domain: str) -> int:
        base = max(1, int(self.num_rollouts or 5))

        try:
            scale = float(getattr(self, "rollouts_scale", 1.0))
        except Exception:
            scale = 1.0
        scale = self._clamp(scale, 0.20, 2.00, default=1.0)

        mode = self._get_mode()
        if mode == "predicting":
            scale *= 1.25
        elif mode == "judging":
            scale *= 1.05
        elif mode == "curiosity":
            scale *= 0.95
        elif mode == "rush":
            scale *= 0.55
        elif mode == "idling":
            scale *= 0.45

        if domain == "survival":
            scale = min(scale, 1.10)

        n = int(round(base * scale))
        return max(1, min(int(getattr(config, "SIMULATOR_MAX_ROLLOUTS", 12)), n))

    def _dynamic_exploration_rate(self, context_profile) -> float:
        dom = self._ctx_domain(context_profile)
        mode = self._get_mode()
        base = float(self.exploration_rate or 0.0)

        if dom == "survival":
            return 0.0

        if mode == "curiosity":
            base = min(0.60, base + 0.15)
        elif mode == "judging":
            base = max(0.00, base - 0.10)
        elif mode == "predicting":
            base = max(0.00, base - 0.05)
        elif mode == "rush":
            base = max(0.00, base - 0.08)
        elif mode == "idling":
            base = max(0.00, base - 0.12)

        return self._clamp(base, 0.0, 0.75, default=0.2)

    def _apply_risk_assessment(self, option: dict) -> dict:
        if not isinstance(option, dict):
            option = {}

        details = option.get("details")
        if not isinstance(details, dict):
            details = {}
            option["details"] = details

        if not self.risk_assessor:
            option["risk"] = self._clamp01(option.get("risk", self.DEFAULT_RISK), default=self.DEFAULT_RISK)
            details["risk"] = option["risk"]
            return option

        cache_key = details.get("option_key") if isinstance(details, dict) else None
        if cache_key and cache_key in self._risk_cache:
            r, reasons = self._risk_cache[cache_key]
            option["risk"] = self._clamp01(r, default=self.DEFAULT_RISK)
            details["risk"] = option["risk"]
            if reasons is not None:
                details["risk_reasons"] = reasons
            return option

        try:
            result = self.risk_assessor.assess(option)
        except Exception as e:
            if getattr(config, "VERBOSE", False):
                print(f"[Simulator] RiskAssessor failed: {e}")
            result = None

        reasons_out = None
        if isinstance(result, (int, float, str)):
            option["risk"] = result
        elif isinstance(result, dict):
            if "risk" in result and not ("name" in result and "details" in result):
                option["risk"] = result.get("risk")
                if result.get("reasons") is not None:
                    reasons_out = result.get("reasons")
            else:
                if "risk" in result:
                    option["risk"] = result.get("risk")
                res_details = result.get("details")
                if isinstance(res_details, dict) and res_details.get("risk_reasons") is not None:
                    reasons_out = res_details.get("risk_reasons")

        option["risk"] = self._clamp01(option.get("risk", self.DEFAULT_RISK), default=self.DEFAULT_RISK)
        details["risk"] = option["risk"]
        if reasons_out is not None:
            details["risk_reasons"] = reasons_out

        if cache_key:
            self._risk_cache[cache_key] = (float(option["risk"]), reasons_out)

        return option

    def _simulate_rollouts(self, option: dict, base_score: float, n_rollouts: int, rng: random.Random):
        rollouts: List[dict] = []
        n = max(1, int(n_rollouts))

        risk = self._clamp01(option.get("risk", self.DEFAULT_RISK), default=self.DEFAULT_RISK)
        competence = self._clamp01(base_score, default=0.0)

        mode = self._get_mode()
        mode_skill = 0.03 if mode in {"predicting", "judging"} else (-0.03 if mode == "rush" else 0.0)

        for _ in range(n):
            p_success = (competence + mode_skill) * (1.0 - 0.6 * risk)
            p_success = max(0.0, min(1.0, p_success))

            success = rng.random() < p_success
            reward = rng.uniform(0.3, 1.0) * max(0.0, competence) if success else 0.0
            cost = rng.uniform(0.0, 1.0) * risk
            utility = reward - cost

            rollouts.append({
                "p_success": round(p_success, 3),
                "success": bool(success),
                "reward": round(reward, 3),
                "cost": round(cost, 3),
                "utility": round(utility, 3),
            })

        utilities = [r["utility"] for r in rollouts]
        expected = sum(utilities) / len(utilities)
        worst = min(utilities)
        mean = expected
        var = sum((u - mean) ** 2 for u in utilities) / len(utilities)
        std = math.sqrt(var)
        p_succ = sum(r["p_success"] for r in rollouts) / len(rollouts)

        return rollouts, expected, worst, std, p_succ

    # ------------------------------------------------------------------
    # Memory extraction / ranking
    # ------------------------------------------------------------------
    def _stable_mem_sort_key(self, m: Dict[str, Any]) -> Tuple:
        if not isinstance(m, dict):
            return ("", "", "", 0.0, 0.0)
        mtype = str(m.get("type") or "")
        dom = str(m.get("domain") or "")
        kind = str(m.get("kind") or "")
        content = str(m.get("content") or m.get("claim") or "")
        ts = float(m.get("timestamp", 0.0) or 0.0)
        w = float(m.get("weight", 0.0) or 0.0)
        c = float(m.get("confidence", 0.0) or 0.0)
        return (mtype, dom, kind, content, -c, -w, -ts)

    def _extract_seed_text(self, msg: dict) -> str:
        if not isinstance(msg, dict):
            return ""

        details = msg.get("details")
        if isinstance(details, dict):
            sm = self._safe_text(details.get("seed_memory"))
            if sm and not self._is_unusable_seed(sm) and not self._is_feedbackish_text(sm):
                return sm

        claim = msg.get("claim")
        if isinstance(claim, str) and claim.strip():
            c1 = self._safe_text(claim)
            if self._seed_quality_ok(c1) and not self._is_feedbackish_text(c1):
                return c1

        c = msg.get("content")
        if isinstance(c, str) and c.strip():
            if self._is_feedbackish_text(c):
                return ""
            if '"action_name"' in c or "'action_name'" in c:
                m = re.search(r"action_name['\"]?\s*[:=]\s*['\"]([^'\"]+)['\"]", c)
                if m:
                    cand = self._safe_text(m.group(1))
                    return cand if self._seed_quality_ok(cand) else ""
            c2 = self._safe_text(c)
            return c2 if self._seed_quality_ok(c2) else ""

        mtype = (msg.get("type") or "").lower()
        if mtype == "committed_action":
            cobj = msg.get("content_obj")
            if isinstance(cobj, dict):
                an = self._safe_text(cobj.get("action_name"))
                if an:
                    return f"Committed action: {an}"

        n = msg.get("name")
        if isinstance(n, str) and n.strip() and not self._is_feedbackish_text(n):
            n2 = self._safe_text(n)
            return n2 if self._seed_quality_ok(n2) else ""

        return ""

    def _make_seed_ref(self, msg: Optional[dict]) -> Optional[dict]:
        if not isinstance(msg, dict):
            return None
        try:
            t = (msg.get("type") or "").strip().lower() or None
            content = self._safe_text(msg.get("content") or msg.get("claim"))
            dom = msg.get("domain")
            dom = str(dom).strip().lower() if dom else None
            dom = self.DOMAIN_ALIASES.get(dom, dom) if dom else None

            tags = msg.get("tags") or []
            if isinstance(tags, set):
                tags = list(tags)
            if isinstance(tags, str):
                tags = [tags]
            if not isinstance(tags, (list, tuple)):
                tags = []
            tags = [str(x).strip().lower() for x in tags if str(x).strip()]

            seen = set()
            tags_u: List[str] = []
            for x in tags:
                if x not in seen:
                    tags_u.append(x)
                    seen.add(x)

            kind = msg.get("kind")
            kind = str(kind).strip().lower() if kind else None

            mid = msg.get("id", None)
            if mid is None:
                mid = msg.get("memory_id", None)

            ref = {"id": mid, "type": t, "content": content, "domain": dom, "tags": tags_u[:10], "kind": kind}
            return {k: v for k, v in ref.items() if v not in (None, "", [])}
        except Exception:
            return None

    def _derive_mem_weight(self, msg: dict) -> float:
        if not isinstance(msg, dict):
            return 0.5
        w = msg.get("weight", 0.5)
        c = msg.get("confidence", 0.8)
        try:
            w = float(w)
        except Exception:
            w = 0.5
        try:
            c = float(c)
        except Exception:
            c = 0.8
        if w < 1e-6:
            return max(0.2, min(1.0, c))
        return max(0.2, min(1.0, w))

    def _memory_planning_score(
        self,
        msg: dict,
        domain: str,
        q_words: Set[str],
        pinned_fam: Optional[str],
    ) -> Tuple[float, Dict[str, float]]:
        seed = self._extract_seed_text(msg)
        if not seed:
            return -999.0, {"empty_seed": -999.0}

        ref = self._make_seed_ref(msg)
        mtype = str(msg.get("type") or "").strip().lower()
        kind = str(msg.get("kind") or "").strip().lower()
        msg_domain = str(msg.get("domain") or "").strip().lower()
        msg_domain = self.DOMAIN_ALIASES.get(msg_domain, msg_domain)

        seed_toks = self._tokens(seed)
        overlap = float(len(seed_toks & q_words))
        proc = float(self._procedural_verb_count(seed))
        factish = 1.0 if self._looks_like_fact_seed(seed) else 0.0

        score = 0.0
        dbg: Dict[str, float] = {}

        score += overlap * 1.15
        dbg["overlap"] = overlap * 1.15

        score += min(2.8, proc * 0.95)
        dbg["procedural"] = min(2.8, proc * 0.95)

        mem_weight = self._derive_mem_weight(msg)
        score += mem_weight * 0.60
        dbg["mem_weight"] = mem_weight * 0.60

        if mtype in {"lesson", "procedure", "reasoning_action", "committed_action"}:
            score += 2.2
            dbg["type_bonus"] = 2.2
        elif mtype in {"fact", "claim"}:
            score += 0.2
            dbg["type_bonus"] = 0.2
        else:
            dbg["type_bonus"] = 0.0

        if kind in {"procedure", "goal"}:
            score += 1.0
            dbg["kind_bonus"] = 1.0
        elif kind == "revision":
            score += 0.45
            dbg["kind_bonus"] = 0.45
        else:
            dbg["kind_bonus"] = 0.0

        if msg_domain and msg_domain == domain:
            score += 1.4
            dbg["domain_match"] = 1.4
        elif msg_domain and msg_domain != domain:
            score -= 1.3
            dbg["domain_penalty"] = -1.3

        if domain == "survival" and pinned_fam in self.HAZARD_FAMILIES:
            hg = self._infer_hazard(seed, ref)
            if hg.hazard == pinned_fam:
                bonus = 4.5 + float(hg.confidence)
                score += bonus
                dbg["hazard_match"] = bonus
            elif hg.hazard in self.HAZARD_FAMILIES and hg.hazard != pinned_fam and hg.confidence >= 0.55:
                score -= 5.0
                dbg["hazard_mismatch"] = -5.0
            elif msg_domain == "survival":
                score += 0.7
                dbg["survival_domain_bonus"] = 0.7

        local_hits = 0
        active_anchor_count = 0

        if domain in self.NON_SURVIVAL_HINTS:
            anchors = set(self.NON_SURVIVAL_HINTS[domain]["anchors"])
            anchor_overlap = float(len(seed_toks & anchors))
            anchor_bonus = anchor_overlap * 0.90
            score += anchor_bonus
            dbg["anchor_overlap"] = anchor_bonus

            local_hits, active_anchor_count = self._local_anchor_overlap(seed_toks, q_words, domain)
            local_bonus = float(local_hits) * 1.35
            score += local_bonus
            dbg["local_anchor_bonus"] = local_bonus

            penalty_terms = set(self.NON_SURVIVAL_HINTS[domain].get("domain_penalty_terms") or set())
            wrong_domain_hits = len(seed_toks & penalty_terms)
            if wrong_domain_hits:
                pen = min(2.0, wrong_domain_hits * 0.55)
                score -= pen
                dbg["wrong_domain_term_penalty"] = -pen

            if domain == "clubhouse_build":
                cb = self._clubhouse_stage_bonus(seed_toks, q_words)
                score += cb
                dbg["clubhouse_stage_bonus"] = cb
            elif domain == "garden":
                gb = self._garden_symptom_bonus(seed_toks, q_words, seed)
                score += gb
            if {"yellow", "yellowing", "leaves", "soggy"} & q_words:
                if not ({"yellow", "yellowing", "soggy", "overwatering"} & seed_toks):
                    if {"seedlings", "transplant", "hardening", "stress"} & seed_toks:
                        score -= 2.2
                        dbg["garden_focus_mismatch_penalty"] = -2.2
                dbg["garden_symptom_bonus"] = gb

            if domain == "garden":
                if {"yellow", "yellowing", "leaves", "soggy"} & q_words:
                    exact_symptom_hits = len(seed_toks & {"yellow", "yellowing", "leaves", "soggy", "overwatering"})
                    if exact_symptom_hits >= 2:
                        bonus2 = 1.4
                        score += bonus2
                        dbg["garden_exact_symptom_bonus"] = bonus2
                        
            elif domain == "electrical":
                eb = self._electrical_symptom_bonus(seed_toks, q_words, seed)
                score += eb
                dbg["electrical_symptom_bonus"] = eb

            if kind == "revision":
                if active_anchor_count >= 2 and local_hits == 0:
                    score -= 4.2
                    dbg["revision_locality_penalty"] = -4.2
                elif local_hits == 1:
                    score -= 1.6
                    dbg["revision_locality_penalty"] = -1.6
                else:
                    score += 0.4
                    dbg["revision_locality_penalty"] = 0.4

        if domain == "clubhouse_build":
            cross_hits = self._clubhouse_forbidden_cross_stage_hits(seed_toks, q_words)
            if cross_hits > 0:
                pen = min(3.2, 1.35 * cross_hits)
                score -= pen
                dbg["clubhouse_cross_stage_penalty"] = -pen

        if domain == "garden":
            symptom_hits = self._garden_symptom_locality_hits(seed_toks, q_words)
            if symptom_hits > 0:
                bonus = min(3.0, 0.95 * symptom_hits)
                score += bonus
                dbg["garden_symptom_locality_bonus"] = bonus

            broad_fact_pen = self._general_garden_fact_penalty(seed_toks, q_words, seed)
            if broad_fact_pen > 0:
                score -= broad_fact_pen
                dbg["garden_general_fact_penalty"] = -broad_fact_pen

        if factish and domain in {"survival", "clubhouse_build", "garden", "electrical"}:
            score -= 3.2
            dbg["fact_penalty"] = -3.2
        elif factish and domain == "general":
            score -= 1.4
            dbg["fact_penalty"] = -1.4

        if self._is_generic_planning_seed(seed):
            score -= 0.8
            dbg["generic_penalty"] = -0.8

        dbg["local_hits"] = float(local_hits)
        dbg["active_anchor_count"] = float(active_anchor_count)

        return score, dbg

    def _candidate_memories(
        self,
        working_memory,
        limit: int = 30,
        context_profile=None,
        pinned_fam: Optional[str] = None,
        q_words: Optional[Set[str]] = None,
    ) -> Tuple[List[dict], Dict[str, int]]:
        debug = {
            "retrieved": 0,
            "dict_rows": 0,
            "seed_ok": 0,
            "feedback_filtered": 0,
            "bad_seed_filtered": 0,
            "fact_demoted": 0,
            "domain_filtered": 0,
            "kept": 0,
        }

        items: List[dict] = []
        try:
            items = working_memory.retrieve(count=limit, context_profile=context_profile, planning_only=True)
        except TypeError:
            try:
                items = working_memory.retrieve(count=limit, context_profile=context_profile)
            except TypeError:
                try:
                    items = working_memory.retrieve(count=limit)
                except Exception:
                    items = []
            except Exception:
                items = []
        except Exception:
            items = []

        domain = self._ctx_domain(context_profile)
        q_words = set(q_words or set())
        debug["retrieved"] = len(items or [])

        scored: List[Tuple[float, dict]] = []
        fallback_facts: List[Tuple[float, dict]] = []

        for m in items or []:
            if not isinstance(m, dict):
                continue
            debug["dict_rows"] += 1

            content = m.get("content", "") or m.get("claim", "")
            if isinstance(content, str) and self._is_feedbackish_text(content):
                debug["feedback_filtered"] += 1
                continue

            seed = self._extract_seed_text(m)
            if not seed:
                debug["bad_seed_filtered"] += 1
                continue
            if not self._seed_quality_ok(seed):
                debug["bad_seed_filtered"] += 1
                continue

            debug["seed_ok"] += 1

            msg_domain = str(m.get("domain") or "").strip().lower()
            msg_domain = self.DOMAIN_ALIASES.get(msg_domain, msg_domain)

            if domain in {"clubhouse_build", "garden", "electrical"} and msg_domain and msg_domain not in {domain, "general"}:
                debug["domain_filtered"] += 1
                continue

            score, score_dbg = self._memory_planning_score(m, domain, q_words, pinned_fam)

            m2 = dict(m)
            m2["_planning_rank"] = float(score)
            m2["_planning_dbg"] = dict(score_dbg)

            if self._looks_like_fact_seed(seed):
                debug["fact_demoted"] += 1
                fallback_facts.append((score, m2))
                continue

            scored.append((score, m2))

        scored.sort(key=lambda x: x[0], reverse=True)
        fallback_facts.sort(key=lambda x: x[0], reverse=True)

        kept = [m for _, m in scored[:limit]]
        if not kept and fallback_facts:
            kept = [m for _, m in fallback_facts[: max(4, min(limit, 8))]]

        debug["kept"] = len(kept)
        return kept, debug

    # ------------------------------------------------------------------
    # Seed selection / rescue
    # ------------------------------------------------------------------
    def _make_exploration_seed(self, question: Optional[str], context_profile=None) -> str:
        if question and self._is_feedbackish_text(question):
            question = None
        q = self._safe_text(question, max_len=140) if question else ""
        if q:
            return f"Explore to gather more information relevant to: {q}"
        dom = self._ctx_domain(context_profile)
        return f"Explore to gather more information for domain '{dom}'."

    def _thin_pool_rescue_seeds(self, question: Optional[str], domain: str, pinned_fam: Optional[str] = None) -> List[str]:
        q = self._safe_text(question, max_len=160) if question else ""
        out: List[str] = []

        if domain == "survival":
            fam = pinned_fam if pinned_fam in self.HAZARD_FAMILIES else "unknown"
            out.extend(self.SURVIVAL_TEMPLATES.get(fam, self.SURVIVAL_TEMPLATES["unknown"]))
            if q:
                out.append(f"Take the safest immediate {fam if fam != 'unknown' else 'survival'} action for: {q}")
            return out

        if domain == "clubhouse_build":
            out.extend([
                "Check level, spacing, and alignment before continuing the build.",
                "Correct the current framing issue before installing the next structural layer.",
                "Verify prerequisites before moving to the next clubhouse construction step.",
            ])
        elif domain == "garden":
            out.extend([
                "Check watering, drainage, and plant stress before changing the whole garden plan.",
                "Correct the likely garden-care cause before adding more water or fertilizer.",
                "Review the symptom and match it to the most likely plant-care issue.",
            ])
        elif domain == "electrical":
            out.extend([
                "Shut off power and verify the circuit is safe before touching wiring.",
                "Use the safest diagnostic step before handling the electrical problem directly.",
                "Check breaker status, de-energization, and obvious fault conditions first.",
            ])
        else:
            if q:
                out.append(f"Clarify the situation and constraints for: {q}")
                out.append(f"List immediate next steps for: {q}")
            else:
                out.append(f"Explore to gather more information for domain '{domain}'.")

        return out

    def _pick_seed(
        self,
        recent_msgs: List[dict],
        q_words: Set[str],
        rng: random.Random,
        question=None,
        context_profile=None,
        pinned_fam: Optional[str] = None,
    ):
        dom = self._ctx_domain(context_profile)
        have_mem = bool(recent_msgs)
        exp_rate = self._dynamic_exploration_rate(context_profile)

        use_memory = have_mem and (rng.random() > float(exp_rate))
        if dom == "survival":
            use_memory = have_mem

        if not use_memory:
            seed_text = self._make_exploration_seed(question, context_profile=context_profile)
            return seed_text, [], 0.5, None, None, True

        if dom in {"clubhouse_build", "garden", "electrical", "survival"}:
            top_k = min(4, len(recent_msgs))
        else:
            top_k = min(6, len(recent_msgs))

        pool = recent_msgs[: max(1, top_k)]

        weights: List[float] = []
        min_rank = min(float(m.get("_planning_rank", 0.0) or 0.0) for m in pool)

        for idx, m in enumerate(pool):
            rank = float(m.get("_planning_rank", 0.0) or 0.0)
            w = max(0.05, (rank - min_rank) + 0.25)

            if dom in {"clubhouse_build", "garden", "electrical"}:
                if idx == 0:
                    w += 2.0
                elif idx == 1:
                    w += 0.8
            elif dom == "survival":
                if idx == 0:
                    w += 1.4
                elif idx == 1:
                    w += 0.5

            weights.append(w)

        total = sum(weights)
        pick = rng.random() * total
        acc = 0.0
        msg = pool[0]
        for m, w in zip(pool, weights):
            acc += w
            if pick <= acc:
                msg = m
                break

        seed_text = self._extract_seed_text(msg)

        if self._is_unusable_seed(seed_text):
            seed_text = self._make_exploration_seed(question, context_profile=context_profile)
            return seed_text, [], 0.5, None, None, True

        ctx = self._ensure_context(msg.get("context"))
        mem_weight = self._derive_mem_weight(msg)
        seed_ref = self._make_seed_ref(msg)
        return seed_text, ctx, mem_weight, msg, seed_ref, False

    # ------------------------------------------------------------------
    # Option construction
    # ------------------------------------------------------------------
    def _build_option(
        self,
        i: int,
        seed_text: str,
        ctx: List[Any],
        mem_weight: float,
        source_mem: Optional[dict],
        seed_ref: Optional[dict],
        domain: str,
        is_explore: bool,
        n_rollouts: int,
        rng: random.Random,
        question_guess: Optional[HazardGuess] = None,
        pinned_fam: Optional[str] = None,
        query_words: Optional[Set[str]] = None,
        debug_counts: Optional[Dict[str, int]] = None,
    ) -> dict:
        seed_for_name = self._strip_action_prefix(seed_text) if self._looks_like_action_name(seed_text) else seed_text
        slug = self._slug(seed_for_name)

        base_name = self._unique_name(f"action_{i + 1}_{slug}", rng)
        base_score = round(self._clamp01(mem_weight, default=0.5) * rng.uniform(0.7, 1.0), 3)

        canonical_seed = self._safe_text(seed_text, max_len=320) or "Explore to gather more information."

        details: Dict[str, Any] = {
            "score": float(base_score),
            "seed_memory": canonical_seed,
            "seed_text": canonical_seed,
            "domain": domain,
            "is_explore": bool(is_explore),
            "mode": self._get_mode(),
            "rollouts_requested": int(n_rollouts),
        }
        if isinstance(seed_ref, dict) and seed_ref:
            details["seed_ref"] = seed_ref
        if isinstance(debug_counts, dict) and debug_counts:
            details["debug_counts"] = dict(debug_counts)

        hazard = None
        if domain == "survival":
            hg = self._infer_hazard(str(details.get("seed_text") or ""), details.get("seed_ref"))
            hazard = hg.hazard if hg.hazard in self.HAZARD_FAMILIES else None
            details["hazard_family"] = hazard
            details["hazard_confidence"] = round(float(hg.confidence), 3)
            if hg.evidence:
                details["hazard_evidence"] = list(hg.evidence)[:6]
            if isinstance(question_guess, HazardGuess):
                details["question_hazard"] = question_guess.hazard
                details["question_hazard_conf"] = round(float(question_guess.confidence), 3)
            if pinned_fam in self.HAZARD_FAMILIES:
                details["pinned_hazard_family"] = pinned_fam

        norm = self._normalize_seed_for_key(details["seed_text"])
        details["option_key"] = f"{domain}|{hazard or 'none'}|{norm}"

        option: Dict[str, Any] = {
            "name": base_name,
            "type": "reasoning_action",
            "weight": float(base_score),
            "details": details,
            "context": ctx,
        }
        if isinstance(source_mem, dict):
            option["source_memory"] = source_mem

        option = self._apply_risk_assessment(option)

        if isinstance(source_mem, dict) and domain in self.NON_SURVIVAL_HINTS:
            planning_dbg = source_mem.get("_planning_dbg")
            if isinstance(planning_dbg, dict):
                local_hits = int(float(planning_dbg.get("local_hits", 0.0) or 0.0))
                active_anchor_count = int(float(planning_dbg.get("active_anchor_count", 0.0) or 0.0))
                option["details"]["local_anchor_hits"] = local_hits
                option["details"]["active_anchor_count"] = active_anchor_count

                src_kind = str(source_mem.get("kind") or "").strip().lower()
                if src_kind == "revision" and active_anchor_count >= 2 and local_hits == 0:
                    option["details"]["revision_locality_penalty_final"] = -2.2
                elif src_kind == "revision" and local_hits == 1:
                    option["details"]["revision_locality_penalty_final"] = -0.8

        rollouts, expected, worst, std, p_success = self._simulate_rollouts(
            option,
            base_score,
            n_rollouts=n_rollouts,
            rng=rng,
        )
        option["details"]["rollouts"] = rollouts
        option["details"]["expected_utility_raw"] = round(expected, 3)
        option["details"]["worst_case_utility"] = round(worst, 3)
        option["details"]["utility_std"] = round(std, 3)
        option["details"]["p_success"] = round(p_success, 3)

        mult = self._priority_multiplier(option)
        option["details"]["priority_mult"] = round(mult, 3)

        expected_adj = expected * mult
        option["details"]["expected_utility"] = round(expected_adj, 3)

        final = expected_adj - (float(self.uncertainty_penalty) * std)
        if self.risk_penalty and float(self.risk_penalty) > 0.0:
            final -= float(self.risk_penalty) * float(option.get("risk", self.DEFAULT_RISK))

        if domain == "survival" and pinned_fam in self.HAZARD_FAMILIES and is_explore:
            final -= 1.10

        if domain == "survival" and self._looks_like_fact_seed(canonical_seed):
            final -= 4.2
        elif domain != "survival" and self._looks_like_fact_seed(canonical_seed):
            final -= 2.2

        if domain == "survival" and pinned_fam in self.HAZARD_FAMILIES:
            if hazard == pinned_fam:
                final += 1.00
            elif hazard in self.HAZARD_FAMILIES and hazard != pinned_fam:
                final -= 1.25

        final += float(option["details"].get("revision_locality_penalty_final", 0.0) or 0.0)

        if domain in {"clubhouse_build", "garden", "electrical"}:
            locality_pen = self._non_survival_option_locality_penalty(
                canonical_seed,
                seed_ref,
                domain,
                query_words,
            )
            if locality_pen > 0:
                final -= locality_pen
                option["details"]["non_survival_locality_penalty"] = round(-locality_pen, 3)

        option["weight"] = round(final, 3)
        return option

    def _score_sequence(self, step1: dict, step2: Optional[dict]) -> float:
        if not isinstance(step1, dict):
            return -999.0

        u1 = float(step1.get("details", {}).get("expected_utility", 0.0))
        s1 = float(step1.get("details", {}).get("utility_std", 0.0))
        r1 = float(step1.get("risk", self.DEFAULT_RISK))

        base1 = u1 - float(self.uncertainty_penalty) * s1
        if self.risk_penalty and float(self.risk_penalty) > 0.0:
            base1 -= float(self.risk_penalty) * r1

        if not step2:
            return round(base1, 3)

        u2 = float(step2.get("details", {}).get("expected_utility", 0.0))
        s2 = float(step2.get("details", {}).get("utility_std", 0.0))
        r2 = float(step2.get("risk", self.DEFAULT_RISK))

        base2 = u2 - float(self.uncertainty_penalty) * s2
        if self.risk_penalty and float(self.risk_penalty) > 0.0:
            base2 -= float(self.risk_penalty) * r2

        return round(base1 + float(self.step_discount) * base2, 3)

    def _similarity(self, a: str, b: str) -> float:
        a2 = self._normalize_seed_for_key(a or "")
        b2 = self._normalize_seed_for_key(b or "")
        if not a2 or not b2:
            return 0.0
        return float(SequenceMatcher(None, a2, b2).ratio())

    def _followup_is_too_similar(self, seed1: str, seed2: str) -> bool:
        s1 = (seed1 or "").strip()
        s2 = (seed2 or "").strip()
        if not s1 or not s2:
            return False
        if self._normalize_seed_for_key(s1) == self._normalize_seed_for_key(s2):
            return True
        sim = self._similarity(s1, s2)
        if sim >= float(self.followup_similarity_max):
            if abs(len(s1) - len(s2)) <= int(self.followup_min_delta_len):
                return True
            if sim >= 0.90:
                return True
        return False

    # ------------------------------------------------------------------
    # Followup / lock helpers
    # ------------------------------------------------------------------
    def _is_generic_safe_followup(self, seed_text: str) -> bool:
        t = (seed_text or "").lower()
        if not t.strip():
            return True
        if "explore to gather more information" in t:
            return True
        if re.search(r"\b(call\s+(911|emergency)|seek\s+help|ask\s+for\s+help)\b", t):
            return True
        if re.search(r"\b(check\s+injur|first\s+aid|stay\s+calm|move\s+to\s+safety)\b", t):
            return True
        return False

    def _survival_seed_strength(self, seed: str, seed_ref: Optional[dict], fam: str) -> float:
        hg = self._infer_hazard(seed, seed_ref)
        if hg.hazard == fam:
            return 4.0 + float(hg.confidence)

        s = (seed or "").lower()
        content_blob = s
        if isinstance(seed_ref, dict):
            content_blob += " " + str(seed_ref.get("content") or "").lower()
            tags = seed_ref.get("tags") or []
            if isinstance(tags, str):
                tags = [tags]
            if isinstance(tags, (list, tuple)):
                content_blob += " " + " ".join(str(x).lower() for x in tags)

        if fam == "flood" and re.search(r"\b(flood|floodwater|flooding|underwater|submerged|higher\s+ground|evacuat|rising\s+water)\b", content_blob):
            return 3.2
        if fam == "earthquake" and re.search(r"\b(earthquake|aftershock|quake|tremor|ground\s+shaking|drop\s+cover)\b", content_blob):
            return 3.2
        if fam == "fire" and re.search(r"\b(fire|smoke|flame|burning|get\s+low|crawl\s+low)\b", content_blob):
            return 3.2
        if fam == "lost" and re.search(r"\b(lost|stay\s+put|signal\s+for\s+help|shelter)\b", content_blob):
            return 3.2

        return 0.0

    def _followup_allowed_for_lock(
        self,
        seed_text: str,
        seed_ref: Optional[dict],
        lock_fam: Optional[str],
        is_explore: bool,
    ) -> bool:
        if lock_fam not in self.HAZARD_FAMILIES:
            return bool(is_explore or self._is_generic_safe_followup(seed_text))

        hg = self._infer_hazard(seed_text, seed_ref)
        if hg.hazard == lock_fam:
            return True

        strength = self._survival_seed_strength(seed_text, seed_ref, lock_fam)
        if strength >= 2.5:
            return True

        if is_explore or self._is_generic_safe_followup(seed_text):
            return True

        return False

    def _lock_from_signals(
        self,
        domain: str,
        question_guess: HazardGuess,
        step1_guess: HazardGuess,
        ctx_fam: Optional[str],
    ) -> str:
        if domain != "survival":
            return "unknown"

        lock_threshold = float(getattr(config, "SIMULATOR_HAZARD_LOCK_THRESHOLD", 0.65))

        if question_guess.hazard in self.HAZARD_FAMILIES and question_guess.confidence >= lock_threshold:
            return question_guess.hazard
        if step1_guess.hazard in self.HAZARD_FAMILIES and step1_guess.confidence >= lock_threshold:
            return step1_guess.hazard
        if ctx_fam in self.HAZARD_FAMILIES:
            return str(ctx_fam)
        return "unknown"

    def _followup_rescue_seeds(self, lock_fam: Optional[str], question: Optional[str], step1_seed: str) -> List[str]:
        q = self._safe_text(question, max_len=140) if question else ""
        s1 = self._safe_text(step1_seed, max_len=140)

        if lock_fam == "fire":
            seeds = [
                "Stay low and continue toward the nearest safe exit away from smoke.",
                "Check whether the exit remains clear and choose the safest alternate route if needed.",
                "Once clear of the fire area, stay outside and call emergency services.",
            ]
        elif lock_fam == "flood":
            seeds = [
                "Keep moving to higher ground and stay out of floodwater.",
                "Avoid submerged roads and move away from low areas immediately.",
                "Once in a safer place, monitor evacuation orders and do not re-enter floodwater.",
            ]
        elif lock_fam == "earthquake":
            seeds = [
                "Remain sheltered until the shaking stops and protect your head and neck.",
                "After the shaking stops, move carefully and watch for falling debris and broken utilities.",
                "Prepare for aftershocks and avoid damaged structures.",
            ]
        elif lock_fam == "lost":
            seeds = [
                "Stay put if possible and make yourself easier to locate.",
                "Signal for help and conserve energy instead of wandering farther.",
                "Secure shelter and maintain a stable position while awaiting rescue.",
            ]
        else:
            seeds = [
                "Take the next safest action that continues reducing immediate danger.",
                "Stay focused on immediate safety and avoid unrelated actions.",
            ]

        if q and lock_fam in self.HAZARD_FAMILIES:
            seeds.append(f"Continue {lock_fam} survival response for: {q}")
        elif s1:
            seeds.append(f"Follow through on the safest next step after: {s1}")

        return seeds

    def _build_followup_seed_pool(
        self,
        recent_msgs: List[dict],
        follow_words: Set[str],
        lock_fam: Optional[str],
        question: Optional[str],
        step1_seed: str,
        domain: str,
    ) -> List[Tuple[str, List[Any], float, Optional[dict], Optional[dict], bool]]:
        pool: List[Tuple[float, Tuple[str, List[Any], float, Optional[dict], Optional[dict], bool]]] = []

        for m in recent_msgs[: max(8, self.followup_candidate_cap)]:
            seed = self._extract_seed_text(m)
            if not seed:
                continue
            ctx = self._ensure_context(m.get("context"))
            mw = self._derive_mem_weight(m)
            ref = self._make_seed_ref(m)
            is_explore = False

            if lock_fam in self.HAZARD_FAMILIES and not self._followup_allowed_for_lock(seed, ref, lock_fam, is_explore):
                continue

            strength = 0.0
            overlap = len(self._tokens(seed) & follow_words)
            if lock_fam in self.HAZARD_FAMILIES:
                strength = self._survival_seed_strength(seed, ref, lock_fam)
                hg = self._infer_hazard(seed, ref)
                if hg.hazard in self.HAZARD_FAMILIES and hg.hazard != lock_fam and hg.confidence >= 0.60:
                    continue

            score = overlap + strength + float(mw)
            pool.append((score, (seed, ctx, mw, m, ref, is_explore)))

        rescue_seeds = self._followup_rescue_seeds(lock_fam, question, step1_seed)
        for idx, seed in enumerate(rescue_seeds):
            ref = None
            if lock_fam in self.HAZARD_FAMILIES:
                ref = {
                    "domain": "survival",
                    "tags": [lock_fam, "survival", "emergency", "followup"],
                    "content": seed,
                    "kind": "followup_rescue_seed",
                }
            mw = max(0.84, self.followup_lock_rescue_bias)
            score = 100.0 - float(idx)
            pool.append((score, (seed, [], mw, None, ref, False)))

        pool.sort(key=lambda x: x[0], reverse=True)

        seen: Set[str] = set()
        deduped: List[Tuple[str, List[Any], float, Optional[dict], Optional[dict], bool]] = []
        for _, item in pool:
            key = self._normalize_seed_for_key(item[0])
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(item)
            if len(deduped) >= max(6, self.followup_candidate_cap):
                break

        return deduped

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_options(self, working_memory, num_actions=None, question=None, context_profile=None, horizon=None):
        num_actions = num_actions if num_actions is not None else getattr(config, "SIMULATOR_DEFAULT_NUM_ACTIONS", 2)
        num_actions = max(1, int(num_actions))

        domain = self._raw_survival_domain(question, context_profile=context_profile)
        horizon_eff = self._effective_horizon(horizon)

        if question and self._is_feedbackish_text(question):
            question = None

        q_words = self._tokens(question) if question else set()
        mode = self._get_mode()

        question_guess = self._question_hazard_pin(question, domain)
        broad_guess = self._broad_raw_hazard_signal(question, context_profile=context_profile)

        pin_threshold = float(getattr(config, "SIMULATOR_HAZARD_PIN_THRESHOLD", 0.55))
        raw_pin = self._raw_hazard_pin(question, context_profile=context_profile)

        q_pin = None
        if domain == "survival":
            if question_guess.hazard in self.HAZARD_FAMILIES and question_guess.confidence >= pin_threshold:
                q_pin = question_guess.hazard
            elif broad_guess.hazard in self.HAZARD_FAMILIES and broad_guess.confidence >= pin_threshold:
                q_pin = broad_guess.hazard
            else:
                q_pin = raw_pin

        if domain == "survival" and q_pin in self.HAZARD_FAMILIES:
            if not isinstance(context_profile, dict):
                context_profile = {"domain": "survival", "tags": [q_pin], "hazard_family": q_pin}
            else:
                context_profile = dict(context_profile)
                context_profile["domain"] = "survival"
                context_profile["hazard_family"] = q_pin
                tags = context_profile.get("tags") or []
                if isinstance(tags, str):
                    tags = [tags]
                if not isinstance(tags, list):
                    tags = list(tags) if isinstance(tags, (set, tuple)) else []
                q_tag = str(q_pin).strip().lower()
                if q_tag and q_tag not in [str(x).strip().lower() for x in tags]:
                    tags.append(q_tag)
                context_profile["tags"] = tags

        rng = self._make_rng(
            question=question,
            domain=domain,
            mode=mode,
            hazard_hint=q_pin,
            context_profile=context_profile,
        )

        n_rollouts = self._effective_rollouts(domain)
        pinned_step1_fam = q_pin if domain == "survival" else None

        recent_msgs, pool_debug = self._candidate_memories(
            working_memory,
            limit=30,
            context_profile=context_profile,
            pinned_fam=pinned_step1_fam,
            q_words=q_words,
        )

        rescue_seeds: List[str] = []
        thin_min = int(getattr(config, "SIMULATOR_THIN_POOL_MIN", 3))
        if len(recent_msgs) < thin_min:
            rescue_seeds = self._thin_pool_rescue_seeds(question, domain, pinned_fam=pinned_step1_fam)

        options: List[dict] = []
        seen_keys: Set[str] = set()

        build_tries = max(8, num_actions * 8)
        i = 0
        rescue_i = 0

        while len(options) < num_actions and i < build_tries:
            i += 1

            use_rescue = bool(rescue_seeds) and (
                len(recent_msgs) == 0
                or (pinned_step1_fam in self.HAZARD_FAMILIES and rescue_i < len(rescue_seeds))
                or (domain in self.NON_SURVIVAL_HINTS and rescue_i < len(rescue_seeds) and i <= len(rescue_seeds))
                or (i % 3 == 0 and rescue_i < len(rescue_seeds))
            )

            if use_rescue:
                seed_text = rescue_seeds[rescue_i % len(rescue_seeds)]
                rescue_i += 1

                if domain == "survival":
                    rescue_hazard = pinned_step1_fam if pinned_step1_fam in self.HAZARD_FAMILIES else None
                    rescue_seed_ref = {
                        "domain": "survival",
                        "tags": [x for x in [rescue_hazard, "survival", "emergency"] if x],
                        "content": seed_text,
                        "kind": "rescue_seed",
                    }
                    rescue_is_explore = False
                    rescue_weight = 0.92
                else:
                    rescue_seed_ref = {
                        "domain": domain,
                        "tags": [domain, "corrective", "rescue"],
                        "content": seed_text,
                        "kind": "rescue_seed",
                    }
                    rescue_is_explore = False
                    rescue_weight = 0.88

                ctx, mem_weight, source_mem, seed_ref, is_explore = (
                    [],
                    rescue_weight,
                    None,
                    rescue_seed_ref,
                    rescue_is_explore,
                )
            else:
                seed_text, ctx, mem_weight, source_mem, seed_ref, is_explore = self._pick_seed(
                    recent_msgs,
                    q_words,
                    rng=rng,
                    question=question,
                    context_profile=context_profile,
                    pinned_fam=pinned_step1_fam,
                )

            opt = self._build_option(
                len(options),
                seed_text,
                ctx,
                mem_weight,
                source_mem,
                seed_ref,
                domain,
                is_explore,
                n_rollouts,
                rng=rng,
                question_guess=question_guess,
                pinned_fam=pinned_step1_fam,
                query_words=q_words,
                debug_counts=pool_debug,
            )

            if domain == "survival" and pinned_step1_fam in self.HAZARD_FAMILIES and not use_rescue:
                od = opt.get("details", {}) if isinstance(opt.get("details"), dict) else {}
                ohf = od.get("hazard_family")
                oconf = float(od.get("hazard_confidence", 0.0) or 0.0)

                if ohf in self.HAZARD_FAMILIES and ohf != pinned_step1_fam and oconf >= 0.55:
                    continue

                if ohf is None:
                    strength = self._survival_seed_strength(seed_text, seed_ref, pinned_step1_fam)
                    if strength < 2.5:
                        continue

            k = opt.get("details", {}).get("option_key") if isinstance(opt.get("details"), dict) else None
            if isinstance(k, str) and k:
                if k in seen_keys:
                    continue
                seen_keys.add(k)

            options.append(opt)

        if domain == "survival" and not options:
            fam = q_pin if q_pin in self.HAZARD_FAMILIES else "unknown"
            fallback = self.SURVIVAL_TEMPLATES.get(fam, self.SURVIVAL_TEMPLATES["unknown"])
            for j, seed in enumerate(fallback[:num_actions]):
                opt = self._build_option(
                    j,
                    seed,
                    [],
                    0.93,
                    None,
                    {
                        "domain": "survival",
                        "tags": [x for x in [fam, "survival", "emergency", "hard_fallback"] if x and x != "unknown"],
                        "content": seed,
                        "kind": "hard_fallback",
                    },
                    "survival",
                    False,
                    n_rollouts,
                    rng=rng,
                    question_guess=question_guess,
                    pinned_fam=(fam if fam in self.HAZARD_FAMILIES else None),
                    query_words=q_words,
                    debug_counts={**pool_debug, "hard_fallback_used": 1},
                )
                options.append(opt)

        if horizon_eff == 2 and options:
            k_followups = int(getattr(config, "SIMULATOR_FOLLOWUP_CANDIDATES", 2))
            k_followups = max(1, min(3, k_followups))

            lock_debug = bool(getattr(config, "SIMULATOR_LOCK_DEBUG", False))
            verbose = bool(getattr(config, "VERBOSE", False))
            ctx_fam = self._ctx_hazard_family(context_profile) if domain == "survival" else None
            lock_threshold = float(getattr(config, "SIMULATOR_HAZARD_LOCK_THRESHOLD", 0.65))

            for step1 in options:
                d1 = step1.get("details", {}) if isinstance(step1.get("details"), dict) else {}
                seed1 = str(d1.get("seed_text") or d1.get("seed_memory") or "")
                seed_ref1 = d1.get("seed_ref") if isinstance(d1.get("seed_ref"), dict) else None

                step1_guess = self._infer_hazard(seed1, seed_ref1)
                lock_fam = self._lock_from_signals(domain, question_guess, step1_guess, ctx_fam)

                if isinstance(d1, dict) and domain == "survival":
                    d1["lock_hazard_family"] = (lock_fam if lock_fam != "unknown" else None)
                    d1["lock_hazard_conf_threshold"] = round(lock_threshold, 3)

                follow_words = set(q_words) | self._tokens(seed1)
                follow_pool = self._build_followup_seed_pool(
                    recent_msgs=recent_msgs,
                    follow_words=follow_words,
                    lock_fam=(lock_fam if lock_fam in self.HAZARD_FAMILIES else None),
                    question=question,
                    step1_seed=seed1,
                    domain=domain,
                )

                if not follow_pool:
                    follow_pool = [(
                        self._make_exploration_seed(question or seed1, context_profile=context_profile),
                        [],
                        0.5,
                        None,
                        None,
                        True,
                    )]

                best2: Optional[dict] = None
                best_seq: Optional[float] = None
                best2_key: Optional[str] = None
                tried_followup_keys: Set[str] = set()

                tries = min(max(k_followups * 4, 8), max(10, len(follow_pool) * 2))
                tries = min(tries, max(12, len(follow_pool)))

                sampled_indices = list(range(len(follow_pool)))
                if not self.deterministic:
                    rng.shuffle(sampled_indices)
                sampled_indices = sampled_indices[:tries]

                for idx in sampled_indices:
                    seed2, ctx2, mw2, src2, seed_ref2, is_explore2 = follow_pool[idx]

                    if self._followup_is_too_similar(seed1, seed2):
                        continue

                    if domain == "survival" and lock_fam in self.HAZARD_FAMILIES:
                        if not self._followup_allowed_for_lock(seed2, seed_ref2, lock_fam, is_explore2):
                            if lock_debug and verbose:
                                step2_guess = self._infer_hazard(seed2, seed_ref2)
                                print(
                                    f"[Simulator] prefilter reject: step1={step1.get('name')} lock={lock_fam} "
                                    f"seed2='{self._safe_text(seed2, 80)}' step2_h={step2_guess.hazard} "
                                    f"conf={step2_guess.confidence:.2f}"
                                )
                            continue

                    merged_ctx = list(step1.get("context", []) or [])
                    merged_ctx.append(f"follow_up_from:{step1.get('name', 'unknown')}")
                    if ctx2:
                        merged_ctx.extend(ctx2)

                    step2 = self._build_option(
                        rng.randint(0, k_followups - 1),
                        seed2,
                        merged_ctx,
                        mw2,
                        src2,
                        seed_ref2,
                        domain,
                        is_explore2,
                        n_rollouts,
                        rng=rng,
                        question_guess=question_guess,
                        pinned_fam=(lock_fam if lock_fam in self.HAZARD_FAMILIES else None),
                        query_words=follow_words,
                        debug_counts=pool_debug,
                    )

                    k2 = step2.get("details", {}).get("option_key") if isinstance(step2.get("details"), dict) else None
                    if isinstance(k2, str) and k2:
                        if k2 in tried_followup_keys:
                            continue
                        tried_followup_keys.add(k2)

                    if isinstance(k2, str) and best2_key and k2 == best2_key:
                        continue

                    seq_score = self._score_sequence(step1, step2)
                    if best_seq is None or seq_score > best_seq:
                        best_seq = seq_score
                        best2 = step2
                        best2_key = k2 if isinstance(k2, str) else None

                planned = None
                if isinstance(best2, dict):
                    bd = best2.get("details", {}) if isinstance(best2.get("details"), dict) else {}
                    planned = {
                        "name": best2.get("name"),
                        "seed_text": bd.get("seed_text"),
                        "seed_memory": bd.get("seed_memory"),
                        "expected_utility": bd.get("expected_utility"),
                        "risk": best2.get("risk"),
                    }
                else:
                    fallback_seed = self._make_exploration_seed(question or seed1, context_profile=context_profile)
                    planned = {
                        "name": "followup_explore",
                        "seed_text": fallback_seed,
                        "seed_memory": fallback_seed,
                        "expected_utility": d1.get("expected_utility", None),
                        "risk": step1.get("risk", self.DEFAULT_RISK),
                    }
                    if best_seq is None:
                        best_seq = self._score_sequence(step1, None)

                if isinstance(step1.get("details"), dict):
                    step1["details"]["planned_next"] = planned
                    step1["details"]["sequence_score"] = best_seq
                    if best_seq is not None:
                        step1["weight"] = float(best_seq)

        options.sort(key=lambda o: float(o.get("weight", -9999.0)), reverse=True)
        self.past_actions.extend([opt.get("name", "unknown") for opt in options if isinstance(opt, dict)])

        if getattr(config, "VERBOSE", False):
            rs = getattr(self, "rollouts_scale", 1.0)
            hh = getattr(self, "horizon_hint", None)
            pin_str = f"{question_guess.hazard}:{question_guess.confidence:.2f}" if domain == "survival" else "n/a"
            broad_str = f"{broad_guess.hazard}:{broad_guess.confidence:.2f}" if domain == "survival" else "n/a"
            raw_pin_str = q_pin or "none"
            print(
                f"[Simulator] v{self.VERSION} Generated {len(options)} candidate action(s) "
                f"(horizon={horizon_eff}, domain={domain}, mode={mode}, rollouts={n_rollouts}, "
                f"rollouts_scale={rs}, horizon_hint={hh}, deterministic={self.deterministic}, "
                f"q_pin={pin_str}, broad_pin={broad_str}, raw_pin={raw_pin_str})."
            )

        return options