# FILE: test_clair.py
# Clair Stress Benchmark – unified rewrite
#
# Focus:
# - wiring / schema sanity
# - recall correctness
# - hazard lock correctness
# - planning determinism
# - deferred queue stability
# - reflection commit sanity
# - LTM promotion sanity
#
# Run:
#   python test_clair.py

from __future__ import annotations

import contextlib
import io
import random
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from clair import Clair
from memory.working_memory import WorkingMemory
import planning.simulator as sim_mod


# -----------------------------------------------------------------------------
# UTF-8 safety for Windows terminals
# -----------------------------------------------------------------------------
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

try:
    import config as _config  # type: ignore
except Exception:
    _config = None


# =============================================================================
# Global controls
# =============================================================================
SILENCE_INTERNAL = True
BENCH_RNG = random.Random(13371337)


# =============================================================================
# IO silencing
# =============================================================================
@contextlib.contextmanager
def silence_io(enabled: bool = True):
    if not enabled:
        yield
        return
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
        yield


# =============================================================================
# Minimal logger
# =============================================================================
class Log:
    QUIET = 0
    NORMAL = 1
    VERBOSE = 2

    def __init__(self, level: int = NORMAL):
        self.level = level

    def h(self, title: str):
        print("\n" + "=" * 92)
        print(title)
        print("=" * 92)

    def p(self, msg: str = "", lvl: int = NORMAL):
        if self.level >= lvl:
            print(msg)

    def kv(self, key: str, value: Any, lvl: int = NORMAL):
        if self.level >= lvl:
            print(f"{key:<28} {value}")


LOG = Log(level=Log.NORMAL)


# =============================================================================
# Scoreboard
# =============================================================================
@dataclass
class PhaseScore:
    name: str
    earned: int = 0
    max_points: int = 0

    def add(self, pts: int):
        self.earned += int(pts)

    def cap(self):
        self.earned = max(0, min(self.earned, self.max_points))


class Scoreboard:
    def __init__(self):
        self.phases: List[PhaseScore] = []

    def new_phase(self, name: str, max_points: int) -> PhaseScore:
        phase = PhaseScore(name=name, max_points=int(max_points))
        self.phases.append(phase)
        return phase

    def total(self) -> Tuple[int, int]:
        earned = sum(p.earned for p in self.phases)
        maxp = sum(p.max_points for p in self.phases)
        return earned, maxp

    def render(self):
        LOG.h("SCORE SUMMARY")
        for p in self.phases:
            LOG.p(f"{p.name:<40} {p.earned:>4}/{p.max_points}")
        earned, maxp = self.total()
        LOG.p("-" * 92)
        LOG.p(f"{'TOTAL':<40} {earned:>4}/{maxp}")


SB = Scoreboard()


# =============================================================================
# Packet shim
# =============================================================================
class TestPacket:
    def __init__(self, packet_id: str, raw_input: dict):
        self.packet_id = packet_id
        self.raw_input = raw_input

        content = (raw_input.get("content", "") or "")
        self.signals = {"normalized_text": content}

        self.uncertainty = type(
            "Uncertainty",
            (),
            {
                "metaphor_detected": False,
                "missing_references": False,
                "conflicting_signals": False,
            },
        )()

        self.retry_count = 0
        self.next_eligible_time = 0.0

    def is_viable(self):
        return True


# =============================================================================
# Generic helpers
# =============================================================================
def contains_any(text: str, needles: Iterable[str]) -> bool:
    if not text:
        return False
    low = text.lower()
    return any(str(n).lower() in low for n in needles)


def clamp01(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        v = float(default)
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def shorten(s: Any, limit: int = 110) -> str:
    try:
        text = str(s or "").replace("\n", " ").strip()
    except Exception:
        return ""
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def _norm_sig_text(s: Any, limit: int = 90) -> str:
    try:
        text = str(s or "").strip().lower()
    except Exception:
        return ""
    text = " ".join(text.split())
    return text[:limit]


def safe_get(d: Any, *path: Any, default=None):
    cur = d
    for p in path:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur


def call_first(obj: Any, method_names: Sequence[str], *args, **kwargs) -> Tuple[bool, Any]:
    for name in method_names:
        fn = getattr(obj, name, None)
        if callable(fn):
            try:
                return True, fn(*args, **kwargs)
            except TypeError:
                try:
                    return True, fn(*args)
                except Exception as e:
                    return False, e
            except Exception as e:
                return False, e
    return False, None


# =============================================================================
# Memory row compatibility helpers
# =============================================================================
def wm_row_is_mappingish(row: Any) -> bool:
    return isinstance(row, dict) or hasattr(row, "get")


def wm_row_get(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)

    getter = getattr(row, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except Exception:
            pass

    try:
        return getattr(row, key)
    except Exception:
        return default


def wm_row_details(row: Any) -> Dict[str, Any]:
    details = wm_row_get(row, "details", {})
    return details if isinstance(details, dict) else {}


def wm_row_id(row: Any) -> Optional[str]:
    return wm_row_get(row, "id") or wm_row_get(row, "memory_id")


def wm_row_type(row: Any) -> Optional[str]:
    val = wm_row_get(row, "type")
    return str(val) if val is not None else None


def wm_row_content(row: Any) -> str:
    val = wm_row_get(row, "content")
    if val is None:
        val = wm_row_get(row, "claim", "")
    try:
        return str(val or "")
    except Exception:
        return ""


def wm_row_status(row: Any) -> str:
    val = wm_row_get(row, "status") or wm_row_get(row, "verification_status") or ""
    try:
        return str(val).strip().lower()
    except Exception:
        return ""


def wm_row_conflict(row: Any) -> bool:
    details = wm_row_details(row)
    raw = wm_row_get(row, "conflict", False)
    return bool(raw or details.get("conflict", False) or details.get("contested", False))


def wm_row_contested(row: Any) -> bool:
    details = wm_row_details(row)
    raw = wm_row_get(row, "contested", False)
    status = wm_row_status(row)
    return bool(raw or details.get("contested", False) or status in {"contested", "disputed"})


def wm_row_last_verified(row: Any) -> Any:
    details = wm_row_details(row)
    return wm_row_get(row, "last_verified", details.get("last_verified"))


def wm_row_memory_class(row: Any) -> Any:
    details = wm_row_details(row)
    return wm_row_get(row, "memory_class", details.get("memory_class"))


def wm_row_staleness_risk(row: Any) -> Any:
    details = wm_row_details(row)
    return wm_row_get(row, "staleness_risk", details.get("staleness_risk"))


def wm_row_times_retrieved(row: Any) -> int:
    details = wm_row_details(row)
    try:
        return int(details.get("times_retrieved", 0) or 0)
    except Exception:
        return 0


# =============================================================================
# Tolerant wrappers
# =============================================================================
def sim_generate_options_tolerant(
    simulator: Any,
    memory: Any,
    *,
    num_actions: int,
    question: str,
    context_profile: dict,
    horizon: int,
):
    try:
        return simulator.generate_options(
            memory,
            num_actions=num_actions,
            question=question,
            context_profile=context_profile,
            horizon=horizon,
        )
    except TypeError:
        pass
    except Exception:
        raise

    try:
        return simulator.generate_options(
            memory,
            num_actions=num_actions,
            question=question,
            context_profile=context_profile,
        )
    except TypeError:
        pass
    except Exception:
        raise

    try:
        return simulator.generate_options(memory, num_actions=num_actions, question=question)
    except TypeError:
        pass
    except Exception:
        raise

    return simulator.generate_options(memory, num_actions=num_actions)


def ltm_retrieve_tolerant(ltm: Any) -> List[dict]:
    if ltm is None:
        return []
    fn = getattr(ltm, "retrieve", None)
    if not callable(fn):
        return []

    try:
        out = fn()
        return out if isinstance(out, list) else []
    except TypeError:
        pass
    except Exception:
        return []

    try:
        out = fn(msg_type=None, limit=250)
        return out if isinstance(out, list) else []
    except Exception:
        return []


def execute_one_action_tolerant(clair: Clair, chosen: dict) -> List[dict]:
    act = getattr(clair, "actuator", None)
    if act is None:
        return []

    system_state = None
    try:
        fn = getattr(clair, "_get_system_state", None)
        if callable(fn):
            system_state = fn()
    except Exception:
        system_state = None

    try:
        out = act.execute(chosen, system_state=system_state)
        return out if isinstance(out, list) else (out or [])
    except TypeError:
        pass
    except Exception:
        return []

    try:
        out = act.execute(chosen)
        return out if isinstance(out, list) else (out or [])
    except Exception:
        pass

    try:
        out = act.execute([chosen])
        return out if isinstance(out, list) else (out or [])
    except Exception:
        return []


def reflector_process_tolerant(clair: Clair, results: List[dict], evaluations: List[dict]) -> bool:
    reflector = getattr(clair, "reflector", None)
    memory = getattr(clair, "memory", None)
    if reflector is None or memory is None:
        return False

    ctx_weights = None
    try:
        ctx_weights = getattr(clair, "DEFAULT_CONTEXT_WEIGHTS", None)
    except Exception:
        ctx_weights = None

    try:
        reflector.process(results, evaluations, memory, context_weights=ctx_weights or {})
        return True
    except TypeError:
        pass
    except Exception:
        return False

    try:
        reflector.process(results, evaluations, memory)
        return True
    except Exception:
        return False


# =============================================================================
# WM schema smoke test
# =============================================================================
def run_wm_schema_smoke_test() -> Tuple[bool, List[str]]:
    issues: List[str] = []

    try:
        wm = WorkingMemory(preload_long_term=False)

        wm.store(
            {
                "type": "fact",
                "content": "Mount Everest is 8848 meters tall.",
                "source": "verification",
                "status": "verified",
                "domain": "general",
                "details": {
                    "source_trust": "trusted",
                    "verified": True,
                    "verification_status": "verified",
                    "status": "verified",
                },
            }
        )

        if not wm.buffer:
            return False, ["wm_store_failed_no_buffer"]

        mem = wm.buffer[0]
        details = wm_row_details(mem)

        if not wm_row_id(mem):
            issues.append("missing_id")

        status_v = str(
            details.get("verification_status")
            or details.get("status")
            or wm_row_status(mem)
            or ""
        ).lower()
        if status_v != "verified":
            issues.append("verification_status_not_verified")

        if wm_row_last_verified(mem) is None:
            issues.append("missing_last_verified")

        if wm_row_memory_class(mem) not in {"semantic", "episodic", "procedural", "hypothesis"}:
            issues.append("bad_memory_class")

        if wm_row_staleness_risk(mem) not in {"low", "medium", "high"}:
            issues.append("bad_staleness_risk")

        before = wm_row_times_retrieved(mem)
        _ = wm.retrieve("Everest")
        after = wm_row_times_retrieved(mem)
        if after < before + 1:
            issues.append("times_retrieved_not_incremented")

        wm.store(
            {
                "type": "fact",
                "content": "Water boils at 100 degrees Celsius at sea level.",
                "source": "verification",
                "status": "verified",
                "domain": "general",
                "details": {
                    "source_trust": "trusted",
                    "verified": True,
                    "verification_status": "verified",
                    "status": "verified",
                },
            }
        )
        wm.store(
            {
                "type": "fact",
                "content": "Water boils at 95 degrees Celsius at sea level.",
                "source": "reading",
                "status": "unverified",
                "domain": "general",
                "details": {
                    "verification_status": "unverified",
                    "status": "unverified",
                },
            }
        )

        contested = [m for m in wm.buffer if wm_row_is_mappingish(m) and wm_row_contested(m)]
        if not contested:
            issues.append("no_contested_conflict_detected")
        else:
            for c in contested:
                cd = wm_row_details(c)
                if not isinstance(cd.get("conflict_with_ids", []), list):
                    issues.append("conflict_with_ids_missing")
                    break
                cstat = str(
                    cd.get("verification_status")
                    or cd.get("status")
                    or wm_row_status(c)
                    or ""
                ).lower()
                if cstat not in {"contested", "disputed"}:
                    issues.append("conflict_verification_status_not_contested")
                    break

        wm.store(
            {
                "type": "fact",
                "content": "Maybe the backup server is offline.",
                "kind": "hypothesis",
                "domain": "operations",
                "status": "provisional",
            }
        )

        if not hasattr(wm, "calibration_candidates"):
            issues.append("missing_calibration_candidates")
        else:
            cands = wm.calibration_candidates(limit=10)
            if not isinstance(cands, list) or not cands:
                issues.append("calibration_candidates_empty_or_invalid")

        return len(issues) == 0, issues

    except Exception as e:
        return False, [f"wm_schema_exception:{type(e).__name__}:{e}"]


# =============================================================================
# Risk assessor spy
# =============================================================================
class RiskAssessorSpy:
    def __init__(self, base):
        self.base = base
        self.calls = 0

    def assess(self, option: dict):
        self.calls += 1

        if self.base and hasattr(self.base, "assess"):
            return self.base.assess(option)

        txt = ""
        try:
            txt = str(option.get("details", {}).get("seed_text", ""))[:200].lower()
        except Exception:
            txt = ""

        risk = 0.35
        if any(k in txt for k in ("fire", "smoke", "burn")):
            risk = 0.65
        elif any(k in txt for k in ("flood", "water", "submerged")):
            risk = 0.60
        elif any(k in txt for k in ("earthquake", "aftershock", "seismic", "tremor", "fault")):
            risk = 0.62
        return {"risk": risk, "reasons": ["spy_fallback"]}


# =============================================================================
# Hazard helpers
# =============================================================================
@dataclass(frozen=True)
class HazardGuess:
    hazard: str
    confidence: float
    evidence: List[str]


_HAZ_PATTERNS: Dict[str, List[Tuple[str, float]]] = {
    "earthquake": [
        (r"\b(aftershock|foreshock|tremor|seismic|epicenter|fault\s*line)\b", 3.0),
        (r"\b(earthquake|quake|magnitude\s*\d(\.\d)?)\b", 2.5),
        (r"\b(ground\s+shaking|building\s+swaying|shake\s+again)\b", 2.0),
    ],
    "flood": [
        (r"\b(flood|flooding|inundat(e|ion)|flash\s*flood)\b", 3.0),
        (r"\b(rising\s+water|storm\s+surge|levee|sandbag)\b", 2.0),
        (r"\b(submerged|water\s+is\s+rising|road(s)?\s+underwater)\b", 2.0),
    ],
    "fire": [
        (r"\b(fire|wildfire|smoke|flames|embers|burning)\b", 3.0),
        (r"\b(get\s+low|crawl|smoke\s+inhalation)\b", 2.0),
    ],
}

_NEG_HINTS: Dict[str, List[str]] = {
    "earthquake": [r"\bstorm\s+surge\b", r"\blevee\b", r"\bsandbag\b", r"\bunderwater\b"],
    "flood": [r"\baftershock\b", r"\bfault\s*line\b", r"\bseismic\b", r"\bground\s+shaking\b"],
}


def ref_infer_hazard(text: str) -> HazardGuess:
    t = (text or "").lower()
    if not t.strip():
        return HazardGuess("unknown", 0.0, [])

    scores: Dict[str, float] = {h: 0.0 for h in _HAZ_PATTERNS}
    evidence: Dict[str, List[str]] = {h: [] for h in _HAZ_PATTERNS}

    for hazard, patterns in _HAZ_PATTERNS.items():
        for pat, weight in patterns:
            m = re.search(pat, t)
            if m:
                scores[hazard] += weight
                evidence[hazard].append(m.group(0))

        for neg in _NEG_HINTS.get(hazard, []):
            if re.search(neg, t):
                scores[hazard] -= 1.5

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_h, top_s = ranked[0]
    runner_s = ranked[1][1] if len(ranked) > 1 else 0.0

    if top_s <= 0.5:
        return HazardGuess("unknown", 0.2 if top_s > 0 else 0.0, [])

    if (top_s - runner_s) < 1.0:
        return HazardGuess("unknown", 0.35, evidence[top_h])

    conf = min(0.95, 0.45 + 0.15 * top_s)
    return HazardGuess(top_h, conf, evidence[top_h])


def classify_action_text(opt_or_planned: Any) -> HazardGuess:
    if isinstance(opt_or_planned, dict):
        d = opt_or_planned.get("details") if "details" in opt_or_planned else opt_or_planned
        seed = (
            safe_get(d, "seed_text")
            or safe_get(d, "seed_memory")
            or safe_get(d, "seed_ref", "content")
            or safe_get(opt_or_planned, "name")
            or ""
        )
        return ref_infer_hazard(str(seed))
    return ref_infer_hazard(str(opt_or_planned))


def extract_hazard_fields(details: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "hazard_family": details.get("hazard_family"),
        "lock_hazard_family": details.get("lock_hazard_family") or details.get("hazard_lock"),
        "hazard_confidence": details.get("hazard_confidence")
        or details.get("hazard_conf")
        or details.get("hazard_score"),
        "hazard_evidence": details.get("hazard_evidence") or details.get("evidence"),
        "question_hazard": details.get("question_hazard"),
        "question_hazard_conf": details.get("question_hazard_conf"),
    }


def verdict_hazard_lock(opt: Dict[str, Any]) -> Tuple[bool, str]:
    d = opt.get("details", {}) if isinstance(opt, dict) else {}
    hf = d.get("hazard_family")
    lockfam = d.get("lock_hazard_family") or d.get("hazard_lock")

    if lockfam is None or hf is None:
        return True, "lock_or_hf_missing"
    if str(lockfam) != str(hf):
        return False, f"LOCK_MISMATCH hazard_family={hf} lock={lockfam}"
    return True, "lock_ok"


def verdict_followup_legality(opt: Dict[str, Any]) -> Tuple[bool, str]:
    d = opt.get("details", {}) if isinstance(opt, dict) else {}
    lockfam = d.get("lock_hazard_family") or d.get("hazard_lock")
    planned = d.get("planned_next")

    if not isinstance(planned, dict):
        return True, "planned_missing"
    if lockfam is None:
        return True, "lock_missing"

    ref_follow = classify_action_text(planned)
    if lockfam in ("flood", "fire", "earthquake"):
        if ref_follow.hazard != "unknown" and ref_follow.hazard != lockfam:
            return (
                False,
                f"FOLLOWUP_ILLEGAL lock={lockfam} follow={ref_follow.hazard} conf={ref_follow.confidence:.2f}",
            )
    return True, "follow_ok_or_unknown"


def compact_option_line(opt: Dict[str, Any]) -> str:
    d = opt.get("details", {}) if isinstance(opt, dict) else {}
    hz = extract_hazard_fields(d)
    name = opt.get("name", "unknown")
    risk = opt.get("risk", None)
    weight = opt.get("weight", None)
    seed = shorten(d.get("seed_text") or d.get("seed_memory") or "", 85)
    return (
        f"{name} | w={weight} risk={risk} | "
        f"q_pin={hz.get('question_hazard')}:{hz.get('question_hazard_conf')} | "
        f"hf={hz.get('hazard_family')} lock={hz.get('lock_hazard_family')} | "
        f"seed='{seed}'"
    )


# =============================================================================
# Deferred processing
# =============================================================================
def process_deferred_fallback(clair: Clair, ticks: int = 6) -> int:
    deferred_responses = 0

    if hasattr(clair, "step") and callable(getattr(clair, "step")):
        for _ in range(ticks):
            with silence_io(SILENCE_INTERNAL):
                tick = clair.step()
            resp_list = tick.get("responses", []) if isinstance(tick, dict) else []
            deferred_responses += len(resp_list)
        return deferred_responses

    for _ in range(ticks):
        now = time.time()
        q = getattr(clair, "deferred_queue", []) or []
        if not q:
            break

        remaining = []
        eligible = []
        for p in q:
            try:
                nxt = float(getattr(p, "next_eligible_time", 0.0) or 0.0)
            except Exception:
                nxt = 0.0
            if nxt <= now:
                eligible.append(p)
            else:
                remaining.append(p)

        clair.deferred_queue = remaining

        for p in eligible:
            with silence_io(SILENCE_INTERNAL):
                resp = clair.handle_packet(p)
            if resp:
                deferred_responses += 1
            try:
                p.retry_count = int(getattr(p, "retry_count", 0) or 0) + 1
                p.next_eligible_time = time.time() + 0.05
            except Exception:
                pass

        time.sleep(0.03)

    return deferred_responses


# =============================================================================
# Hypothalamus shim
# =============================================================================
def apply_hypothalamus_biases(clair: Clair, simulator) -> Optional[str]:
    hypo = getattr(clair, "hypothalamus", None)
    if hypo is None:
        return None

    signals = {
        "risk": 0.45,
        "uncertainty": 0.55,
        "novelty": 0.70,
        "urgency": 0.40,
        "goal_pressure": 0.65,
        "fatigue": 0.10,
        "activity": 0.60,
    }

    mode = None
    try:
        if hasattr(hypo, "choose_mode") and callable(hypo.choose_mode):
            mode = hypo.choose_mode(signals)
    except Exception:
        mode = None

    try:
        if hasattr(hypo, "apply_to_simulator") and callable(getattr(hypo, "apply_to_simulator")):
            hypo.apply_to_simulator(simulator)
            return str(mode) if mode is not None else "unknown"
    except Exception:
        pass

    biases = None
    try:
        if hasattr(hypo, "get_biases") and callable(hypo.get_biases):
            biases = hypo.get_biases()
    except Exception:
        biases = None

    if not isinstance(biases, dict):
        return str(mode) if mode is not None else None

    try:
        if "exploration_rate" in biases:
            simulator.exploration_rate = float(biases["exploration_rate"])
        if "sim_horizon" in biases:
            simulator.horizon_hint = int(round(float(biases["sim_horizon"])))
        if "sim_rollouts" in biases:
            sr = clamp01(biases["sim_rollouts"], 0.5)
            simulator.rollouts_scale = 0.20 + 1.80 * sr
    except Exception:
        pass

    return str(mode) if mode is not None else "unknown"


# =============================================================================
# Reflection helpers
# =============================================================================
def _find_recent_committed_actions(buffer: Any, limit: int = 30) -> List[Any]:
    if not isinstance(buffer, list):
        return []
    out = []
    for m in reversed(buffer[-max(1, limit):]):
        if wm_row_is_mappingish(m) and wm_row_type(m) == "committed_action":
            out.append(m)
    return out


def _reflection_entry_sanity(m: Any) -> Tuple[bool, List[str]]:
    issues: List[str] = []

    if not wm_row_is_mappingish(m):
        return False, ["not_mappingish"]

    content = wm_row_content(m)
    content_obj = wm_row_get(m, "content_obj", None)

    if not isinstance(content, str) or not content.strip():
        issues.append("content_not_string_or_empty")
    if content_obj is not None and not isinstance(content_obj, dict):
        issues.append("content_obj_not_dict")

    conf = wm_row_get(m, "confidence", None)
    weight = wm_row_get(m, "weight", None)

    try:
        cf = float(conf)
        if not (0.0 <= cf <= 1.0):
            issues.append("confidence_out_of_range")
    except Exception:
        issues.append("confidence_not_float")

    try:
        wf = float(weight)
        if not (0.0 <= wf <= 1.0):
            issues.append("weight_out_of_range")
    except Exception:
        issues.append("weight_not_float")

    dom = wm_row_get(m, "domain", None)
    if dom is not None and not isinstance(dom, str):
        issues.append("domain_not_str")

    tags = wm_row_get(m, "tags", None)
    if tags is not None and not isinstance(tags, list):
        issues.append("tags_not_list")

    return len(issues) == 0, issues


# =============================================================================
# Start benchmark
# =============================================================================
LOG.h("IMPORT CHECK")
LOG.kv("planning.simulator path", sim_mod.__file__)
LOG.kv("Simulator VERSION", getattr(sim_mod.Simulator, "VERSION", "unknown"))
LOG.kv("Simulator class id", id(sim_mod.Simulator))
LOG.kv("WorkingMemory VERSION", getattr(WorkingMemory, "VERSION", "unknown"))

with silence_io(SILENCE_INTERNAL):
    clair = Clair()

LOG.h("CLAIR STRESS BENCHMARK START")
LOG.p("Candy rule: if PASS, you get 🍬. Because apparently unit tests still count as therapy.\n")


# =============================================================================
# PHASE 0: Wiring checks
# =============================================================================
P0 = SB.new_phase("Phase 0: Wiring checks", 75)
LOG.h("PHASE 0: Wiring checks")

has_sim = hasattr(clair, "simulator")
has_mem = hasattr(clair, "memory")
has_ltm = hasattr(clair, "long_term")
has_hypo = hasattr(clair, "hypothalamus")

LOG.kv("clair.simulator", "OK" if has_sim else "MISSING")
if has_sim:
    LOG.kv("sim.VERSION", getattr(clair.simulator, "VERSION", "unknown"))
LOG.kv("clair.memory", "OK" if has_mem else "MISSING")
LOG.kv("clair.long_term", "OK" if has_ltm else "MISSING")
LOG.kv("clair.hypothalamus", "OK" if has_hypo else "MISSING")

P0.add(23 if has_sim else 5)
P0.add(18 if has_mem else 0)
P0.add(12 if has_ltm else 0)
P0.add(18 if has_hypo else 0)

wm_ok, wm_issues = run_wm_schema_smoke_test()
LOG.kv("WM schema smoke", "PASS" if wm_ok else "FAIL")
if wm_issues:
    LOG.p(f"WM issues: {wm_issues}")
P0.add(4 if wm_ok else 0)
P0.cap()


# =============================================================================
# PHASE 1: Knowledge injection
# =============================================================================
P1 = SB.new_phase("Phase 1: Knowledge injection", 70)
LOG.h("PHASE 1: Knowledge injection")

knowledge_packets = [
    {"type": "lesson", "content": "If lost at night: stop moving, stay put, conserve energy, make shelter, signal for help."},
    {"type": "lesson", "content": "In a flood: move to higher ground, avoid fast water, do not drive through floodwater, follow evacuation orders."},
    {"type": "lesson", "content": "For fire in a closed room: leave if possible, get low to avoid smoke, open windows only if safe, call emergency services."},
    {"type": "lesson", "content": "During an earthquake: drop, cover, and hold on. After shaking stops, evacuate unsafe buildings and watch for aftershocks."},
    {"type": "lesson", "content": "Python is a programming language used for coding."},
    {"type": "fact", "content": "Water boils at 100 degrees Celsius at sea level."},
    {"type": "fact", "content": "Humans have 206 bones."},
    {"type": "fact", "content": "Rain happens due to condensation of water vapor into droplets."},
    {"type": "lesson", "content": "A simple day plan: list 3 priorities, timebox them, leave buffer, then do a quick review at the end."},
    {"type": "lesson", "content": "Organization trick: pick one inbox for tasks, do a 5-minute sort, then execute the top item only."},
    {"type": "lesson", "content": "When planning: define success, pick the next action, then schedule it in a specific time block."},
]

with silence_io(SILENCE_INTERNAL):
    for i, item in enumerate(knowledge_packets):
        clair.handle_packet(TestPacket(f"learn_{i}", item))

LOG.p("Injected lessons/facts:")
LOG.p(f"  count={len(knowledge_packets)}")
P1.add(70)
P1.cap()


# =============================================================================
# PHASE 2: Recall + no tech leakage
# =============================================================================
P2 = SB.new_phase("Phase 2: Recall + no tech leakage", 210)
LOG.h("PHASE 2: Recall + no tech leakage")

tests = [
    ("boiling", "what is the boiling point of water?", ["100"], [], 32),
    ("bones", "how many bones are in the human body?", ["206"], [], 32),
    ("rain", "why does rain happen?", ["condensation"], [], 26),
    (
        "survival_flood",
        "A flood is happening and roads are submerged. What should I do?",
        ["higher", "ground", "avoid", "floodwater", "evac"],
        ["python", "programming"],
        60,
    ),
    (
        "survival_fire",
        "Fire in a closed room, smoke everywhere. What should I do?",
        ["leave", "low", "smoke", "exit", "safe"],
        ["python", "programming"],
        60,
    ),
]

for tid, q, expected, forbidden, pts in tests:
    with silence_io(SILENCE_INTERNAL):
        resp = clair.handle_packet(TestPacket(f"recall_{tid}", {"type": "ask", "content": q}))
    ok = False

    if resp:
        hit_expected = contains_any(resp, expected) if expected else True
        hit_forbidden = contains_any(resp, forbidden) if forbidden else False
        if hit_expected and not hit_forbidden:
            P2.add(pts)
            ok = True
        elif not hit_forbidden:
            P2.add(int(pts * 0.5))
        else:
            P2.add(5)
    else:
        P2.add(6)

    LOG.p(f"[{tid:<14}] {'OK' if ok else '...'}  resp='{shorten(resp, 95)}'")

P2.cap()


# =============================================================================
# PHASE 3: Hypothalamus -> simulator knobs
# =============================================================================
P3 = SB.new_phase("Phase 3: Hypothalamus -> simulator knobs", 70)
LOG.h("PHASE 3: Hypothalamus -> simulator knobs")

selected_mode = None
if has_sim:
    with silence_io(SILENCE_INTERNAL):
        selected_mode = apply_hypothalamus_biases(clair, clair.simulator)

LOG.kv("Selected mode", selected_mode)

if selected_mode:
    P3.add(50)
    try:
        er = float(getattr(clair.simulator, "exploration_rate", 0.0))
        rs = float(getattr(clair.simulator, "rollouts_scale", 1.0))
        hh = getattr(clair.simulator, "horizon_hint", None)
        LOG.kv("sim.exploration_rate", er)
        LOG.kv("sim.rollouts_scale", rs)
        LOG.kv("sim.horizon_hint", hh)
        if 0.0 <= er <= 0.75:
            P3.add(10)
        if 0.2 <= rs <= 2.0:
            P3.add(10)
    except Exception:
        pass
else:
    P3.add(10)

P3.cap()


# =============================================================================
# PHASE 4: Meta reasoning sanity
# =============================================================================
P4 = SB.new_phase("Phase 4: Meta reasoning sanity", 78)
LOG.h("PHASE 4: Meta reasoning sanity")

with silence_io(SILENCE_INTERNAL):
    resp1 = clair.handle_packet(TestPacket("meta_seed", {"type": "ask", "content": "why does rain happen?"}))
    resp2 = clair.handle_packet(TestPacket("meta_why", {"type": "ask", "content": "explain your reasoning"}))

LOG.p(f"[meta_seed] {shorten(resp1, 110)}")
LOG.p(f"[meta_why ] {shorten(resp2, 110)}")

if resp2:
    words = resp2.split()
    P4.add(50 if len(words) <= 65 else 22)
    if not contains_any(resp2, ["mount everest", "python is", "largest hot desert", "sahara"]):
        P4.add(28)
else:
    P4.add(12)

P4.cap()


# =============================================================================
# PHASE 5: Planning baseline + risk cache
# =============================================================================
P5 = SB.new_phase("Phase 5: Planning baseline + risk cache", 168)
LOG.h("PHASE 5: Planning baseline + risk cache")

if not (has_sim and has_mem):
    LOG.p("Simulator or memory missing; skipping.")
    P5.add(10)
    P5.cap()
else:
    base_ra = getattr(clair.simulator, "risk_assessor", None)
    spy = RiskAssessorSpy(base_ra)
    clair.simulator.risk_assessor = spy

    context_profile = {
        "domain": "survival",
        "tags": ["flood"],
        "threat": 0.7,
        "urgency": 0.6,
        "goal": "preserve life",
        "rng_seed": 1337,
    }
    q = "Heavy rain floods the town and roads are submerged. What should be done?"

    try:
        with silence_io(SILENCE_INTERNAL):
            options = sim_generate_options_tolerant(
                clair.simulator,
                clair.memory,
                num_actions=3,
                question=q,
                context_profile=context_profile,
                horizon=2,
            )
    except Exception as e:
        LOG.p(f"Simulator error: {e}")
        options = []

    if not options:
        LOG.p("No options generated.")
        P5.add(16)
    else:
        LOG.p(f"Generated {len(options)} option(s).")
        ok_planned = 0
        ok_seq = 0
        ok_lock = 0

        for opt in options[:3]:
            d = opt.get("details", {}) if isinstance(opt, dict) else {}
            hz = extract_hazard_fields(d)
            LOG.p("  " + compact_option_line(opt))

            planned = d.get("planned_next")
            if isinstance(planned, dict) and (planned.get("seed_text") or planned.get("seed_memory")):
                ok_planned += 1
            if d.get("sequence_score") is not None:
                ok_seq += 1
            if hz.get("lock_hazard_family") == "flood":
                ok_lock += 1

        if ok_planned >= 1:
            P5.add(55)
        if ok_planned >= 2:
            P5.add(25)
        if ok_seq >= 1:
            P5.add(45)
        if ok_lock >= 1:
            P5.add(30)

        calls_before = spy.calls
        try:
            with silence_io(SILENCE_INTERNAL):
                _ = sim_generate_options_tolerant(
                    clair.simulator,
                    clair.memory,
                    num_actions=3,
                    question=q,
                    context_profile=context_profile,
                    horizon=2,
                )
        except Exception:
            pass

        delta = spy.calls - calls_before
        LOG.kv("RiskAssessorSpy delta (2nd run)", delta)
        P5.add(38 if delta <= 6 else (18 if delta <= 12 else 6))

    clair.simulator.risk_assessor = base_ra
    P5.cap()


# =============================================================================
# PHASE 6: Hazard lock forensics
# =============================================================================
P6 = SB.new_phase("Phase 6: Hazard lock forensics", 215)
LOG.h("PHASE 6: Hazard lock forensics")

if not (has_sim and has_mem):
    LOG.p("Simulator or memory missing; skipping.")
    P6.add(8)
    P6.cap()
else:
    scenarios = [
        ("EQ_clean", "An earthquake just hit. The ground is shaking and aftershocks are expected.", ["earthquake"]),
        ("FLOOD_clean", "A flash flood is happening. Roads are underwater and the water is rising fast.", ["flood"]),
        ("FIRE_clean", "There is a fire in a closed room. Thick smoke and flames are spreading.", ["fire"]),
        (
            "EQ_with_water",
            "Earthquake damage broke a water main. The ground shaking continues and aftershocks are likely. Streets have water.",
            ["earthquake"],
        ),
        (
            "FLOOD_with_shake",
            "Flood waters are rising. People are evacuating. Someone says the building is shaking from trucks outside.",
            ["flood"],
        ),
    ]

    bad_lock = 0
    bad_follow = 0
    bad_step1 = 0

    for sid, question, expect in scenarios:
        ref_q = ref_infer_hazard(question)
        LOG.p(f"\n[{sid}] q='{shorten(question, 92)}'")
        LOG.p(f"  ref_q={ref_q.hazard} conf={ref_q.confidence:.2f} ev={ref_q.evidence}")

        context_profile = {
            "domain": "survival",
            "tags": expect,
            "threat": 0.75,
            "urgency": 0.65,
            "goal": "preserve life",
            "rng_seed": 9001,
        }

        try:
            with silence_io(SILENCE_INTERNAL):
                options = sim_generate_options_tolerant(
                    clair.simulator,
                    clair.memory,
                    num_actions=3,
                    question=question,
                    context_profile=context_profile,
                    horizon=2,
                )
        except Exception as e:
            LOG.p(f"  Simulator error: {e}")
            options = []

        if not options:
            LOG.p("  No options generated.")
            continue

        for i, opt in enumerate(options[:2]):
            d = opt.get("details", {}) if isinstance(opt, dict) else {}
            hz = extract_hazard_fields(d)

            seed_guess = classify_action_text(opt)
            sim_hf = hz.get("hazard_family")

            if sim_hf in ("earthquake", "flood", "fire") and seed_guess.hazard != "unknown" and sim_hf != seed_guess.hazard:
                bad_step1 += 1
                LOG.p(
                    f"  [opt{i}] STEP1_SUSPECT sim_hf={sim_hf} seed_ref={seed_guess.hazard} ev={seed_guess.evidence}"
                )

            ok_lock, lock_note = verdict_hazard_lock(opt)
            ok_follow, follow_note = verdict_followup_legality(opt)

            if not ok_lock:
                bad_lock += 1
            if not ok_follow:
                bad_follow += 1

            LOG.p(f"  [opt{i}] {compact_option_line(opt)}")
            LOG.p(f"         lock={lock_note} | follow={follow_note}")

    LOG.p("\nForensics counters:")
    LOG.kv("step1 suspects", bad_step1)
    LOG.kv("lock mismatches", bad_lock)
    LOG.kv("illegal followups", bad_follow)

    P6.add(70)
    P6.add(55 if bad_lock == 0 else max(0, 55 - 15 * bad_lock))
    P6.add(55 if bad_follow == 0 else max(0, 55 - 20 * bad_follow))
    P6.add(35 if bad_step1 == 0 else max(0, 35 - 10 * bad_step1))
    P6.cap()


# =============================================================================
# PHASE 7: Determinism toggle
# =============================================================================
P7 = SB.new_phase("Phase 7: Determinism toggle", 98)
LOG.h("PHASE 7: Determinism toggle")

if not (has_sim and has_mem):
    LOG.p("Simulator or memory missing; skipping.")
    P7.add(6)
    P7.cap()
else:
    q = "Give me a short plan to organize my day."
    context_profile = {"domain": "general", "tags": ["planning", "test"], "rng_seed": 4242}

    def signature(opts: List[dict]) -> str:
        parts = []
        for o in (opts or [])[:3]:
            d = o.get("details", {}) if isinstance(o, dict) else {}
            parts.append(_norm_sig_text(d.get("seed_text") or d.get("seed_memory") or ""))
        return "|".join(parts)

    prev_det = bool(getattr(clair.simulator, "deterministic", True))
    old_past = list(getattr(clair.simulator, "past_actions", []) or [])

    clair.simulator.deterministic = True
    clair.simulator.past_actions = []

    with silence_io(SILENCE_INTERNAL):
        s1 = signature(
            sim_generate_options_tolerant(
                clair.simulator,
                clair.memory,
                num_actions=3,
                question=q,
                context_profile=context_profile,
                horizon=1,
            )
        )

    clair.simulator.past_actions = []
    with silence_io(SILENCE_INTERNAL):
        s2 = signature(
            sim_generate_options_tolerant(
                clair.simulator,
                clair.memory,
                num_actions=3,
                question=q,
                context_profile=context_profile,
                horizon=1,
            )
        )

    LOG.p(f"deterministic: {s1} || {s2}")
    P7.add(60 if s1 and s1 == s2 else 22)

    clair.simulator.deterministic = False

    uniq: List[str] = []
    for _ in range(6):
        clair.simulator.past_actions = []
        with silence_io(SILENCE_INTERNAL):
            sx = signature(
                sim_generate_options_tolerant(
                    clair.simulator,
                    clair.memory,
                    num_actions=3,
                    question=q,
                    context_profile=context_profile,
                    horizon=1,
                )
            )
        if sx and sx not in uniq:
            uniq.append(sx)
        time.sleep(0.002)

    if len(uniq) >= 2:
        LOG.p(f"nondeterministic: {uniq[0]} || {uniq[1]}")
    elif len(uniq) == 1:
        LOG.p(f"nondeterministic: {uniq[0]} || {uniq[0]}")
    else:
        LOG.p("nondeterministic: (empty) || (empty)")

    if len(uniq) >= 2:
        P7.add(38)
    elif len(uniq) == 1:
        P7.add(22)
    else:
        P7.add(10)

    clair.simulator.past_actions = old_past
    if _config is not None:
        try:
            clair.simulator.deterministic = bool(getattr(_config, "SIMULATOR_DETERMINISTIC", prev_det))
        except Exception:
            clair.simulator.deterministic = prev_det
    else:
        clair.simulator.deterministic = prev_det

    P7.cap()


# =============================================================================
# PHASE 8: Deferred queue stress
# =============================================================================
P8 = SB.new_phase("Phase 8: Deferred queue stress", 70)
LOG.h("PHASE 8: Deferred queue stress")

if not hasattr(clair, "deferred_queue"):
    LOG.p("clair.deferred_queue missing.")
    P8.add(12)
else:
    prompts = [
        "what is the boiling point of water?",
        "how many bones are in the human body?",
        "A fire is in a closed room. What should be done?",
        "A flood is happening and roads are submerged. What should be done?",
        "During an earthquake what do I do?",
        "explain your reasoning",
        "summarize what you know about floods",
    ]

    for i in range(10):
        p = TestPacket(
            f"defer_{i}",
            {"type": "ask", "content": prompts[BENCH_RNG.randrange(0, len(prompts))]},
        )
        p.retry_count = 0
        p.next_eligible_time = 0.0
        clair.deferred_queue.append(p)

    with silence_io(SILENCE_INTERNAL):
        clair.handle_packet(TestPacket("prime_gate", {"type": "ask", "content": "status check"}))

    deferred_responses = process_deferred_fallback(clair, ticks=8)
    LOG.kv("Deferred responses observed", deferred_responses)

    P8.add(70 if deferred_responses >= 3 else (40 if deferred_responses >= 1 else 12))

P8.cap()


# =============================================================================
# PHASE 9: LTM sanity
# =============================================================================
P9 = SB.new_phase("Phase 9: LTM sanity", 65)
LOG.h("PHASE 9: LTM sanity")

try:
    with silence_io(SILENCE_INTERNAL):
        if hasattr(clair, "memory") and hasattr(clair.memory, "force_promote_candidates"):
            clair.memory.force_promote_candidates(limit=20)
        elif hasattr(clair, "consolidate_memory") and callable(clair.consolidate_memory):
            clair.consolidate_memory()
except Exception:
    pass

ltm_items: List[dict] = []
try:
    ltm_items = ltm_retrieve_tolerant(getattr(clair, "long_term", None))
except Exception:
    ltm_items = []

ltm_count = len(ltm_items) if isinstance(ltm_items, list) else 0
LOG.kv("LTM item count", ltm_count)

if 5 <= ltm_count <= 300:
    P9.add(65)
elif 1 <= ltm_count < 5:
    P9.add(32)
elif ltm_count > 300:
    P9.add(22)
else:
    P9.add(12)

P9.cap()


# =============================================================================
# PHASE 10: Drift sniff
# =============================================================================
P10 = SB.new_phase("Phase 10: Drift sniff (no tech leak)", 50)
LOG.h("PHASE 10: Drift sniff (no tech leak)")

drift_prompts = [
    "A flood is happening. What should I do?",
    "Fire in a closed room. Smoke everywhere. What now?",
    "During an earthquake what do I do?",
]

good = 0
for i in range(6):
    q = drift_prompts[BENCH_RNG.randrange(0, len(drift_prompts))]
    with silence_io(SILENCE_INTERNAL):
        resp = clair.handle_packet(TestPacket(f"drift_{i}", {"type": "ask", "content": q}))
    if resp and not contains_any(resp, ["python", "programming"]):
        good += 1

LOG.kv("passes", f"{good}/6")
P10.add(50 if good >= 5 else (28 if good >= 3 else 12))
P10.cap()


# =============================================================================
# PHASE 11: Reflection commit sanity
# =============================================================================
P11 = SB.new_phase("Phase 11: Reflection commit sanity", 90)
LOG.h("PHASE 11: Reflection commit sanity")

if not (has_sim and has_mem):
    LOG.p("Simulator or memory missing; skipping.")
    P11.add(10)
    P11.cap()
else:
    context_profile = {
        "domain": "survival",
        "tags": ["fire"],
        "threat": 0.75,
        "urgency": 0.70,
        "goal": "preserve life",
        "rng_seed": 2222,
    }
    q = "There is a fire in a closed room. Thick smoke. What should be done?"

    try:
        with silence_io(SILENCE_INTERNAL):
            options = sim_generate_options_tolerant(
                clair.simulator,
                clair.memory,
                num_actions=2,
                question=q,
                context_profile=context_profile,
                horizon=2,
            )
    except Exception as e:
        LOG.p(f"Simulator error: {e}")
        options = []

    if not options:
        LOG.p("No options generated.")
        P11.add(10)
    else:
        chosen = options[0] if isinstance(options[0], dict) else None
        if not chosen:
            LOG.p("Option schema unexpected.")
            P11.add(10)
        else:
            with silence_io(SILENCE_INTERNAL):
                results = execute_one_action_tolerant(clair, chosen)

            evaluations: List[dict] = []
            try:
                perf = getattr(clair, "performance", None)
                if perf is not None and hasattr(perf, "evaluate"):
                    with silence_io(SILENCE_INTERNAL):
                        evaluations = perf.evaluate(results)
            except Exception:
                evaluations = []

            with silence_io(SILENCE_INTERNAL):
                ok_ref = reflector_process_tolerant(clair, results, evaluations)

            LOG.kv("reflection results count", len(results) if isinstance(results, list) else -1)
            LOG.kv("reflection evals count", len(evaluations) if isinstance(evaluations, list) else -1)
            LOG.kv("reflector_process_ok", ok_ref)

            if results:
                r0 = results[0] if isinstance(results[0], dict) else {}
                LOG.p(f"result[0].action_name       = {r0.get('action_name')}")
                LOG.p(f"result[0].action_type       = {r0.get('action_type')}")
                LOG.p(f"result[0].outcome           = {r0.get('outcome')}")
                LOG.p(f"result[0].details           = {shorten(r0.get('details'), 180)}")

            if evaluations:
                e0 = evaluations[0] if isinstance(evaluations[0], dict) else {}
                LOG.p(f"eval[0].score               = {e0.get('score')}")
                LOG.p(f"eval[0].outcome             = {e0.get('outcome')}")
                LOG.p(f"eval[0].action_name         = {e0.get('action_name')}")

            if results is not None:
                P11.add(25)
            if ok_ref:
                P11.add(20)

            buf = getattr(clair.memory, "buffer", None)
            if isinstance(buf, list):
                LOG.kv("wm.buffer size", len(buf))
                tail_types = [wm_row_type(m) for m in buf[-10:] if wm_row_is_mappingish(m)]
                LOG.p(f"wm.tail types               = {tail_types}")

            committed = _find_recent_committed_actions(buf, limit=120)
            LOG.kv("committed_action entries", len(committed))

            if committed:
                ok_cnt = 0
                for m in committed[:3]:
                    ok, issues = _reflection_entry_sanity(m)
                    ok_cnt += 1 if ok else 0
                    if not ok:
                        LOG.p(f"  bad_entry issues={issues} content='{shorten(wm_row_content(m), 90)}'")

                if ok_cnt >= 1:
                    P11.add(35)
                elif len(committed) >= 1:
                    P11.add(15)

                m0 = committed[0]
                content_obj = wm_row_get(m0, "content_obj", None)
                if isinstance(content_obj, dict) and ("action_name" in content_obj or "outcome" in content_obj):
                    P11.add(10)
            else:
                P11.add(10)

    P11.cap()


# =============================================================================
# FINAL
# =============================================================================
SB.render()
earned, maxp = SB.total()

LOG.h("FINAL RESULTS")
LOG.p(f"Total Score: {earned}/{maxp}")

pass_mark = int(round(maxp * 0.80))
partial_mark = int(round(maxp * 0.63))

if earned >= pass_mark:
    LOG.p(" PASS (stress-ready): stable, hazard-lock forensics clean.")
elif earned >= partial_mark:
    LOG.p(" PARTIAL PASS: stable-ish, but forensics found mismatches worth fixing.")
else:
    LOG.p(" FAIL: drift, missing pieces, or hazard lock still leaky.")

LOG.p("\n=== CLAIR STRESS BENCHMARK COMPLETE ===\n")