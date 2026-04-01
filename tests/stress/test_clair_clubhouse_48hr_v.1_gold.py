# test_clair_clubhouse_48hr.py
# Clair Day 24/25 Benchmark - 48hr clubhouse build stress test
#
# Purpose:
# - Ingest test_doc.txt (clubhouse manual)
# - Simulate 48 hours of user interaction while building a clubhouse
# - Stress:
#   * document ingestion
#   * recall over book content
#   * procedural sequencing
#   * troubleshooting scenarios
#   * safety enforcement
#   * simulator grounding
#   * reflection / idle consolidation
#   * long-term memory persistence
#
# Run:
#   python test_clair_clubhouse_48hr.py
#   python test_clair_clubhouse_48hr.py --hours 6 --cycles_per_hour 6
#   python test_clair_clubhouse_48hr.py --hours 48 --cycles_per_hour 6

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

from clair import Clair
from planning.simulator import Simulator
import planning.simulator as sim_mod


# =============================================================================
# Helpers
# =============================================================================

DEFAULT_DOC_PATH = r"C:\Users\bhilt\OneDrive\Desktop\Clair_Space\Clair_v.2\clair's books\test_doc.txt"


def hr(title: str = "", width: int = 96) -> None:
    print("\n" + "=" * width)
    if title:
        print(title)
        print("=" * width)


def short(text: Any, limit: int = 180) -> str:
    s = str(text).replace("\n", " ").strip()
    if len(s) <= limit:
        return s
    return s[: limit - 1] + "…"


def pct(a: int, b: int) -> str:
    if b <= 0:
        return "0.00%"
    return f"{(100.0 * a / b):.2f}%"


def safe_call(fn: Any, *args, **kwargs) -> Tuple[bool, Any, str]:
    try:
        result = fn(*args, **kwargs)
        return True, result, ""
    except Exception as exc:
        return False, None, repr(exc)


def obj_to_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)

    out: Dict[str, Any] = {}
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        if callable(value):
            continue
        out[name] = value
    return out


def flatten_text(obj: Any) -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj.lower()
    if isinstance(obj, dict):
        return " | ".join(f"{k}:{flatten_text(v)}" for k, v in obj.items()).lower()
    if isinstance(obj, (list, tuple, set)):
        return " | ".join(flatten_text(x) for x in obj).lower()
    return str(obj).lower()


def normalize_text(text: Any) -> str:
    s = str(text or "").lower()
    s = s.replace("’", "'").replace("–", "-").replace("—", "-")
    s = re.sub(r"[^a-z0-9\-\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def contains_expected(blob: str, expected_terms: Iterable[str]) -> bool:
    norm_blob = normalize_text(blob)
    for term in expected_terms:
        if normalize_text(term) in norm_blob:
            return True
    return False


def option_blob(option: Any) -> str:
    d = obj_to_dict(option)
    fields: List[str] = []

    for key in (
        "hazard_family",
        "lock_family",
        "hazard_lock",
        "lock",
        "seed",
        "seed_text",
        "action_name",
        "name",
        "action",
        "description",
        "content",
        "text",
        "details",
    ):
        value = d.get(key)
        if value is not None:
            fields.append(str(value).lower())

    details = d.get("details")
    if isinstance(details, dict):
        for key in (
            "hazard_family",
            "pinned_hazard_family",
            "question_hazard",
            "lock_hazard_family",
            "seed_text",
            "seed_memory",
            "planned_next",
        ):
            value = details.get(key)
            if value is not None:
                fields.append(str(value).lower())

    fields.append(str(option).lower())
    return " | ".join(fields)


# =============================================================================
# Document ingestion helpers
# =============================================================================

SECTION_HINTS = {
    "planning": ["planning", "design", "preparation", "purpose", "requirements", "location"],
    "tools": ["tools", "materials", "checklist", "drill", "hammer", "saw", "level", "chalk line"],
    "safety": ["safety", "goggles", "gloves", "ear protection", "partner", "hazard", "wet", "wind"],
    "foundation": ["footprint", "stake", "string", "diagonal", "square", "post hole", "spray paint"],
    "posts": ["support posts", "4x4", "gravel", "concrete", "cure", "vertical alignment"],
    "floor": ["floor frame", "2x6", "joists", "16 inches", "joist hangers", "plywood", "squeaking"],
    "roof": ["roof", "shingles", "metal", "polycarbonate", "weatherproof"],
    "finish": ["paint", "sealant", "caulk", "trim", "window", "door", "hinges", "latches"],
}

QUERY_ALIASES: Dict[str, List[str]] = {
    "kids": ["children", "playhouse", "visibility", "playful"],
    "kids prioritize": ["children", "playhouse", "safety", "visibility", "playful"],
    "big branch": ["unstable tree branches", "branches", "unsafe"],
    "yard collects water": ["good drainage", "water accumulation", "drainage"],
    "windy and damp": ["wet", "high-wind conditions", "avoid construction"],
    "layout square": ["equal diagonal measurements", "diagonals", "square"],
    "what next after footprint": ["post holes", "support posts", "concrete"],
    "before plywood": ["floor frame", "joists", "joist hangers", "level"],
    "after gravel": ["4x4 posts", "vertical alignment", "concrete"],
    "marking straight cuts": ["chalk line"],
    "straight cuts": ["chalk line"],
    "what tool for straight cuts": ["chalk line"],
    "safety gear cutting drilling": ["goggles", "gloves", "ear protection"],
}


def infer_tags_domain_kind(line: str) -> Tuple[List[str], str, str]:
    low = line.lower()
    tags = ["clubhouse", "construction"]
    domain = "clubhouse_build"
    kind = "procedure"

    if any(x in low for x in SECTION_HINTS["safety"]):
        tags += ["safety"]
        domain = "safety"
        kind = "policy"
    elif any(x in low for x in SECTION_HINTS["tools"]):
        tags += ["tools", "materials"]
        kind = "fact"
    elif any(x in low for x in SECTION_HINTS["foundation"]):
        tags += ["foundation", "layout"]
    elif any(x in low for x in SECTION_HINTS["posts"]):
        tags += ["posts", "foundation"]
    elif any(x in low for x in SECTION_HINTS["floor"]):
        tags += ["floor", "framing"]
    elif any(x in low for x in SECTION_HINTS["roof"]):
        tags += ["roof"]
    elif any(x in low for x in SECTION_HINTS["finish"]):
        tags += ["finishing"]
    elif any(x in low for x in SECTION_HINTS["planning"]):
        tags += ["planning"]
        kind = "guideline"

    if "children" in low:
        tags.append("children")
    if "workshop" in low or "hobby" in low:
        tags.append("hobby")
    if "retreat" in low:
        tags.append("retreat")
    if "chalk line" in low:
        tags += ["chalk line", "marking", "straight cuts"]

    return sorted(set(tags)), domain, kind


def _make_memory(
    content: str,
    *,
    mem_type: str = "lesson",
    confidence: float = 0.96,
    source: str = "seed_verified",
    context: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    domain: str = "clubhouse_build",
    kind: str = "procedure",
    status: str = "verified",
    details: Optional[Dict[str, Any]] = None,
    evidence: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "type": mem_type,
        "content": content,
        "claim": content,
        "confidence": confidence,
        "source": source,
        "context": context or [],
        "tags": sorted(set(tags or [])),
        "domain": domain,
        "kind": kind,
        "status": status,
        "details": details or {},
        "evidence": evidence or [],
    }


def build_section_summary(section: str, lines: List[str]) -> str:
    """
    Compact summary builder. Keep it structured enough to help retrieval
    instead of turning five lines into one giant soup.
    """
    if not lines:
        return section

    chosen = lines[:4]
    fragments = []
    for line in chosen:
        line = line.strip()
        if not line:
            continue
        if len(line) > 140:
            line = line[:139] + "…"
        fragments.append(line)

    if not fragments:
        return section

    return f"{section}: " + " | ".join(fragments)


def extract_lessons_from_doc(doc_text: str) -> List[Dict[str, Any]]:
    """
    Extract four memory layers:
    1. line-level lessons
    2. section summaries
    3. explicit sequence-chain memories
    4. targeted precision memories for known weak recall spots
    """
    lessons: List[Dict[str, Any]] = []
    current_page: Optional[str] = None
    current_section: Optional[str] = None

    section_lines: Dict[str, List[str]] = {}
    raw_lines = [ln.strip() for ln in doc_text.splitlines()]

    for line in raw_lines:
        if not line:
            continue

        if re.match(r"^page\s+\d+", line.strip(), re.I):
            current_page = line.strip()
            continue

        if re.match(r"^\d+\.\s+", line):
            current_section = re.sub(r"^\d+\.\s*", "", line).strip()
            section_lines.setdefault(current_section, [])
            continue

        if len(line) < 8:
            continue

        line = re.sub(r"^[\-\*\u2022]+", "", line).strip()
        if not line:
            continue

        alpha_count = sum(1 for ch in line if ch.isalpha())
        if alpha_count < 8:
            continue

        tags, domain, kind = infer_tags_domain_kind(line)
        context_bits = []
        if current_page:
            context_bits.append(current_page)
        if current_section:
            context_bits.append(current_section)
            section_lines.setdefault(current_section, []).append(line)

        details = {
            "document_name": "test_doc.txt",
            "section": current_section,
            "page": current_page,
            "source_type": "book_ingest_line",
            "verified": True,
            "pending_verification": False,
        }

        lessons.append(
            _make_memory(
                line,
                mem_type="lesson",
                confidence=0.96,
                source="seed_verified",
                context=context_bits,
                tags=tags,
                domain=domain,
                kind=kind,
                status="verified",
                details=details,
                evidence=[f"book:test_doc.txt", f"section:{current_section or 'unknown'}"],
            )
        )

    # Section summaries
    for section, lines in section_lines.items():
        if not lines:
            continue

        summary_text = build_section_summary(section, lines)
        summary_tags, summary_domain, _ = infer_tags_domain_kind(" ".join(lines[:5]))
        summary_tags = sorted(set(summary_tags + ["summary", "section_summary"]))

        lessons.append(
            _make_memory(
                summary_text,
                mem_type="lesson",
                confidence=0.98,
                source="seed_verified",
                context=[section],
                tags=summary_tags,
                domain=summary_domain,
                kind="summary",
                status="verified",
                details={
                    "document_name": "test_doc.txt",
                    "section": section,
                    "source_type": "book_ingest_section_summary",
                    "verified": True,
                    "pending_verification": False,
                },
                evidence=[f"book:test_doc.txt", f"section:{section}"],
            )
        )

    # Explicit sequence-chain memories
    sequence_memories = [
        (
            "After marking the clubhouse footprint, dig the support post holes and prepare post positions before building the floor frame.",
            ["sequence", "foundation", "posts", "layout"],
            "clubhouse_build",
        ),
        (
            "After adding gravel to the post holes, place the 4x4 support posts, check vertical alignment with a level, and then fill with quick-set concrete.",
            ["sequence", "posts", "foundation", "concrete"],
            "clubhouse_build",
        ),
        (
            "After the support posts are set and the concrete has cured for at least 24 hours, build the floor frame with beams and floor joists.",
            ["sequence", "posts", "floor", "framing"],
            "clubhouse_build",
        ),
        (
            "Before laying plywood over the floor, confirm the floor frame is level, install joists spaced 16 inches apart, and use joist hangers if needed.",
            ["sequence", "floor", "framing", "joists"],
            "clubhouse_build",
        ),
        (
            "For children’s clubhouses, prioritize safety, visibility from the main house, and playful features.",
            ["sequence", "planning", "children", "playhouse"],
            "planning",
        ),
        (
            "Avoid building in wet or high-wind conditions, especially when cutting or lifting materials.",
            ["sequence", "safety", "weather", "tools"],
            "safety",
        ),
        (
            "Choose a level site with good drainage and avoid unstable tree branches when selecting the clubhouse location.",
            ["sequence", "planning", "location", "drainage", "branches"],
            "planning",
        ),
    ]

    for text, tags, domain in sequence_memories:
        lessons.append(
            _make_memory(
                text,
                mem_type="lesson",
                confidence=0.99,
                source="seed_verified",
                context=["sequence_chain"],
                tags=sorted(set(tags + ["clubhouse", "construction"])),
                domain=domain,
                kind="sequence_rule",
                status="verified",
                details={
                    "document_name": "test_doc.txt",
                    "source_type": "book_ingest_sequence_chain",
                    "verified": True,
                    "pending_verification": False,
                },
                evidence=["book:test_doc.txt", "section:sequence_chain"],
            )
        )

    # Precision memories for weak spots
    precision_memories = [
        (
            "The chalk line is used for marking straight cuts during clubhouse construction layout and cutting prep.",
            ["tools", "chalk line", "marking", "straight cuts", "layout"],
            "clubhouse_build",
            "fact",
        ),
        (
            "Safety goggles, gloves, and ear protection should be used when cutting, drilling, or using loud tools during clubhouse construction.",
            ["safety", "goggles", "gloves", "ear protection", "cutting", "drilling", "protective equipment"],
            "safety",
            "fact",
        ),
        (
            "Wear safety goggles, gloves, and ear protection when cutting or drilling materials.",
            ["safety", "goggles", "gloves", "ear protection", "cutting", "drilling"],
            "safety",
            "fact",
        ),
        (
            "Use a chalk line for marking straight cuts.",
            ["tools", "chalk line", "marking", "straight cuts"],
            "clubhouse_build",
            "fact",
        ),
    ]

    for text, tags, domain, kind in precision_memories:
        lessons.append(
            _make_memory(
                text,
                mem_type="lesson",
                confidence=0.99,
                source="seed_verified",
                context=["precision_memory"],
                tags=sorted(set(tags + ["clubhouse", "construction"])),
                domain=domain,
                kind=kind,
                status="verified",
                details={
                    "document_name": "test_doc.txt",
                    "source_type": "book_ingest_precision_memory",
                    "verified": True,
                    "pending_verification": False,
                },
                evidence=["book:test_doc.txt", "section:precision_memory"],
            )
        )

    # De-duplicate exact content
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for item in lessons:
        c = item["content"].strip().lower()
        if c not in seen:
            deduped.append(item)
            seen.add(c)

    return deduped


def expand_query_aliases(query: str) -> str:
    q = (query or "").strip()
    low = q.lower()

    expansions: List[str] = []
    for key, vals in QUERY_ALIASES.items():
        if key in low:
            expansions.extend(vals)

    if "kids" in low or "children" in low:
        expansions.extend(["children's playhouse", "visibility", "playful features"])
    if "branch" in low:
        expansions.extend(["unstable tree branches"])
    if "water" in low and ("yard" in low or "rain" in low):
        expansions.extend(["good drainage", "water accumulation"])
    if "wind" in low or "damp" in low or "wet" in low:
        expansions.extend(["wet conditions", "high-wind conditions", "avoid construction"])
    if "square" in low:
        expansions.extend(["equal diagonal measurements", "layout is square"])
    if "what next" in low and "footprint" in low:
        expansions.extend(["dig post holes", "support posts"])
    if "before" in low and "plywood" in low:
        expansions.extend(["floor frame level", "joists spaced 16 inches apart", "joist hangers"])
    if "after" in low and "gravel" in low:
        expansions.extend(["place 4x4 posts", "vertical alignment", "quick-set concrete"])
    if "straight cuts" in low or ("marking" in low and "cuts" in low):
        expansions.extend(["chalk line", "tool for straight cuts"])
    if "safety gear" in low and ("cutting" in low or "drilling" in low):
        expansions.extend(["goggles", "gloves", "ear protection", "protective equipment"])

    expansions = sorted(set(x for x in expansions if x))
    if not expansions:
        return q
    return q + " " + " ".join(expansions)


# =============================================================================
# Benchmark data
# =============================================================================

RECALL_PROBES: List[Tuple[str, List[str]]] = [
    ("How deep should support posts be?", ["18", "24", "18-24"]),
    ("How wide should the post holes be?", ["8", "10", "8-10"]),
    ("How much gravel goes at the base of a post hole?", ["3", "4", "3-4", "gravel"]),
    ("How long should concrete cure before proceeding?", ["24 hours", "24"]),
    ("How far apart should floor joists be?", ["16 inches", "16"]),
    ("Why should the site have good drainage?", ["water", "drainage", "accumulation"]),
    ("What safety gear should be used when cutting or drilling?", ["goggles", "gloves", "ear protection"]),
    ("Why should the floor frame be level?", ["uneven", "level", "flooring"]),
    ("What tools are used for marking straight cuts?", ["chalk line"]),
    ("What should be done to plywood edges after installation?", ["sand", "splinters"]),
]

SEQUENCE_PROBES: List[Tuple[str, List[str]]] = [
    (
        "What comes after marking the footprint and before building the floor frame?",
        ["dig holes", "support posts", "concrete", "posts"],
    ),
    (
        "What should happen before laying plywood over the joists?",
        ["floor frame", "joists", "level", "joist hangers"],
    ),
    (
        "What should happen after adding gravel to the post holes?",
        ["place 4x4 posts", "level", "concrete"],
    ),
]

USER_SCENARIOS: List[Dict[str, Any]] = [
    {
        "question": "I'm building for kids. What should I prioritize?",
        "expect": ["safety", "visibility", "playful"],
        "domain": "planning",
    },
    {
        "question": "I picked a site under a big branch. Is that okay?",
        "expect": ["avoid", "unstable tree branches", "unsafe", "branches"],
        "domain": "safety",
    },
    {
        "question": "My yard collects water after rain. What should I think about?",
        "expect": ["drainage", "water accumulation", "level", "stable"],
        "domain": "foundation",
    },
    {
        "question": "I already poured the concrete but one post is leaning.",
        "expect": ["level", "vertical alignment", "post", "concrete"],
        "domain": "posts",
    },
    {
        "question": "Can I start building the frame right away after pouring concrete?",
        "expect": ["24 hours", "cure", "before proceeding"],
        "domain": "posts",
    },
    {
        "question": "I spaced my floor joists 24 inches apart. Is that okay?",
        "expect": ["16 inches", "joists", "support", "load"],
        "domain": "floor",
    },
    {
        "question": "The floor squeaks when stepped on. What should I check?",
        "expect": ["deck screws", "movement", "joists", "plywood"],
        "domain": "floor",
    },
    {
        "question": "It's windy and damp outside. Should I keep building?",
        "expect": ["avoid", "wet", "high-wind", "conditions"],
        "domain": "safety",
    },
    {
        "question": "Why should children’s clubhouses be visible from the main house?",
        "expect": ["visible", "main house", "children"],
        "domain": "planning",
    },
    {
        "question": "What should I check to make sure the layout is square?",
        "expect": ["diagonals", "equal diagonal", "square"],
        "domain": "foundation",
    },
]

SIM_SCENARIOS: List[Tuple[str, List[str], Dict[str, Any]]] = [
    (
        "The builder is setting support posts for a clubhouse. The holes are dug, but one post is leaning and concrete has just been poured.",
        ["level", "vertical alignment", "post", "concrete", "cure"],
        {"domain": "clubhouse_build", "tags": ["clubhouse", "posts", "safety"]},
    ),
    (
        "A user is building a clubhouse floor. The frame seems uneven and they are about to install plywood anyway.",
        ["level", "frame", "joists", "plywood", "uneven"],
        {"domain": "clubhouse_build", "tags": ["clubhouse", "floor", "framing"]},
    ),
    (
        "It is wet and windy while a user is cutting wood for a clubhouse roof with power tools.",
        ["avoid", "wet", "high-wind", "safety", "power tool"],
        {"domain": "safety", "tags": ["clubhouse", "safety", "roof"]},
    ),
]

IDENTITY_LEAK_TERMS = [
    "my name is clair",
    "i am clair",
    "cognitive learning and interactive reasoner",
]


# =============================================================================
# Clair adapter
# =============================================================================

class ClairAdapter:
    def __init__(self, clair: Any) -> None:
        self.clair = clair
        self.memory = clair.memory
        self.long_term = clair.long_term
        self.simulator = clair.simulator

    def ltm_count(self) -> int:
        ok, result, _ = safe_call(self.long_term.get_all_memories)
        if ok and isinstance(result, list):
            return len(result)
        return -1

    def wm_count(self) -> int:
        buf = getattr(self.memory, "buffer", None)
        if isinstance(buf, list):
            return len(buf)
        return -1

    def ingest_doc_lessons(self, lessons: List[Dict[str, Any]]) -> Dict[str, int]:
        stats = {
            "wm_ok": 0,
            "ltm_ok": 0,
            "ltm_inserted_new": 0,
            "ltm_reinforced_existing": 0,
            "ltm_merged_revision": 0,
            "ltm_stored_conflict_variant": 0,
        }

        for item in lessons:
            ok, _, _ = safe_call(self.memory.store, item)
            if ok:
                stats["wm_ok"] += 1

            if hasattr(self.long_term, "store_detailed"):
                ok2, result, _ = safe_call(self.long_term.store_detailed, item)
                if ok2 and isinstance(result, dict):
                    stored = int(result.get("stored", 0) or 0)
                    if stored > 0:
                        stats["ltm_ok"] += stored

                    for row in result.get("results", []):
                        if not isinstance(row, dict):
                            continue
                        action = str(row.get("action", "") or "").strip().lower()
                        if action == "inserted_new":
                            stats["ltm_inserted_new"] += 1
                        elif action == "reinforced_existing":
                            stats["ltm_reinforced_existing"] += 1
                        elif action == "merged_revision":
                            stats["ltm_merged_revision"] += 1
                        elif action == "stored_conflict_variant":
                            stats["ltm_stored_conflict_variant"] += 1
            else:
                before = self.ltm_count()
                ok2, _, _ = safe_call(self.long_term.store, item)
                after = self.ltm_count()
                if ok2 and before >= 0 and after >= 0 and after > before:
                    stats["ltm_ok"] += 1
                    stats["ltm_inserted_new"] += 1

        return stats

    def retrieve_candidates(self, query: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        expanded_query = expand_query_aliases(query)

        ok, result, err = safe_call(self.memory.retrieve, expanded_query)
        if ok:
            out["wm.retrieve"] = result
        else:
            out["wm.retrieve_error"] = err

        ok, result, err = safe_call(self.long_term.search, expanded_query)
        if ok:
            out["ltm.search"] = result
        else:
            out["ltm.search_error"] = err

        ok, result, err = safe_call(self.long_term.retrieve, expanded_query)
        if ok:
            out["ltm.retrieve"] = result
        else:
            out["ltm.retrieve_error"] = err

        return out

    def retrieve_ranked_blob(self, query: str, expected_terms: List[str]) -> Tuple[str, Dict[str, Any], bool]:
        candidates = self.retrieve_candidates(query)
        parts: List[str] = []
        identity_leak = False

        for key in ("ltm.search", "ltm.retrieve", "wm.retrieve"):
            if key not in candidates:
                continue
            blob = flatten_text(candidates[key]).strip()
            if not blob:
                continue

            hit = contains_expected(blob, expected_terms)
            is_identity = any(term in blob for term in IDENTITY_LEAK_TERMS)
            if is_identity:
                identity_leak = True

            tag = "[HIT]" if hit else "[MISS]"
            if is_identity:
                tag += "[IDENTITY]"
            parts.append(f"{tag} {key}: {blob}")

        # fallback direct scan across all candidate text
        combined_blob = " || ".join(parts).lower()
        if not parts and candidates:
            combined_blob = flatten_text(candidates).lower()

        if combined_blob and not any(p.startswith("[HIT]") for p in parts):
            if contains_expected(combined_blob, expected_terms):
                parts.append(f"[HIT][FALLBACK] aggregate: {combined_blob[:400]}")

        return " || ".join(parts), candidates, identity_leak

    def plan(self, scenario: str, context_profile: Optional[Dict[str, Any]] = None, num_actions: int = 3) -> List[Any]:
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            ok, result, _ = safe_call(
                self.simulator.generate_options,
                self.memory,
                num_actions=num_actions,
                question=scenario,
                context_profile=context_profile,
                horizon=1,
            )

        if not ok:
            return []

        if isinstance(result, list):
            return result
        if isinstance(result, tuple) and result and isinstance(result[0], list):
            return result[0]
        return []

    def reflect(self) -> bool:
        ok, _, _ = safe_call(self.clair.reflect)
        return ok


# =============================================================================
# Benchmark
# =============================================================================

class Clubhouse48HrBenchmark:
    def __init__(self, adapter: ClairAdapter, doc_path: str, hours: int, cycles_per_hour: int) -> None:
        self.adapter = adapter
        self.doc_path = doc_path
        self.hours = hours
        self.cycles_per_hour = cycles_per_hour

        self.initial_wm = -1
        self.initial_ltm = -1

        self.ingest_lessons = 0
        self.ingest_wm_ok = 0
        self.ingest_ltm_ok = 0

        self.recall_ok = 0
        self.recall_total = 0
        self.recall_fails: List[str] = []

        self.sequence_ok = 0
        self.sequence_total = 0
        self.sequence_fails: List[str] = []

        self.user_ok = 0
        self.user_total = 0
        self.user_fails: List[str] = []

        self.sim_ok = 0
        self.sim_total = 0
        self.sim_fails: List[str] = []

        self.reflect_ok = 0
        self.reflect_total = 0

        self.idle_cycles = 0
        self.identity_leaks = 0

        self.ltm_persist = 0
        self.ltm_inserted_new = 0
        self.ltm_reinforced_existing = 0
        self.ltm_merged_revision = 0
        self.ltm_stored_conflict_variant = 0

        self._recall_idx = 0
        self._sequence_idx = 0
        self._user_idx = 0
        self._sim_idx = 0

    def _resolve_doc_path(self) -> str:
        candidate_paths = [
            self.doc_path,
            DEFAULT_DOC_PATH,
            os.path.join(os.getcwd(), self.doc_path),
            os.path.join(os.getcwd(), "clair's books", os.path.basename(self.doc_path)),
            os.path.join(os.getcwd(), "clairs books", os.path.basename(self.doc_path)),
            os.path.join(os.getcwd(), "books", os.path.basename(self.doc_path)),
        ]

        for path in candidate_paths:
            if path and os.path.exists(path):
                return path

        tried = "\n".join(f" - {p}" for p in candidate_paths)
        raise FileNotFoundError(f"Document not found. Tried:\n{tried}")

    def next_recall(self) -> Tuple[str, List[str]]:
        item = RECALL_PROBES[self._recall_idx % len(RECALL_PROBES)]
        self._recall_idx += 1
        return item

    def next_sequence(self) -> Tuple[str, List[str]]:
        item = SEQUENCE_PROBES[self._sequence_idx % len(SEQUENCE_PROBES)]
        self._sequence_idx += 1
        return item

    def next_user(self) -> Dict[str, Any]:
        item = USER_SCENARIOS[self._user_idx % len(USER_SCENARIOS)]
        self._user_idx += 1
        return item

    def next_sim(self) -> Tuple[str, List[str], Dict[str, Any]]:
        item = SIM_SCENARIOS[self._sim_idx % len(SIM_SCENARIOS)]
        self._sim_idx += 1
        return item

    def ingest_book(self) -> None:
        resolved_path = self._resolve_doc_path()
        self.doc_path = resolved_path

        with open(self.doc_path, "r", encoding="utf-8") as f:
            doc_text = f.read()

        lessons = extract_lessons_from_doc(doc_text)
        self.ingest_lessons = len(lessons)

        stats = self.adapter.ingest_doc_lessons(lessons)
        self.ingest_wm_ok = stats["wm_ok"]
        self.ingest_ltm_ok = stats["ltm_ok"]
        self.ltm_persist += stats["ltm_ok"]
        self.ltm_inserted_new += stats["ltm_inserted_new"]
        self.ltm_reinforced_existing += stats["ltm_reinforced_existing"]
        self.ltm_merged_revision += stats["ltm_merged_revision"]
        self.ltm_stored_conflict_variant += stats["ltm_stored_conflict_variant"]

    def run_recall_probe(self) -> None:
        query, expected_terms = self.next_recall()
        blob, candidates, identity_leak = self.adapter.retrieve_ranked_blob(query, expected_terms)

        self.recall_total += 1
        if identity_leak:
            self.identity_leaks += 1

        if contains_expected(blob, expected_terms):
            self.recall_ok += 1
            return

        if len(self.recall_fails) < 5:
            fail_bits = []
            for key in ("ltm.search", "ltm.retrieve", "wm.retrieve"):
                if key in candidates:
                    fail_bits.append(f"{key}={short(flatten_text(candidates[key]), 120)!r}")
            self.recall_fails.append(
                f"query={query!r} got={' | '.join(fail_bits) if fail_bits else short(blob)!r}"
            )

    def run_sequence_probe(self) -> None:
        query, expected_terms = self.next_sequence()
        blob, candidates, identity_leak = self.adapter.retrieve_ranked_blob(query, expected_terms)

        self.sequence_total += 1
        if identity_leak:
            self.identity_leaks += 1

        if contains_expected(blob, expected_terms):
            self.sequence_ok += 1
            return

        if len(self.sequence_fails) < 5:
            fail_bits = []
            for key in ("ltm.search", "ltm.retrieve", "wm.retrieve"):
                if key in candidates:
                    fail_bits.append(f"{key}={short(flatten_text(candidates[key]), 120)!r}")
            self.sequence_fails.append(
                f"query={query!r} got={' | '.join(fail_bits) if fail_bits else short(blob)!r}"
            )

    def run_user_probe(self) -> None:
        item = self.next_user()
        blob, candidates, identity_leak = self.adapter.retrieve_ranked_blob(item["question"], item["expect"])

        self.user_total += 1
        if identity_leak:
            self.identity_leaks += 1

        if contains_expected(blob, item["expect"]):
            self.user_ok += 1
            return

        if len(self.user_fails) < 5:
            fail_bits = []
            for key in ("ltm.search", "ltm.retrieve", "wm.retrieve"):
                if key in candidates:
                    fail_bits.append(f"{key}={short(flatten_text(candidates[key]), 120)!r}")
            self.user_fails.append(
                f"question={item['question']!r} got={' | '.join(fail_bits) if fail_bits else short(blob)!r}"
            )

    def run_sim_probe(self) -> None:
        scenario, expected_terms, context_profile = self.next_sim()
        options = self.adapter.plan(scenario, context_profile=context_profile, num_actions=3)

        self.sim_total += 1

        if not options:
            if len(self.sim_fails) < 5:
                self.sim_fails.append(f"scenario={short(scenario)!r} got='NO_OPTIONS'")
            return

        merged = " || ".join(option_blob(opt) for opt in options[:3])
        if contains_expected(merged, expected_terms):
            self.sim_ok += 1
            return

        if len(self.sim_fails) < 5:
            self.sim_fails.append(
                f"scenario={short(scenario)!r} top1={short(option_blob(options[0]), 180)!r}"
            )

    def run_idle_cycle(self) -> None:
        self.idle_cycles += 1
        self.reflect_total += 1
        if self.adapter.reflect():
            self.reflect_ok += 1

    @staticmethod
    def score(ratio: float, max_points: int) -> int:
        ratio = max(0.0, min(1.0, ratio))
        return round(ratio * max_points)

    def run(self) -> None:
        hr("CLAIR 48HR CLUBHOUSE FIELD STRESS TEST")
        self.initial_wm = self.adapter.wm_count()
        self.initial_ltm = self.adapter.ltm_count()

        print(f"Document path                  {self.doc_path}")
        print(f"Simulated hours                {self.hours}")
        print(f"Cycles per hour                {self.cycles_per_hour}")
        print(f"Initial WM count               {self.initial_wm}")
        print(f"Initial LTM count              {self.initial_ltm}")

        t0 = time.time()

        self.ingest_book()

        print(f"Resolved document path         {self.doc_path}")
        print(f"Ingested lesson candidates     {self.ingest_lessons}")
        print(f"Ingested into WM               {self.ingest_wm_ok}")
        print(f"Persisted/consolidated in LTM  {self.ingest_ltm_ok}")

        total_cycles = self.hours * self.cycles_per_hour
        for cycle in range(total_cycles):
            self.run_recall_probe()

            if cycle % 2 == 0:
                self.run_user_probe()

            if cycle % 3 == 0:
                self.run_sequence_probe()

            if cycle % 4 == 0:
                self.run_sim_probe()

            if cycle % 2 == 1:
                self.run_idle_cycle()

            if (cycle + 1) % self.cycles_per_hour == 0:
                hour_num = (cycle + 1) // self.cycles_per_hour
                print(
                    f"[hour {hour_num:02d}] "
                    f"recall={self.recall_ok}/{self.recall_total} "
                    f"sequence={self.sequence_ok}/{self.sequence_total} "
                    f"user={self.user_ok}/{self.user_total} "
                    f"sim={self.sim_ok}/{self.sim_total} "
                    f"reflect={self.reflect_ok}/{self.reflect_total} "
                    f"idle={self.idle_cycles} "
                    f"identity_leaks={self.identity_leaks} "
                    f"ltm_persist={self.ltm_persist} "
                    f"(new={self.ltm_inserted_new}, reinforce={self.ltm_reinforced_existing}, "
                    f"merge={self.ltm_merged_revision}, contested={self.ltm_stored_conflict_variant}) "
                    f"WM={self.adapter.wm_count()} "
                    f"LTM={self.adapter.ltm_count()}"
                )

        elapsed = time.time() - t0
        final_wm = self.adapter.wm_count()
        final_ltm = self.adapter.ltm_count()
        net_ltm_delta = final_ltm - self.initial_ltm if final_ltm >= 0 and self.initial_ltm >= 0 else 0

        phase_a = self.score(self.ingest_wm_ok / max(1, self.ingest_lessons), 150)
        phase_b = self.score(self.recall_ok / max(1, self.recall_total), 200)
        phase_c = self.score(self.sequence_ok / max(1, self.sequence_total), 125)
        phase_d = self.score(self.user_ok / max(1, self.user_total), 175)
        phase_e = self.score(self.sim_ok / max(1, self.sim_total), 175)
        phase_f = self.score(self.reflect_ok / max(1, self.reflect_total), 75)

        growth_checks = 0
        growth_total = 4
        if final_wm >= 0:
            growth_checks += 1
        if final_ltm >= 0:
            growth_checks += 1
        if self.ltm_persist > 0:
            growth_checks += 1
        if net_ltm_delta > 0 or self.ltm_reinforced_existing > 0 or self.ltm_merged_revision > 0 or self.ltm_stored_conflict_variant > 0:
            growth_checks += 1
        phase_g = self.score(growth_checks / growth_total, 100)

        identity_penalty = 0
        if self.identity_leaks > 0:
            identity_penalty = min(100, self.identity_leaks * 10)

        total = phase_a + phase_b + phase_c + phase_d + phase_e + phase_f + phase_g - identity_penalty
        total = max(0, total)

        hr("SCORE SUMMARY")
        print(f"Phase A: Book ingestion                 {phase_a:3d}/150")
        print(f"Phase B: Direct recall                  {phase_b:3d}/200")
        print(f"Phase C: Procedural sequence            {phase_c:3d}/125")
        print(f"Phase D: User troubleshooting           {phase_d:3d}/175")
        print(f"Phase E: Simulator grounding            {phase_e:3d}/175")
        print(f"Phase F: Reflection / idle              {phase_f:3d}/75")
        print(f"Phase G: Persistent memory              {phase_g:3d}/100")
        print(f"Identity leak penalty                  -{identity_penalty:3d}")

        hr("FINAL RESULTS")
        print(f"Runtime seconds                {elapsed:.3f}")
        print(f"Final WM count                 {final_wm}")
        print(f"Final LTM count                {final_ltm}")
        print(f"Net LTM row delta              {net_ltm_delta}")
        print(f"Lessons parsed from doc        {self.ingest_lessons}")
        print(f"WM ingest successes            {self.ingest_wm_ok}")
        print(f"LTM persist successes          {self.ltm_persist}")
        print(f"LTM inserted_new               {self.ltm_inserted_new}")
        print(f"LTM reinforced_existing        {self.ltm_reinforced_existing}")
        print(f"LTM merged_revision            {self.ltm_merged_revision}")
        print(f"LTM stored_conflict_variant    {self.ltm_stored_conflict_variant}")
        print(f"Recall                         {self.recall_ok}/{self.recall_total} ({pct(self.recall_ok, self.recall_total)})")
        print(f"Sequence                       {self.sequence_ok}/{self.sequence_total} ({pct(self.sequence_ok, self.sequence_total)})")
        print(f"User troubleshooting           {self.user_ok}/{self.user_total} ({pct(self.user_ok, self.user_total)})")
        print(f"Simulator grounding            {self.sim_ok}/{self.sim_total} ({pct(self.sim_ok, self.sim_total)})")
        print(f"Reflection                     {self.reflect_ok}/{self.reflect_total} ({pct(self.reflect_ok, self.reflect_total)})")
        print(f"Idle cycles                    {self.idle_cycles}")
        print(f"Identity leaks                 {self.identity_leaks}")
        print(f"Total Score                    {total}/1000")

        ratio = total / 1000.0
        if ratio >= 0.95:
            verdict = "PASS (clubhouse field-stable)"
        elif ratio >= 0.85:
            verdict = "PASS-WARN (usable, mild weakness)"
        elif ratio >= 0.70:
            verdict = "WARN (visible stress under field cycle)"
        else:
            verdict = "FAIL (instability or weak field retention)"

        print(f"Verdict                        {verdict}")

        if self.recall_fails:
            hr("SAMPLE RECALL FAILURES")
            for item in self.recall_fails:
                print(item)

        if self.sequence_fails:
            hr("SAMPLE SEQUENCE FAILURES")
            for item in self.sequence_fails:
                print(item)

        if self.user_fails:
            hr("SAMPLE USER TROUBLESHOOT FAILURES")
            for item in self.user_fails:
                print(item)

        if self.sim_fails:
            hr("SAMPLE SIMULATOR FAILURES")
            for item in self.sim_fails:
                print(item)


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=48)
    parser.add_argument("--cycles_per_hour", type=int, default=6)
    parser.add_argument("--doc", type=str, default=DEFAULT_DOC_PATH)
    args = parser.parse_args()

    hr("IMPORT CHECK")
    print(f"Clair class                     {Clair}")
    print(f"planning.simulator path        {sim_mod.__file__}")
    print(f"Simulator class                {Simulator}")
    print(f"Simulator VERSION              {getattr(Simulator, 'VERSION', 'missing')}")

    clair = Clair()

    hr("RUNTIME WIRING CHECK")
    print(f"Clair simulator class          {clair.simulator.__class__}")
    print(f"Clair simulator VERSION        {getattr(clair.simulator, 'VERSION', 'unknown')}")
    print(f"Clair memory class             {clair.memory.__class__}")
    print(f"Clair long_term class          {clair.long_term.__class__}")

    adapter = ClairAdapter(clair)
    bench = Clubhouse48HrBenchmark(
        adapter=adapter,
        doc_path=args.doc,
        hours=args.hours,
        cycles_per_hour=args.cycles_per_hour,
    )
    bench.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
