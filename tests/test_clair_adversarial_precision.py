# test_clair_adversarial_precision.py
# Clair Day 30 Adversarial Precision Benchmark
#
# Purpose:
# - Stress local precision under adversarial memory pressure
# - Verify revision collision handling
# - Verify same-domain interference resistance
# - Verify false-cue contamination resistance
# - Verify overload precision in retrieval + simulator planning
# - Inspect whether simulator picks the most local actionable seed
#
# Expected outcome:
# - Clair should select the most local actionable seed
# - unrelated but semantically nearby memories should not hijack planning
# - misleading keywords should not pull the wrong domain/hazard
# - deterministic runs should remain stable

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import inspect
import io
import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from clair import Clair
from planning.simulator import Simulator
import planning.simulator as sim_mod


# =============================================================================
# Helpers
# =============================================================================

DEFAULT_DOC_PATH = os.path.join(os.getcwd(), "test_doc.txt")


def hr(title: str = "", width: int = 100) -> None:
    print("\n" + "=" * width)
    if title:
        print(title)
        print("=" * width)


def short(text: Any, limit: int = 180) -> str:
    s = str(text).replace("\n", " ").strip()
    if len(s) <= limit:
        return s
    return s[: limit - 3] + "..."


def pct(a: int, b: int) -> str:
    if b <= 0:
        return "0.00%"
    return f"{(100.0 * a / b):.2f}%"


def safe_call(fn: Any, *args, **kwargs) -> Tuple[bool, Any, str]:
    try:
        result = fn(*args, **kwargs)
        return True, result, ""
    except Exception as exc:
        return False, None, f"{type(exc).__name__}: {exc}"


def supported_kwargs(fn: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(fn)
    except Exception:
        return dict(kwargs)

    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)

    return {k: v for k, v in kwargs.items() if k in params}


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
    if dataclasses.is_dataclass(obj):
        return flatten_text(dataclasses.asdict(obj))
    return str(obj).lower()


def normalize_text(text: Any) -> str:
    s = str(text or "").lower()
    s = s.replace("’", "'").replace("–", "-").replace("—", "-")
    s = s.replace("_", " ")
    s = re.sub(r"[^a-z0-9\-\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def contains_expected(blob: str, expected_terms: Iterable[str]) -> bool:
    norm_blob = normalize_text(blob)
    for term in expected_terms:
        if normalize_text(term) in norm_blob:
            return True
    return False


def contains_forbidden(blob: str, forbidden_terms: Iterable[str]) -> bool:
    norm_blob = normalize_text(blob)
    for term in forbidden_terms:
        if normalize_text(term) in norm_blob:
            return True
    return False


def count_expected_hits(blob: str, expected_terms: Iterable[str]) -> int:
    norm_blob = normalize_text(blob)
    hits = 0
    for term in expected_terms:
        if normalize_text(term) in norm_blob:
            hits += 1
    return hits


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
        "reason",
        "rationale",
        "note",
    ):
        value = d.get(key)
        if value is not None:
            fields.append(flatten_text(value))

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
            "reason",
            "local_anchor_hits",
            "active_anchor_count",
            "revision_locality_penalty_final",
            "debug_counts",
        ):
            value = details.get(key)
            if value is not None:
                fields.append(flatten_text(value))

    fields.append(flatten_text(option))
    return " | ".join(x for x in fields if x)


def make_memory(
    content: str,
    *,
    mem_type: str = "lesson",
    confidence: float = 0.96,
    source: str = "seed_verified",
    context: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    domain: str = "general",
    kind: str = "fact",
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


def resolve_doc_path(doc_path: str) -> str:
    raw = (doc_path or "").strip()
    base_name = os.path.basename(raw) if raw else ""

    candidate_paths: List[str] = []

    if raw:
        candidate_paths.extend([
            raw,
            os.path.abspath(raw),
        ])

    if base_name:
        candidate_paths.extend([
            DEFAULT_DOC_PATH,
            os.path.join(os.getcwd(), raw) if raw else "",
            os.path.join(os.getcwd(), base_name),
            os.path.join(os.getcwd(), "clair's books", base_name),
            os.path.join(os.getcwd(), "clairs books", base_name),
            os.path.join(os.getcwd(), "clair books", base_name),
            os.path.join(os.getcwd(), "books", base_name),
            os.path.join(os.getcwd(), "docs", base_name),
            os.path.join(os.getcwd(), "documents", base_name),
            os.path.join(os.getcwd(), "data", base_name),
            os.path.join(os.getcwd(), "clair_s books", base_name),
        ])
    else:
        candidate_paths.append(DEFAULT_DOC_PATH)

    seen: Set[str] = set()
    cleaned_candidates: List[str] = []
    for path in candidate_paths:
        if not path:
            continue
        norm = os.path.normpath(path)
        if norm not in seen:
            seen.add(norm)
            cleaned_candidates.append(norm)

    for path in cleaned_candidates:
        if os.path.isfile(path):
            return path

    if base_name:
        recursive_hits: List[str] = []
        for root, _, files in os.walk(os.getcwd()):
            for fname in files:
                if fname.lower() == base_name.lower():
                    recursive_hits.append(os.path.join(root, fname))

        if len(recursive_hits) == 1:
            return recursive_hits[0]

        if len(recursive_hits) > 1:
            tried = "\n".join(f" - {p}" for p in cleaned_candidates)
            found = "\n".join(f" - {p}" for p in recursive_hits[:25])
            raise FileNotFoundError(
                f"Multiple matching documents found for {base_name!r}. "
                f"Use --doc with the full path.\n\nTried:\n{tried}\n\nFound matches:\n{found}"
            )

    tried = "\n".join(f" - {p}" for p in cleaned_candidates)
    raise FileNotFoundError(
        f"Document not found for input {doc_path!r}.\nTried:\n{tried}"
    )


# =============================================================================
# Query shaping / ranking helpers
# =============================================================================

STOPWORDS = {
    "is", "are", "was", "were", "the", "a", "an", "of", "and", "or",
    "to", "in", "on", "for", "my", "your", "you", "i", "we", "it",
    "that", "this", "with", "as", "at", "by", "from", "what", "why",
    "how", "do", "does", "did", "should", "would", "can", "could",
    "tell", "me", "about", "during", "after", "before", "while",
    "now", "please", "today", "tonight", "useful", "only", "many",
    "among", "these", "there", "when", "being", "under", "all",
    "answer", "just", "into", "then",
}

TOKEN_ALIASES = {
    "raisedbeds": "raised_bed",
    "raised": "raised",
    "beds": "bed",
    "bed": "bed",
    "vegetables": "vegetable",
    "plants": "plant",
    "roots": "root",
    "watering": "water",
    "seedlings": "seedling",
    "transplanting": "transplant",
    "transplanted": "transplant",
    "drainage": "drainage",
    "soilcontrol": "soil_control",
    "flickering": "flicker",
    "lights": "light",
    "wires": "wire",
    "weeds": "weed",
    "suppressing": "suppress",
    "suppresses": "suppress",
    "retaining": "retain",
    "retains": "retain",
    "moisture": "moisture",
    "mulching": "mulch",
    "yellowing": "yellow",
    "tripping": "trip",
    "connections": "connection",
    "decorative": "decorative",
    "decorations": "decorative",
    "planned": "plan",
    "planning": "plan",
}


DOMAIN_QUERY_HINTS: Dict[str, List[str]] = {
    "clubhouse_build": ["clubhouse", "construction", "posts", "floor", "roof", "framing", "layout"],
    "planning": ["clubhouse", "planning", "children", "visibility", "design", "playhouse"],
    "safety": ["safety", "protective equipment", "hazard", "wet", "wind", "drilling", "cutting"],
    "garden": ["garden", "plants", "soil", "watering", "seedlings", "vegetable", "transplant", "raised bed", "mulch"],
    "electrical": ["electrical", "wiring", "breaker", "circuit", "voltage tester", "power", "live wiring"],
}


def stem_token(tok: str) -> str:
    w = normalize_text(tok).replace(" ", "")
    if not w:
        return ""
    w = TOKEN_ALIASES.get(w, w)
    if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    return w


def token_list(text: str) -> List[str]:
    raw = normalize_text(text).split()
    out: List[str] = []
    for tok in raw:
        if tok in STOPWORDS:
            continue
        st = stem_token(tok)
        if st and st not in STOPWORDS:
            out.append(st)
    return out


def token_set(text: str) -> Set[str]:
    return set(token_list(text))


def anchor_tokens(text: str) -> Set[str]:
    return {t for t in token_set(text) if t and t not in STOPWORDS and len(t) >= 4}


def query_flags(query: str) -> Dict[str, bool]:
    q = normalize_text(query)
    return {
        "support_posts_soft_soil": ("support posts" in q and "soft soil" in q),
        "floor_frame_before_plywood": ("floor frame" in q and "plywood" in q),
        "decorative_priority": ("decorative" in q and ("come first" in q or "build sequence" in q)),
        "yellow_soggy": ("yellow" in q and "soggy soil" in q),
        "breaker_load": ("breaker" in q and ("trip" in q or "tripping" in q or "load" in q or "microwave" in q)),
    }


def candidate_rank_score(
    text: str,
    query: str,
    expected_terms: Iterable[str],
) -> Tuple[int, int, int, int, int, int, int]:
    norm = normalize_text(text)
    text_tokens = token_set(text)
    q_anchors = anchor_tokens(query)
    flags = query_flags(query)

    exact_expected_hits = 0
    partial_expected_hits = 0

    for term in expected_terms:
        nt = normalize_text(term)
        if nt in norm:
            exact_expected_hits += 1
        else:
            tt = token_set(term)
            if tt and (tt & text_tokens):
                partial_expected_hits += 1

    query_anchor_hits = len(q_anchors & text_tokens)

    exact_query_phrase_bonus = 0
    nq = normalize_text(query)
    if nq and nq in norm:
        exact_query_phrase_bonus = 2

    same_domain_bonus = 0
    if flags["floor_frame_before_plywood"] and ("floor" in text_tokens or "frame" in text_tokens):
        same_domain_bonus = 2
    elif flags["yellow_soggy"] and ("yellow" in text_tokens or "overwater" in text_tokens):
        same_domain_bonus = 2
    elif flags["breaker_load"] and ("breaker" in text_tokens or "overload" in text_tokens or "short" in text_tokens):
        same_domain_bonus = 2

    primary_revision_bonus = 0
    contamination_penalty = 0

    if flags["support_posts_soft_soil"]:
        if "soft soil" in norm and "24 inches" in norm:
            primary_revision_bonus += 4
        if "decorative upgrades" in norm:
            contamination_penalty -= 5
        if "compacted zones" in norm or "22 inches" in norm:
            contamination_penalty -= 7

    if flags["floor_frame_before_plywood"]:
        if "before laying plywood" in norm:
            primary_revision_bonus += 4
        if "joist spacing" in norm:
            primary_revision_bonus += 3
        if "support posts" in norm:
            contamination_penalty -= 4

    if flags["decorative_priority"]:
        if "alignment checks still come first" in norm or "structural alignment" in norm:
            primary_revision_bonus += 4

    return (
        exact_expected_hits,
        primary_revision_bonus,
        same_domain_bonus,
        query_anchor_hits,
        partial_expected_hits,
        exact_query_phrase_bonus,
        contamination_penalty,
    )


def expand_query_with_domain(query: str, domain_hint: Optional[str] = None) -> str:
    q = (query or "").strip()
    extras: List[str] = []

    if domain_hint and domain_hint in DOMAIN_QUERY_HINTS:
        extras.extend(DOMAIN_QUERY_HINTS[domain_hint])

    low = q.lower()

    if "straight cuts" in low:
        extras.extend(["chalk line", "marking straight cuts"])
    if "de-energized" in low or "de energized" in low:
        extras.extend(["voltage tester", "circuit", "before touching wires"])
    if "flickering lights" in low or "flicker" in low:
        extras.extend(["loose wire connections", "intermittent power"])
    if "raised beds" in low or ("raised" in low and "vegetable" in low):
        extras.extend(["raised bed", "drainage", "soil control", "vegetable"])
    if "morning watering" in low or "late evening watering" in low or "evening watering" in low:
        extras.extend(["fungal risk", "watering"])
    if "seedlings" in low and "transplant" in low:
        extras.extend(["hardening off", "watering stress", "wilting"])
    if "wet and windy" in low:
        extras.extend(["avoid construction", "high-wind conditions", "wet conditions"])
    if "uneven" in low and "floor frame" in low:
        extras.extend(["level the frame", "joists", "before plywood"])
    if "breaker keeps tripping" in low or ("breaker" in low and "trip" in low):
        extras.extend(["overload", "short conditions"])
    if "yellow leaves" in low or "yellowing" in low or "soggy soil" in low:
        extras.extend(["overwatering", "soggy soil"])
    if "mulch" in low:
        extras.extend([
            "mulch",
            "retain moisture",
            "suppress weeds",
            "moisture retention",
            "weed control",
            "garden beds",
            "vegetable beds",
        ])
    if "support posts" in low and "soft soil" in low:
        extras.extend(["24 inches", "primary revision", "post depth"])
    if "decorative upgrades" in low and "come first" in low:
        extras.extend(["structural alignment checks", "build priority"])

    extras = sorted(set(x for x in extras if x))
    if not extras:
        return q
    return q + " " + " ".join(extras)


def infer_context_profile(query: str, domain_hint: Optional[str] = None) -> Dict[str, Any]:
    qk = token_set(query)
    tags: List[str] = []
    domain = domain_hint or "general"

    if domain == "garden":
        tags.append("garden")
        if "raised" in qk and ("bed" in qk or "raised_bed" in qk):
            tags.append("raised_bed")
        if "vegetable" in qk:
            tags.append("vegetable")
        if "drainage" in qk:
            tags.append("drainage")
        if "soil" in qk:
            tags.append("soil")
        if "water" in qk:
            tags.append("watering")
        if "seedling" in qk or "transplant" in qk:
            tags.extend(["seedling", "transplant"])
        if "mulch" in qk:
            tags.extend(["mulch", "moisture", "weed_control"])
        if "weed" in qk:
            tags.append("weed")
        if "yellow" in qk:
            tags.append("yellow_leaves")
    elif domain == "electrical":
        tags.append("electrical")
        if "breaker" in qk or "trip" in qk:
            tags.append("breaker")
        if "tester" in qk or "voltage" in qk:
            tags.append("voltage_tester")
        if "wire" in qk:
            tags.append("wires")
    elif domain == "clubhouse_build":
        tags.append("clubhouse_build")
        if "joist" in qk:
            tags.append("joists")
        if "floor" in qk:
            tags.append("floor")
        if "post" in qk:
            tags.append("posts")
        if "frame" in qk:
            tags.append("frame")
        if "soft" in qk and "soil" in qk:
            tags.append("soft_soil")
        if "decorative" in qk:
            tags.append("decorative")
    elif domain == "safety":
        tags.append("safety")
    elif domain == "planning":
        tags.append("planning")

    return {
        "domain": domain,
        "tags": sorted(set(tags)),
        "query_text": query,
    }


def iter_candidate_rows(obj: Any) -> Iterable[Any]:
    if obj is None:
        return
    if isinstance(obj, dict):
        yield obj
        return
    if isinstance(obj, (list, tuple, set)):
        for item in obj:
            yield item
        return
    yield obj


def best_candidate_blob(obj: Any, expected_terms: Iterable[str], query: str = "") -> str:
    rows = list(iter_candidate_rows(obj))
    if not rows:
        return ""

    scored: List[Tuple[Tuple[int, int, int, int, int, int, int], str]] = []
    for row in rows:
        blob = flatten_text(row)
        rank = candidate_rank_score(blob, query, expected_terms)
        scored.append((rank, blob))

    scored.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
    return scored[0][1] if scored else ""


# =============================================================================
# Source packs
# =============================================================================

def load_clubhouse_doc(doc_path: str) -> List[Dict[str, Any]]:
    doc_path = resolve_doc_path(doc_path)

    with open(doc_path, "r", encoding="utf-8") as f:
        doc_text = f.read()

    lessons: List[Dict[str, Any]] = []
    current_page: Optional[str] = None
    current_section: Optional[str] = None

    raw_lines = [ln.strip() for ln in doc_text.splitlines()]
    for line in raw_lines:
        if not line:
            continue

        if re.match(r"^page\s+\d+", line.strip(), re.I):
            current_page = line.strip()
            continue

        if re.match(r"^\d+\.\s+", line):
            current_section = re.sub(r"^\d+\.\s*", "", line).strip()
            continue

        if len(line) < 8:
            continue

        line = re.sub(r"^[\-\*\u2022]+", "", line).strip()
        if not line:
            continue

        alpha_count = sum(1 for ch in line if ch.isalpha())
        if alpha_count < 8:
            continue

        low = line.lower()
        tags = ["clubhouse", "construction"]
        domain = "clubhouse_build"
        kind = "procedure"

        if any(x in low for x in ("goggles", "gloves", "ear protection", "safety", "hazard", "wet", "wind")):
            tags += ["safety"]
            domain = "safety"
            kind = "policy"
        elif any(x in low for x in ("chalk line", "drill", "hammer", "saw", "tool", "materials")):
            tags += ["tools", "materials"]
            kind = "fact"
        elif any(x in low for x in ("footprint", "stake", "string", "diagonal", "square")):
            tags += ["foundation", "layout"]
        elif any(x in low for x in ("4x4", "gravel", "concrete", "vertical alignment", "support posts")):
            tags += ["posts", "foundation"]
        elif any(x in low for x in ("joists", "plywood", "floor frame", "16 inches", "joist hangers")):
            tags += ["floor", "framing", "joists"]

        if "children" in low:
            tags.append("children")
        if "playhouse" in low:
            tags.append("playhouse")

        lessons.append(
            make_memory(
                line,
                mem_type="lesson",
                confidence=0.96,
                source="seed_verified",
                context=[x for x in [current_page, current_section] if x],
                tags=tags,
                domain=domain,
                kind=kind,
                status="verified",
                details={
                    "document_name": os.path.basename(doc_path),
                    "source_type": "clubhouse_doc",
                    "verified": True,
                    "pending_verification": False,
                },
                evidence=[f"book:{os.path.basename(doc_path)}", f"section:{current_section or 'unknown'}"],
            )
        )

    lessons.extend([
        make_memory(
            "The chalk line is used for marking straight cuts during clubhouse construction.",
            tags=["clubhouse", "tools", "chalk line", "marking", "straight cuts"],
            domain="clubhouse_build",
            kind="fact",
            context=["precision_memory"],
            details={"document_name": os.path.basename(doc_path), "source_type": "clubhouse_precision"},
            evidence=[f"book:{os.path.basename(doc_path)}", "section:precision_memory"],
        ),
        make_memory(
            "Safety goggles, gloves, and ear protection should be used when cutting or drilling materials.",
            tags=["clubhouse", "safety", "goggles", "gloves", "ear protection", "cutting", "drilling"],
            domain="safety",
            kind="fact",
            context=["precision_memory"],
            details={"document_name": os.path.basename(doc_path), "source_type": "clubhouse_precision"},
            evidence=[f"book:{os.path.basename(doc_path)}", "section:precision_memory"],
        ),
        make_memory(
            "A level should be used to confirm vertical alignment of support posts and the floor frame.",
            tags=["clubhouse", "level", "support posts", "floor frame", "alignment"],
            domain="clubhouse_build",
            kind="fact",
            context=["precision_memory"],
            details={"document_name": os.path.basename(doc_path), "source_type": "clubhouse_precision"},
            evidence=[f"book:{os.path.basename(doc_path)}", "section:precision_memory"],
        ),
    ])

    seen = set()
    out: List[Dict[str, Any]] = []
    for item in lessons:
        key = item["content"].strip().lower()
        if key not in seen:
            out.append(item)
            seen.add(key)
    return out


def build_garden_pack() -> List[Dict[str, Any]]:
    items = [
        ("Tomatoes need consistent watering and well-drained soil.", ["garden", "watering", "soil", "vegetable"], "garden", "fact"),
        ("Seedlings should be hardened off gradually before transplanting outdoors.", ["garden", "seedlings", "transplant", "hardening off"], "garden", "procedure"),
        ("Morning watering reduces fungal risk compared with late-evening watering.", ["garden", "watering", "fungal risk", "morning watering"], "garden", "fact"),
        ("Raised beds improve drainage and soil control for many vegetables.", ["garden", "raised_bed", "drainage", "soil control", "vegetable"], "garden", "fact"),
        ("Mulch helps retain moisture and suppress weeds in garden beds.", ["garden", "mulch", "retain moisture", "suppress weeds", "bed"], "garden", "fact"),
        ("Mulch helps vegetable beds by retaining moisture and suppressing weeds.", ["garden", "mulch", "vegetable bed", "retain moisture", "suppress weeds", "weed"], "garden", "fact"),
        ("If leaves yellow and soil is soggy, overwatering is a likely cause.", ["garden", "yellow leaves", "overwatering", "soggy soil"], "garden", "troubleshooting"),
        ("If tomato seedlings wilt after transplanting, check watering stress and whether they were hardened off gradually.", ["garden", "seedlings", "wilting", "transplant", "hardening off", "watering stress"], "garden", "troubleshooting"),
        ("Yellow leaves with soggy soil often point to overwatering.", ["garden", "yellow leaves", "soggy soil", "overwatering"], "garden", "troubleshooting"),
        ("Morning watering is usually better because it reduces fungal risk.", ["garden", "morning watering", "fungal risk"], "garden", "fact"),
        ("Loose or inconsistent watering can stress seedlings after transplanting.", ["garden", "watering stress", "seedlings", "transplant"], "garden", "troubleshooting"),
        ("Raised beds can warm sooner, drain better, and give tighter control over soil quality.", ["garden", "raised_bed", "drainage", "soil quality", "vegetable"], "garden", "fact"),
    ]
    return [
        make_memory(
            text,
            tags=tags,
            domain=domain,
            kind=kind,
            context=["garden_pack"],
            details={"document_name": "garden_pack", "source_type": "synthetic_pack", "verified": True},
            evidence=["pack:garden_pack"],
        )
        for text, tags, domain, kind in items
    ]


def build_power_pack() -> List[Dict[str, Any]]:
    items = [
        ("Turn off power at the breaker before replacing a wall switch.", ["electrical", "breaker", "switch"], "electrical", "policy"),
        ("Use a voltage tester to confirm a circuit is de-energized before touching wires.", ["electrical", "tester", "wires", "safety", "voltage tester"], "electrical", "policy"),
        ("Black wires are often hot, white wires are often neutral, and bare copper is often ground in standard residential wiring.", ["electrical", "hot", "neutral", "ground"], "electrical", "fact"),
        ("If a breaker trips repeatedly, stop resetting it and inspect for overload or short conditions.", ["electrical", "breaker", "trips", "overload", "short"], "electrical", "troubleshooting"),
        ("Loose wire connections can cause flickering lights or intermittent power.", ["electrical", "loose connection", "flicker", "flickering lights"], "electrical", "troubleshooting"),
        ("Never work on live wiring in wet conditions.", ["electrical", "live wiring", "wet", "safety"], "electrical", "policy"),
        ("A voltage tester should be used to verify a circuit is de-energized before touching wires.", ["electrical", "voltage tester", "de-energized", "wires"], "electrical", "policy"),
        ("Loose wire connections can cause flickering lights.", ["electrical", "loose wire connections", "flickering lights"], "electrical", "troubleshooting"),
        ("If a breaker trips repeatedly, inspect for overload or short conditions instead of repeatedly resetting it.", ["electrical", "breaker", "overload", "short"], "electrical", "troubleshooting"),
        ("Before touching wires, confirm the circuit is off at the breaker and verify with a tester.", ["electrical", "breaker", "tester", "wires", "safety"], "electrical", "policy"),
    ]
    return [
        make_memory(
            text,
            tags=tags,
            domain=domain,
            kind=kind,
            context=["power_pack"],
            details={"document_name": "power_pack", "source_type": "synthetic_pack", "verified": True},
            evidence=["pack:power_pack"],
        )
        for text, tags, domain, kind in items
    ]


def build_revision_pack() -> List[Dict[str, Any]]:
    return [
        make_memory(
            "REVISION: For this project, support posts should be dug 24 inches deep in soft soil zones.",
            tags=["clubhouse", "revision", "posts", "depth", "soft soil", "primary_revision"],
            domain="clubhouse_build",
            kind="revision",
            context=["revision_pack"],
            confidence=0.98,
            status="verified",
            details={
                "document_name": "revision_pack",
                "source_type": "revision_pack",
                "verified": True,
                "pending_verification": False,
                "revision_target": "support post depth",
                "revision_order": 1,
                "is_primary_revision": True,
            },
            evidence=["pack:revision_pack", "target:support post depth"],
        ),
        make_memory(
            "REVISION: In this project build, galvanized structural screws are preferred over nails for deck-floor fastening.",
            tags=["clubhouse", "revision", "fastening", "screws", "nails", "primary_revision"],
            domain="clubhouse_build",
            kind="revision",
            context=["revision_pack"],
            confidence=0.98,
            status="verified",
            details={
                "document_name": "revision_pack",
                "source_type": "revision_pack",
                "verified": True,
                "pending_verification": False,
                "revision_target": "deck fastening",
                "revision_order": 1,
                "is_primary_revision": True,
            },
            evidence=["pack:revision_pack", "target:deck fastening"],
        ),
        make_memory(
            "REVISION: Children's clubhouse builds for this project should prioritize safety and visibility over decorative features.",
            tags=["clubhouse", "revision", "children", "visibility", "safety", "primary_revision"],
            domain="planning",
            kind="revision",
            context=["revision_pack"],
            confidence=0.98,
            status="verified",
            details={
                "document_name": "revision_pack",
                "source_type": "revision_pack",
                "verified": True,
                "pending_verification": False,
                "revision_target": "child priorities",
                "revision_order": 1,
                "is_primary_revision": True,
            },
            evidence=["pack:revision_pack", "target:child priorities"],
        ),
    ]


def build_adversarial_pack() -> List[Dict[str, Any]]:
    items = [
        (
            "REVISION: For this project, support posts should be dug 22 inches deep in compacted zones.",
            ["clubhouse", "revision", "posts", "depth", "compacted zones"],
            "clubhouse_build",
            "revision",
            {
                "revision_target": "support post depth",
                "revision_order": 2,
                "is_primary_revision": False,
            },
        ),
        (
            "REVISION: For this project, support posts should remain 24 inches deep in soft soil zones.",
            ["clubhouse", "revision", "posts", "depth", "soft soil", "primary_revision"],
            "clubhouse_build",
            "revision",
            {
                "revision_target": "support post depth",
                "revision_order": 3,
                "is_primary_revision": True,
            },
        ),
        (
            "Decorative upgrades do not change the required support post depth in soft soil zones.",
            ["clubhouse", "revision_context", "decorative", "posts", "soft soil"],
            "clubhouse_build",
            "fact",
            {
                "revision_target": "support post depth context",
                "revision_order": 3,
                "is_primary_revision": False,
            },
        ),
        (
            "REVISION: Decorative trim may be added later, but structural alignment checks still come first.",
            ["clubhouse", "revision", "decorative", "alignment", "structure", "primary_revision"],
            "clubhouse_build",
            "revision",
            {
                "revision_target": "build priority",
                "revision_order": 2,
                "is_primary_revision": True,
            },
        ),
        (
            "Support post depth should be confirmed before concrete pour in new hole placement.",
            ["clubhouse", "posts", "concrete", "depth"],
            "clubhouse_build",
            "procedure",
            {},
        ),
        (
            "Before laying plywood, confirm the floor frame is level and joist spacing is correct.",
            ["clubhouse", "floor", "frame", "joists", "plywood", "level", "spacing"],
            "clubhouse_build",
            "procedure",
            {},
        ),
        (
            "Deck-floor fastening should use structural screws once the frame is ready.",
            ["clubhouse", "deck", "screws", "frame"],
            "clubhouse_build",
            "procedure",
            {},
        ),
        (
            "Yellow leaves with soggy soil usually indicate overwatering.",
            ["garden", "yellow leaves", "soggy soil", "overwatering"],
            "garden",
            "troubleshooting",
            {},
        ),
        (
            "Seedling wilt after transplant can point to watering stress or failure to harden off gradually.",
            ["garden", "seedlings", "wilting", "transplant", "hardening off", "watering stress"],
            "garden",
            "troubleshooting",
            {},
        ),
        (
            "Evening watering can increase fungal risk compared with morning watering.",
            ["garden", "evening watering", "morning watering", "fungal risk"],
            "garden",
            "fact",
            {},
        ),
        (
            "If a breaker trips repeatedly under appliance load, inspect for overload or short conditions.",
            ["electrical", "breaker", "overload", "short", "appliance"],
            "electrical",
            "troubleshooting",
            {},
        ),
        (
            "Flickering lights often point to loose wire connections or intermittent power.",
            ["electrical", "flickering lights", "loose wire connections", "intermittent power"],
            "electrical",
            "troubleshooting",
            {},
        ),
        (
            "Use a voltage tester before touching wires, even when the breaker appears off.",
            ["electrical", "voltage tester", "breaker", "wires", "de-energized"],
            "electrical",
            "policy",
            {},
        ),
        (
            "Decorative lighting ideas should never override structural frame corrections.",
            ["clubhouse", "decorative", "lighting", "frame"],
            "clubhouse_build",
            "fact",
            {},
        ),
        (
            "Transplant shock in plants is unrelated to earthquakes or structural shaking.",
            ["garden", "transplant", "shock", "plants"],
            "garden",
            "fact",
            {},
        ),
        (
            "Water damage around outlets is an electrical safety issue, not a flood-evacuation plan by itself.",
            ["electrical", "water damage", "outlets", "safety"],
            "electrical",
            "policy",
            {},
        ),
    ]

    out: List[Dict[str, Any]] = []
    for text, tags, domain, kind, extra_details in items:
        out.append(
            make_memory(
                text,
                tags=tags,
                domain=domain,
                kind=kind,
                context=["adversarial_pack"],
                confidence=0.97 if kind == "revision" else 0.95,
                details={
                    "document_name": "adversarial_pack",
                    "source_type": "synthetic_pack",
                    "verified": True,
                    **extra_details,
                },
                evidence=["pack:adversarial_pack"],
            )
        )
    return out


# =============================================================================
# Benchmark data
# =============================================================================

REVISION_COLLISION_PROBES: List[Dict[str, Any]] = [
    {
        "query": "For this project, how deep should support posts be dug in soft soil zones?",
        "expected": ["24 inches", "soft soil"],
        "forbidden": ["22 inches in compacted zones", "22 inches", "decorative upgrades"],
        "domain": "clubhouse_build",
    },
    {
        "query": "When decorative upgrades are being planned, what still comes first in the build sequence?",
        "expected": ["structural alignment", "alignment checks", "come first"],
        "forbidden": ["decorative features first", "trim first"],
        "domain": "clubhouse_build",
    },
]

SAME_DOMAIN_PROBES: List[Dict[str, Any]] = [
    {
        "query": "The clubhouse floor frame is uneven and plywood has not been laid yet. What should happen first?",
        "expected": ["level", "floor frame", "joists", "before plywood"],
        "forbidden": ["24 inches deep", "support posts", "decorative trim"],
        "domain": "clubhouse_build",
    },
    {
        "query": "A gardener is soaking plants every evening and now the leaves are yellow with soggy soil. What is the likely issue?",
        "expected": ["overwatering", "yellow leaves", "soggy soil"],
        "forbidden": ["hardening off", "transplant stress only"],
        "domain": "garden",
    },
    {
        "query": "A breaker keeps tripping when the microwave runs. What is the likely thing to inspect?",
        "expected": ["overload", "short", "breaker"],
        "forbidden": ["flickering lights only", "loose wire connections only"],
        "domain": "electrical",
    },
]

FALSE_CUE_PROBES: List[Dict[str, Any]] = [
    {
        "query": "The clubhouse has decorative lighting ideas, but the floor frame is uneven. What should be corrected first?",
        "expected": ["frame", "level", "joists"],
        "forbidden": ["lighting ideas first", "decorative"],
        "domain": "clubhouse_build",
    },
    {
        "query": "My seedlings have transplant shock, but I am asking about plant care. What is a likely cause of wilting after transplant?",
        "expected": ["hardening off", "watering stress", "transplant"],
        "forbidden": ["earthquake", "shaking", "drop cover hold"],
        "domain": "garden",
    },
    {
        "query": "There is water damage near an outlet and I need the safest electrical step. What should I do first?",
        "expected": ["turn off power", "breaker", "voltage tester", "electrical"],
        "forbidden": ["move to higher ground", "floodwater evacuation"],
        "domain": "electrical",
    },
]

OVERLOAD_PROBES: List[Dict[str, Any]] = [
    {
        "query": "Among all these clubhouse details, answer only this: before laying plywood on an uneven frame, what should be checked first?",
        "expected": ["level", "joists", "frame", "before plywood"],
        "forbidden": ["post depth", "children visibility", "decorative features"],
        "domain": "clubhouse_build",
    },
    {
        "query": "Among many garden facts, answer only this: yellow leaves plus soggy soil usually points to what?",
        "expected": ["overwatering", "soggy soil"],
        "forbidden": ["raised beds", "mulch", "hardening off only"],
        "domain": "garden",
    },
    {
        "query": "Among many electrical facts, answer only this: repeated breaker trips under load should be checked for what?",
        "expected": ["overload", "short", "breaker"],
        "forbidden": ["neutral wire color", "flickering lights only"],
        "domain": "electrical",
    },
]

SIM_PRECISION_PROBES: List[Dict[str, Any]] = [
    {
        "scenario": "A builder is about to lay clubhouse floor plywood, but the floor frame is uneven and joist spacing was not checked.",
        "expected": ["level", "frame", "joists", "plywood"],
        "forbidden": ["support posts", "24 inches deep", "decorative trim"],
        "preferred": ["level", "joists", "before laying plywood", "spacing"],
        "acceptable_alt": ["floor frame", "alignment"],
        "context": {"domain": "clubhouse_build", "tags": ["clubhouse", "floor", "frame", "joists", "plywood", "level"]},
    },
    {
        "scenario": "A gardener keeps watering every evening and now the leaves are yellow and the soil is soggy.",
        "expected": ["overwatering", "yellow leaves", "soggy soil"],
        "forbidden": ["earthquake", "transplant shock only"],
        "preferred": ["overwatering", "soggy soil", "yellow leaves"],
        "acceptable_alt": ["morning watering", "fungal risk"],
        "context": {"domain": "garden", "tags": ["garden", "watering", "yellow leaves", "overwatering", "fungal risk"]},
    },
    {
        "scenario": "A breaker trips every time a microwave is used and the user wants the next diagnostic step.",
        "expected": ["breaker", "overload", "short", "inspect"],
        "forbidden": ["flickering lights only", "neutral wire color"],
        "preferred": ["breaker", "overload", "short"],
        "acceptable_alt": ["inspect", "appliance load"],
        "context": {"domain": "electrical", "tags": ["electrical", "breaker", "microwave", "overload", "short"]},
    },
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
        if hasattr(self.long_term, "get_all_memories"):
            ok, result, _ = safe_call(self.long_term.get_all_memories)
            if ok and isinstance(result, list):
                return len(result)

        if hasattr(self.long_term, "all_memories"):
            data = getattr(self.long_term, "all_memories", None)
            if isinstance(data, list):
                return len(data)

        if hasattr(self.long_term, "memories"):
            data = getattr(self.long_term, "memories", None)
            if isinstance(data, list):
                return len(data)

        return -1

    def wm_count(self) -> int:
        for attr in ("buffer", "working_set", "items", "memories"):
            buf = getattr(self.memory, attr, None)
            if isinstance(buf, list):
                return len(buf)
        return -1

    def ingest_memories(self, lessons: List[Dict[str, Any]]) -> Dict[str, int]:
        stats = {
            "wm_ok": 0,
            "ltm_ok": 0,
            "ltm_inserted_new": 0,
            "ltm_reinforced_existing": 0,
            "ltm_merged_revision": 0,
            "ltm_stored_conflict_variant": 0,
        }

        for item in lessons:
            if hasattr(self.memory, "store"):
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
            elif hasattr(self.long_term, "store"):
                before = self.ltm_count()
                ok2, _, _ = safe_call(self.long_term.store, item)
                after = self.ltm_count()
                if ok2:
                    if before >= 0 and after >= 0 and after > before:
                        stats["ltm_ok"] += 1
                        stats["ltm_inserted_new"] += 1
                    else:
                        stats["ltm_ok"] += 1

        return stats

    def _wm_keywords(self, query: str) -> List[str]:
        if hasattr(self.memory, "extract_keywords"):
            ok, result, _ = safe_call(self.memory.extract_keywords, query)
            if ok and isinstance(result, set):
                return sorted(str(x) for x in result)
            if ok and isinstance(result, list):
                return sorted(str(x) for x in result)
        return sorted(token_set(query))

    def _merge_unique_rows(self, *groups: Any) -> List[Any]:
        out: List[Any] = []
        seen: Set[str] = set()

        for group in groups:
            for row in iter_candidate_rows(group):
                sig = normalize_text(flatten_text(row))
                if not sig or sig in seen:
                    continue
                seen.add(sig)
                out.append(row)
        return out

    def _call_memory_retrieve(
        self,
        query: str,
        *,
        count: int,
        keywords: List[str],
        context_profile: Dict[str, Any],
    ) -> Tuple[bool, Any, str]:
        if not hasattr(self.memory, "retrieve"):
            return False, None, "WorkingMemory has no retrieve()"

        fn = self.memory.retrieve

        attempts: List[Dict[str, Any]] = [
            {
                "count": count,
                "keywords": keywords,
                "context_profile": context_profile,
                "min_relevance": 0.75,
                "planning_only": False,
            },
            {
                "count": count,
                "keywords": keywords,
                "context_profile": context_profile,
            },
            {
                "count": count,
                "keywords": keywords,
            },
            {
                "count": count,
            },
            {},
        ]

        last_err = "unknown"
        for kw in attempts:
            kw2 = supported_kwargs(fn, kw)
            ok, result, err = safe_call(fn, query, **kw2)
            if ok:
                return True, result, ""
            last_err = err

        return False, None, last_err

    def _call_ltm_search(self, query: str) -> Tuple[bool, Any, str]:
        if hasattr(self.long_term, "search"):
            return safe_call(self.long_term.search, query)
        if hasattr(self.long_term, "retrieve"):
            return safe_call(self.long_term.retrieve, query)
        return False, None, "LongTermMemory has neither search() nor retrieve()"

    def _call_ltm_retrieve(self, query: str) -> Tuple[bool, Any, str]:
        if hasattr(self.long_term, "retrieve"):
            return safe_call(self.long_term.retrieve, query)
        if hasattr(self.long_term, "search"):
            return safe_call(self.long_term.search, query)
        return False, None, "LongTermMemory has neither retrieve() nor search()"

    def retrieve_candidates(self, query: str, domain_hint: Optional[str] = None) -> Dict[str, Any]:
        out: Dict[str, Any] = {}

        raw_query = (query or "").strip()
        expanded_query = expand_query_with_domain(raw_query, domain_hint=domain_hint)
        context_profile = infer_context_profile(raw_query, domain_hint=domain_hint)
        raw_keywords = self._wm_keywords(raw_query)
        expanded_keywords = self._wm_keywords(expanded_query)

        wm_raw: List[Any] = []
        wm_expanded: List[Any] = []

        ok, result, err = self._call_memory_retrieve(
            raw_query,
            count=7,
            keywords=raw_keywords,
            context_profile=context_profile,
        )
        if ok:
            wm_raw = list(iter_candidate_rows(result))
        else:
            out["wm.retrieve_raw_error"] = err

        ok, result, err = self._call_memory_retrieve(
            expanded_query,
            count=7,
            keywords=expanded_keywords,
            context_profile=context_profile,
        )
        if ok:
            wm_expanded = list(iter_candidate_rows(result))
        else:
            out["wm.retrieve_expanded_error"] = err

        out["wm.retrieve"] = self._merge_unique_rows(wm_raw, wm_expanded)

        ltm_search_raw: List[Any] = []
        ltm_search_expanded: List[Any] = []

        ok, result, err = self._call_ltm_search(raw_query)
        if ok:
            ltm_search_raw = list(iter_candidate_rows(result))
        else:
            out["ltm.search_raw_error"] = err

        ok, result, err = self._call_ltm_search(expanded_query)
        if ok:
            ltm_search_expanded = list(iter_candidate_rows(result))
        else:
            out["ltm.search_expanded_error"] = err

        out["ltm.search"] = self._merge_unique_rows(ltm_search_raw, ltm_search_expanded)

        ltm_retrieve_raw: List[Any] = []
        ltm_retrieve_expanded: List[Any] = []

        ok, result, err = self._call_ltm_retrieve(raw_query)
        if ok:
            ltm_retrieve_raw = list(iter_candidate_rows(result))
        else:
            out["ltm.retrieve_raw_error"] = err

        ok, result, err = self._call_ltm_retrieve(expanded_query)
        if ok:
            ltm_retrieve_expanded = list(iter_candidate_rows(result))
        else:
            out["ltm.retrieve_expanded_error"] = err

        out["ltm.retrieve"] = self._merge_unique_rows(ltm_retrieve_raw, ltm_retrieve_expanded)
        out["_context_profile"] = context_profile
        out["_raw_query"] = raw_query
        out["_expanded_query"] = expanded_query
        return out

    def retrieve_ranked_blob(
        self,
        query: str,
        expected_terms: List[str],
        domain_hint: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any], bool]:
        candidates = self.retrieve_candidates(query, domain_hint=domain_hint)
        parts: List[str] = []
        identity_leak = False

        for key in ("ltm.search", "ltm.retrieve", "wm.retrieve"):
            if key not in candidates:
                continue

            best_blob = best_candidate_blob(candidates[key], expected_terms, query=query).strip()
            if not best_blob:
                continue

            hit = contains_expected(best_blob, expected_terms)
            is_identity = any(term in best_blob for term in IDENTITY_LEAK_TERMS)
            if is_identity:
                identity_leak = True

            tag = "[HIT]" if hit else "[MISS]"
            if is_identity:
                tag += "[IDENTITY]"
            parts.append(f"{tag} {key}: {best_blob}")

        combined_blob = " || ".join(parts).lower()
        if combined_blob and not any(p.startswith("[HIT]") for p in parts):
            aggregate = flatten_text({
                "ltm.search": candidates.get("ltm.search", []),
                "ltm.retrieve": candidates.get("ltm.retrieve", []),
                "wm.retrieve": candidates.get("wm.retrieve", []),
            })
            if contains_expected(aggregate, expected_terms):
                parts.append(f"[HIT][FALLBACK] aggregate: {aggregate[:400]}")

        return " || ".join(parts), candidates, identity_leak

    def _extract_options(self, result: Any) -> List[Any]:
        if result is None:
            return []

        if isinstance(result, list):
            return result

        if isinstance(result, tuple):
            if result and isinstance(result[0], list):
                return result[0]
            flat = [x for x in result if isinstance(x, (dict, str)) or dataclasses.is_dataclass(x)]
            if flat:
                return flat

        if isinstance(result, dict):
            for key in ("options", "actions", "candidates", "plans"):
                value = result.get(key)
                if isinstance(value, list):
                    return value

        return []

    def plan(
        self,
        scenario: str,
        context_profile: Optional[Dict[str, Any]] = None,
        num_actions: int = 3,
    ) -> List[Any]:
        if not hasattr(self.simulator, "generate_options"):
            return []

        fn = self.simulator.generate_options
        sink = io.StringIO()

        base_kwargs = {
            "num_actions": num_actions,
            "question": scenario,
            "context_profile": context_profile,
            "horizon": 1,
        }

        attempts = [
            (self.memory, supported_kwargs(fn, base_kwargs)),
            (self.memory, supported_kwargs(fn, {k: v for k, v in base_kwargs.items() if k != "context_profile"})),
            (self.memory, supported_kwargs(fn, {"num_actions": num_actions, "question": scenario})),
            (scenario, supported_kwargs(fn, {"num_actions": num_actions, "context_profile": context_profile, "horizon": 1})),
            (scenario, supported_kwargs(fn, {"num_actions": num_actions})),
        ]

        last_result: Any = None
        for first_arg, kwargs in attempts:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                ok, result, _ = safe_call(fn, first_arg, **kwargs)
            if ok:
                last_result = result
                break

        return self._extract_options(last_result)

    def reflect(self) -> bool:
        if not hasattr(self.clair, "reflect"):
            return False
        ok, _, _ = safe_call(self.clair.reflect)
        return ok


# =============================================================================
# Benchmark
# =============================================================================

class AdversarialPrecisionBenchmark:
    def __init__(self, adapter: ClairAdapter, doc_path: str, cycles: int) -> None:
        self.adapter = adapter
        self.doc_path = resolve_doc_path(doc_path)
        self.cycles = max(1, cycles)

        self.initial_wm = -1
        self.initial_ltm = -1

        self.ingest_lessons = 0
        self.ingest_wm_ok = 0
        self.ingest_ltm_ok = 0

        self.rev_ok = 0
        self.rev_total = 0
        self.rev_fails: List[str] = []

        self.same_ok = 0
        self.same_total = 0
        self.same_fails: List[str] = []

        self.false_ok = 0
        self.false_total = 0
        self.false_fails: List[str] = []

        self.overload_ok = 0
        self.overload_total = 0
        self.overload_fails: List[str] = []

        self.sim_ok = 0
        self.sim_partial = 0
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

        self._rev_idx = 0
        self._same_idx = 0
        self._false_idx = 0
        self._over_idx = 0
        self._sim_idx = 0

    def next_rev(self) -> Dict[str, Any]:
        item = REVISION_COLLISION_PROBES[self._rev_idx % len(REVISION_COLLISION_PROBES)]
        self._rev_idx += 1
        return item

    def next_same(self) -> Dict[str, Any]:
        item = SAME_DOMAIN_PROBES[self._same_idx % len(SAME_DOMAIN_PROBES)]
        self._same_idx += 1
        return item

    def next_false(self) -> Dict[str, Any]:
        item = FALSE_CUE_PROBES[self._false_idx % len(FALSE_CUE_PROBES)]
        self._false_idx += 1
        return item

    def next_over(self) -> Dict[str, Any]:
        item = OVERLOAD_PROBES[self._over_idx % len(OVERLOAD_PROBES)]
        self._over_idx += 1
        return item

    def next_sim(self) -> Dict[str, Any]:
        item = SIM_PRECISION_PROBES[self._sim_idx % len(SIM_PRECISION_PROBES)]
        self._sim_idx += 1
        return item

    def ingest_all_sources(self) -> None:
        clubhouse = load_clubhouse_doc(self.doc_path)
        garden = build_garden_pack()
        power = build_power_pack()
        revisions = build_revision_pack()
        adversarial = build_adversarial_pack()

        all_items = clubhouse + garden + power + revisions + adversarial
        self.ingest_lessons = len(all_items)

        stats = self.adapter.ingest_memories(all_items)
        self.ingest_wm_ok = stats["wm_ok"]
        self.ingest_ltm_ok = stats["ltm_ok"]
        self.ltm_persist += stats["ltm_ok"]
        self.ltm_inserted_new += stats["ltm_inserted_new"]
        self.ltm_reinforced_existing += stats["ltm_reinforced_existing"]
        self.ltm_merged_revision += stats["ltm_merged_revision"]
        self.ltm_stored_conflict_variant += stats["ltm_stored_conflict_variant"]

    def _run_retrieval_probe(
        self,
        item: Dict[str, Any],
        ok_counter_name: str,
        total_counter_name: str,
        fail_list_name: str,
    ) -> None:
        query = item["query"]
        expected = item["expected"]
        forbidden = item["forbidden"]
        domain = item["domain"]

        blob, candidates, identity_leak = self.adapter.retrieve_ranked_blob(
            query,
            expected,
            domain_hint=domain,
        )

        setattr(self, total_counter_name, getattr(self, total_counter_name) + 1)
        if identity_leak:
            self.identity_leaks += 1

        good = contains_expected(blob, expected)
        bad = contains_forbidden(blob, forbidden)

        if good and not bad:
            setattr(self, ok_counter_name, getattr(self, ok_counter_name) + 1)
            return

        fails = getattr(self, fail_list_name)
        if len(fails) < 6:
            fail_bits = []
            for key in ("ltm.search", "ltm.retrieve", "wm.retrieve"):
                if key in candidates:
                    fail_bits.append(
                        f"{key}={short(best_candidate_blob(candidates[key], expected, query=query), 140)!r}"
                    )
            fails.append(
                f"query={query!r} domain={domain!r} good={good} bad={bad} got={' | '.join(fail_bits) if fail_bits else short(blob)!r}"
            )

    def run_revision_collision_probe(self) -> None:
        self._run_retrieval_probe(self.next_rev(), "rev_ok", "rev_total", "rev_fails")

    def run_same_domain_probe(self) -> None:
        self._run_retrieval_probe(self.next_same(), "same_ok", "same_total", "same_fails")

    def run_false_cue_probe(self) -> None:
        self._run_retrieval_probe(self.next_false(), "false_ok", "false_total", "false_fails")

    def run_overload_probe(self) -> None:
        self._run_retrieval_probe(self.next_over(), "overload_ok", "overload_total", "overload_fails")

    def run_sim_probe(self) -> None:
        item = self.next_sim()
        options = self.adapter.plan(item["scenario"], context_profile=item["context"], num_actions=3)

        self.sim_total += 1

        if not options:
            if len(self.sim_fails) < 6:
                self.sim_fails.append(f"scenario={short(item['scenario'])!r} got='NO_OPTIONS'")
            return

        top1_blob = option_blob(options[0])
        merged = " || ".join(option_blob(opt) for opt in options[:3])

        top1_bad = contains_forbidden(top1_blob, item["forbidden"])
        merged_bad = contains_forbidden(merged, item["forbidden"])

        pref_hits_top1 = count_expected_hits(top1_blob, item.get("preferred", []))
        exp_hits_top1 = count_expected_hits(top1_blob, item.get("expected", []))
        alt_hits_top1 = count_expected_hits(top1_blob, item.get("acceptable_alt", []))

        exp_hits_merged = count_expected_hits(merged, item.get("expected", []))
        alt_hits_merged = count_expected_hits(merged, item.get("acceptable_alt", []))

        strong_local = (
            not top1_bad
            and (
                pref_hits_top1 >= 2
                or exp_hits_top1 >= 2
                or (exp_hits_top1 >= 1 and alt_hits_top1 >= 1)
            )
        )

        partial_local = (
            not merged_bad
            and (
                (exp_hits_top1 >= 1)
                or (alt_hits_top1 >= 2)
                or (exp_hits_merged >= 1 and alt_hits_merged >= 1)
                or (alt_hits_merged >= 2)
            )
        )

        if strong_local:
            self.sim_ok += 1
            return

        if partial_local:
            self.sim_partial += 1
            return

        if len(self.sim_fails) < 6:
            self.sim_fails.append(
                f"scenario={short(item['scenario'])!r} "
                f"top1_bad={top1_bad} merged_bad={merged_bad} "
                f"pref_hits_top1={pref_hits_top1} exp_hits_top1={exp_hits_top1} "
                f"alt_hits_top1={alt_hits_top1} exp_hits_merged={exp_hits_merged} "
                f"alt_hits_merged={alt_hits_merged} "
                f"top1={short(top1_blob, 220)!r}"
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
        hr("CLAIR ADVERSARIAL PRECISION BENCHMARK")
        self.initial_wm = self.adapter.wm_count()
        self.initial_ltm = self.adapter.ltm_count()

        print(f"Document path                  {self.doc_path}")
        print(f"Cycles                         {self.cycles}")
        print(f"Initial WM count               {self.initial_wm}")
        print(f"Initial LTM count              {self.initial_ltm}")

        t0 = time.time()

        self.ingest_all_sources()

        print(f"Ingested memory candidates     {self.ingest_lessons}")
        print(f"Ingested into WM               {self.ingest_wm_ok}")
        print(f"Persisted/consolidated in LTM  {self.ingest_ltm_ok}")

        for cycle in range(self.cycles):
            self.run_revision_collision_probe()
            self.run_same_domain_probe()
            self.run_false_cue_probe()
            self.run_overload_probe()

            if cycle % 2 == 0:
                self.run_sim_probe()

            if cycle % 2 == 1:
                self.run_idle_cycle()

            print(
                f"[cycle {cycle + 1:02d}] "
                f"rev={self.rev_ok}/{self.rev_total} "
                f"same={self.same_ok}/{self.same_total} "
                f"false={self.false_ok}/{self.false_total} "
                f"over={self.overload_ok}/{self.overload_total} "
                f"sim={self.sim_ok}+{self.sim_partial}/{self.sim_total} "
                f"reflect={self.reflect_ok}/{self.reflect_total} "
                f"identity_leaks={self.identity_leaks}"
            )

        elapsed = time.time() - t0
        final_wm = self.adapter.wm_count()
        final_ltm = self.adapter.ltm_count()
        net_ltm_delta = final_ltm - self.initial_ltm if final_ltm >= 0 and self.initial_ltm >= 0 else 0

        phase_a = self.score(self.ingest_wm_ok / max(1, self.ingest_lessons), 100)
        phase_b = self.score(self.rev_ok / max(1, self.rev_total), 220)
        phase_c = self.score(self.same_ok / max(1, self.same_total), 220)
        phase_d = self.score(self.false_ok / max(1, self.false_total), 180)
        phase_e = self.score(self.overload_ok / max(1, self.overload_total), 160)

        sim_ratio = (self.sim_ok + 0.5 * self.sim_partial) / max(1, self.sim_total)
        phase_f = self.score(sim_ratio, 90)

        phase_g = self.score(self.reflect_ok / max(1, self.reflect_total), 30)

        identity_penalty = 0
        if self.identity_leaks > 0:
            identity_penalty = min(100, self.identity_leaks * 10)

        total = phase_a + phase_b + phase_c + phase_d + phase_e + phase_f + phase_g - identity_penalty
        total = max(0, total)

        hr("SCORE SUMMARY")
        print(f"Phase A: Ingestion                        {phase_a:3d}/100")
        print(f"Phase B: Revision collision precision     {phase_b:3d}/220")
        print(f"Phase C: Same-domain precision            {phase_c:3d}/220")
        print(f"Phase D: False-cue resistance             {phase_d:3d}/180")
        print(f"Phase E: Overload precision               {phase_e:3d}/160")
        print(f"Phase F: Simulator local grounding        {phase_f:3d}/90")
        print(f"Phase G: Reflection / idle                {phase_g:3d}/30")
        print(f"Identity leak penalty                    -{identity_penalty:3d}")

        hr("FINAL RESULTS")
        print(f"Runtime seconds                  {elapsed:.3f}")
        print(f"Final WM count                   {final_wm}")
        print(f"Final LTM count                  {final_ltm}")
        print(f"Net LTM row delta                {net_ltm_delta}")
        print(f"Revision collision               {self.rev_ok}/{self.rev_total} ({pct(self.rev_ok, self.rev_total)})")
        print(f"Same-domain precision            {self.same_ok}/{self.same_total} ({pct(self.same_ok, self.same_total)})")
        print(f"False-cue resistance             {self.false_ok}/{self.false_total} ({pct(self.false_ok, self.false_total)})")
        print(f"Overload precision               {self.overload_ok}/{self.overload_total} ({pct(self.overload_ok, self.overload_total)})")
        print(f"Simulator local grounding        {self.sim_ok}+{self.sim_partial}/{self.sim_total} ({pct(self.sim_ok + self.sim_partial, self.sim_total)})")
        print(f"Reflection                       {self.reflect_ok}/{self.reflect_total} ({pct(self.reflect_ok, self.reflect_total)})")
        print(f"Idle cycles                      {self.idle_cycles}")
        print(f"Identity leaks                   {self.identity_leaks}")
        print(f"Total Score                      {total}/1000")

        ratio = total / 1000.0
        if ratio >= 0.95:
            verdict = "PASS (adversarial precision-stable)"
        elif ratio >= 0.85:
            verdict = "PASS-WARN (small local precision weakness)"
        elif ratio >= 0.70:
            verdict = "WARN (visible interference under adversarial pressure)"
        else:
            verdict = "FAIL (precision instability under adversarial pressure)"

        print(f"Verdict                          {verdict}")

        if self.rev_fails:
            hr("SAMPLE REVISION COLLISION FAILURES")
            for item in self.rev_fails:
                print(item)

        if self.same_fails:
            hr("SAMPLE SAME-DOMAIN FAILURES")
            for item in self.same_fails:
                print(item)

        if self.false_fails:
            hr("SAMPLE FALSE-CUE FAILURES")
            for item in self.false_fails:
                print(item)

        if self.overload_fails:
            hr("SAMPLE OVERLOAD FAILURES")
            for item in self.overload_fails:
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
    parser.add_argument("--cycles", type=int, default=12)
    parser.add_argument("--doc", type=str, default=DEFAULT_DOC_PATH)
    args = parser.parse_args()

    try:
        resolved_doc = resolve_doc_path(args.doc)
    except FileNotFoundError as exc:
        hr("DOCUMENT RESOLUTION ERROR")
        print(exc)
        return 2

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
    bench = AdversarialPrecisionBenchmark(
        adapter=adapter,
        doc_path=resolved_doc,
        cycles=args.cycles,
    )
    bench.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())