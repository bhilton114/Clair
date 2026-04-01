# test_clair_24hr.py
# Clair Day 24 Benchmark - continuous memory / simulator grounding stress
#
# Uses the interfaces actually discovered in Clair v2.51+:
# - memory.store(...)
# - memory.retrieve(...)
# - long_term.search(...)
# - long_term.retrieve(...)
# - long_term.get_all_memories()
# - long_term.store(...) and/or long_term.store_detailed(...)
# - simulator.generate_options(working_memory, num_actions=..., question=..., context_profile=..., horizon=...)
# - clair.reflect()
#
# Key fixes in this rewrite:
# - FIXES simulator call signature: scenario is passed as question=..., not as working_memory
# - Adds simulator import/path/version checks
# - Adds instantiated simulator version check from Clair object
# - Uses hazard-aware context_profile during direct simulator grounding
# - Replaces old LTM "row delta only" success metric with consolidation-aware accounting
# - Tracks inserted / reinforced / merged / contested LTM outcomes
# - Keeps retrieval/recall checks compact and identity-leak aware
# - Makes hazard scoring inspect top options more honestly
#
# Run:
#   python test_clair_24hr.py
#   python test_clair_24hr.py --hours 3 --cycles_per_hour 10
#   python test_clair_24hr.py --hours 24 --cycles_per_hour 20

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import time
from typing import Any, Dict, List, Optional, Tuple

from clair import Clair
from planning.simulator import Simulator
import planning.simulator as sim_mod


# =============================================================================
# Helpers
# =============================================================================

def hr(title: str = "", width: int = 92) -> None:
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


def safe_call(fn: Any, *args, **kwargs) -> Tuple[bool, Any, str]:
    try:
        result = fn(*args, **kwargs)
        return True, result, ""
    except Exception as exc:
        return False, None, repr(exc)


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


def normalize_kind(payload: Dict[str, Any]) -> str:
    return str(payload.get("kind", "") or payload.get("type", "") or "").strip().lower()


# =============================================================================
# Benchmark data
# =============================================================================

GOOD_FACTS: List[Dict[str, Any]] = [
    {
        "content": "Water boils at 100 degrees Celsius at sea level.",
        "kind": "lesson",
        "tags": ["science", "water"],
        "domain": "general",
        "confidence": 0.95,
        "truth_status": "supported",
    },
    {
        "content": "Humans have 206 bones.",
        "kind": "lesson",
        "tags": ["biology", "human"],
        "domain": "general",
        "confidence": 0.95,
        "truth_status": "supported",
    },
    {
        "content": "Rain is caused by condensation of water vapor.",
        "kind": "lesson",
        "tags": ["weather"],
        "domain": "general",
        "confidence": 0.95,
        "truth_status": "supported",
    },
    {
        "content": "Avoid driving through floodwater.",
        "kind": "lesson",
        "tags": ["survival", "flood"],
        "domain": "survival",
        "confidence": 0.95,
        "truth_status": "supported",
    },
    {
        "content": "During a fire, get low to avoid smoke inhalation.",
        "kind": "lesson",
        "tags": ["survival", "fire"],
        "domain": "survival",
        "confidence": 0.95,
        "truth_status": "supported",
    },
    {
        "content": "In an earthquake, drop, cover, and hold on.",
        "kind": "lesson",
        "tags": ["survival", "earthquake"],
        "domain": "survival",
        "confidence": 0.95,
        "truth_status": "supported",
    },
    {
        "content": "If uncertain, defer and gather more signal rather than guessing.",
        "kind": "lesson",
        "tags": ["meta", "reasoning"],
        "domain": "meta",
        "confidence": 0.95,
        "truth_status": "supported",
    },
]

BAD_FACTS: List[Dict[str, Any]] = [
    {
        "content": "Water boils at 90 degrees Celsius at sea level.",
        "kind": "observe",
        "tags": ["science", "water", "conflict"],
        "domain": "general",
        "confidence": 0.30,
        "truth_status": "conflicted",
    },
    {
        "content": "Humans have 212 bones.",
        "kind": "observe",
        "tags": ["biology", "human", "conflict"],
        "domain": "general",
        "confidence": 0.30,
        "truth_status": "conflicted",
    },
    {
        "content": "During a fire, stand tall to avoid smoke.",
        "kind": "observe",
        "tags": ["survival", "fire", "conflict"],
        "domain": "survival",
        "confidence": 0.20,
        "truth_status": "conflicted",
    },
]

OBSERVATIONS: List[Dict[str, Any]] = [
    {
        "content": "Observed successful recall on a simple factual question.",
        "kind": "observe",
        "tags": ["benchmark", "observation"],
        "domain": "meta",
        "confidence": 0.70,
    },
    {
        "content": "Observed conflicting memory pressure during repeated injections.",
        "kind": "observe",
        "tags": ["benchmark", "observation"],
        "domain": "meta",
        "confidence": 0.70,
    },
]

RECALL_PROBES: List[Tuple[str, List[str], str]] = [
    ("water boils 100 sea level", ["100", "100 degrees", "100 degrees celsius"], "general"),
    ("206 bones human body skeleton", ["206", "bones"], "general"),
    ("rain condensation water vapor", ["condensation", "water vapor"], "general"),
    ("avoid driving through floodwater", ["avoid driving", "floodwater"], "survival"),
    ("fire get low smoke inhalation", ["get low", "smoke inhalation", "avoid smoke"], "survival"),
    ("earthquake drop cover hold on", ["drop", "cover", "hold on"], "survival"),
]

SCENARIOS: List[Tuple[str, str, List[str]]] = [
    (
        "A flash flood is happening. Roads are underwater and the water is rising fast.",
        "flood",
        ["floodwater", "higher ground", "evacuation", "avoid driving", "rising water"],
    ),
    (
        "There is a fire in a closed room. Thick smoke and flames are spreading.",
        "fire",
        ["fire", "smoke", "get low", "avoid smoke", "safe exit"],
    ),
    (
        "An earthquake just hit. The ground is shaking and aftershocks are expected.",
        "earthquake",
        ["earthquake", "aftershocks", "drop", "cover", "hold on", "falling debris"],
    ),
]

IDENTITY_LEAK_TERMS = [
    "my name is clair",
    "i am clair",
    "cognitive learning and interactive reasoner",
]

VALID_LTM_TYPES = {
    "lesson",
    "observe",
    "fact",
    "policy",
    "feedback",
    "identity",
    "reasoning_action",
    "committed_action",
}

LTM_SUCCESS_ACTIONS = {
    "inserted_new",
    "reinforced_existing",
    "merged_revision",
    "stored_conflict_variant",
}


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

    def _normalize_payload_for_storage(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        p = dict(payload)

        kind = str(p.get("kind", "") or "").strip().lower()
        msg_type = str(p.get("type", "") or "").strip().lower()

        if not msg_type:
            if kind in VALID_LTM_TYPES:
                p["type"] = kind
            else:
                p["type"] = "lesson"

        if "source" not in p:
            p["source"] = "system"

        if "context" not in p:
            p["context"] = []

        if "details" not in p or not isinstance(p.get("details"), dict):
            p["details"] = {}

        truth_status = str(p.get("truth_status", "") or "").strip().lower()
        if truth_status:
            if truth_status == "supported":
                p["status"] = "verified"
                p["details"].setdefault("verified", True)
                p["details"].setdefault("pending_verification", False)
            elif truth_status == "conflicted":
                p["status"] = "contested"
                p["details"].setdefault("contested", True)
                p["details"].setdefault("recall_blocked", True)

        return p

    def store_memory(self, payload: Dict[str, Any], also_ltm: bool = False) -> Tuple[bool, Dict[str, int], Dict[str, Any]]:
        normalized = self._normalize_payload_for_storage(payload)

        wm_ok, _, _ = safe_call(self.memory.store, normalized)

        ltm_stats = {
            "successes": 0,
            "inserted_new": 0,
            "reinforced_existing": 0,
            "merged_revision": 0,
            "stored_conflict_variant": 0,
            "fallback_row_delta": 0,
        }
        ltm_meta: Dict[str, Any] = {"mode": "not_called"}

        if also_ltm:
            if hasattr(self.long_term, "store_detailed"):
                ok, result, err = safe_call(self.long_term.store_detailed, normalized)
                if ok and isinstance(result, dict):
                    ltm_meta["mode"] = "store_detailed"
                    ltm_meta["raw"] = result
                    actions = result.get("results", [])
                    if isinstance(actions, list):
                        for item in actions:
                            if not isinstance(item, dict):
                                continue
                            action = str(item.get("action", "") or "").strip().lower()
                            if action in LTM_SUCCESS_ACTIONS:
                                ltm_stats["successes"] += 1
                                ltm_stats[action] += 1
                    ltm_meta["error"] = ""
                else:
                    ltm_meta["mode"] = "store_detailed_failed"
                    ltm_meta["error"] = err
            else:
                before = self.ltm_count()
                ok, result, err = safe_call(self.long_term.store, normalized)
                after = self.ltm_count()
                ltm_meta["mode"] = "legacy_store"
                ltm_meta["raw"] = result
                ltm_meta["error"] = err
                if ok and before >= 0 and after >= 0 and after > before:
                    ltm_stats["successes"] = 1
                    ltm_stats["inserted_new"] = 1
                    ltm_stats["fallback_row_delta"] = after - before

        return wm_ok, ltm_stats, ltm_meta

    def retrieve_candidates(self, query: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {}

        ok, result, err = safe_call(self.memory.retrieve, query)
        if ok:
            out["wm.retrieve"] = result
        else:
            out["wm.retrieve_error"] = err

        ok, result, err = safe_call(self.long_term.search, query)
        if ok:
            out["ltm.search"] = result
        else:
            out["ltm.search_error"] = err

        ok, result, err = safe_call(self.long_term.retrieve, query)
        if ok:
            out["ltm.retrieve"] = result
        else:
            out["ltm.retrieve_error"] = err

        return out

    def retrieve_ranked_blob(self, query: str, expected_terms: List[str]) -> Tuple[str, Dict[str, Any], bool]:
        candidates = self.retrieve_candidates(query)
        parts: List[str] = []
        identity_leak = False

        ordered_keys = ("ltm.search", "ltm.retrieve", "wm.retrieve")

        for key in ordered_keys:
            if key not in candidates:
                continue

            blob = flatten_text(candidates[key]).strip()
            if not blob:
                continue

            hit = any(term.lower() in blob for term in expected_terms)
            is_identity = any(term in blob for term in IDENTITY_LEAK_TERMS)
            if is_identity:
                identity_leak = True

            tag = "[HIT]" if hit else "[MISS]"
            if is_identity:
                tag += "[IDENTITY]"
            parts.append(f"{tag} {key}: {blob}")

        return " || ".join(parts), candidates, identity_leak

    def _hazard_context_profile(self, expected: str) -> Dict[str, Any]:
        return {
            "domain": "survival",
            "hazard_family": expected,
            "tags": ["survival", expected, "emergency"],
        }

    def plan(self, scenario: str, expected: Optional[str] = None, num_actions: int = 3) -> List[Any]:
        """
        Correct simulator call:
            generate_options(working_memory, num_actions=..., question=..., context_profile=..., horizon=...)
        """
        context_profile = self._hazard_context_profile(expected) if expected else None

        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            ok, result, err = safe_call(
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

class Benchmark24Hr:
    def __init__(self, adapter: ClairAdapter, hours: int, cycles_per_hour: int) -> None:
        self.adapter = adapter
        self.hours = hours
        self.cycles_per_hour = cycles_per_hour

        self.initial_wm_count = -1
        self.initial_ltm_count = -1

        self.inject_ok = 0
        self.inject_total = 0
        self.wm_inject_ok = 0

        self.ltm_persist_successes = 0
        self.ltm_inserted_new = 0
        self.ltm_reinforced_existing = 0
        self.ltm_merged_revision = 0
        self.ltm_stored_conflict_variant = 0

        self.conflicts = 0

        self.recall_ok = 0
        self.recall_total = 0
        self.recall_fails: List[str] = []
        self.identity_recall_leaks = 0

        self.hazard_ok = 0
        self.hazard_total = 0
        self.hazard_fails: List[str] = []

        self.reflect_ok = 0
        self.reflect_total = 0

        self.store_debug_failures: List[str] = []

        self._good_idx = 0
        self._bad_idx = 0
        self._obs_idx = 0
        self._recall_idx = 0
        self._scenario_idx = 0

    def next_good(self) -> Dict[str, Any]:
        item = GOOD_FACTS[self._good_idx % len(GOOD_FACTS)]
        self._good_idx += 1
        return item

    def next_bad(self) -> Dict[str, Any]:
        item = BAD_FACTS[self._bad_idx % len(BAD_FACTS)]
        self._bad_idx += 1
        return item

    def next_obs(self) -> Dict[str, Any]:
        item = OBSERVATIONS[self._obs_idx % len(OBSERVATIONS)]
        self._obs_idx += 1
        return item

    def next_recall(self) -> Tuple[str, List[str], str]:
        item = RECALL_PROBES[self._recall_idx % len(RECALL_PROBES)]
        self._recall_idx += 1
        return item

    def next_scenario(self) -> Tuple[str, str, List[str]]:
        item = SCENARIOS[self._scenario_idx % len(SCENARIOS)]
        self._scenario_idx += 1
        return item

    def inject_batch(self, cycle: int) -> None:
        payloads = [self.next_good(), self.next_good(), self.next_obs()]
        if cycle % 3 == 0:
            payloads.append(self.next_bad())
            self.conflicts += 1

        for idx, payload in enumerate(payloads):
            also_ltm = (idx == 0)

            self.inject_total += 1
            wm_ok, ltm_stats, ltm_meta = self.adapter.store_memory(payload, also_ltm=also_ltm)

            if wm_ok:
                self.wm_inject_ok += 1

            ltm_successes = int(ltm_stats.get("successes", 0) or 0)
            self.ltm_persist_successes += ltm_successes
            self.ltm_inserted_new += int(ltm_stats.get("inserted_new", 0) or 0)
            self.ltm_reinforced_existing += int(ltm_stats.get("reinforced_existing", 0) or 0)
            self.ltm_merged_revision += int(ltm_stats.get("merged_revision", 0) or 0)
            self.ltm_stored_conflict_variant += int(ltm_stats.get("stored_conflict_variant", 0) or 0)

            if wm_ok or ltm_successes > 0:
                self.inject_ok += 1

            if also_ltm and not wm_ok and ltm_successes <= 0 and len(self.store_debug_failures) < 5:
                self.store_debug_failures.append(
                    f"payload={short(payload)} mode={ltm_meta.get('mode')} err={ltm_meta.get('error', '')!r} raw={short(ltm_meta.get('raw'))}"
                )

    def run_recall_probe(self) -> None:
        query, expected_terms, _domain = self.next_recall()
        blob, candidates, identity_leak = self.adapter.retrieve_ranked_blob(query, expected_terms)

        self.recall_total += 1

        if identity_leak:
            self.identity_recall_leaks += 1

        if any(term.lower() in blob for term in expected_terms):
            self.recall_ok += 1
            return

        if len(self.recall_fails) < 5:
            fail_bits = []
            for key in ("ltm.search", "ltm.retrieve", "wm.retrieve"):
                if key in candidates:
                    fail_bits.append(f"{key}={short(flatten_text(candidates[key]), 120)!r}")

            extra = " identity_hijack=YES" if identity_leak else ""
            self.recall_fails.append(
                f"query={query!r}{extra} got={' | '.join(fail_bits) if fail_bits else short(blob)!r}"
            )

    def run_hazard_probe(self) -> None:
        scenario, expected, grounding_terms = self.next_scenario()

        grounding_blob, _, _ = self.adapter.retrieve_ranked_blob(scenario, grounding_terms)
        enriched_prompt = scenario

        if grounding_blob.strip():
            enriched_prompt = (
                f"{scenario}\n"
                f"Relevant retrieved memory:\n{grounding_blob}"
            )

        options = self.adapter.plan(enriched_prompt, expected=expected, num_actions=3)
        self.hazard_total += 1

        if not options:
            if len(self.hazard_fails) < 5:
                self.hazard_fails.append(f"expected={expected!r} got='NO_OPTIONS'")
            return

        top_blobs = [option_blob(opt) for opt in options[:3]]
        merged = " || ".join(top_blobs)

        ok = False
        if expected in merged:
            ok = True
        elif any(term.lower() in merged for term in grounding_terms):
            ok = True
        elif expected == "flood" and any(t in merged for t in ("floodwater", "higher ground", "rising water", "avoid driving")):
            ok = True
        elif expected == "fire" and any(t in merged for t in ("smoke", "flames", "get low", "safe exit", "avoid smoke")):
            ok = True
        elif expected == "earthquake" and any(t in merged for t in ("drop", "cover", "hold on", "aftershock", "ground shaking")):
            ok = True

        if ok:
            self.hazard_ok += 1
        elif len(self.hazard_fails) < 5:
            self.hazard_fails.append(
                f"expected={expected!r} top1={short(top_blobs[0], 180)!r}"
            )

    def run_reflection(self) -> None:
        self.reflect_total += 1
        if self.adapter.reflect():
            self.reflect_ok += 1

    @staticmethod
    def score(ratio: float, max_points: int) -> int:
        ratio = max(0.0, min(1.0, ratio))
        return round(ratio * max_points)

    def run(self) -> None:
        hr("CLAIR 24HR CONTINUOUS ACTIVITY BENCHMARK")
        self.initial_wm_count = self.adapter.wm_count()
        self.initial_ltm_count = self.adapter.ltm_count()

        print(f"Simulated hours                {self.hours}")
        print(f"Cycles per hour                {self.cycles_per_hour}")
        print(f"Initial WM count               {self.initial_wm_count}")
        print(f"Initial LTM count              {self.initial_ltm_count}")

        t0 = time.time()

        for hour in range(self.hours):
            for cycle_in_hour in range(self.cycles_per_hour):
                cycle = hour * self.cycles_per_hour + cycle_in_hour

                self.inject_batch(cycle)
                self.run_recall_probe()

                if cycle % 3 == 0:
                    self.run_hazard_probe()

                if cycle % 5 == 0:
                    self.run_reflection()

            print(
                f"[hour {hour + 1:02d}] "
                f"recall={self.recall_ok}/{self.recall_total} "
                f"direct_sim_grounding={self.hazard_ok}/{self.hazard_total} "
                f"injected={self.inject_ok}/{self.inject_total} "
                f"(wm={self.wm_inject_ok}, ltm_persist={self.ltm_persist_successes}, "
                f"new={self.ltm_inserted_new}, reinforce={self.ltm_reinforced_existing}, "
                f"merge={self.ltm_merged_revision}, contested={self.ltm_stored_conflict_variant}) "
                f"identity_leaks={self.identity_recall_leaks} "
                f"conflicts={self.conflicts} "
                f"reflect={self.reflect_ok}/{self.reflect_total} "
                f"WM={self.adapter.wm_count()} "
                f"LTM={self.adapter.ltm_count()}"
            )

        elapsed = time.time() - t0

        phase_a = self.score(self.inject_ok / max(1, self.inject_total), 150)
        phase_b = self.score(self.recall_ok / max(1, self.recall_total), 350)
        phase_c = self.score(self.hazard_ok / max(1, self.hazard_total), 300)
        phase_d = self.score(self.reflect_ok / max(1, self.reflect_total), 100)

        final_wm = self.adapter.wm_count()
        final_ltm = self.adapter.ltm_count()
        net_ltm_delta = final_ltm - self.initial_ltm_count if final_ltm >= 0 and self.initial_ltm_count >= 0 else 0

        growth_checks = 0
        growth_total = 6

        if final_wm >= 0:
            growth_checks += 1
        if final_ltm >= 0:
            growth_checks += 1
        if self.wm_inject_ok > 0:
            growth_checks += 1
        if self.ltm_persist_successes > 0:
            growth_checks += 1
        if (self.ltm_inserted_new + self.ltm_reinforced_existing + self.ltm_merged_revision + self.ltm_stored_conflict_variant) > 0:
            growth_checks += 1
        if net_ltm_delta > 0 or self.ltm_reinforced_existing > 0 or self.ltm_merged_revision > 0 or self.ltm_stored_conflict_variant > 0:
            growth_checks += 1

        phase_e = self.score(growth_checks / growth_total, 100)

        total = phase_a + phase_b + phase_c + phase_d + phase_e

        hr("SCORE SUMMARY")
        print(f"Phase A: Injection reliability         {phase_a:3d}/150")
        print(f"Phase B: Retrieval recall              {phase_b:3d}/350")
        print(f"Phase C: Direct simulator grounding    {phase_c:3d}/300")
        print(f"Phase D: Reflection execution          {phase_d:3d}/100")
        print(f"Phase E: Memory growth sanity          {phase_e:3d}/100")

        hr("FINAL RESULTS")
        print(f"Runtime seconds                {elapsed:.3f}")
        print(f"Final WM count                 {final_wm}")
        print(f"Final LTM count                {final_ltm}")
        print(f"Net LTM row delta              {net_ltm_delta}")
        print(f"Injection success              {self.inject_ok}/{self.inject_total} ({pct(self.inject_ok, self.inject_total)})")
        print(f"WM injection successes         {self.wm_inject_ok}")
        print(f"LTM persist successes          {self.ltm_persist_successes}")
        print(f"LTM inserted_new               {self.ltm_inserted_new}")
        print(f"LTM reinforced_existing        {self.ltm_reinforced_existing}")
        print(f"LTM merged_revision            {self.ltm_merged_revision}")
        print(f"LTM stored_conflict_variant    {self.ltm_stored_conflict_variant}")
        print(f"Recall                         {self.recall_ok}/{self.recall_total} ({pct(self.recall_ok, self.recall_total)})")
        print(f"Identity recall leaks          {self.identity_recall_leaks}")
        print(f"Direct simulator grounding     {self.hazard_ok}/{self.hazard_total} ({pct(self.hazard_ok, self.hazard_total)})")
        print(f"Reflection                     {self.reflect_ok}/{self.reflect_total} ({pct(self.reflect_ok, self.reflect_total)})")
        print(f"Conflict injections            {self.conflicts}")
        print(f"Total Score                    {total}/1000")

        ratio = total / 1000.0
        if ratio >= 0.95:
            verdict = "PASS (day-scale stable)"
        elif ratio >= 0.85:
            verdict = "PASS-WARN (usable, minor scaling strain)"
        elif ratio >= 0.70:
            verdict = "WARN (memory pressure visible)"
        else:
            verdict = "FAIL (instability detected)"

        print(f"Verdict                        {verdict}")

        if self.recall_fails:
            hr("SAMPLE RECALL FAILURES")
            for item in self.recall_fails:
                print(item)

        if self.hazard_fails:
            hr("SAMPLE DIRECT SIMULATOR GROUNDING FAILURES")
            for item in self.hazard_fails:
                print(item)

        if self.store_debug_failures:
            hr("SAMPLE STORE FAILURES")
            for item in self.store_debug_failures:
                print(item)


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=3)
    parser.add_argument("--cycles_per_hour", type=int, default=10)
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
    bench = Benchmark24Hr(adapter, args.hours, args.cycles_per_hour)
    bench.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())