from __future__ import annotations

import os
import tempfile
from pprint import pprint
from typing import Any, Dict, List

import config

from memory.long_term_memory import LongTermMemory
from memory.working_memory import WorkingMemory
from learning.epistemic_tagger import EpistemicTagger
from learning.hippocampus_ingest import HippocampusIngestor
from calibration.ACC import ACC


def hr(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def safe_close(obj: Any) -> None:
    try:
        if obj is not None and hasattr(obj, "close"):
            obj.close()
    except Exception as exc:
        print(f"[close warning] {exc}")


def short_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "memory_id": row.get("memory_id"),
        "type": row.get("type"),
        "domain": row.get("domain"),
        "kind": row.get("kind"),
        "status": row.get("status") or row.get("verification_status"),
        "content": (row.get("content") or row.get("claim") or "")[:100],
    }


def short_claim(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "claim_key": row.get("claim_key"),
        "layers": row.get("layers"),
        "row_count": row.get("row_count"),
        "mirror_only": row.get("mirror_only"),
        "epistemic_state": row.get("epistemic_state"),
        "canonical_text": (row.get("canonical_text") or "")[:120],
    }


def contains_text(rows: List[Dict[str, Any]], needle: str) -> bool:
    target = (needle or "").strip().lower()
    if not target:
        return False
    for row in rows:
        text = (
            row.get("content")
            or row.get("claim")
            or row.get("canonical_text")
            or ""
        )
        if target in str(text).strip().lower():
            return True
    return False


def main() -> None:
    temp_dir = tempfile.mkdtemp(prefix="clair_smoke_")
    db_path = os.path.join(temp_dir, "ltm_smoke.db")

    old_db_path = getattr(config, "LONG_TERM_DB_PATH", None)
    old_verbose = getattr(config, "VERBOSE", False)

    config.LONG_TERM_DB_PATH = db_path
    config.VERBOSE = True

    ltm: LongTermMemory | None = None
    wm: WorkingMemory | None = None

    try:
        hr("LTM INIT")
        ltm = LongTermMemory()
        print("LTM version:", getattr(ltm, "VERSION", "unknown"))
        print("DB path:", db_path)

        hr("SEED BASELINE MEMORIES")
        seed_payload = [
            {
                "type": "claim",
                "content": "Water boils at 100 C at sea level.",
                "confidence": 0.90,
                "source": "reading",
                "domain": "science",
                "tags": ["science", "boiling"],
                "kind": "fact",
                "status": "verified",
                "details": {
                    "verified": True,
                    "source_trust": "trusted",
                    "status": "verified",
                    "verification_status": "verified",
                },
                "evidence": [],
            },
            {
                "type": "chapter_summary",
                "content": "Chapter 3 argues that perception can distort reality.",
                "confidence": 0.82,
                "source": "reading",
                "domain": "literature",
                "tags": ["literature", "reading", "chapter_3"],
                "kind": "summary",
                "status": "unverified",
                "details": {
                    "verified": False,
                    "status": "unverified",
                    "verification_status": "unverified",
                },
                "evidence": [],
            },
        ]

        seed_result = ltm.store_detailed(seed_payload)
        print("store_detailed result:")
        pprint(seed_result)

        hr("LTM RETRIEVE")
        all_ltm = ltm.retrieve(limit=20)
        for row in all_ltm:
            pprint(row)

        hr("WORKING MEMORY INIT + PRELOAD")
        wm = WorkingMemory(
            max_history=50,
            decay_rate=getattr(config, "MEMORY_DECAY_RATE", 0.98),
            preload_long_term=True,
        )
        print("WM version:", getattr(wm, "VERSION", "unknown"))
        print("wm stats:")
        pprint(wm.stats())

        hr("WM RETRIEVE - SCIENCE")
        wm_science = wm.retrieve("water boiling science", count=5)
        pprint(wm_science)

        hr("WM RETRIEVE - LITERATURE")
        wm_lit = wm.retrieve("chapter 3 perception reality", count=5)
        pprint(wm_lit)

        hr("EPISTEMIC TAGGER CHECK")
        tagger = EpistemicTagger(strict=False)

        tagged_fact = tagger.tag(
            "Water boils at 100 C at sea level.",
            context_profile={"domain": "science", "tags": ["science"]},
            doc_meta={"is_nonfiction": True},
        )
        pprint(tagged_fact)

        tagged_lit = tagger.tag(
            "The narrator suggests that perception distorts reality.",
            context_profile={
                "domain": "literature",
                "tags": ["literature", "reading"],
                "goal": "interpret",
            },
            doc_meta={"genre_hint": "novel", "is_fiction": True},
        )
        pprint(tagged_lit)

        hr("HIPPOCAMPUS INGEST CHECK")
        ingestor = HippocampusIngestor(working_memory=wm, long_term_memory=ltm)

        stored_claims, stored_frames, stored_summaries = ingestor.store_claims(
            doc_id="book_demo",
            chunk_id=3,
            claims=[
                {
                    "text": "Water boils at 90 C at sea level.",
                    "claim_type": "claim",
                    "speaker": "author",
                    "modality": "asserted",
                },
                {
                    "text": "The reading argues that perception shapes what people believe is real.",
                    "claim_type": "claim",
                    "speaker": "author",
                    "modality": "asserted",
                },
            ],
            domain="literature",
            tags=["reading", "literature"],
            confidence=0.72,
            weight=0.66,
            persist_to_ltm=False,
            doc_meta={
                "title": "Chapter 3",
                "book_title_hint": "Demo Book",
                "chapter_hint": "3",
                "is_fiction": True,
                "genre_hint": "novel",
            },
            section_summary="Section 3 focuses on how interpretation can distort reality.",
            chapter_summary="Chapter 3 argues that perception can distort reality and shape belief.",
        )
        print(
            "stored_claims, stored_frames, stored_summaries =",
            (stored_claims, stored_frames, stored_summaries),
        )

        hr("WM STATS AFTER INGEST")
        wm_stats_after = wm.stats()
        pprint(wm_stats_after)

        hr("WM CONTEXT SNAPSHOT")
        wm_context = wm.context_snapshot()
        pprint(wm_context)

        hr("ACC INIT")
        acc = ACC(wm, ltm)
        print("ACC version:", getattr(acc, "VERSION", "unknown"))

        hr("ACC WIRING CHECK")
        print("wm object id:       ", id(wm))
        print("acc.wm object id:   ", id(acc.wm))
        print("ltm object id:      ", id(ltm))
        print("acc.ltm object id:  ", id(acc.ltm))
        print("wm is acc.wm:       ", wm is acc.wm)
        print("ltm is acc.ltm:     ", ltm is acc.ltm)
        print("len(wm.buffer):     ", len(getattr(wm, "buffer", []) or []))
        print("len(acc.wm.buffer): ", len(getattr(acc.wm, "buffer", []) or []))
        print("ACC wm type:        ", type(acc.wm))
        print("ACC ltm type:       ", type(acc.ltm))

        hr("ACC DIRECT MEMORY SNAPSHOT")
        acc_memories = acc.get_memories(limit=50)
        print("ACC get_memories count:", len(acc_memories))
        for i, row in enumerate(acc_memories[:10]):
            print(i, short_row(row))

        hr("ACC CANONICAL CLAIM SNAPSHOT")
        canonical_claims = acc.build_canonical_claims(acc_memories)
        print("ACC canonical claim count:", len(canonical_claims))
        for i, row in enumerate(canonical_claims[:10]):
            print(i, short_claim(row))

        hr("ACC AUDIT")
        acc_audit = acc.full_audit()
        pprint(acc_audit)

        hr("ACC NEXT QUESTION")
        acc_question = acc.next_question()
        pprint(acc_question)

        hr("LTM SEARCH")
        ltm_hits = ltm.search("perception reality chapter 3", limit=5)
        pprint(ltm_hits)

        hr("WM RETRIEVE POST-INGEST")
        post_ingest_hits = wm.retrieve("chapter 3 perception reality", count=10)
        pprint(post_ingest_hits)

        numeric_conflicts = acc_audit.get("numeric_conflicts") or []
        flagged_conflicts = acc_audit.get("flagged_conflicts") or []
        negation_conflicts = acc_audit.get("negation_conflicts") or []
        duplicates = acc_audit.get("duplicates") or []
        mirror_groups = acc_audit.get("mirror_row_groups") or []

        raw_memory_count = int(acc_audit.get("raw_memory_count", 0) or 0)
        canonical_claim_count = int(acc_audit.get("canonical_claim_count", 0) or 0)
        audit_memory_count = int(acc_audit.get("memory_count", 0) or 0)

        question_kind = acc_question.get("kind") if isinstance(acc_question, dict) else None
        question_text = (
            acc_question.get("question") or acc_question.get("prompt") or ""
            if isinstance(acc_question, dict)
            else ""
        )

        hr("PASS/FAIL SUMMARY")
        pass_flags = {
            "ltm_has_rows": len(all_ltm) >= 2,
            "wm_preloaded": wm.stats().get("count", 0) >= 2,
            "hippocampus_added_material": stored_claims >= 1 and stored_summaries >= 1,
            "acc_audit_present": isinstance(acc_audit, dict) and "memory_count" in acc_audit,
            "acc_question_available": acc_question is None or isinstance(acc_question, dict),

            # Wiring checks
            "acc_wm_is_same_object": acc.wm is wm,
            "acc_ltm_is_same_object": acc.ltm is ltm,
            "acc_buffer_matches_wm": len(getattr(acc.wm, "buffer", []) or []) == len(getattr(wm, "buffer", []) or []),

            # ACC sees live memory and ingest results
            "acc_sees_wm_ingest": len(acc_memories) >= 4,
            "acc_sees_100c_claim": contains_text(acc_memories, "Water boils at 100 C at sea level."),
            "acc_sees_90c_claim": contains_text(acc_memories, "Water boils at 90 C at sea level."),

            # Canonical view should exist and be sensible
            "acc_has_canonical_claims": len(canonical_claims) >= 3,
            "audit_exposes_raw_count": raw_memory_count >= len(canonical_claims),
            "audit_exposes_canonical_count": canonical_claim_count >= 3,
            "audit_memory_count_matches_canonical": audit_memory_count == canonical_claim_count,

            # Conflict behavior
            "acc_found_numeric_or_flagged_conflict": bool(numeric_conflicts) or bool(flagged_conflicts),
            "acc_conflict_question_available": (
                (question_kind == "conflict")
                or ("conflict" in str(question_text).lower())
                or bool(numeric_conflicts)
                or bool(flagged_conflicts)
            ),

            # Mirror-aware audit behavior
            "mirror_groups_exposed_or_not_needed": isinstance(mirror_groups, list),
            "duplicates_structure_valid": isinstance(duplicates, list),
            "negation_structure_valid": isinstance(negation_conflicts, list),
        }
        pprint(pass_flags)

        failed = [k for k, v in pass_flags.items() if not v]
        if failed:
            print("\nSMOKE TEST RESULT: FAIL")
            print("Failed checks:", failed)
        else:
            print("\nSMOKE TEST RESULT: PASS")

        hr("SMOKE TEST NOTES")
        print("raw_memory_count       =", raw_memory_count)
        print("canonical_claim_count  =", canonical_claim_count)
        print("audit memory_count     =", audit_memory_count)
        print("numeric_conflicts      =", len(numeric_conflicts))
        print("flagged_conflicts      =", len(flagged_conflicts))
        print("negation_conflicts     =", len(negation_conflicts))
        print("duplicates             =", len(duplicates))
        print("mirror_row_groups      =", len(mirror_groups))
        print("question_kind          =", question_kind)

    finally:
        wm_ltm = wm.long_term if wm is not None and hasattr(wm, "long_term") else None
        if wm_ltm is ltm:
            safe_close(ltm)
        else:
            safe_close(wm_ltm)
            safe_close(ltm)

        config.LONG_TERM_DB_PATH = old_db_path
        config.VERBOSE = old_verbose


if __name__ == "__main__":
    main()