# FILE: learning/hippocampus_ingest.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import time
import re

try:
    from learning.epistemic_tagger import EpistemicTagger
except Exception:
    EpistemicTagger = None  # type: ignore


ClaimLike = Union[str, Dict[str, Any]]


class HippocampusIngestor:
    """
    Intake bridge from reader/processor output into WorkingMemory/LTM.

    Responsibilities:
    - normalize extracted claims / frames / summaries
    - apply epistemic tagging
    - shape memory payloads for WM
    - optionally persist to LTM
    - preserve literature summaries as first-class memories

    Key protections:
    - literature summaries remain first-class
    - real-world numeric factual claims can override document-literature bias
    - claim payloads preserve routing and verification hints
    - ingest now stamps a clearer epistemic route / scope for downstream audit
    """

    _BINARY_MARKERS = ("%pdf", "endobj", "xref", "endstream", "stream")
    _WORD_RE = re.compile(r"[A-Za-z0-9']+")
    _SPACE_RE = re.compile(r"\s+")
    _CHAPTER_RE = re.compile(r"\bchapter[_\s]*(\d+)\b", re.IGNORECASE)
    _SECTION_RE = re.compile(r"\bsection[_\s]*(\d+)\b", re.IGNORECASE)
    _NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")

    _REALWORLD_NUMERIC_MARKERS = {
        "water", "boil", "boils", "boiling",
        "celsius", "fahrenheit", "degree", "degrees",
        "meter", "meters", "km", "mile", "miles",
        "bone", "bones", "everest", "sea", "level",
        "human", "humans",
    }

    _LITERATURE_ONLY_CUES = {
        "narrator", "character", "chapter", "section",
        "author", "novel", "poem", "story", "theme",
        "symbol", "metaphor",
    }

    _DOCUMENT_SCOPED_CUES = {
        "chapter", "section", "the reading", "the text", "the author",
        "the narrator", "the story", "the novel", "the poem",
        "argues", "suggests", "implies", "portrays",
    }

    def __init__(self, working_memory, long_term_memory=None):
        self.wm = working_memory
        self.ltm = long_term_memory
        self.tagger = EpistemicTagger(strict=False) if EpistemicTagger is not None else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def store_claims(
        self,
        *,
        doc_id: str,
        chunk_id: int,
        claims: List[ClaimLike],
        domain: str = "general",
        tags: Optional[List[str]] = None,
        confidence: float = 0.65,
        weight: float = 0.6,
        persist_to_ltm: bool = False,
        doc_meta: Optional[Dict[str, Any]] = None,
        context_profile: Optional[Dict[str, Any]] = None,
        section_summary: Optional[str] = None,
        chapter_summary: Optional[str] = None,
        frame_candidates: Optional[List[Union[str, Dict[str, Any]]]] = None,
    ) -> Tuple[int, int, int]:
        """
        Returns:
            (stored_claims_count, stored_frames_count, stored_summaries_count)
        """
        tags = [str(t).strip() for t in (tags or []) if str(t).strip()]
        doc_meta = dict(doc_meta or {})
        frame_candidates = frame_candidates or []

        dom = str(domain or "general").strip().lower()
        if dom == "literature" and "literature" not in [t.lower() for t in tags]:
            tags = tags + ["literature"]

        cp = context_profile or {
            "domain": dom,
            "tags": tags,
            "goal": "interpret" if dom == "literature" else "learn",
        }

        payload: List[Dict[str, Any]] = []
        ts = time.time()

        stored_frames = 0
        stored_claims = 0
        stored_summaries = 0

        normalized_meta = self._normalized_doc_meta(doc_meta, doc_id=doc_id, default_domain=dom)
        evidence_list = self._build_evidence_list(doc_id, chunk_id, normalized_meta)

        # 1) legacy embedded narrative frames
        for item in claims or []:
            if isinstance(item, dict) and item.get("_type") == "narrative_frame":
                frame = self._build_frame_mem(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    item=item,
                    dom=dom,
                    tags=tags,
                    ts=ts,
                    doc_meta=normalized_meta,
                    evidence=evidence_list,
                )
                if frame:
                    payload.append(frame)
                    stored_frames += 1

        # 2) generic frame candidates
        for item in frame_candidates:
            frame = self._build_generic_frame_mem(
                doc_id=doc_id,
                chunk_id=chunk_id,
                item=item,
                dom=dom,
                tags=tags,
                ts=ts,
                doc_meta=normalized_meta,
                evidence=evidence_list,
            )
            if frame:
                payload.append(frame)
                stored_frames += 1

        # 3) claims
        for item in claims or []:
            if isinstance(item, dict) and item.get("_type") == "narrative_frame":
                continue

            norm = self._normalize_claim_item(item)
            if not norm:
                continue

            claim_text = self._clean_text(norm["text"])
            if not claim_text or self._looks_binaryish(claim_text):
                continue

            speaker = norm.get("speaker")
            modality = norm.get("modality")
            claim_type = norm.get("claim_type")
            confidence_text = norm.get("confidence_text")
            evidence_span = norm.get("evidence_span")

            res = None
            if self.tagger is not None:
                try:
                    res = self.tagger.tag(
                        claim_text,
                        context_profile=cp,
                        doc_meta=normalized_meta,
                        speaker=speaker,
                        modality=modality,
                        claim_type=claim_type,
                        verified_facts=None,
                        verified_fact_keywords=None,
                    )
                except Exception:
                    res = None

            truth_label = (getattr(res, "truth_label", None) if res else None) or "unknown"
            conf_truth = float(getattr(res, "confidence_truth", 0.5) if res else 0.5)
            reasons = list(getattr(res, "reasons", []) if res else [])
            speaker_out = getattr(res, "speaker", None) if res else speaker
            modality_out = getattr(res, "modality", None) if res else modality
            claim_type_out = getattr(res, "claim_type", None) if res else claim_type

            memory_kind_hint = (getattr(res, "memory_kind_hint", None) if res else None) or "unresolved"
            verification_status_hint = (
                getattr(res, "verification_status_hint", None) if res else None
            ) or "unverified"
            domain_hint = (getattr(res, "domain_hint", None) if res else None) or dom
            source_trust_hint = (getattr(res, "source_trust_hint", None) if res else None) or "normal"
            should_store_semantic = bool(getattr(res, "should_store_semantic", True) if res else True)
            should_route_to_verifier = bool(getattr(res, "should_route_to_verifier", False) if res else False)
            hazard_family = (getattr(res, "hazard_family", None) if res else None) or None
            res_tags = list(getattr(res, "tags", []) if res else [])

            override = self._realworld_fact_override(
                claim_text=claim_text,
                current_domain=domain_hint,
                current_kind=memory_kind_hint,
                current_truth=truth_label,
                current_status=verification_status_hint,
                current_route=should_route_to_verifier,
            )
            if override is not None:
                domain_hint = override["domain_hint"]
                memory_kind_hint = override["memory_kind_hint"]
                truth_label = override["truth_label"]
                verification_status_hint = override["verification_status_hint"]
                should_route_to_verifier = override["should_route_to_verifier"]
                source_trust_hint = override["source_trust_hint"]
                res_tags.extend(override["extra_tags"])
                reasons = reasons + override["reasons"]

            truth_label = self._normalize_truth_label(truth_label, domain_hint, claim_text)

            if not should_store_semantic and memory_kind_hint not in {"reflection", "summary"}:
                continue

            final_kind = self._normalize_memory_kind(
                memory_kind_hint,
                truth_label,
                domain_hint,
                claim_type_out,
            )

            ingest_route = self._infer_ingest_route(
                claim_text=claim_text,
                domain=domain_hint,
                truth_label=truth_label,
                final_kind=final_kind,
                claim_type=claim_type_out,
                doc_meta=normalized_meta,
            )

            final_status = self._normalize_status_hint(
                verification_status_hint,
                truth_label=truth_label,
                domain=domain_hint,
                source_trust_hint=source_trust_hint,
                ingest_route=ingest_route,
            )

            base_weight, mem_conf = self._score_policy(
                truth_label=truth_label,
                default_weight=float(weight),
                default_conf=float(confidence),
                truth_conf=float(conf_truth),
                domain=domain_hint,
                memory_kind_hint=final_kind,
                verification_status_hint=final_status,
                ingest_route=ingest_route,
            )

            if domain_hint == "literature" and final_kind == "fact":
                base_weight = min(float(base_weight), 0.66)
                mem_conf = min(float(mem_conf), 0.72)

            pending_verification = final_status in {"provisional", "unverified"}
            verified_flag = final_status == "verified"
            contested_flag = final_status == "contested"

            scope = self._infer_scope(
                ingest_route=ingest_route,
                domain=domain_hint,
                truth_label=truth_label,
            )
            storage_class = self._infer_storage_class(
                ingest_route=ingest_route,
                final_kind=final_kind,
                domain=domain_hint,
            )

            semantic_tags = self._build_tags(
                tags,
                extra=res_tags + [
                    truth_label,
                    str(claim_type_out or "claim"),
                    final_kind,
                    "reading",
                    hazard_family,
                    ingest_route,
                    scope,
                ],
                doc_meta=normalized_meta,
            )

            cobj = {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "speaker": speaker_out,
                "modality": modality_out,
                "claim_type": claim_type_out or "claim",
                "truth_label": truth_label,
                "confidence_truth": mem_conf,
                "confidence_text": float(confidence_text) if isinstance(confidence_text, (int, float)) else None,
                "reasons": reasons[:8],
                "evidence_span": str(evidence_span or claim_text[:220]),
                "evidence_ids": list(evidence_list),
                "doc_meta": normalized_meta,
                "chapter_hint": normalized_meta.get("chapter_hint"),
                "section_hint": normalized_meta.get("section_hint"),
                "book_title_hint": normalized_meta.get("book_title_hint"),
                "title": normalized_meta.get("title"),
                "file_name": normalized_meta.get("file_name") or normalized_meta.get("filename"),
                "reader_name": normalized_meta.get("reader_name"),
                "domain_hint": domain_hint,
                "hazard_family": hazard_family,
                "memory_kind_hint": final_kind,
                "verification_status_hint": final_status,
                "should_route_to_verifier": should_route_to_verifier,
                "ingest_route": ingest_route,
                "scope": scope,
                "storage_class": storage_class,
            }
            cobj = {k: v for k, v in cobj.items() if v is not None}

            details = self._base_details(
                doc_id=doc_id,
                chunk_id=chunk_id,
                doc_meta=normalized_meta,
                evidence=evidence_list,
                status=final_status,
                pending_verification=pending_verification,
                summary_priority=0,
            )
            details["truth_label"] = truth_label
            details["claim_type"] = str(claim_type_out or "claim")
            details["confidence_truth"] = float(mem_conf)
            details["evidence_span"] = str(evidence_span or claim_text[:220])
            details["speaker"] = speaker_out
            details["modality"] = modality_out
            details["reasons"] = reasons[:8]
            details["verified"] = verified_flag
            details["pending_verification"] = pending_verification
            details["contested"] = contested_flag
            details["verification_status"] = final_status
            details["source_trust"] = source_trust_hint
            details["hazard_family"] = hazard_family
            details["should_route_to_verifier"] = should_route_to_verifier
            details["memory_kind_hint"] = final_kind
            details["epistemic_route"] = ingest_route
            details["scope"] = scope
            details["knowledge_scope"] = scope
            details["storage_class"] = storage_class
            details["document_scoped"] = scope == "document_scoped"
            details["global_candidate"] = scope == "global"
            details["real_world_candidate"] = ingest_route == "real_world_fact"

            mem_type = self._infer_mem_type(
                final_kind=final_kind,
                domain=domain_hint,
                truth_label=truth_label,
                claim_type=claim_type_out,
                ingest_route=ingest_route,
            )

            payload.append(
                self._make_memory_base(
                    mem_type=mem_type,
                    content=claim_text,
                    content_obj=cobj,
                    confidence=mem_conf,
                    weight=float(base_weight),
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    doc_meta=normalized_meta,
                    domain=domain_hint,
                    tags=semantic_tags,
                    timestamp=ts,
                    details=details,
                    evidence=evidence_list,
                    kind=final_kind,
                    status=final_status,
                    hazard_family=hazard_family,
                )
            )
            stored_claims += 1

        # 4) section summary
        if section_summary and not self._looks_binaryish(section_summary):
            mem = self._build_summary_mem(
                summary_text=section_summary,
                summary_type="section_summary",
                doc_id=doc_id,
                chunk_id=chunk_id,
                dom=dom,
                tags=tags,
                ts=ts,
                doc_meta=normalized_meta,
                confidence=confidence,
                weight=weight,
                evidence=evidence_list,
            )
            if mem:
                payload.append(mem)
                stored_summaries += 1

        # 5) chapter summary
        if chapter_summary and not self._looks_binaryish(chapter_summary):
            mem = self._build_summary_mem(
                summary_text=chapter_summary,
                summary_type="chapter_summary",
                doc_id=doc_id,
                chunk_id=chunk_id,
                dom=dom,
                tags=tags,
                ts=ts,
                doc_meta=normalized_meta,
                confidence=max(0.70, float(confidence)),
                weight=max(0.72, float(weight)),
                evidence=evidence_list,
            )
            if mem:
                payload.append(mem)
                stored_summaries += 1

        if payload:
            self.wm.store(payload)
            if persist_to_ltm and self.ltm is not None:
                self.ltm.store(payload)
                for m in payload:
                    m["persisted"] = True

        return stored_claims, stored_frames, stored_summaries

    # ------------------------------------------------------------------
    # Metadata / normalization helpers
    # ------------------------------------------------------------------
    def _normalized_doc_meta(self, doc_meta: Dict[str, Any], *, doc_id: str, default_domain: str) -> Dict[str, Any]:
        meta = dict(doc_meta or {})

        filename = str(
            meta.get("file_name")
            or meta.get("filename")
            or doc_id
            or ""
        ).strip()

        title = str(
            meta.get("title")
            or meta.get("document_title")
            or meta.get("book_title_hint")
            or filename
        ).strip()

        chapter_hint = meta.get("chapter_hint") or meta.get("chapter")
        section_hint = meta.get("section_hint") or meta.get("section")
        book_title_hint = meta.get("book_title_hint") or meta.get("book_title")
        domain_hint = meta.get("domain_hint") or meta.get("document_domain_hint")
        reader_name = meta.get("reader_name") or "document_reader"
        file_type = meta.get("file_type")
        evidence_id = meta.get("evidence_id")
        preview = meta.get("preview")

        if not chapter_hint:
            m = self._CHAPTER_RE.search(title) or self._CHAPTER_RE.search(filename)
            if m:
                chapter_hint = m.group(1)

        if not section_hint:
            m = self._SECTION_RE.search(title) or self._SECTION_RE.search(filename)
            if m:
                section_hint = m.group(1)

        if not book_title_hint:
            if title and not re.fullmatch(r"lesson\d+", title.strip().lower()):
                book_title_hint = title

        if not domain_hint:
            domain_hint = default_domain or "general"

        meta["filename"] = filename
        meta["file_name"] = filename
        meta["title"] = title
        meta["document_title"] = title
        meta["chapter_hint"] = chapter_hint
        meta["section_hint"] = section_hint
        meta["book_title_hint"] = book_title_hint
        meta["domain_hint"] = str(domain_hint).strip().lower()
        meta["reader_name"] = reader_name
        meta["file_type"] = file_type
        meta["evidence_id"] = evidence_id
        meta["preview"] = preview
        return meta

    def _clean_text(self, text: str) -> str:
        t = str(text or "")
        t = t.replace("\x00", " ")
        t = t.replace("“", '"').replace("”", '"').replace("’", "'")
        t = self._SPACE_RE.sub(" ", t).strip()
        return t

    def _clean_summary_text(self, text: str) -> str:
        t = self._clean_text(text)
        t = re.sub(r"\s*\.{2,}\s*", ". ", t)
        t = re.sub(r"\s+([,.;:!?])", r"\1", t)
        t = t.strip()

        if len(t) > 900:
            t = t[:899].rstrip() + "…"
        return t

    def _build_context(self, doc_id: str, chunk_id: int, doc_meta: Dict[str, Any]) -> List[str]:
        ctx = [f"doc:{doc_id}", f"chunk:{chunk_id}"]

        chapter = doc_meta.get("chapter_hint") or doc_meta.get("chapter")
        section = doc_meta.get("section_hint") or doc_meta.get("section")
        book = doc_meta.get("book_title_hint") or doc_meta.get("book_title")
        title = doc_meta.get("title") or doc_meta.get("document_title")

        if chapter:
            ctx.append(f"chapter:{chapter}")
        if section:
            ctx.append(f"section:{section}")
        if book:
            ctx.append(f"book:{book}")
        if title:
            ctx.append(f"title:{title}")

        return ctx

    def _build_tags(
        self,
        base_tags: List[str],
        extra: Optional[List[Any]] = None,
        doc_meta: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        out = [str(t).strip() for t in (base_tags or []) if str(t).strip()]
        doc_meta = doc_meta or {}

        chapter = doc_meta.get("chapter_hint") or doc_meta.get("chapter")
        section = doc_meta.get("section_hint") or doc_meta.get("section")
        book = doc_meta.get("book_title_hint") or doc_meta.get("book_title")
        domain_hint = doc_meta.get("domain_hint") or doc_meta.get("document_domain_hint")
        file_type = doc_meta.get("file_type")

        if book:
            out.append(str(book).strip().lower().replace(" ", "_"))
        if chapter:
            out.append(f"chapter_{str(chapter).strip().lower()}")
        if section:
            out.append(f"section_{str(section).strip().lower()}")
        if domain_hint:
            out.append(str(domain_hint).strip().lower())
        if file_type:
            out.append(str(file_type).strip().lower())

        for item in (extra or []):
            if item is None:
                continue
            s = str(item).strip()
            if s:
                out.append(s)

        seen = set()
        final: List[str] = []
        for t in out:
            key = t.lower()
            if key in seen:
                continue
            seen.add(key)
            final.append(t)
        return final

    def _build_evidence_list(self, doc_id: str, chunk_id: int, doc_meta: Dict[str, Any]) -> List[str]:
        evidence_id = str(doc_meta.get("evidence_id") or f"{doc_id}:chunk_{chunk_id}").strip()
        return [evidence_id] if evidence_id else [f"{doc_id}:chunk_{chunk_id}"]

    def _base_details(
        self,
        *,
        doc_id: str,
        chunk_id: int,
        doc_meta: Dict[str, Any],
        evidence: List[str],
        status: str,
        pending_verification: bool,
        summary_priority: int,
    ) -> Dict[str, Any]:
        return {
            "verified": status == "verified",
            "pending_verification": bool(pending_verification),
            "contested": status == "contested",
            "superseded": False,
            "recall_blocked": status == "contested",
            "status": status,
            "verification_status": status,
            "summary_priority": int(summary_priority),
            "source": "reading",
            "source_trust": "normal",
            "evidence": list(evidence),
            "doc_id": doc_id,
            "chunk_index": chunk_id,
            "file_name": doc_meta.get("file_name") or doc_meta.get("filename"),
            "file_type": doc_meta.get("file_type"),
            "title": doc_meta.get("title"),
            "book_title_hint": doc_meta.get("book_title_hint"),
            "chapter_hint": doc_meta.get("chapter_hint"),
            "section_hint": doc_meta.get("section_hint"),
            "reader_name": doc_meta.get("reader_name"),
            "preview": doc_meta.get("preview"),
        }

    def _make_memory_base(
        self,
        *,
        mem_type: str,
        content: str,
        content_obj: Dict[str, Any],
        confidence: float,
        weight: float,
        doc_id: str,
        chunk_id: int,
        doc_meta: Dict[str, Any],
        domain: str,
        tags: List[str],
        timestamp: float,
        details: Dict[str, Any],
        evidence: List[str],
        kind: str,
        status: str,
        hazard_family: Optional[str],
    ) -> Dict[str, Any]:
        return {
            "type": mem_type,
            "content": content,
            "content_obj": content_obj,
            "confidence": float(confidence),
            "weight": float(weight),
            "context": self._build_context(doc_id, chunk_id, doc_meta),
            "source": "reading",
            "evidence": list(evidence),
            "last_verified": timestamp if status == "verified" else None,
            "conflict": status == "contested",
            "persisted": False,
            "domain": domain,
            "hazard_family": hazard_family,
            "tags": list(tags),
            "kind": kind,
            "timestamp": timestamp,
            "ts": timestamp,
            "details": details,
        }

    # ------------------------------------------------------------------
    # Frame builders
    # ------------------------------------------------------------------
    def _build_frame_mem(
        self,
        doc_id: str,
        chunk_id: int,
        item: Dict[str, Any],
        dom: str,
        tags: List[str],
        ts: float,
        doc_meta: Optional[Dict[str, Any]] = None,
        evidence: Optional[List[str]] = None,
    ):
        summary = self._clean_text(str(item.get("summary") or "").strip())
        if not summary and not item.get("themes") and not item.get("events"):
            return None

        doc_meta = dict(doc_meta or {})
        evidence = list(evidence or self._build_evidence_list(doc_id, chunk_id, doc_meta))

        cobj = {
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "entities": list(item.get("entities") or []),
            "themes": list(item.get("themes") or []),
            "tone": str(item.get("tone") or "neutral"),
            "events": list(item.get("events") or []),
            "summary": summary,
            "doc_meta": doc_meta,
            "chapter_hint": doc_meta.get("chapter_hint"),
            "section_hint": doc_meta.get("section_hint"),
            "book_title_hint": doc_meta.get("book_title_hint"),
            "title": doc_meta.get("title"),
            "evidence_ids": list(evidence),
            "ingest_route": "literary_frame" if dom == "literature" else "document_claim",
            "scope": "document_scoped",
            "storage_class": "semantic_document",
        }

        details = self._base_details(
            doc_id=doc_id,
            chunk_id=chunk_id,
            doc_meta=doc_meta,
            evidence=evidence,
            status="unverified",
            pending_verification=False,
            summary_priority=2,
        )
        details["epistemic_route"] = "literary_frame" if dom == "literature" else "document_claim"
        details["scope"] = "document_scoped"
        details["knowledge_scope"] = "document_scoped"
        details["storage_class"] = "semantic_document"
        details["document_scoped"] = True

        return self._make_memory_base(
            mem_type="literary_frame",
            content=summary,
            content_obj=cobj,
            confidence=0.68 if dom == "literature" else 0.55,
            weight=0.60 if dom == "literature" else 0.45,
            doc_id=doc_id,
            chunk_id=chunk_id,
            doc_meta=doc_meta,
            domain="literature" if dom == "literature" else dom,
            tags=self._build_tags(tags, extra=["frame", "literary_frame", "reading", "document_scoped"], doc_meta=doc_meta),
            timestamp=ts,
            details=details,
            evidence=evidence,
            kind="summary",
            status="unverified",
            hazard_family=None,
        )

    def _build_generic_frame_mem(
        self,
        doc_id: str,
        chunk_id: int,
        item: Union[str, Dict[str, Any]],
        dom: str,
        tags: List[str],
        ts: float,
        doc_meta: Optional[Dict[str, Any]] = None,
        evidence: Optional[List[str]] = None,
    ):
        doc_meta = dict(doc_meta or {})
        evidence = list(evidence or self._build_evidence_list(doc_id, chunk_id, doc_meta))

        if isinstance(item, str):
            text = self._clean_text(item.strip())
            if not text:
                return None
            content = text
        elif isinstance(item, dict):
            text = self._clean_text(str(item.get("text") or item.get("content") or item.get("summary") or "").strip())
            if not text:
                return None
            content = text
        else:
            return None

        cobj = {
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "frame_name": content,
            "doc_meta": doc_meta,
            "chapter_hint": doc_meta.get("chapter_hint"),
            "section_hint": doc_meta.get("section_hint"),
            "book_title_hint": doc_meta.get("book_title_hint"),
            "title": doc_meta.get("title"),
            "evidence_ids": list(evidence),
            "ingest_route": "literary_frame" if dom == "literature" else "document_claim",
            "scope": "document_scoped",
            "storage_class": "semantic_document",
        }

        details = self._base_details(
            doc_id=doc_id,
            chunk_id=chunk_id,
            doc_meta=doc_meta,
            evidence=evidence,
            status="unverified",
            pending_verification=False,
            summary_priority=3,
        )
        details["epistemic_route"] = "literary_frame" if dom == "literature" else "document_claim"
        details["scope"] = "document_scoped"
        details["knowledge_scope"] = "document_scoped"
        details["storage_class"] = "semantic_document"
        details["document_scoped"] = True

        return self._make_memory_base(
            mem_type="concept_frame",
            content=content,
            content_obj=cobj,
            confidence=0.70 if dom == "literature" else 0.62,
            weight=0.64 if dom == "literature" else 0.52,
            doc_id=doc_id,
            chunk_id=chunk_id,
            doc_meta=doc_meta,
            domain=dom,
            tags=self._build_tags(tags, extra=["frame", "concept_frame", "reading", "document_scoped"], doc_meta=doc_meta),
            timestamp=ts,
            details=details,
            evidence=evidence,
            kind="summary",
            status="unverified",
            hazard_family=None,
        )

    # ------------------------------------------------------------------
    # Summary builders
    # ------------------------------------------------------------------
    def _build_summary_mem(
        self,
        *,
        summary_text: str,
        summary_type: str,
        doc_id: str,
        chunk_id: int,
        dom: str,
        tags: List[str],
        ts: float,
        doc_meta: Dict[str, Any],
        confidence: float,
        weight: float,
        evidence: Optional[List[str]] = None,
    ):
        text = self._clean_summary_text(summary_text)
        if not text or len(text) < 24:
            return None

        if summary_type not in {"section_summary", "chapter_summary"}:
            summary_type = "section_summary"

        base_conf = max(0.72, min(0.94, float(confidence)))
        base_weight = max(0.72, min(0.96, float(weight)))
        summary_priority = 4

        if summary_type == "chapter_summary":
            base_conf = max(base_conf, 0.82)
            base_weight = max(base_weight, 0.86)
            summary_priority = 5
        else:
            base_conf = max(base_conf, 0.76)
            base_weight = max(base_weight, 0.78)

        evidence = list(evidence or self._build_evidence_list(doc_id, chunk_id, doc_meta))

        route = "literary_summary" if dom == "literature" else "document_claim"

        cobj = {
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "summary_type": summary_type,
            "doc_meta": doc_meta,
            "chapter_hint": doc_meta.get("chapter_hint"),
            "section_hint": doc_meta.get("section_hint"),
            "book_title_hint": doc_meta.get("book_title_hint"),
            "title": doc_meta.get("title"),
            "evidence_ids": list(evidence),
            "ingest_route": route,
            "scope": "document_scoped",
            "storage_class": "semantic_document",
        }

        details = self._base_details(
            doc_id=doc_id,
            chunk_id=chunk_id,
            doc_meta=doc_meta,
            evidence=evidence,
            status="unverified",
            pending_verification=False,
            summary_priority=summary_priority,
        )
        details["prefer_for_opinion"] = True
        details["prefer_for_summary_queries"] = True
        details["epistemic_route"] = route
        details["scope"] = "document_scoped"
        details["knowledge_scope"] = "document_scoped"
        details["storage_class"] = "semantic_document"
        details["document_scoped"] = True

        return self._make_memory_base(
            mem_type=summary_type,
            content=text,
            content_obj=cobj,
            confidence=base_conf,
            weight=base_weight,
            doc_id=doc_id,
            chunk_id=chunk_id,
            doc_meta=doc_meta,
            domain="literature" if dom == "literature" else dom,
            tags=self._build_tags(
                tags,
                extra=[
                    "summary",
                    summary_type,
                    "reading",
                    "literary_summary" if dom == "literature" else None,
                    "document_scoped",
                    "literature" if dom == "literature" else None,
                ],
                doc_meta=doc_meta,
            ),
            timestamp=ts,
            details=details,
            evidence=evidence,
            kind="summary",
            status="unverified",
            hazard_family=None,
        )

    # ------------------------------------------------------------------
    # Claim normalization / safety
    # ------------------------------------------------------------------
    def _normalize_claim_item(self, item: ClaimLike) -> Optional[Dict[str, Any]]:
        if isinstance(item, str):
            text = item.strip()
            if not text:
                return None
            return {"text": text}

        if isinstance(item, dict):
            text = str(item.get("text") or item.get("content") or "").strip()
            if not text:
                return None
            out = {"text": text}
            for k in ("speaker", "modality", "claim_type", "confidence_text", "evidence_span"):
                if k in item:
                    out[k] = item.get(k)
            return out

        return None

    def _looks_binaryish(self, s: str) -> bool:
        low = (s or "").lower()
        if any(m in low for m in self._BINARY_MARKERS):
            return True
        if len(s) >= 120 and (self._printable_ratio(s) < 0.78):
            return True
        words = self._WORD_RE.findall(s)
        if len(s) >= 120 and len(words) <= 3:
            return True
        return False

    def _printable_ratio(self, s: str) -> float:
        printable = sum(1 for ch in s if (32 <= ord(ch) <= 126) or ch in "\n\r\t")
        return printable / max(1, len(s))

    def _normalize_truth_label(self, truth_label: str, domain: str, text: str) -> str:
        tl = str(truth_label or "unknown").strip().lower()

        if domain == "literature":
            if tl in {"unknown", "uncertain"}:
                if '"' in text or "“" in text or "’" in text or "—" in text:
                    return "quote"
                return "fictional"
            return tl

        return tl

    def _normalize_memory_kind(
        self,
        memory_kind_hint: str,
        truth_label: str,
        domain: str,
        claim_type: Optional[str],
    ) -> str:
        mk = str(memory_kind_hint or "unresolved").strip().lower()
        ct = str(claim_type or "").strip().lower()

        if mk in {"fact", "summary", "procedure", "reflection", "scenario", "rule", "unresolved"}:
            return mk

        if domain == "literature" and truth_label in {"fictional", "quote"}:
            return "summary"
        if ct in {"instruction"}:
            return "procedure"
        return "fact"

    def _normalize_status_hint(
        self,
        verification_status_hint: str,
        *,
        truth_label: str,
        domain: str,
        source_trust_hint: str,
        ingest_route: str,
    ) -> str:
        s = str(verification_status_hint or "unverified").strip().lower()

        if s in {"verified", "unverified", "provisional", "contested", "rejected"}:
            if s == "rejected":
                return "contested"
            if ingest_route in {"document_claim", "literary_summary", "literary_frame"} and s == "verified":
                return "unverified"
            return s

        if ingest_route == "real_world_fact":
            if source_trust_hint == "trusted" and truth_label == "likely_true":
                return "verified"
            if truth_label == "likely_false":
                return "contested"
            if truth_label in {"uncertain", "unknown"}:
                return "provisional"
            return "unverified"

        if domain == "literature" or ingest_route in {"document_claim", "literary_summary", "literary_frame"}:
            return "unverified"

        return "unverified"

    def _infer_mem_type(
        self,
        *,
        final_kind: str,
        domain: str,
        truth_label: str,
        claim_type: Optional[str],
        ingest_route: str,
    ) -> str:
        ct = str(claim_type or "").strip().lower()

        if final_kind == "summary":
            if domain == "literature":
                return "claim"
            return "summary"
        if final_kind == "procedure":
            return "procedure"
        if final_kind == "rule":
            return "policy"
        if final_kind == "reflection":
            return "feedback"
        if final_kind == "scenario":
            return "scenario"
        if ct:
            return ct if ct in {"claim", "fact", "policy", "feedback", "scenario"} else "claim"
        if truth_label in {"fictional", "quote"} and domain == "literature":
            return "claim"
        if ingest_route == "document_claim":
            return "claim"
        return "claim"

    def _score_policy(
        self,
        *,
        truth_label: str,
        default_weight: float,
        default_conf: float,
        truth_conf: float,
        domain: str,
        memory_kind_hint: str,
        verification_status_hint: str,
        ingest_route: str,
    ) -> Tuple[float, float]:
        tl = str(truth_label or "unknown").lower()
        mk = str(memory_kind_hint or "unresolved").lower()
        vs = str(verification_status_hint or "unverified").lower()

        mem_conf = max(0.25, min(0.95, float(truth_conf if truth_conf is not None else default_conf)))
        base_weight = float(default_weight)

        if ingest_route in {"literary_summary", "literary_frame", "document_claim"}:
            base_weight = max(0.45, min(0.86, base_weight))
            mem_conf = max(0.55, min(0.88, mem_conf))

            if mk == "summary":
                base_weight = max(base_weight, 0.70)
                mem_conf = max(mem_conf, 0.72)

            if vs == "contested":
                mem_conf = min(mem_conf, 0.50)

            return float(base_weight), float(mem_conf)

        if domain == "literature":
            if mk == "summary":
                base_weight = max(0.70, min(0.88, base_weight))
                mem_conf = max(mem_conf, 0.72)
            elif tl in {"fictional", "quote", "metaphor", "opinion"}:
                base_weight = max(0.45, min(0.68, base_weight))
            elif tl == "likely_true":
                base_weight = max(0.55, min(0.80, base_weight))
            else:
                base_weight = max(0.45, min(0.70, base_weight))

            if vs == "verified":
                mem_conf = max(mem_conf, 0.85)
            elif vs == "contested":
                mem_conf = min(mem_conf, 0.50)

            return float(base_weight), float(mem_conf)

        if tl == "likely_true":
            base_weight = min(1.0, base_weight + 0.10)
        elif tl in {"fictional", "opinion", "metaphor"}:
            base_weight = max(0.35, base_weight - 0.15)
        elif tl in {"uncertain", "unknown", "quote", "question", "instruction"}:
            base_weight = max(0.40, base_weight - 0.10)

        if mk == "rule":
            base_weight = max(base_weight, 0.72)
        elif mk == "procedure":
            base_weight = max(base_weight, 0.64)
        elif mk == "reflection":
            base_weight = min(base_weight, 0.55)

        if vs == "verified":
            mem_conf = max(mem_conf, 0.88)
        elif vs == "contested":
            mem_conf = min(mem_conf, 0.50)

        return float(base_weight), float(mem_conf)

    # ------------------------------------------------------------------
    # Ingest route / scope helpers
    # ------------------------------------------------------------------
    def _tokenize(self, text: str) -> List[str]:
        return [t.lower() for t in self._WORD_RE.findall(text or "")]

    def _looks_realworld_numeric_fact(self, text: str) -> bool:
        low = (text or "").lower().strip()
        toks = set(self._tokenize(low))
        if not self._NUM_RE.search(low):
            return False
        if toks & self._LITERATURE_ONLY_CUES:
            return False
        return bool(toks & self._REALWORLD_NUMERIC_MARKERS)

    def _looks_document_scoped_claim(self, text: str, doc_meta: Dict[str, Any]) -> bool:
        low = (text or "").lower().strip()
        if not low:
            return False

        if any(cue in low for cue in self._DOCUMENT_SCOPED_CUES):
            return True

        chapter_hint = self._safe_meta_str(doc_meta.get("chapter_hint"))
        section_hint = self._safe_meta_str(doc_meta.get("section_hint"))
        title = self._safe_meta_str(doc_meta.get("title")).lower()

        if chapter_hint and f"chapter {chapter_hint}" in low:
            return True
        if section_hint and f"section {section_hint}" in low:
            return True
        if title and title.lower() in low and len(title) > 4:
            return True

        return False

    def _infer_ingest_route(
        self,
        *,
        claim_text: str,
        domain: str,
        truth_label: str,
        final_kind: str,
        claim_type: Optional[str],
        doc_meta: Dict[str, Any],
    ) -> str:
        ct = str(claim_type or "").strip().lower()

        if final_kind == "procedure" or ct == "instruction":
            return "procedure"
        if final_kind == "rule":
            return "rule"
        if final_kind == "reflection":
            return "reflection"
        if final_kind == "scenario":
            return "scenario"

        if self._looks_realworld_numeric_fact(claim_text):
            return "real_world_fact"

        if domain == "literature":
            if final_kind == "summary" or truth_label in {"fictional", "quote", "metaphor"}:
                return "document_claim"

        if self._looks_document_scoped_claim(claim_text, doc_meta):
            return "document_claim"

        if domain in {"science", "general"} and truth_label in {"likely_true", "unknown", "uncertain"}:
            return "real_world_fact"

        return "document_claim" if domain == "literature" else "real_world_fact"

    def _infer_scope(self, *, ingest_route: str, domain: str, truth_label: str) -> str:
        if ingest_route in {"document_claim", "literary_summary", "literary_frame"}:
            return "document_scoped"
        if ingest_route == "real_world_fact":
            return "global"
        if domain == "literature" and truth_label in {"fictional", "quote", "metaphor"}:
            return "document_scoped"
        return "global"

    def _infer_storage_class(self, *, ingest_route: str, final_kind: str, domain: str) -> str:
        if ingest_route in {"document_claim", "literary_summary", "literary_frame"}:
            return "semantic_document"
        if final_kind in {"procedure", "rule"}:
            return "semantic_operational"
        if final_kind in {"reflection", "scenario"}:
            return "semantic_contextual"
        if domain == "literature":
            return "semantic_document"
        return "semantic_global"

    def _safe_meta_str(self, value: Any) -> str:
        if value is None:
            return ""
        try:
            return str(value).strip()
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Real-world factual override
    # ------------------------------------------------------------------
    def _guess_realworld_domain(self, text: str, current_domain: str) -> str:
        low = (text or "").lower()
        if "water" in low and ("boil" in low or "boiling" in low):
            return "science"
        if "everest" in low:
            return "science"
        if "bone" in low and "human" in low:
            return "science"
        if current_domain and current_domain != "literature":
            return current_domain
        return "general"

    def _realworld_fact_override(
        self,
        *,
        claim_text: str,
        current_domain: str,
        current_kind: str,
        current_truth: str,
        current_status: str,
        current_route: bool,
    ) -> Optional[Dict[str, Any]]:
        if not self._looks_realworld_numeric_fact(claim_text):
            return None

        domain_hint = self._guess_realworld_domain(claim_text, current_domain)
        extra_tags = ["realworld_numeric", "fact_override"]

        status = str(current_status or "unverified").strip().lower()
        if status not in {"verified", "contested"}:
            status = "provisional"

        truth = str(current_truth or "unknown").strip().lower()
        if truth in {"fictional", "quote", "metaphor"}:
            truth = "unknown"

        return {
            "domain_hint": domain_hint,
            "memory_kind_hint": "fact",
            "truth_label": truth,
            "verification_status_hint": status,
            "should_route_to_verifier": True or current_route,
            "source_trust_hint": "normal",
            "extra_tags": extra_tags,
            "reasons": ["real-world numeric fact override applied"],
        }