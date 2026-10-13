# Document-Grounded Answering as an Authority-Preservation Problem

**Blake Hilton · Independent researcher · Project Clair · 13 October 2026**  
**ORCID:** https://orcid.org/0009-0009-9187-8508  
**Status:** Public research working draft; not peer reviewed.

> **Evidence notice:** Case study includes a documented failed source-boundary check (R84); no claim that end-to-end document authority is solved.  
> **Format notice:** This is a full-text, layout-simplified mirror transcribed from the October 13 research-collection PDF. Some figure placement, mathematical typesetting, and table formatting may differ from the editable Word/PDF master files. It is not a DOI-registered publication.

---

Document-Grounded Answering as an Authority-
Preservation Problem
Source scope, structured extraction, and refusal under evidence conflict

Blake Hilton | Independent Researcher and System Architect | Project Clair

ORCID: https://orcid.org/0009-0009-9187-8508 | Evidence cutoff: 13 October 2026

WORKING RESEARCH PREPRINT

PUBLICATION-SAFE WORKING PREPRINT. Conceptual and externally observable behavior is
described. This document intentionally omits nonpublic schemas, thresholds, private filesystem
paths, secrets, gate algorithms, and reconstruction-ready implementation details. This text is a
research draft for author review; it is not evidence of peer review, acceptance, novelty priority, or
completed cross-system validation.


Abstract
Document question answering is often evaluated by whether relevant text was retrieved and a
plausible answer generated. Project Clair motivates an additional criterion: the response must
preserve authority from the user-designated document through extraction, evidence binding, and
final answer composition. We define document-source substitution as an end-to-end failure in
which an otherwise valid downstream result answers from the wrong evidence channel. The
prototype has passed focused document and tabular-extraction regressions, yet a 12 October 2026
live test accepted unrelated externally retrieved material under a document-grounded request. We
report this counterexample transparently, describe a non-self-authorizing architecture, and
propose negative-control and ablation evaluations. This is a problem-formulation and systems-
methods preprint; reduction in hallucination or improved benchmark accuracy has not been
established.

Keywords: document QA; evidence provenance; source authority; grounded generation; semantic
drift; refusal


1. Introduction
A user may ask an agent to inspect a supplied PDF, spreadsheet, or technical manual, then answer a
question from that document. A conventional retrieval-augmented response pipeline often prizes
answer fluency or text overlap with a passage. But a true document-grounded task includes an
implicit or explicit authorization constraint: *the answer must follow from the specified source and
not a plausible substitute*. If an external web lookup produces a credible but unrelated answer, the
task remains unsolved.

This paper distinguishes three events that are frequently merged: recognizing content in a
document, recognizing that the content is relevant to the user’s question, and authorizing the final
answer as supported by that content. The first can be a parsing success while the last is a
governance failure.

2. Background and related research
Retrieval-Augmented Generation (Lewis et al., 2020) combines parametric generation with an
accessible non-parametric evidence store, improving performance on knowledge-intensive tasks.
However, availability of retrieved passages does not by itself establish that the final claim matches
the requested document. Liu et al. (2024) show that long-context behavior is sensitive to where
relevant information appears. Work such as AgentDojo (Debenedetti et al., 2024) illustrates the
distinct need to treat encountered tool and document content as untrusted input. The current
research asks how a *system outside the generator* can preserve the designated source constraint
and enforce appropriate abstention.


3. Document-source authority model
Let the user’s task be represented as Q = (D, S, R, C), where D identifies the permitted document or
set of documents, S is a requested subsection, row, table, or page context when specified, R is the
requested relation or attribute, and C is the operational context. A candidate answer X is admissible
only if its supporting evidence E belongs to the permitted document scope, is traceably related to R,
has not been invalidated by contradictory or missing context, and passes the independent answer
policy.

This typed representation is a research abstraction. Project Clair's protected implementation may
use different internal structures. The critical distinction is between a valid extracted string and an
authorized response for Q. Source selection is part of task meaning, not a metadata field added after
generation.

3.1 Four failure classes

-   Source substitution: a web result, unrelated memory, or model completion is treated as support
    from the supplied document.
-   Structural misbinding: a value in one table row, column, or heading is attributed to a different
    parameter or device.
-   Semantic role confusion: a label such as 'fax', 'range', 'pressure', or 'valve position' is read
    correctly but assigned the wrong requested relation.
-   Unsupported completion: a summary fills missing steps, units, or conditions using plausible
    domain knowledge that the document did not provide.

The correct behavior can be to answer with a source pointer, request clarification, or state that the
document lacks sufficient support. A wrong-scope source should not silently rescue an otherwise
unsupported document question.


4. Project Clair case record
In 12 October testing, a live document question about a technical control range received an answer
describing a 0–100 position and a direction of bypass/line-pressure change, accompanied by an
evidence confidence value. This demonstrated a bounded successful document-related answer in
one run; it did not establish universal table comprehension. In a subsequent live refusal control, the
agent accepted an unrelated color-palette answer through an `external_lookup` route rather than

refusing the off-document claim. That observed R84 failure is material evidence that the final
answer path could bypass the intended source authority even while targeted tabular extraction
checks were green.

The evidence base is the developer's recorded contract outcomes and live demonstrations, not an
independent benchmark. In a 13 October 2026 development checkpoint, a consolidated, targeted
regression group reported 100/100 passing checks; individual reported groups included an
experience-retrieval bridge (13/13), an authority contract (6/6), a fail-closed consumer (5/5), and
adversarial prototypes (18/18). These are overlapping or distinct suites of unspecified membership
and must not be summed. Earlier checkpoints included 167/167 targeted regressions on 12 October
and 199/199 cumulative scoped checks on 9 October; dates, code revisions, and scopes differ.
Evidence of durable experience persistence across builder rebuilds and a verified local server
identity was also recorded. Crucially, a 12 October live document question exposed an unrelated
answer accepted via an external-lookup path (the R84 failure), showing that end-to-end evidence-
source authority was not yet closed. A 17-case bridge-hardening test and live document-to-
experience-to-later-conversation improvement demonstration remained pending on 13 October. No
independent replication or causal improvement result is claimed.

The juxtaposition matters more than the pass count. Component correctness and some successful
live answers coexisted with a negative-control failure. The scientific claim is therefore a falsifiable
system-level gap, not a completed robustness result.


5. Proposed end-to-end enforcement
An admissible response needs a continuous trail from document identity to extracted region, target
relation, candidate evidence, and final response. The guard can reject a completion or request
additional extraction when required source links are absent. The system may use models or OCR as
*candidate producers* but should not let them grant their own evidence permissions. Optional
fallback routes should be classified as authorized substitutions only after a task-level user decision
or policy-defined change of source scope.

5.1 Source selection and trust

An explicitly named document outranks the convenience of an unrelated general web search for a
document-grounded question. A local document may still contain false statements or prompt-
injection instructions. Binding to it establishes source fidelity, not external factual truth. A high-
fidelity answer might correctly report that the document claims an incorrect thing. Truth about the
world and fidelity to a specified source are separately measurable.

5.2 Treatment of tables and units

A structured document can place qualifiers far from values. A caption, row header, unit, operating
mode, and nearby footnote may all be required to bind a number. The evaluator should capture
those relationships; extracting '0–100' without knowing which parameter, device, or mode it
describes is not sufficient. Negative cases must intentionally permute rows, swap units, and
introduce duplicated numeric values.

6. Experimental protocol
Construct a public or permission-cleared document corpus with plain text, tabular PDFs, DOCX,
spreadsheets, and distractor documents. For each item create supported, unsupported, and
conflicting questions. Specify source constraints in the user prompt. Hold parsers, retrievers, model,
and tool availability constant while switching on a source-authority validation layer. Include seeded
web results whose answer strings are plausible but not present in the designated documents.

The primary endpoint is wrong-source answer admission rate, computed as wrong-source
answers divided by all attempted document-grounded items; report the denominator and
confidence interval. Secondary measures are answer exactness, table-binding accuracy, quote/span
traceability, justified abstention, false refusal, time to answer, and retrieval/tool cost. Evaluate both
positive questions and negative controls, with a separate score for source fidelity versus external
factual correctness.

An ablation should distinguish structural parsing improvements from authority preservation. Run
conditions with (a) parsing only, (b) parsing plus source binding, and (c) parsing plus source
binding plus final response gate. Improvements that vanish in the live user interface but exist in
unit tests are evidence of incomplete integration.


7. Evaluation risks and limitations
This paper does not assert that one authority rule will solve prompt injection, cross-document
contradictions, or OCR uncertainty. Binding can be over-restrictive and reject useful evidence; both
correct refusal and missed correct answers must be tracked. The current negative-control failure
demonstrates one vulnerability but gives no population estimate. A validated repair requires
rerunning the original live failure with a recorded code identity and independent scoring.


8. Conclusion
Document grounding is an authority constraint as well as a retrieval problem. The strongest lesson
of the Project Clair tests is that a system may parse a table correctly and still fail to respect which
evidence it was allowed to use. The next research result must be a controlled, end-to-end reduction
in wrong-source admissions, not an inflated count of successful extractors.


References
1. Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
    NeurIPS. https://arxiv.org/abs/2005.11401
2. Liu, N. F., et al. (2024). Lost in the Middle: How Language Models Use Long Contexts.
    Transactions of the Association for Computational Linguistics, 12, 157–173.
    https://arxiv.org/abs/2307.03172
3. Sumers, T. R., Yao, S., Narasimhan, K., & Griffiths, T. L. (2024). Cognitive Architectures for
    Language Agents. Transactions on Machine Learning Research. https://arxiv.org/abs/2309.02427
4. Debenedetti, E., et al. (2024). AgentDojo: A Dynamic Environment to Evaluate Prompt
    Injection Attacks and Defenses for LLM Agents. arXiv:2406.13352.
    https://arxiv.org/abs/2406.13352

5. Gallo, N. (2026). Proof-of-Continuity: A Temporal Model for Authority Propagation in
   Distributed Systems and AI Agents. arXiv:2607.08906. https://arxiv.org/abs/2607.08906
6. Autio, C., et al. (2024). Artificial Intelligence Risk Management Framework: Generative
   Artificial Intelligence Profile, NIST AI 600-1. https://doi.org/10.6028/NIST.AI.600-1
