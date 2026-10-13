# Bounded Epistemic Inquiry for Local-First Cognitive Agents

**Blake Hilton · Independent researcher · Project Clair · 13 October 2026**  
**ORCID:** https://orcid.org/0009-0009-9187-8508  
**Status:** Public research working draft; not peer reviewed.

> **Evidence notice:** Methods proposal and bounded prototype evidence; held-out generalization study still required.  
> **Format notice:** This is a full-text, layout-simplified mirror transcribed from the October 13 research-collection PDF. Some figure placement, mathematical typesetting, and table formatting may differ from the editable Word/PDF master files. It is not a DOI-registered publication.

---

Bounded Epistemic Inquiry for Local-First Cognitive Agents
Converting uncertainty into auditable, finite information-seeking operations

Blake Hilton | Independent Researcher and System Architect | Project Clair

ORCID: https://orcid.org/0009-0009-9187-8508 | Evidence cutoff: 13 October 2026

WORKING RESEARCH PREPRINT

PUBLICATION-SAFE WORKING PREPRINT. Conceptual and externally observable behavior is
described. This document intentionally omits nonpublic schemas, thresholds, private filesystem
paths, secrets, gate algorithms, and reconstruction-ready implementation details. This text is a
research draft for author review; it is not evidence of peer review, acceptance, novelty priority, or
completed cross-system validation.


Abstract
An assistant that encounters missing or conflicting information must decide whether to retrieve,
ask, defer, or decline. Project Clair studies the control-plane design of that decision using an explicit
epistemic workspace that represents target constraints, evidence candidates, unresolved gaps, and
finite inquiry budgets. The architecture treats language-model proposals and tool outputs as
observations rather than automatic truth. Early workspace integration was deliberately
observation-first, allowing state-representation and refusal contracts to be tested before live
recursive search authority. We specify a finite-state research model, describe development-level
checkpoints, and propose an evaluation of correct answer, correct abstention, unsupported closure,
and cost. The work is an architectural proposal, not a validated theorem of optimal search or
general information-gathering success.

Keywords: epistemic uncertainty; bounded agents; recursive inquiry; evidence binding; abstention;
local-first AI


1. The problem of knowing when to look again
A one-pass assistant may fill a missing fact with plausible text; an unrestricted agent may continue
exploring indefinitely. Both approaches conflate task completion with confident narration. A useful
middle ground is a structured state that remembers *which part of the question remains
unresolved*, whether a permitted observation is likely to help, and what resources remain
available.

A missing phone number, an ambiguous legal entity, a conflicting manual table, and a question
about the user's private preference are all epistemic gaps, but they require different authorities.
Public search may help with the first, entity disambiguation with the second, document evidence
with the third, and user confirmation with the fourth. A general-purpose search loop that ignores
those distinctions cannot be considered reliable.

2. Research formulation
We represent a bounded inquiry episode as W = (T, K, G, E, P, B, H), where T is the target task and
user-authorized source scope, K the currently established knowledge, G an explicitly unresolved gap
set, E candidate observations and their lineage, P applicable policies and permissions, B a finite
budget, and H the transition history. The controller's action belongs to a permitted set: answer,
inspect local memory, consult the specified document, seek authorized external evidence, ask the
user, defer, or decline.

A transition may be made only when it respects P and consumes a declared amount of B. New
observations can close a gap, split a gap into subproblems, expose a conflict, or fail to improve the
state. A stopping rule terminates when the answer gate accepts adequate evidence, no useful
permitted route remains, or resources are exhausted. These are conceptual contracts; detailed
proprietary thresholds and scoring rules are intentionally omitted.

2.1 Why uncertainty is state, not style

A model can say 'I'm uncertain' after outputting an answer it already selected. An epistemic
workspace must make uncertainty consequential before the final response: it can block authority,
route inquiry, or require deferral. That design offers stronger evaluation targets than subjective
impressions of cautious wording.

2.2 Observation does not imply verification

An external search result, document excerpt, or locally retrieved episode changes the set of
observations, not necessarily the set of verified claims. A result is usable only if it supports the
target entity, geography, attribute, and source requirement. A larger result set can increase noise;
quantity alone cannot discharge an epistemic gap.


3. Relationship to previous systems
ReAct-style language agents combine reasoning and tool actions; CoALA (Sumers et al., 2024) frames
modular agent memory and decisions; Global Workspace Agents (Shang, 2026) examine centrally
coordinated sustained computation. Project Clair's proposed emphasis is distinct: a bounded
workspace that carries source and task authority, explicit unknowns, and policy-limited transitions
across modules while allowing the local language model to be optional. These are related
architectural choices, not proof that the approach is unprecedented.


4. Development chronology
At the August 2026 stage, the epistemic workspace and observation adapters were intentionally
introduced without unrestricted recursive activation. The adapter could represent unusable results
as provisional candidates and unresolved gaps rather than granting them immediate answer
authority. Earlier demonstrations of company contact lookup showed why entity and geography
binding are necessary: records may contain syntactically valid phone-like strings while supplying
no reliable answer to the requested organization's attribute.

Subsequent experience and document work reinforced the same principle. A 12 October live
document refusal failure showed that when an alternate output route is not governed by the same
source contract, an epistemic boundary can still be bypassed. The October 13 checkpoint's isolated
experience-retrieval bridge was not yet connected as live reasoning authority.

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


5. Experimental design
Create a benchmark with both satisfiable and unsatisfiable information needs. Seed ambiguous
entities, wrong locations, adversarially plausible numbers, duplicate names, stale records, and
incompatible sources. For each task record the gold answer or a justified 'unknown' label with
source-scope constraints. Compare one-pass generation, unbounded retry within a high resource
ceiling, bounded inquiry without typed source authority, and bounded inquiry with typed
authority. Match available models and tools.

The primary outcome is jointly reported supported resolution rate and unsupported closure
rate. Additional measurements include correct abstention, false abstention, tool calls, latency,
source-lineage retention, percentage of cases exceeding a budget, and stability across restart. A
finite budget should not be awarded a high score merely because it avoids difficult tasks. Its
purpose is to trade off cost, accuracy, and justified non-resolution transparently.

5.1 Perturbation and negative controls

-   Result containing the right number but wrong entity.
-   Result matching the entity but wrong municipality.
-   Correct entity and location but mismatched requested attribute.
-   High-confidence model answer with no inspectable source.
-   Document evidence that contradicts previously remembered information.
-   Missing or corrupted lineage fields on an otherwise relevant candidate.
-   An authenticated user instruction explicitly changing the task's source scope.

5.2 Generalization check

Hold out entity classes, document types, and tool providers from development. Freeze the code
before evaluation. Release counts of attempts, accepted answers, rejected answers, and error
categories, not merely a single accuracy figure. Evaluate whether the controller's choice of next
route changes appropriately when a relevant tool is unavailable.


6. Limits and failure handling
A bounded workspace cannot resolve a genuinely unanswerable question. It can also perpetuate
error if the initial target is incorrectly represented. User-provided context can be ambiguous, and
verification rules can be brittle in unfamiliar domains. The research must measure the cost of such
rigidity as well as the reliability benefit. A local-first system may avoid network dependence for
baseline operations while still lacking information that only external sources can provide.


7. Conclusion
The contribution proposed here is a governance model for finite information seeking. Recursive
inquiry is useful only when each recurrence inherits the task target, consumes a bounded resource,
and updates a justified epistemic state. Experimental success will require showing that explicit gaps
and authority controls reduce unsupported closure without destroying useful answer coverage.


References
1. Sumers, T. R., Yao, S., Narasimhan, K., & Griffiths, T. L. (2024). Cognitive Architectures for
    Language Agents. Transactions on Machine Learning Research. https://arxiv.org/abs/2309.02427
2. Shang, W. (2026). “Theater of Mind” for LLMs: A Cognitive Architecture Based on Global
    Workspace Theory. arXiv:2604.08206. https://arxiv.org/abs/2604.08206
3. Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
    NeurIPS. https://arxiv.org/abs/2005.11401
4. Shinn, N., et al. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning.
    NeurIPS. https://arxiv.org/abs/2303.11366
5. Autio, C., et al. (2024). Artificial Intelligence Risk Management Framework: Generative
    Artificial Intelligence Profile, NIST AI 600-1. https://doi.org/10.6028/NIST.AI.600-1
6. Gallo, N. (2026). Proof-of-Continuity: A Temporal Model for Authority Propagation in
    Distributed Systems and AI Agents. arXiv:2607.08906. https://arxiv.org/abs/2607.08906
