# Governed Experience Learning in a Persistent Local AI Agent

**Blake Hilton · Independent researcher · Project Clair · 13 October 2026**  
**ORCID:** https://orcid.org/0009-0009-9187-8508  
**Status:** Public research working draft; not peer reviewed.

> **Evidence notice:** Concept and subsystem regressions; controlled held-out cross-session learning improvement is not established.  
> **Format notice:** This is a full-text, layout-simplified mirror transcribed from the October 13 research-collection PDF. Some figure placement, mathematical typesetting, and table formatting may differ from the editable Word/PDF master files. It is not a DOI-registered publication.

---

Governed Experience Learning in a Persistent Local AI
Agent
Separating episodic retention, retrieval, lesson proposals, and truth authority

Blake Hilton | Independent Researcher and System Architect | Project Clair

ORCID: https://orcid.org/0009-0009-9187-8508 | Evidence cutoff: 13 October 2026

WORKING RESEARCH PREPRINT

PUBLICATION-SAFE WORKING PREPRINT. Conceptual and externally observable behavior is
described. This document intentionally omits nonpublic schemas, thresholds, private filesystem
paths, secrets, gate algorithms, and reconstruction-ready implementation details. This text is a
research draft for author review; it is not evidence of peer review, acceptance, novelty priority, or
completed cross-system validation.


Abstract
Persistent agents need to exploit past outcomes without mistaking remembered text or model-
generated reflection for authoritative knowledge. This paper presents a design and evaluation
protocol for governed experience in Project Clair, a local-first cognitive research prototype. The
proposed separation is between retaining an episode, retrieving a contextually eligible trace,
proposing a lesson, and authorizing its influence over future reasoning. We report development-
level evidence of durable experience records, rebuild continuity, scoped selection, and fail-closed
retrieval contracts through 13 October 2026. We explicitly do not claim demonstrated end-to-end
learning improvement: the retrieved-experience bridge had not yet been admitted into the live
reasoning path, and a held-out multi-conversation improvement study was pending. The
contribution is therefore a falsifiable architecture for learning under durable-state and authority
constraints, together with a controlled evaluation plan.

Keywords: persistent agents; experiential learning; memory governance; provenance; local-first AI;
calibrated uncertainty


1. Problem and research question
A conversational system can appear to learn simply by restating earlier text. A long-lived agent
requires a stronger definition: an experience must be recorded, remain available across the
intended temporal boundary, be eligible for a relevant new task, and improve a measured outcome
without granting obsolete or fabricated information greater epistemic force. The relevant question
is not whether an episode can be serialized, but whether the system can use it *safely and
beneficially* later.

Three common shortcuts are inadequate. Storing every interaction indiscriminately conflates
observations, private user statements, and facts. Retrieving high-similarity memories ignores scope
and changing ground truth. Promoting a fluent reflective summary to a rule lets the generator

authorize its own policy. The engineering response is to separate these responsibilities into
bounded lifecycle decisions.


2. Relation to prior work
Generative Agents (Park et al., 2023) combine memory, reflection, and planning to support coherent
longer-term behaviors. Reflexion (Shinn et al., 2023) demonstrates that linguistic feedback can
improve later attempts without gradient updates. MemGPT (Packer et al., 2023) manages distinct
memory tiers and extended interaction beyond a single model context. These works motivate
persistence and re-use, but do not remove the deployment-specific need to specify *when a
retrieved memory is permissible evidence* for a fresh question. CoALA (Sumers et al., 2024) offers a
taxonomy of memory and action that supports modular description; Project Clair emphasizes
explicit non-self-authorizing transitions between these components.

The phrase 'learning' is reserved here for a causal, prospective advantage under controlled
conditions. Persisting or surfacing a record alone is not sufficient.


3. Architectural claim
The architectural proposal is a four-stage experience plane. Observation creates an episode from
bounded event, action, and outcome context. Retention accepts a durable record only with
applicable permission, owner scope, origin, timestamps, and integrity checks. Retrieval filters
candidate episodes according to relevance and admissible scope; a result is advisory, not a verified
fact. Consumption enables a downstream answer or plan to consider the retrieved trace, while the
existing answer and action gates retain sole authority over final outputs and durable promotion.

This is an authority separation, not a requirement that every task consult history. Some tasks
should ignore episodes because they are private, stale, disputed, unrelated, or contradicted by
higher-quality fresh evidence. A correct rejection is a positive outcome if the episode would
otherwise contaminate the answer.

3.1 Experience is not long-term truth

An episode describes 'what the system observed or did at a time and scope.' A durable factual belief
describes 'what may be asserted now under evidence policy.' These are different types of claim. An
episode can be accurate yet irrelevant, or useful for procedural planning yet unsuitable to establish
a current external fact. Retention policy must therefore not shortcut verification, memory consent,
or source attribution.

3.2 The retrieval bridge as an untrusted producer

The October implementation isolates an ExperienceRetrievalBridge from the live reasoning/server
pathway. This permits the developer to exercise selection, failure modes, and typed consumers
without implicitly allowing retrieved text to answer users. A retrieval consumer should refuse
malformed records, missing required fields, unauthenticated provenance, ambiguous owners, and
scope mismatch. The bridge's success is a component contract until end-to-end integration is
observed.

3.3 An authority-safe lesson proposal

Reflection may propose that a previous decision strategy was useful, or that a pattern should be
avoided. Promotion requires separate evidence of outcome, target domain, reversibility, and lack of
contradiction. Even an accepted operational lesson cannot assert a new external fact without the
normal evidentiary route. A lesson may influence search order or a simulator's candidates while
remaining barred from independent answer authority.


4. Existing evidence and evidence boundary
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

This record establishes a meaningful engineering progression: episodes persist, can be recovered
within tested rebuilds, and have been exercised through scoped selection and rejection tests. It does
not measure improved reasoning compared with the same system without experience. It also does
not prove a low false-promotion rate over a realistic workload, resistance to malicious episodes, or
long-horizon stability under software upgrades.


5. Evaluation design
The principal experiment is paired, held-out, cross-session task transfer. A frozen task family
contains learning opportunities with a verified outcome in session A and structurally related but
non-identical tasks in session B after a clean restart. Compare (i) no experience retrieval, (ii)
ungoverned similarity-based retrieval, and (iii) the governed experience bridge, under identical
models, tools, prompts, and source corpora. Vary whether a prior episode is relevant, irrelevant,
contradictory, private, stale, deliberately poisoned, or absent.

The primary metric is change in held-out task correctness after the system has accumulated an
eligible experience. Secondary measures include unsupported assertion rate, incorrect experience
adoption rate, correct abstention, owner/scope leakage, latency, number of additional tool calls, and
post-restart recall consistency. Report stratified results by episode type, not just aggregate mean. A
retrieval approach that seems to improve raw correctness but raises false assertions should not
count as a governance success.

5.1 Suggested hypotheses and falsifiers

-   H1: Governed retrieval improves accuracy on held-out related tasks relative to no retrieval,
    once live integration exists.
-   H2: Governed retrieval reduces use of unrelated or conflicted episodes compared with
    similarity-only retrieval.
-   H3: Rebuild or restart retains eligible episodes without leaking them across user or conversation
    boundaries.
-   Falsifiers: no accuracy benefit in paired tasks, significant contamination, or a failure to preserve
    eligible records across controlled restarts.

5.2 Pre-registration details

Define episode eligibility before test execution. Randomize task order and blind outcome scoring to
condition. Specify a minimal meaningful improvement and required confidence interval. Separate
handcrafted unit contracts from human or evaluator-rated answer quality. Freeze test corpus
hashes and the software commit. Include a negative control in which an apparently relevant
episode refers to a different entity or geography.


6. Security and deployment tradeoffs
Persistence increases attack surface. A hostile document or user could introduce a narrative that
later resembles a useful lesson. Authentication, scope isolation, provenance binding, and revocation
must therefore be applied to the experience plane, not merely to the original document or chat.
SQLite durability does not imply immutability or tamper evidence by itself. Trigger-enforced
append-only rules, cryptographic receipts, and bounded graph traversal are proposed hardening
directions, not asserted as fully installed in this paper.

Local-first storage increases data control but also creates responsibilities for migration, corruption
recovery, and backups. The system should distinguish physical data persistence from logical
authority preservation, and should test both.


7. Limitations
The study currently relies on development reports rather than a released corpus or independent
replication. Test pass counts cover named contracts and cannot be converted into an end-to-end
learning percentage. At the evidence cutoff, safe consumption in the final reasoning route and the
complete document-to-experience-to-later-answer loop remained unproven. There is no claim of
autonomous self-improvement, weights-based learning, or general intelligence.


8. Conclusion
Governed experience learning treats remembered episodes as potentially useful but non-
authoritative candidates. The architectural contribution is the explicit separation of retention,
retrieval, lesson proposal, and authorized influence. Its scientific value depends on a held-out
causal evaluation demonstrating useful transfer without memory or answer contamination. Project
Clair supplies a prototype testbed for that proposition, not yet a completed empirical proof.

References
1. Park, J. S., et al. (2023). Generative Agents: Interactive Simulacra of Human Behavior. UIST.
    https://arxiv.org/abs/2304.03442
2. Shinn, N., et al. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning.
    NeurIPS. https://arxiv.org/abs/2303.11366
3. Packer, C., et al. (2023). MemGPT: Towards LLMs as Operating Systems.
    https://arxiv.org/abs/2310.08560
4. Sumers, T. R., Yao, S., Narasimhan, K., & Griffiths, T. L. (2024). Cognitive Architectures for
    Language Agents. Transactions on Machine Learning Research. https://arxiv.org/abs/2309.02427
5. He, J., & Yu, D. (2026). Beyond Memory: A Transactional Continuity Kernel for Long-Lived AI
    Agents. arXiv:2608.11632. https://arxiv.org/abs/2608.11632
6. Autio, C., et al. (2024). Artificial Intelligence Risk Management Framework: Generative
    Artificial Intelligence Profile, NIST AI 600-1. https://doi.org/10.6028/NIST.AI.600-1
