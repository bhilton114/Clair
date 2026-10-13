# Cross-Authority State Continuity in Modular Cognitive Systems

**Blake Hilton · Independent researcher · Project Clair · 13 October 2026**  
**ORCID:** https://orcid.org/0009-0009-9187-8508  
**Status:** Public research working draft; not peer reviewed.

> **Evidence notice:** Formal research proposal and implementation observations; the paired baseline-versus-continuity ablation is not complete.  
> **Format notice:** This is a full-text, layout-simplified mirror transcribed from the October 13 research-collection PDF. Some figure placement, mathematical typesetting, and table formatting may differ from the editable Word/PDF master files. It is not a DOI-registered publication.

---

Cross-Authority State Continuity in Modular Cognitive
Systems
Revised manuscript from the author's existing "publication1.txt" draft

Blake Hilton | Independent Researcher | Project Clair | 13 October 2026

ORCID: https://orcid.org/0009-0009-9187-8508

PUBLICATION-SAFE WORKING PREPRINT. Conceptual and externally observable behavior is
described. This document intentionally omits nonpublic schemas, thresholds, private filesystem
paths, secrets, gate algorithms, and reconstruction-ready implementation details. This text is a
research draft for author review; it is not evidence of peer review, acceptance, novelty priority, or
completed cross-system validation.

Editorial note: Sections 1–15 preserve the argument and study plan of the earlier manuscript; the
October supplement below corrects the formalization and adds bounded engineering observations.
No claim of experimental superiority is implied.


Abstract
Modular artificial cognitive systems increasingly distribute perception, memory, reasoning,
retrieval, verification, planning, and response generation across specialized components. While
modularity improves separation of concerns and allows individual subsystems to be tested
independently, it introduces a less obvious failure mode: a system may remain locally correct at
each subsystem boundary while becoming globally incorrect as the semantic authority of the
original task is reconstructed, weakened, substituted, or lost across transitions.

This paper introduces cross-authority state continuity, a systems principle requiring authority-
bearing elements of cognitive state to remain traceably preserved or explicitly revised as processing
moves between modules. We distinguish task authority from computational state and execution
provenance, and identify several forms of authority-bearing state, including entity identity,
requested attribute, geographic scope, evidence provenance, verification state, and epistemic status.

We propose an Authority Continuity Invariant under which downstream cognitive operations
may preserve, narrow, reject, or evidence-justify changes to inherited authority, but may not
silently manufacture or broaden authority required to complete a task. The paper develops a formal
representation of this constraint, describes failure classes caused by authority reconstruction at
subsystem seams, and proposes a controlled evaluation methodology comparing implicit authority
reconstruction against explicit authority-continuity propagation.

The central hypothesis is that explicit preservation of task authority reduces false resolution,
unsupported claims, semantic target substitution, provenance loss, and globally incorrect outputs
even when the constituent modules remain individually contract-correct. The work is motivated by
observed failures in a modular local-first cognitive architecture and is intended as a general
systems principle applicable beyond the originating implementation.

1. Introduction
Modern artificial intelligence systems increasingly consist not of a single inference procedure but of
collections of specialized computational components. A single request may pass through intent
recognition, entity extraction, memory retrieval, planning, tool selection, external information
acquisition, evidence scoring, verification, contradiction handling, calibration, and answer
generation before producing an output.

This decomposition provides substantial engineering advantages. Modules can have narrow
responsibilities, explicit contracts, isolated testing, and replaceable implementations. Contemporary
language-agent architectures similarly organize cognition around structured memory, actions,
shared workspaces, specialized agents, and generalized control processes [1,2].

However, modular correctness does not necessarily imply end-to-end cognitive correctness.

Consider a request seeking a particular attribute of a particular entity within a particular
geographic scope. One subsystem may correctly identify the intended entity. Another may correctly
retrieve candidate evidence. A third may correctly recognize a valid instance of the requested
attribute. A fourth may correctly verify that the candidate originates from a credible source.

Nevertheless, the final answer may still be wrong.

This can occur when different components independently reconstruct what the request means
rather than inheriting a common authoritative representation of that meaning.

For example:
   Original request:

     entity    =E

     geography = G

     attribute = A

   Module 1:

     correctly identifies E

   Module 2:

     retrieves evidence associated with E'

   Module 3:

     correctly identifies an instance of A

   Module 4:

     verifies the evidence source

   Final result:

     verified A belonging to E'

Every local operation may satisfy its immediate contract while the global cognitive result violates
the original request.

We refer to this class of failure as cross-authority drift.

The central argument of this paper is that cognitive architectures require not merely shared
computational state but explicit continuity of the authority that gives that state meaning.


2. Problem Definition

2.1 Computational State and Authority State

Let the complete cognitive state at transition (t) be:

S_t = (C_t, A_t)

where:

-   (C_t) represents ordinary computational state;
-   (A_t) represents authority-bearing cognitive state.

Computational state may include temporary values, candidate results, counters, scores,
intermediate representations, or tool outputs.

Authority state represents the conditions under which those values may legitimately contribute to
resolving the task.

For a factual retrieval task, authority state might include:

A(t) = {entity, geography, requested attribute, provenance, verification, epistemic state}

where:

-   (E) = entity authority;
-   (G) = geographic or contextual authority;
-   (R) = requested relation or attribute;
-   (P) = provenance authority;
-   (V) = verification authority;
-   (Q) = epistemic qualification.

The distinction matters because possession of a value does not establish authority to use that value
as the answer.

A subsystem can possess:
    479-555-1234

without having established:
    entity      = requested entity

    location     = requested location

    attribute    = requested phone type

    source      = bound source

     verification = sufficient


2.2 Authority Reconstruction

We define authority reconstruction as a transition in which a downstream subsystem
independently infers authority-bearing state that should have been inherited or explicitly justified.

Let module (M_i) perform:

S_{t+1}=M_i(S_t)

An unsafe reconstruction occurs when:

Unsafe expansion: proposed authority not justified by inherited authority and J(i).

where (J_i) represents authority justified by newly acquired evidence during operation (M_i).

In plain terms:
     A downstream component must not acquire more authority than it inherited unless it can identify the evidence
     or rule that justifies the increase.


3. Authority Continuity Invariant
We propose the following invariant.

Authority Continuity Invariant

For every cognitive transition:

S_t → M_i → S_{t+1}

any change to authority-bearing state must belong to one of four classes:

1. Preservation

    Existing authority is carried forward unchanged.

2. Narrowing

    Existing authority is restricted.

3. Rejection or uncertainty

    Existing authority is invalidated, disputed, or reduced in confidence.

4. Evidence-justified extension

    New authority is introduced together with an explicit justification.

Therefore:

Proposed shorthand: A(t+1) ⊆ A(t) ∪ J(i). See typed formalization in Section 16.

subject to explicit revision records.

A module must not silently:

-     substitute another entity;

-   broaden geographic scope;
-   change the requested attribute;
-   convert provisional evidence into verified evidence;
-   detach a claim from its provenance;
-   treat absence of binding information as positive binding;
-   reconstruct missing authority merely because execution requires it.


4. Failure Classes
We identify several recurring classes of cross-authority failure.

4.1 Entity Substitution

Evidence belongs to an entity other than the original target, but downstream processing treats the
candidate as bound to the target.

4.2 Geographic Drift

The correct attribute is retrieved for the wrong branch, city, jurisdiction, or region.

4.3 Attribute Substitution

A valid value of one attribute is mistaken for the requested attribute.

Examples include:
    main office phone → customer service phone

    fax number      → telephone

    mailing address → physical address


4.4 Provenance Detachment

A claim survives while the evidence that originally justified the claim is discarded or no longer
associated with it.

4.5 Verification Inflation

A downstream component interprets the existence of a validated value as validation of the entire
claim.

For example:
    valid telephone syntax

does not imply:
    verified telephone number

    for the requested organization

    at the requested location.

4.6 Epistemic Promotion Without Authority

A candidate progresses from:
      possible

to:
      supported

or:
      verified

without an explicit state transition justified by evidence.


5. Relationship to Existing Work
This proposal intersects several established research directions but addresses a different layer of
the problem.

Cognitive Architectures for Language Agents (CoALA) formalizes language agents in terms of
modular memory, structured action spaces, and generalized decision processes [1].

Global Workspace Agent architectures investigate centralized shared cognitive state and
coordination among specialized processors [2].

Recent reasoning-provenance work argues that computational checkpoints alone do not preserve
the reasons, observations, and evidence relationships underlying agent decisions [3].

Proof-of-Continuity introduces a related non-expansion principle for propagation of authorization
authority through distributed execution chains [4]. That work concerns authorization and causal
privilege propagation, whereas the present work concerns preservation of semantic and epistemic
task authority across cognitive processing boundaries.

The distinction is important:
      Authorization authority:

        "What is this component allowed to do?"

      Task authority:

        "What does the current evidence actually

        authorize the system to conclude?"

The two share a structural concern with continuity but govern different objects.


6. Research Questions
The proposed evaluation addresses four primary questions.

RQ1

Can individually contract-correct cognitive modules produce globally incorrect outputs when
authority-bearing state is reconstructed independently?

RQ2

Does explicit propagation of authority state reduce these failures?

RQ3

Which forms of authority drift contribute most strongly to incorrect task resolution?

RQ4

What computational and implementation overhead is introduced by explicit authority continuity?


7. Hypotheses

H1: Global Failure Under Local Correctness

A modular system using implicit authority reconstruction will exhibit measurable cases in which all
participating modules satisfy their local contracts but the final answer violates the original task
constraints.

H2: Continuity Reduces False Resolution

Explicit authority continuity will reduce false-resolution rate relative to implicit reconstruction.

H3: Continuity Improves Rejection

Explicit continuity will increase correct rejection or uncertainty reporting when required authority
cannot be established.

H4: Continuity Preserves Provenance

Explicit authority propagation will reduce the frequency with which final claims become detached
from the evidence supporting them.

H5: Continuity Has Bounded Cost

The reliability improvement produced by explicit authority state will be achievable without
requiring unbounded search, model inference, or computational growth.


8. Experimental Design
Two system conditions will be compared.

Condition A: Implicit Authority Reconstruction

Subsystems receive ordinary task and result representations and may locally infer contextual
relationships required for their operations.

Condition B: Explicit Authority Continuity

Subsystems receive an authoritative state object carrying relevant task constraints and may modify
authority-bearing fields only through permitted transition classes.

The same tasks, tools, evidence corpus, and subsystem implementations should be used wherever
possible.

The independent variable is therefore the mechanism by which authority state crosses subsystem
boundaries.


9. Perturbation Suite
Evaluation cases should deliberately introduce ambiguous or incomplete seam conditions,
including:

-   ambiguous entity names;
-   multiple organizations with similar names;
-   incorrect geographic results;
-   multiple candidate values;
-   valid but unrelated values;
-   conflicting sources;
-   missing provenance;
-   unavailable geography;
-   unsupported attribute labels;
-   stale results;
-   partial entity matches;
-   high-confidence but unrelated evidence.

The perturbation suite should contain both positive cases, where resolution is possible, and
negative cases, where the correct outcome is uncertainty or refusal to resolve.


10. Metrics
The principal dependent measures are:

False Resolution Rate

FRR = \frac{\text{incorrect tasks marked resolved}} {\text{all resolved tasks}}

Authority Substitution Rate

ASR = \frac{\text{transitions containing unjustified authority substitution}} {\text{authority-bearing
transitions}}

Target Preservation Rate

The proportion of completed runs in which entity, geography, and requested relation remain
correctly bound.

Provenance Preservation Rate

The proportion of final factual claims retaining a valid evidence lineage.

Correct Rejection Rate

The proportion of intentionally underdetermined cases correctly retained as unresolved,
provisional, or unknown.

End-to-End Accuracy

Task-level correctness measured independently of subsystem-level contract success.

Computational Cost

Where available:

-   number of search attempts;
-   tool calls;
-   inference calls;
-   execution latency;
-   state-transition count.


11. Implementation Case Study
The motivating implementation is a modular, local-first cognitive system containing independently
testable components for intent analysis, memory, reasoning, retrieval, evidence evaluation,
verification, contradiction handling, calibration, and response composition.

During generalized information-retrieval testing, failures were observed in which individual
processing stages remained locally valid while the complete cognitive chain failed to preserve
entity, geographic, attribute, or evidence relationships.

These observations motivated explicit binding and authority-state propagation across subsystem
seams.

For publication, the implementation case study will report only the architectural information
required to reproduce the failure mechanism and evaluate the proposed invariant. Unrelated
proprietary implementation details are outside the scope of the paper.


12. Required Empirical Evidence Before Final Submission
The final manuscript must replace this section with measured results.

At minimum, the evaluation should contain:

1. a fixed test corpus;

2. an explicit baseline condition;
3. an authority-continuity condition;
4. positive and negative controls;
5. seam perturbations;
6. repeated deterministic trials where applicable;
7. exact failure classifications;
8. aggregate measurements;
9. regression results;
10. sufficient implementation description for reproduction.

No improvement claim should be made before these measurements are complete.


13. Threats to Validity
Several limitations must be considered.

First, experiments conducted on a single architecture may not establish that the failure mechanism
occurs equally across all modular cognitive systems.

Second, task classes emphasizing factual retrieval may overrepresent entity and provenance
authority relative to planning or simulation tasks.

Third, deterministic contracts may make authority errors easier to observe than in systems where
state is represented primarily in natural language.

Fourth, introducing explicit state objects may improve performance partly through increased
structure rather than authority continuity specifically. Ablation testing is required to distinguish
these effects.

Finally, successful prevention of the identified failure classes does not establish general intelligence,
human-equivalent cognition, or superiority over existing cognitive architectures.


14. Broader Implications
If supported experimentally, cross-authority continuity suggests that the correctness of modular
cognitive systems depends on more than the correctness of their constituent modules.

The relevant unit of analysis becomes the cognitive transition.

A system must preserve not only data but the constraints governing what that data is permitted to
mean.

This distinction may become increasingly important as artificial cognitive systems distribute
processing across specialized models, tools, memory systems, planners, verifiers, and external
services.

In such architectures, the question is not merely:
    Did each subsystem perform its assigned function correctly?

It is also:
    Did the complete system preserve the authority required for the final conclusion to remain about the same task?

15. Conclusion
This paper proposes cross-authority state continuity as a general correctness principle for modular
cognitive systems.

The principle addresses a failure mode in which individually correct subsystems collectively
produce incorrect conclusions because authority-bearing task state is reconstructed or altered
across subsystem boundaries.

We propose an Authority Continuity Invariant requiring downstream authority to remain non-
expansive except where explicit evidence justifies revision. We further define a controlled
experimental methodology for determining whether this mechanism reduces false resolution,
target substitution, provenance loss, and unsupported epistemic promotion.

The contribution is intentionally narrow. It does not claim a general theory of machine cognition.
Instead, it identifies a specific systems-level correctness problem, formalizes the relevant invariant,
and provides a falsifiable method for evaluating the proposed solution.


References
[1] Sumers, T. R., Yao, S., Narasimhan, K., & Griffiths, T. L. *Cognitive Architectures for Language
Agents*. arXiv:2309.02427, 2023.

[2] Shang, W. *“Theater of Mind” for LLMs: A Cognitive Architecture Based on Global Workspace
Theory*. arXiv:2604.08206, 2026.

[3] Vispute, N. *Reasoning Provenance for Autonomous AI Agents: Structured Behavioral Analytics
Beyond State Checkpoints and Execution Traces*. arXiv:2603.21692, 2026.

[4] Gallo, N. *Proof-of-Continuity: A Temporal Model for Authority Propagation in Distributed
Systems and AI Agents*. arXiv:2607.08906, 2026.


16. October 2026 Revision: Typed Authority Continuity

16.1 Why the original set relation needs refinement

The expression A(t+1) ⊆ A(t) ∪ J(i) is an intuitive shorthand, not a valid enforcement specification
for every authority field. Entity identity, requested relation, source lineage, permission, epistemic
status, and geographic context are heterogeneous typed objects. The set inclusion operator cannot
by itself determine when an attribute was changed, a provisional observation was upgraded, or an
entity match became overbroad. A rigorous implementation must check field-specific predicates
and explicitly recorded justifications.

Let A(t) be a structured authority record containing (T, E, P, V, U), where T is the original task target
and scope, E the bound evidence lineage, P applicable permissions, V the verification state, and U
unresolved uncertainty. Let J(i) be the documented reason and the proof-bearing observation for a
proposed revision at component i. Define AcceptTransition(A(t), A(t+1), J(i)) to hold only if: (a)
original task constraints are preserved or a user-authorized task change is recorded; (b) any change
in entity, relation, or geography has a valid source binding; (c) evidence lineage and permissions do

not silently weaken; and (d) epistemic promotion is supported by the relevant verification rule. This
is a proposal for an executable contract, not a claim of a formal proof of global correctness.

16.2 Distinguishing three authorities

Task authority specifies *which question is being answered*. Evidence authority specifies *what
sources support the proposed answer in that question's scope*. Execution authority specifies *what
effects the component is permitted to commit*. These overlap but must not be merged. A
component may be allowed to query a tool without thereby gaining authority to assert the result. A
document may be readable without its content becoming a durable memory. A correct-looking
extracted attribute may remain unrelated to the user's target.

16.3 October case-study update

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

The 12 October document counterexample is especially relevant. The document-extraction route
could perform correctly yet final output admitted content from an external-lookup route that did
not answer the document-grounded question. The case is evidence of a plausible authority-
continuity defect, not an estimate of frequency. The 13 October isolated ExperienceRetrievalBridge
shows another boundary where retrieved experience must remain advisory until explicit
scope/provenance checks authorize its use. Neither record proves that continuity enforcement
reduces an aggregate error rate.

16.4 Evaluation protocol required for a testable claim

Establish a paired corpus with real positive examples, insufficient-evidence negatives, entity
substitutions, wrong geographies, wrong attributes, provenance truncations, stale evidence, and
high-confidence but irrelevant model completions. Hold the retriever, tool availability, model
settings, and output policy fixed. Compare two branches: typed authority passed across seams
versus structurally equivalent data without admission checks. Pre-register the primary endpoint
(incorrectly resolved tasks / all resolved tasks); report denominator, 95% confidence interval,
severity, correct abstention, and overall coverage. Verify that the system does not trivially improve
false-resolution by abstaining on everything; report utility and false rejection. Re-run with a fresh
session and restart to test state lineage.

A code-execution effect or SQL mutation must additionally be tested against execution authorization
rather than inferred from semantic relevance. A falsifying result would be no meaningful reduction
in target substitution after controlling for additional structure, or unacceptable reduction in correct
resolution for benign cases.

16.5 Related-work clarification

The contemporaneous Continuity Kernel paper by He and Yu (2026) addresses atomic activation of
agent state, and Gallo (2026) addresses non-expansive execution authority along causal chains.
Those works are independent and not Project Clair publications. The present hypothesis concerns
continuity of a user's semantic target and the evidentiary authority for the answer across modular
cognition. The distinction is substantive only if the experiments operationalize and measure it. This
manuscript makes no priority or novelty claim over undiscovered prior work.

16.6 Release status

The manuscript now contains an operationally refined invariant and an explicit October case study.
It remains a methods / hypothesis preprint until the ablation dataset, protocol, repeated runs, and
aggregate results are attached. Exact proprietary gate implementation remains withheld.


Additional October references
-   He, J., & Yu, D. (2026). Beyond Memory: A Transactional Continuity Kernel for Long-Lived AI
    Agents. arXiv:2608.11632. https://arxiv.org/abs/2608.11632
-   Gallo, N. (2026). Proof-of-Continuity: A Temporal Model for Authority Propagation in Distributed
    Systems and AI Agents. arXiv:2607.08906. https://arxiv.org/abs/2607.08906
-   Vispute, N. (2026). Reasoning Provenance for Autonomous AI Agents: Structured Behavioral
    Analytics Beyond State Checkpoints and Execution Traces. arXiv:2603.21692.
    https://arxiv.org/abs/2603.21692
-   Sumers, T. R., Yao, S., Narasimhan, K., & Griffiths, T. L. (2024). Cognitive Architectures for
    Language Agents. Transactions on Machine Learning Research. https://arxiv.org/abs/2309.02427
-   Autio, C., et al. (2024). Artificial Intelligence Risk Management Framework: Generative Artificial
    Intelligence Profile, NIST AI 600-1. https://doi.org/10.6028/NIST.AI.600-1
