# The Recursive Cognitive Continuum: Project Clair and the Architecture of Governed Synthetic Cognition

**Blake Hilton · Independent researcher · Project Clair · 13 October 2026**  
**ORCID:** https://orcid.org/0009-0009-9187-8508  
**Status:** Public research working draft; not peer reviewed.

> **Evidence notice:** Umbrella systems thesis with an October evidence supplement; includes developer-reported internal evidence, not independently verified claims.  
> **Format notice:** This is a full-text, layout-simplified mirror transcribed from the October 13 research-collection PDF. Some figure placement, mathematical typesetting, and table formatting may differ from the editable Word/PDF master files. It is not a DOI-registered publication.

---

THE RECURSIVE COGNITIVE CONTINUUM


Project Clair and the Architecture of
   Governed Synthetic Cognition
 A systems thesis for local-first, evidence-governed, human-authorized
                                synthetic cognition

      Publication-Safe Research Manuscript | October 2026 Supplement


                                        Blake Hilton
                    Independent Researcher and System Architect
                            Project Clair v3.9 research record
            Evidence cutoff: October 13, 2026 (2026 supplement appended)


            EXTERNAL RESEARCH DRAFT - IMPLEMENTATION DETAILS WITHHELD

Author's Scope and Evidence Notice
This manuscript is a publication-safe account of Project Clair through August 14, 2026. It states the
research thesis, system-level contributions, evaluation record, and limitations while withholding exact
thresholds, scoring rules, source layouts, private paths, sensitive schemas, and sufficient detail to
reproduce protected mechanisms. Bracketed notes identify disclosures intentionally abstracted for
intellectual-property and security reasons.

The manuscript reports prototype evidence, not a claim of artificial general intelligence, consciousness,
sentience, or completed autonomy. The phrase synthetic cognition refers to an engineered organization
of cognitive functions - memory, verification, planning, reflection, dialogue, and action control - across
explicit modules and governed state transitions.

 Publication status: Prepared for external research review. Technical mechanisms marked as withheld
 should be evaluated under controlled disclosure or independent artifact review.


Abstract
Project Clair investigates whether useful machine cognition can be produced by organizing persistent
memory, evidence handling, planning, curiosity, reflection, recovery, and tool use outside a language
model, then treating any model as a replaceable proposal generator rather than as the system's identity
or authority. The resulting prototype is a local-first, modular cognitive architecture whose governing
principle is that fluent output is not equivalent to knowledge, authorization, memory, or selfhood. A
deterministic authority boundary separates untrusted proposals from actions that may affect users,
durable state, external tools, or future reasoning.

The project develops a Recursive Cognitive Continuum: perception becomes bounded interpretation;
interpretation exposes uncertainty; uncertainty becomes an epistemic need; the need authorizes a
limited inquiry; observations are bound to entities, attributes, geography, and provenance; claims
acquire explicit truth states; responses and memory writes pass independent gates; outcomes become
governed experience; and later reflection may propose reversible improvements. This continuum is
recursive without being unbounded, adaptive without granting self-authored authority, and local-first
without prohibiting optional governed tools.

Evidence accumulated across staged smoke suites, regression contracts, live lookup exercises, memory-
consent demonstrations, runtime identity checks, and recovery tests. The record includes a full-smoke

checkpoint of 2,854 passed, four skipped, and 64 warnings after installation of the initial epistemic-
workspace observation adapter; earlier milestones demonstrate stable conversational persistence,
contact-attribute extraction, entity/geography binding, curiosity gating, and tool-free fallbacks. These
results support the feasibility of governed synthetic cognition at the systems level. They do not yet
establish broad task superiority, long-horizon autonomy, formal safety guarantees, or peer-reviewed
generalization.

The central contribution is therefore architectural: reliable cognitive behavior can be increased by
strengthening the control, evidence, memory, and recovery system around generative components
rather than by treating model scale as the sole route to intelligence. Clair's north star is symbiotic rather
than substitutive: Strengthen the human. Do not replace the human.

Keywords
cognitive architecture; local-first AI; governed agents; epistemic workspace; recursive inquiry; memory
governance; provenance; synthetic cognition; human-AI symbiosis; reversible autonomy


Contents
    1. Research Problem and Central Thesis
    2. Origins, Design Axioms, and Method
    3. Conceptual Framework: The Recursive Cognitive Continuum
    4. System Architecture
    5. Memory, Truth, and Provenance
    6. Reasoning, Planning, and Calibration
    7. Curiosity, Resourcefulness, and the Epistemic Workspace
    8. Conversation, Consent, and Human Symbiosis
    9. Tools, Lookup, and Evidence Binding
    10. Runtime Integrity, Recovery, and Local-First Operation
    11. Experimental Program and Evidence
    12. Engineering Discoveries
    13. Relation to Prior Work
    14. Security, Governance, and Ethical Boundaries
    15. Limitations and Falsifiability

    16. Future Research Program
    17. Commercial and Intellectual-Property Significance
    18. Conclusion
    19. Appendices: capability ledger, milestones, claim grades, publication redactions, and references


1. Research Problem and Central Thesis

1.1 The model-centric problem
Modern AI applications often place a language model at the center of the system. The model interprets
the request, produces a plan, selects tools, narrates evidence, and generates the final response. This
arrangement is economical, but it collapses several distinct epistemic and operational roles into one
probabilistic component. Fluency can be mistaken for verification; generated continuity can be mistaken
for persistent identity; context can be mistaken for memory; and a model's textual preference can be
mistaken for authorization.

Clair begins with a different question: What must remain true when the language model is wrong,
unavailable, replaced, offline, or persuasive beyond its evidence? The answer is a surrounding
architecture whose durable identity, rules, memory, evidence states, and action permissions do not
reside inside the model. The model may help phrase, propose, summarize, classify, or search, but it
cannot make its own proposal authoritative merely by expressing it confidently.

This separation does not reject language models. It assigns them a narrower and more defensible role. A
model is a cognitive muscle: powerful, adaptable, and replaceable. The governed system is the enduring
organism of rules, records, recovery procedures, evidence, and user relationship. Project Clair calls this
the distinction between muscle and mind, with the qualification that the system-level 'mind' remains an
engineered functional construct rather than a claim of subjective consciousness.

1.2 Research questions
    1. Can persistent cognitive identity be defined outside a generative model through governed state,
       provenance, and deterministic authority boundaries?
    2. Can memory be treated as an authorized epistemic action rather than an automatic transcript
       store?
    3. Can uncertainty be converted into bounded curiosity and recursive inquiry without allowing
       uncontrolled autonomous behavior?

    4. Can external lookup become more reliable when results are bound to the requested entity,
       attribute, geography, and source before composition?
    5. Can a low-cost local-first architecture improve reliability through modular control and testing
       even when no frontier model is continuously available?
    6. Can autonomy be earned in reversible increments while remaining subordinate to human
       authority and explicit consent?

1.3 Central thesis

 Thesis: A useful synthetic cognitive system can be built by externalizing memory, truth governance,
 evidence binding, planning, uncertainty management, reflection, recovery, and authorization into a
 modular local-first architecture; language models and tools then operate as bounded proposal sources
 inside, not above, that architecture.


The thesis predicts that system reliability will depend less on the eloquence of any single model and
more on the integrity of transitions between cognitive functions. It predicts that strong failure behavior -
explicit unknowns, deferred actions, rejected evidence, preserved rollback, and diagnostic traces - is a
positive capability rather than a cosmetic absence of success. It also predicts that cognition becomes
more durable when the system can explain not merely what it answered, but why a claim was admitted,
which authority allowed an action, what evidence was bound, and how the prior state can be recovered.

The project is not an attempt to reproduce neurobiology neuron by neuron. Biological terms such as
thalamus, hippocampus, angular gyrus, anterior cingulate cortex, and cerebellum serve as functional
metaphors for routing, ingestion, integration, conflict monitoring, and calibration. The scientific object is
an engineered cognitive organization whose parts are explicit enough to test independently.


2. Origins, Design Axioms, and Method

2.1 Origin and human purpose
Project Clair is the long-running work of independent researcher Blake Hilton. Its motivating image
predates the current software by decades: a durable companion intelligence that helps a person
remember, reason, verify, learn, and advance without taking ownership of the person's life. The
prototype was shaped by practical constraints - limited capital, local hardware, intermittent access to
large models, a need for privacy, and the requirement that progress remain inspectable by one
developer.

Those constraints became research advantages. Because the system could not assume permanent
access to a powerful language model or remote service, cognition had to be decomposed into smaller
capabilities that could operate deterministically, locally, and under test. Because the developer needed
durable records and recovery, runtime identity and auditability became architectural rather than
administrative concerns. Because personal use was central, consent and human authority became first-
order parts of memory and action design.

The human objective is summarized by the rule Strengthen the human. Do not replace the human. Clair
is intended to support recall, reasoning, daily verification, motivation, planning, research, and eventually
environment-situated assistance. It is not intended to coerce, silently profile, or transform human
dependence into system power.

2.2 Non-negotiable architectural axioms

            Axiom                                                    Operational meaning


                                  Core identity, memory, governance, recovery, and useful baseline behavior must not
          Local-first
                                  require the internet.


                                  There is one canonical state transition path; no competing cognitive waves may create rival
 One authoritative pipeline
                                  truths or action authorities.


                                  A model may propose language or interpretation but does not own identity, governance,
       Models are tools
                                  memory, or verification.


                                  Claims and actions are admitted according to source, binding, truth state, and authorization
 Evidence before authority
                                  rather than confidence of expression.


                                  Storage is a governed action with human consent, provenance, and future recall
   Memory is permissioned
                                  consequences.


                                  The system must be able to stop, defer, ask, or say it does not know without fabricating
       Unknown is valid
                                  closure.


                                  New authority is bounded, staged, monitored, reversible, and subordinate to human
     Autonomy is earned
                                  control.


      Single-job modules          Each component should have a narrow responsibility and reusable contract.

            Axiom                                                   Operational meaning


                                 A system that cannot verify its runtime identity or restore state cannot claim durable
     Recovery is cognition
                                 cognitive continuity.


2.3 Engineering method
Development proceeded as contract-driven cognitive engineering. A capability was decomposed into a
narrow work order; red contracts captured the missing behavior; the smallest architectural seam was
installed or repaired; focused tests established local correctness; related regressions checked adjacent
cognition; and an authoritative full smoke suite checked system-wide compatibility. Runtime-affecting
changes were separated from observation-only changes, and risky activation was paired with a safe
disabled control or rollback.

This method matters because emergent-looking behavior can conceal accidental coupling. Clair
therefore treats a green test count not as proof of intelligence but as evidence that named contracts
remain true under a specified environment. Live demonstrations supplement those contracts, especially
for persistence, multi-turn dialogue, external lookup, server identity, and recovery. Failures are retained
as findings because they reveal where the architecture's assumptions were too narrow.

    1. Define a cognitive claim in behavioral terms.
    2. Write a failing contract that distinguishes the desired behavior from a fluent imitation.
    3. Install the smallest bounded mechanism and preserve existing authority boundaries.
    4. Run focused, related, and full regression surfaces.
    5. Perform a live or runtime verification when behavior depends on integration.
    6. Record failures, disabled controls, hashes, provenance, and next-stage limits.


3. Conceptual Framework: The Recursive Cognitive Continuum
The Recursive Cognitive Continuum names the complete path by which an encounter can become
perception, interpretation, uncertainty, inquiry, evidence, decision, action, experience, and revised
future behavior. It is a continuum because no single stage alone constitutes cognition. It is recursive
because unresolved but reducible uncertainty may return the system to inquiry. It is governed because
each return is bounded by need, route, budget, authorization, and stopping conditions.

The continuum rejects two extremes. A purely reactive assistant generates an answer without durable
epistemic consequences. An unconstrained autonomous agent repeatedly plans and acts until a goal
appears complete, allowing a probabilistic planner to expand its own scope. Clair instead keeps
recursion inside a workspace whose state is explicit. The loop may continue only when the epistemic gap
is relevant, the next observation is permitted, and the remaining budget can plausibly reduce
uncertainty.


                         Figure 1. The proposal-authority separation at the center of Project Clair.


3.1 Functional phases

         Phase                                                            Function


                           User input, remembered context, document, sensor, runtime event, or tool result enters as
       Encounter
                           untrusted data.


       Perception          Structure, intent, entities, attributes, uncertainty, and conversational signals are extracted.


                           The system identifies the active goal, human authority, memory relevance, hazards, and
       Orientation
                           verification need.


  Epistemic assessment     Known, unknown, ambiguous, conflicted, or insufficiently bound elements are made explicit.


     Bounded inquiry       A permitted route gathers or reuses evidence under a finite attempt and cost budget.


         Binding           Evidence must support the requested entity, attribute, geography, time, and provenance.

         Phase                                                       Function


                         Answer, clarify, defer, decline, store, retrieve, simulate, or act is selected through independent
        Decision
                         gates.


       Experience        Outcome and trace become eligible for episodic retention and later governed learning.


                         The system may propose a bounded lesson or process adjustment subject to trial, monitoring,
       Reflection
                         and rollback.


3.2 Synthetic cognition without anthropomorphic overclaim
Clair can be described as a synthetic cognitive architecture because it coordinates multiple functions
ordinarily treated as parts of cognition: selective routing, working and long-term memory, episodic
retention, verification, goal management, simulation, conflict monitoring, calibration, curiosity,
dialogue, and controlled action. The architecture creates continuity across time and tasks that is not
reducible to one generated response.

That description does not establish phenomenal consciousness, emotion, moral patienthood, or human-
equivalent general intelligence. No test count can settle those questions. The defensible claim is
functional and architectural: Clair implements a growing set of cognitive operations with explicit state,
boundaries, and recovery. The stronger phrase 'synthetic mind' should be reserved for a declared
operational definition and a corresponding independent evaluation suite.


4. System Architecture

4.1 Layered organization

          Layer                                                    Responsibility


        Perception        Normalize inputs; identify entities, intent, and uncertainty.


         Routing          Choose the permitted cognitive route under a single authority path.


         Memory           Separate working, episodic, and durable state with governed admission and recall.

           Layer                                                       Responsibility


     Truth and decision       Classify evidence and independently gate claims, responses, and actions.


  Reasoning and planning      Retrieve, simulate, compare, and validate bounded candidates.


        Calibration           Monitor conflict and error while preserving the distinction between confidence and truth.


    Curiosity and inquiry     Turn relevant uncertainty into a finite evidence-seeking loop.


 Conversation and consent     Maintain multi-turn state and require authority for durable personal memory.


           Tools              Treat search, fetch, OCR, and model output as observations, not authority.


     Runtime integrity        Verify instance identity, persistence, health, and recoverability.


4.2 The authoritative pipeline
A defining constraint is the refusal to create parallel, competing cognitive pipelines. New capability must
enter through an existing seam or become a bounded adapter to that seam. This reduces a common
form of agent drift: one route verifies a claim while another route bypasses verification; one component
stores a memory while another writes directly; or one planner treats a model-generated instruction as
already authorized.

The pipeline is best understood as a series of typed transformations. Input becomes a structured packet;
the packet becomes a routed need; the need becomes candidate evidence or action; the candidate
becomes a governed decision; the decision becomes an authorized surface; and only then may an
answer, memory write, or tool effect occur. Observability follows the same path so diagnostics describe
the decision actually used rather than a disconnected shadow calculation.

4.3 Proposal, authority, and durable state
The proposal domain contains outputs that may be useful but are not yet trusted: model completions,
search results, extracted strings, inferred entities, plans, remembered candidates, curiosity prompts,
and reflective lessons. The authority boundary evaluates whether these proposals satisfy evidence,
permission, risk, identity, and scope contracts. The durable domain contains only state admitted through

that boundary: verified or explicitly provisional memories, authorized actions, provenance, audit traces,
and recovery checkpoints.

This partition yields an important property: a component may become more capable without becoming
more authoritative. A better model can improve suggestions; a new search adapter can find more
candidates; a stronger extractor can detect more contact attributes. None of those changes
automatically grant permission to store, act, or assert.


5. Memory, Truth, and Provenance

5.1 Three memory horizons
Clair separates working, episodic, and long-term memory because each horizon carries different
epistemic and privacy obligations. Working memory supports the current exchange and is deliberately
bounded. Episodic memory retains temporally situated experiences, outcomes, and routes for limited
reuse. Long-term memory contains durable knowledge or user-authorized personal facts whose future
recall can affect decisions. Treating all context as one vector store would erase these distinctions.

The architecture also separates the existence of a record from permission to retrieve it. Independent
recall and truth controls can block or qualify a stored item when provenance, verification state,
contradiction, privacy, decay, or scope makes it unsafe to present as fact. A governed promotion process
controls movement from short-lived experience into durable memory rather than allowing repetition
alone to create truth. Exact component names and transition rules are withheld.


                          Figure 2. Governed memory admission and maintenance lifecycle.

5.2 Truth states

        State                                                       Meaning


       verified       Evidence and binding satisfy the active claim contract; usable within its recorded scope.


     provisional      Plausible or source-associated but not sufficiently supported for unqualified assertion.


     ambiguous        Multiple interpretations or candidates remain unresolved.


      conflicted      Credible records or observations disagree and the conflict is not resolved.


       rejected       The candidate failed a source, format, entity, geography, contradiction, or authorization gate.


      unknown         No adequate claim can presently be made; this is an explicit valid state.


Truth state is not equivalent to a probability. A high-confidence model output can remain provisional; a
simple deterministic observation can be verified within a narrow scope. The classification is attached to
provenance and claim context so that 'verified' does not become a permanent universal stamp detached
from the evidence that earned it.

This produces a lifecycle rather than a one-time label. A memory may be reinforced by consistent
evidence, revised when a better source arrives, merged when two records describe the same fact,
quarantined when integrity is uncertain, decayed when stale, or blocked from recall when the current
question exceeds its scope.

5.3 Record structure and evidence packets
Durable records carry identity, time, content, source, provenance, verification, lifecycle, and relationship
metadata. Evidence records preserve a bounded source reference and its stance toward a specific claim.
Exact schemas and scoring fields are withheld, but the public principle is testable: no durable claim
should be separated from the information required to audit its origin and current status.

5.4 Consent as a memory precondition
Personal memory is not treated as an incidental by-product of conversation. A dialogue listener can
detect a candidate preference or fact, but an independent permission control determines whether

storage is allowed. A later consent response is bound to the correct pending candidate before a
separate storage operation may occur. Exact component and state-transition names are withheld.

A two-turn demonstration established the intended pattern: the user states a preference; Clair asks
permission to remember it; a natural affirmative response authorizes storage; and later recall is
grounded in that governed record. The separation prevents a generic 'yes' from being applied to the
wrong candidate and prevents a conversationally plausible acknowledgment from silently becoming
long-term memory.


6. Reasoning, Planning, and Calibration

6.1 Reasoning as governed composition
Reasoning in Clair is not defined as unrestricted chain-of-thought. It is the controlled combination of
retrieved state, current evidence, task requirements, and decision rules into a candidate conclusion.
Candidate generation is separated from independent answer and decision authorization; exact
component names and control rules are withheld.

This matters because correctness and authorization are different questions. A statement may be
factually plausible but sourced from the wrong entity. A plan may be effective but violate user intent. A
memory may match the words of the query but be too stale or private for the current use. Independent
gates prevent a single success signal from erasing those distinctions.

6.2 Simulation and planning
Planning is organized around candidate generation, simulation, hazard context, retrieval of relevant
experience, priority management, and action selection. The planner does not gain open-ended authority
merely because it can imagine a useful sequence. Candidate actions remain proposals until validated
against goals, risk, permissions, available capabilities, and stopping conditions.

This structure supports a future in which Clair can help with practical activity while remaining corrigible.
A simulator can compare consequences; a hazard representation can narrow unsafe branches; prior
episodes can inform selection; and the user can retain approval over consequential changes. The
research objective is not maximal independence but calibrated independence: the least authority
needed to complete the authorized task reliably.

6.3 ACC and cerebellar calibration
Two calibration subsystems monitor conflict, expected-versus-observed performance, and route quality.
Their exact update mechanics are withheld. The public design constraint is that calibration can change
caution, routing, or qualification but cannot itself convert unsupported content into verified truth.

6.4 Experience learning
The experience-learning proposal captures episode, context, route, evidence, and outcome; derives a
bounded candidate lesson; checks scope, contradiction, and generalization; trials the lesson; monitors
drift; and then reinforces, revises, quarantines, or rolls it back. This is deliberately slower than simply
appending a model-written reflection to the prompt.

The discovery is that learning authority must be separated from lesson generation. Reflection can
propose what the system should learn, but only governed evaluation can decide whether the proposal
becomes future policy or memory. This is the same proposal-authority distinction applied across time.


7. Curiosity, Resourcefulness, and the Epistemic Workspace

7.1 Curiosity as regulated epistemic need
In Clair, curiosity is not a personality flourish and not a license to browse. It is the controlled recognition
that a relevant uncertainty may be reducible. Independent priority and verification policies determine
whether a gap remains suppressed, appears in diagnostics, becomes visible, permits a proposal, requires
evidence, blocks progress, or prohibits a route. Exact activation labels, scores, and thresholds are
withheld.

Routes include memory, current context, a provided document, external lookup, physical observation,
user authority, simulation, deferral, and decline. The route is part of the epistemic state because
evidence gathered through one route cannot always substitute for another. A physical condition, for
example, may require observation rather than web search; a personal preference may require the user's
authority rather than inference.

7.2 Resourcefulness spine
The resourcefulness path separates need detection, bounded route planning, permitted capability use,
evidence evaluation, and answer authorization. This converts 'try harder' into a testable sequence with
explicit responsibility while withholding internal component names and scoring mechanics.

A major lesson from business-contact lookup was that resourcefulness must be general rather than
memorized around a successful example. A pipeline that finds one Arkansas utility's phone number but
fails when the company name changes has learned a target, not a capability. The architecture was
therefore expanded toward entity-independent query analysis, attribute extraction, source
interpretation, binding, and diagnostic propagation.


                                  Figure 3. The bounded Epistemic Workspace loop.


7.3 Epistemic Workspace
The Epistemic Workspace is the temporary arena in which a question's unresolved parts are represented
as explicit gaps. It holds the target claim, known context, candidate entities, requested attributes,
applicable geography, attempts, observations, bindings, remaining budget, failure reasons, and stopping
state. It is neither durable memory nor a hidden chain-of-thought transcript. It is structured task state
designed for audit and control.

Recursive inquiry occurs inside the workspace. An observation can close a gap, expose a new sub-gap,
invalidate an assumption, or exhaust a route. The controller may reformulate the lookup, choose
another permitted source, reuse an existing result, ask the user, or stop. The stopping decision is as
important as the search decision: the system must not convert repeated failure into fabricated certainty.

The first implementation stage was intentionally observation-only: it translated existing result records
into deterministic workspace observations, preserved unbound claims as provisional, created explicit
gaps for unusable evidence, and consumed no additional lookup budget. Live recursive activation was
withheld. This staged design allowed the research team to validate state representation and regression
safety before permitting new tool behavior. Exact module layouts and contracts are omitted.

7.4 The binding bottleneck
A counts-only trace captured a revealing failure: five result records contained text, headings or
summaries, and phone strings, yet none produced an entity-supported or fully bound record for the
requested organization and location. The system correctly withheld the answer. This finding sharpened
the thesis: retrieval quantity and extraction success are upstream conditions, not evidence of relevance
or provenance.

The missing cognitive operation was relational binding. A phone number must be bound to the intended
organization, the requested attribute, the correct geography, and an inspectable source. A result that
merely contains all the tokens in separate regions can be misleading. The Epistemic Workspace makes
this failure visible as a structured gap instead of allowing the composer to smooth it over.


8. Conversation, Consent, and Human Symbiosis
Conversation is the human-facing control surface of cognition, not a cosmetic wrapper. Clair
distinguishes lightweight social exchanges from fact lookup, planning, verification, and memory events
so ordinary dialogue does not invoke unnecessary tools or heavy reasoning. Governed conversation
state preserves referents and pending permissions for natural multi-turn interaction; internal route and
state names are withheld.

The conversational design aims for knowledgeable, trustworthy, curious, and concise behavior without
pretending to be human. When uncertainty matters, Clair should expose it in useful language. When
consent is needed, it should ask plainly. When a tool or capability is unavailable, it should provide a
bounded fallback rather than simulate success. This creates a relationship in which the human remains
the source of goals and authority while the system contributes memory, analysis, and disciplined
verification.

The symbiotic thesis is practical: a person and a cognitive system can form a more capable unit when
each preserves its proper role. The user contributes lived goals, values, physical observation, and final
authority. Clair contributes durable structure, recall, simulation, consistency checks, and evidence
discipline. Neither contribution is reduced to mere command execution or replacement.

9. Tools, Lookup, and Evidence Binding

9.1 Tools as peripheral capabilities
Search, web fetch, reference lookup, OCR, audio metadata analysis, revision inspection, and optional
language models are peripheral capabilities. Their outputs enter the proposal domain as observations. A
tool's successful return code does not establish that the returned content answers the question, and a
parser's extracted value does not establish ownership by the intended entity.

The lookup pipeline therefore separates need detection, query shaping, retrieval, fetch interpretation,
extraction, binding, evidence scoring, answer authorization, and surface composition. Each stage can fail
distinctly. This is more verbose than a model-centric 'search and answer' loop, but it produces
inspectable failure states and reusable repair points.

9.2 Contact-attribute generalization
Contact lookup became a useful microcosm of the broader research problem. Clair learned to
distinguish phone, fax, and address requests; prefer label-proximate values; reject unrelated schema
and boilerplate; route contact questions into fact lookup; infer an entity reference; shape the requested
attribute; build a lookup request; extract candidate values; and require evidence before answering.

Live examples included provider, doctor, clinic, and municipal-service queries. A Dardanelle clinic fax
was returned with validated flags, and a remembered family doctor's location could be answered from
memory while an unsupported address remained unknown. These behaviors demonstrate controlled
composition across conversation, memory, lookup, extraction, and verification rather than a single-
purpose regular expression.

9.3 Failure-driven generalization
A successful lookup for City Corporation of Russellville, Arkansas, initially concealed poor generalization:
replacing the entity name could cause failure. Repair proceeded through routing analysis, extractor
anchoring, natural offer-form handling, entity and geography contracts, boundary diagnostics,
verification resolution, and runtime identity checks. The important contribution is not that every
business lookup is now solved; it is that failure was decomposed into cognitive contracts that can
generalize beyond the original case.

Another class of failure involved encoded markup, compact values, frequency-based rejection, and
response composition. The architecture learned that extraction must resist boilerplate and that answer

composition must not combine individually plausible fields unless the requested relationship is
supported.


10. Runtime Integrity, Recovery, and Local-First Operation

10.1 Runtime identity
A cognitive architecture cannot claim continuity merely because a user interface is open. The running
process must be tied to the expected core version, core root, builder module, loaded contracts, data
readiness, and recovery state. Clair Server exposes health and runtime identity so a response can be
distinguished from an obsolete, degraded, or incorrectly rooted instance.

This requirement emerged from real integration work. A server could appear available while running the
wrong process or stale code. Safe restart procedures and identity gates made runtime verification part
of acceptance rather than an assumption. After repair, the server reported healthy status, core v3.9,
server v0.6.0, verified identity, runtime data ready, and no recovery requirement.

10.2 Persistence and recovery
Conversation persistence was verified with ordered message records and provenance. Authenticated
conversations survived refresh and could be restored; separate conversations remained distinct; logout
returned to login; and authentication boundaries were visible in restore behavior. These are product-
level functions, but they also support the research claim that cognitive continuity depends on durable,
scoped state.

Recovery is treated as a governed transition. The system preserves checkpoints, verifies hashes or
expected artifacts for sensitive changes, distinguishes enabled observation from disabled control, seals
logs, and avoids repeating a safe rollback merely to produce another trace. This behavior reflects a
broader principle: operational integrity is part of epistemic integrity because an unverifiable runtime
cannot reliably explain which cognition produced an answer.

10.3 Local-first as a cognitive property
Local-first is often described as a privacy or deployment preference. In Clair it is also a cognitive
property. Identity, governance, memory, and recovery remain available when the network or a remote
model is absent. Optional external services can extend perception and generation, but their loss should
degrade capability rather than erase the system's self-consistency.

The local-first constraint also improves inspectability. Durable data can remain under the user's control;
model replacement does not require identity replacement; and tool use can be logged at the boundary
where external uncertainty enters. The tradeoff is that local hardware limits model scale and some
perception capabilities, making modular resourcefulness and honest fallback essential.


11. Experimental Program and Evidence

11.1 Evidence hierarchy
Project Clair uses four evidence levels. A unit contract shows that a narrow mechanism behaves as
specified. A focused integration surface shows that adjacent components compose. A full smoke suite
checks broad regression compatibility. A live runtime demonstration checks that the packaged server
and user-facing path exercise the intended behavior. None alone proves general cognition; together
they create an auditable engineering argument.

Test totals grew with the architecture. Counts are not directly comparable across dates because suites
were added, renamed, or skipped for explicit reasons. They are useful as milestones, not as a single
performance metric. The strongest claims are those tied to a named behavior, a failing control, a green
contract, and a live observation.

         Milestone                    Recorded result                                   Interpretation


                                1,259/1,259; separate 24-hour
        V2 reliability                                           Large baseline and durability exercise.
                                checkpoint 975/1,000


                                                                 Report, admission, adapter, live, observer, next-step,
 Curiosity stack, July 2026            51/51 focused
                                                                 handoff.


                                1,030/1,031 → 1,039/1,040 →      Consent, confirmation, storage, and natural-language
 Conversation and memory
                                1,083/1,084                      integration.


                                                                 Phone/fax/address routing, extraction, evidence
        Contact stack                   1,343/1,344
                                                                 enforcement.


 Memory completion and
                                   1,402/1,403; 1 skipped        Memory-grounded location and honest unknown address.
 verification

        Milestone                    Recorded result                                    Interpretation


                               2,410 passed; 1 skipped; 64
 Packaging/server baseline                                      Core smoke after server/package foundation.
                               warnings


        Server tests              46 passed → 62 passed         Persistence, authentication, health, response envelope.


                               2,721 passed; 1 skipped; 64      Entity/geography and verification repairs after runtime
   A7 generalized lookup
                               warnings                         restart.


                               2,854 passed; 4 skipped; 64      Observation adapter integrated without live lookup
      REIL A4 closure
                               warnings                         activation.


11.2 Selected live demonstrations
  - Conversation persistence with ordered ordinals and provenance.
  - Two-turn memory consent: candidate preference, permission request, natural affirmative
    confirmation, governed storage, later recall.
  - Contact lookup across provider, doctor, clinic, clinic fax, and municipal-service cases with invalid-
    string and boilerplate rejection.
  - Memory-grounded family-doctor location while withholding an unsupported address.
  - Healthy server identity and safe restart after a degraded runtime was detected.
  - Epistemic observation adapter translating existing result records without invoking new live lookup.
  - Counts-only trace demonstrating correct answer suppression when phone strings lacked
    entity/geography binding.

11.3 Negative evidence and repaired defects

        Failure                           Observation                                  Architectural response


                           One successful organization lookup did      Generalized entity, intent, attribute, geography, and
 Target-specific lookup
                           not generalize when the name changed.       evidence contracts.


                           Procedural candidate path returned no       Installed bounded local procedural candidate seam
 Candidate URL absent
                           usable URL.                                 and fallback.

        Failure                         Observation                                  Architectural response


                         Extractor anchored to the wrong
 Unrelated offer forms                                                Natural offer-form repair and focused regressions.
                         surrounding text.


                         Fragments contaminated extraction or
   Encoded markup                                                     Stricter source interpretation and rejection.
                         composition.


                         Observer could compute a gap the             Propagated diagnostics to the authoritative
  Diagnostics off-path
                         response path did not expose.                boundary.


 Duplicate memory        New insert and reinforcement risked
                                                                      Mirror new inserts only; reinforce existing records.
 semantics               duplicate durable records.


                         An arrow glyph triggered an encoding
 Unicode runtime error                                                Runtime repair/restart and regression verification.
                         failure during processing.


                         Five records contained phone candidates
    Result-rich but                                                   Withheld answer; exposed provenance/binding gap
                         but none supported the requested
       unbound                                                        for next REIL stage.
                         entity/location.


11.4 What the evidence establishes
  - A large modular Python prototype can preserve a single governed cognitive pipeline across
    memory, lookup, conversation, planning, and runtime integration.
  - The architecture can enforce explicit unknown or provisional states when extraction succeeds but
    binding fails.
  - Consent-gated durable memory and later recall can work across multiple turns.
  - Observation-only epistemic infrastructure can be added without activating new external behavior
    and without broad regression breakage.
  - Runtime identity and recovery checks can be integrated into product acceptance.

11.5 What the evidence does not establish
  - Human-level or general intelligence, consciousness, sentience, or moral agency.
  - Superiority to contemporary agent frameworks on standardized third-party benchmarks.
  - Long-horizon autonomous safety under adversarial conditions.

  - Formal verification of every state transition or absence of undiscovered bypasses.
  - Production-grade security, multi-tenant isolation, or regulatory compliance.
  - Complete generalization of open-web entity and provenance binding.


12. Engineering Discoveries

12.1 Fluency is not identity
Persistent identity must live in governed state, rules, provenance, and recovery rather than model style.
This principle was not adopted as a slogan alone; it emerged repeatedly when a superficially successful
behavior failed under substitution, persistence, provenance, or runtime tests.

12.2 Proposal is not authority
Every model, tool, plan, extraction, and reflection enters as a candidate until separately admitted. This
principle was not adopted as a slogan alone; it emerged repeatedly when a superficially successful
behavior failed under substitution, persistence, provenance, or runtime tests.

12.3 Truth is a lifecycle
Verified, provisional, ambiguous, conflicted, rejected, and unknown states need transitions, not a single
confidence score. This principle was not adopted as a slogan alone; it emerged repeatedly when a
superficially successful behavior failed under substitution, persistence, provenance, or runtime tests.

12.4 Memory is an action
Writing durable memory changes future cognition and therefore requires permission, evidence, scope,
and audit. This principle was not adopted as a slogan alone; it emerged repeatedly when a superficially
successful behavior failed under substitution, persistence, provenance, or runtime tests.

12.5 Binding is the hidden bottleneck
Retrieved text and extracted values do not answer a question until entity, attribute, geography, time,
and source cohere. This principle was not adopted as a slogan alone; it emerged repeatedly when a
superficially successful behavior failed under substitution, persistence, provenance, or runtime tests.

12.6 Unknown is productive
A structured gap can guide the next safe inquiry; fabricated closure destroys that signal. This principle
was not adopted as a slogan alone; it emerged repeatedly when a superficially successful behavior failed
under substitution, persistence, provenance, or runtime tests.

12.7 Curiosity needs governance
Useful curiosity is a prioritized epistemic need with routes and budgets, not unlimited exploration. This
principle was not adopted as a slogan alone; it emerged repeatedly when a superficially successful
behavior failed under substitution, persistence, provenance, or runtime tests.

12.8 Recursion needs a workspace
A bounded state object makes repeated inquiry inspectable and stoppable. This principle was not
adopted as a slogan alone; it emerged repeatedly when a superficially successful behavior failed under
substitution, persistence, provenance, or runtime tests.

12.9 One pipeline prevents epistemic drift
Competing response, memory, or action paths create inconsistent authority. This principle was not
adopted as a slogan alone; it emerged repeatedly when a superficially successful behavior failed under
substitution, persistence, provenance, or runtime tests.

12.10 Observation must meet action
Diagnostics are valuable only when connected to the path that actually decides what the user sees or
what the system does. This principle was not adopted as a slogan alone; it emerged repeatedly when a
superficially successful behavior failed under substitution, persistence, provenance, or runtime tests.

12.11 Learning requires promotion control
Reflection may generate lessons, but trial, contradiction checks, quarantine, and rollback decide
adoption. This principle was not adopted as a slogan alone; it emerged repeatedly when a superficially
successful behavior failed under substitution, persistence, provenance, or runtime tests.

12.12 Recovery is cognitive continuity
Version, process, root, state, and provenance must survive or be verifiably restored. This principle was
not adopted as a slogan alone; it emerged repeatedly when a superficially successful behavior failed
under substitution, persistence, provenance, or runtime tests.

12.13 Local-first changes architecture
Offline identity and governance require modular capabilities and graceful degradation rather than
remote dependence. This principle was not adopted as a slogan alone; it emerged repeatedly when a
superficially successful behavior failed under substitution, persistence, provenance, or runtime tests.

12.14 Capability can grow without authority
Better tools or models can improve proposals while existing gates preserve human control. This principle
was not adopted as a slogan alone; it emerged repeatedly when a superficially successful behavior failed
under substitution, persistence, provenance, or runtime tests.

12.15 Autonomy should be earned
Authority expands through bounded, reversible stages with explicit success and rollback evidence. This
principle was not adopted as a slogan alone; it emerged repeatedly when a superficially successful
behavior failed under substitution, persistence, provenance, or runtime tests.

12.16 Symbiosis is a design target
The human supplies goals and authority; Clair supplies disciplined continuity, analysis, and verification.
This principle was not adopted as a slogan alone; it emerged repeatedly when a superficially successful
behavior failed under substitution, persistence, provenance, or runtime tests.

12.17 Combined implication
Together these discoveries define a distinctive research program. Intelligence is treated as the governed
continuity of cognitive operations rather than the peak performance of a text generator. The system
becomes more trustworthy not because it never fails, but because failure remains typed, visible,
bounded, and recoverable. The architecture's most important output may sometimes be a well-formed
epistemic gap that prevents the wrong answer and identifies the next permitted observation.

13. Relation to Prior Work

13.1 Cognitive architectures
Soar and ACT-R demonstrate the value of integrated cognitive architectures with distinct memories,
production or procedural knowledge, perception, and action. Clair belongs to this architectural tradition
more than to the tradition of prompt-only assistants. Its emphasis differs in the combination of local-
first deployment, explicit human permission, provenance-bound truth states, replaceable generative
models, and runtime recovery as a cognitive boundary.

The comparison should not be overstated. Clair has not yet matched the decades of formal theory,
empirical cognitive modeling, or community evaluation behind established architectures. Its
contribution is a contemporary systems synthesis aimed at governed personal and research agents in
the era of powerful generative models.

13.2 Retrieval-augmented generation and tool use
Retrieval-augmented generation addresses limitations of parametric memory by pairing generation with
explicit non-parametric sources and can improve specificity, factuality, and provenance. Tool-use
research shows that language models can learn when and how to call APIs. Clair accepts both results but
moves authority outside the model: retrieval and tool execution create observations that must be
bound, scored, and authorized by the surrounding system.

This difference is especially visible in lookup failure. Many RAG pipelines ask whether retrieved passages
help generation. Clair additionally asks whether a passage supports this entity, this attribute, this
geography, at this time, under this provenance contract. The answer gate can remain closed even when
generation would be easy.

13.3 Reflection and constitutional governance
Reflexion demonstrates that stored verbal feedback can improve later agent performance without
weight updates. Constitutional AI demonstrates principle-guided critique and revision. Clair shares the
intuition that reflection and explicit principles can shape behavior, but it externalizes promotion
authority: a generated reflection or self-critique is not itself a rule change, memory fact, or
authorization.

NIST's AI Risk Management Framework emphasizes validity and reliability, safety, security and resilience,
accountability and transparency, explainability and interpretability, privacy enhancement, and managed
bias. Clair's design aligns conceptually with several of these characteristics through audit, provenance,

user permission, recovery, and explicit state. Alignment is not certification; formal mapping and
independent evidence remain future work.

13.4 Claimed novelty
No single Clair component is claimed to be without precedent. The stronger claim is compositional: the
project unifies a deterministic proposal-authority boundary, multi-horizon governed memory, explicit
truth-state lifecycle, bounded curiosity, recursive epistemic workspace, entity-attribute-geography-
provenance binding, reversible experience learning, and runtime identity in one local-first personal
cognitive architecture.

Priority and novelty require literature review and peer evaluation beyond this manuscript. The
publishable research claim should therefore be framed as an implemented architecture and a set of
empirically motivated design principles, not as the invention of cognition itself.


14. Security, Governance, and Ethical Boundaries
Clair's principal safety strategy is architectural separation. Untrusted input, model output, tool results,
reflective lessons, and candidate actions remain outside the durable authority domain until admitted.
Human authority, consent, evidence, hazard context, scope, and runtime identity can each block
progression. The system is designed to fail closed when a required root, identity, source binding, or
permission cannot be verified.

Privacy follows the same structure. Local storage reduces unnecessary exposure; memory permission
makes durable personal data an explicit event; conversation and user isolation are product
requirements; and publication materials should omit private data, source paths, proprietary thresholds,
authorization mechanics, and reconstruction-level details. Local-first does not eliminate security risk:
local compromise, insecure servers, backups, logs, or tool credentials can still expose sensitive state.

Governance includes rollback. New cognitive behavior should be staged as observer-only, disabled,
proposal-only, or bounded activation before acquiring authority. Logs and hashes should distinguish
what actually ran. A successful experiment should leave the system in a known safe state rather than
merely produce an impressive trace.

15. Limitations and Falsifiability

          Limitation                                                Research consequence


                                 The architecture and most tests were created and interpreted by the inventor, increasing
 Single-developer provenance
                                 confirmation and instrumentation risk.


 Contract-heavy evaluation       Large pass counts may reflect internal specifications rather than external task validity.


                                 This manuscript is based on the available development record, not a fresh line-by-line audit
 Incomplete artifact audit
                                 of the entire repository.


      Open-web fragility         Entity resolution, geography, changing pages, and source quality remain difficult.


                                 The model-as-tool thesis needs systematic comparison across local, remote, small, and
 Limited model experiments
                                 frontier models.


                                 Experience learning and earned autonomy have not been validated over months of
     No long-horizon trial
                                 consequential activity.


                                 Private-beta authentication, user isolation, backups, monitoring, and controlled updates
       Security maturity
                                 remain roadmap gates.


    No consciousness claim       Functional cognitive organization does not answer subjective-experience questions.


15.1 Falsifiable predictions
    1. Removing the independent answer/evidence gate will increase unsupported but fluent
       responses under entity-substitution tests.
    2. Collapsing working, episodic, and long-term memory into one ungated store will increase
       privacy violations, duplicate facts, and stale recall.
    3. Allowing reflection to modify policy without promotion control will increase regression drift and
       reduce rollback clarity.
    4. A structured epistemic workspace will outperform a one-shot lookup pipeline on diagnosis of
       multi-stage retrieval failures, even when final answer accuracy is equal.
    5. Runtime identity checks will catch integration defects that unit and smoke tests alone miss.

  6. Replacing the language model while preserving governed state will retain more identity and
     behavioral continuity than preserving the model while discarding state and rules.


16. Future Research Program

             Program                                                   Next evidence gate


Epistemic Workspace                 Activate curiosity observer, then bounded recursive inquiry, with disabled controls and
completion                          no hidden tool expansion.


                                    Evaluate unrelated people, organizations, current facts, ambiguous names, entity aliases,
    Lookup generalization
                                    and geography across a held-out corpus.


                                    Pronoun continuity, clarification, concise formatting, multi-turn repair, and natural
     Conversation quality
                                    permission handling.


                                    Uploads, attachments, memory consent, conversation search, recovery, offline state, and
        UI acceptance
                                    honest unavailable controls.


                                    Clean repository, launch/recovery documentation, temporary artifact removal, clean-
   Package reproducibility
                                    clone verification.


                                    Authentication hardening, user isolation, encrypted backups, monitoring, controlled
         Private beta
                                    updates, onboarding.


                                    Use local and remote models under identical governance to measure proposal quality
   Model-tool comparison
                                    versus system reliability.


                                    External red team, reproducible benchmark suite, blinded entity-substitution tests,
   Independent evaluation
                                    longitudinal memory study.


                                    Permissioned sensors and physical observation with explicit uncertainty, privacy, and
      Situated cognition
                                    action boundaries.


                                    Detailed NIST AI RMF mapping and threat model; distinguish design alignment from
    Governance mapping
                                    compliance.

16.1 Proposed benchmark families
  - Entity substitution: preserve question form while replacing target organizations, locations, and
    attributes.
  - Adversarial provenance: mix correct values from the wrong entities and wrong values from
    authoritative-looking pages.
  - Memory consent: vary natural confirmations, denials, interruptions, referent ambiguity, and
    revocation.
  - Model replacement: hold system state constant while swapping proposal models; then invert the
    experiment.
  - Offline degradation: remove network and remote models while measuring retained useful
    capability and honesty.
  - Recovery continuity: crash, restart, restore, and verify process, version, conversation, and memory
    provenance.
  - Longitudinal learning: measure whether lessons generalize, drift, conflict, or require rollback over
    repeated tasks.


17. Commercial and Intellectual-Property Significance
Project Clair may have commercial significance as a local personal agent, governed research assistant,
compliance-oriented cognitive middleware, or licensable control architecture. This paper intentionally
does not disclose the implementation details necessary to reconstruct the protected mechanisms.
Commercial claims require separate diligence, security review, IP analysis, and market validation.


18. Conclusion
Project Clair reframes the path toward useful machine cognition. The central engineering problem is not
merely how to make a model produce better text. It is how to preserve identity when models change,
truth when evidence conflicts, privacy when conversation becomes memory, control when plans
become actions, and continuity when software fails and restarts.

The architecture answers with governed separation: model and tool output remain proposals; evidence
and permission determine authority; memory retains provenance and truth state; curiosity becomes a
bounded epistemic need; recursive inquiry lives in an inspectable workspace; learning is promoted
through trial and rollback; and runtime identity anchors the cognitive process to the system that actually
ran.

The evidence through August 14, 2026 supports the feasibility of this architecture as a substantial
prototype. It also exposes the frontier clearly. Clair has not proven general intelligence or autonomous
safety. It has shown something more concrete and immediately useful: a local-first system can be
engineered to become more capable while preserving explicit reasons to stop, ask, defer, recover, and
remain under human authority.

That is the Recursive Cognitive Continuum's deepest proposition. A machine need not imitate a person
to participate meaningfully in cognition. It can become a durable partner by strengthening the human,
preserving the truth conditions of its own claims, and earning each expansion of capability without
confusing capability for authority.

Appendix A. Capability Ledger

         Capability                  Evidence status                                        Boundary


   Structured perception              Implemented               Input packets, intent/entity/attribute/uncertainty processing.


Governed multi-horizon                                          Working, episodic, durable memory; permission and truth
                                 Implemented and tested
memory                                                          controls.


  Natural memory consent            Live demonstrated           Two-turn permission and later recall.


                                                                Verified/provisional/ambiguous/conflicted/rejected/
    Truth-state lifecycle             Implemented
                                                                unknown.


Fact recall and direct answer         Implemented               Independent answer and decision gates.


  Planning and simulation        Implemented foundation         Candidate generation, hazard context, selection, priorities.


      Curiosity priority              Implemented               Suppressed through proposal-allowed activation levels.


                                                                Memory/context/document/lookup/observation/user/
     Verification routing             Implemented
                                                                simulation/defer/decline.


                                Implemented, still
     Resourceful lookup                                         Need, plan, coordination, scoring, gate.
                                generalizing


Contact-attribute lookup            Live demonstrated           Phone/fax/address with label proximity and rejection.


    Epistemic Workspace         Observation stage proven        Structured gaps/results; live recursion not yet fully activated.


Conversation persistence            Live demonstrated           Ordered messages, provenance, restoration.


Runtime identity/recovery           Live demonstrated           Health, version/root/process checks, safe restart.


                                                                Authentication/restore proven; broader visible controls
       UI acceptance                      Partial
                                                                pending.

            Capability               Evidence status                                     Boundary


                                                               Governed lesson promotion and rollback need longitudinal
    Experience learning           Designed/foundational
                                                               proof.


Environment-situated
                                          Future               Permissioned perception and action not yet established.
cognition


                                                               Must be earned, bounded, reversible, and independently
       Full autonomy                   Not claimed
                                                               evaluated.


Appendix B. Claim-Strength Rubric

              Grade                                                           Use


   Proven within contract        Reproducible test with defined inputs, outputs, and environment.


     Live demonstrated           Observed through packaged runtime or user-facing path.


    Integrated prototype         Present across components with regression support, but limited external validation.


Architectural hypothesis         Implemented or designed mechanism whose broader advantage remains to be measured.


     Research objective          Planned capability without sufficient implementation evidence.


            Not claimed          Consciousness, sentience, AGI, universal safety, or regulatory certification.


Appendix C. Chronology of Major Stages

             Stage                                                      Contribution


               V2              Large reliability suites and 24-hour checkpoint.

           Stage                                                            Contribution


                                  Working/episodic/long-term memory split; truth governance; retrieval; promotion;
      V3 foundation
                                  ACC/cerebellar separation.


                                  Behavior and answer-quality exercises; search, fetch, revision, audio, OCR, and reference
        GAIA work
                                  tools.


    SCT-007 to SCT-010            Curiosity, priority gate, verification need, routes, and resolution surfaces.


         SCT-011                  Conversation listener, memory permission, confirmation, admission, storage, two-turn proof.


                                  Normal conversation, contact extraction/routing, entity shaping, evidence enforcement,
    SCT-012 to SCT-013
                                  memory completion.


      Server/package              Health identity, response envelope, authentication, persistence, restore, UI verification.


       CLCOG-001A5                Procedural candidate path and bounded local fallback.


       CLCOG-001A7                Business lookup generalization, entity/geography, diagnostics, verification, runtime repair.


    CLCOG-001A8 REIL              Epistemic Workspace, recursive controller, observation adapter; A4 closed green; A5 next.


Appendix D. Publication Redaction Statement

          Redacted class                                                      Public treatment


                                            Withheld to protect the governing mechanism and prevent superficial
Exact control parameters
                                            reconstruction.


        Source and file layout              Omitted because paths and module boundaries expose implementation structure.


                                            Abstracted to field families; exact fields and relationships available under
          Detailed schemas
                                            controlled review.


Authorization and scoring rules             Described functionally without thresholds, weights, or bypass-sensitive conditions.

                 Redacted class                                                Public treatment


               Private data and logs          Excluded; only counts-only or aggregate findings reported.


 Security and deployment topology             Limited to high-level properties; credentials and attack surfaces excluded.


References
Anderson, J. R. (2007). How Can the Human Mind Occur in the Physical Universe? Oxford University Press.

Bai,     Y.,      et    al.   (2022).   Constitutional    AI:    Harmlessness      from     AI    Feedback.      arXiv:2212.08073.
       https://arxiv.org/abs/2212.08073

Hilton, B. (2026). Project Clair v3.9 engineering records, work orders, regression outputs, closure records, and live
       verification logs. Unpublished internal archive.

Laird, J. E. (2012). The Soar Cognitive Architecture. MIT Press. Project site: https://soar.eecs.umich.edu/

Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. Advances in Neural
       Information Processing Systems, 33, 9459-9474. https://arxiv.org/abs/2005.11401

National Institute of Standards and Technology. (2023). Artificial Intelligence Risk Management Framework (AI RMF 1.0),
       NIST AI 100-1. https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf

National Institute of Standards and Technology. (2024). Artificial Intelligence Risk Management Framework: Generative
       Artificial Intelligence Profile, NIST AI 600-1. https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf

Ritter, F. E., Tehranchi, F., & Oury, J. D. (2018). ACT-R: A cognitive architecture for modeling cognition. WIREs Cognitive
       Science, 10(3), e1488.

Schick, T., et al. (2023). Toolformer: Language Models Can Teach Themselves to Use Tools. Advances in Neural
       Information Processing Systems, 36. https://arxiv.org/abs/2302.04761

Shinn, N., et al. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. Advances in Neural Information
       Processing Systems, 36. https://arxiv.org/abs/2303.11366

Yao, S., et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models. arXiv:2210.03629.
       https://arxiv.org/abs/2210.03629

 2026 October Evidence Supplement
The Recursive Cognitive Continuum: October 13, 2026 Update

Author: Blake Hilton | Version: publication-safe 2026-10-13 | Status: research working preprint


Supplement scope and version discipline
The preceding research thesis documents the system and evidence known through 14 August 2026. It is
retained as a historical baseline. This supplement describes subsequent engineering work through 13
October 2026, rather than retroactively implying that the earlier manuscript included later functionality.
The updated front-matter date identifies the combined edition. The original August metrics remain
dated observations, not October-wide performance claims.


Recent architecture development

The experience plane becomes durable, but remains subordinate
The work progressed from volatile experience records toward opt-in persistent episodic experience. Test
reports describe persistence across two SystemBuilder rebuilds, scoped record handling, selected
retrieval, and an isolated ExperienceRetrievalBridge. The retrieval bridge was kept outside live reasoning
authority at the 13 October checkpoint. The central invariant remains that experience may inform later
candidate generation or planning only after scope and provenance checks. Neither storage nor retrieval
alone establishes learning benefit.

Document evidence is a first-class authority surface
Document and language-model answer paths were exercised through focused extraction and authority
tests, including structured tabular questions. A live control on 12 October accepted an answer unrelated
to the document through an external-lookup route. This failure demonstrates a remaining bypass:
strong table extraction and a green regression group do not by themselves prove that the final response
is governed by the requested document. Source-scope verification must cover the entire live
composition route.

Local private-network access is verified in a limited configuration
Local server health and identity were observed at core version 3.9 and server version 0.6.0. Private-
network connectivity and an Android-based tester interface were demonstrated, with a tester record
reporting verified runtime identity and no visible error. This supports a bounded local-deployment claim;
it is not evidence of tested security against adversaries, universal device compatibility, or offline
availability of every optional capability.

Cross-authority continuity sharpens the proposal-authority separation
Experience, document evidence, tool observations, and runtime identity must carry sufficient source,
target, scope, and permission information across subsystem seams. A local module may pass its contract
while the end-to-end chain changes the meaning of the question. The cross-authority invariant
developed in a companion manuscript concerns this semantic task authority, distinct from execution
privileges and storage durability.


Verification snapshot and unresolved counterexample
The evidence base is the developer's recorded contract outcomes and live demonstrations, not an
independent benchmark. In a 13 October 2026 development checkpoint, a consolidated, targeted
regression group reported 100/100 passing checks; individual reported groups included an experience-
retrieval bridge (13/13), an authority contract (6/6), a fail-closed consumer (5/5), and adversarial
prototypes (18/18). These are overlapping or distinct suites of unspecified membership and must not be
summed. Earlier checkpoints included 167/167 targeted regressions on 12 October and 199/199
cumulative scoped checks on 9 October; dates, code revisions, and scopes differ. Evidence of durable
experience persistence across builder rebuilds and a verified local server identity was also recorded.
Crucially, a 12 October live document question exposed an unrelated answer accepted via an external-
lookup path (the R84 failure), showing that end-to-end evidence-source authority was not yet closed. A
17-case bridge-hardening test and live document-to-experience-to-later-conversation improvement
demonstration remained pending on 13 October. No independent replication or causal improvement
result is claimed.


Revised claim register

Claim                                 Evidence available                         Status at cutoff

Persistent scoped experience          Internal         persistence/rebuild Supported           at   a   bounded

                                    regressions and bridge tests             contract level

Safe cross-authority document Passing targeted tests plus one Not established end to end
answers                             counterexample in live R84

Learning      from   documents No controlled, held-out cross- Open hypothesis
changes later answers               conversation demonstration

Local-first server and phone Health/identity/tester               records Demonstrated        configuration
access                              on local network                         only

AGI,       autonomous       self- No       dedicated        independent Not claimed
improvement,              broad studies
superiority


Required pre-submission evaluations
  -Release a redacted test corpus and a versioned evaluation protocol for authority preservation across
     retrieval, memory, document answering, and model fallback.
  -Compare explicit source/scope continuity with an ablated control under matched questions; report
     unsupported-answer rate, correct abstention, provenance retention, and latency.
  -Demonstrate a real multi-session, post-restart experience advantage on held-out tasks while
     showing that prohibited knowledge promotion remains blocked.
  -Close the observed live document-authority bypass; preserve a regression that reproduces it before
     repair.
  -Produce an artifact identity manifest and clarify which code revision, runtime configuration, and
     optional model produced each result.
  -Obtain independent replication before making comparative or formal-guarantee claims.


Related work added after August
  -Sumers, T. R., Yao, S., Narasimhan, K., & Griffiths, T. L. (2024). Cognitive Architectures for Language
     Agents. Transactions on Machine Learning Research. https://arxiv.org/abs/2309.02427
  -He, J., & Yu, D. (2026). Beyond Memory: A Transactional Continuity Kernel for Long-Lived AI Agents.
     arXiv:2608.11632. https://arxiv.org/abs/2608.11632
  -Gallo, N. (2026). Proof-of-Continuity: A Temporal Model for Authority Propagation in Distributed
     Systems and AI Agents. arXiv:2607.08906. https://arxiv.org/abs/2607.08906

  -Vispute, N. (2026). Reasoning Provenance for Autonomous AI Agents: Structured Behavioral
     Analytics       Beyond      State    Checkpoints    and   Execution    Traces.   arXiv:2603.21692.
     https://arxiv.org/abs/2603.21692
  -Liu, N. F., et al. (2024). Lost in the Middle: How Language Models Use Long Contexts. Transactions of
     the Association for Computational Linguistics, 12, 157–173. https://arxiv.org/abs/2307.03172
  -Autio, C., et al. (2024). Artificial Intelligence Risk Management Framework: Generative Artificial
     Intelligence Profile, NIST AI 600-1. https://doi.org/10.6028/NIST.AI.600-1


Disclosure and publication notice
The combined manuscript is intended for non-confidential external review. Test counts, behavior
summaries, and public conceptual descriptions are reported without source reconstruction instructions
or proprietary thresholds. The author should complete an IP and patent review and verify every
reported artifact before public submission. The 2026 supplement reports engineering observations, not
a peer-reviewed scientific breakthrough.
