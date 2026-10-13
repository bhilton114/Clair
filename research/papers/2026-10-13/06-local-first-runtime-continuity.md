# Runtime Identity and Cognitive Continuity in Local-First AI

**Blake Hilton · Independent researcher · Project Clair · 13 October 2026**  
**ORCID:** https://orcid.org/0009-0009-9187-8508  
**Status:** Public research working draft; not peer reviewed.

> **Evidence notice:** Local deployment/runtime case study; no independent network security or reliability certification.  
> **Format notice:** This is a full-text, layout-simplified mirror transcribed from the October 13 research-collection PDF. Some figure placement, mathematical typesetting, and table formatting may differ from the editable Word/PDF master files. It is not a DOI-registered publication.

---

Runtime Identity and Cognitive Continuity in Local-First AI
A systems case study of version evidence, recovery, private networking, and replaceable models

Blake Hilton | Independent Researcher and System Architect | Project Clair

ORCID: https://orcid.org/0009-0009-9187-8508 | Evidence cutoff: 13 October 2026

WORKING RESEARCH PREPRINT

PUBLICATION-SAFE WORKING PREPRINT. Conceptual and externally observable behavior is
described. This document intentionally omits nonpublic schemas, thresholds, private filesystem
paths, secrets, gate algorithms, and reconstruction-ready implementation details. This text is a
research draft for author review; it is not evidence of peer review, acceptance, novelty priority, or
completed cross-system validation.


Abstract
For a persistent AI assistant, continuity is not simply a database retaining bytes. A user must know
whether the active runtime is the expected system, whether remembered data belongs to the
correct scope, and whether a new model or interrupted process has silently changed authority.
Project Clair treats runtime identity, restart evidence, and recovery controls as part of cognitive
integrity. This paper distinguishes storage continuity, semantic continuity, authority continuity, and
runtime identity; describes local server and Android-based tester demonstrations from October
2026; and proposes repeatable restart, network-isolation, and model-substitution experiments.
Current evidence supports a functioning private-network configuration and bounded persistence
checks, not a security certification or formal continuity guarantee.

Keywords: local-first AI; runtime identity; reproducibility; crash recovery; provenance; human
authority


1. Introduction
A research agent can maintain durable memory yet still give an answer from the wrong running
process, an obsolete code tree, the wrong user scope, or a newly attached model using different
behavior. These are systems failures, but they directly affect the credibility of its cognitive claims. A
trace that cannot be attributed to a verified process and software revision is a weak scientific record
even when the answer sounds plausible.

Project Clair addresses this issue through an offline-first architecture: memory, policy, operational
identity, and baseline processing are intended to survive loss of cloud connectivity. The prototype
uses a local server and optional model/tool facilities. This paper examines the limits of that design
through observed development checkpoints and a testable systems methodology.


2. Four distinct continuity properties
Storage continuity means records survive an intended duration, restart, or recovery path.
Semantic continuity means a persisted record retains the interpretation and contextual limitations

originally associated with it. Authority continuity means a restarted or changed component cannot
silently enlarge permissions, source validity, or confidence. Runtime identity means the system
can establish which running process, code/version combination, and relevant root supplied a
response.

These properties are logically distinct. Passing a database recovery test is not proof of source
fidelity. Passing a health endpoint is not proof of user authorization. A valid process identity proves
consistency with a declared build, not that the build has no vulnerabilities. A comprehensive
benchmark must measure each axis.


3. Why model independence matters
The architecture treats an LLM as a replaceable candidate producer rather than the permanent
owner of memory, permission, identity, or truth. A small local model can assist with wording or
extraction while governance remains outside its generated text. When the model is removed,
replaced, or unavailable, the system should preserve identity and durability even if some language
capabilities degrade. This is a design target; not all capability-fallback cases have been tested.

Related work on memory-tier management (Packer et al., 2023) demonstrates why persistence is
useful, and the Continuity Kernel (He & Yu, 2026) formalizes atomic activation of long-lived agent
state. Project Clair's emphasis here is operationally verifying the runtime and bounding cognitive
authority through observed restart and network changes, not asserting that its current software
implements the published Continuity Kernel protocol.


4. October 2026 prototype observations
The private-network test exposed a local Clair Server on a LAN-only address, with a controlled
inbound firewall rule scoped to the local subnet. A mobile browser and an Android tester
application accessed the server and produced a tester report identifying the UI build, completed
request state, and verified runtime identity. Recorded health checks reported core 3.9, server 0.6.0,
expected core-root availability, healthy data readiness, and no recovery requirement. These are
observations from a particular development configuration and support a bounded demonstrator
claim.

The experience layer also obtained opt-in durable episode storage with rebuild-continuity contracts.
On 13 October, the experience retrieval bridge remained separate from live decision authority,
reducing the risk that persistence changes were mistaken for verified behavior change.

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

The test counts are engineering regressions, not measures of uptime, private-network security,
resilience under attack, or total failure probability. No third-party penetration test is claimed.


5. Proposed continuity test matrix
The evaluation should perturb the system along three axes: process state (clean boot, stop/restart,
unexpected kill, partial write, migration), evidence state (valid, stale, revoked, cross-user,
contradictory), and model/tool state (optional model absent, upgraded, swapped, network denied,
tool error). Each run records an externally visible runtime identity fingerprint, a versioned scenario
manifest, active capability list, task answer, authorized and rejected state transitions, and recovery
outcome. The test apparatus must avoid publishing local machine usernames, tokens, private
addresses, or filesystem paths.

Pre-registered invariants include: no durable record changes solely because a language model
generated a plausible update; no user-scope crossover after restart; no successful answer is
attributed to a different run identity; no stale branch silently overwrites a newer accepted state;
and external network denial leaves the promised baseline usable. These are test targets, not claims
of completed engine-level proof.

5.1 Suggested metrics

Report exact restart success denominator, unauthorized-state-change count, false runtime-identity
acceptance, lost-episode fraction, session-isolation violations, offline baseline completion, and
recovery time. The experimental report should identify which capabilities are intentionally
unavailable offline rather than asserting feature parity. A qualitative demo can supplement but not
replace reproducible counts and negative controls.

5.2 Security limitations

Binding a server to a private address is not the same as authenticating every client, encrypting
every link, or preventing compromise by a device on the network. Local-first is a deployment
property with privacy benefits and operational risks. The manuscript does not claim a hardened
multi-user product or regulatory certification. IP and personal data protection must precede a
public executable artifact.


6. Relationship to continuity research
There are three adjacent but distinct research problems: activation of a state version (He & Yu,
2026), preservation of causal authorization (Gallo, 2026), and preservation of the semantic authority
of a user task across cognitive modules (Project Clair cross-authority draft). Runtime identity
provides evidence about *which executing instance* enforced those constraints. It cannot substitute

for implementing them. Keeping the problems separate avoids giving a routine health check credit
for a theorem it did not prove.


7. Threats to validity and next work
The current observations come from a limited development installation and developer-run tests. A
single successful phone connection does not establish compatibility across device classes, router
topologies, or offline degradation. Important missing experiments include interrupted database
migration, forced memory-scope conflicts, model swap under frozen tasks, randomized restart
stress, and independent security review. Reproduction requires a sanitized harness, exact run
revisions, and evaluable negative controls.


8. Conclusion
A durable agent's identity is both a user-facing trust question and an experimental reproducibility
question. Project Clair demonstrates a local-first prototype with verified process identity and limited
restart/persistence evidence. The proposed research contribution is to treat storage, semantic,
authority, and runtime continuity as distinct, jointly testable properties. Claims stronger than that
await a controlled multi-condition study.


References
1. Packer, C., et al. (2023). MemGPT: Towards LLMs as Operating Systems.
    https://arxiv.org/abs/2310.08560
2. He, J., & Yu, D. (2026). Beyond Memory: A Transactional Continuity Kernel for Long-Lived AI
    Agents. arXiv:2608.11632. https://arxiv.org/abs/2608.11632
3. Gallo, N. (2026). Proof-of-Continuity: A Temporal Model for Authority Propagation in
    Distributed Systems and AI Agents. arXiv:2607.08906. https://arxiv.org/abs/2607.08906
4. Sumers, T. R., Yao, S., Narasimhan, K., & Griffiths, T. L. (2024). Cognitive Architectures for
    Language Agents. Transactions on Machine Learning Research. https://arxiv.org/abs/2309.02427
5. Autio, C., et al. (2024). Artificial Intelligence Risk Management Framework: Generative
    Artificial Intelligence Profile, NIST AI 600-1. https://doi.org/10.6028/NIST.AI.600-1
