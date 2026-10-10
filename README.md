# Clair 3.9

> **Clair V4: a local-first cognitive architecture where LLMs are tools, not the mind.**

**Models may propose. Clair must govern.**

Clair is a cognitive AI research project built around a separation-of-authority idea: language models, search systems, document readers, calculators, and other tools may contribute information or candidate output, but the surrounding cognitive system retains responsibility for memory, verification, planning, uncertainty handling, and final answer acceptance.

This public repository contains the **Clair 3.9 development lineage**, which became the architectural foundation for the private Clair V4 system.

### Start here

- [Research Note: What Happens When the LLM Is No Longer the AI?](docs/RESEARCH_NOTE.md)
- [V4 Public Status](V4_PUBLIC_STATUS.md)
- [Architecture Overview](docs/architecture.md)
- [Roadmap](ROADMAP.md)
- [Security / Public Disclosure Policy](SECURITY.md)

### Current public milestone

**October 9, 2026 update:** Clair V4 completed a local-network tester milestone: the server was reached from an Android mobile browser on the same Wi-Fi network, with a completed tester report, HTTP 200 health response, verified runtime identity, and no visible UI error in the recorded session. This is a bounded LAN demonstration, not a claim of universal Android compatibility, remote internet access, or a production release.

The Experience Engine also completed a targeted authority-boundary regression checkpoint (20/20 passing checks) covering denial of disputed, blocked, rejected, and conflicting evidence while preserving defined fallback behavior. This is a *targeted regression result*, not an overall suite total or evidence of autonomous learning in all settings.

Clair V4 has been transferred to and validated on a second physical Windows machine from a fresh environment, including startup, authentication, document ingestion, and governed document reasoning.

Local language-model assistance has also been demonstrated experimentally as a bounded, non-authoritative resource.

The active V4 implementation remains private.

## Repository Status

This public repository represents the **Clair 3.9 lineage**.

Clair 3.9 was the major architecture-development stage in which the project established and integrated the systems that later became the basis for Clair V4.

The active project has now advanced beyond this public codebase.

### Current project status

As of October 2026:

- Clair 3.9 completed its major architecture and integration work
- the LIVE02 validation campaign has been closed
- the resulting system was cleaned and packaged as **Clair V4**
- Clair V4 has been installed and validated on a second physical Windows machine
- runtime startup, local authentication, document ingestion, and governed document reasoning passed after transfer
- multiple document formats have been validated during development
- local language-model assistance has been demonstrated as a bounded, non-authoritative resource
- a closed-network mobile tester session and Experience Engine authority-boundary regression have been recorded
- controlled live-runtime model integration, broader robustness testing, and Full-profile validation remain active work

Clair V4 is currently a working research prototype, not a production-certified release.

## What Clair 3.9 Established

The 3.9 development line established the core architectural direction for Clair, including:

- local-first cognitive control
- separation of reasoning, memory, verification, calibration, and tool use
- bounded working and long-term memory
- governed memory admission
- explicit uncertainty and truth-state handling
- planning and simulation
- resourcefulness and external tool use
- verification and answer gating
- reflection and experience handling
- document reasoning
- provider-independent tool architecture
- explicit separation between model output and system authority

These capabilities were developed as parts of one governed system rather than as independent chatbot features.

## Architectural Principle

Clair is not designed around the idea that a language model is the entire AI system.

Its governing architecture is intended to retain ownership of:

- identity
- memory
- verification
- reasoning control
- planning
- tool arbitration
- acceptance of candidate answers

External tools and language models may provide information or candidate output, but they are not intended to become Clair's truth authority or system identity.

## From 3.9 to V4

Clair V4 is the packaging and deployment milestone built from the architecture proven during the 3.9 development cycle.

The transition to V4 focused on:

- cleaning the development tree
- separating retained runtime material from historical development artifacts
- packaging the server and core system
- validating fresh installation
- proving operation outside the original development machine
- improving document ingestion and release usability
- beginning controlled local-model integration

The current V4 source tree remains private while validation and intellectual-property work continue.

## Verified V4 Deployment Milestone

The V4 package has been transferred to a second physical Windows machine and validated from a fresh environment.

Publicly verified:

```text
Package transfer            PASS
Fresh Base installation     PASS
Server startup              PASS
Runtime identity            PASS
Local authentication        PASS
Normal conversation         PASS
Document upload             PASS
Document ingestion          PASS
Governed document reasoning PASS
```

This demonstrates reproducible deployment beyond the development PC. It does not claim universal portability or production readiness.

## Local-First Access and Experience Work

A local-network tester workflow has been demonstrated: the Clair server runs on a Windows host and a separate device on the same private Wi-Fi network accesses its tester interface. No public internet connection is required for that LAN path. The recorded Android browser report establishes a successful session, not full deployment or device-compatibility coverage. The APK packaging/testing workflow remains experimental.

Experience handling is being developed with explicit separation between observed or candidate material and information admitted into authoritative state. Recent bounded tests exercised rejection and conflict handling. Wider persistence, replay, learning quality, and adversarial robustness still require continued validation.

## Current Research Priorities

Current Project Clair work is focused on:

1. controlled local-model integration
2. preserving strict authority and resource boundaries
3. Full-profile installation validation
4. robustness and failure-path testing
5. repeatable clean-start demonstrations
6. capability benchmarking and gap discovery
7. independent technical evaluation

## Public Repository Scope

This repository is intentionally limited to public documentation, historical public code, selected examples, and non-sensitive research material.

It does **not** publish:

- the active Clair V4 source tree
- private regression or benchmark fixtures
- internal governance implementation
- security-sensitive controls
- exact thresholds or guard values
- proprietary routing or continuity logic
- credentials, secrets, or private deployment configuration

Some source, tests, and examples in this repository are historical and should not be interpreted as the current V4 implementation.

See:

- `V4_PUBLIC_STATUS.md`
- `ROADMAP.md`
- `SECURITY.md`
- `docs/`

## Research

Project Clair research includes work on governed memory and long-lived agent continuity.

Paper:

**Beyond Memory: A Transactional Continuity Kernel for Long-Lived AI Agents**

## License

Apache License 2.0. See `LICENSE`.

## Project

Created and developed by Blake Hilton.
