# Clair 3.9

**Public repository for the Clair 3.9 development lineage and the research architecture that became the foundation of Clair V4.**

Clair is a local-first cognitive AI research project focused on governed reasoning, verification, memory, planning, tool use, uncertainty handling, and long-lived system continuity.

This repository does **not** contain the current private Clair V4 implementation.

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
