# Clair

**A local-first cognitive AI research project focused on governed reasoning, verification, memory, and bounded tool use.**

Clair is being developed as a cognitive architecture rather than an LLM wrapper. Its public research direction emphasizes separation of responsibilities, explicit uncertainty, governed memory, evidence discipline, and local-first operation.

## Current Status

**Current milestone: Clair V4**

As of October 2026, Clair V4 has reached a verified deployment milestone.

Publicly verified:

- a cleaned V4 package has been assembled
- the Base installation profile was validated on a second physical Windows machine
- runtime startup, authentication, and document reasoning passed after transfer
- multiple document formats have been validated during development
- local language-model assistance has been demonstrated as a bounded, non-authoritative resource
- broader runtime integration and additional validation are still in progress

Clair V4 is a working research prototype. It is not presented as production-certified software.

## Design Principles

Clair's public design goals are:

- local-first operation
- explicit uncertainty handling
- separation of cognitive responsibilities
- governed memory
- verification and evidence discipline
- bounded tool use
- provider-independent cognition
- honest failure when evidence is insufficient

## Local-First Architecture

Clair is designed so its governing state and core decision structure can remain local.

External services and language models may extend capability, but the architecture is intended to prevent them from becoming the system's identity, memory authority, or truth authority.

## Deployment Evidence

The V4 package has been transferred to a second physical Windows machine and successfully validated from a fresh environment.

This demonstrates reproducible deployment beyond the development machine. It does not claim universal operating-system compatibility or production readiness.

## Research Direction

Current public research areas include:

- governed local model assistance
- deployment reproducibility
- document reasoning
- memory continuity
- verification
- long-lived agent stability
- capability benchmarking
- independent evaluation

## Research Paper

Project Clair research includes:

**Beyond Memory: A Transactional Continuity Kernel for Long-Lived AI Agents**

## Public Repository Scope

This repository is intentionally limited to public documentation, research direction, milestone summaries, historical examples, and selected non-sensitive material.

The full active V4 implementation, private evaluation fixtures, internal governance logic, security controls, exact thresholds, and proprietary control mechanisms are not published here.

See:

- `V4_PUBLIC_STATUS.md`
- `SECURITY.md`
- `ROADMAP.md`
- `docs/`

## License

Apache License 2.0. See `LICENSE`.

## Project

Created and developed by Blake Hilton.
