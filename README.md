# Project Clair

### Local-first cognitive AI research

**A governed reasoning architecture where models propose and the system decides what can be trusted.**

Project Clair is an independent research project developed by **Blake Hilton**. It explores persistent AI systems in which verification, memory admission, planning, uncertainty, and answer acceptance remain governed outside optional language models.

> **Current status (October 2026):** Clair V4 is an active **private** research prototype. This is the **canonical public project hub**, not the released V4 application or a production-ready product.

## Start here

| Explore | What you will find |
| --- | --- |
| [Architecture](docs/architecture.md) | High-level cognitive system and trust boundaries |
| [V4 development status](V4_PUBLIC_STATUS.md) | Recent reported deployment and validation milestones |
| [Research and papers](research/README.md) | Research notes, manuscript status, publication links |
| [Development stages](docs/PROJECT_STAGES.md) | One timeline linking the historical V3 and V3.9 repositories |
| [Roadmap](ROADMAP.md) | Current engineering and research priorities |
| [Documentation map](docs/SITE_MAP.md) | Quick navigation by audience |
| [Security and disclosure](SECURITY.md) | Responsible reporting and limits of public details |

## The research thesis

Traditional model-centered assistants can conflate generated text, retrieved information, and trusted memory. Clair investigates explicit boundaries between:

1. **Proposals** produced by language models and other tools.
2. **Evidence** evaluated for provenance, relevance, uncertainty, and conflict.
3. **Authority** to change persistent memory, influence decisions, or approve actions.

The goal is a reusable, testable cognitive control architecture, not a claim of artificial general intelligence or consciousness.

## High-level system

```text
Input / local client
      |
Perception & routing
      |
Reasoning / planning <---- optional models, search, documents
      |
Evidence & verification
      |
Authority / answer gates
      |
Response and governed reflection
      |
Memory / experience admission
```

This is a public-facing conceptual model, not a full inventory or implementation specification.

## Development history

| Stage | Core emphasis | Public home |
| --- | --- | --- |
| Early versions | Cognitive foundations and reasoning loops | This repository's earlier development history |
| V3.3–V3.4 | Resourcefulness, fallback, document reasoning, memory verification | [Clair_v.3](https://github.com/bhilton114/Clair_v.3) |
| V3.9 | Governance, memory authority, uncertainty, validation | [Project-Clair-v3.9](https://github.com/bhilton114/Project-Clair-v3.9) |
| V4 | Private packaging, runtime integration, local-network testing, governed experience reuse | [Public status](V4_PUBLIC_STATUS.md) |

**One project, multiple research stages.** Older repositories remain accessible for provenance. They are not competing current editions.

## Selected local development evidence

- Earlier public V3 smoke checkpoint: **416 passed / 1 skipped**.
- October 9, 2026: a reported **20/20** targeted experience authority-boundary regression pass.
- October 2026: developer-reported successful local Wi-Fi mobile-browser tester session, including health HTTP 200 and verified runtime identity.

These are different test scopes. They do **not** constitute independent replication, commercial certification, or proof of unrestricted autonomous learning. See [current status](V4_PUBLIC_STATUS.md) and the [research index](research/README.md) for context.

## Publications

Research notes and preprint drafts are catalogued in the [research index](research/README.md). Draft manuscripts are labeled as such; verify author metadata and stable identifiers before citing them in ORCID.

## Public disclosure boundaries

The current Clair V4 source, sensitive governance implementation, security controls, private test fixtures, credentials, and proprietary research internals are **not** included here. Public documentation is intended to make the research understandable and its evidence inspectable without misrepresenting the available artifacts.

## Author and project

**Blake Hilton** · Independent developer and researcher  
**Project Clair** · Active research prototype

For technical background, [read the architecture](docs/architecture.md). For the development chronology, [start with the stages](docs/PROJECT_STAGES.md).
