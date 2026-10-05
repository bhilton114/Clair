# Clair

**A local-first cognitive AI architecture for governed reasoning, verification, memory, and tool use.**

Clair is not designed as an LLM wrapper. Its core architecture keeps identity, memory, verification, calibration, planning, and answer acceptance outside any language model. LLMs and external services are treated as bounded tools rather than cognitive authorities.

## Current Status

**Project milestone: Clair V4**

As of October 2026, the V4 package has reached a verified deployment milestone:

- Cleaned and packaged local server/runtime
- Fresh installation validated on a second physical Windows 11 machine
- Runtime identity verification passed after transfer
- Independent local user creation and authentication passed
- Document upload and governed document reasoning passed on the second machine
- TXT, DOCX, PDF, CSV, and XLSX document paths validated in development
- Local LLM attachment demonstrated through Clair's ToolRegistry using Ollama and `llama3.2:3b`
- LLM output remains explicitly non-authoritative candidate material
- Live automatic LLM routing is still under controlled integration and is not yet enabled

This is a working research prototype, not a production-certified system.

## Design Principle

Clair follows a simple architectural rule:

> **Models may propose. Clair must govern.**

The system separates responsibilities so that no single model call owns truth, identity, memory, routing, or final acceptance.

## Cognitive Architecture

A simplified path is:

```text
Input
  ↓
Perception / Intake
  ↓
Routing
  ↓
Reasoning
  ↓
Resourcefulness / Tool Use
  ↓
Calibration
  ↓
Verification
  ↓
Answer Gate
  ↓
Response
  ↓
Reflection / Governed Memory
```

Supporting subsystems include working memory, long-term memory, episodic memory, planning, simulation, uncertainty handling, evidence scoring, resource recovery, and post-answer reflection.

## Tool and Provider Boundary

Clair's tool layer uses explicit request/result contracts and registry-based execution.

External providers do not receive truth authority.

A current LLM integration proof follows this shape:

```text
ToolRequest
  ↓
ToolRegistry
  ↓
LLMTool
  ↓
Local model backend
  ↓
ToolResult
  ↓
Clair-side interpretation and acceptance
```

The isolated local-model test returned:

- provider: Ollama
- model: `llama3.2:3b`
- `candidate_only: true`
- `authority: none`

The next integration step is a dedicated loopback inference boundary so local model traffic can be authorized without weakening Clair's existing public-network resource boundary.

## Local-First

Clair is designed so its governing cognition does not depend on cloud infrastructure.

Network tools may extend capability, but the architecture is intended to preserve local identity, memory, reasoning control, and governance when external services are unavailable.

## Reliability Philosophy

Clair is designed to:

- expose uncertainty rather than conceal it
- separate evidence gathering from truth assignment
- use bounded tool execution
- preserve provenance and lineage where required
- reject unsupported answers when evidence is insufficient
- keep memory writes governed
- prevent external tools or models from silently becoming system authority

No AI system can guarantee zero hallucinations. Clair's goal is to reduce unsupported output through explicit architecture, verification, and acceptance controls.

## V4 Deployment Proof

The V4 release package was transferred to a second physical Windows machine and validated from a fresh environment.

Verified on that machine:

```text
Release transfer              PASS
Fresh Python environment      PASS
Base installation             PASS
Server startup                PASS
Runtime identity              PASS
Local authentication          PASS
Normal conversation           PASS
Document upload               PASS
DOCX ingestion                PASS
Governed document reasoning   PASS
```

This demonstrates reproducible deployment beyond the development machine. Broader production portability remains a separate validation target.

## Research Direction

Current work is focused on:

1. completing governed local LLM integration
2. preserving strict network/resource boundaries
3. Full-profile installation validation
4. robustness and failure-path testing
5. repeatable clean-start demonstrations
6. capability benchmarking and gap discovery
7. independent technical evaluation

## Research Paper

Project Clair research includes work on long-lived agent continuity, governed memory, and transactional identity/state handling.

Paper title:

**Beyond Memory: A Transactional Continuity Kernel for Long-Lived AI Agents**

## Repository Scope

This public repository documents Clair's architecture, research direction, examples, and selected implementation material.

The active V4 development system contains additional components and validation infrastructure that are not necessarily mirrored here.

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Project

Created and developed by Blake Hilton.

Clair began as an independent effort to explore whether a durable synthetic cognitive system could be built around governed reasoning rather than model authority.
