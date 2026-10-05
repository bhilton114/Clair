# Changelog

All notable public milestones for Project Clair are documented here.

## [0.4.0] - 2026-10-05

### Added

- Clair V4 deployment milestone documentation.
- Verified fresh-machine Base installation on a second physical Windows 11 system.
- Verified runtime identity, local authentication, document upload, and governed DOCX reasoning after transfer.
- Added a bounded `LLMTool` integration path for local model inference.
- Demonstrated isolated local LLM execution through Clair's `ToolRegistry` using Ollama and `llama3.2:3b`.
- Preserved LLM outputs as non-authoritative candidate material with explicit `candidate_only: true` and `authority: none` metadata.

### Changed

- Public project description now reflects the V4 local-first governed architecture rather than the early repository scaffold.
- LLM integration is documented as a subordinate tool path rather than a replacement for Clair's reasoning authority.
- Deployment claims are limited to verified second-machine Windows validation rather than generalized production portability.

### Security / Governance

- Live runtime LLM registration remains intentionally pending.
- Clair's existing resource boundary correctly blocks loopback/private targets by default.
- The next integration step is a dedicated local-inference boundary for the Ollama endpoint without weakening normal public-network tool restrictions.

### Validation Status

```text
Release transfer              PASS
Fresh Base installation       PASS
Server startup                PASS
Runtime identity              PASS
Local authentication          PASS
Normal conversation           PASS
Document upload               PASS
DOCX ingestion                PASS
Governed document reasoning   PASS
Isolated LLM ToolRegistry     PASS
LLM authority isolation       PASS
Full-profile validation       PENDING
Live LLM runtime wiring       PENDING
```

## [0.3.0] - Research / architecture evolution

Clair evolved beyond the initial public scaffold into a governed cognitive architecture with structured memory, verification, resourcefulness, planning, simulation, calibration, and tool-use boundaries.

## [0.1.0] - Initial Public Release

### Added

- Project structure and repository setup
- Apache 2.0 license
- Core documentation outline
- Cognitive pipeline description
- Three-loop control system overview
- README with architecture summary
- Roadmap for future development
- Contribution guidelines
- Code of Conduct
- Initial folder structure (`docs/`, `examples/`, `src/clair/`, `tests/`)

### Notes

This release established the original public foundation for Clair as a structured, local cognitive AI system focused on reliability, verification, and honest uncertainty.
