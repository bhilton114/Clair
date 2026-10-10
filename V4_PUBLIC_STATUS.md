# Clair V4 Public Status

**Status date: October 2026**

This page summarizes what can be stated publicly about Clair V4 without exposing private implementation details.

## Verified Milestones

- V4 release package assembled and cleaned
- fresh Base installation validated on a second physical Windows 11 machine
- runtime identity verification passed after transfer
- new local authentication bootstrap passed
- normal conversation path passed
- document upload and governed DOCX reasoning passed on the second machine
- multiple document formats validated during development
- local language-model assistance demonstrated through Clair's bounded tool layer
- model output preserved as non-authoritative candidate material

## Current Work

- completing a narrow local-inference network boundary
- integrating local model access into the shared runtime without weakening existing resource controls
- Full-profile installation validation
- broader robustness testing
- repeatable clean-start demonstrations
- capability benchmarking and independent evaluation

## Important Limits

The following are not claimed yet:

- universal OS portability
- production certification
- automatic local-model routing in the live runtime
- Full-profile cross-machine validation
- complete elimination of hallucination

Clair V4 should currently be described as a working local-first cognitive AI research prototype with cross-machine deployment evidence and governed tool/model integration under active validation.

## October 9, 2026: Local-Network and Experience Checkpoints

- A tester interface was exercised from an Android mobile browser over a private local Wi-Fi network to the Windows-hosted Clair server.
- The recorded tester session completed, with an HTTP 200 health response, verified runtime identity, and no visible UI error.
- A targeted Experience Engine authority-boundary regression reported 20/20 passing checks against rejected, disputed, blocked, conflicting and otherwise excluded evidence paths, with expected fallbacks preserved.

These checkpoints do not establish broad APK compatibility, internet-facing deployment readiness, complete experience learning, or overall system correctness. The APK workflow and expanded validation remain experimental.
