# Project Clair Research

## October 2026 full-text research manuscripts

The current public working manuscript collection has **six versioned papers** with individual citation URLs and BibTeX entries. **[Browse the collection](papers/2026-10-13/README.md)**. These are not peer-reviewed papers or arXiv publications; the index explains the evidence status and pending evaluations.

This directory collects public, non-sensitive research material from Project Clair.

The documents here describe the ideas, architecture, and validation evidence behind the project without publishing the active Clair V4 source tree, private tests, exact thresholds, security-sensitive controls, or proprietary implementation seams.

## Working Papers and Research Notes

1. [Governed Cognitive Architecture](01-governed-cognitive-architecture.md)  
   A public overview of Clair's core research direction: cognition as a governed system rather than a single model.

2. [Models as Non-Authoritative Resources](02-models-as-non-authoritative-resources.md)  
   Why Clair treats language models, search, documents, calculators, and other tools as contributors rather than truth authorities.

3. [Anti-Drift, Continuity, and Long-Lived AI](03-anti-drift-and-continuity.md)  
   A research note on epistemic, memory, authority, identity, and behavioral drift in persistent AI systems.

4. [What Happens When the LLM Is No Longer the AI?](../docs/RESEARCH_NOTE.md)  
   A concise introduction to the central architectural question behind Project Clair.

## Current Evidence

Publicly described Clair V4 validation includes:

- package transfer to a second physical Windows machine
- fresh-environment installation
- successful server startup
- fresh local authentication
- normal conversation flow
- document upload and ingestion
- governed document reasoning
- multiple document-format tests during development
- experimental use of a local language model as a bounded, non-authoritative resource

These are engineering validation results, not claims of benchmark superiority, production certification, consciousness, or general intelligence.

## Citation and Research Linking

These materials are intended to provide a stable GitHub reference for Project Clair research, including links from researcher profiles, preprints, technical notes, demonstrations, and future publications.

When a formal paper or preprint is published elsewhere, this directory can host the related implementation notes, public architecture summaries, validation evidence, and reproducibility material that are safe to disclose.

## Disclosure Boundary

Project Clair deliberately separates public research claims from private implementation details.

Not published here:

- active Clair V4 source
- exact routing and authority logic
- exact guard values or thresholds
- private benchmark or regression fixtures
- security-sensitive network and execution controls
- proprietary continuity mechanisms
- credentials or deployment secrets

## Author

Blake Hilton  
Project Clair

## Manuscripts undergoing review

- **Governed Experience Reuse in a Local-First Cognitive Agent** (Blake Hilton, 2026), version 0.1, technical preprint draft. [Read the review copy](https://github.com/bhilton114/Clair_v.3/blob/docs/clair-v4-public-overview-20261012/research/2026-governed-experience-reuse.md). It is on a review branch; it is **not** yet a stable versioned release, peer-reviewed work, or DOI-registered publication.

## Research navigation and publication integrity

- [Development stages](../docs/PROJECT_STAGES.md) explains the historical repositories and active V4 status.
- [Public documentation map](../docs/SITE_MAP.md) provides a single starting point for technical reviewers.
- **Authorship and publication verification:** An earlier project README lists *Beyond Memory: A Transactional Continuity Kernel for Long-Lived AI Agents*. Verify the original publication's authorship and identifier independently before adding it to an ORCID record.
- For ORCID, prefer a reviewed, released manuscript with a stable public URL and, where available, an archive DOI.
