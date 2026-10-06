# Project Clair Research

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
