# Architecture Overview

Clair V4 is a local-first cognitive AI research system built around separation of responsibilities and explicit governance.

This public document intentionally describes the architecture at a high level. Internal implementation details, control logic, security boundaries, private evaluation fixtures, and proprietary integration methods are not published here.

## Public Architecture Principles

Clair separates major cognitive responsibilities rather than placing them inside a single model call.

At a high level, the system contains functions for:

- input interpretation
- reasoning
- uncertainty assessment
- evidence use
- verification
- memory
- planning
- tool use
- answer acceptance
- reflection

The core design rule is that no external model or tool automatically becomes the authority for truth, identity, memory, or final output.

## Local-First Operation

Clair is designed so that its governing state and core cognitive control can remain local. External services may provide optional capability, but they are treated as resources rather than system identity.

## Model Independence

Language models can be used as bounded resources. Their outputs are treated as candidate material subject to Clair-side processing and acceptance.

## Public Scope

The public repository documents research goals, verified milestones, and architecture principles. It does not publish the full V4 implementation or private operational details.
