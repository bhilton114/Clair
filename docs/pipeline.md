# Public Cognitive Pipeline

Clair uses a governed processing pipeline.

A simplified public view is:

```text
Input
  ↓
Interpretation
  ↓
Reasoning and resource use
  ↓
Calibration
  ↓
Verification
  ↓
Acceptance
  ↓
Response
  ↓
Governed learning / reflection
```

This diagram is intentionally conceptual.

## Key Properties

- evidence gathering is separate from truth assignment
- uncertainty can survive through the pipeline
- tools and models provide bounded inputs
- final response acceptance remains under Clair-side governance
- memory updates are not assumed from every interaction

## What Is Not Public

The repository does not publish internal routing rules, thresholds, ownership contracts, execution order details, security controls, or private validation fixtures.
