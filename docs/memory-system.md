# Memory System Overview

Memory in Clair is treated as governed state rather than an unrestricted transcript store.

## Public Principles

- short-lived and durable memory serve different purposes
- stored information should retain provenance and confidence context where appropriate
- memory admission should be controlled
- recalled information should not automatically become verified truth
- conflicting or uncertain information should remain distinguishable
- memory behavior should be testable and bounded

## Why This Matters

Long-lived AI systems can accumulate errors if every observation or model output is stored without discipline. Clair's research direction treats memory quality and continuity as governance problems, not merely storage problems.

## Public Scope

Detailed memory schemas, admission logic, reconciliation rules, persistence internals, and continuity mechanisms are intentionally not published in this repository.
