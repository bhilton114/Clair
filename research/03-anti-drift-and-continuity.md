# Anti-Drift, Continuity, and Long-Lived AI

**Project Clair Working Paper**

**Author:** Blake Hilton  
**Project:** Clair  
**Status:** Public research note, October 2026

## Abstract

Long-lived AI systems face a problem that short conversations can hide: state changes over time.

Memories accumulate. Tools change. Models are replaced. Contradictory information appears. Earlier conclusions may become stale. Repeated interaction can gradually alter system behavior.

Project Clair treats this collection of problems as several related forms of **drift** rather than one phenomenon.

This note defines the public conceptual categories and describes how the Clair architecture is intended to limit uncontrolled change.

## 1. Drift Is More Than Personality Change

In AI discussions, drift is sometimes described as a model behaving differently over time.

For persistent cognitive systems, the problem is broader.

Project Clair considers at least five conceptual forms:

### Epistemic Drift

Unsupported, stale, or conflicting claims gradually become treated as reliable knowledge.

### Memory Drift

Repeated storage or revision changes the meaning or reliability of persistent memory.

### Authority Drift

A tool, model, or subsystem gradually begins making decisions outside its intended responsibility.

### Identity Drift

The persistent system becomes dependent on transient components in a way that changes what constitutes the continuing agent.

### Behavioral Drift

Outputs or decisions gradually move away from governing rules, evidence requirements, or intended operating constraints.

## 2. Why Long-Lived Systems Are Different

A one-shot model call has little continuity to protect.

A system expected to operate across days, months, restarts, changing models, changing tools, and accumulated experience has a different problem.

It must preserve useful change without allowing every change to become authoritative.

That means continuity cannot simply mean "keep everything."

Some information should be reinforced.

Some should remain provisional.

Some should be revised.

Some should be rejected.

Some uncertainty should remain unresolved.

## 3. Clair's Public Anti-Drift Strategy

Clair's architecture addresses drift through separation of responsibilities.

Conceptually:

```text
New information
      ↓
Interpretation
      ↓
Uncertainty / evidence evaluation
      ↓
Governed acceptance decision
      ↓
Possible effect on persistent state
```

The goal is to prevent direct uncontrolled paths such as:

```text
tool output → truth
model output → memory
new claim → identity change
single interaction → permanent policy change
```

The exact implementation controls remain private.

## 4. Memory Is Not Truth

One of the most important principles is that persistent storage and epistemic acceptance are not identical.

A system may retain information about a claim without treating the claim as verified truth.

This matters because useful cognition requires remembering ambiguity and conflict, not merely remembering conclusions.

A long-lived AI should be able to represent:

- verified material
- provisional material
- ambiguity
- conflict
- rejection
- unknowns

Without such distinctions, memory can slowly become a collection of accumulated assertions with no meaningful reliability structure.

## 5. Model Replacement and Continuity

Clair's governing architecture is designed to remain outside any single language model.

This creates a testable continuity hypothesis:

> If a language model is replaced, the system's persistent identity, governed memory, and authority structure should remain coherent.

That does not mean behavior will be perfectly identical. Different models have different capabilities.

The important distinction is whether the **resource changed** or whether the **governing system changed**.

## 6. What Has and Has Not Been Proven

Clair has substantial architecture intended to reduce drift, and the system has undergone repeated regression, integration, packaging, restart, and deployment work.

However, long-term anti-drift performance should not be claimed as solved until it is measured directly.

A serious evaluation campaign should deliberately attempt to cause drift.

Possible tests include:

- thousands of sequential interactions
- repeated contradictory claims
- memory reinforcement and revision cycles
- deliberate low-quality evidence
- model replacement
- tool replacement
- repeated restart and recovery
- adversarial candidate output
- stale information
- conflicting authoritative sources
- attempts to make one resource exceed its authority

## 7. Proposed Measurements

Useful metrics could include:

- rate of unsupported claims entering trusted state
- rate of previously rejected claims becoming accepted without new evidence
- identity invariants preserved across model swaps
- memory conflict detection rate
- answer-withholding rate when evidence is insufficient
- recovery behavior after tool failure
- percentage of state changes with traceable justification
- divergence in decisions before and after long-duration stress

These would turn "anti-drift" from an architectural claim into an empirical property.

## 8. Research Position

Project Clair does not assume that drift can be eliminated.

The more defensible objective is:

**make drift observable, bounded, testable, and recoverable.**

A cognitive system should be able to change because learning requires change.

The problem is uncontrolled change without evidence, authority, or traceable reasoning.

## Conclusion

Long-lived AI requires more than good short-term answers.

It requires a disciplined relationship between new information and persistent state.

Project Clair's anti-drift research focuses on preserving that discipline while still allowing the system to learn, revise, and adapt.
