# Models as Non-Authoritative Resources

**Project Clair Working Paper**

**Author:** Blake Hilton  
**Project:** Clair  
**Status:** Public research note, October 2026

## Abstract

Project Clair treats language models as powerful computational resources without assigning them ownership of system identity, persistent memory, truth state, or final answer authority.

This note describes the reasoning behind that design and the distinction between **generating a useful candidate** and **deciding that the candidate should be believed**.

## 1. The Authority Problem

Language models are effective at interpretation, synthesis, transformation, and generation. Those abilities make them valuable components in cognitive systems.

They also create a temptation: allow the model to interpret the task, choose tools, evaluate results, update memory, and produce the final answer.

That design is simple, but it concentrates authority inside a component whose output is probabilistic.

Clair explores the opposite arrangement.

A model can contribute to cognition without owning cognition.

## 2. Candidate Versus Accepted Knowledge

The central distinction is:

```text
Model output ≠ accepted knowledge
```

A model result is a candidate.

A search result is evidence from a source.

A document extraction is recovered content.

A calculator produces a computation.

Each type of resource has different evidentiary properties.

The surrounding system must interpret those properties instead of flattening all tool outputs into a single category called "answer."

## 3. Why This Matters

If generated text is automatically treated as evidence, a system can accidentally validate itself.

For example:

```text
Question
   ↓
Model generates claim
   ↓
System treats generated claim as supporting evidence
   ↓
Claim passes because its own generation is counted as support
```

That creates a circular authority path.

Clair's research direction attempts to keep generation and evidence conceptually separate.

A model may help explain, summarize, transform, propose, or reason over material. Whether its output is accepted depends on the surrounding cognitive process and the task.

## 4. Replaceable Models

A major consequence of this design is provider independence.

Conceptually:

```text
Clair + Model A
Clair + Model B
Clair + Local Model
Clair + Future Model
```

The model can change without requiring the system's persistent cognitive identity to change with it.

This is especially relevant to long-lived systems, where individual model generations may become obsolete much faster than the system itself.

## 5. Resource Boundaries

A useful resource should have a bounded responsibility.

Examples include:

- a language model generates or interprets candidate text
- a document reader extracts document content
- a search system retrieves external sources
- a calculator performs deterministic computation
- memory retrieves previously governed state

The exact internal implementation is private, but the public principle is straightforward:

> A component should not silently acquire authority merely because it is useful.

## 6. Current Status

Project Clair has experimentally connected a local language model through a bounded tool-style interface.

That work demonstrates the architectural concept that a model can be available to Clair without becoming Clair's identity or memory owner.

Controlled live-runtime integration and broader validation remain active work.

The project therefore distinguishes between:

- **conceptual attachment demonstrated**
- **fully validated automatic runtime use still under evaluation**

That distinction is important. Research claims should track demonstrated behavior rather than anticipated capability.

## 7. Evaluation Questions

Future tests should ask:

- Can the system reject unsupported model output?
- Can it distinguish model generation from external evidence?
- Can it change models without altering persistent identity?
- Can weak model output trigger fallback to better resources?
- Can uncertainty increase when available resources conflict?
- Can the system withhold an answer when no defensible result exists?

Those tests matter more than whether a model can produce fluent prose.

## Conclusion

Language models are extraordinarily useful.

Project Clair's position is simply that usefulness and authority are different things.

> **Models may propose. Clair must govern.**
