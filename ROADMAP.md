# Clair Roadmap

A structured plan for the evolution of a local-first governed cognitive AI system.

Clair is built around a separation-of-authority model: models and tools may propose or provide evidence, while Clair retains ownership of identity, memory, verification, routing, and final acceptance.

## Current Milestone: Clair V4

### Proven

- Local server/runtime package assembled and cleaned
- Base installation validated on a second physical Windows 11 machine
- Runtime identity verification passed after transfer
- Fresh local user creation and authentication passed
- Document upload and governed DOCX reasoning passed on the second machine
- TXT, DOCX, PDF, CSV, and XLSX document paths validated during development
- Isolated local LLM attachment proven through `ToolRegistry`
- Ollama + `llama3.2:3b` verified as a subordinate inference resource
- LLM output explicitly marked non-authoritative

### Immediate Next Steps

1. Build a dedicated loopback-only local inference boundary for Ollama.
2. Route Ollama HTTP through Clair's bounded `SafeFetchTransport`.
3. Register `llm_tool` in the shared runtime registry behind explicit configuration.
4. Add `llm_assistance -> llm_tool` to the capability map.
5. Prove normal Clair behavior is unchanged when the LLM capability is enabled.
6. Keep automatic LLM selection disabled until explicit runtime invocation is proven.
7. Validate the Full installation profile.
8. Run failure-path and malformed-input robustness tests.
9. Freeze a repeatable clean-start demo.
10. Begin capability benchmarking and gap discovery.

## Architecture Direction

### Local-first cognition

Clair should remain operational without cloud model access. External providers may extend capability, but they must not become the system's identity, memory authority, or truth authority.

### Governed model use

Target path:

```text
Task / Need Detection
        ↓
Capability Planning
        ↓
Tool Selection
        ↓
LLMTool
        ↓
Local or remote provider
        ↓
ToolResult
        ↓
Clair interpretation
        ↓
Calibration / Verification
        ↓
Answer Gate
```

### Resource-bound tool execution

Network-capable tools must remain behind explicit boundaries. Local inference will receive a narrow loopback exception rather than a general private-network permission.

## Validation Roadmap

### Deployment

- [x] Clean V4 package
- [x] Second-machine ZIP integrity verification
- [x] Fresh Base install
- [x] Server startup
- [x] Runtime identity verification
- [x] Authentication bootstrap
- [x] Foreign-machine document reasoning
- [ ] Fresh Full-profile install
- [ ] Additional OS / environment validation

### Documents

- [x] TXT
- [x] DOCX
- [x] PDF
- [x] CSV
- [x] XLSX
- [x] Negative no-source behavior
- [x] Conversation isolation
- [ ] Broader malformed/unsupported-file robustness suite

### LLM integration

- [x] Provider-independent LLM tool contract
- [x] Ollama backend proof
- [x] Isolated ToolRegistry execution
- [x] Candidate-only authority boundary
- [ ] Dedicated local-inference network boundary
- [ ] Shared runtime registration
- [ ] Explicit Clair-side invocation
- [ ] Governed interpretation + answer-gate proof
- [ ] Automatic selection policy
- [ ] Additional provider adapters

### Evaluation

- [ ] Capability matrix freeze
- [ ] Repeatable benchmark harness
- [ ] GAIA re-evaluation without answer hardcoding
- [ ] Independent architecture review
- [ ] External evaluator package
- [ ] Regression tracking across future capability additions

## Research Direction

Long-term research priorities include:

- transactional continuity for long-lived agents
- memory truth discipline
- recursive inquiry under bounded authority
- simulation and experience learning
- environmental evidence resolution
- post-answer reflection and governed admission
- anti-drift and long-term stability mechanisms
- provider-independent cognition
- situated local-first intelligence

## Long-Term Vision

Clair aims to explore a class of AI systems that are:

- local-first
- transparent
- verifiable
- structured
- provider-independent
- memory-governed
- resistant to unsupported answers
- capable of using LLMs without being defined by them

The goal is not merely a better chatbot. It is a durable cognitive architecture in which learned models are resources inside a governed system.
