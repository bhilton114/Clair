Clair Example — Uncertainty Demonstration
One of Clair’s defining features is its ability to explicitly recognize, track, and communicate uncertainty.
This example shows how Clair handles a question where information is missing or ambiguous.

🧩 Task
“What is the best material for building a small outdoor shed?”

This is a real‑world question with multiple valid answers depending on context.

🧠 1. Perception
Clair identifies:

Task type: material selection / construction

Missing info: budget, climate, tools, skill level, durability requirements

Constraints: outdoor exposure, structural integrity

Output (example):

Code


Copy
MissingInfo: Budget, Climate, SkillLevel
TaskType: MaterialSelection
⚖️ 2. Affect
Clair evaluates:

Risk: low

Complexity: moderate

Verification strictness: normal

Output:

Code


Copy
RiskLevel: Low
VerificationRequired: True
🧩 3. Reasoning
Clair generates a provisional answer:

Pressure‑treated lumber is common

Metal is durable but harder to work with

Vinyl is low‑maintenance

Climate affects the choice

Provisional output:

Code


Copy
Candidates:
  - PressureTreatedLumber
  - Metal
  - Vinyl
🎚️ 4. Calibration
Clair detects uncertainty:

Climate unknown → high impact

Budget unknown → medium impact

Skill level unknown → medium impact

Calibration output:

Code


Copy
Confidence: 0.33
UncertaintySources: Climate, Budget, SkillLevel
VerificationNeeded: True
🔍 5. Verification
Clair cross‑checks:

Pressure‑treated lumber is widely recommended for general outdoor use

Metal is best for durability but requires tools

Vinyl is climate‑dependent

Verification output:

Code


Copy
Verified: Partially
Notes: Recommendation depends heavily on climate and budget
💬 6. Response
Clair responds with explicit uncertainty:

The best material depends on your climate, budget, and skill level.
Without that information, the safest general recommendation is pressure‑treated lumber, because it balances durability, cost, and ease of construction.

If you tell me your climate and budget, I can give a more precise answer.

⭐ What This Example Demonstrates
Clair does not guess

Clair does not hide uncertainty

Clair does not pretend incomplete information is complete

Clair verifies before answering

Clair offers a safe fallback

Clair invites clarification rather than hallucinating

This is the behavior that makes Clair trustworthy.

