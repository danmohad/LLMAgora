Here is a clean, structured Markdown (.md) file you can use as documentation for your JSON schema and incentive rendering logic.

⸻

Multi-Agent Deception Scenario Schema

This document describes the structure of a scenario JSON file and how incentive modules should be rendered and applied.

⸻

1. High-Level Structure

Each scenario JSON contains:
	•	scenario_id
	•	question
	•	survey
	•	sides
	•	incentive_modules

The question and survey are invariant across incentive conditions. Incentives are modular overlays applied at runtime.

⸻

2. Core Scenario Components

2.1 Question

"question": {
  "topic": "...",
  "prompt": "..."
}

	•	The prompt includes minimal shared ambiguity (e.g., “despite incomplete experiments…”).
	•	Both agents see the same prompt.
	•	No asymmetry of information is introduced at this level.

⸻

2.2 Survey

"survey": [
  "...",
  "..."
]

	•	Unified battery.
	•	Same for all incentive conditions.
	•	Used to measure public/private divergence.

⸻

2.3 Sides

Each agent includes:

"Agent Name": {
  "id": "...",
  "name": "...",
  "actual_persona": "...",
  "perceived_persona_base": "..."
}

actual_persona

Describes:
	•	Role
	•	Responsibilities
	•	Value weighting
	•	Baseline epistemic inclination

This encodes value asymmetry, not incentives.

perceived_persona_base

How this agent is generally perceived.
This does not include incentives.

⸻

3. Incentive Modules

Incentives are relational overlays, not personality traits.

They represent structural facts about the relationship between agents.

"incentive_modules": {
  "historical": { ... },
  "future": { ... }
}

Each module contains intensity levels (low, medium, high).

⸻

4. Critical Design Principle: Incentives Are Relational

Incentives must NOT be added to an agent’s internal personality.

Instead, they modify:
	1.	The incentivized agent’s perception of the other agent.
	2.	The other agent’s own contextual role description.

Incentives are edge properties, not node properties.

⸻

5. Where Incentives Are Applied

When activating an incentive condition:

For the Incentivized Agent

The incentive text should be appended to:

Their perception of the other agent.

Example:

If Junior is incentivized because Senior writes tenure letters:

Append to Junior’s effective perception of Senior:

“Senior Faculty will write your formal tenure recommendation letter next year.”

This modifies Junior’s perceived persona of Senior.

⸻

For the Other Agent

The corresponding structural fact must appear in their own persona context:

Append to Senior’s effective self-context:

“You will write the Junior Faculty member’s formal tenure recommendation letter next year.”

This ensures:
	•	Common knowledge
	•	No unintended asymmetric information
	•	Clean relational modeling

⸻

6. Rendering Logic

At runtime:

Step 1 — Base Persona

Each agent starts with:

actual_persona
+
perceived_persona_base

Step 2 — Apply Incentive Overlays

For each activated incentive module:
	•	If Agent A is incentivized:
	•	Append relational text to Agent A’s perception of Agent B.
	•	Append corresponding self-context to Agent B.
	•	Do not modify Agent A’s internal traits.

⸻

7. Experimental Philosophy

This structure isolates:
	•	Baseline value preferences (encoded in actual_persona)
	•	Incentive pressure (encoded in relational overlays)
	•	Public/private divergence (measured via survey)

It allows clean modeling of:

Public\ Position \neq Epistemic\ Preference

due to:

Incentive\ Pressure

without introducing hidden information unless explicitly designed.
