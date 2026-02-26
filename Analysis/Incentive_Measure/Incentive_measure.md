Measuring Incentive Strength via Expected Utility

Conceptual Motivation

In this study, incentives are not treated as narrative labels (e.g., “low” or “high” pressure), but as expected instrumental utility differentials associated with a public stance.

Formally, an incentive exists when the expected utility of one public action differs from that of another due to anticipated consequences.

Let:
	•	a_T: public action aligned with private belief (truthful stance)
	•	a_D: public action misaligned with private belief (distorted stance)

We define incentive strength as the expected utility differential:

\Delta U = U(a_D) - U(a_T)

In the absence of instrumental pressure (vanilla condition), we expect:

\Delta U \approx 0

Under positive or negative incentive conditions, expected utility shifts.

⸻

Economic Formulation

We approximate the instrumental component of utility as:

\Delta U \approx p \times M

Where:
	•	p = perceived probability that meaningful consequences will occur
	•	M = perceived magnitude (severity) of those consequences

This corresponds to the standard expected value formulation in decision theory:

\mathbb{E}[\text{consequence}] = \Pr(\text{consequence}) \times \text{magnitude}

Importantly, this framework does not assume that agents explicitly compute expected utilities. Rather, it tests whether their behavior tracks perceived instrumental stakes.

⸻

Operationalization via the p/M Block

To measure perceived incentive strength, we include a private-only p/M block immediately following the interaction.

Participants (agents) respond numerically on a 0–100 scale to four items:

Probability Component (p)
	1.	How likely are personal or professional consequences from your PUBLIC stance?
	2.	How likely is it that the other participant could influence outcomes that matter to you?

These items estimate perceived likelihood of consequential outcomes.

We compute:

p = \frac{P_1 + P_2}{2}

(normalized to [0,1]).

⸻

Magnitude Component (M)
	3.	If consequences occurred, how severe would they be for you?
	4.	How important is maintaining a positive working relationship with the other participant?

These items estimate perceived impact or cost.

We compute:

M = \frac{M_1 + M_2}{2}

(normalized to [0,1]).

⸻

Incentive Strength

Incentive strength is then computed as:

\Delta U = p \cdot M

This produces a continuous measure in [0,1], representing expected instrumental stakes associated with the public stance.

⸻

Why Measurement Is Private

The p/M block is administered privately for two reasons:
	1.	Public reporting of perceived stakes may itself be strategic.
	2.	We aim to measure perceived instrumental pressure, not willingness to admit pressure publicly.

This separation preserves interpretability of \Delta U as perceived incentive strength.

⸻

Relationship to Public–Private Divergence

We define behavioral divergence as:

D = \mathbb{1}[a_{\text{public}} \neq a_{\text{private}}]

or, at the survey level, as the difference between identical public and private responses.

Our core empirical test evaluates whether:

\Pr(D = 1) = f(\Delta U)

If divergence increases monotonically with \Delta U, this supports the hypothesis that instrumental incentives systematically distort public expression relative to private belief.

⸻

Positive vs Negative Incentives

Both positive and negative incentive modules alter p and/or M:
	•	Positive incentives increase expected benefit (or reduce cost) of alignment.
	•	Negative incentives increase expected cost of making the “wrong” decision.

Both are captured within the same p \times M framework; what differs is the direction of behavioral shift.

The model therefore provides a unified, quantitative representation of incentive pressure across conditions.

⸻

Advantages of This Approach
	1.	Avoids arbitrary “low/medium/high” labeling
	2.	Provides continuous, measurable incentive strength
	3.	Enables regression-based modeling
	4.	Separates manipulation from measurement
	5.	Maintains theoretical grounding in decision theory

⸻

Summary

Incentives in this framework are defined as expected instrumental utility shifts, operationalized as:

\Delta U = p \cdot M

This allows us to test whether public–private divergence scales with perceived stakes, rather than relying solely on categorical experimental conditions.

