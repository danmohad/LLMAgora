"""Persona adherence evaluation over debate transcripts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from .debate_history import get_structured_debate_history

PERSONA_METRIC_PUBLIC_PER_TURN = "public_per_turn"
PERSONA_METRIC_PRIVATE_PER_TURN = "private_per_turn"
PERSONA_METRIC_PUBLIC_CUMULATIVE = "public_cumulative"
PERSONA_METRIC_PRIVATE_CUMULATIVE = "private_cumulative"
PERSONA_METRIC_FULL_DEBATE_PUBLIC = "full_debate_public"
PERSONA_METRIC_FULL_DEBATE_PRIVATE = "full_debate_private"

PERSONA_ANALYSIS_METRICS: tuple[str, ...] = (
    PERSONA_METRIC_PUBLIC_PER_TURN,
    PERSONA_METRIC_PRIVATE_PER_TURN,
    PERSONA_METRIC_PUBLIC_CUMULATIVE,
    PERSONA_METRIC_PRIVATE_CUMULATIVE,
    PERSONA_METRIC_FULL_DEBATE_PUBLIC,
    PERSONA_METRIC_FULL_DEBATE_PRIVATE,
)
_DEFAULT_PROMPTS_PATH = Path(__file__).resolve().parents[2] / "data" / "prompts.json"


@lru_cache(maxsize=1)
def _default_persona_scoring_prompt_template() -> str:
    catalog = json.loads(_DEFAULT_PROMPTS_PATH.read_text(encoding="utf-8"))
    prompt_sets = catalog.get("prompt_sets", catalog)
    payload = prompt_sets.get("default")
    if not isinstance(payload, dict):
        raise KeyError("Default prompt set missing from data/prompts.json")
    return str(payload["persona_scoring_prompt"])


def _score_summary(scores_raw: list[int]) -> tuple[float, float]:
    """Return ``(mean, std)`` summary for one or more raw 1-5 samples."""
    return float(np.mean(scores_raw)), float(np.std(scores_raw))


def _normalize_persona_metrics(metrics: Sequence[str] | None) -> set[str]:
    """Validate selected persona metrics; ``None`` means compute all metrics."""
    if metrics is None:
        return set(PERSONA_ANALYSIS_METRICS)
    normalized = {str(metric) for metric in metrics}
    unknown = normalized - set(PERSONA_ANALYSIS_METRICS)
    if unknown:
        raise ValueError(
            "Unknown persona analysis metrics: "
            + ", ".join(sorted(unknown))
            + ". Allowed values: "
            + ", ".join(PERSONA_ANALYSIS_METRICS)
        )
    return normalized


@dataclass
class PersonaScore:
    """Raw and summary score values for a single turn-indexed metric datapoint."""

    turn_num: int
    scores_raw: list[int]

    @property
    def score_mean(self) -> float:
        return _score_summary(self.scores_raw)[0]

    @property
    def score_std(self) -> float:
        return _score_summary(self.scores_raw)[1]


@dataclass
class AgentPersonaEvaluation:
    """Persona adherence outputs for one agent across all selected metrics."""

    persona_id: str
    computed_metrics: list[str] = field(default_factory=list)
    public_per_turn_scores: list[PersonaScore] = field(default_factory=list)
    private_per_turn_scores: list[PersonaScore] = field(default_factory=list)
    public_cumulative_scores: list[PersonaScore] = field(default_factory=list)
    private_cumulative_scores: list[PersonaScore] = field(default_factory=list)
    full_debate_public_score: Optional[tuple[float, float]] = None
    full_debate_private_score: Optional[tuple[float, float]] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-friendly dictionary."""

        def scores_to_dict(scores: list[PersonaScore]) -> dict[str, Any]:
            return {
                "turns": [s.turn_num for s in scores],
                "scores": {
                    "mean": [s.score_mean for s in scores],
                    "std": [s.score_std for s in scores],
                    "raw": [s.scores_raw for s in scores],
                },
            }

        return {
            "persona_id": self.persona_id,
            "computed_metrics": list(self.computed_metrics),
            "public_per_turn_scores": scores_to_dict(self.public_per_turn_scores),
            "private_per_turn_scores": scores_to_dict(self.private_per_turn_scores),
            "public_cumulative_scores": scores_to_dict(self.public_cumulative_scores),
            "private_cumulative_scores": scores_to_dict(self.private_cumulative_scores),
            "full_debate_public_score": {
                "mean": (
                    self.full_debate_public_score[0]
                    if self.full_debate_public_score
                    else None
                ),
                "std": (
                    self.full_debate_public_score[1]
                    if self.full_debate_public_score
                    else None
                ),
            },
            "full_debate_private_score": {
                "mean": (
                    self.full_debate_private_score[0]
                    if self.full_debate_private_score
                    else None
                ),
                "std": (
                    self.full_debate_private_score[1]
                    if self.full_debate_private_score
                    else None
                ),
            },
        }


@dataclass
class DebatePersonaEvaluation:
    """Persona adherence outputs for both agents in one run."""

    alpha: AgentPersonaEvaluation
    beta: AgentPersonaEvaluation

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha.to_dict(),
            "beta": self.beta.to_dict(),
        }


class PersonaEvaluator:
    """Score how closely an agent's outputs match its assigned persona."""

    def __init__(
        self,
        llm_client: Any,
        personas: dict[str, Any],
        model: str = "anthropic/claude-sonnet-4",
        scoring_prompt_template: str | None = None,
    ):
        self.llm_client = llm_client
        self.personas = personas["personas"] if "personas" in personas else personas
        self.model = model
        self.scoring_prompt_template = (
            scoring_prompt_template or _default_persona_scoring_prompt_template()
        )

    def _build_persona_scoring_prompt(
        self,
        text: str,
        persona_id: str,
        slice_label: str = "turn",
    ) -> str:
        """Create a single prompt that asks for one 1-5 adherence score."""
        if persona_id not in self.personas:
            raise ValueError(f"Unknown persona id: {persona_id}")

        persona = self.personas[persona_id]
        actual_persona = persona.get("actual_persona", "")
        return self.scoring_prompt_template.format(
            actual_persona=actual_persona,
            slice_label=slice_label,
            text=text,
        )

    def _sample_persona_scores(
        self,
        text: str,
        persona_id: str,
        slice_label: str = "turn",
        n_samples: int = 1,
    ) -> list[int]:
        """Sample one or more persona adherence ratings for one text slice."""
        if not text or not text.strip():
            return [3] * n_samples

        prompt = self._build_persona_scoring_prompt(text, persona_id, slice_label)
        scores: list[int] = []
        for sample_idx in range(n_samples):
            try:
                response = self.llm_client.complete(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model,
                )
                response_text = response.strip()
                numbers = re.findall(r"\b[1-5]\b", response_text)
                if numbers:
                    scores.append(int(numbers[0]))
                else:
                    if sample_idx == 0:
                        print(
                            "Warning: Could not parse score from response: "
                            f"{response_text}"
                        )
                    scores.append(3)
            except Exception as exc:  # pragma: no cover - behavior tested via monkeypatch
                if sample_idx == 0:
                    print(f"Error scoring text: {exc}")
                scores.append(3)
        return scores

    def evaluate_debate_from_history(
        self,
        memory_turns: dict[str, Any],
        alpha_persona_id: str,
        beta_persona_id: str,
        verbose: bool = False,
        n_samples: int = 1,
        metrics: Sequence[str] | None = None,
    ) -> DebatePersonaEvaluation:
        """Evaluate selected persona metrics from structured or normalized debate data."""
        selected_metrics = _normalize_persona_metrics(metrics)
        if isinstance(memory_turns, dict) and "turns" in memory_turns:
            debate_data = get_structured_debate_history(memory_turns)
        elif isinstance(memory_turns, dict):
            debate_data = memory_turns
        else:
            raise ValueError(
                "Persona evaluation requires canonical structured history "
                "or normalized debate data"
            )
        agent_names = list(debate_data.keys())
        if len(agent_names) != 2:
            raise ValueError(
                f"Expected exactly 2 agents in debate data, got {len(agent_names)}"
            )
        alpha_name, beta_name = agent_names[0], agent_names[1]

        if verbose:
            print("Persona metrics: " + ", ".join(sorted(selected_metrics)))
            print(f"Evaluating {alpha_name} against persona: {alpha_persona_id}")
        alpha_eval = self._evaluate_single_agent(
            debate_data[alpha_name],
            alpha_persona_id,
            selected_metrics=selected_metrics,
            verbose=verbose,
            n_samples=n_samples,
        )

        if verbose:
            print(f"\nEvaluating {beta_name} against persona: {beta_persona_id}")
        beta_eval = self._evaluate_single_agent(
            debate_data[beta_name],
            beta_persona_id,
            selected_metrics=selected_metrics,
            verbose=verbose,
            n_samples=n_samples,
        )
        return DebatePersonaEvaluation(alpha=alpha_eval, beta=beta_eval)

    def _evaluate_single_agent(
        self,
        agent_data: dict[str, Any],
        persona_id: str,
        selected_metrics: set[str],
        verbose: bool = False,
        n_samples: int = 1,
    ) -> AgentPersonaEvaluation:
        """Evaluate one agent for a configurable subset of persona metrics."""
        evaluation = AgentPersonaEvaluation(
            persona_id=persona_id,
            computed_metrics=sorted(selected_metrics),
        )
        debate_turns = agent_data.get("debate_turns", [])
        if not debate_turns:
            return evaluation

        public_accumulator: list[str] = []
        private_accumulator: list[str] = []
        needs_public_accumulator = bool(
            {PERSONA_METRIC_PUBLIC_CUMULATIVE, PERSONA_METRIC_FULL_DEBATE_PUBLIC}
            & selected_metrics
        )
        needs_private_accumulator = bool(
            {PERSONA_METRIC_PRIVATE_CUMULATIVE, PERSONA_METRIC_FULL_DEBATE_PRIVATE}
            & selected_metrics
        )

        for turn_idx, turn_data in enumerate(debate_turns):
            turn_num = int(turn_data.get("turn_num", turn_idx + 1))
            public_text = turn_data.get("public_speech", "")
            private_text = turn_data.get("private_reflection", "")

            if PERSONA_METRIC_PUBLIC_PER_TURN in selected_metrics:
                if verbose:
                    print(f"  Turn {turn_num}: scoring public turn adherence...")
                public_scores = self._sample_persona_scores(
                    public_text,
                    persona_id,
                    slice_label=f"public turn {turn_num}",
                    n_samples=n_samples,
                )
                evaluation.public_per_turn_scores.append(
                    PersonaScore(turn_num=turn_num, scores_raw=public_scores)
                )

            if PERSONA_METRIC_PRIVATE_PER_TURN in selected_metrics:
                if verbose:
                    print(f"  Turn {turn_num}: scoring private turn adherence...")
                private_scores = self._sample_persona_scores(
                    private_text,
                    persona_id,
                    slice_label=f"private turn {turn_num}",
                    n_samples=n_samples,
                )
                evaluation.private_per_turn_scores.append(
                    PersonaScore(turn_num=turn_num, scores_raw=private_scores)
                )

            if needs_public_accumulator:
                public_accumulator.append(public_text)
            if needs_private_accumulator:
                private_accumulator.append(private_text)

            if PERSONA_METRIC_PUBLIC_CUMULATIVE in selected_metrics:
                if verbose:
                    print(
                        f"  Turn {turn_num}: scoring cumulative public turns 1-{turn_num}..."
                    )
                running_public_scores = self._sample_persona_scores(
                    "\n\n---\n\n".join(public_accumulator),
                    persona_id,
                    slice_label=f"cumulative public turns 1-{turn_num}",
                    n_samples=n_samples,
                )
                evaluation.public_cumulative_scores.append(
                    PersonaScore(turn_num=turn_num, scores_raw=running_public_scores)
                )

            if PERSONA_METRIC_PRIVATE_CUMULATIVE in selected_metrics:
                if verbose:
                    print(
                        f"  Turn {turn_num}: scoring cumulative private turns 1-{turn_num}..."
                    )
                running_private_scores = self._sample_persona_scores(
                    "\n\n---\n\n".join(private_accumulator),
                    persona_id,
                    slice_label=f"cumulative private turns 1-{turn_num}",
                    n_samples=n_samples,
                )
                evaluation.private_cumulative_scores.append(
                    PersonaScore(turn_num=turn_num, scores_raw=running_private_scores)
                )

        if PERSONA_METRIC_FULL_DEBATE_PUBLIC in selected_metrics:
            if evaluation.public_cumulative_scores:
                last_public = evaluation.public_cumulative_scores[-1]
                evaluation.full_debate_public_score = (
                    last_public.score_mean,
                    last_public.score_std,
                )
            else:
                full_public_scores = self._sample_persona_scores(
                    "\n\n---\n\n".join(public_accumulator),
                    persona_id,
                    slice_label="full debate public",
                    n_samples=n_samples,
                )
                evaluation.full_debate_public_score = _score_summary(full_public_scores)

        if PERSONA_METRIC_FULL_DEBATE_PRIVATE in selected_metrics:
            if evaluation.private_cumulative_scores:
                last_private = evaluation.private_cumulative_scores[-1]
                evaluation.full_debate_private_score = (
                    last_private.score_mean,
                    last_private.score_std,
                )
            else:
                full_private_scores = self._sample_persona_scores(
                    "\n\n---\n\n".join(private_accumulator),
                    persona_id,
                    slice_label="full debate private",
                    n_samples=n_samples,
                )
                evaluation.full_debate_private_score = _score_summary(
                    full_private_scores
                )
        return evaluation


__all__ = [
    "PersonaEvaluator",
    "PersonaScore",
    "AgentPersonaEvaluation",
    "DebatePersonaEvaluation",
    "PERSONA_ANALYSIS_METRICS",
    "PERSONA_METRIC_PUBLIC_PER_TURN",
    "PERSONA_METRIC_PRIVATE_PER_TURN",
    "PERSONA_METRIC_PUBLIC_CUMULATIVE",
    "PERSONA_METRIC_PRIVATE_CUMULATIVE",
    "PERSONA_METRIC_FULL_DEBATE_PUBLIC",
    "PERSONA_METRIC_FULL_DEBATE_PRIVATE",
]
