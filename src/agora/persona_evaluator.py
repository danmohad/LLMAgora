"""Persona adherence evaluation over debate transcripts."""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np

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

    def to_dict(self) -> Dict:
        """Convert to a JSON-friendly dictionary."""

        def scores_to_dict(scores: list[PersonaScore]) -> Dict:
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

    def to_dict(self) -> Dict:
        return {
            "alpha": self.alpha.to_dict(),
            "beta": self.beta.to_dict(),
        }


class PersonaEvaluator:
    """Score how closely an agent's outputs match its assigned persona."""

    def __init__(
        self,
        llm_client: Any,
        personas: Dict,
        model: str = "anthropic/claude-sonnet-4",
    ):
        self.llm_client = llm_client
        self.personas = personas["personas"] if "personas" in personas else personas
        self.model = model

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
        return f"""You are evaluating how well a debate participant adheres to their assigned persona.

ASSIGNED PERSONA:
{actual_persona}

TEXT TO EVALUATE ({slice_label}):
{text}

On a scale of 1-5, how well does this text match the assigned persona?

1 = Not at all - completely inconsistent with the persona
2 = Poorly - mostly inconsistent, with only minor alignment
3 = Moderately - some consistency, but significant deviations
4 = Well - mostly consistent, with minor deviations
5 = Perfectly - completely consistent and authentic to the persona

Consider:
- Does the language and tone match what this persona would use?
- Are the arguments and perspectives consistent with this persona's background and interests?
- Does the content reflect the constraints, obligations, and stakes mentioned in the persona?
- Is the level of detail and specificity appropriate for this persona?

Respond with ONLY a single number from 1 to 5, nothing else."""

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
        memory_turns: list,
        alpha_persona_id: str,
        beta_persona_id: str,
        verbose: bool = False,
        n_samples: int = 1,
        metrics: Sequence[str] | None = None,
    ) -> DebatePersonaEvaluation:
        """Evaluate selected persona metrics for both agents from debate history."""
        selected_metrics = _normalize_persona_metrics(metrics)
        debate_data = get_structured_debate_history(memory_turns)
        agent_names = list(debate_data.keys())
        if len(agent_names) != 2:
            raise ValueError(
                f"Expected exactly 2 agents in debate data, got {len(agent_names)}"
            )
        alpha_name, beta_name = agent_names[0], agent_names[1]

        if verbose:
            print(
                "Persona metrics: "
                + ", ".join(sorted(selected_metrics))
            )
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
        agent_data: Dict,
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
            {
                PERSONA_METRIC_PUBLIC_CUMULATIVE,
                PERSONA_METRIC_FULL_DEBATE_PUBLIC,
            }
            & selected_metrics
        )
        needs_private_accumulator = bool(
            {
                PERSONA_METRIC_PRIVATE_CUMULATIVE,
                PERSONA_METRIC_FULL_DEBATE_PRIVATE,
            }
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
                evaluation.full_debate_private_score = _score_summary(full_private_scores)
        return evaluation

def _get_or_create_turn(agent_turns: list[dict], turn_num: int) -> dict:
    for turn in agent_turns:
        if int(turn.get("turn_num", 0)) == int(turn_num):
            return turn
    entry = {
        "turn_num": int(turn_num),
        "public_speech": "",
        "private_reflection": "",
        "public_stance": "",
    }
    agent_turns.append(entry)
    agent_turns.sort(key=lambda item: int(item.get("turn_num", 0)))
    return entry


def get_structured_debate_history(memory_turns: Any) -> Dict:
    """
    Convert raw memory turns into structured debate data.
    
    This function organizes the debate history by agent and separates
    public speeches from private reflections.
    
    Args:
        memory_turns: List of MemoryTurn objects from Agora.history()
    
    Returns:
        Dictionary mapping agent names to their debate data:
        {
            'agent_name': {
                'debate_turns': [
                    {
                        'turn_num': int,
                        'public_speech': str,
                        'private_reflection': str,
                        'public_stance': str,  # extracted metadata if available
                    },
                    ...
                ],
                'pre_interview': str or None,
                'post_interview': str or None,
            },
            ...
        }
    """
    # Preferred path: canonical structured history payload from Agora.
    if isinstance(memory_turns, dict) and "turns" in memory_turns:
        agent_data: Dict[str, Dict[str, Any]] = {
            "Alpha": {"debate_turns": [], "pre_interview": None, "post_interview": None},
            "Beta": {"debate_turns": [], "pre_interview": None, "post_interview": None},
        }

        pre_interviews = memory_turns.get("pre_interviews", {})
        post_interviews = memory_turns.get("post_interviews", {})
        for slot in ("Alpha", "Beta"):
            pre_stage = pre_interviews.get(slot, {})
            post_stage = post_interviews.get(slot, {})
            agent_data[slot]["pre_interview"] = pre_stage.get("response")
            agent_data[slot]["post_interview"] = post_stage.get("response")

        for turn in memory_turns.get("turns", []):
            turn_num = int(turn.get("turn_num", 0))
            for slot in ("Alpha", "Beta"):
                subturn = turn.get(slot, {})
                agent_data[slot]["debate_turns"].append(
                    {
                        "turn_num": turn_num,
                        "public_speech": subturn.get("public_utterance") or "",
                        "private_reflection": subturn.get("private_utterance") or "",
                        "public_stance": "",
                    }
                )
        return agent_data

    # Fallback path: legacy list of MemoryTurn-like records.
    agent_data: Dict[str, Dict[str, Any]] = {}
    agent_turn_nums: Dict[str, int] = {}

    for turn in memory_turns:
        speaker_name = turn.metadata.get("speaker_name", turn.speaker_id)

        if speaker_name not in agent_data:
            agent_data[speaker_name] = {
                "debate_turns": [],
                "pre_interview": None,
                "post_interview": None,
            }
            agent_turn_nums[speaker_name] = 0

        if turn.role == "pre_interview":
            agent_data[speaker_name]["pre_interview"] = turn.private_reflection
            continue

        if turn.role == "post_interview":
            agent_data[speaker_name]["post_interview"] = turn.private_reflection
            continue

        # Prefer explicit macro turn metadata when available.
        metadata_turn_num = turn.metadata.get("turn_num")
        event_type = turn.metadata.get("event_type")
        if metadata_turn_num is not None and event_type in {
            "public_utterance",
            "private_utterance",
        }:
            turn_data = _get_or_create_turn(
                agent_data[speaker_name]["debate_turns"], int(metadata_turn_num)
            )
            if event_type == "public_utterance":
                turn_data["public_speech"] = turn.public_speech or ""
            elif event_type == "private_utterance":
                turn_data["private_reflection"] = turn.private_reflection or ""
            continue

        # Legacy ordering fallback.
        if turn.role == "assistant":
            agent_turn_nums[speaker_name] += 1
            turn_num = agent_turn_nums[speaker_name]
            agent_data[speaker_name]["debate_turns"].append(
                {
                    "turn_num": turn_num,
                    "public_speech": turn.public_speech or "",
                    "private_reflection": "",
                    "public_stance": "",
                }
            )
        elif turn.role == "reflection" and agent_data[speaker_name]["debate_turns"]:
            agent_data[speaker_name]["debate_turns"][-1]["private_reflection"] = (
                turn.private_reflection or ""
            )

    return agent_data


def plot_persona_adherence(
    eval_dict: Dict,
    alpha_persona_name: str,
    beta_persona_name: str,
    save_path: str = None,
    show_plot: bool = True,
):
    """
    Plot persona adherence scores over time with error bars.

    The plot tolerates partial metric selections. If a given series is missing or
    empty, that line is omitted.
    
    Args:
        eval_dict: Dictionary from DebatePersonaEvaluation.to_dict()
        alpha_persona_name: Display name for alpha persona
        beta_persona_name: Display name for beta persona
        save_path: If provided, save plot to this path
        show_plot: If True, display the plot
    
    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Persona Adherence Scores Over Time', fontsize=16)
    
    # Extract data
    alpha_data = eval_dict['alpha']
    beta_data = eval_dict['beta']
    
    # Define colors for each agent
    alpha_color = 'tab:blue'
    beta_color = 'tab:orange'
    
    def _apply_integer_xticks(ax, *series_turns: list[int]) -> None:
        all_turns = sorted({int(turn) for turns in series_turns for turn in turns})
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
        if all_turns:
            ax.set_xticks(all_turns)

    def _plot_series_if_present(
        ax: Any,
        series: dict[str, Any],
        *,
        marker: str,
        label: str,
        color: str,
        linestyle: str,
    ) -> list[int]:
        turns = list(series.get("turns", []))
        means = list(series.get("scores", {}).get("mean", []))
        stds = list(series.get("scores", {}).get("std", []))
        if not turns:
            return []
        ax.errorbar(
            turns,
            means,
            yerr=stds,
            marker=marker,
            label=label,
            linewidth=2,
            capsize=5,
            alpha=0.8,
            color=color,
            linestyle=linestyle,
        )
        return turns

    # Left panel: per-turn scores.
    ax = axes[0]
    alpha_pub_ind = alpha_data.get("public_per_turn_scores", {})
    alpha_priv_ind = alpha_data.get("private_per_turn_scores", {})
    beta_pub_ind = beta_data.get("public_per_turn_scores", {})
    beta_priv_ind = beta_data.get("private_per_turn_scores", {})

    _plot_series_if_present(
        ax,
        alpha_pub_ind,
        marker="o",
        label=f"{alpha_persona_name} - Public",
        color=alpha_color,
        linestyle="-",
    )
    _plot_series_if_present(
        ax,
        alpha_priv_ind,
        marker="o",
        label=f"{alpha_persona_name} - Private",
        color=alpha_color,
        linestyle="--",
    )
    _plot_series_if_present(
        ax,
        beta_pub_ind,
        marker="s",
        label=f"{beta_persona_name} - Public",
        color=beta_color,
        linestyle="-",
    )
    _plot_series_if_present(
        ax,
        beta_priv_ind,
        marker="s",
        label=f"{beta_persona_name} - Private",
        color=beta_color,
        linestyle="--",
    )

    _apply_integer_xticks(
        ax,
        alpha_pub_ind.get("turns", []),
        alpha_priv_ind.get("turns", []),
        beta_pub_ind.get("turns", []),
        beta_priv_ind.get("turns", []),
    )
    ax.set_title("Individual Turn Scores")
    ax.set_xlabel("Turn Number")
    ax.set_ylabel("Score (1-5)")
    ax.set_ylim(0.5, 5.5)
    if ax.has_data():
        ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right panel: cumulative scores.
    ax = axes[1]
    alpha_pub_cum = alpha_data.get("public_cumulative_scores", {})
    alpha_priv_cum = alpha_data.get("private_cumulative_scores", {})
    beta_pub_cum = beta_data.get("public_cumulative_scores", {})
    beta_priv_cum = beta_data.get("private_cumulative_scores", {})

    _plot_series_if_present(
        ax,
        alpha_pub_cum,
        marker="o",
        label=f"{alpha_persona_name} - Public",
        color=alpha_color,
        linestyle="-",
    )
    _plot_series_if_present(
        ax,
        alpha_priv_cum,
        marker="o",
        label=f"{alpha_persona_name} - Private",
        color=alpha_color,
        linestyle="--",
    )
    _plot_series_if_present(
        ax,
        beta_pub_cum,
        marker="s",
        label=f"{beta_persona_name} - Public",
        color=beta_color,
        linestyle="-",
    )
    _plot_series_if_present(
        ax,
        beta_priv_cum,
        marker="s",
        label=f"{beta_persona_name} - Private",
        color=beta_color,
        linestyle="--",
    )

    _apply_integer_xticks(
        ax,
        alpha_pub_cum.get("turns", []),
        alpha_priv_cum.get("turns", []),
        beta_pub_cum.get("turns", []),
        beta_priv_cum.get("turns", []),
    )
    ax.set_title("Cumulative Scores")
    ax.set_xlabel("Turn Number")
    ax.set_ylabel("Score (1-5)")
    ax.set_ylim(0.5, 5.5)
    if ax.has_data():
        ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


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
    "get_structured_debate_history",
    "plot_persona_adherence",
]
