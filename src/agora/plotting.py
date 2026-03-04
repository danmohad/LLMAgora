"""Plot helpers used by experiment output generation."""

import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

from agora.agent import Agent
from agora.survey import (
    SURVEY_GROUP_DEFAULT,
    SURVEY_GROUP_DIRECT,
    SURVEY_GROUP_SENTIMENT,
    normalize_survey_questions,
)


def plot_survey_responses(
    responses: dict,
    agents: list[Agent],
    survey_questions: list[str] | list[dict[str, str]] | dict[str, list[str]],
    title: str,
    output_path: Path,
):
    """
    Plots survey responses from a debate.
    """
    agents_dict = {agents[i].id: agents[i].name for i in range(len(agents))}
    agent_ids = list(responses.keys())
    if not agent_ids:
        return

    questions = sorted(
        {
            q
            for agent_data in responses.values()
            for round_data in agent_data.values()
            for q in round_data
        }
    )

    if not questions:
        return

    question_specs = _normalize_plot_question_specs(survey_questions)
    panels = _build_question_panels(question_specs, questions)
    if not panels:
        return

    n = len(panels)
    ncols = min(5, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3 * ncols, 3 * nrows),
        sharex=True,
        sharey=True,
    )

    axes = axes.flatten() if n > 1 else [axes]

    all_rounds = sorted(
        {
            int(turn_num)
            for agent_data in responses.values()
            for turn_num in agent_data.keys()
        }
    )

    for ax, panel in zip(axes, panels):
        for agent_id in agent_ids:
            agent_data = responses[agent_id]

            sorted_rounds = sorted(int(turn_num) for turn_num in agent_data.keys())

            y = []
            for round_num in sorted_rounds:
                round_data = agent_data.get(round_num, agent_data.get(str(round_num), {}))
                y.append(_survey_panel_value(round_data, panel["questions"]))

            ax.plot(
                sorted_rounds,
                y,
                marker="o",
                label=agents_dict[agent_id],  # short id
            )

        ax.axhline(0)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
        if all_rounds:
            ax.set_xticks(all_rounds)
        ax.set_title(_truncate_label(panel["label"]))
        ax.grid(True)

    # Hide unused subplots
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(title)
    fig.supxlabel("Turn Number")
    fig.supylabel("Response score")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(agent_ids))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_survey_distance(
    public_responses: dict,
    private_responses: dict,
    agents: list[Agent],
    survey_questions: list[str] | list[dict[str, str]] | dict[str, list[str]],
    title: str,
    output_path: Path,
    y_limits_base: tuple[float, float] | None = None,
    y_limits_avg: tuple[float, float] | None = None,
):
    """
    Plot per-round distance between public and private survey answers.

    NOTE: This helper is intentionally retained while the integration work is
    still in progress. It is not fully wired into the high-level experiment flow yet.
    """
    agents_dict = {agent.id: agent.name for agent in agents}
    agent_ids = list(public_responses.keys())
    if not agent_ids:
        return

    question_specs = _normalize_plot_question_specs(survey_questions)
    group_by_question = {
        f"Q{index}": spec["group"] for index, spec in enumerate(question_specs, start=1)
    }
    labels_by_question = {
        f"Q{index}": spec["text"] for index, spec in enumerate(question_specs, start=1)
    }
    individual_questions = [
        q_key
        for q_key, group in group_by_question.items()
        if group in {SURVEY_GROUP_DEFAULT, SURVEY_GROUP_DIRECT}
    ]
    sentiment_questions = [
        q_key for q_key, group in group_by_question.items() if group == SURVEY_GROUP_SENTIMENT
    ]
    if not individual_questions and not sentiment_questions:
        return

    if y_limits_base is None:
        y_limits_base = (-4, 4)
    if y_limits_avg is None:
        y_limits_avg = (0, 4)
    individual_distances = {
        agent_id: {
            question: {"rounds": [], "distance": []}
            for question in individual_questions
        }
        for agent_id in agent_ids
    }
    sentiment_distances = {
        agent_id: {"rounds": [], "distance": []} for agent_id in agent_ids
    }

    for agent_id in agent_ids:
        public_agent_data = public_responses.get(agent_id, {})
        private_agent_data = private_responses.get(agent_id, {})

        # Use rounds from public responses as the primary source
        sorted_rounds = sorted(int(turn_num) for turn_num in public_agent_data.keys())

        for my_round in sorted_rounds:
            public_round_data = public_agent_data.get(
                my_round, public_agent_data.get(str(my_round), {})
            )
            private_round_data = private_agent_data.get(
                my_round, private_agent_data.get(str(my_round), {})
            )

            all_questions = set(public_round_data.keys()) | set(
                private_round_data.keys()
            )
            if not all_questions:
                continue

            sum_diff = 0
            count = 0
            for question in all_questions:
                public_score = public_round_data.get(question)
                private_score = private_round_data.get(question)

                if public_score is None or private_score is None:
                    continue
                response_diff = public_score - private_score

                if question in individual_questions:
                    individual_distances[agent_id][question]["rounds"].append(my_round)
                    individual_distances[agent_id][question]["distance"].append(
                        response_diff
                    )
                elif question in sentiment_questions:
                    sum_diff += abs(response_diff)
                    count += 1

            if count > 0:
                sentiment_distances[agent_id]["rounds"].append(my_round)
                sentiment_distances[agent_id]["distance"].append(sum_diff / count)

    n = len(individual_questions) + (1 if sentiment_questions else 0)
    ncols = min(6, n)
    nrows = math.ceil(n / ncols)

    # We apply different y-limits for base vs average panels, so don't share y-axes.
    sharey = False

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3 * ncols, 3 * nrows),
        sharex=True,
        sharey=sharey,
    )

    axes = axes.flatten() if n > 1 else [axes]

    for ax, q_key in zip(axes, individual_questions):
        question_rounds: set[int] = set()
        for agent_id in agent_ids:
            agent_plot_data = individual_distances[agent_id][q_key]
            if agent_plot_data["rounds"]:
                ax.plot(
                    agent_plot_data["rounds"],
                    agent_plot_data["distance"],
                    marker="o",
                    label=agents_dict[agent_id],
                )
                question_rounds.update(agent_plot_data["rounds"])

        ax.axhline(0, color='grey', linewidth=0.8)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
        if question_rounds:
            ax.set_xticks(sorted(question_rounds))
        ax.set_title(_truncate_label(labels_by_question.get(q_key, q_key)))
        ax.grid(True)
        if y_limits_base is not None:
            ax.set_ylim(*y_limits_base)

    if sentiment_questions:
        ax = axes[len(individual_questions)]
        avg_rounds: set[int] = set()
        for agent_id in agent_ids:
            agent_plot_data = sentiment_distances[agent_id]
            if agent_plot_data["rounds"]:
                ax.plot(
                    agent_plot_data["rounds"],
                    agent_plot_data["distance"],
                    marker="o",
                    label=agents_dict[agent_id],
                )
                avg_rounds.update(agent_plot_data["rounds"])
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
        if avg_rounds:
            ax.set_xticks(sorted(avg_rounds))
        ax.set_title("Avg. Sentiment Dist.")
        ax.grid(True)
        if y_limits_avg is not None:
            ax.set_ylim(*y_limits_avg)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(title)
    fig.supxlabel("Turn Number")
    fig.supylabel("Public - Private Response Score")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(agent_ids), bbox_to_anchor=(0.5, 0.95))
    fig.tight_layout(rect=[0, 0.03, 1, 0.9])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _normalize_plot_question_specs(
    survey_questions: list[str] | list[dict[str, str]] | dict[str, list[str]],
) -> list[dict[str, str]]:
    return normalize_survey_questions(
        survey_questions, default_group=SURVEY_GROUP_DEFAULT
    )


def _build_question_panels(
    question_specs: list[dict[str, str]],
    available_questions: Iterable[str],
) -> list[dict[str, Any]]:
    available = set(available_questions)
    panels: list[dict[str, Any]] = []
    sentiment_keys: list[str] = []

    for index, spec in enumerate(question_specs, start=1):
        q_key = f"Q{index}"
        if q_key not in available:
            continue
        if spec["group"] == SURVEY_GROUP_SENTIMENT:
            sentiment_keys.append(q_key)
            continue
        panels.append({"label": spec["text"], "questions": [q_key]})

    if sentiment_keys:
        panels.append({"label": "Avg. Sentiment", "questions": sentiment_keys})

    known_questions = {f"Q{index}" for index in range(1, len(question_specs) + 1)}
    for q_key in sorted(available - known_questions):
        panels.append({"label": q_key, "questions": [q_key]})

    return panels


def _survey_panel_value(round_data: dict[str, Any], questions: list[str]) -> float | None:
    values = [round_data.get(question) for question in questions]
    present_values = [value for value in values if value is not None]
    if not present_values:
        return None
    if len(questions) == 1:
        return present_values[0]
    return sum(present_values) / len(present_values)


def _truncate_label(label: str, max_length: int = 20) -> str:
    return label[:max_length] + "..." if len(label) > max_length else label


def plot_persona_adherence(
    eval_dict: dict[str, Any],
    alpha_persona_name: str,
    beta_persona_name: str,
    save_path: str | None = None,
    show_plot: bool = True,
):
    """Plot persona adherence scores over time with error bars.

    The plot tolerates partial metric selections. If a given series is missing
    or empty, that line is omitted.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Persona Adherence Scores Over Time", fontsize=16)

    alpha_data = eval_dict["alpha"]
    beta_data = eval_dict["beta"]

    alpha_color = "tab:blue"
    beta_color = "tab:orange"

    def _apply_integer_xticks(ax: Any, *series_turns: list[int]) -> None:
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
