"""Plot helpers used by experiment output generation."""

import math
import textwrap
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


def _wrap_label(label: str, width: int = 28, max_lines: int = 2) -> str:
    """Wrap *label* to at most *max_lines* lines of *width* characters each."""
    lines = textwrap.wrap(label, width)
    if len(lines) <= max_lines:
        return "\n".join(lines)
    truncated = lines[max_lines - 1]
    if len(truncated) > width - 3:
        truncated = truncated[: width - 3] + "..."
    return "\n".join(lines[: max_lines - 1] + [truncated])


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
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
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
        ses = list(series.get("scores", {}).get("se", []))
        if not turns:
            return []
        ax.errorbar(
            turns,
            means,
            yerr=ses,
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

    # Third panel: full-debate summary bar chart.
    ax = axes[2]
    bar_width = 0.6
    full_debate_bars = [
        ("alpha", "full_debate_public_score",  "",   f"{alpha_persona_name}\nPublic",  alpha_color),
        ("alpha", "full_debate_private_score", "//",  f"{alpha_persona_name}\nPrivate", alpha_color),
        ("beta",  "full_debate_public_score",  "",   f"{beta_persona_name}\nPublic",   beta_color),
        ("beta",  "full_debate_private_score", "//",  f"{beta_persona_name}\nPrivate",  beta_color),
    ]
    has_full_debate = False
    for i, (role, key, hatch, xlabel, color) in enumerate(full_debate_bars):
        score = eval_dict.get(role, {}).get(key, {}) or {}
        mean_val = score.get("mean")
        se_val = score.get("se", 0.0) or 0.0
        if mean_val is not None:
            ax.bar(
                i, mean_val, bar_width,
                color=color, hatch=hatch, alpha=0.8,
                yerr=se_val, capsize=4,
                error_kw={"elinewidth": 1.5, "ecolor": "black"},
            )
            has_full_debate = True
    ax.set_xticks(range(4))
    ax.set_xticklabels(
        [b[3] for b in full_debate_bars], fontsize=8
    )
    ax.set_ylabel("Score (1-5)")
    ax.set_ylim(0.5, 5.5)
    ax.set_title("Full-Debate Scores")
    ax.grid(axis="y", alpha=0.3)
    if not has_full_debate:
        ax.text(
            0.5, 0.5, "not computed",
            ha="center", va="center", transform=ax.transAxes, color="grey",
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# Group-level plot functions (aggregate across repeats)
# ---------------------------------------------------------------------------

def build_emotion_style(field_results_list: list[dict]) -> dict:
    """Build a consistent colour+marker style map for emotion labels.

    Parameters
    ----------
    field_results_list:
        One or more dicts in the format returned by
        :meth:`~agora.sweep_results.GroupAnalysisResult.run_emotion_analysis`.

    Returns
    -------
    dict
        ``{label: {"color": <rgba>, "marker": <str>}}`` ordered alphabetically.
    """
    import matplotlib.cm as cm

    all_labels = sorted(
        {
            label
            for field_result in field_results_list
            for agent_data in field_result.values()
            for label in agent_data.get("emotions", {}).keys()
        }
    )
    markers = ["o", "s", "^", "D", "v", "P", "*", "X", "h", "8", "<", ">"]
    palette = cm.get_cmap("tab10", max(len(all_labels), 1))
    return {
        label: {"color": palette(i % 10), "marker": markers[i % len(markers)]}
        for i, label in enumerate(all_labels)
    }


def plot_group_semantic_similarity(
    aggregated: dict,
    alpha_name: str = "Alpha",
    beta_name: str = "Beta",
    show: bool = True,
) -> None:
    """Plot semantic similarity metrics aggregated across repeats with error bars.

    Produces up to two figures: one for self-consistency and one for
    cross-agent alignment (public and/or private).

    Parameters
    ----------
    aggregated:
        Output of :meth:`~agora.sweep_results.GroupAnalysisResult.aggregate_semantic`.
    alpha_name, beta_name:
        Display names for the two agents (used in titles).
    show:
        Whether to call ``plt.show()`` after each figure.
    """

    def _xticks(ax: Any, turns: list) -> None:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
        if turns:
            ax.set_xticks(sorted({int(t) for t in turns}))

    sc = aggregated.get("self_consistency")
    if sc:
        fig, ax = plt.subplots(figsize=(9, 4))
        for i, (agent_id, data) in enumerate(sc.items()):
            ax.errorbar(
                data["turns"],
                data["mean"],
                yerr=data["se"],
                marker="os"[i % 2],
                capsize=4,
                linewidth=2,
                alpha=0.85,
                label=agent_id,
            )
        _xticks(ax, [t for d in sc.values() for t in d["turns"]])
        ax.set_title("Self-Consistency  (private vs public — mean ± SE across repeats)")
        ax.set_xlabel("Debate Turn")
        ax.set_ylabel("Cosine Similarity [−1–1]")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(alpha=0.4)
        plt.tight_layout()
        if show:
            plt.show()

    cpa = aggregated.get("cross_agent_public_alignment")
    cpriva = aggregated.get("cross_agent_private_alignment")
    if cpa or cpriva:
        fig, ax = plt.subplots(figsize=(9, 4))
        if cpa:
            ax.errorbar(
                cpa["turns"],
                cpa["mean"],
                yerr=cpa["se"],
                marker="o",
                capsize=4,
                linewidth=2,
                alpha=0.85,
                label="Public alignment",
            )
        if cpriva:
            ax.errorbar(
                cpriva["turns"],
                cpriva["mean"],
                yerr=cpriva["se"],
                marker="s",
                capsize=4,
                linewidth=2,
                alpha=0.85,
                linestyle="--",
                label="Private alignment",
            )
        _xticks(ax, (cpa or cpriva)["turns"])
        ax.set_title("Cross-Agent Semantic Alignment  (mean ± SE across repeats)")
        ax.set_xlabel("Debate Turn")
        ax.set_ylabel("Cosine Similarity [−1–1]")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(alpha=0.4)
        plt.tight_layout()
        if show:
            plt.show()


def plot_group_nli(
    aggregated_nli: dict,
    alpha_name: str = "Alpha",
    beta_name: str = "Beta",
    show: bool = True,
) -> None:
    """Plot NLI class distributions aggregated across repeats with ±1σ shaded bands.

    Produces one figure for self-consistency (one subplot per agent) and, if
    available, one figure for cross-agent public alignment.

    Parameters
    ----------
    aggregated_nli:
        Output of :meth:`~agora.sweep_results.GroupAnalysisResult.run_nli_analysis`.
    alpha_name, beta_name:
        Display names used in the cross-agent alignment title.
    show:
        Whether to call ``plt.show()`` after each figure.
    """
    import numpy as np

    _NLI_COLORS = {
        "contradiction": "#d62728",
        "neutral": "#aec7e8",
        "entailment": "#2ca02c",
    }

    def _color(label: str) -> str:
        lw = label.lower()
        for key, clr in _NLI_COLORS.items():
            if key in lw:
                return clr
        return "#999999"

    def _draw_nli(ax: Any, data: dict, title: str) -> None:
        turns = data["turns"]
        label_names = data["label_names"]
        n_labels = len(label_names)
        n_turns = len(turns)
        x = np.arange(n_turns)
        group_width = 0.75
        bar_width = group_width / max(n_labels, 1)
        offsets = [(i - (n_labels - 1) / 2.0) * bar_width for i in range(n_labels)]
        for i, label_name in enumerate(label_names):
            dist = data["distributions"][label_name]
            means = np.array(dist["mean"])
            ses = np.array(dist["se"])
            clr = _color(label_name)
            ax.bar(
                x + offsets[i], means, bar_width * 0.9,
                color=clr, alpha=0.8, label=label_name,
                yerr=ses, capsize=3, error_kw={"elinewidth": 1.5, "ecolor": "black"},
            )
        ax.set_xticks(x)
        ax.set_xticklabels([f"T{t}" for t in turns])
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Probability")
        ax.set_xlabel("Turn")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    sc = aggregated_nli.get("self_consistency", {})
    if sc:
        agent_ids = list(sc.keys())
        fig, axes = plt.subplots(
            1, len(agent_ids), figsize=(7 * len(agent_ids), 5), sharey=True
        )
        if len(agent_ids) == 1:
            axes = [axes]
        for ax, agent_id in zip(axes, agent_ids):
            _draw_nli(
                ax,
                sc[agent_id],
                f"Self-Consistency NLI\n{agent_id}\n(private → public)",
            )
        fig.suptitle(
            "NLI Self-Consistency: mean ± SE across repeats", fontsize=12
        )
        plt.tight_layout()
        if show:
            plt.show()

    cpa = aggregated_nli.get("cross_agent_public")
    if cpa:
        sc_keys = list(sc.keys()) if sc else []
        a0 = sc_keys[0] if sc_keys else alpha_name
        a1 = sc_keys[1] if len(sc_keys) > 1 else beta_name
        fig, ax = plt.subplots(figsize=(max(6, len(cpa["turns"]) * 1.5), 5))
        _draw_nli(ax, cpa, "")
        fig.suptitle(
            f"NLI Cross-Agent Public Alignment\n{a0} ↔ {a1}  (mean ± SE across repeats)",
            fontsize=12,
        )
        plt.tight_layout()
        if show:
            plt.show()


def plot_group_emotions(
    aggregated_emotions: dict,
    field_label: str,
    alpha_name: str = "Alpha",
    beta_name: str = "Beta",
    emotion_style: dict | None = None,
    show: bool = True,
) -> None:
    """Plot emotion probabilities aggregated across repeats with ±1σ shaded bands.

    Parameters
    ----------
    aggregated_emotions:
        Output of :meth:`~agora.sweep_results.GroupAnalysisResult.run_emotion_analysis`.
    field_label:
        Title suffix, e.g. ``"Public Utterances"`` or ``"Private Reflections"``.
    alpha_name, beta_name:
        Display names for the first and second agent.
    emotion_style:
        Optional style dict from :func:`build_emotion_style`.  Built from
        ``aggregated_emotions`` when *None*.
    show:
        Whether to call ``plt.show()``.
    """
    import numpy as np

    agent_ids = list(aggregated_emotions.keys())
    if not agent_ids:
        print(f"No emotion data for {field_label}.")
        return

    if emotion_style is None:
        emotion_style = build_emotion_style([aggregated_emotions])
    all_labels = sorted(emotion_style.keys())

    agent_display = {agent_ids[0]: alpha_name}
    if len(agent_ids) > 1:
        agent_display[agent_ids[1]] = beta_name

    n_cols = max(len(agent_ids), 2)
    fig, axes = plt.subplots(1, n_cols, figsize=(15, 5), sharey=True)
    fig.suptitle(
        f"Emotion Probabilities Over Turns — {field_label}\n(mean ± SE across repeats)",
        fontsize=13,
    )

    def _draw(ax: Any, agent_id: str) -> None:
        display = agent_display.get(agent_id, agent_id)
        data = aggregated_emotions.get(agent_id, {})
        turns = data.get("turns", [])
        emotions = data.get("emotions", {})
        if not turns or not emotions:
            ax.set_title(f"{display}\n(no data)")
            return
        x = list(range(len(turns)))
        for label in all_labels:
            if label not in emotions:
                continue
            style = emotion_style[label]
            means = np.array(emotions[label]["mean"])
            ses = np.array(emotions[label]["se"])
            ax.plot(
                x, means,
                marker=style["marker"], color=style["color"],
                label=label, linewidth=2, markersize=5,
            )
            ax.fill_between(x, means - ses, means + ses, alpha=0.1, color=style["color"])
        ax.set_xticks(x)
        ax.set_xticklabels([f"T{t}" for t in turns])
        ax.set_xlabel("Debate Turn")
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1.0)
        ax.set_title(display)
        ax.legend(loc="upper right", fontsize=9, ncol=1)
        ax.grid(alpha=0.35)

    for i, ax in enumerate(axes):
        if i < len(agent_ids):
            _draw(ax, agent_ids[i])
        else:
            ax.set_visible(False)

    plt.tight_layout()
    if show:
        plt.show()


def plot_group_survey(
    aggregated_survey: dict,
    alpha_name: str = "Alpha",
    beta_name: str = "Beta",
    survey_questions: list[str] | list[dict[str, str]] | dict[str, list[str]] | None = None,
    show: bool = True,
) -> None:
    """Plot survey responses aggregated across repeats with error bars.

    Produces up to three figures: public responses, private responses, and
    the public − private difference (mirroring :func:`plot_survey_distance`
    but for aggregated data).  Each figure has one subplot per question panel.

    Parameters
    ----------
    aggregated_survey:
        Output of :meth:`~agora.sweep_results.GroupAnalysisResult.aggregate_survey`.
        Structure::

            {
                "public":  {"Alpha": {q_key: {"turns": [...], "mean": [...], "se": [...]}}, ...},
                "private": {"Alpha": {q_key: ...}, ...},
                "diff":    {"Alpha": {q_key: ...}, ...},  # public − private
            }

    alpha_name, beta_name:
        Display names for the two agents (``"Alpha"`` and ``"Beta"`` slots).
    survey_questions:
        Optional question specs for panel labels and group structure.
        Accepts the same format as :func:`plot_survey_responses`.
        When *None*, each question key (Q1, Q2, …) becomes its own panel.
    show:
        Whether to call ``plt.show()`` after each figure.
    """
    _COLORS = {"Alpha": "tab:blue", "Beta": "tab:orange"}
    _MARKERS = {"Alpha": "o", "Beta": "s"}
    _FALLBACK_COLORS = ["tab:green", "tab:red"]
    _FALLBACK_MARKERS = ["^", "D"]

    _SLOT_DISPLAY = {"Alpha": alpha_name, "Beta": beta_name}

    def _display(slot: str, i: int) -> str:
        if slot in _SLOT_DISPLAY:
            return _SLOT_DISPLAY[slot]
        return slot

    def _color(slot: str, i: int) -> str:
        return _COLORS.get(slot, _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)])

    def _marker(slot: str, i: int) -> str:
        return _MARKERS.get(slot, _FALLBACK_MARKERS[i % len(_FALLBACK_MARKERS)])

    def _plot_figure(title: str, by_slot: dict, ylabel: str) -> None:
        if not by_slot:
            return

        all_q_keys = sorted({q for slot_data in by_slot.values() for q in slot_data})
        if not all_q_keys:
            return

        if survey_questions is not None:
            question_specs = _normalize_plot_question_specs(survey_questions)
            panels = _build_question_panels(question_specs, all_q_keys)
        else:
            panels = [{"label": q_key, "questions": [q_key]} for q_key in all_q_keys]

        if not panels:
            return

        n = len(panels)
        ncols = min(5, n)
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(3 * ncols, 3 * nrows),
            sharex=True, sharey=True,
        )
        axes = axes.flatten() if n > 1 else [axes]

        slot_ids = list(by_slot.keys())

        for ax, panel in zip(axes, panels):
            for i, slot in enumerate(slot_ids):
                slot_data = by_slot[slot]
                panel_q_keys = [q for q in panel["questions"] if q in slot_data]
                if not panel_q_keys:
                    continue

                all_turns = sorted({t for q in panel_q_keys for t in slot_data[q]["turns"]})
                means: list[float] = []
                ses: list[float] = []
                for t in all_turns:
                    q_vals = [
                        slot_data[q]["mean"][slot_data[q]["turns"].index(t)]
                        for q in panel_q_keys if t in slot_data[q]["turns"]
                    ]
                    q_ses = [
                        slot_data[q]["se"][slot_data[q]["turns"].index(t)]
                        for q in panel_q_keys if t in slot_data[q]["turns"]
                    ]
                    means.append(sum(q_vals) / len(q_vals) if q_vals else 0.0)
                    n_q = len(q_ses)
                    ses.append(
                        math.sqrt(sum(s**2 for s in q_ses)) / n_q if n_q else 0.0
                    )

                ax.errorbar(
                    all_turns, means, yerr=ses,
                    marker=_marker(slot, i),
                    color=_color(slot, i),
                    label=_display(slot, i),
                    linewidth=2,
                    capsize=4,
                    alpha=0.85,
                )

            ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
            ax.set_title(_wrap_label(panel["label"]), fontsize=8)
            ax.grid(True, alpha=0.3)

        for ax in axes[n:]:
            ax.set_visible(False)

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=len(slot_ids))
        fig.suptitle(title)
        fig.supxlabel("Turn Number")
        fig.supylabel(ylabel)
        plt.tight_layout()
        if show:
            plt.show()

    _plot_figure(
        "Aggregated Public Survey Responses  (mean \u00b1 SE across repeats)",
        aggregated_survey.get("public", {}),
        "Response score (mean \u00b1 SE)",
    )
    _plot_figure(
        "Aggregated Private Survey Responses  (mean \u00b1 SE across repeats)",
        aggregated_survey.get("private", {}),
        "Response score (mean \u00b1 SE)",
    )
    _plot_figure(
        "Aggregated Public \u2212 Private Survey Distance  (mean \u00b1 SE across repeats)",
        aggregated_survey.get("diff", {}),
        "Public \u2212 Private (mean \u00b1 SE)",
    )
