"""Plot helpers used by experiment output generation."""

import math
from pathlib import Path

import matplotlib.pyplot as plt

from agora.agent import Agent


def plot_survey_responses(
    responses: dict,
    agents: list[Agent],
    survey_questions: list[str],
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

    n = len(questions)
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

    for ax, q in zip(axes, questions):
        for agent_id in agent_ids:
            agent_data = responses[agent_id]

            sorted_rounds = sorted(int(turn_num) for turn_num in agent_data.keys())

            y = [
                agent_data.get(r, agent_data.get(str(r), {})).get(q, None)
                for r in sorted_rounds
            ]

            ax.plot(
                sorted_rounds,
                y,
                marker="o",
                label=agents_dict[agent_id],  # short id
            )

        ax.axhline(0)
        max_length = 20
        my_str = survey_questions[int(q[1:]) - 1]
        truncated_str = my_str[:max_length] + "..." if len(my_str) > 20 else my_str
        ax.set_title(truncated_str)
        ax.grid(True)

    # Hide unused subplots
    for ax in axes[len(questions) :]:
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
    survey_questions: list[str],
    title: str,
    output_path: Path,
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

    num_base_questions = 5
    base_questions_distances = {
        agent_id: {
            question + 1: {"rounds": [], "distance": []}
            for question in range(num_base_questions)
        }
        for agent_id in agent_ids
    }
    distances = {agent_id: {"rounds": [], "distance": []} for agent_id in agent_ids}

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
                q_num = int(question.removeprefix("Q"))

                public_score = public_round_data.get(question)
                private_score = private_round_data.get(question)
                
                if public_score is None or private_score is None:
                    continue
                response_diff = public_score - private_score
                
                if q_num <= num_base_questions:
                    base_questions_distances[agent_id][q_num]["rounds"].append(my_round)
                    base_questions_distances[agent_id][q_num]["distance"].append(response_diff)
                else:
                    sum_diff += response_diff
                    count += 1

            if count > 0:
                distances[agent_id]["rounds"].append(my_round)
                distances[agent_id]["distance"].append(sum_diff/count)

    n = num_base_questions + 1
    ncols = min(6, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3 * ncols, 3 * nrows),
        sharex=True,
        sharey=True,
    )

    axes = axes.flatten() if n > 1 else [axes]

    for ax, q_num in zip(axes, range(num_base_questions)):
        for agent_id in agent_ids:
            agent_plot_data = base_questions_distances[agent_id][q_num + 1]
            if agent_plot_data["rounds"]:
                ax.plot(
                    agent_plot_data["rounds"],
                    agent_plot_data["distance"],
                    marker="o",
                    label=agents_dict[agent_id],
                )

        ax.axhline(0, color='grey', linewidth=0.8)
        max_length = 20
        my_str = survey_questions[q_num]
        truncated_str = my_str[:max_length] + "..." if len(my_str) > max_length else my_str
        ax.set_title(truncated_str)
        ax.grid(True)
    
    # Plot average distance for other questions
    ax = axes[num_base_questions]
    for agent_id in agent_ids:
        agent_plot_data = distances[agent_id]
        if agent_plot_data["rounds"]:
            ax.plot(
                agent_plot_data["rounds"],
                agent_plot_data["distance"],
                marker="o",
                label=agents_dict[agent_id],
            )
    ax.axhline(0, color='grey', linewidth=0.8)
    ax.set_title("Avg. Other Qs Dist.")
    ax.grid(True)


    # Hide unused subplots
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


