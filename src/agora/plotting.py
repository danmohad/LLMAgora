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

            sorted_rounds = sorted(agent_data.keys())
            round_index = list(range(len(sorted_rounds)))

            y = [agent_data[r].get(q, None) for r in sorted_rounds]

            ax.plot(
                round_index,
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
    fig.supxlabel("Round index (order only)")
    fig.supylabel("Response score")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(agent_ids))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_survey_distance(
    public_responses: dict,
    private_responses: dict,
    agents: list[Agent],
    title: str,
    output_path: Path,
):
    """
    Plots the distance between public and private survey responses.
    """
    agents_dict = {agent.id: agent.name for agent in agents}
    agent_ids = list(public_responses.keys())
    if not agent_ids:
        return

    distances = {agent_id: {"rounds": [], "distance": []} for agent_id in agent_ids}

    for agent_id in agent_ids:
        public_agent_data = public_responses.get(agent_id, {})
        private_agent_data = private_responses.get(agent_id, {})
        
        # Use rounds from public responses as the primary source
        sorted_rounds = sorted(public_agent_data.keys())
        
        for r in sorted_rounds:
            public_round_data = public_agent_data.get(r, {})
            private_round_data = private_agent_data.get(r, {})
            
            all_questions = set(public_round_data.keys()) | set(private_round_data.keys())
            
            if not all_questions:
                continue

            sum_sq_diff = 0
            count = 0
            for q in all_questions:
                public_score = public_round_data.get(q)
                private_score = private_round_data.get(q)
                
                # Ensure both scores are available
                if public_score is not None and private_score is not None:
                    sum_sq_diff += (public_score - private_score) ** 2
                    count += 1
            
            if count > 0:
                distance = math.sqrt(sum_sq_diff)
                distances[agent_id]["rounds"].append(r)
                distances[agent_id]["distance"].append(distance)

    fig, ax = plt.subplots(figsize=(10, 6))

    for agent_id, data in distances.items():
        if data["rounds"]:
            ax.plot(
                range(len(data["rounds"])),
                data["distance"],
                marker="o",
                label=agents_dict.get(agent_id, agent_id),
            )

    ax.set_title(title)
    ax.set_xlabel("Round index (order only)")
    ax.set_ylabel("Euclidean Distance")
    ax.grid(True)
    ax.legend()
    ax.set_ylim(bottom=0)
    plt.tight_layout()

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
