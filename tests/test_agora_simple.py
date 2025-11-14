import pytest

from agora.agora import Agora
from agora.agent import Agent


def test_agora_runs_with_turn_limit(stub_llm_factory):
    """Ensure turn-taking alternates and stops when each agent hits the quota."""

    agent_a = Agent(
        name="Alpha",
        model="demo-model",
        llm_client=stub_llm_factory(["Alpha turn 1", "Alpha turn 2"]),
        system_prompt="You are Alpha.",
    )
    agent_b = Agent(
        name="Beta",
        model="demo-model",
        llm_client=stub_llm_factory(["Beta turn 1", "Beta turn 2"]),
        system_prompt="You are Beta.",
    )

    agora = Agora([agent_a, agent_b])

    history = agora.run(max_turns_per_agent=2)

    assert len(history) == 4
    assert [turn.metadata["speaker_name"] for turn in history] == [
        "Alpha",
        "Beta",
        "Alpha",
        "Beta",
    ]
    assert agent_a.view_history() == history
    assert agent_b.view_history() == history
    assert len(agent_a.memory) == 4
    assert len(agent_b.memory) == 4


def test_agora_rejects_invalid_turn_limit(stub_llm_factory):
    """Verify invalid turn thresholds raise user-friendly errors."""

    agent = Agent(
        name="Solo",
        model="demo-model",
        llm_client=stub_llm_factory(["only response"]),
    )
    agora = Agora([agent])
    with pytest.raises(ValueError):
        agora.run(max_turns_per_agent=0)
