import pytest

from agora.agora import Agora
from agora.agent import Agent
from agora.memory import MemoryTurn


def test_agora_runs_with_turn_limit(stub_llm_factory):
    """Ensure turn-taking alternates and stops when each agent hits the quota."""

    agent_a = Agent(
        name="Alpha",
        model="demo-model",
        llm_client=stub_llm_factory(["Alpha turn 1", "Alpha turn 2"]),
        system_prompt="You are Alpha.",
        response_instruction="Alpha respond.",
    )
    agent_b = Agent(
        name="Beta",
        model="demo-model",
        llm_client=stub_llm_factory(["Beta turn 1", "Beta turn 2"]),
        system_prompt="You are Beta.",
        response_instruction="Beta respond.",
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
        response_instruction="Solo respond.",
    )
    agora = Agora([agent])
    with pytest.raises(ValueError):
        agora.run(max_turns_per_agent=0)


def test_agent_message_roles_follow_schema(stub_llm_factory):
    """Ensure chat payloads mark self turns as assistant and others as user."""

    llm_a = stub_llm_factory(["Alpha turn 1", "Alpha turn 2"])
    llm_b = stub_llm_factory(["Beta turn 1", "Beta turn 2"])
    agent_a = Agent(
        name="Alpha",
        model="demo-model",
        llm_client=llm_a,
        system_prompt="You are Alpha.",
        response_instruction="Alpha respond.",
    )
    agent_b = Agent(
        name="Beta",
        model="demo-model",
        llm_client=llm_b,
        system_prompt="You are Beta.",
        response_instruction="Beta respond.",
    )
    Agora([agent_a, agent_b]).run(max_turns_per_agent=2)

    # Alpha's first call should just have system + final user prompt.
    alpha_messages = llm_a.calls[0]["messages"]
    assert alpha_messages[0]["role"] == "system"
    assert alpha_messages[-1]["role"] == "user"

    # Beta's first call should include Alpha's speech tagged as user.
    beta_first = llm_b.calls[0]["messages"]
    assert any(msg["role"] == "user" and msg["content"] == "Alpha turn 1" for msg in beta_first)

    # Beta's second call should include its previous turn tagged as assistant.
    beta_second = llm_b.calls[1]["messages"]
    assert any(msg["role"] == "assistant" and msg["content"] == "Beta turn 1" for msg in beta_second)


def test_multi_agent_histories_label_user_messages(stub_llm_factory):
    """When more than two speakers exist, user messages include the speaker name."""

    llm_a = stub_llm_factory(["Alpha turn 1", "Alpha turn 2"])
    llm_b = stub_llm_factory(["Beta turn 1", "Beta turn 2"])
    llm_c = stub_llm_factory(["Gamma turn 1", "Gamma turn 2"])
    agent_a = Agent(name="Alpha", model="demo", llm_client=llm_a, response_instruction="Alpha respond.")
    agent_b = Agent(name="Beta", model="demo", llm_client=llm_b, response_instruction="Beta respond.")
    agent_c = Agent(name="Gamma", model="demo", llm_client=llm_c, response_instruction="Gamma respond.")

    Agora([agent_a, agent_b, agent_c]).run(max_turns_per_agent=1)

    beta_messages = llm_b.calls[0]["messages"]
    assert any(
        msg["role"] == "user" and msg["content"].startswith("Alpha:")
        for msg in beta_messages
    )


def test_memory_turn_openrouter_response_conversion():
    """MemoryTurn should emit OpenRouter-compatible structures."""

    turn = MemoryTurn(
        turn_id=1,
        speaker_id="agent-a",
        role="assistant",
        public_speech="Hello world",
        metadata={"speaker_name": "Alpha"},
        message_id="msg-123",
    )
    message = turn.to_openrouter_response(viewer_id="agent-b", multi_party=True)
    assert message["type"] == "message"
    assert message["role"] == "user"
    assert message["content"][0]["text"].startswith("Alpha:")
    assert message["id"] == "msg-123"
