import pytest

from agora.agora import Agora
from agora.agent import Agent, build_system_prompt
from agora.memory import MemoryTurn
from agora.persistence import (
    load_history,
    load_snapshot,
    save_history,
    save_snapshot,
)


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


def test_private_reflections_are_private(stub_llm_factory):
    """Private reflections should only enter the speaking agent's memory."""

    llm_alpha = stub_llm_factory(["Alpha thinks", "Alpha turn 1"])
    llm_beta = stub_llm_factory(["Beta turn 1"])
    agent_a = Agent(
        name="Alpha",
        model="demo",
        llm_client=llm_alpha,
        response_instruction="Alpha public",
        private_response_instruction="Alpha private",
    )
    agent_b = Agent(name="Beta", model="demo", llm_client=llm_beta, response_instruction="Beta public")

    agora = Agora([agent_a, agent_b])
    history = agora.run(max_turns_per_agent=1)

    assert len(history) == 3  # reflection + two public turns
    assert history[0].role == "reflection"
    assert history[0].private_reflection == "Alpha thinks"

    alpha_history = agent_a.view_history()
    beta_history = agent_b.view_history()
    assert any(turn.private_reflection == "Alpha thinks" for turn in alpha_history)
    assert all(turn.private_reflection is None for turn in beta_history)


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
    assert message is not None
    assert message["type"] == "message"
    assert message["role"] == "user"
    assert message["content"][0]["text"].startswith("Alpha:")
    assert message["id"] == "msg-123"

    reflection = MemoryTurn(
        turn_id=2,
        speaker_id="agent-a",
        role="reflection",
        private_reflection="Thinking...",
    )
    assert reflection.to_openrouter_response(viewer_id="agent-b") is None
    my_view = reflection.to_openrouter_response(viewer_id="agent-a")
    assert my_view is not None
    assert my_view["role"] == "assistant"


def test_memory_turn_serialization_roundtrip():
    """Full serialization should preserve private reflections."""

    original = MemoryTurn(
        turn_id=42,
        speaker_id="agent-x",
        role="reflection",
        public_speech=None,
        private_reflection="Thinking aloud",
        metadata={"speaker_name": "X"},
        message_id="msg-x",
        status="completed",
    )
    clone = MemoryTurn.from_dict(original.to_dict())
    assert clone == original


def test_history_save_and_load(tmp_path):
    """save_history/load_history should round-trip JSON data."""

    turns = [
        MemoryTurn(turn_id=1, speaker_id="a", role="assistant", public_speech="Hi"),
        MemoryTurn(turn_id=2, speaker_id="a", role="reflection", private_reflection="Thinking"),
    ]
    path = tmp_path / "history.json"
    save_history(path, turns)
    loaded = load_history(path)
    assert loaded == turns


def test_agora_snapshot_roundtrip(tmp_path, stub_llm_factory):
    """Agora snapshots should restore agents, including private memory."""

    agent_a = Agent(
        name="Alpha",
        model="demo",
        llm_client=stub_llm_factory(["Alpha thinks", "Alpha turn"]),
        response_instruction="Alpha respond",
        private_response_instruction="Alpha private",
    )
    agent_b = Agent(
        name="Beta",
        model="demo",
        llm_client=stub_llm_factory(["Beta turn"]),
        response_instruction="Beta respond",
    )
    agora = Agora([agent_a, agent_b])
    agora.run(max_turns_per_agent=1)

    snapshot_path = tmp_path / "snapshot.json"
    save_snapshot(snapshot_path, agora)

    def factory(state):
        return stub_llm_factory([f"{state.name} restored"])

    restored = load_snapshot(snapshot_path, factory)
    assert [t.to_dict() for t in restored.history()] == [
        t.to_dict() for t in agora.history()
    ]

    restored_agents = {agent.name: agent for agent in restored.agents}
    assert any(turn.private_reflection for turn in restored_agents["Alpha"].memory)


def test_build_system_prompt_falls_back_to_raw():
    """Raw system_prompt should be used when provided (and self_role absent)."""

    cfg = {"system_prompt": "Use me verbatim."}
    result = build_system_prompt(cfg, total_agents=2)
    assert result == "Use me verbatim."


def test_build_system_prompt_from_roles():
    """Structured roles should concatenate self + perceived roles."""

    cfg = {
        "self_role": "You are Alpha.",
        "perceived_nonself_roles": [{"name": "Beta", "role": "Beta is your buyer."}],
    }
    result = build_system_prompt(cfg, total_agents=2)
    assert "You are Alpha." in result
    assert "Beta: Beta is your buyer." in result


def test_build_system_prompt_requires_roles():
    """Missing role information should raise."""

    with pytest.raises(ValueError):
        build_system_prompt({}, total_agents=2)

    with pytest.raises(ValueError):
        build_system_prompt({"system_prompt": "x", "self_role": "y"}, total_agents=2)

    with pytest.raises(ValueError):
        build_system_prompt(
            {
                "self_role": "Only me",
                "perceived_nonself_roles": [{"name": "Beta", "role": "hi"}],
            },
            total_agents=4,
        )


def test_interviews_respect_keep_flag(stub_llm_factory):
    """Pre/post interviews can be excluded from agent memory but still appear in history."""

    llm = stub_llm_factory(["pre", "public", "post"])
    agent = Agent(
        name="Alpha",
        model="demo",
        llm_client=llm,
        response_instruction="public",
        pre_interview_instruction="pre",
        pre_interview_keep=False,
        post_interview_instruction="post",
        post_interview_keep=False,
    )
    agora = Agora([agent])
    history = agora.run(max_turns_per_agent=1, verbose=False)
    assert [t.role for t in history] == ["pre_interview", "assistant", "post_interview"]
    assert len(agent.memory) == 1  # only the public turn is kept


def test_private_reflection_keep_flag(stub_llm_factory):
    """Private reflections can be excluded from agent memory when keep=False."""

    llm = stub_llm_factory(["think", "say"])
    agent = Agent(
        name="Alpha",
        model="demo",
        llm_client=llm,
        response_instruction="say",
        private_response_instruction="think",
        private_response_keep=False,
    )
    agora = Agora([agent])
    history = agora.run(max_turns_per_agent=1)
    assert any(t.role == "reflection" for t in history)
    assert all(t.role != "reflection" for t in agent.memory)


def test_skip_first_reflection_even_after_pre_interview(stub_llm_factory):
    """Skip flag should suppress the first reflection even if pre-interviews advance the counter."""

    llm = stub_llm_factory(["pre", "think", "say"])
    agent = Agent(
        name="Alpha",
        model="demo",
        llm_client=llm,
        response_instruction="say",
        private_response_instruction="think",
        private_response_keep=True,
        pre_interview_instruction="pre",
        pre_interview_keep=False,
    )
    history = Agora([agent]).run(max_turns_per_agent=1, skip_first_agent_first_reflection=True)
    # Should see pre-interview + public turn only
    assert [t.role for t in history] == ["pre_interview", "assistant"]
