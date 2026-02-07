import pytest

from agora.agora import Agora
from agora.agent import Agent, build_system_prompt
from agora.memory import MemoryTurn
from agora.persistence import load_snapshot, save_snapshot


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

    history = agora.run(num_turns=2)

    assert len(history) == 4
    assert [turn.metadata["speaker_name"] for turn in history] == [
        "Alpha",
        "Beta",
        "Alpha",
        "Beta",
    ]
    assert agent_a.view_history() == history
    assert agent_b.view_history() == history
    assert len(agent_a.view_history()) == 4
    assert len(agent_b.view_history()) == 4


def test_agora_rejects_invalid_turn_limit(stub_llm_factory):
    """Verify invalid turn thresholds raise user-friendly errors."""

    agent_a = Agent(
        name="Alpha",
        model="demo-model",
        llm_client=stub_llm_factory(["only response"]),
        response_instruction="Alpha respond.",
    )
    agent_b = Agent(
        name="Beta",
        model="demo-model",
        llm_client=stub_llm_factory(["only response"]),
        response_instruction="Beta respond.",
    )
    agora = Agora([agent_a, agent_b])
    with pytest.raises(ValueError):
        agora.run(num_turns=0)


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
    Agora([agent_a, agent_b]).run(num_turns=2)

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


def test_opening_instruction_used_for_first_public_turn(stub_llm_factory):
    """First public speaker should receive the opening instruction."""

    llm_a = stub_llm_factory(["Alpha turn 1"])
    llm_b = stub_llm_factory(["Beta turn 1"])
    agent_a = Agent(
        name="Alpha",
        model="demo-model",
        llm_client=llm_a,
        response_instruction="Alpha respond.",
        opening_instruction="Alpha open.",
    )
    agent_b = Agent(
        name="Beta",
        model="demo-model",
        llm_client=llm_b,
        response_instruction="Beta respond.",
    )

    Agora([agent_a, agent_b]).run(num_turns=1)

    alpha_messages = llm_a.calls[0]["messages"]
    beta_messages = llm_b.calls[0]["messages"]
    assert alpha_messages[-1]["content"] == "Alpha open."
    assert beta_messages[-1]["content"] == "Beta respond."


def test_agora_rejects_more_than_two_agents(stub_llm_factory):
    """Agora should require exactly two agents."""

    llm = stub_llm_factory(["Alpha turn 1"])
    solo = Agent(name="Solo", model="demo", llm_client=llm, response_instruction="respond.")
    with pytest.raises(ValueError, match="exactly two agents"):
        Agora([solo])

    llm_a = stub_llm_factory(["Alpha turn 1", "Alpha turn 2"])
    llm_b = stub_llm_factory(["Beta turn 1", "Beta turn 2"])
    llm_c = stub_llm_factory(["Gamma turn 1", "Gamma turn 2"])
    agent_a = Agent(name="Alpha", model="demo", llm_client=llm_a, response_instruction="Alpha respond.")
    agent_b = Agent(name="Beta", model="demo", llm_client=llm_b, response_instruction="Beta respond.")
    agent_c = Agent(name="Gamma", model="demo", llm_client=llm_c, response_instruction="Gamma respond.")

    with pytest.raises(ValueError, match="exactly two agents"):
        Agora([agent_a, agent_b, agent_c])


def test_private_reflections_are_private(stub_llm_factory):
    """Private reflections should only enter the speaking agent's memory."""

    llm_alpha = stub_llm_factory(["Alpha turn 1", "Alpha thinks"])
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
    history = agora.run(num_turns=1)

    assert len(history) == 3  # two public turns + reflection
    assert history[1].role == "reflection"
    assert history[1].private_reflection == "Alpha thinks"

    alpha_history = agent_a.view_history()
    beta_history = agent_b.view_history()
    assert any(turn.private_reflection == "Alpha thinks" for turn in alpha_history)
    assert all(turn.private_reflection is None for turn in beta_history)


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
    agora.run(num_turns=1)

    snapshot_path = tmp_path / "snapshot.json"
    save_snapshot(snapshot_path, agora)

    def factory(state):
        return stub_llm_factory([f"{state.name} restored"])

    restored = load_snapshot(snapshot_path, factory)
    assert [t.to_dict() for t in restored.history()] == [
        t.to_dict() for t in agora.history()
    ]

    restored_agents = {agent.name: agent for agent in restored.agents}
    assert any(
        turn.private_reflection for turn in restored_agents["Alpha"].view_history()
    )


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
    assert "Beta is your buyer." in result
    assert "Beta:" not in result


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
    beta = Agent(
        name="Beta",
        model="demo",
        llm_client=stub_llm_factory(["beta public"]),
        response_instruction="public",
    )
    agora = Agora([agent, beta])
    history = agora.run(num_turns=1, verbose=False)
    assert [t.role for t in history] == ["pre_interview", "assistant", "assistant", "post_interview"]
    assert len(agent.view_history()) == 2  # both public turns are visible


def test_private_reflection_keep_flag(stub_llm_factory):
    """Private reflections can be excluded from agent memory when keep=False."""

    llm = stub_llm_factory(["say", "think"])
    agent = Agent(
        name="Alpha",
        model="demo",
        llm_client=llm,
        response_instruction="say",
        private_response_instruction="think",
        private_response_keep=False,
    )
    beta = Agent(
        name="Beta",
        model="demo",
        llm_client=stub_llm_factory(["beta says"]),
        response_instruction="say",
    )
    agora = Agora([agent, beta])
    history = agora.run(num_turns=1)
    assert any(t.role == "reflection" for t in history)
    assert all(t.role != "reflection" for t in agent.view_history())


def test_skip_first_reflection_even_after_pre_interview(stub_llm_factory):
    """Skip flag should suppress the first reflection even if pre-interviews advance the counter."""

    llm = stub_llm_factory(["pre", "say"])
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
    beta = Agent(
        name="Beta",
        model="demo",
        llm_client=stub_llm_factory(["beta says"]),
        response_instruction="say",
    )
    history = Agora([agent, beta]).run(num_turns=1, skip_first_agent_first_reflection=True)
    # Should see pre-interview + public turn only
    assert [t.role for t in history] == ["pre_interview", "assistant", "assistant"]


def test_agora_event_order_validation(stub_llm_factory):
    alpha = Agent(
        name="Alpha",
        model="demo",
        llm_client=stub_llm_factory(["a pub", "a priv"]),
        response_instruction="say",
        private_response_instruction="think",
    )
    beta = Agent(
        name="Beta",
        model="demo",
        llm_client=stub_llm_factory(["b pub"]),
        response_instruction="say",
    )

    with pytest.raises(ValueError, match="event_order must not be empty"):
        Agora([alpha, beta], event_order=[])
    with pytest.raises(ValueError, match="unknown events"):
        Agora([alpha, beta], event_order=["public_utterance", "not_real"])
    with pytest.raises(ValueError, match="must not contain duplicates"):
        Agora([alpha, beta], event_order=["public_utterance", "public_utterance"])
    with pytest.raises(ValueError, match="must match enabled events 1:1"):
        Agora([alpha, beta], event_order=["public_utterance"])

    agora = Agora([alpha, beta])
    with pytest.raises(ValueError, match="Unknown interview stage"):
        agora._empty_interview_stage(stage="invalid")


def test_agora_load_structured_history_rebuilds_all_event_types(stub_llm_factory):
    alpha = Agent(
        name="Alpha",
        model="demo",
        llm_client=stub_llm_factory(["unused"]),
        response_instruction="say",
        private_response_instruction="think",
        survey_questions=["q1"],
        survey_public_prompt="Public\n",
        survey_private_prompt="Private\n",
        enable_public_survey=True,
        enable_private_survey=True,
        public_survey_keep=False,
        private_survey_keep=True,
        post_interview_instruction="post",
        post_interview_keep=True,
    )
    beta = Agent(
        name="Beta",
        model="demo",
        llm_client=stub_llm_factory(["unused"]),
        response_instruction="say",
        private_response_instruction="think",
        survey_questions=["q1"],
        survey_public_prompt="Public\n",
        survey_private_prompt="Private\n",
        enable_public_survey=True,
        enable_private_survey=True,
        public_survey_keep=False,
        private_survey_keep=True,
        post_interview_instruction="post",
        post_interview_keep=True,
    )
    agora = Agora(
        [alpha, beta],
        event_order=[
            "public_utterance",
            "private_utterance",
            "public_survey",
            "private_survey",
        ],
    )

    agora.load_structured_history(
        event_order=[
            "public_utterance",
            "private_utterance",
            "public_survey",
            "private_survey",
        ],
        pre_interviews={
            "Alpha": {
                "speaker_id": "missing",
                "speaker_name": "Alpha",
                "response": "alpha pre",
                "keep": True,
            },
            "Beta": {
                "speaker_id": beta.id,
                "speaker_name": "Beta",
                "response": None,
                "keep": True,
            },
        },
        turns=[
            {
                "turn_num": 1,
                "Alpha": {
                    "speaker_id": alpha.id,
                    "speaker_name": "Alpha",
                    "public_utterance": "alpha public",
                    "private_utterance": "alpha private",
                    "public_survey": {"Q1": 0},
                    "private_survey": None,
                },
                "Beta": {
                    "speaker_id": beta.id,
                    "speaker_name": "Beta",
                    "public_utterance": None,
                    "private_utterance": "beta private",
                    "public_survey": None,
                    "private_survey": {"Q1": 2},
                },
            }
        ],
        post_interviews={
            "Alpha": {
                "speaker_id": "missing",
                "speaker_name": "Alpha",
                "response": "alpha post",
                "keep": True,
            },
            "Beta": {
                "speaker_id": beta.id,
                "speaker_name": "Beta",
                "response": None,
                "keep": True,
            },
        },
    )

    roles = [turn.role for turn in agora.history()]
    assert "public_survey" in roles
    assert "private_survey" in roles
    alpha_visible_roles = [turn.role for turn in agora.history_for_agent(alpha.id)]
    assert "public_survey" not in alpha_visible_roles
    assert "private_survey" not in alpha_visible_roles
    beta_visible_roles = [turn.role for turn in agora.history_for_agent(beta.id)]
    assert "private_survey" in beta_visible_roles
