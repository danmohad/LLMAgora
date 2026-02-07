import json

import pytest

from agora.agent import Agent, build_system_prompt
from agora.agora import Agora
from agora.memory import MemoryTurn


class QueueLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def complete(self, *, messages, model, survey_questions=None):
        self.calls.append(
            {"messages": list(messages), "model": model, "survey_questions": survey_questions}
        )
        return self._responses.pop(0)


def test_agent_requires_agora_for_history():
    agent = Agent(
        name="Solo",
        model="demo",
        llm_client=QueueLLM(["hi"]),
        response_instruction="respond",
    )
    with pytest.raises(RuntimeError):
        agent.view_history()


def test_agent_private_reflection_requires_instruction():
    agent = Agent(
        name="Solo",
        model="demo",
        llm_client=QueueLLM(["hi"]),
        response_instruction="respond",
    )
    with pytest.raises(RuntimeError):
        agent.generate_private_reflection()


def test_agent_survey_requires_prompt_public():
    agent = Agent(
        name="Solo",
        model="demo",
        llm_client=QueueLLM(["hi"]),
        response_instruction="respond",
        survey_questions=["q1"],
        survey_public_prompt=None,
        survey_private_prompt=None,
    )
    with pytest.raises(RuntimeError):
        agent.generate_public_survey_response(["q1"])

def test_agent_survey_requires_prompt_private():
    agent = Agent(
        name="Solo",
        model="demo",
        llm_client=QueueLLM(["hi"]),
        response_instruction="respond",
        survey_questions=["q1"],
        survey_public_prompt=None,
        survey_private_prompt=None,
    )
    with pytest.raises(RuntimeError):
        agent.generate_private_survey_response(["q1"])


def test_agent_generate_survey_passes_questions():
    llm_client = QueueLLM([json.dumps({"Q1": "Neutral"})])
    agent = Agent(
        name="Solo",
        model="demo",
        llm_client=llm_client,
        response_instruction="respond",
        survey_questions=["q1"],
        survey_public_prompt="Base\n",
    )
    agent.generate_public_survey_response(["q1"])
    assert llm_client.calls[0]["survey_questions"] == ["q1"]
    assert "Q1." in llm_client.calls[0]["messages"][-1]["content"]

def test_agent_generate_survey_passes_questions_private():
    llm_client = QueueLLM([json.dumps({"Q1": "Neutral"})])
    agent = Agent(
        name="Solo",
        model="demo",
        llm_client=llm_client,
        response_instruction="respond",
        survey_questions=["q1"],
        survey_private_prompt="Base\n",
    )
    agent.generate_private_survey_response(["q1"])
    assert llm_client.calls[0]["survey_questions"] == ["q1"]
    assert "Q1." in llm_client.calls[0]["messages"][-1]["content"]


def test_agent_survey_generation_rejects_disabled_modes():
    agent = Agent(
        name="Solo",
        model="demo",
        llm_client=QueueLLM(["hi"]),
        response_instruction="respond",
        survey_questions=["q1"],
        survey_public_prompt="public",
        survey_private_prompt="private",
        enable_public_survey=False,
        enable_private_survey=False,
    )
    with pytest.raises(RuntimeError, match="Public survey requested but disabled"):
        agent.generate_public_survey_response(["q1"])
    with pytest.raises(RuntimeError, match="Private survey requested but disabled"):
        agent.generate_private_survey_response(["q1"])


def test_strip_speaker_prefix():
    agent = Agent(
        name="Alpha",
        model="demo",
        llm_client=QueueLLM(["hi"]),
        response_instruction="respond",
    )
    agent.observe_turn(
        MemoryTurn(
            turn_id=1,
            speaker_id="b",
            role="assistant",
            public_speech="Hello",
            metadata={"speaker_name": "Beta"},
        )
    )
    assert agent._strip_speaker_prefix("Alpha: Hi") == "Hi"
    assert agent._strip_speaker_prefix("Beta: Hi") == "Hi"


def test_agent_property_accessors():
    agent = Agent(
        name="Alpha",
        model="demo",
        llm_client=QueueLLM(["hi"]),
        response_instruction="respond",
        system_prompt="system",
        opening_instruction="open",
    )
    assert agent.system_prompt == "system"
    assert agent.response_instruction == "respond"
    assert agent.opening_instruction == "open"


def test_build_system_prompt_without_perceived_roles():
    cfg = {"self_role": "You are Alpha"}
    assert build_system_prompt(cfg, total_agents=1) == "You are Alpha"


def test_build_system_prompt_validation_errors():
    cfg = {"self_role": "You are Alpha", "perceived_nonself_roles": "nope"}
    with pytest.raises(ValueError):
        build_system_prompt(cfg, total_agents=2)

    cfg = {"self_role": "You are Alpha", "perceived_nonself_roles": ["nope"]}
    with pytest.raises(ValueError):
        build_system_prompt(cfg, total_agents=2)

    cfg = {"self_role": "You are Alpha", "perceived_nonself_roles": [{"name": ""}]}
    with pytest.raises(ValueError):
        build_system_prompt(cfg, total_agents=2)


def test_agora_requires_agents():
    with pytest.raises(ValueError, match="exactly two agents"):
        Agora([])


def test_agora_requires_unique_ids():
    llm_client = QueueLLM(["hi", "hello"])
    agent_a = Agent(
        name="Alpha",
        model="demo",
        llm_client=llm_client,
        response_instruction="respond",
        agent_id="same",
    )
    agent_b = Agent(
        name="Beta",
        model="demo",
        llm_client=llm_client,
        response_instruction="respond",
        agent_id="same",
    )
    with pytest.raises(ValueError):
        Agora([agent_a, agent_b])


def test_agora_survey_flow_and_unknown_agent(capsys):
    responses = [
        json.dumps({"Q1": "Neutral"}),
        json.dumps({"Q1": "Agree"}),
        "public",
        json.dumps({"Q1": "Agree"}),
        json.dumps({"Q1": "Neutral"}),
    ]
    llm_client = QueueLLM(responses)
    agent = Agent(
        name="Alpha",
        model="demo",
        llm_client=llm_client,
        response_instruction="respond",
        survey_questions=["q1"],
        survey_public_prompt="Base\n",
        survey_private_prompt="Base\n",
    )
    beta = Agent(
        name="Beta",
        model="demo",
        llm_client=QueueLLM(["beta public"]),
        response_instruction="respond",
    )

    agora = Agora([agent, beta])
    history = agora.run(max_turns_per_agent=1, verbose=True)
    assert history[0].role == "assistant"
    assert 0 in agora.survey_public_response[agent.id]
    assert any("survey response" in line for line in capsys.readouterr().out.splitlines())

    with pytest.raises(KeyError):
        agora.history_for_agent("missing")


def test_agora_public_survey_keep_appends_to_speech():
    survey_payload = json.dumps({"Q1": "Neutral"})
    responses = [
        survey_payload,
        json.dumps({"Q1": "Agree"}),
        "public speech",
        survey_payload,
        json.dumps({"Q1": "Agree"}),
    ]
    llm_client = QueueLLM(responses)
    agent = Agent(
        name="Alpha",
        model="demo",
        llm_client=llm_client,
        response_instruction="respond",
        survey_questions=["q1"],
        survey_public_prompt="Base\n",
        survey_private_prompt="Base\n",
        public_survey_keep=True,
    )
    beta = Agent(
        name="Beta",
        model="demo",
        llm_client=QueueLLM(["beta public"]),
        response_instruction="respond",
    )

    agora = Agora([agent, beta])
    history = agora.run(max_turns_per_agent=1)
    alpha_turn = next(turn for turn in history if turn.metadata.get("speaker_name") == "Alpha")
    assert alpha_turn.public_speech.startswith("public speech")
    assert alpha_turn.public_speech.endswith(survey_payload)


def test_agora_public_survey_only():
    survey_payload = json.dumps({"Q1": "Neutral"})
    llm_client = QueueLLM([survey_payload, "public speech", survey_payload])
    agent = Agent(
        name="Alpha",
        model="demo",
        llm_client=llm_client,
        response_instruction="respond",
        survey_questions=["q1"],
        survey_public_prompt="Base\n",
        survey_private_prompt="Base\n",
        enable_public_survey=True,
        enable_private_survey=False,
    )
    beta = Agent(
        name="Beta",
        model="demo",
        llm_client=QueueLLM(["beta public"]),
        response_instruction="respond",
    )

    agora = Agora([agent, beta])
    agora.run(max_turns_per_agent=1)
    assert agent.id in agora.survey_public_response
    assert agent.id not in agora.survey_private_response


def test_agora_private_survey_only():
    survey_payload = json.dumps({"Q1": "Agree"})
    llm_client = QueueLLM([survey_payload, "public speech", survey_payload])
    agent = Agent(
        name="Alpha",
        model="demo",
        llm_client=llm_client,
        response_instruction="respond",
        survey_questions=["q1"],
        survey_public_prompt="Base\n",
        survey_private_prompt="Base\n",
        enable_public_survey=False,
        enable_private_survey=True,
    )
    beta = Agent(
        name="Beta",
        model="demo",
        llm_client=QueueLLM(["beta public"]),
        response_instruction="respond",
    )

    agora = Agora([agent, beta])
    history = agora.run(max_turns_per_agent=1)
    assert agent.id in agora.survey_private_response
    assert agent.id not in agora.survey_public_response
    alpha_turn = next(turn for turn in history if turn.metadata.get("speaker_name") == "Alpha")
    assert alpha_turn.public_speech == "public speech"


def test_agora_pre_post_keep_true():
    responses = ["pre", "public", "post"]
    llm_client = QueueLLM(responses)
    agent = Agent(
        name="Alpha",
        model="demo",
        llm_client=llm_client,
        response_instruction="respond",
        pre_interview_instruction="pre",
        post_interview_instruction="post",
    )
    beta = Agent(
        name="Beta",
        model="demo",
        llm_client=QueueLLM(["beta public"]),
        response_instruction="respond",
    )
    agora = Agora([agent, beta])
    history = agora.run(max_turns_per_agent=1)
    assert [turn.role for turn in history] == ["pre_interview", "assistant", "assistant", "post_interview"]
    assert any(turn.role == "pre_interview" for turn in agent.view_history())
    assert any(turn.role == "post_interview" for turn in agent.view_history())


def test_agora_verbose_excluded_notes(capsys):
    responses = ["pre", "reflect", "public", "post"]
    llm_client = QueueLLM(responses)
    agent = Agent(
        name="Alpha",
        model="demo",
        llm_client=llm_client,
        response_instruction="respond",
        private_response_instruction="reflect",
        private_response_keep=False,
        pre_interview_instruction="pre",
        pre_interview_keep=False,
        post_interview_instruction="post",
        post_interview_keep=False,
    )
    beta = Agent(
        name="Beta",
        model="demo",
        llm_client=QueueLLM(["beta public"]),
        response_instruction="respond",
    )
    Agora([agent, beta]).run(max_turns_per_agent=1, verbose=True)
    output = capsys.readouterr().out
    assert "(excluded)" in output
