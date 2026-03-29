import json

import pytest

from agora.agent import Agent, build_system_prompt
from agora.agora import Agora
from agora.memory import MemoryTurn


class QueueLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def complete(
        self,
        *,
        messages,
        model,
        survey_questions=None,
        survey_question_groups=None,
    ):
        self.calls.append(
            {
                "messages": list(messages),
                "model": model,
                "survey_questions": survey_questions,
                "survey_question_groups": survey_question_groups,
            }
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
    llm_client = QueueLLM([json.dumps({"Q1": "Agree"})])
    agent = Agent(
        name="Solo",
        model="demo",
        llm_client=llm_client,
        response_instruction="respond",
        survey_questions=["q1"],
        survey_question_groups={"Q1": "evaluative"},
        survey_public_prompt="Base\n{scale}\n",
    )
    agent.generate_public_survey_response(["q1"])
    assert llm_client.calls[0]["survey_questions"] == ["q1"]
    assert llm_client.calls[0]["survey_question_groups"] == {"Q1": "evaluative"}
    assert "Use the following Likert scale:" in llm_client.calls[0]["messages"][-1]["content"]
    assert "Q1. [Likert] q1" in llm_client.calls[0]["messages"][-1]["content"]

def test_agent_generate_survey_passes_questions_private():
    llm_client = QueueLLM([json.dumps({"Q1": "Neutral"})])
    agent = Agent(
        name="Solo",
        model="demo",
        llm_client=llm_client,
        response_instruction="respond",
        survey_questions=["q1"],
        survey_question_groups={"Q1": "incentive"},
        survey_private_prompt="Base\n{scale}\n",
    )
    agent.generate_private_survey_response(["q1"])
    assert llm_client.calls[0]["survey_questions"] == ["q1"]
    assert llm_client.calls[0]["survey_question_groups"] == {"Q1": "incentive"}
    assert "Strongly disagree" in llm_client.calls[0]["messages"][-1]["content"]
    assert "Q1. [Likert] q1" in llm_client.calls[0]["messages"][-1]["content"]


def test_agent_generate_survey_expands_partial_question_groups():
    llm_client = QueueLLM([json.dumps({"Q1": "Neutral", "Q2": "Agree"})])
    agent = Agent(
        name="Solo",
        model="demo",
        llm_client=llm_client,
        response_instruction="respond",
        survey_questions=["q1", "q2"],
        survey_question_groups={"Q2": "incentive"},
        survey_public_prompt="Base\n{scale}\n",
    )

    agent.generate_public_survey_response(["q1", "q2"])

    assert llm_client.calls[0]["survey_question_groups"] == {
        "Q1": "deliberative",
        "Q2": "incentive",
    }
    prompt = llm_client.calls[0]["messages"][-1]["content"]
    assert "Use the following Likert scale:" in prompt
    assert "Q1. [Likert] q1" in prompt
    assert "Q2. [Likert] q2" in prompt


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


def test_agent_strips_transcript_labels_from_responses():
    agent = Agent(
        name="Alpha",
        model="demo",
        llm_client=QueueLLM(["[Current instruction]\nAlpha: Hi"]),
        response_instruction="respond",
    )

    assert agent.generate_public_speech() == "Hi"


def test_agent_strips_speaker_prefix_before_transcript_label():
    agent = Agent(
        name="Alpha",
        model="demo",
        llm_client=QueueLLM(["Alpha: [Current instruction]\nHi"]),
        response_instruction="respond",
    )

    assert agent.generate_public_speech() == "Hi"


def test_agent_normalizes_apostrophes_in_responses():
    responses = [
        "It\u2019s public.",
        "It\u2019s private.",
        "It\u2019s interview.",
        json.dumps({"Q1": "It\u2019s public survey."}),
        json.dumps({"Q1": "It\u2019s private survey."}),
    ]
    agent = Agent(
        name="Solo",
        model="demo",
        llm_client=QueueLLM(responses),
        response_instruction="respond",
        private_response_instruction="reflect",
        survey_questions=["q1"],
        survey_public_prompt="Base\n",
        survey_private_prompt="Base\n",
    )
    assert agent.generate_public_speech() == "It's public."
    assert agent.generate_private_reflection() == "It's private."
    assert agent.generate_interview_response("Interview") == "It's interview."
    assert agent.generate_public_survey_response(["q1"]) == '{"Q1": "It\'s public survey."}'
    assert agent.generate_private_survey_response(["q1"]) == '{"Q1": "It\'s private survey."}'


def test_agent_normalizes_empty_responses():
    agent = Agent(
        name="Solo",
        model="demo",
        llm_client=QueueLLM([""]),
        response_instruction="respond",
    )
    assert agent.generate_public_speech() == ""


def test_agent_property_accessors():
    agent = Agent(
        name="Alpha",
        model="demo",
        llm_client=QueueLLM(["hi"]),
        response_instruction="respond",
        system_prompt="system",
        opening_instruction="open",
        public_survey_keep=True,
        private_survey_keep=True,
    )
    assert agent.system_prompt == "system"
    assert agent.response_instruction == "respond"
    assert agent.opening_instruction == "open"
    assert agent.public_survey_keep is True
    assert agent.private_survey_keep is True


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
        "public",
        json.dumps({"Q1": "Neutral"}),
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
    )
    beta = Agent(
        name="Beta",
        model="demo",
        llm_client=QueueLLM(["beta public"]),
        response_instruction="respond",
    )

    agora = Agora([agent, beta])
    history = agora.run(num_turns=1, verbose=True)
    assert history[0].role == "assistant"
    structured = agora.structured_history()
    assert structured["turns"][0]["Alpha"]["public_survey"] == {"Q1": 0}
    assert any("public survey" in line for line in capsys.readouterr().out.splitlines())

    with pytest.raises(KeyError):
        agora.history_for_agent("missing")


def test_agora_public_survey_keep_does_not_modify_public_speech():
    survey_payload = json.dumps({"Q1": "Neutral"})
    responses = [
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
    history = agora.run(num_turns=1)
    alpha_turn = next(turn for turn in history if turn.metadata.get("speaker_name") == "Alpha")
    assert alpha_turn.public_speech == "public speech"


def test_agora_public_survey_only():
    survey_payload = json.dumps({"Q1": "Neutral"})
    llm_client = QueueLLM(["public speech", survey_payload])
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
    agora.run(num_turns=1)
    structured = agora.structured_history()
    assert structured["turns"][0]["Alpha"]["public_survey"] == {"Q1": 0}
    assert structured["turns"][0]["Alpha"]["private_survey"] is None


def test_agora_private_survey_only():
    survey_payload = json.dumps({"Q1": "Agree"})
    llm_client = QueueLLM(["public speech", survey_payload])
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
    history = agora.run(num_turns=1)
    structured = agora.structured_history()
    assert structured["turns"][0]["Alpha"]["private_survey"] == {"Q1": 1}
    assert structured["turns"][0]["Alpha"]["public_survey"] is None
    alpha_turn = next(turn for turn in history if turn.metadata.get("speaker_name") == "Alpha")
    assert alpha_turn.public_speech == "public speech"


def test_agora_survey_calls_include_same_turn_context():
    alpha_llm = QueueLLM(
        [
            "alpha public",
            "alpha private",
            json.dumps({"Q1": "Neutral"}),
            json.dumps({"Q1": "Agree"}),
        ]
    )
    alpha = Agent(
        name="Alpha",
        model="demo",
        llm_client=alpha_llm,
        response_instruction="respond",
        private_response_instruction="reflect",
        private_response_keep=True,
        survey_questions=["q1"],
        survey_public_prompt="Base\n",
        survey_private_prompt="Base\n",
        enable_public_survey=True,
        enable_private_survey=True,
    )
    beta = Agent(
        name="Beta",
        model="demo",
        llm_client=QueueLLM(["beta public"]),
        response_instruction="respond",
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
    agora.run(num_turns=1)

    public_survey_messages = alpha_llm.calls[2]["messages"]
    private_survey_messages = alpha_llm.calls[3]["messages"]

    assert any(
        msg["content"] == "[You | public statement]\nalpha public"
        for msg in public_survey_messages
    )
    assert any(
        msg["content"] == "[You | public statement]\nalpha public"
        for msg in private_survey_messages
    )
    assert any(
        msg["content"] == "[You | private note]\nalpha private"
        for msg in public_survey_messages
    )
    assert any(
        msg["content"] == "[You | private note]\nalpha private"
        for msg in private_survey_messages
    )


def test_agora_structured_history_includes_exact_survey_receipt():
    alpha_llm = QueueLLM(
        [
            "alpha public",
            json.dumps({"Q1": "Agree"}),
        ]
    )
    alpha = Agent(
        name="Alpha",
        model="demo",
        llm_client=alpha_llm,
        response_instruction="respond",
        survey_questions=["q1"],
        survey_question_groups={"Q1": "incentive"},
        survey_public_prompt="Base\n{scale}\n",
        survey_private_prompt="Private\n",
        enable_public_survey=True,
        enable_private_survey=False,
    )
    beta = Agent(
        name="Beta",
        model="demo",
        llm_client=QueueLLM(["beta public"]),
        response_instruction="respond",
    )

    agora = Agora([alpha, beta], event_order=["public_utterance", "public_survey"])
    agora.run(num_turns=1)

    receipts = agora.structured_history()["llm_receipts"]
    survey_receipt = next(
        receipt for receipt in receipts if receipt["event_type"] == "public_survey"
    )

    prompt = survey_receipt["request"]["messages"][-1]["content"]
    assert "{scale}" not in prompt
    assert "Use the following Likert scale:" in prompt
    assert "- Strongly disagree" in prompt
    assert "- Strongly agree" in prompt
    assert "Q1. [Likert] q1" in prompt
    enum_values = survey_receipt["request"]["response_format"]["json_schema"][
        "schema"
    ]["properties"]["Q1"]["enum"]
    assert enum_values == [
        "Strongly disagree",
        "Disagree",
        "Neutral",
        "Agree",
        "Strongly agree",
    ]
    assert survey_receipt["response"] == '{"Q1": "Agree"}'


def test_agora_append_completion_receipt_requires_pending_receipt():
    alpha = Agent(
        name="Alpha",
        model="demo",
        llm_client=QueueLLM(["alpha public"]),
        response_instruction="respond",
    )
    beta = Agent(
        name="Beta",
        model="demo",
        llm_client=QueueLLM(["beta public"]),
        response_instruction="respond",
    )
    agora = Agora([alpha, beta])

    with pytest.raises(RuntimeError, match="Expected exactly one completion receipt"):
        agora._append_completion_receipt(
            alpha,
            turn_num=1,
            subturn="Alpha",
            event_type="public_utterance",
        )


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
    history = agora.run(num_turns=1)
    assert [turn.role for turn in history] == ["pre_interview", "assistant", "assistant", "post_interview"]
    assert any(turn.role == "pre_interview" for turn in agent.view_history())
    assert any(turn.role == "post_interview" for turn in agent.view_history())


def test_agora_verbose_excluded_notes(capsys):
    responses = ["pre", "public", "reflect", "post"]
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
    Agora([agent, beta]).run(num_turns=1, verbose=True)
    output = capsys.readouterr().out
    assert "(excluded)" in output


def test_agora_emits_sweep_progress_markers(capsys, monkeypatch):
    agent = Agent(
        name="Alpha",
        model="demo",
        llm_client=QueueLLM(["alpha public"]),
        response_instruction="respond",
    )
    beta = Agent(
        name="Beta",
        model="demo",
        llm_client=QueueLLM(["beta public"]),
        response_instruction="respond",
    )

    Agora([agent, beta]).run(num_turns=1, emit_progress_markers=True)

    output = capsys.readouterr().out
    assert "[agora progress] Turn 1/1" in output
