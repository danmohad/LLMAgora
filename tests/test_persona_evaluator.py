import pytest

from agora.memory import MemoryTurn
from agora.persona_evaluator import (
    AgentPersonaEvaluation,
    DebatePersonaEvaluation,
    PersonaEvaluator,
    PersonaScore,
    get_structured_debate_history,
)


class StubClient:
    def __init__(self, responses=None, fail=False):
        self.responses = list(responses or [])
        self.fail = fail
        self.calls = []

    def complete(self, *, messages, model):
        self.calls.append({"messages": list(messages), "model": model})
        if self.fail:
            raise RuntimeError("boom")
        if not self.responses:
            return "3"
        return self.responses.pop(0)


def _personas():
    return {"personas": {"p1": {"actual_persona": "Test persona"}}}


def test_score_text_handles_empty():
    evaluator = PersonaEvaluator(StubClient(), _personas())
    assert evaluator._score_text("", "p1") == 3


def test_score_text_parses_number():
    evaluator = PersonaEvaluator(StubClient(["Score: 4"]), _personas())
    assert evaluator._score_text("Hello", "p1") == 4


def test_score_text_handles_invalid_response(capsys):
    evaluator = PersonaEvaluator(StubClient(["no score"]), _personas())
    assert evaluator._score_text("Hello", "p1") == 3
    captured = capsys.readouterr()
    assert "Warning" in captured.out


def test_score_text_handles_exception(capsys):
    evaluator = PersonaEvaluator(StubClient(fail=True), _personas())
    assert evaluator._score_text("Hello", "p1") == 3
    captured = capsys.readouterr()
    assert "Error scoring text" in captured.out


def test_create_prompt_requires_known_persona():
    evaluator = PersonaEvaluator(StubClient(), _personas())
    with pytest.raises(ValueError):
        evaluator._create_evaluation_prompt("text", "missing")


def test_evaluate_debate_requires_two_agents():
    evaluator = PersonaEvaluator(StubClient(), _personas())
    with pytest.raises(ValueError):
        evaluator.evaluate_debate({"Alpha": {}}, "p1", "p1")


def test_evaluate_debate_scores_and_cumulative(monkeypatch):
    evaluator = PersonaEvaluator(StubClient(), _personas())
    calls = {"count": 0}

    def fake_score(text, persona_id, turn_label="turn"):
        calls["count"] += 1
        return 5

    monkeypatch.setattr(evaluator, "_score_text", fake_score)

    debate_data = {
        "Alpha": {
            "debate_turns": [{"public_speech": "A", "private_reflection": "A0"}],
            "pre_interview": None,
            "post_interview": None,
        },
        "Beta": {
            "debate_turns": [{"public_speech": "B", "private_reflection": "B0"}],
            "pre_interview": None,
            "post_interview": None,
        },
    }

    result = evaluator.evaluate_debate(debate_data, "p1", "p1")
    assert result.alpha.full_debate_public_score == 5
    assert result.beta.full_debate_private_score == 5
    assert calls["count"] == 8


def test_evaluate_debate_verbose_output(capsys, monkeypatch):
    evaluator = PersonaEvaluator(StubClient(), _personas())

    def fake_score(text, persona_id, turn_label="turn"):
        return 3

    monkeypatch.setattr(evaluator, "_score_text", fake_score)

    debate_data = {
        "Alpha": {"debate_turns": [{"public_speech": "A", "private_reflection": "A0"}]},
        "Beta": {"debate_turns": [{"public_speech": "B", "private_reflection": "B0"}]},
    }

    evaluator.evaluate_debate(debate_data, "p1", "p1", verbose=True)
    output = capsys.readouterr().out
    assert "Evaluating Alpha" in output
    assert "Scoring individual public speech" in output


def test_evaluate_agent_empty_turns():
    evaluator = PersonaEvaluator(StubClient(), _personas())
    evaluation = evaluator._evaluate_agent({"debate_turns": []}, "p1")
    assert evaluation.persona_id == "p1"


def test_evaluation_to_dict_roundtrip():
    score = PersonaScore(turn_num=1, score=4)
    agent_eval = AgentPersonaEvaluation(persona_id="p1")
    agent_eval.public_turn_scores.append(score)
    debate_eval = DebatePersonaEvaluation(alpha=agent_eval, beta=agent_eval)

    agent_dict = agent_eval.to_dict()
    debate_dict = debate_eval.to_dict()

    assert agent_dict["public_turn_scores"] == [(1, 4)]
    assert debate_dict["alpha"]["persona_id"] == "p1"


def test_get_structured_debate_history_captures_turns():
    turns = [
        MemoryTurn(
            turn_id=1,
            speaker_id="a",
            role="pre_interview",
            private_reflection="pre",
            metadata={"speaker_name": "Alpha"},
        ),
        MemoryTurn(
            turn_id=2,
            speaker_id="a",
            role="reflection",
            private_reflection="reflect-first",
            metadata={"speaker_name": "Alpha"},
        ),
        MemoryTurn(
            turn_id=3,
            speaker_id="a",
            role="assistant",
            public_speech="public",
            metadata={"speaker_name": "Alpha"},
        ),
        MemoryTurn(
            turn_id=4,
            speaker_id="a",
            role="reflection",
            private_reflection="reflect-after",
            metadata={"speaker_name": "Alpha"},
        ),
        MemoryTurn(
            turn_id=5,
            speaker_id="a",
            role="post_interview",
            private_reflection="post",
            metadata={"speaker_name": "Alpha"},
        ),
    ]

    structured = get_structured_debate_history(turns)
    alpha = structured["Alpha"]
    assert alpha["pre_interview"] == "pre"
    assert alpha["post_interview"] == "post"
    assert alpha["debate_turns"][0]["public_speech"] == "public"
    assert alpha["debate_turns"][0]["private_reflection"] == "reflect-after"
