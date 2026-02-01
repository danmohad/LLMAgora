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
    result = evaluator._score_text("", "p1", n_samples=1)
    assert result == [3]


def test_score_text_parses_number():
    evaluator = PersonaEvaluator(StubClient(["Score: 4"]), _personas())
    result = evaluator._score_text("Hello", "p1", n_samples=1)
    assert result == [4]


def test_score_text_multiple_samples():
    evaluator = PersonaEvaluator(StubClient(["4", "5", "4"]), _personas())
    result = evaluator._score_text("Hello", "p1", n_samples=3)
    assert result == [4, 5, 4]


def test_score_text_handles_invalid_response(capsys):
    evaluator = PersonaEvaluator(StubClient(["no score"]), _personas())
    result = evaluator._score_text("Hello", "p1", n_samples=1)
    assert result == [3]
    captured = capsys.readouterr()
    assert "Warning" in captured.out


def test_score_text_handles_exception(capsys):
    evaluator = PersonaEvaluator(StubClient(fail=True), _personas())
    result = evaluator._score_text("Hello", "p1", n_samples=1)
    assert result == [3]
    captured = capsys.readouterr()
    assert "Error scoring text" in captured.out


def test_score_text_only_prints_warning_once(capsys):
    evaluator = PersonaEvaluator(StubClient(["no score", "no score", "no score"]), _personas())
    result = evaluator._score_text("Hello", "p1", n_samples=3)
    assert result == [3, 3, 3]
    captured = capsys.readouterr()
    # Should only print warning once, not three times
    assert captured.out.count("Warning") == 1


def test_create_prompt_requires_known_persona():
    evaluator = PersonaEvaluator(StubClient(), _personas())
    with pytest.raises(ValueError):
        evaluator._create_evaluation_prompt("text", "missing")


def test_evaluate_debate_from_history_requires_two_agents():
    evaluator = PersonaEvaluator(StubClient(), _personas())
    # Create memory turns with only one agent
    turns = [
        MemoryTurn(
            turn_id=1,
            speaker_id="a",
            role="assistant",
            public_speech="test",
            metadata={"speaker_name": "Alpha"},
        ),
    ]
    with pytest.raises(ValueError, match="Expected exactly 2 agents"):
        evaluator.evaluate_debate_from_history(turns, "p1", "p1")


def test_evaluate_debate_scores_and_cumulative(monkeypatch):
    evaluator = PersonaEvaluator(StubClient(), _personas())
    calls = {"count": 0}

    def fake_score(text, persona_id, turn_label="turn", n_samples=1):
        calls["count"] += 1
        return [5] * n_samples

    monkeypatch.setattr(evaluator, "_score_text", fake_score)

    # Create proper memory turns
    turns = [
        MemoryTurn(
            turn_id=1,
            speaker_id="a",
            role="assistant",
            public_speech="A",
            metadata={"speaker_name": "Alpha"},
        ),
        MemoryTurn(
            turn_id=2,
            speaker_id="a",
            role="reflection",
            private_reflection="A0",
            metadata={"speaker_name": "Alpha"},
        ),
        MemoryTurn(
            turn_id=3,
            speaker_id="b",
            role="assistant",
            public_speech="B",
            metadata={"speaker_name": "Beta"},
        ),
        MemoryTurn(
            turn_id=4,
            speaker_id="b",
            role="reflection",
            private_reflection="B0",
            metadata={"speaker_name": "Beta"},
        ),
    ]

    result = evaluator.evaluate_debate_from_history(turns, "p1", "p1", n_samples=1)
    
    # full_debate scores are now tuples (mean, std)
    assert result.alpha.full_debate_public_score == (5.0, 0.0)
    assert result.beta.full_debate_private_score == (5.0, 0.0)
    
    # Should have 8 calls: 2 agents × 1 turn × 4 score types (pub ind, priv ind, pub cum, priv cum)
    assert calls["count"] == 8


def test_evaluate_debate_verbose_output(capsys, monkeypatch):
    evaluator = PersonaEvaluator(StubClient(), _personas())

    def fake_score(text, persona_id, turn_label="turn", n_samples=1):
        return [3] * n_samples

    monkeypatch.setattr(evaluator, "_score_text", fake_score)

    turns = [
        MemoryTurn(
            turn_id=1,
            speaker_id="a",
            role="assistant",
            public_speech="A",
            metadata={"speaker_name": "Alpha"},
        ),
        MemoryTurn(
            turn_id=2,
            speaker_id="a",
            role="reflection",
            private_reflection="A0",
            metadata={"speaker_name": "Alpha"},
        ),
        MemoryTurn(
            turn_id=3,
            speaker_id="b",
            role="assistant",
            public_speech="B",
            metadata={"speaker_name": "Beta"},
        ),
        MemoryTurn(
            turn_id=4,
            speaker_id="b",
            role="reflection",
            private_reflection="B0",
            metadata={"speaker_name": "Beta"},
        ),
    ]

    evaluator.evaluate_debate_from_history(turns, "p1", "p1", verbose=True, n_samples=1)
    output = capsys.readouterr().out
    assert "Evaluating Alpha" in output
    assert "Scoring individual public speech" in output


def test_evaluate_agent_empty_turns():
    evaluator = PersonaEvaluator(StubClient(), _personas())
    evaluation = evaluator._evaluate_agent({"debate_turns": []}, "p1", n_samples=1)
    assert evaluation.persona_id == "p1"
    assert len(evaluation.public_turn_scores) == 0


def test_persona_score_statistics():
    score = PersonaScore(turn_num=1, scores_raw=[4, 4, 5, 4, 4])
    assert score.score_mean == 4.2
    assert score.score_std == pytest.approx(0.4, abs=0.01)


def test_persona_score_single_value():
    score = PersonaScore(turn_num=1, scores_raw=[5])
    assert score.score_mean == 5.0
    assert score.score_std == 0.0


def test_evaluation_to_dict_new_structure():
    score = PersonaScore(turn_num=1, scores_raw=[4, 5, 4])
    agent_eval = AgentPersonaEvaluation(persona_id="p1")
    agent_eval.public_turn_scores.append(score)
    agent_eval.full_debate_public_score = (4.33, 0.47)
    
    agent_dict = agent_eval.to_dict()
    
    # Check new structure
    assert agent_dict["public_turn_scores"]["turns"] == [1]
    assert agent_dict["public_turn_scores"]["scores"]["mean"] == [pytest.approx(4.33, abs=0.01)]
    assert agent_dict["public_turn_scores"]["scores"]["std"] == [pytest.approx(0.47, abs=0.01)]
    assert agent_dict["public_turn_scores"]["scores"]["raw"] == [[4, 5, 4]]
    assert agent_dict["full_debate_public_score"]["mean"] == pytest.approx(4.33, abs=0.01)
    assert agent_dict["full_debate_public_score"]["std"] == pytest.approx(0.47, abs=0.01)


def test_evaluation_to_dict_roundtrip():
    score1 = PersonaScore(turn_num=1, scores_raw=[4, 4, 5])
    score2 = PersonaScore(turn_num=2, scores_raw=[5, 5, 5])
    
    agent_eval = AgentPersonaEvaluation(persona_id="p1")
    agent_eval.public_turn_scores.extend([score1, score2])
    agent_eval.full_debate_public_score = (4.67, 0.47)
    
    debate_eval = DebatePersonaEvaluation(alpha=agent_eval, beta=agent_eval)
    debate_dict = debate_eval.to_dict()

    assert debate_dict["alpha"]["persona_id"] == "p1"
    assert debate_dict["beta"]["persona_id"] == "p1"
    assert len(debate_dict["alpha"]["public_turn_scores"]["turns"]) == 2
    assert debate_dict["alpha"]["full_debate_public_score"]["mean"] == pytest.approx(4.67, abs=0.01)


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
    assert len(alpha["debate_turns"]) == 1
    assert alpha["debate_turns"][0]["public_speech"] == "public"
    assert alpha["debate_turns"][0]["private_reflection"] == "reflect-after"


def test_get_structured_debate_history_multiple_turns():
    turns = [
        # Alpha turn 1
        MemoryTurn(
            turn_id=1,
            speaker_id="a",
            role="assistant",
            public_speech="alpha-pub-1",
            metadata={"speaker_name": "Alpha"},
        ),
        MemoryTurn(
            turn_id=2,
            speaker_id="a",
            role="reflection",
            private_reflection="alpha-priv-1",
            metadata={"speaker_name": "Alpha"},
        ),
        # Beta turn 1
        MemoryTurn(
            turn_id=3,
            speaker_id="b",
            role="assistant",
            public_speech="beta-pub-1",
            metadata={"speaker_name": "Beta"},
        ),
        MemoryTurn(
            turn_id=4,
            speaker_id="b",
            role="reflection",
            private_reflection="beta-priv-1",
            metadata={"speaker_name": "Beta"},
        ),
        # Alpha turn 2
        MemoryTurn(
            turn_id=5,
            speaker_id="a",
            role="assistant",
            public_speech="alpha-pub-2",
            metadata={"speaker_name": "Alpha"},
        ),
        MemoryTurn(
            turn_id=6,
            speaker_id="a",
            role="reflection",
            private_reflection="alpha-priv-2",
            metadata={"speaker_name": "Alpha"},
        ),
    ]

    structured = get_structured_debate_history(turns)
    
    assert len(structured["Alpha"]["debate_turns"]) == 2
    assert len(structured["Beta"]["debate_turns"]) == 1
    
    assert structured["Alpha"]["debate_turns"][0]["public_speech"] == "alpha-pub-1"
    assert structured["Alpha"]["debate_turns"][0]["private_reflection"] == "alpha-priv-1"
    assert structured["Alpha"]["debate_turns"][1]["public_speech"] == "alpha-pub-2"
    assert structured["Alpha"]["debate_turns"][1]["private_reflection"] == "alpha-priv-2"
    
    assert structured["Beta"]["debate_turns"][0]["public_speech"] == "beta-pub-1"
    assert structured["Beta"]["debate_turns"][0]["private_reflection"] == "beta-priv-1"


def test_personas_with_nested_structure():
    """Test that evaluator handles both nested and flat persona structures."""
    nested = {"personas": {"p1": {"actual_persona": "Test"}}}
    flat = {"p1": {"actual_persona": "Test"}}
    
    eval_nested = PersonaEvaluator(StubClient(), nested)
    eval_flat = PersonaEvaluator(StubClient(), flat)
    
    # Both should work
    assert eval_nested.personas["p1"]["actual_persona"] == "Test"
    assert eval_flat.personas["p1"]["actual_persona"] == "Test"


def test_n_samples_propagates_through_evaluation(monkeypatch):
    """Test that n_samples parameter is correctly passed through all evaluation calls."""
    evaluator = PersonaEvaluator(StubClient(), _personas())
    
    n_samples_used = []
    
    def fake_score(text, persona_id, turn_label="turn", n_samples=1):
        n_samples_used.append(n_samples)
        return [4] * n_samples
    
    monkeypatch.setattr(evaluator, "_score_text", fake_score)
    
    turns = [
        MemoryTurn(
            turn_id=1,
            speaker_id="a",
            role="assistant",
            public_speech="A",
            metadata={"speaker_name": "Alpha"},
        ),
        MemoryTurn(
            turn_id=2,
            speaker_id="a",
            role="reflection",
            private_reflection="A0",
            metadata={"speaker_name": "Alpha"},
        ),
        MemoryTurn(
            turn_id=3,
            speaker_id="b",
            role="assistant",
            public_speech="B",
            metadata={"speaker_name": "Beta"},
        ),
        MemoryTurn(
            turn_id=4,
            speaker_id="b",
            role="reflection",
            private_reflection="B0",
            metadata={"speaker_name": "Beta"},
        ),
    ]
    
    evaluator.evaluate_debate_from_history(turns, "p1", "p1", n_samples=7)
    
    # All calls should have n_samples=7
    assert all(n == 7 for n in n_samples_used)
    assert len(n_samples_used) == 8  # 2 agents × 4 score types


def test_full_debate_scores_match_last_cumulative():
    """Test that full_debate scores match the last cumulative scores."""
    score1 = PersonaScore(turn_num=1, scores_raw=[4, 4, 5])
    score2 = PersonaScore(turn_num=2, scores_raw=[5, 5, 5])
    
    agent_eval = AgentPersonaEvaluation(persona_id="p1")
    agent_eval.public_cumulative_scores.extend([score1, score2])
    
    # Simulate what _evaluate_agent does
    last_score = agent_eval.public_cumulative_scores[-1]
    agent_eval.full_debate_public_score = (last_score.score_mean, last_score.score_std)
    
    assert agent_eval.full_debate_public_score[0] == score2.score_mean
    assert agent_eval.full_debate_public_score[1] == score2.score_std