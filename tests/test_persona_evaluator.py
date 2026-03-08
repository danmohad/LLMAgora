import matplotlib
import pytest

from agora.debate_history import get_structured_debate_history
from agora.persona_adherence_evaluator import (
    AgentPersonaEvaluation,
    DebatePersonaEvaluation,
    PERSONA_METRIC_FULL_DEBATE_PRIVATE,
    PERSONA_METRIC_FULL_DEBATE_PUBLIC,
    PersonaEvaluator,
    PersonaScore,
)
from agora.plotting import plot_persona_adherence


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


def _structured_history(
    turn_rows: list[tuple[str | None, str | None, str | None, str | None]],
    *,
    alpha_pre: str | None = None,
    beta_pre: str | None = None,
    alpha_post: str | None = None,
    beta_post: str | None = None,
):
    return {
        "pre_interviews": {
            "Alpha": {"response": alpha_pre},
            "Beta": {"response": beta_pre},
        },
        "turns": [
            {
                "turn_num": turn_num,
                "Alpha": {
                    "public_utterance": alpha_public,
                    "private_utterance": alpha_private,
                },
                "Beta": {
                    "public_utterance": beta_public,
                    "private_utterance": beta_private,
                },
            }
            for turn_num, (
                alpha_public,
                alpha_private,
                beta_public,
                beta_private,
            ) in enumerate(turn_rows, start=1)
        ],
        "post_interviews": {
            "Alpha": {"response": alpha_post},
            "Beta": {"response": beta_post},
        },
    }


def test_sample_persona_scores_handles_empty():
    evaluator = PersonaEvaluator(StubClient(), _personas())
    result = evaluator._sample_persona_scores("", "p1", n_samples=1)
    assert result == [3]


def test_sample_persona_scores_parses_number():
    evaluator = PersonaEvaluator(StubClient(["Score: 4"]), _personas())
    result = evaluator._sample_persona_scores("Hello", "p1", n_samples=1)
    assert result == [4]


def test_sample_persona_scores_multiple_samples():
    evaluator = PersonaEvaluator(StubClient(["4", "5", "4"]), _personas())
    result = evaluator._sample_persona_scores("Hello", "p1", n_samples=3)
    assert result == [4, 5, 4]


def test_sample_persona_scores_handles_invalid_response(capsys):
    evaluator = PersonaEvaluator(StubClient(["no score"]), _personas())
    result = evaluator._sample_persona_scores("Hello", "p1", n_samples=1)
    assert result == [3]
    captured = capsys.readouterr()
    assert "Warning" in captured.out


def test_sample_persona_scores_handles_exception(capsys):
    evaluator = PersonaEvaluator(StubClient(fail=True), _personas())
    result = evaluator._sample_persona_scores("Hello", "p1", n_samples=1)
    assert result == [3]
    captured = capsys.readouterr()
    assert "Error scoring text" in captured.out


def test_sample_persona_scores_only_prints_warning_once(capsys):
    evaluator = PersonaEvaluator(StubClient(["no score", "no score", "no score"]), _personas())
    result = evaluator._sample_persona_scores("Hello", "p1", n_samples=3)
    assert result == [3, 3, 3]
    captured = capsys.readouterr()
    # Should only print warning once, not three times
    assert captured.out.count("Warning") == 1


def test_create_prompt_requires_known_persona():
    evaluator = PersonaEvaluator(StubClient(), _personas())
    with pytest.raises(ValueError):
        evaluator._build_persona_scoring_prompt("text", "missing")


def test_evaluate_debate_from_history_requires_two_agents():
    evaluator = PersonaEvaluator(StubClient(), _personas())
    turns = {
        "Alpha": {
            "debate_turns": [
                {
                    "turn_num": 1,
                    "public_speech": "test",
                    "private_reflection": "",
                    "public_stance": "",
                }
            ],
            "pre_interview": None,
            "post_interview": None,
        }
    }
    with pytest.raises(ValueError, match="Expected exactly 2 agents"):
        evaluator.evaluate_debate_from_history(turns, "p1", "p1")


def test_evaluate_debate_scores_and_cumulative(monkeypatch):
    evaluator = PersonaEvaluator(StubClient(), _personas())
    calls = {"count": 0}

    def fake_score(text, persona_id, slice_label="turn", n_samples=1):
        calls["count"] += 1
        return [5] * n_samples

    monkeypatch.setattr(evaluator, "_sample_persona_scores", fake_score)

    turns = _structured_history([("A", "A0", "B", "B0")])

    result = evaluator.evaluate_debate_from_history(turns, "p1", "p1", n_samples=1)
    
    # full_debate scores are now tuples (mean, std)
    assert result.alpha.full_debate_public_score == (5.0, 0.0)
    assert result.beta.full_debate_private_score == (5.0, 0.0)
    
    # Should have 8 calls: 2 agents × 1 turn × 4 score types (pub ind, priv ind, pub cum, priv cum)
    assert calls["count"] == 8


def test_evaluate_debate_verbose_output(capsys, monkeypatch):
    evaluator = PersonaEvaluator(StubClient(), _personas())

    def fake_score(text, persona_id, slice_label="turn", n_samples=1):
        return [3] * n_samples

    monkeypatch.setattr(evaluator, "_sample_persona_scores", fake_score)

    turns = _structured_history([("A", "A0", "B", "B0")])

    evaluator.evaluate_debate_from_history(turns, "p1", "p1", verbose=True, n_samples=1)
    output = capsys.readouterr().out
    assert "Evaluating Alpha" in output
    assert "scoring public turn adherence" in output


def test_evaluate_single_agent_empty_turns():
    evaluator = PersonaEvaluator(StubClient(), _personas())
    evaluation = evaluator._evaluate_single_agent(
        {"debate_turns": []},
        "p1",
        selected_metrics={"public_per_turn"},
        n_samples=1,
    )
    assert evaluation.persona_id == "p1"
    assert len(evaluation.public_per_turn_scores) == 0


def test_evaluate_debate_rejects_unknown_metric():
    evaluator = PersonaEvaluator(StubClient(), _personas())
    turns = _structured_history([("A", None, "B", None)])
    with pytest.raises(ValueError, match="Unknown persona analysis metrics"):
        evaluator.evaluate_debate_from_history(
            turns,
            "p1",
            "p1",
            metrics=["not_real"],
        )


def test_full_debate_metrics_can_run_without_cumulative(monkeypatch):
    evaluator = PersonaEvaluator(StubClient(), _personas())
    labels = []

    def fake_score(_text, _persona_id, slice_label="turn", n_samples=1):
        labels.append(slice_label)
        return [4] * n_samples

    monkeypatch.setattr(evaluator, "_sample_persona_scores", fake_score)
    turns = _structured_history([("A", "A0", "B", "B0")])

    result = evaluator.evaluate_debate_from_history(
        turns,
        "p1",
        "p1",
        metrics=[
            PERSONA_METRIC_FULL_DEBATE_PUBLIC,
            PERSONA_METRIC_FULL_DEBATE_PRIVATE,
        ],
    )
    assert result.alpha.full_debate_public_score is not None
    assert result.alpha.full_debate_private_score is not None
    assert "full debate public" in labels
    assert "full debate private" in labels


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
    agent_eval.public_per_turn_scores.append(score)
    agent_eval.full_debate_public_score = (4.33, 0.47)
    
    agent_dict = agent_eval.to_dict()
    
    # Check new structure
    assert agent_dict["public_per_turn_scores"]["turns"] == [1]
    assert agent_dict["public_per_turn_scores"]["scores"]["mean"] == [pytest.approx(4.33, abs=0.01)]
    assert agent_dict["public_per_turn_scores"]["scores"]["std"] == [pytest.approx(0.47, abs=0.01)]
    assert agent_dict["public_per_turn_scores"]["scores"]["raw"] == [[4, 5, 4]]
    assert agent_dict["full_debate_public_score"]["mean"] == pytest.approx(4.33, abs=0.01)
    assert agent_dict["full_debate_public_score"]["std"] == pytest.approx(0.47, abs=0.01)


def test_evaluation_to_dict_roundtrip():
    score1 = PersonaScore(turn_num=1, scores_raw=[4, 4, 5])
    score2 = PersonaScore(turn_num=2, scores_raw=[5, 5, 5])
    
    agent_eval = AgentPersonaEvaluation(persona_id="p1")
    agent_eval.public_per_turn_scores.extend([score1, score2])
    agent_eval.full_debate_public_score = (4.67, 0.47)
    
    debate_eval = DebatePersonaEvaluation(alpha=agent_eval, beta=agent_eval)
    debate_dict = debate_eval.to_dict()

    assert debate_dict["alpha"]["persona_id"] == "p1"
    assert debate_dict["beta"]["persona_id"] == "p1"
    assert len(debate_dict["alpha"]["public_per_turn_scores"]["turns"]) == 2
    assert debate_dict["alpha"]["full_debate_public_score"]["mean"] == pytest.approx(4.67, abs=0.01)


def test_get_structured_debate_history_captures_turns():
    turns = _structured_history(
        [("public", "reflect-after", None, None)],
        alpha_pre="pre",
        alpha_post="post",
    )

    structured = get_structured_debate_history(turns)
    alpha = structured["Alpha"]
    assert alpha["pre_interview"] == "pre"
    assert alpha["post_interview"] == "post"
    assert len(alpha["debate_turns"]) == 1
    assert alpha["debate_turns"][0]["public_speech"] == "public"
    assert alpha["debate_turns"][0]["private_reflection"] == "reflect-after"


def test_get_structured_debate_history_multiple_turns():
    turns = _structured_history(
        [
            ("alpha-pub-1", "alpha-priv-1", "beta-pub-1", "beta-priv-1"),
            ("alpha-pub-2", "alpha-priv-2", "beta-pub-2", "beta-priv-2"),
        ]
    )

    structured = get_structured_debate_history(turns)
    
    assert len(structured["Alpha"]["debate_turns"]) == 2
    assert len(structured["Beta"]["debate_turns"]) == 2
    
    assert structured["Alpha"]["debate_turns"][0]["public_speech"] == "alpha-pub-1"
    assert structured["Alpha"]["debate_turns"][0]["private_reflection"] == "alpha-priv-1"
    assert structured["Alpha"]["debate_turns"][1]["public_speech"] == "alpha-pub-2"
    assert structured["Alpha"]["debate_turns"][1]["private_reflection"] == "alpha-priv-2"
    
    assert structured["Beta"]["debate_turns"][0]["public_speech"] == "beta-pub-1"
    assert structured["Beta"]["debate_turns"][0]["private_reflection"] == "beta-priv-1"
    assert structured["Beta"]["debate_turns"][1]["public_speech"] == "beta-pub-2"
    assert structured["Beta"]["debate_turns"][1]["private_reflection"] == "beta-priv-2"


def test_get_structured_debate_history_from_canonical_turn_structure():
    payload = {
        "pre_interviews": {
            "Alpha": {"response": "alpha pre"},
            "Beta": {"response": "beta pre"},
        },
        "turns": [
            {
                "turn_num": 1,
                "Alpha": {
                    "public_utterance": "alpha pub",
                    "private_utterance": "alpha priv",
                },
                "Beta": {
                    "public_utterance": "beta pub",
                    "private_utterance": "beta priv",
                },
            }
        ],
        "post_interviews": {
            "Alpha": {"response": "alpha post"},
            "Beta": {"response": "beta post"},
        },
    }

    structured = get_structured_debate_history(payload)
    assert structured["Alpha"]["pre_interview"] == "alpha pre"
    assert structured["Beta"]["post_interview"] == "beta post"
    assert structured["Alpha"]["debate_turns"][0]["turn_num"] == 1
    assert structured["Alpha"]["debate_turns"][0]["private_reflection"] == "alpha priv"


def test_get_structured_debate_history_rejects_noncanonical_input():
    with pytest.raises(ValueError, match="canonical structured history format"):
        get_structured_debate_history([])


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
    
    def fake_score(text, persona_id, slice_label="turn", n_samples=1):
        n_samples_used.append(n_samples)
        return [4] * n_samples

    monkeypatch.setattr(evaluator, "_sample_persona_scores", fake_score)
    
    turns = _structured_history([("A", "A0", "B", "B0")])
    
    evaluator.evaluate_debate_from_history(turns, "p1", "p1", n_samples=7)
    
    # All calls should have n_samples=7
    assert all(n == 7 for n in n_samples_used)
    assert len(n_samples_used) == 8  # 2 agents × 4 score types


def test_evaluate_debate_rejects_raw_turn_list():
    evaluator = PersonaEvaluator(StubClient(), _personas())
    with pytest.raises(ValueError, match="canonical structured history"):
        evaluator.evaluate_debate_from_history([], "p1", "p1")


def test_full_debate_scores_match_last_cumulative():
    """Test that full_debate scores match the last cumulative scores."""
    score1 = PersonaScore(turn_num=1, scores_raw=[4, 4, 5])
    score2 = PersonaScore(turn_num=2, scores_raw=[5, 5, 5])
    
    agent_eval = AgentPersonaEvaluation(persona_id="p1")
    agent_eval.public_cumulative_scores.extend([score1, score2])
    
    # Simulate what evaluator does for full-debate summaries
    last_score = agent_eval.public_cumulative_scores[-1]
    agent_eval.full_debate_public_score = (last_score.score_mean, last_score.score_std)
    
    assert agent_eval.full_debate_public_score[0] == score2.score_mean
    assert agent_eval.full_debate_public_score[1] == score2.score_std


def _sample_eval_dict():
    def _scores():
        return {"turns": [1], "scores": {"mean": [4.333], "se": [0.333], "raw": [[4, 5, 4]]}}

    agent_entry = {
        "persona_id": "p1",
        "computed_metrics": [],
        "public_per_turn_scores": _scores(),
        "private_per_turn_scores": _scores(),
        "public_cumulative_scores": _scores(),
        "private_cumulative_scores": _scores(),
        "full_debate_public_score": {"mean": None, "std": None},
        "full_debate_private_score": {"mean": None, "std": None},
    }
    beta_entry = {**agent_entry, "persona_id": "p2"}
    return {"alpha": agent_entry, "beta": beta_entry}


def test_plot_persona_adherence_saves_and_closes(tmp_path):
    matplotlib.use("Agg", force=True)
    output_path = tmp_path / "persona.png"
    fig = plot_persona_adherence(
        _sample_eval_dict(),
        "Alpha",
        "Beta",
        save_path=str(output_path),
        show_plot=False,
    )
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert fig is not None


def test_plot_persona_adherence_show(monkeypatch):
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    calls = {"count": 0}

    def fake_show():
        calls["count"] += 1

    monkeypatch.setattr(plt, "show", fake_show)
    fig = plot_persona_adherence(_sample_eval_dict(), "Alpha", "Beta", show_plot=True)
    assert calls["count"] == 1
    plt.close(fig)


def test_plot_persona_adherence_handles_missing_series():
    matplotlib.use("Agg", force=True)
    sparse = {
        "alpha": {
            "public_per_turn_scores": {"turns": [], "scores": {"mean": [], "std": [], "raw": []}},
            "private_per_turn_scores": {"turns": [], "scores": {"mean": [], "std": [], "raw": []}},
            "public_cumulative_scores": {"turns": [], "scores": {"mean": [], "std": [], "raw": []}},
            "private_cumulative_scores": {"turns": [], "scores": {"mean": [], "std": [], "raw": []}},
            "full_debate_public_score": {"mean": None, "std": None},
            "full_debate_private_score": {"mean": None, "std": None},
            "persona_id": "p1",
        },
        "beta": {
            "public_per_turn_scores": {"turns": [], "scores": {"mean": [], "std": [], "raw": []}},
            "private_per_turn_scores": {"turns": [], "scores": {"mean": [], "std": [], "raw": []}},
            "public_cumulative_scores": {"turns": [], "scores": {"mean": [], "std": [], "raw": []}},
            "private_cumulative_scores": {"turns": [], "scores": {"mean": [], "std": [], "raw": []}},
            "full_debate_public_score": {"mean": None, "std": None},
            "full_debate_private_score": {"mean": None, "std": None},
            "persona_id": "p2",
        },
    }
    fig = plot_persona_adherence(sparse, "Alpha", "Beta", show_plot=False)
    assert fig is not None
