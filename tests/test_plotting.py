from types import SimpleNamespace
from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt

from agora import plotting
from agora.plotting import plot_survey_distance, plot_survey_responses


def test_plot_survey_responses_no_agents(tmp_path):
    matplotlib.use("Agg", force=True)
    output_path = tmp_path / "plot.png"
    plot_survey_responses({}, [], ["q1"], "title", output_path)
    assert not output_path.exists()


def test_plot_survey_responses_no_questions(tmp_path):
    matplotlib.use("Agg", force=True)
    output_path = tmp_path / "plot.png"
    agents = [SimpleNamespace(id="a", name="Alpha")]
    plot_survey_responses({"a": {}}, agents, ["q1"], "title", output_path)
    assert not output_path.exists()


def test_plot_survey_responses_saves_file(tmp_path):
    matplotlib.use("Agg", force=True)
    output_path = tmp_path / "plot.png"
    agents = [SimpleNamespace(id="a", name="Alpha")]
    responses = {
        "a": {
            0: {f"Q{i}": i for i in range(1, 7)},
            1: {f"Q{i}": i + 1 for i in range(1, 7)},
        }
    }
    questions = {
        "default": [
            "This is a long survey question that should be truncated.",
            "Short 2",
        ],
        "direct": ["Short 3", "Short 4"],
        "sentiment": ["Short 5", "Short 6"],
    }
    plot_survey_responses(responses, agents, questions, "Survey", output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_survey_responses_returns_when_panels_empty(tmp_path, monkeypatch):
    matplotlib.use("Agg", force=True)
    output_path = tmp_path / "plot.png"
    agents = [SimpleNamespace(id="a", name="Alpha")]
    responses = {"a": {0: {"Q1": 1}}}

    monkeypatch.setattr(plotting, "_build_question_panels", lambda *_args, **_kwargs: [])

    plot_survey_responses(responses, agents, ["q1"], "Survey", output_path)

    assert not output_path.exists()


def test_plot_survey_responses_hides_unused_subplots(tmp_path, monkeypatch):
    matplotlib.use("Agg", force=True)
    output_path = tmp_path / "plot.png"
    agents = [SimpleNamespace(id="a", name="Alpha")]
    responses = {
        "a": {
            0: {f"Q{i}": i for i in range(1, 7)},
            1: {f"Q{i}": i + 1 for i in range(1, 7)},
        }
    }
    questions = {"default": [f"Question {i}" for i in range(1, 7)]}

    real_subplots = plt.subplots
    captured: dict[str, object] = {}

    def capturing_subplots(*args, **kwargs):
        fig, axes = real_subplots(*args, **kwargs)
        captured["axes"] = axes
        return fig, axes

    monkeypatch.setattr(plt, "subplots", capturing_subplots)
    monkeypatch.setattr(plt, "close", lambda *_args, **_kwargs: None)

    plot_survey_responses(responses, agents, questions, "Survey", output_path)

    axes = captured["axes"]
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]
    assert output_path.exists()
    assert axes_list[6].get_visible() is False


def test_plot_survey_responses_hides_unused_grouped_subplots(tmp_path, monkeypatch):
    matplotlib.use("Agg", force=True)
    output_path = tmp_path / "plot.png"
    agents = [SimpleNamespace(id="a", name="Alpha")]
    responses = {
        "a": {
            0: {f"Q{i}": i for i in range(1, 12)},
            1: {f"Q{i}": i + 1 for i in range(1, 12)},
        }
    }
    questions = {
        "default": [f"Question {i}" for i in range(1, 7)],
        "sentiment": [f"Sentiment {i}" for i in range(7, 12)],
    }

    real_subplots = plt.subplots
    captured: dict[str, object] = {}

    def capturing_subplots(*args, **kwargs):
        fig, axes = real_subplots(*args, **kwargs)
        captured["axes"] = axes
        return fig, axes

    monkeypatch.setattr(plt, "subplots", capturing_subplots)
    monkeypatch.setattr(plt, "close", lambda *_args, **_kwargs: None)

    plot_survey_responses(responses, agents, questions, "Survey", output_path)

    axes = captured["axes"]
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]
    assert output_path.exists()
    assert axes_list[6].get_visible() is True
    assert all(ax.get_visible() is False for ax in axes_list[7:])


def test_plot_survey_distance_no_agents(tmp_path):
    matplotlib.use("Agg", force=True)
    output_path = tmp_path / "plot.png"
    plot_survey_distance({}, {}, [], [], "Distance", output_path)
    assert not output_path.exists()


def test_plot_survey_distance_saves_file(tmp_path):
    matplotlib.use("Agg", force=True)
    output_path = tmp_path / "plot.png"
    agents = [SimpleNamespace(id="a", name="Alpha")]
    public_responses = {
        "a": {
            0: {"Q1": 1, "Q2": 2},
            1: {"Q1": 2},
            2: {},
        },
        "b": {},
    }
    private_responses = {
        "a": {
            0: {"Q1": 3, "Q2": 1},
            1: {"Q1": 2, "Q2": None},
            2: {},
        }
    }
    plot_survey_distance(
        public_responses,
        private_responses,
        agents,
        {"default": ["question1"], "sentiment": ["question2"]},
        "Distance",
        output_path,
    )
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_survey_distance_returns_without_configured_groups(tmp_path):
    matplotlib.use("Agg", force=True)
    output_path = tmp_path / "plot.png"
    agents = [SimpleNamespace(id="a", name="Alpha")]
    public_responses = {"a": {0: {"Q1": 1}}}
    private_responses = {"a": {0: {"Q1": 0}}}

    plot_survey_distance(
        public_responses,
        private_responses,
        agents,
        [],
        "Distance",
        output_path,
    )

    assert not output_path.exists()


def test_plot_survey_distance_with_extra_questions(tmp_path):
    matplotlib.use("Agg", force=True)
    output_path = tmp_path / "plot.png"
    agents = [
        SimpleNamespace(id="a", name="Alpha"),
        SimpleNamespace(id="b", name="Beta"),
    ]
    public_responses = {
        "a": {
            0: {"Q1": 1, "Q2": 2, "Q6": 3, "Q7": 4},
            1: {"Q1": 2, "Q3": 1, "Q8": 2, "Q9": 1},
        },
        "b": {
            0: {"Q1": 1, "Q10": 5},
            1: {"Q2": 2},
        },
    }
    private_responses = {
        "a": {
            0: {"Q1": 2, "Q2": 1, "Q6": 2, "Q7": 3},
            1: {"Q1": 3, "Q3": 2, "Q8": 1, "Q9": 2},
        },
        "b": {
            0: {"Q1": 2, "Q10": 4},
            1: {"Q2": 3},
        },
    }
    questions = {
        "default": [
            "question1",
            "question2",
            "question3",
            "question4",
            "question5",
        ],
        "sentiment": ["extra6", "extra7", "extra8", "extra9", "extra10"],
    }
    plot_survey_distance(
        public_responses,
        private_responses,
        agents,
        questions,
        "Distance with Extra Questions",
        output_path,
    )
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_survey_distance_hides_unused_grouped_subplots(tmp_path, monkeypatch):
    matplotlib.use("Agg", force=True)
    output_path = tmp_path / "plot.png"
    agents = [SimpleNamespace(id="a", name="Alpha")]
    public_responses = {
        "a": {
            0: {f"Q{i}": i for i in range(1, 13)},
        }
    }
    private_responses = {
        "a": {
            0: {f"Q{i}": i - 1 for i in range(1, 13)},
        }
    }
    questions = {
        "default": [f"Question {i}" for i in range(1, 8)],
        "sentiment": [f"Sentiment {i}" for i in range(8, 13)],
    }

    real_subplots = plt.subplots
    captured: dict[str, object] = {}

    def capturing_subplots(*args, **kwargs):
        fig, axes = real_subplots(*args, **kwargs)
        captured["axes"] = axes
        return fig, axes

    monkeypatch.setattr(plt, "subplots", capturing_subplots)
    monkeypatch.setattr(plt, "close", lambda *_args, **_kwargs: None)

    plot_survey_distance(
        public_responses,
        private_responses,
        agents,
        questions,
        "Distance",
        output_path,
    )

    axes = captured["axes"]
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]
    assert output_path.exists()
    assert axes_list[7].get_visible() is True
    assert all(ax.get_visible() is False for ax in axes_list[8:])


def test_plot_survey_distance_sets_distinct_y_limits(tmp_path, monkeypatch):
    matplotlib.use("Agg", force=True)
    output_path = tmp_path / "plot.png"

    # Capture the created axes so we can assert y-limits after the call.
    import matplotlib.pyplot as plt

    real_subplots = plt.subplots
    captured: dict[str, object] = {}

    def capturing_subplots(*args, **kwargs):
        fig, axes = real_subplots(*args, **kwargs)
        captured["fig"] = fig
        captured["axes"] = axes
        return fig, axes

    monkeypatch.setattr(plt, "subplots", capturing_subplots)
    monkeypatch.setattr(plt, "close", lambda *args, **kwargs: None)

    agents = [SimpleNamespace(id="a", name="Alpha")]
    public_responses = {"a": {0: {"Q1": 1, "Q3": 4}}}
    private_responses = {"a": {0: {"Q1": 4, "Q3": 1}}}

    # Q1 is per-question, while Q2/Q3 are sentiment questions and collapse into
    # the Avg. Sentiment Dist. panel.
    plot_survey_distance(
        public_responses,
        private_responses,
        agents,
        {"default": ["question1"], "sentiment": ["question2", "question3"]},
        "Distance",
        output_path,
        y_limits_base=(-4, 4),
        y_limits_avg=(0, 4),
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    axes = captured["axes"]
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    base_ax_0 = axes_list[0]
    avg_ax = axes_list[1]

    assert base_ax_0.get_ylim() == (-4.0, 4.0)
    assert avg_ax.get_ylim() == (0.0, 4.0)


def test_build_question_panels_adds_unknown_question_panel():
    panels = plotting._build_question_panels(
        [{"text": "Known", "group": "default"}],
        ["Q1", "Q3"],
    )

    assert panels == [
        {"label": "Known", "questions": ["Q1"]},
        {"label": "Q3", "questions": ["Q3"]},
    ]


def test_survey_panel_value_returns_none_when_no_values_present():
    assert plotting._survey_panel_value({"Q1": None}, ["Q1"]) is None


# ---------------------------------------------------------------------------
# plot_group_survey
# ---------------------------------------------------------------------------

def test_wrap_label_short_fits_one_line():
    from agora.plotting import _wrap_label
    assert _wrap_label("Short") == "Short"


def test_wrap_label_long_wraps_to_two_lines():
    from agora.plotting import _wrap_label
    result = _wrap_label("This is a fairly long question label", width=20, max_lines=2)
    assert "\n" in result
    assert result.count("\n") == 1


def test_wrap_label_very_long_truncates_second_line():
    from agora.plotting import _wrap_label
    long = "word " * 30
    result = _wrap_label(long, width=20, max_lines=2)
    assert result.count("\n") == 1
    assert result.endswith("...")


def test_plot_group_survey_public_and_private_smoke():
    matplotlib.use("Agg", force=True)
    from agora.plotting import plot_group_survey

    agg = {
        "public": {
            "Alpha": {"Q1": {"turns": [1, 2], "mean": [1.0, -1.0], "se": [0.1, 0.2]}},
            "Beta": {"Q1": {"turns": [1, 2], "mean": [0.0, 0.5], "se": [0.05, 0.1]}},
        },
        "private": {
            "Alpha": {"Q1": {"turns": [1, 2], "mean": [2.0, 0.0], "se": [0.3, 0.1]}},
        },
        "diff": {
            "Alpha": {"Q1": {"turns": [1, 2], "mean": [-1.0, -1.0], "se": [0.2, 0.15]}},
        },
    }
    with patch("agora.plotting.plt.show"):
        plot_group_survey(agg, "Alpha Agent", "Beta Agent")


def test_plot_group_survey_empty_does_not_crash():
    matplotlib.use("Agg", force=True)
    from agora.plotting import plot_group_survey

    with patch("agora.plotting.plt.show"):
        plot_group_survey({})
        plot_group_survey({"public": {}, "private": {}})


def test_plot_group_survey_with_questions():
    matplotlib.use("Agg", force=True)
    from agora.plotting import plot_group_survey

    agg = {
        "public": {
            "Alpha": {
                "Q1": {"turns": [1], "mean": [1.0], "se": [0.1]},
                "Q2": {"turns": [1], "mean": [-1.0], "se": [0.2]},
            },
        },
        "diff": {
            "Alpha": {
                "Q1": {"turns": [1], "mean": [0.5], "se": [0.1]},
            },
        },
    }
    questions = {"default": ["Question one", "Question two"]}
    with patch("agora.plotting.plt.show"):
        plot_group_survey(agg, survey_questions=questions)


def test_plot_group_survey_hides_unused_subplots(monkeypatch):
    matplotlib.use("Agg", force=True)
    from agora.plotting import plot_group_survey

    # 6 questions → 5-column layout → one unused subplot
    agg = {
        "public": {
            "Alpha": {
                f"Q{i}": {"turns": [1], "mean": [float(i)], "se": [0.1]}
                for i in range(1, 7)
            },
        },
    }
    real_subplots = plt.subplots
    captured: dict = {}

    def capturing_subplots(*args, **kwargs):
        fig, axes = real_subplots(*args, **kwargs)
        captured["axes"] = axes
        return fig, axes

    monkeypatch.setattr(plt, "subplots", capturing_subplots)
    with patch("agora.plotting.plt.show"):
        plot_group_survey(agg)

    axes = captured["axes"]
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]
    assert axes_list[6].get_visible() is False


def test_plot_group_survey_panels_empty_does_not_crash(monkeypatch):
    matplotlib.use("Agg", force=True)
    from agora.plotting import plot_group_survey

    monkeypatch.setattr(plotting, "_build_question_panels", lambda *_a, **_kw: [])
    agg = {
        "public": {
            "Alpha": {"Q1": {"turns": [1], "mean": [1.0], "se": [0.0]}},
        },
    }
    with patch("agora.plotting.plt.show"):
        plot_group_survey(agg, survey_questions=["A question"])


# ---------------------------------------------------------------------------
# Additional coverage tests
# ---------------------------------------------------------------------------

def test_plot_group_semantic_similarity_cpriva_show_true(monkeypatch):
    """Cross-agent private alignment triggers the second plt.show() call (line 668)."""
    matplotlib.use("Agg", force=True)
    from agora.plotting import plot_group_semantic_similarity

    calls = []
    monkeypatch.setattr(plt, "show", lambda: calls.append(1))
    agg = {
        "cross_agent_private_alignment": {"turns": [1], "mean": [0.6], "se": [0.05]},
    }
    plot_group_semantic_similarity(agg, show=True)
    assert len(calls) == 1


def test_plot_group_nli_unknown_label_fallback_color():
    """A label not matching contradiction/neutral/entailment uses the grey fallback."""
    matplotlib.use("Agg", force=True)
    from agora.plotting import plot_group_nli

    _nli_dist = {
        "turns": [1],
        "label_names": ["unusual_label"],
        "distributions": {"unusual_label": {"mean": [0.5], "se": [0.1]}},
    }
    agg = {
        "id2label": {0: "unusual_label"},
        "self_consistency": {"AgentA": _nli_dist},
    }
    with patch("agora.plotting.plt.show"):
        plot_group_nli(agg)


def test_plot_group_emotions_without_emotion_style():
    """Call without pre-built emotion_style: builds it internally (line 837).
    Agent with empty turns triggers 'no data' path (lines 857-858).
    Single agent: second axis is hidden (line 885).
    """
    matplotlib.use("Agg", force=True)
    from agora.plotting import plot_group_emotions

    agg = {"AgentA": {"turns": [], "emotions": {}}}
    with patch("agora.plotting.plt.show"):
        plot_group_emotions(agg, "Public Utterances")  # no emotion_style


def test_plot_group_emotions_skips_labels_not_in_agent():
    """An emotion label present in the style but absent from the agent is skipped (line 862)."""
    matplotlib.use("Agg", force=True)
    from agora.plotting import build_emotion_style, plot_group_emotions

    agg = {
        "AgentA": {
            "turns": [1],
            "emotions": {"joy": {"mean": [0.5], "se": [0.05]}},
        }
    }
    # style contains "anger" which AgentA does not have → triggers the `continue`
    style = build_emotion_style([{
        "AgentA": {
            "turns": [1],
            "emotions": {
                "joy": {"mean": [0.5], "se": [0.05]},
                "anger": {"mean": [0.3], "se": [0.05]},
            },
        }
    }])
    with patch("agora.plotting.plt.show"):
        plot_group_emotions(agg, "Public Utterances", emotion_style=style)


def test_plot_group_survey_with_custom_slot_name():
    """A slot name that is neither Alpha nor Beta falls through _display (line 936)."""
    matplotlib.use("Agg", force=True)
    from agora.plotting import plot_group_survey

    agg = {
        "public": {
            "CustomAgent": {"Q1": {"turns": [1], "mean": [1.0], "se": [0.1]}},
        }
    }
    with patch("agora.plotting.plt.show"):
        plot_group_survey(agg)


def test_plot_group_survey_slot_with_no_q_keys():
    """A by_slot where all slot_data dicts are empty → all_q_keys is empty → early return (line 950)."""
    matplotlib.use("Agg", force=True)
    from agora.plotting import plot_group_survey

    agg = {
        "public": {"EmptySlot": {}},
    }
    with patch("agora.plotting.plt.show"):
        plot_group_survey(agg)


def test_plot_group_survey_slot_missing_panel_questions():
    """A slot whose data lacks the panel's questions skips that slot (line 979 continue)."""
    matplotlib.use("Agg", force=True)
    from agora.plotting import plot_group_survey

    agg = {
        "public": {
            "Alpha": {"Q1": {"turns": [1], "mean": [1.0], "se": [0.1]}},
            "Beta": {},  # Beta has no Q1 → panel_q_keys is empty → continue
        }
    }
    with patch("agora.plotting.plt.show"):
        plot_group_survey(agg)


# ---------------------------------------------------------------------------
# plot_group_response_decisions
# ---------------------------------------------------------------------------

def _decision_series(turns=(1, 2), mean=(0.6, 0.7), se=(0.1, 0.08)):
    return {"turns": list(turns), "mean": list(mean), "se": list(se)}


def test_plot_group_response_decisions_smoke():
    """Full two-figure smoke test with public and private data for both agents."""
    matplotlib.use("Agg", force=True)
    from agora.plotting import plot_group_response_decisions

    agg = {
        "decision_label": "ENDORSE",
        "by_slot": {
            "Alpha": {"public": _decision_series(), "private": _decision_series()},
            "Beta":  {"public": _decision_series(), "private": _decision_series()},
        },
    }
    with patch("agora.plotting.plt.show"):
        plot_group_response_decisions(agg, "PolicyDirector", "CoalitionChair")


def test_plot_group_response_decisions_empty_turns():
    """A slot with no turns still renders without error (empty subplot path)."""
    matplotlib.use("Agg", force=True)
    from agora.plotting import plot_group_response_decisions

    agg = {
        "decision_label": "YES",
        "by_slot": {
            "Alpha": {"public": {"turns": [], "mean": [], "se": []}},
        },
    }
    with patch("agora.plotting.plt.show"):
        plot_group_response_decisions(agg)


def test_plot_group_response_decisions_missing_private_channel():
    """Only public channel present — private bars are simply absent."""
    matplotlib.use("Agg", force=True)
    from agora.plotting import plot_group_response_decisions

    agg = {
        "decision_label": "ENDORSE",
        "by_slot": {
            "Alpha": {"public": _decision_series()},
            "Beta":  {"public": _decision_series()},
        },
    }
    with patch("agora.plotting.plt.show"):
        plot_group_response_decisions(agg)


# ============================================================ plot_group_response_decisions_all_repeats


def _per_repeat_series(turns=(1, 2), decisions=(1, 0)):
    return {"turns": list(turns), "decisions": list(decisions)}


def test_plot_group_response_decisions_all_repeats_smoke():
    """Full two-figure smoke with public + private data for two repeats."""
    matplotlib.use("Agg", force=True)
    from agora.plotting import plot_group_response_decisions_all_repeats

    per_repeat = {
        "decision_label": "ENDORSE",
        "repeats": [
            {
                "Alpha": {"public": _per_repeat_series(), "private": _per_repeat_series((1, 2), (0, 1))},
                "Beta":  {"public": _per_repeat_series(), "private": _per_repeat_series()},
            },
            {
                "Alpha": {"public": _per_repeat_series((1,), (1,))},
                "Beta":  {"public": _per_repeat_series((1,), (0,)), "private": _per_repeat_series()},
            },
        ],
    }
    with patch("agora.plotting.plt.show"):
        plot_group_response_decisions_all_repeats(per_repeat, "PolicyDir", "CoalitionChair")


def test_plot_group_response_decisions_all_repeats_empty_repeats():
    """No repeats at all — figures still render without error."""
    matplotlib.use("Agg", force=True)
    from agora.plotting import plot_group_response_decisions_all_repeats

    per_repeat = {"decision_label": "YES", "repeats": []}
    with patch("agora.plotting.plt.show"):
        plot_group_response_decisions_all_repeats(per_repeat)


def test_plot_group_response_decisions_all_repeats_missing_private():
    """Only public channel data present for all repeats."""
    matplotlib.use("Agg", force=True)
    from agora.plotting import plot_group_response_decisions_all_repeats

    per_repeat = {
        "decision_label": "ENDORSE",
        "repeats": [
            {
                "Alpha": {"public": _per_repeat_series()},
                "Beta":  {"public": _per_repeat_series()},
            },
        ],
    }
    with patch("agora.plotting.plt.show"):
        plot_group_response_decisions_all_repeats(per_repeat)


def test_plot_group_response_decisions_all_repeats_empty_series_in_repeat():
    """A series entry with empty turns triggers the 'if not turns: continue' branch."""
    matplotlib.use("Agg", force=True)
    from agora.plotting import plot_group_response_decisions_all_repeats

    per_repeat = {
        "decision_label": "YES",
        "repeats": [
            {
                "Alpha": {
                    "public":  {"turns": [], "decisions": []},
                    "private": _per_repeat_series(),
                },
                "Beta": {"public": _per_repeat_series()},
            },
        ],
    }
    with patch("agora.plotting.plt.show"):
        plot_group_response_decisions_all_repeats(per_repeat)

