from types import SimpleNamespace

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
