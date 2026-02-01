from types import SimpleNamespace

import matplotlib

from agora.plotting import plot_survey_responses


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
    questions = [
        "This is a long survey question that should be truncated.",
        "Short 2",
        "Short 3",
        "Short 4",
        "Short 5",
        "Short 6",
    ]
    plot_survey_responses(responses, agents, questions, "Survey", output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0
