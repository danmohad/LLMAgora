from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

from agora import cli
from agora.experiment import ExperimentConfig, ExperimentResult


class DummyAgora:
    def history(self):
        return []


class DummyAgent:
    def __init__(self, name):
        self.name = name


def _result(run_dir: Optional[Path], run_id=None):
    return ExperimentResult(
        agora=DummyAgora(),
        agents=[DummyAgent("Alpha")],
        eval_data={},
        run_dir=run_dir,
        run_id=run_id,
        semantic_analyzer=None,
        persona_adherence_eval=None,
    )


def test_build_parser_registers_run_subcommand():
    parser = cli.build_parser()
    args = parser.parse_args(["run", "--scenario-id", "s1"])
    assert args.func is cli._run
    args_incentive = parser.parse_args(
        ["run", "--scenario-id", "s1", "--incentive-direction", "none"]
    )
    assert args_incentive.incentive_direction == "none"
    args_semantic_clear = parser.parse_args(
        ["run", "--scenario-id", "s1", "--semantic-analysis-metrics"]
    )
    assert args_semantic_clear.semantic_analysis_metrics == []
    args_persona_clear = parser.parse_args(
        ["run", "--scenario-id", "s1", "--persona-analysis-metrics"]
    )
    assert args_persona_clear.persona_analysis_metrics == []
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--scenario-id", "s1", "--skip-first-agent-first-reflection"])


def test_run_uses_config_and_cli_overrides(tmp_path, monkeypatch, capsys):
    captured = {}
    cfg_from_file = ExperimentConfig(scenario_id="from-file")

    def fake_load(path):
        captured["config_path"] = path
        return cfg_from_file

    def fake_merge(base, overrides):
        captured["base"] = base
        captured["overrides"] = overrides
        return ExperimentConfig(scenario_id="override-scenario", indexed_output=True)

    def fake_run_persona_experiment(config):
        captured["final_cfg"] = config
        captured["called"] = True
        return _result(tmp_path / "outputs" / "abc123", run_id="abc123")

    monkeypatch.setattr(cli, "load_experiment_config", fake_load)
    monkeypatch.setattr(cli, "_merge_config", fake_merge)
    monkeypatch.setattr(cli, "run_persona_experiment", fake_run_persona_experiment)
    monkeypatch.setattr(cli, "print_agent_histories", lambda agents: captured.setdefault("printed", True))

    args = SimpleNamespace(
        config=tmp_path / "example.json",
        scenario_id="override-scenario",
        incentive_direction=None,
        incentive_type=None,
        prompt_set=None,
        alpha_model=None,
        beta_model=None,
        num_turns=None,
        subturn_event_order=None,
        verbose=None,
        keep_private_reflection=None,
        enable_pre_interview=None,
        keep_pre_interview=None,
        enable_post_interview=None,
        keep_post_interview=None,
        keep_public_survey=None,
        keep_private_survey=None,
        semantic_analysis_metrics=[],
        persona_analysis_metrics=[],
        persona_scoring_model=None,
        persona_scoring_verbose=None,
        persona_score_samples=None,
        save_plots=None,
        show_plots=None,
        load_snapshot=None,
        load_dir=None,
        save_snapshot=None,
        outputs_root=None,
        run_name=None,
        indexed_output=True,
        index_csv=None,
        catalog_path=None,
        prompts_path=None,
        print_histories=True,
    )

    cli._run(args)

    assert captured["config_path"] == args.config
    assert captured["base"] == asdict(cfg_from_file)
    assert captured["overrides"]["scenario_id"] == "override-scenario"
    assert captured["overrides"]["semantic_analysis_metrics"] == []
    assert captured["overrides"]["persona_analysis_metrics"] == []
    assert captured["called"] is True
    assert captured["printed"] is True

    output = capsys.readouterr().out
    assert "Run directory:" in output
    assert "Run ID: abc123" in output


def test_run_without_config_calls_build_config(tmp_path, monkeypatch):
    captured = {}

    def fake_build(payload):
        captured["payload"] = payload
        return ExperimentConfig(scenario_id="from-flags")

    def fake_run_persona_experiment(config):
        captured["cfg"] = config
        captured["called"] = True
        return _result(tmp_path / "outputs" / "named")

    monkeypatch.setattr(cli, "build_experiment_config", fake_build)
    monkeypatch.setattr(cli, "run_persona_experiment", fake_run_persona_experiment)

    args = SimpleNamespace(
        config=None,
        scenario_id="from-flags",
        incentive_direction="positive",
        incentive_type="future",
        prompt_set="default",
        alpha_model="a",
        beta_model="b",
        num_turns=2,
        subturn_event_order=["public_utterance"],
        verbose=False,
        keep_private_reflection=False,
        enable_pre_interview=False,
        keep_pre_interview=False,
        enable_post_interview=False,
        keep_post_interview=False,
        keep_public_survey=False,
        keep_private_survey=False,
        semantic_analysis_metrics=[],
        persona_analysis_metrics=[],
        persona_scoring_model="m",
        persona_scoring_verbose=False,
        persona_score_samples=1,
        save_plots=False,
        show_plots=False,
        load_snapshot=False,
        load_dir=None,
        save_snapshot=False,
        outputs_root=tmp_path / "outputs",
        run_name="demo",
        indexed_output=False,
        index_csv=tmp_path / "outputs" / "index.csv",
        catalog_path=tmp_path / "catalog.json",
        prompts_path=tmp_path / "prompts.json",
        print_histories=False,
    )

    cli._run(args)

    assert captured["payload"]["scenario_id"] == "from-flags"
    assert captured["called"] is True


def test_run_without_outputs_prints_none_directory(tmp_path, monkeypatch, capsys):
    def fake_build(_payload):
        return ExperimentConfig(scenario_id="from-flags")

    def fake_run_persona_experiment(_config):
        return _result(None)

    monkeypatch.setattr(cli, "build_experiment_config", fake_build)
    monkeypatch.setattr(cli, "run_persona_experiment", fake_run_persona_experiment)

    args = SimpleNamespace(
        config=None,
        scenario_id="from-flags",
        incentive_direction="none",
        incentive_type="historical",
        prompt_set="default",
        alpha_model="a",
        beta_model="b",
        num_turns=2,
        subturn_event_order=["public_utterance"],
        verbose=False,
        keep_private_reflection=False,
        enable_pre_interview=False,
        keep_pre_interview=False,
        enable_post_interview=False,
        keep_post_interview=False,
        keep_public_survey=False,
        keep_private_survey=False,
        semantic_analysis_metrics=[],
        persona_analysis_metrics=[],
        persona_scoring_model="m",
        persona_scoring_verbose=False,
        persona_score_samples=1,
        save_plots=False,
        show_plots=False,
        load_snapshot=False,
        load_dir=None,
        save_snapshot=False,
        outputs_root=tmp_path / "outputs",
        run_name="demo",
        indexed_output=False,
        index_csv=tmp_path / "outputs" / "index.csv",
        catalog_path=tmp_path / "catalog.json",
        prompts_path=tmp_path / "prompts.json",
        print_histories=False,
    )

    cli._run(args)
    output = capsys.readouterr().out
    assert "<none> (outputs disabled by config)" in output


def test_main_dispatches(monkeypatch):
    calls = {}

    class DummyParser:
        def parse_args(self, argv):
            calls["argv"] = argv
            return SimpleNamespace(func=lambda _: calls.setdefault("called", True))

    monkeypatch.setattr(cli, "build_parser", lambda: DummyParser())
    monkeypatch.setattr(cli, "load_dotenv", lambda: calls.setdefault("dotenv", True))

    cli.main(["run"])

    assert calls["dotenv"] is True
    assert calls["called"] is True
