import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import pytest

from agora import experiment
from agora.experiment import (
    DEFAULT_CATALOG_PATH,
    DEFAULT_INDEX_CSV,
    DEFAULT_OUTPUTS_ROOT,
    DEFAULT_PROMPTS_PATH,
    ExperimentConfig,
    _merge_config,
    _plot_inter_scores,
    _plot_intra_scores,
    _prompt_set_payload,
    _resolve_run_dir,
    _resolve_index_csv,
    _scenario_entry,
    _should_write_outputs,
    _slug,
    build_experiment_config,
    load_experiment_config,
    run_persona_experiment,
)


@dataclass
class DummyAgent:
    id: str
    name: str


class DummyAgora:
    def __init__(self):
        self.survey_public_response = {}
        self.survey_private_response = {}

    def history(self):
        return []


def _catalog_payload():
    return {
        "scenarios": [
            {
                "id": "s1",
                "question": {
                    "topic": "A topic",
                    "agreeable": "Q agree",
                    "controversial": "Q contro",
                },
                "surveys": {
                    "agreeable": ["scenario agreeable"],
                    "controversial": ["scenario controversial"],
                },
                "side_1": {
                    "id": "p1",
                    "name": "Persona One",
                    "actual_persona": "alpha",
                    "perceived_persona": "perceived beta",
                    "debate_arena": "arena1",
                },
                "side_2": {
                    "id": "p2",
                    "name": "Persona Two",
                    "actual_persona": "beta",
                    "perceived_persona": "perceived alpha",
                    "debate_arena": "arena2",
                },
            }
        ]
    }


def _prompt_payload(with_neutral=True):
    payload = {
        "default": {
            "base_prompt": "base",
            "perceived_prompt": "perceived",
            "debate_arena_prompt": "arena {debate_arena}",
            "public_instruction": "public",
            "opening_instruction": "opening",
            "private_instruction": "private",
            "pre_interview_instruction": "pre",
            "post_interview_instruction": "post",
            "survey_public_prompt": "pub survey",
            "survey_private_prompt": "priv survey",
            "survey_questions": ["default survey"],
        }
    }
    if with_neutral:
        payload["default"]["neutral_arena_prompt"] = "neutral"
    return payload


def _write_json(path: Path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_experiment_config_and_helpers(tmp_path):
    cfg = build_experiment_config(
        {
            "scenario_id": "s1",
            "outputs_root": str(tmp_path / "outputs"),
            "catalog_path": str(tmp_path / "catalog.json"),
            "prompts_path": str(tmp_path / "prompts.json"),
        }
    )
    assert cfg.scenario_id == "s1"
    assert cfg.outputs_root == tmp_path / "outputs"
    assert cfg.index_csv is None
    assert cfg.load_dir is None
    assert cfg.catalog_path == tmp_path / "catalog.json"
    assert cfg.prompts_path == tmp_path / "prompts.json"

    cfg_paths = build_experiment_config(
        {
            "scenario_id": "s1",
            "outputs_root": tmp_path / "already-path",
            "index_csv": str(tmp_path / "already-path" / "index.csv"),
            "load_snapshot": True,
            "load_dir": str(tmp_path / "already-path" / "from-run"),
            "catalog_path": tmp_path / "already-path" / "catalog.json",
            "prompts_path": tmp_path / "already-path" / "prompts.json",
        }
    )
    assert cfg_paths.outputs_root == tmp_path / "already-path"
    assert cfg_paths.index_csv == tmp_path / "already-path" / "index.csv"
    assert cfg_paths.load_dir == tmp_path / "already-path" / "from-run"

    cfg_paths_obj = build_experiment_config(
        {
            "scenario_id": "s1",
            "index_csv": tmp_path / "already-path-2" / "index.csv",
        }
    )
    assert cfg_paths_obj.index_csv == tmp_path / "already-path-2" / "index.csv"

    with pytest.raises(ValueError):
        build_experiment_config({"scenario_id": "s1", "turns_per_agent": 0})
    with pytest.raises(ValueError):
        build_experiment_config({"scenario_id": "s1", "persona_n_samples": 0})
    with pytest.raises(ValueError):
        build_experiment_config({"scenario_id": "s1", "side_order": "bad"})
    with pytest.raises(ValueError):
        build_experiment_config({"scenario_id": "s1", "question_variant": "bad"})
    with pytest.raises(ValueError):
        build_experiment_config({"scenario_id": "s1", "show_plots": True, "save_plots": False})
    with pytest.raises(ValueError):
        build_experiment_config({"scenario_id": "s1", "keep_public_survey": True, "enable_public_survey": False})
    with pytest.raises(ValueError):
        build_experiment_config({"scenario_id": "s1", "keep_private_survey": True, "enable_private_survey": False})
    with pytest.raises(ValueError):
        build_experiment_config({"scenario_id": "s1", "persona_eval_verbose": True, "enable_persona_evaluation": False})
    with pytest.raises(ValueError):
        build_experiment_config({"scenario_id": "s1", "load_snapshot": True})
    with pytest.raises(ValueError):
        build_experiment_config({"scenario_id": "s1", "load_dir": "outputs/somewhere"})
    with pytest.raises(ValueError):
        build_experiment_config({})

    payload_file = tmp_path / "cfg.json"
    _write_json(payload_file, {"scenario_id": "s1"})
    loaded = load_experiment_config(payload_file)
    assert loaded.scenario_id == "s1"

    bad_file = tmp_path / "bad.json"
    _write_json(bad_file, [1, 2, 3])
    with pytest.raises(ValueError):
        load_experiment_config(bad_file)

    merged = _merge_config({"scenario_id": "s1", "turns_per_agent": 2}, {"turns_per_agent": None, "verbose": True})
    assert merged.turns_per_agent == 2
    assert merged.verbose is True

    assert _slug("  hi there  ") == "hi_there"
    assert _slug("***") == "run"

    prompts = _prompt_payload()
    assert _prompt_set_payload(prompts, "default")["public_instruction"] == "public"
    with pytest.raises(KeyError):
        _prompt_set_payload({}, "default")
    with pytest.raises(ValueError):
        _prompt_set_payload({"default": "bad"}, "default")

    catalog = _catalog_payload()
    assert _scenario_entry(catalog, "s1")["id"] == "s1"
    with pytest.raises(KeyError):
        _scenario_entry(catalog, "missing")

    cfg_readable = ExperimentConfig(scenario_id="s1", outputs_root=tmp_path / "outputs1")
    first, run_id = _resolve_run_dir(cfg_readable)
    assert run_id is None
    assert first.name == "s1_controversial_12_biased"
    second, _ = _resolve_run_dir(cfg_readable)
    assert second.name.endswith("_2")

    cfg_indexed = ExperimentConfig(scenario_id="s1", outputs_root=tmp_path / "outputs2", indexed_output=True)
    indexed_path, indexed_id = _resolve_run_dir(cfg_indexed)
    assert indexed_id is not None
    assert indexed_path.name == indexed_id

    # Force indexed collision path to cover retry loop.
    class FakeUUID:
        def __init__(self, value):
            self.hex = value

    collision_root = tmp_path / "outputs-collision"
    collision_root.mkdir(parents=True, exist_ok=True)
    (collision_root / "aaaaaa").mkdir()
    uuid_values = iter([FakeUUID("aaaaaa111"), FakeUUID("bbbbbb222")])
    monkey = pytest.MonkeyPatch()
    monkey.setattr(experiment, "uuid4", lambda: next(uuid_values))
    try:
        _, collision_id = _resolve_run_dir(
            ExperimentConfig(scenario_id="s1", outputs_root=collision_root, indexed_output=True)
        )
    finally:
        monkey.undo()
    assert collision_id == "bbbbbb"

    # Force readable-name collision to increment beyond _2.
    readable_root = tmp_path / "outputs-readable-collision"
    base = readable_root / "s1_controversial_12_biased"
    base_2 = readable_root / "s1_controversial_12_biased_2"
    base.mkdir(parents=True, exist_ok=True)
    base_2.mkdir(parents=True, exist_ok=True)
    resolved, _ = _resolve_run_dir(ExperimentConfig(scenario_id="s1", outputs_root=readable_root))
    assert resolved.name.endswith("_3")


def test_defaults_constants():
    assert DEFAULT_OUTPUTS_ROOT == Path("outputs")
    assert DEFAULT_INDEX_CSV == Path("outputs/index.csv")
    assert DEFAULT_CATALOG_PATH == Path("data/scenarios.json")
    assert DEFAULT_PROMPTS_PATH == Path("data/prompts.json")
    assert _should_write_outputs(ExperimentConfig(scenario_id="s1")) is False
    assert _should_write_outputs(ExperimentConfig(scenario_id="s1", enable_analyzer=True)) is True
    assert _should_write_outputs(ExperimentConfig(scenario_id="s1", enable_public_survey=True)) is True
    assert _should_write_outputs(ExperimentConfig(scenario_id="s1", enable_private_survey=True)) is True
    assert (
        _should_write_outputs(
            ExperimentConfig(
                scenario_id="s1",
                load_snapshot=True,
                load_dir=Path("outputs/existing"),
            )
        )
        is False
    )
    assert _should_write_outputs(ExperimentConfig(scenario_id="s1", indexed_output=True)) is True
    assert _resolve_index_csv(ExperimentConfig(scenario_id="s1")) == Path("outputs/index.csv")
    assert _resolve_index_csv(
        ExperimentConfig(scenario_id="s1", index_csv=Path("custom/index.csv"))
    ) == Path("custom/index.csv")


def test_plot_helpers_show_branch(tmp_path, monkeypatch):
    matplotlib.use("Agg", force=True)
    calls = {"show": 0}
    monkeypatch.setattr(experiment.plt, "show", lambda: calls.__setitem__("show", calls["show"] + 1))

    intra = {"Alpha": {"turns": [0, 1], "scores": [0.5, 0.6]}}
    _plot_intra_scores(intra, {"Alpha": "Alpha"}, tmp_path / "intra.png", "title", show_plot=True)
    assert (tmp_path / "intra.png").exists()

    inter = {"turns": [0, 1], "scores": [0.3, 0.4]}
    _plot_inter_scores(inter, inter, tmp_path / "inter.png", "title", show_plot=True)
    assert (tmp_path / "inter.png").exists()
    assert calls["show"] == 2


def test_run_persona_experiment_collapses_optional_features(tmp_path, monkeypatch):
    catalog_path = tmp_path / "catalog.json"
    prompts_path = tmp_path / "prompts.json"
    _write_json(catalog_path, _catalog_payload())
    _write_json(prompts_path, _prompt_payload())

    captured = {}

    def fake_build_scenario_agent_configs(**kwargs):
        captured["build_kwargs"] = kwargs
        return [
            {
                "name": "Alpha",
                "model": "a",
                "self_role": "alpha",
                "response_instruction": "pub",
                "private_response": {"instruction": "private", "keep": True},
                "pre_interview": {"instruction": "pre", "keep": True},
                "post_interview": {"instruction": "post", "keep": True},
                "survey": {
                    "survey_questions": ["q"],
                    "survey_public_prompt": "sp",
                    "survey_private_prompt": "spr",
                    "public_survey_keep": True,
                },
            },
            {
                "name": "Beta",
                "model": "b",
                "self_role": "beta",
                "response_instruction": "pub",
                "private_response": {"instruction": "private", "keep": True},
                "pre_interview": {"instruction": "pre", "keep": True},
                "post_interview": {"instruction": "post", "keep": True},
                "survey": {
                    "survey_questions": ["q"],
                    "survey_public_prompt": "sp",
                    "survey_private_prompt": "spr",
                    "public_survey_keep": True,
                },
            },
        ]

    def fake_run_debate_session(
        agent_configs,
        *,
        turns_per_agent,
        verbose,
        skip_first_agent_first_reflection,
        snapshot_path,
        load_snapshot_flag,
        save_snapshot_flag,
    ):
        captured["agent_configs"] = agent_configs
        captured["session_args"] = {
            "turns": turns_per_agent,
            "verbose": verbose,
            "skip_first": skip_first_agent_first_reflection,
            "snapshot_path": snapshot_path,
            "load_snapshot": load_snapshot_flag,
            "save_snapshot": save_snapshot_flag,
        }
        return DummyAgora(), [DummyAgent("alpha", "Alpha"), DummyAgent("beta", "Beta")]

    monkeypatch.setattr(experiment, "build_scenario_agent_configs", fake_build_scenario_agent_configs)
    monkeypatch.setattr(experiment, "run_debate_session", fake_run_debate_session)

    cfg = ExperimentConfig(
        scenario_id="s1",
        outputs_root=tmp_path / "outputs",
        catalog_path=catalog_path,
        prompts_path=prompts_path,
        run_name="my_run",
        enable_private_reflection=False,
        enable_pre_interview=False,
        enable_post_interview=False,
        enable_public_survey=False,
        enable_private_survey=False,
        enable_analyzer=False,
        enable_persona_evaluation=False,
        save_plots=False,
        save_snapshot=False,
    )

    result = run_persona_experiment(cfg)

    assert result.run_dir is None

    for agent_cfg in captured["agent_configs"]:
        assert agent_cfg["private_response"]["instruction"] is None
        assert agent_cfg["pre_interview"]["instruction"] is None
        assert agent_cfg["post_interview"]["instruction"] is None
        assert agent_cfg["survey"]["survey_questions"] == []

    assert captured["session_args"]["save_snapshot"] is False
    assert captured["session_args"]["snapshot_path"] is None
    assert result.eval_data["intra_agent_honesty"] is None
    assert result.eval_data["persona_adherence"] is None
    assert not (tmp_path / "outputs").exists()


def test_run_persona_experiment_loads_snapshot_from_load_dir_without_output_dir(tmp_path, monkeypatch):
    catalog_path = tmp_path / "catalog.json"
    prompts_path = tmp_path / "prompts.json"
    load_dir = tmp_path / "resume_here"
    load_dir.mkdir(parents=True, exist_ok=True)
    _write_json(catalog_path, _catalog_payload())
    _write_json(prompts_path, _prompt_payload())

    captured = {}

    def fake_run_debate_session(
        agent_configs,
        *,
        turns_per_agent,
        verbose,
        skip_first_agent_first_reflection,
        snapshot_path,
        load_snapshot_flag,
        save_snapshot_flag,
    ):
        captured["session_args"] = {
            "snapshot_path": snapshot_path,
            "load_snapshot": load_snapshot_flag,
            "save_snapshot": save_snapshot_flag,
            "turns": turns_per_agent,
        }
        return DummyAgora(), [DummyAgent("alpha", "Alpha"), DummyAgent("beta", "Beta")]

    monkeypatch.setattr(experiment, "run_debate_session", fake_run_debate_session)

    cfg = ExperimentConfig(
        scenario_id="s1",
        outputs_root=tmp_path / "outputs",
        catalog_path=catalog_path,
        prompts_path=prompts_path,
        load_snapshot=True,
        load_dir=load_dir,
        save_snapshot=False,
    )

    result = run_persona_experiment(cfg)
    assert result.run_dir is None
    assert captured["session_args"]["snapshot_path"] == load_dir / "debate_snapshot.json"
    assert captured["session_args"]["load_snapshot"] is True
    assert captured["session_args"]["save_snapshot"] is False
    assert not (tmp_path / "outputs").exists()


def test_run_persona_experiment_writes_expected_files_when_outputs_enabled(tmp_path, monkeypatch):
    catalog_path = tmp_path / "catalog.json"
    prompts_path = tmp_path / "prompts.json"
    _write_json(catalog_path, _catalog_payload())
    _write_json(prompts_path, _prompt_payload())

    def fake_run_debate_session(
        _agent_configs,
        *,
        turns_per_agent,
        verbose,
        skip_first_agent_first_reflection,
        snapshot_path,
        load_snapshot_flag,
        save_snapshot_flag,
    ):
        assert turns_per_agent == 2
        assert verbose is False
        assert skip_first_agent_first_reflection is False
        assert load_snapshot_flag is False
        assert save_snapshot_flag is True
        assert snapshot_path is not None
        return DummyAgora(), [DummyAgent("alpha", "Alpha"), DummyAgent("beta", "Beta")]

    monkeypatch.setattr(experiment, "run_debate_session", fake_run_debate_session)

    cfg = ExperimentConfig(
        scenario_id="s1",
        outputs_root=tmp_path / "outputs",
        catalog_path=catalog_path,
        prompts_path=prompts_path,
        run_name="my_run",
        save_snapshot=True,
    )

    result = run_persona_experiment(cfg)
    assert result.run_dir is not None
    assert result.run_dir.name == "my_run"
    assert (result.run_dir / "config.json").exists()
    assert not (result.run_dir / "eval_data.json").exists()
    assert not (result.run_dir / "scenarios.json").exists()
    assert not (result.run_dir / "prompts.json").exists()
    assert not (result.run_dir / "run_metadata.json").exists()

def test_run_persona_experiment_writes_eval_data_only_when_enabled(tmp_path, monkeypatch):
    catalog_path = tmp_path / "catalog.json"
    prompts_path = tmp_path / "prompts.json"
    _write_json(catalog_path, _catalog_payload())
    _write_json(prompts_path, _prompt_payload())

    def fake_run_debate_session(*args, **kwargs):
        return DummyAgora(), [DummyAgent("alpha", "Alpha"), DummyAgent("beta", "Beta")]

    class FakeAnalyzer:
        def __init__(self, _turns):
            pass

        def compute_intra_agent_honesty(self):
            return {"Alpha": {"turns": [0], "scores": [0.5]}}

        def compute_inter_agent_alignment(self, _a, _b):
            return {"turns": [0], "scores": [0.3]}

    monkeypatch.setattr(experiment, "run_debate_session", fake_run_debate_session)
    monkeypatch.setattr(experiment, "DebateAnalyzer", FakeAnalyzer)

    cfg = ExperimentConfig(
        scenario_id="s1",
        outputs_root=tmp_path / "outputs",
        catalog_path=catalog_path,
        prompts_path=prompts_path,
        run_name="with_eval",
        save_snapshot=True,
        enable_analyzer=True,
    )
    result = run_persona_experiment(cfg)
    assert result.run_dir is not None
    assert (result.run_dir / "eval_data.json").exists()
    eval_payload = json.loads((result.run_dir / "eval_data.json").read_text(encoding="utf-8"))
    assert eval_payload["intra_agent_honesty"] is not None


def test_run_persona_experiment_private_survey_only(tmp_path, monkeypatch):
    catalog_path = tmp_path / "catalog.json"
    prompts_path = tmp_path / "prompts.json"
    _write_json(catalog_path, _catalog_payload())
    _write_json(prompts_path, _prompt_payload())

    captured = {}

    class AgoraWithPrivateSurvey(DummyAgora):
        def __init__(self):
            super().__init__()
            self.survey_private_response = {"alpha": {0: {"Q1": 1}}}

    def fake_build_scenario_agent_configs(**kwargs):
        captured["build_kwargs"] = kwargs
        return [
            {
                "name": "Alpha",
                "model": "a",
                "self_role": "alpha",
                "response_instruction": "pub",
                "private_response": {"instruction": "private", "keep": True},
                "pre_interview": {"instruction": "pre", "keep": True},
                "post_interview": {"instruction": "post", "keep": True},
                "survey": {
                    "survey_questions": ["default survey", "scenario controversial"],
                    "survey_public_prompt": "sp",
                    "survey_private_prompt": "spr",
                    "public_survey_keep": True,
                },
            },
            {
                "name": "Beta",
                "model": "b",
                "self_role": "beta",
                "response_instruction": "pub",
                "private_response": {"instruction": "private", "keep": True},
                "pre_interview": {"instruction": "pre", "keep": True},
                "post_interview": {"instruction": "post", "keep": True},
                "survey": {
                    "survey_questions": ["default survey", "scenario controversial"],
                    "survey_public_prompt": "sp",
                    "survey_private_prompt": "spr",
                    "public_survey_keep": True,
                },
            },
        ]

    def fake_run_debate_session(agent_configs, **_kwargs):
        captured["agent_configs"] = agent_configs
        return AgoraWithPrivateSurvey(), [DummyAgent("alpha", "Alpha"), DummyAgent("beta", "Beta")]

    monkeypatch.setattr(experiment, "build_scenario_agent_configs", fake_build_scenario_agent_configs)
    monkeypatch.setattr(experiment, "run_debate_session", fake_run_debate_session)

    cfg = ExperimentConfig(
        scenario_id="s1",
        outputs_root=tmp_path / "outputs",
        catalog_path=catalog_path,
        prompts_path=prompts_path,
        run_name="private_only",
        enable_public_survey=False,
        enable_private_survey=True,
        keep_public_survey=False,
        save_plots=True,
    )

    result = run_persona_experiment(cfg)

    assert result.run_dir is not None
    for agent_cfg in captured["agent_configs"]:
        assert agent_cfg["survey"]["enable_public_survey"] is False
        assert agent_cfg["survey"]["enable_private_survey"] is True
        assert agent_cfg["survey"]["survey_public_prompt"] is None
        assert agent_cfg["survey"]["public_survey_keep"] is False
    assert (result.run_dir / "private_survey.png").exists()
    assert not (result.run_dir / "public_survey.png").exists()
    assert (result.run_dir / "eval_data.json").exists()


def test_run_persona_experiment_public_survey_only(tmp_path, monkeypatch):
    catalog_path = tmp_path / "catalog.json"
    prompts_path = tmp_path / "prompts.json"
    _write_json(catalog_path, _catalog_payload())
    _write_json(prompts_path, _prompt_payload())

    captured = {}

    def fake_build_scenario_agent_configs(**kwargs):
        captured["build_kwargs"] = kwargs
        return [
            {
                "name": "Alpha",
                "model": "a",
                "self_role": "alpha",
                "response_instruction": "pub",
                "private_response": {"instruction": "private", "keep": True},
                "pre_interview": {"instruction": "pre", "keep": True},
                "post_interview": {"instruction": "post", "keep": True},
                "survey": {
                    "survey_questions": ["default survey", "scenario controversial"],
                    "survey_public_prompt": "sp",
                    "survey_private_prompt": "spr",
                    "public_survey_keep": True,
                },
            },
            {
                "name": "Beta",
                "model": "b",
                "self_role": "beta",
                "response_instruction": "pub",
                "private_response": {"instruction": "private", "keep": True},
                "pre_interview": {"instruction": "pre", "keep": True},
                "post_interview": {"instruction": "post", "keep": True},
                "survey": {
                    "survey_questions": ["default survey", "scenario controversial"],
                    "survey_public_prompt": "sp",
                    "survey_private_prompt": "spr",
                    "public_survey_keep": True,
                },
            },
        ]

    def fake_run_debate_session(agent_configs, **_kwargs):
        captured["agent_configs"] = agent_configs
        return DummyAgora(), [DummyAgent("alpha", "Alpha"), DummyAgent("beta", "Beta")]

    monkeypatch.setattr(experiment, "build_scenario_agent_configs", fake_build_scenario_agent_configs)
    monkeypatch.setattr(experiment, "run_debate_session", fake_run_debate_session)

    cfg = ExperimentConfig(
        scenario_id="s1",
        outputs_root=tmp_path / "outputs",
        catalog_path=catalog_path,
        prompts_path=prompts_path,
        run_name="public_only",
        enable_public_survey=True,
        enable_private_survey=False,
        keep_public_survey=True,
    )

    run_persona_experiment(cfg)

    for agent_cfg in captured["agent_configs"]:
        assert agent_cfg["survey"]["enable_public_survey"] is True
        assert agent_cfg["survey"]["enable_private_survey"] is False
        assert agent_cfg["survey"]["survey_private_prompt"] is None


def test_run_persona_experiment_with_all_features_and_indexed_output(tmp_path, monkeypatch):
    matplotlib.use("Agg", force=True)

    catalog_path = tmp_path / "catalog.json"
    prompts_path = tmp_path / "prompts.json"
    _write_json(catalog_path, _catalog_payload())
    _write_json(prompts_path, _prompt_payload())

    class AgoraWithSurvey(DummyAgora):
        def __init__(self):
            super().__init__()
            self.survey_public_response = {"alpha": {0: {"Q1": 1}, 1: {"Q1": 2}}}
            self.survey_private_response = {"alpha": {0: {"Q1": 1}, 1: {"Q1": 2}}}

    calls = {"run_session": None, "persona": None, "client_closed": False}

    def fake_build_scenario_agent_configs(**kwargs):
        return [
            {
                "name": "Alpha",
                "model": "a",
                "self_role": "alpha",
                "response_instruction": "pub",
                "private_response": {"instruction": "private", "keep": True},
                "pre_interview": {"instruction": "pre", "keep": True},
                "post_interview": {"instruction": "post", "keep": True},
                "survey": {
                    "survey_questions": ["default survey", "scenario controversial"],
                    "survey_public_prompt": "sp",
                    "survey_private_prompt": "spr",
                    "public_survey_keep": True,
                },
            },
            {
                "name": "Beta",
                "model": "b",
                "self_role": "beta",
                "response_instruction": "pub",
                "private_response": {"instruction": "private", "keep": True},
                "pre_interview": {"instruction": "pre", "keep": True},
                "post_interview": {"instruction": "post", "keep": True},
                "survey": {
                    "survey_questions": ["default survey", "scenario controversial"],
                    "survey_public_prompt": "sp",
                    "survey_private_prompt": "spr",
                    "public_survey_keep": True,
                },
            },
        ]

    def fake_run_debate_session(
        agent_configs,
        *,
        turns_per_agent,
        verbose,
        skip_first_agent_first_reflection,
        snapshot_path,
        load_snapshot_flag,
        save_snapshot_flag,
    ):
        calls["run_session"] = {
            "load_snapshot": load_snapshot_flag,
            "save_snapshot": save_snapshot_flag,
            "snapshot_path": snapshot_path,
            "turns": turns_per_agent,
            "verbose": verbose,
            "skip_first": skip_first_agent_first_reflection,
        }
        return AgoraWithSurvey(), [DummyAgent("alpha", "Alpha"), DummyAgent("beta", "Beta")]

    class FakeAnalyzer:
        def __init__(self, turns):
            self.turns = turns

        def compute_intra_agent_honesty(self):
            return {"Alpha": {"turns": [0], "scores": [0.5]}, "Beta": {"turns": [0], "scores": [0.4]}}

        def compute_inter_agent_alignment(self, a, b):
            assert (a, b) in {
                ("public_speech", "public_speech"),
                ("private_reflection", "private_reflection"),
            }
            return {"turns": [0], "scores": [0.3]}

    class FakeEvalResult:
        def to_dict(self):
            return {
                "alpha": {
                    "public_turn_scores": {"turns": [1], "scores": {"mean": [4], "std": [0], "raw": [[4]]}},
                    "private_turn_scores": {"turns": [1], "scores": {"mean": [4], "std": [0], "raw": [[4]]}},
                    "public_cumulative_scores": {"turns": [1], "scores": {"mean": [4], "std": [0], "raw": [[4]]}},
                    "private_cumulative_scores": {"turns": [1], "scores": {"mean": [4], "std": [0], "raw": [[4]]}},
                    "full_debate_public_score": {"mean": 4, "std": 0},
                    "full_debate_private_score": {"mean": 4, "std": 0},
                    "persona_id": "p1",
                },
                "beta": {
                    "public_turn_scores": {"turns": [1], "scores": {"mean": [4], "std": [0], "raw": [[4]]}},
                    "private_turn_scores": {"turns": [1], "scores": {"mean": [4], "std": [0], "raw": [[4]]}},
                    "public_cumulative_scores": {"turns": [1], "scores": {"mean": [4], "std": [0], "raw": [[4]]}},
                    "private_cumulative_scores": {"turns": [1], "scores": {"mean": [4], "std": [0], "raw": [[4]]}},
                    "full_debate_public_score": {"mean": 4, "std": 0},
                    "full_debate_private_score": {"mean": 4, "std": 0},
                    "persona_id": "p2",
                },
            }

    class FakePersonaEvaluator:
        def __init__(self, llm_client, personas, model):
            calls["persona_init"] = {"personas": personas, "model": model}

        def evaluate_debate_from_history(self, *, memory_turns, alpha_persona_id, beta_persona_id, verbose, n_samples):
            calls["persona"] = {
                "alpha": alpha_persona_id,
                "beta": beta_persona_id,
                "verbose": verbose,
                "samples": n_samples,
                "history": memory_turns,
            }
            return FakeEvalResult()

    class FakeClient:
        def close(self):
            calls["client_closed"] = True

    monkeypatch.setattr(experiment, "build_scenario_agent_configs", fake_build_scenario_agent_configs)
    monkeypatch.setattr(experiment, "run_debate_session", fake_run_debate_session)
    monkeypatch.setattr(experiment, "DebateAnalyzer", FakeAnalyzer)
    monkeypatch.setattr(experiment, "PersonaEvaluator", FakePersonaEvaluator)
    monkeypatch.setattr(experiment, "OpenRouterClient", FakeClient)

    cfg = ExperimentConfig(
        scenario_id="s1",
        outputs_root=tmp_path / "outputs",
        catalog_path=catalog_path,
        prompts_path=prompts_path,
        indexed_output=True,
        index_csv=None,
        use_neutral_arena=True,
        enable_private_reflection=True,
        keep_private_reflection=True,
        enable_pre_interview=True,
        keep_pre_interview=True,
        enable_post_interview=True,
        keep_post_interview=True,
        enable_public_survey=True,
        enable_private_survey=True,
        keep_public_survey=True,
        enable_analyzer=True,
        enable_persona_evaluation=True,
        persona_eval_model="eval-model",
        persona_eval_verbose=True,
        persona_n_samples=3,
        save_plots=True,
        show_plots=False,
        load_snapshot=True,
        load_dir=tmp_path / "resume_from",
        save_snapshot=True,
        verbose=True,
        skip_first_agent_first_reflection=True,
    )

    result = run_persona_experiment(cfg)

    assert result.run_id is not None
    assert result.analyzer is not None
    assert result.persona_eval is not None
    assert calls["run_session"]["load_snapshot"] is True
    assert calls["run_session"]["save_snapshot"] is True
    assert calls["run_session"]["verbose"] is True
    assert calls["run_session"]["skip_first"] is True
    assert calls["persona"]["samples"] == 3
    assert calls["client_closed"] is True

    # Plots produced by enabled features
    assert (result.run_dir / "intra_agent.png").exists()
    assert (result.run_dir / "inter_agent.png").exists()
    assert (result.run_dir / "persona_adherence.png").exists()
    assert (result.run_dir / "public_survey.png").exists()
    assert (result.run_dir / "private_survey.png").exists()

    # Indexed CSV row written
    with (tmp_path / "outputs" / "index.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["run_id"] == result.run_id
    assert rows[0]["scenario_id"] == "s1"

    # Second run appends a row (covers append branch)
    run_persona_experiment(cfg)
    with (tmp_path / "outputs" / "index.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2


def test_run_persona_experiment_requires_neutral_prompt_when_enabled(tmp_path):
    catalog_path = tmp_path / "catalog.json"
    prompts_path = tmp_path / "prompts.json"
    _write_json(catalog_path, _catalog_payload())
    _write_json(prompts_path, _prompt_payload(with_neutral=False))

    cfg = ExperimentConfig(
        scenario_id="s1",
        outputs_root=tmp_path / "outputs",
        catalog_path=catalog_path,
        prompts_path=prompts_path,
        use_neutral_arena=True,
    )

    with pytest.raises(KeyError):
        run_persona_experiment(cfg)
