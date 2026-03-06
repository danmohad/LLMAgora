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
    SEMANTIC_METRIC_SELF_CONSISTENCY,
    _merge_config,
    _plot_inter_scores,
    _plot_intra_scores,
    _prompt_set_payload,
    _resolve_run_dir,
    _resolve_index_csv,
    _scenario_entry,
    _survey_responses_by_agent,
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
    def __init__(self, turns=None):
        self._structured = {
            "event_order": ["public_utterance"],
            "pre_interviews": {
                "Alpha": {"speaker_id": "alpha", "speaker_name": "Alpha", "response": None, "keep": False},
                "Beta": {"speaker_id": "beta", "speaker_name": "Beta", "response": None, "keep": False},
            },
            "turns": list(turns or []),
            "post_interviews": {
                "Alpha": {"speaker_id": "alpha", "speaker_name": "Alpha", "response": None, "keep": False},
                "Beta": {"speaker_id": "beta", "speaker_name": "Beta", "response": None, "keep": False},
            },
        }

    def history(self):
        return []

    def structured_history(self):
        return self._structured


def _catalog_payload():
    return {
        "scenarios": [
            {
                "scenario_id": "s1",
                "question": {
                    "topic": "A topic",
                    "prompt": "Q prompt",
                },
                "survey": {
                    "direct": ["scenario survey"],
                },
                "sides": {
                    "Persona One": {
                        "id": "p1",
                        "name": "Persona One",
                        "actual_persona": "alpha",
                        "perceived_persona_base": "perceived beta",
                    },
                    "Persona Two": {
                        "id": "p2",
                        "name": "Persona Two",
                        "actual_persona": "beta",
                        "perceived_persona_base": "perceived alpha",
                    },
                },
                "incentive_modules": {
                    "positive": {
                        "historical": {
                            "views": {
                                "Persona One": "alpha pos hist",
                                "Persona Two": "beta pos hist",
                            }
                        },
                        "future": {
                            "views": {
                                "Persona One": "alpha pos future",
                                "Persona Two": "beta pos future",
                            }
                        },
                    },
                    "negative": {
                        "historical": {
                            "views": {
                                "Persona One": "alpha neg hist",
                                "Persona Two": "beta neg hist",
                            }
                        },
                        "future": {
                            "views": {
                                "Persona One": "alpha neg future",
                                "Persona Two": "beta neg future",
                            }
                        },
                    },
                },
            }
        ]
    }


def _prompt_payload():
    payload = {
        "default": {
            "base_prompt": "base",
            "perceived_prompt": "perceived",
            "incentive_prompt": " incentive={incentive}",
            "public_instruction": "public",
            "opening_instruction": "opening",
            "private_instruction": "private",
            "pre_interview_instruction": "pre",
            "post_interview_instruction": "post",
            "survey_public_prompt": "pub survey {scale}",
            "survey_private_prompt": "priv survey {scale}",
            "survey_questions": {"default": ["default survey"]},
        }
    }
    return payload


def _write_json(path: Path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_experiment_config_and_helpers(tmp_path):
    cfg = build_experiment_config(
        {
            "scenario_id": "s1",
            "model": "shared-model",
            "outputs_root": str(tmp_path / "outputs"),
            "catalog_path": str(tmp_path / "catalog.json"),
            "prompts_path": str(tmp_path / "prompts.json"),
        }
    )
    assert cfg.scenario_id == "s1"
    assert cfg.model == "shared-model"
    assert cfg.outputs_root == tmp_path / "outputs"
    assert cfg.index_csv is None
    assert cfg.load_dir is None
    assert cfg.catalog_path == tmp_path / "catalog.json"
    assert cfg.prompts_path == tmp_path / "prompts.json"
    assert cfg.subturn_event_order == ["public_utterance"]

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
    cfg_output_dir = build_experiment_config(
        {
            "scenario_id": "s1",
            "output_dir": str(tmp_path / "leaf-run"),
        }
    )
    assert cfg_output_dir.output_dir == tmp_path / "leaf-run"
    cfg_zero_turn_snapshot = build_experiment_config(
        {
            "scenario_id": "s1",
            "num_turns": 0,
            "load_snapshot": True,
            "load_dir": tmp_path / "already-path-3",
        }
    )
    assert cfg_zero_turn_snapshot.num_turns == 0
    cfg_reuse_outputs = build_experiment_config(
        {
            "scenario_id": "s1",
            "num_turns": 0,
            "load_snapshot": True,
            "load_dir": tmp_path / "already-path-4",
            "reuse_load_dir_for_outputs": True,
        }
    )
    assert cfg_reuse_outputs.reuse_load_dir_for_outputs is True

    with pytest.raises(ValueError):
        build_experiment_config({"scenario_id": "s1", "num_turns": 0})
    with pytest.raises(ValueError):
        build_experiment_config({"scenario_id": "s1", "num_turns": -1})
    with pytest.raises(ValueError):
        build_experiment_config({"scenario_id": "s1", "reuse_load_dir_for_outputs": True})
    with pytest.raises(ValueError):
        build_experiment_config(
            {
                "scenario_id": "s1",
                "num_turns": 1,
                "load_snapshot": True,
                "load_dir": tmp_path / "already-path-5",
                "reuse_load_dir_for_outputs": True,
            }
        )
    with pytest.raises(ValueError):
        build_experiment_config(
            {
                "scenario_id": "s1",
                "num_turns": 0,
                "load_snapshot": True,
                "load_dir": tmp_path / "already-path-6",
                "reuse_load_dir_for_outputs": True,
                "indexed_output": True,
            }
        )
    with pytest.raises(ValueError):
        build_experiment_config(
            {
                "scenario_id": "s1",
                "output_dir": tmp_path / "leaf-run-2",
                "indexed_output": True,
            }
        )
    with pytest.raises(ValueError):
        build_experiment_config(
            {
                "scenario_id": "s1",
                "output_dir": tmp_path / "leaf-run-3",
                "run_name": "named",
            }
        )
    with pytest.raises(ValueError):
        build_experiment_config(
            {
                "scenario_id": "s1",
                "output_dir": tmp_path / "leaf-run-4",
                "index_csv": tmp_path / "index.csv",
            }
        )
    with pytest.raises(ValueError):
        build_experiment_config(
            {
                "scenario_id": "s1",
                "output_dir": tmp_path / "leaf-run-5",
                "num_turns": 0,
                "load_snapshot": True,
                "load_dir": tmp_path / "already-path-7",
                "reuse_load_dir_for_outputs": True,
            }
        )
    with pytest.raises(ValueError):
        build_experiment_config({"scenario_id": "s1", "persona_score_samples": 0})
    with pytest.raises(ValueError):
        build_experiment_config({"scenario_id": "s1", "incentive_direction": "bad"})
    with pytest.raises(ValueError):
        build_experiment_config({"scenario_id": "s1", "incentive_type": "bad"})
    with pytest.raises(ValueError):
        build_experiment_config({"scenario_id": "s1", "show_plots": True, "save_plots": False})
    with pytest.raises(ValueError):
        build_experiment_config({"scenario_id": "s1", "semantic_similarity_method": "bad"})
    with pytest.raises(ValueError):
        build_experiment_config({"scenario_id": "s1", "semantic_similarity_device": "cuda"})
    with pytest.raises(ValueError):
        build_experiment_config({"scenario_id": "s1", "keep_private_reflection": True})
    with pytest.raises(ValueError):
        build_experiment_config({"scenario_id": "s1", "keep_public_survey": True})
    with pytest.raises(ValueError):
        build_experiment_config({"scenario_id": "s1", "keep_private_survey": True})
    with pytest.raises(ValueError):
        build_experiment_config({"scenario_id": "s1", "persona_scoring_verbose": True, "persona_analysis_metrics": []})
    with pytest.raises(ValueError):
        build_experiment_config({"scenario_id": "s1", "semantic_analysis_metrics": ["bad_metric"]})
    with pytest.raises(ValueError):
        build_experiment_config({"scenario_id": "s1", "persona_analysis_metrics": ["bad_metric"]})
    with pytest.raises(ValueError):
        build_experiment_config({"scenario_id": "s1", "semantic_analysis_metrics": ["self_consistency", "self_consistency"]})
    cfg_metric_strings = build_experiment_config(
        {
            "scenario_id": "s1",
            "semantic_analysis_metrics": "self_consistency,cross_agent_public_alignment",
            "persona_analysis_metrics": "public_per_turn,full_debate_public",
        }
    )
    assert cfg_metric_strings.semantic_analysis_metrics == [
        "self_consistency",
        "cross_agent_public_alignment",
    ]
    assert cfg_metric_strings.persona_analysis_metrics == [
        "public_per_turn",
        "full_debate_public",
    ]
    cfg_similarity_opts = build_experiment_config(
        {
            "scenario_id": "s1",
            "semantic_similarity_method": "nli",
            "semantic_similarity_model": "dleemiller/finecat-nli-l",
            "semantic_similarity_device": "mps",
        }
    )
    assert cfg_similarity_opts.semantic_similarity_method == "nli"
    assert cfg_similarity_opts.semantic_similarity_model == "dleemiller/finecat-nli-l"
    assert cfg_similarity_opts.semantic_similarity_device == "mps"
    cfg_similarity_null = build_experiment_config(
        {
            "scenario_id": "s1",
            "semantic_similarity_method": None,
        }
    )
    assert cfg_similarity_null.semantic_similarity_method is None
    with pytest.raises(ValueError):
        build_experiment_config(
            {
                "scenario_id": "s1",
                "semantic_analysis_metrics": {"bad": "value"},
            }
        )
    with pytest.raises(ValueError):
        build_experiment_config({"scenario_id": "s1", "load_snapshot": True})
    with pytest.raises(ValueError):
        build_experiment_config({"scenario_id": "s1", "load_dir": "outputs/somewhere"})
    with pytest.raises(TypeError):
        build_experiment_config({"scenario_id": "s1", "enable_private_reflection": True})
    with pytest.raises(ValueError):
        build_experiment_config({"scenario_id": "s1", "subturn_event_order": []})
    with pytest.raises(ValueError):
        build_experiment_config(
            {
                "scenario_id": "s1",
                "subturn_event_order": ["public_utterance", "public_utterance"],
            }
        )
    with pytest.raises(ValueError):
        build_experiment_config(
            {
                "scenario_id": "s1",
                "subturn_event_order": ["public_utterance", "not_real"],
            }
        )
    cfg_from_string = build_experiment_config(
        {
            "scenario_id": "s1",
            "subturn_event_order": "public_utterance,private_utterance",
        }
    )
    assert cfg_from_string.subturn_event_order == [
        "public_utterance",
        "private_utterance",
    ]
    assert cfg_from_string.enable_private_reflection is True
    cfg_private_first = build_experiment_config(
        {
            "scenario_id": "s1",
            "subturn_event_order": "private_utterance,public_utterance",
        }
    )
    assert cfg_private_first.subturn_event_order == [
        "private_utterance",
        "public_utterance",
    ]
    with pytest.raises(ValueError):
        build_experiment_config(
            {"scenario_id": "s1", "subturn_event_order": {"bad": "value"}}
        )
    with pytest.raises(ValueError):
        build_experiment_config({"scenario_id": "s1", "subturn_event_order": ["private_utterance"]})
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

    merged = _merge_config({"scenario_id": "s1", "num_turns": 2}, {"verbose": True})
    assert merged.num_turns == 2
    assert merged.verbose is True
    cleared = _merge_config(
        {"scenario_id": "s1", "incentive_direction": "positive"},
        {"incentive_direction": None},
    )
    assert cleared.incentive_direction is None

    assert _slug("  hi there  ") == "hi_there"
    assert _slug("***") == "run"

    prompts = _prompt_payload()
    assert _prompt_set_payload(prompts, "default")["public_instruction"] == "public"
    with pytest.raises(KeyError):
        _prompt_set_payload({}, "default")
    with pytest.raises(ValueError):
        _prompt_set_payload({"default": "bad"}, "default")

    catalog = _catalog_payload()
    assert _scenario_entry(catalog, "s1")["scenario_id"] == "s1"
    with pytest.raises(KeyError):
        _scenario_entry(catalog, "missing")

    cfg_readable = ExperimentConfig(scenario_id="s1", outputs_root=tmp_path / "outputs1")
    first, run_id = _resolve_run_dir(cfg_readable)
    assert run_id is None
    assert first.name == "s1_no_incentive"
    second, _ = _resolve_run_dir(cfg_readable)
    assert second.name.endswith("_2")

    cfg_indexed = ExperimentConfig(scenario_id="s1", outputs_root=tmp_path / "outputs2", indexed_output=True)
    indexed_path, indexed_id = _resolve_run_dir(cfg_indexed)
    assert indexed_id is not None
    assert indexed_path.name == indexed_id


def test_load_experiment_config_resolves_relative_paths_from_config_dir(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "run.json"
    config_path.write_text(
        json.dumps(
            {
                "scenario_id": "s1",
                "outputs_root": "../outputs",
                "index_csv": "../reports/index.csv",
                "load_snapshot": True,
                "load_dir": "../snapshots/run-1",
                "catalog_path": "../data/catalog.json",
                "prompts_path": "../data/prompts.json",
            }
        ),
        encoding="utf-8",
    )

    cfg = load_experiment_config(config_path)

    assert cfg.outputs_root == tmp_path / "outputs"
    assert cfg.index_csv == tmp_path / "reports" / "index.csv"
    assert cfg.load_dir == tmp_path / "snapshots" / "run-1"
    assert cfg.catalog_path == tmp_path / "data" / "catalog.json"
    assert cfg.prompts_path == tmp_path / "data" / "prompts.json"

    output_config_path = config_dir / "run_with_output_dir.json"
    output_config_path.write_text(
        json.dumps(
            {
                "scenario_id": "s1",
                "output_dir": "../cases/case-1",
            }
        ),
        encoding="utf-8",
    )

    output_cfg = load_experiment_config(output_config_path)

    assert output_cfg.output_dir == tmp_path / "cases" / "case-1"


def test_resolve_experiment_payload_paths_keeps_absolute_and_none_values(tmp_path):
    base_dir = tmp_path / "configs"
    absolute_prompts = (tmp_path / "shared" / "prompts.json").resolve()

    resolved = experiment.resolve_experiment_payload_paths(
        {
            "prompts_path": absolute_prompts,
            "index_csv": None,
        },
        base_dir=base_dir,
    )

    assert resolved["prompts_path"] == absolute_prompts
    assert resolved["index_csv"] is None

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
    base = readable_root / "s1_no_incentive"
    base_2 = readable_root / "s1_no_incentive_2"
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
    assert (
        _should_write_outputs(
            ExperimentConfig(
                scenario_id="s1",
                semantic_analysis_metrics=[SEMANTIC_METRIC_SELF_CONSISTENCY],
            )
        )
        is True
    )
    assert (
        _should_write_outputs(
            ExperimentConfig(scenario_id="s1", subturn_event_order=["public_utterance", "public_survey"])
        )
        is True
    )
    assert (
        _should_write_outputs(
            ExperimentConfig(scenario_id="s1", subturn_event_order=["public_utterance", "private_survey"])
        )
        is True
    )
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
    assert (
        _should_write_outputs(
            ExperimentConfig(scenario_id="s1", output_dir=Path("outputs/fixed"))
        )
        is True
    )
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


def test_survey_responses_by_agent_skips_missing_speaker_ids():
    turns = [
        {
            "turn_num": 1,
            "Alpha": {"speaker_id": "alpha", "public_survey": {"Q1": 1}},
            "Beta": {"speaker_id": None, "public_survey": {"Q1": -1}},
        }
    ]
    responses = _survey_responses_by_agent(turns, "public_survey")
    assert responses == {"alpha": {1: {"Q1": 1}}}


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
        num_turns,
        event_order,
        verbose,
        skip_first_agent_first_reflection,
        snapshot_path,
        load_snapshot_flag,
        save_snapshot_flag,
    ):
        captured["agent_configs"] = agent_configs
        captured["session_args"] = {
            "turns": num_turns,
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
        model="shared-model",
        outputs_root=tmp_path / "outputs",
        catalog_path=catalog_path,
        prompts_path=prompts_path,
        run_name="my_run",
        semantic_analysis_metrics=[],
        persona_analysis_metrics=[],
        save_plots=False,
        save_snapshot=False,
    )

    result = run_persona_experiment(cfg)

    assert result.run_dir is None

    for agent_cfg in captured["agent_configs"]:
        assert agent_cfg["private_response"]["instruction"] is None
        assert agent_cfg["pre_interview"]["instruction"] == "pre"
        assert agent_cfg["post_interview"]["instruction"] == "post"
        assert agent_cfg["survey"]["survey_questions"] == []
    assert captured["build_kwargs"]["model"] == "shared-model"
    assert captured["build_kwargs"]["pre_interview_keep"] is False
    assert captured["build_kwargs"]["post_interview_keep"] is False

    assert captured["session_args"]["save_snapshot"] is False
    assert captured["session_args"]["snapshot_path"] is None
    assert result.eval_data["semantic_similarity"]["self_consistency"] is None
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
        num_turns,
        event_order,
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
            "turns": num_turns,
        }
        return DummyAgora(), [DummyAgent("alpha", "Alpha"), DummyAgent("beta", "Beta")]

    monkeypatch.setattr(experiment, "run_debate_session", fake_run_debate_session)

    cfg = ExperimentConfig(
        scenario_id="s1",
        outputs_root=tmp_path / "outputs",
        catalog_path=catalog_path,
        prompts_path=prompts_path,
        num_turns=0,
        load_snapshot=True,
        load_dir=load_dir,
        save_snapshot=False,
    )

    result = run_persona_experiment(cfg)
    assert result.run_dir is None
    assert captured["session_args"]["snapshot_path"] == load_dir / "debate_snapshot.json"
    assert captured["session_args"]["load_snapshot"] is True
    assert captured["session_args"]["save_snapshot"] is False
    assert captured["session_args"]["turns"] == 0
    assert not (tmp_path / "outputs").exists()


def test_run_persona_experiment_writes_expected_files_when_outputs_enabled(tmp_path, monkeypatch):
    catalog_path = tmp_path / "catalog.json"
    prompts_path = tmp_path / "prompts.json"
    _write_json(catalog_path, _catalog_payload())
    _write_json(prompts_path, _prompt_payload())

    def fake_run_debate_session(
        _agent_configs,
        *,
        num_turns,
        event_order,
        verbose,
        skip_first_agent_first_reflection,
        snapshot_path,
        load_snapshot_flag,
        save_snapshot_flag,
    ):
        assert num_turns == 2
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


def test_run_persona_experiment_uses_fixed_output_dir(tmp_path, monkeypatch):
    catalog_path = tmp_path / "catalog.json"
    prompts_path = tmp_path / "prompts.json"
    output_dir = tmp_path / "cases" / "abc123def456"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(catalog_path, _catalog_payload())
    _write_json(prompts_path, _prompt_payload())

    def fake_run_debate_session(*args, **kwargs):
        return DummyAgora(), [DummyAgent("alpha", "Alpha"), DummyAgent("beta", "Beta")]

    monkeypatch.setattr(experiment, "run_debate_session", fake_run_debate_session)

    cfg = ExperimentConfig(
        scenario_id="s1",
        output_dir=output_dir,
        outputs_root=tmp_path / "unused",
        catalog_path=catalog_path,
        prompts_path=prompts_path,
    )

    result = run_persona_experiment(cfg)

    assert result.run_dir == output_dir
    assert (output_dir / "config.json").exists()
    assert not (tmp_path / "unused").exists()


def test_run_persona_experiment_derives_skip_first_from_event_order(tmp_path, monkeypatch):
    catalog_path = tmp_path / "catalog.json"
    prompts_path = tmp_path / "prompts.json"
    _write_json(catalog_path, _catalog_payload())
    _write_json(prompts_path, _prompt_payload())

    captured = {}

    def fake_run_debate_session(
        _agent_configs,
        *,
        num_turns,
        event_order,
        verbose,
        skip_first_agent_first_reflection,
        snapshot_path,
        load_snapshot_flag,
        save_snapshot_flag,
    ):
        captured["skip_first"] = skip_first_agent_first_reflection
        captured["event_order"] = list(event_order)
        return DummyAgora(), [DummyAgent("alpha", "Alpha"), DummyAgent("beta", "Beta")]

    monkeypatch.setattr(experiment, "run_debate_session", fake_run_debate_session)

    cfg = ExperimentConfig(
        scenario_id="s1",
        outputs_root=tmp_path / "outputs",
        catalog_path=catalog_path,
        prompts_path=prompts_path,
        subturn_event_order=["private_utterance", "public_utterance"],
    )
    run_persona_experiment(cfg)

    assert captured["event_order"] == ["private_utterance", "public_utterance"]
    assert captured["skip_first"] is True


def test_run_persona_experiment_writes_eval_data_only_when_enabled(tmp_path, monkeypatch):
    catalog_path = tmp_path / "catalog.json"
    prompts_path = tmp_path / "prompts.json"
    _write_json(catalog_path, _catalog_payload())
    _write_json(prompts_path, _prompt_payload())

    def fake_run_debate_session(*args, **kwargs):
        return DummyAgora(), [DummyAgent("alpha", "Alpha"), DummyAgent("beta", "Beta")]

    class FakeAnalyzer:
        def __init__(self, _turns, **_kwargs):
            pass

        def compute_self_consistency_scores(self):
            return {"Alpha": {"turns": [0], "scores": [0.5]}}

        def compute_cross_agent_alignment_scores(self, _a, _b):
            return {"turns": [0], "scores": [0.3]}

    monkeypatch.setattr(experiment, "run_debate_session", fake_run_debate_session)
    monkeypatch.setattr(experiment, "SemanticSimilarityAnalyzer", FakeAnalyzer)

    cfg = ExperimentConfig(
        scenario_id="s1",
        outputs_root=tmp_path / "outputs",
        catalog_path=catalog_path,
        prompts_path=prompts_path,
        run_name="with_eval",
        save_snapshot=True,
        semantic_analysis_metrics=[
            "self_consistency",
            "cross_agent_public_alignment",
            "cross_agent_private_alignment",
        ],
    )
    result = run_persona_experiment(cfg)
    assert result.run_dir is not None
    assert (result.run_dir / "eval_data.json").exists()
    eval_payload = json.loads((result.run_dir / "eval_data.json").read_text(encoding="utf-8"))
    assert set(eval_payload.keys()) == {
        "semantic_similarity",
        "persona_adherence",
    }
    assert eval_payload["semantic_similarity"]["self_consistency"] is not None


def test_run_persona_experiment_defaults_semantic_method_when_config_explicitly_null(
    tmp_path, monkeypatch
):
    catalog_path = tmp_path / "catalog.json"
    prompts_path = tmp_path / "prompts.json"
    _write_json(catalog_path, _catalog_payload())
    _write_json(prompts_path, _prompt_payload())
    captured = {}

    def fake_run_debate_session(*args, **kwargs):
        return DummyAgora(), [DummyAgent("alpha", "Alpha"), DummyAgent("beta", "Beta")]

    class FakeAnalyzer:
        def __init__(self, _turns, **kwargs):
            captured["method"] = kwargs["method"]

        def compute_self_consistency_scores(self):
            return {"Alpha": {"turns": [0], "scores": [0.5]}}

        def compute_cross_agent_alignment_scores(self, _a, _b):
            return {"turns": [0], "scores": [0.3]}

    monkeypatch.setattr(experiment, "run_debate_session", fake_run_debate_session)
    monkeypatch.setattr(experiment, "SemanticSimilarityAnalyzer", FakeAnalyzer)

    result = run_persona_experiment(
        {
            "scenario_id": "s1",
            "outputs_root": tmp_path / "outputs",
            "catalog_path": catalog_path,
            "prompts_path": prompts_path,
            "run_name": "null_semantic_method",
            "save_snapshot": True,
            "semantic_analysis_metrics": ["self_consistency"],
            "semantic_similarity_method": None,
        }
    )

    assert result.eval_data["semantic_similarity"]["self_consistency"] is not None
    assert captured["method"] == experiment.SEMANTIC_SIMILARITY_METHOD_COSINE


def test_run_persona_experiment_reuses_load_dir_for_outputs(tmp_path, monkeypatch):
    catalog_path = tmp_path / "catalog.json"
    prompts_path = tmp_path / "prompts.json"
    load_dir = tmp_path / "existing_run"
    load_dir.mkdir(parents=True, exist_ok=True)
    (load_dir / "config.json").write_text('{"preserve": true}', encoding="utf-8")
    _write_json(catalog_path, _catalog_payload())
    _write_json(prompts_path, _prompt_payload())

    def fake_run_debate_session(*args, **kwargs):
        return DummyAgora(), [DummyAgent("alpha", "Alpha"), DummyAgent("beta", "Beta")]

    class FakeAnalyzer:
        def __init__(self, _turns, **_kwargs):
            pass

        def compute_self_consistency_scores(self):
            return {"Alpha": {"turns": [0], "scores": [0.5]}}

        def compute_cross_agent_alignment_scores(self, _a, _b):
            return {"turns": [0], "scores": [0.3]}

    monkeypatch.setattr(experiment, "run_debate_session", fake_run_debate_session)
    monkeypatch.setattr(experiment, "SemanticSimilarityAnalyzer", FakeAnalyzer)

    cfg = ExperimentConfig(
        scenario_id="s1",
        outputs_root=tmp_path / "outputs",
        catalog_path=catalog_path,
        prompts_path=prompts_path,
        num_turns=0,
        load_snapshot=True,
        load_dir=load_dir,
        reuse_load_dir_for_outputs=True,
        semantic_analysis_metrics=["self_consistency"],
    )

    result = run_persona_experiment(cfg)
    assert result.run_dir == load_dir
    assert (load_dir / "eval_data.json").exists()
    assert (load_dir / "config.json").read_text(encoding="utf-8") == '{"preserve": true}'
    assert not (tmp_path / "outputs").exists()


def test_run_persona_experiment_private_survey_only(tmp_path, monkeypatch):
    catalog_path = tmp_path / "catalog.json"
    prompts_path = tmp_path / "prompts.json"
    _write_json(catalog_path, _catalog_payload())
    _write_json(prompts_path, _prompt_payload())

    captured = {}

    class AgoraWithPrivateSurvey(DummyAgora):
        def __init__(self):
            super().__init__(
                turns=[
                    {
                        "turn_num": 1,
                        "Alpha": {
                            "speaker_id": "alpha",
                            "speaker_name": "Alpha",
                            "public_utterance": "alpha public",
                            "private_utterance": None,
                            "public_survey": None,
                            "private_survey": {"Q1": 1},
                        },
                        "Beta": {
                            "speaker_id": "beta",
                            "speaker_name": "Beta",
                            "public_utterance": "beta public",
                            "private_utterance": None,
                            "public_survey": None,
                            "private_survey": None,
                        },
                    }
                ]
            )

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
                    "survey_questions": ["default survey", "scenario survey"],
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
                    "survey_questions": ["default survey", "scenario survey"],
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
        subturn_event_order=["public_utterance", "private_survey"],
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
    assert not (result.run_dir / "eval_data.json").exists()


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
                    "survey_questions": ["default survey", "scenario survey"],
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
                    "survey_questions": ["default survey", "scenario survey"],
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
        subturn_event_order=["public_utterance", "public_survey"],
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
            super().__init__(
                turns=[
                    {
                        "turn_num": 1,
                        "Alpha": {
                            "speaker_id": "alpha",
                            "speaker_name": "Alpha",
                            "public_utterance": "alpha public 1",
                            "private_utterance": "alpha private 1",
                            "public_survey": {"Q1": 1},
                            "private_survey": {"Q1": 1},
                        },
                        "Beta": {
                            "speaker_id": "beta",
                            "speaker_name": "Beta",
                            "public_utterance": "beta public 1",
                            "private_utterance": "beta private 1",
                            "public_survey": {"Q1": 1},
                            "private_survey": {"Q1": 1},
                        },
                    },
                    {
                        "turn_num": 2,
                        "Alpha": {
                            "speaker_id": "alpha",
                            "speaker_name": "Alpha",
                            "public_utterance": "alpha public 2",
                            "private_utterance": "alpha private 2",
                            "public_survey": {"Q1": 2},
                            "private_survey": {"Q1": 2},
                        },
                        "Beta": {
                            "speaker_id": "beta",
                            "speaker_name": "Beta",
                            "public_utterance": "beta public 2",
                            "private_utterance": "beta private 2",
                            "public_survey": {"Q1": 2},
                            "private_survey": {"Q1": 2},
                        },
                    },
                ]
            )

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
                    "survey_questions": ["default survey", "scenario survey"],
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
                    "survey_questions": ["default survey", "scenario survey"],
                    "survey_public_prompt": "sp",
                    "survey_private_prompt": "spr",
                    "public_survey_keep": True,
                },
            },
        ]

    def fake_run_debate_session(
        agent_configs,
        *,
        num_turns,
        event_order,
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
            "turns": num_turns,
            "verbose": verbose,
            "skip_first": skip_first_agent_first_reflection,
        }
        return AgoraWithSurvey(), [DummyAgent("alpha", "Alpha"), DummyAgent("beta", "Beta")]

    class FakeAnalyzer:
        def __init__(self, turns, **kwargs):
            self.turns = turns
            calls["semantic_init"] = kwargs

        def compute_self_consistency_scores(self):
            return {"Alpha": {"turns": [0], "scores": [0.5]}, "Beta": {"turns": [0], "scores": [0.4]}}

        def compute_cross_agent_alignment_scores(self, a, b):
            assert (a, b) in {
                ("public_speech", "public_speech"),
                ("private_reflection", "private_reflection"),
            }
            return {"turns": [0], "scores": [0.3]}

    class FakeEvalResult:
        def to_dict(self):
            return {
                "alpha": {
                    "public_per_turn_scores": {"turns": [1], "scores": {"mean": [4], "std": [0], "raw": [[4]]}},
                    "private_per_turn_scores": {"turns": [1], "scores": {"mean": [4], "std": [0], "raw": [[4]]}},
                    "public_cumulative_scores": {"turns": [1], "scores": {"mean": [4], "std": [0], "raw": [[4]]}},
                    "private_cumulative_scores": {"turns": [1], "scores": {"mean": [4], "std": [0], "raw": [[4]]}},
                    "full_debate_public_score": {"mean": 4, "std": 0},
                    "full_debate_private_score": {"mean": 4, "std": 0},
                    "persona_id": "p1",
                },
                "beta": {
                    "public_per_turn_scores": {"turns": [1], "scores": {"mean": [4], "std": [0], "raw": [[4]]}},
                    "private_per_turn_scores": {"turns": [1], "scores": {"mean": [4], "std": [0], "raw": [[4]]}},
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

        def evaluate_debate_from_history(self, *, memory_turns, alpha_persona_id, beta_persona_id, verbose, n_samples, metrics):
            calls["persona"] = {
                "alpha": alpha_persona_id,
                "beta": beta_persona_id,
                "verbose": verbose,
                "samples": n_samples,
                "metrics": metrics,
                "history": memory_turns,
            }
            return FakeEvalResult()

    class FakeClient:
        def close(self):
            calls["client_closed"] = True

    monkeypatch.setattr(experiment, "build_scenario_agent_configs", fake_build_scenario_agent_configs)
    monkeypatch.setattr(experiment, "run_debate_session", fake_run_debate_session)
    monkeypatch.setattr(experiment, "SemanticSimilarityAnalyzer", FakeAnalyzer)
    monkeypatch.setattr(experiment, "PersonaEvaluator", FakePersonaEvaluator)
    monkeypatch.setattr(experiment, "OpenRouterClient", FakeClient)

    cfg = ExperimentConfig(
        scenario_id="s1",
        outputs_root=tmp_path / "outputs",
        catalog_path=catalog_path,
        prompts_path=prompts_path,
        indexed_output=True,
        index_csv=None,
        incentive_direction="positive",
        incentive_type="future",
        subturn_event_order=[
            "public_utterance",
            "private_utterance",
            "public_survey",
            "private_survey",
        ],
        keep_private_reflection=True,
        keep_pre_interview=True,
        keep_post_interview=True,
        keep_public_survey=True,
        semantic_analysis_metrics=[
            "self_consistency",
            "cross_agent_public_alignment",
            "cross_agent_private_alignment",
        ],
        persona_analysis_metrics=[
            "public_per_turn",
            "private_per_turn",
            "public_cumulative",
            "private_cumulative",
            "full_debate_public",
            "full_debate_private",
        ],
        persona_scoring_model="eval-model",
        persona_scoring_verbose=True,
        persona_score_samples=3,
        save_plots=True,
        show_plots=False,
        load_snapshot=True,
        load_dir=tmp_path / "resume_from",
        save_snapshot=True,
        verbose=True,
    )

    result = run_persona_experiment(cfg)

    assert result.run_id is not None
    assert result.semantic_analyzer is not None
    assert result.persona_adherence_eval is not None
    assert calls["run_session"]["load_snapshot"] is True
    assert calls["run_session"]["save_snapshot"] is True
    assert calls["run_session"]["verbose"] is True
    assert calls["run_session"]["skip_first"] is False
    assert calls["semantic_init"]["method"] == "cosine"
    assert calls["semantic_init"]["model_name"] is None
    assert calls["semantic_init"]["device"] is None
    assert calls["persona"]["samples"] == 3
    assert calls["client_closed"] is True

    # Plots produced by enabled features
    assert (result.run_dir / "semantic_self_consistency.png").exists()
    assert (result.run_dir / "semantic_cross_agent_alignment.png").exists()
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


def test_run_persona_experiment_requires_questions_when_survey_enabled(tmp_path):
    catalog_path = tmp_path / "catalog.json"
    prompts_path = tmp_path / "prompts.json"
    catalog = _catalog_payload()
    catalog["scenarios"][0]["survey"] = {}
    prompts = _prompt_payload()
    prompts["default"]["survey_questions"] = []
    _write_json(catalog_path, catalog)
    _write_json(prompts_path, prompts)

    cfg = ExperimentConfig(
        scenario_id="s1",
        outputs_root=tmp_path / "outputs",
        catalog_path=catalog_path,
        prompts_path=prompts_path,
        subturn_event_order=["public_utterance", "public_survey"],
    )

    with pytest.raises(ValueError, match="Survey is enabled but no survey questions"):
        run_persona_experiment(cfg)


def test_run_persona_experiment_passes_incentive_selection_to_builder(
    tmp_path, monkeypatch
):
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
                "private_response": {"instruction": None, "keep": False},
                "pre_interview": {"instruction": None, "keep": False},
                "post_interview": {"instruction": None, "keep": False},
                "survey": {},
            },
            {
                "name": "Beta",
                "model": "b",
                "self_role": "beta",
                "response_instruction": "pub",
                "private_response": {"instruction": None, "keep": False},
                "pre_interview": {"instruction": None, "keep": False},
                "post_interview": {"instruction": None, "keep": False},
                "survey": {},
            },
        ]

    def fake_run_debate_session(*args, **kwargs):
        return DummyAgora(), [DummyAgent("alpha", "Alpha"), DummyAgent("beta", "Beta")]

    monkeypatch.setattr(experiment, "build_scenario_agent_configs", fake_build_scenario_agent_configs)
    monkeypatch.setattr(experiment, "run_debate_session", fake_run_debate_session)

    cfg = ExperimentConfig(
        scenario_id="s1",
        incentive_direction="negative",
        incentive_type="future",
        subturn_event_order=["public_utterance", "public_survey"],
        outputs_root=tmp_path / "outputs",
        catalog_path=catalog_path,
        prompts_path=prompts_path,
    )

    run_persona_experiment(cfg)
    assert captured["build_kwargs"]["incentive_direction"] == "negative"
    assert captured["build_kwargs"]["incentive_type"] == "future"
    assert captured["build_kwargs"]["survey_questions"] == [
        "default survey",
        "scenario survey",
    ]
    assert captured["build_kwargs"]["survey_question_groups"] == {
        "Q1": "default",
        "Q2": "direct",
    }


def test_run_persona_experiment_requires_two_sides(tmp_path):
    catalog_path = tmp_path / "catalog.json"
    prompts_path = tmp_path / "prompts.json"
    catalog = _catalog_payload()
    catalog["scenarios"][0]["sides"] = {"Persona One": catalog["scenarios"][0]["sides"]["Persona One"]}
    _write_json(catalog_path, catalog)
    _write_json(prompts_path, _prompt_payload())

    cfg = ExperimentConfig(
        scenario_id="s1",
        outputs_root=tmp_path / "outputs",
        catalog_path=catalog_path,
        prompts_path=prompts_path,
    )

    with pytest.raises(ValueError, match="exactly two sides"):
        run_persona_experiment(cfg)


def test_run_persona_experiment_requires_survey_questions(tmp_path):
    catalog_path = tmp_path / "catalog.json"
    prompts_path = tmp_path / "prompts.json"
    catalog = _catalog_payload()
    catalog["scenarios"][0]["survey"] = {}
    prompts = _prompt_payload()
    prompts["default"]["survey_questions"] = []
    _write_json(catalog_path, catalog)
    _write_json(prompts_path, prompts)

    cfg = ExperimentConfig(
        scenario_id="s1",
        outputs_root=tmp_path / "outputs",
        catalog_path=catalog_path,
        prompts_path=prompts_path,
        subturn_event_order=["public_utterance", "public_survey"],
    )

    with pytest.raises(ValueError, match="Survey is enabled but no survey questions"):
        run_persona_experiment(cfg)


def test_run_persona_experiment_falls_back_to_question_label_default(tmp_path, monkeypatch):
    catalog_path = tmp_path / "catalog.json"
    prompts_path = tmp_path / "prompts.json"
    catalog = _catalog_payload()
    del catalog["scenarios"][0]["question"]["topic"]
    _write_json(catalog_path, catalog)
    prompts = _prompt_payload()
    _write_json(prompts_path, prompts)

    class FakeAnalyzer:
        def __init__(self, _turns, **_kwargs):
            pass

        def compute_self_consistency_scores(self):
            return {"Alpha": {"turns": [0], "scores": [0.5]}}

        def compute_cross_agent_alignment_scores(self, _a, _b):
            return {"turns": [0], "scores": [0.3]}

    def fake_run_debate_session(*args, **kwargs):
        return DummyAgora(), [DummyAgent("alpha", "Alpha"), DummyAgent("beta", "Beta")]

    monkeypatch.setattr(experiment, "run_debate_session", fake_run_debate_session)
    monkeypatch.setattr(experiment, "SemanticSimilarityAnalyzer", FakeAnalyzer)

    cfg = ExperimentConfig(
        scenario_id="s1",
        outputs_root=tmp_path / "outputs",
        catalog_path=catalog_path,
        prompts_path=prompts_path,
        save_plots=True,
        semantic_analysis_metrics=["self_consistency"],
    )

    result = run_persona_experiment(cfg)
    assert (result.run_dir / "semantic_self_consistency.png").exists()
