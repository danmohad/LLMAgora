import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agora.sweep_results import ExperimentGroup, SweepCase, SweepManifest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manifest(tmp_path: Path, *, cases: list[dict], notes: str | None = "test run") -> Path:
    manifest = {
        "schema_version": 1,
        "generated_at": "2026-01-01T00:00:00+00:00",
        "sweep_root": str(tmp_path),
        "runner_defaults": {"max_parallel_jobs": 4, "stop_on_error": False},
        "number_of_repeats": 2,
        "notes": notes,
        "total_cases": len(cases),
        "cases": cases,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _case(fp: str, repeat: int, repeat_count: int, model: str, case_id: str) -> dict:
    return {
        "case_id": case_id,
        "case_dir": f"cases/{case_id}",
        "config_path": f"cases/{case_id}/config.json",
        "label": f'model="{model}" (repeat {repeat}/{repeat_count})',
        "repeat_number": repeat,
        "repeat_count": repeat_count,
        "sweep_values": {"model": model},
        "config_fingerprint": fp,
    }


FP_A = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
FP_B = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

TWO_MODELS = [
    _case(FP_A, 1, 2, "openai/gpt-x", "case001"),
    _case(FP_A, 2, 2, "openai/gpt-x", "case002"),
    _case(FP_B, 1, 2, "anthropic/claude-y", "case003"),
    _case(FP_B, 2, 2, "anthropic/claude-y", "case004"),
]


# ---------------------------------------------------------------------------
# SweepCase
# ---------------------------------------------------------------------------

def test_sweep_case_fields():
    sc = SweepCase(
        case_id="abc123",
        case_dir=Path("cases/abc123"),
        config_path=Path("cases/abc123/config.json"),
        label="model=\"x\" (repeat 1/2)",
        repeat_number=1,
        repeat_count=2,
        sweep_values={"model": "x"},
    )
    assert sc.case_id == "abc123"
    assert sc.case_dir == Path("cases/abc123")
    assert sc.config_path == Path("cases/abc123/config.json")
    assert sc.repeat_number == 1
    assert sc.repeat_count == 2
    assert sc.sweep_values == {"model": "x"}


# ---------------------------------------------------------------------------
# ExperimentGroup
# ---------------------------------------------------------------------------

def test_experiment_group_repeat_count_and_case_ids():
    cases = [
        SweepCase("c1", Path("cases/c1"), Path("cases/c1/config.json"), "lbl1", 1, 2, {"m": "a"}),
        SweepCase("c2", Path("cases/c2"), Path("cases/c2/config.json"), "lbl2", 2, 2, {"m": "a"}),
    ]
    group = ExperimentGroup(
        config_fingerprint=FP_A,
        sweep_values={"m": "a"},
        cases=cases,
    )
    assert group.repeat_count == 2
    assert group.case_ids == ["c1", "c2"]


def test_experiment_group_abs_paths(tmp_path: Path):
    cases = [
        SweepCase("c1", Path("cases/c1"), Path("cases/c1/config.json"), "lbl", 1, 1, {}),
    ]
    group = ExperimentGroup(FP_A, {}, cases)
    assert group.abs_case_dirs(tmp_path) == [tmp_path / "cases/c1"]
    assert group.abs_config_paths(tmp_path) == [tmp_path / "cases/c1/config.json"]


# ---------------------------------------------------------------------------
# SweepManifest.from_path — basic loading
# ---------------------------------------------------------------------------

def test_sweep_manifest_loads_from_file(tmp_path: Path):
    path = _make_manifest(tmp_path, cases=TWO_MODELS)
    m = SweepManifest.from_path(path)

    assert m.schema_version == 1
    assert m.generated_at == "2026-01-01T00:00:00+00:00"
    assert m.notes == "test run"
    assert m.total_cases == 4
    assert m.sweep_root == tmp_path.resolve()


def test_sweep_manifest_loads_from_directory(tmp_path: Path):
    _make_manifest(tmp_path, cases=TWO_MODELS)
    m = SweepManifest.from_path(tmp_path)
    assert m.total_cases == 4


def test_sweep_manifest_groups_by_fingerprint(tmp_path: Path):
    path = _make_manifest(tmp_path, cases=TWO_MODELS)
    m = SweepManifest.from_path(path)

    assert m.experiment_count == 2
    fps = {g.config_fingerprint for g in m}
    assert fps == {FP_A, FP_B}


def test_sweep_manifest_repeats_per_group(tmp_path: Path):
    path = _make_manifest(tmp_path, cases=TWO_MODELS)
    m = SweepManifest.from_path(path)

    for group in m:
        assert group.repeat_count == 2
        assert len(group.case_ids) == 2


def test_sweep_manifest_notes_none(tmp_path: Path):
    # manifest without a notes key
    raw = {
        "schema_version": 1,
        "generated_at": "2026-01-01T00:00:00+00:00",
        "total_cases": 0,
        "cases": [],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(raw), encoding="utf-8")
    m = SweepManifest.from_path(tmp_path)
    assert m.notes is None


# ---------------------------------------------------------------------------
# SweepManifest accessors
# ---------------------------------------------------------------------------

def test_sweep_manifest_len_and_iter(tmp_path: Path):
    path = _make_manifest(tmp_path, cases=TWO_MODELS)
    m = SweepManifest.from_path(path)

    assert len(m) == 2
    groups = list(m)
    assert len(groups) == 2


def test_sweep_manifest_getitem_by_index(tmp_path: Path):
    path = _make_manifest(tmp_path, cases=TWO_MODELS)
    m = SweepManifest.from_path(path)

    assert isinstance(m[0], ExperimentGroup)
    assert isinstance(m[1], ExperimentGroup)


def test_sweep_manifest_getitem_by_fingerprint(tmp_path: Path):
    path = _make_manifest(tmp_path, cases=TWO_MODELS)
    m = SweepManifest.from_path(path)

    group_a = m[FP_A]
    assert group_a.sweep_values == {"model": "openai/gpt-x"}
    assert group_a.repeat_count == 2


def test_sweep_manifest_getitem_unknown_fingerprint_raises(tmp_path: Path):
    path = _make_manifest(tmp_path, cases=TWO_MODELS)
    m = SweepManifest.from_path(path)

    with pytest.raises(KeyError):
        _ = m["nonexistent_fp"]


# ---------------------------------------------------------------------------
# Single-repeat sweep (repeat_count == 1)
# ---------------------------------------------------------------------------

def test_sweep_manifest_single_repeat(tmp_path: Path):
    single_cases = [
        {
            "case_id": "solo01",
            "case_dir": "cases/solo01",
            "config_path": "cases/solo01/config.json",
            "label": "base",
            "repeat_number": 1,
            "repeat_count": 1,
            "sweep_values": {},
            "config_fingerprint": FP_A,
        }
    ]
    path = _make_manifest(tmp_path, cases=single_cases)
    m = SweepManifest.from_path(path)

    assert m.experiment_count == 1
    group = m[0]
    assert group.repeat_count == 1
    assert group.case_ids == ["solo01"]


# ---------------------------------------------------------------------------
# abs_case_dirs / abs_config_paths use sweep_root from manifest
# ---------------------------------------------------------------------------

def test_abs_paths_use_sweep_root(tmp_path: Path):
    path = _make_manifest(tmp_path, cases=TWO_MODELS[:2])  # only FP_A repeats
    m = SweepManifest.from_path(path)

    group = m[FP_A]
    dirs = group.abs_case_dirs(m.sweep_root)
    cfgs = group.abs_config_paths(m.sweep_root)

    assert dirs == [m.sweep_root / "cases/case001", m.sweep_root / "cases/case002"]
    assert cfgs == [
        m.sweep_root / "cases/case001/config.json",
        m.sweep_root / "cases/case002/config.json",
    ]


# ---------------------------------------------------------------------------
# run_analysis — SweepCase
# ---------------------------------------------------------------------------

def _write_case_config(case_dir: Path, model: str = "openai/gpt-x") -> None:
    """Write a minimal config.json into a case directory (including legacy machine-specific paths)."""
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "config.json").write_text(
        json.dumps({
            "scenario_id": "s1",
            "model": model,
            "num_turns": 2,
            # Simulate legacy machine-specific paths present in real sweep configs.
            "output_dir": "/old/machine/path/to/case",
            "catalog_path": "/old/machine/LLMAgora/data/scenarios.json",
            "prompts_path": "/old/machine/LLMAgora/data/prompts.json",
        }),
        encoding="utf-8",
    )


def test_sweep_case_run_analysis_calls_experiment(tmp_path: Path):
    """run_analysis builds the correct offline payload and delegates to run_persona_experiment."""
    case_dir = tmp_path / "cases" / "case001"
    _write_case_config(case_dir)

    sc = SweepCase(
        case_id="case001",
        case_dir=Path("cases/case001"),
        config_path=Path("cases/case001/config.json"),
        label="base",
        repeat_number=1,
        repeat_count=1,
        sweep_values={},
    )

    fake_result = MagicMock()
    captured = {}

    def fake_run_persona_experiment(cfg):
        captured["cfg"] = cfg
        return fake_result

    with patch("agora.experiment.run_persona_experiment", fake_run_persona_experiment):
        result = sc.run_analysis(tmp_path)

    assert result is fake_result
    cfg = captured["cfg"]
    assert cfg.num_turns == 0
    assert cfg.load_snapshot is True
    assert cfg.load_dir == case_dir
    assert cfg.reuse_load_dir_for_outputs is True
    assert cfg.save_snapshot is False
    assert cfg.indexed_output is False
    assert cfg.scenario_id == "s1"
    # Legacy machine-specific paths from the original run must be stripped,
    # falling back to ExperimentConfig defaults.
    assert cfg.output_dir is None
    assert cfg.catalog_path == Path("data/scenarios.json")
    assert cfg.prompts_path == Path("data/prompts.json")


def test_sweep_case_run_analysis_applies_postpro_overrides(tmp_path: Path):
    """Keyword overrides passed to run_analysis end up in the config."""
    case_dir = tmp_path / "cases" / "case_x"
    _write_case_config(case_dir)

    sc = SweepCase(
        case_id="case_x",
        case_dir=Path("cases/case_x"),
        config_path=Path("cases/case_x/config.json"),
        label="base",
        repeat_number=1,
        repeat_count=1,
        sweep_values={},
    )

    captured = {}

    def fake_run_persona_experiment(cfg):
        captured["cfg"] = cfg
        return MagicMock()

    with patch("agora.experiment.run_persona_experiment", fake_run_persona_experiment):
        sc.run_analysis(
            tmp_path,
            semantic_analysis_metrics=["self_consistency"],
            persona_analysis_metrics=[],
        )

    assert captured["cfg"].semantic_analysis_metrics == ["self_consistency"]


# ---------------------------------------------------------------------------
# run_analysis — ExperimentGroup
# ---------------------------------------------------------------------------

def test_experiment_group_run_analysis_runs_all_repeats(tmp_path: Path):
    """ExperimentGroup.run_analysis delegates to each SweepCase and collects results."""
    for cid in ("r1", "r2"):
        _write_case_config(tmp_path / "cases" / cid)

    cases = [
        SweepCase(cid, Path(f"cases/{cid}"), Path(f"cases/{cid}/config.json"), "lbl", i + 1, 2, {})
        for i, cid in enumerate(("r1", "r2"))
    ]
    group = ExperimentGroup(FP_A, {}, cases)

    call_count = 0

    def fake_run_persona_experiment(cfg):
        nonlocal call_count
        call_count += 1
        return MagicMock()

    with patch("agora.experiment.run_persona_experiment", fake_run_persona_experiment):
        results = group.run_analysis(tmp_path)

    assert call_count == 2
    assert len(results) == 2

