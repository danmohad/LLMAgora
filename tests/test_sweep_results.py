import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agora.sweep_results import (
    ExperimentGroup,
    GroupAnalysisResult,
    SweepCase,
    SweepManifest,
    _agg_nli_by_turn,
    _agg_persona_per_turn,
    _agg_persona_role,
    _agg_turn_scores,
    _classify_decision,
    _nli_bidirectional,
)


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
            semantic_similarity_method="cosine",
            semantic_similarity_model="all-mpnet-base-v2",
            persona_analysis_metrics=[],
        )

    assert captured["cfg"].semantic_analysis_metrics == ["self_consistency"]
    assert captured["cfg"].semantic_similarity_method == "cosine"
    assert captured["cfg"].semantic_similarity_model == "all-mpnet-base-v2"


# ---------------------------------------------------------------------------
# run_analysis — ExperimentGroup
# ---------------------------------------------------------------------------

def _fake_result(agent_names=("Alpha", "Beta"), sem=None, pers=None):
    """Build a minimal mock ExperimentResult."""
    agents = [MagicMock(name=n, id=n.lower()) for n in agent_names]
    for agent, name in zip(agents, agent_names):
        agent.name = name
    agora_mock = MagicMock()
    agora_mock.structured_history.return_value = {"turns": []}
    return MagicMock(agents=agents, eval_data={"semantic_similarity": sem or {}, "persona_adherence": pers}, agora=agora_mock)


def test_experiment_group_run_analysis_returns_group_result(tmp_path: Path):
    """ExperimentGroup.run_analysis returns a GroupAnalysisResult, not a list."""
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
        group_result = group.run_analysis(tmp_path)

    assert call_count == 2
    assert isinstance(group_result, GroupAnalysisResult)
    assert group_result.n_repeats == 2
    assert group_result.group is group


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def test_agg_turn_scores_basic():
    series = [
        {"turns": [1, 2], "scores": [0.8, 0.6]},
        {"turns": [1, 2], "scores": [0.6, 0.4]},
    ]
    result = _agg_turn_scores(series)
    assert result["turns"] == [1, 2]
    assert result["mean"] == pytest.approx([0.7, 0.5])
    assert result["se"] == pytest.approx([0.1 / 2**0.5, 0.1 / 2**0.5])


def test_agg_turn_scores_partial_turns():
    """Turns that appear in only some repeats are still included."""
    series = [
        {"turns": [1, 2], "scores": [0.9, 0.7]},
        {"turns": [1], "scores": [0.5]},
    ]
    result = _agg_turn_scores(series)
    assert 1 in result["turns"]
    assert result["mean"][result["turns"].index(1)] == pytest.approx(0.7)


def test_agg_persona_per_turn_basic():
    series = [
        {"turns": [1, 2], "scores": {"mean": [3.0, 4.0], "std": [0.0, 0.0]}},
        {"turns": [1, 2], "scores": {"mean": [5.0, 2.0], "std": [0.0, 0.0]}},
    ]
    result = _agg_persona_per_turn(series)
    assert result["turns"] == [1, 2]
    assert result["scores"]["mean"] == pytest.approx([4.0, 3.0])
    assert result["scores"]["se"] == pytest.approx([1.0 / 2**0.5, 1.0 / 2**0.5])


def test_agg_persona_role_aggregates_per_turn_and_scalar():
    role_data = [
        {
            "public_per_turn_scores": {"turns": [1], "scores": {"mean": [3.0], "std": [0.0]}},
            "private_per_turn_scores": {},
            "public_cumulative_scores": {},
            "private_cumulative_scores": {},
            "full_debate_public_score": {"mean": 3.0, "std": 0.0},
            "full_debate_private_score": None,
            "computed_metrics": ["full_debate_public"],
        },
        {
            "public_per_turn_scores": {"turns": [1], "scores": {"mean": [5.0], "std": [0.0]}},
            "private_per_turn_scores": {},
            "public_cumulative_scores": {},
            "private_cumulative_scores": {},
            "full_debate_public_score": {"mean": 5.0, "std": 0.0},
            "full_debate_private_score": None,
            "computed_metrics": ["full_debate_public"],
        },
    ]
    result = _agg_persona_role(role_data)
    assert result["public_per_turn_scores"]["scores"]["mean"] == pytest.approx([4.0])
    assert result["full_debate_public_score"]["mean"] == pytest.approx(4.0)
    assert result["full_debate_public_score"]["se"] == pytest.approx(1.0 / 2**0.5)
    assert result["computed_metrics"] == ["full_debate_public"]


def test_agg_nli_by_turn_basic():
    # Two repeats, two turns, 3 classes
    turn_dict = {
        1: [[0.1, 0.3, 0.6], [0.2, 0.4, 0.4]],
        2: [[0.3, 0.3, 0.4], [0.1, 0.5, 0.4]],
    }
    id2label = {0: "contradiction", 1: "neutral", 2: "entailment"}
    result = _agg_nli_by_turn(turn_dict, id2label)
    assert result["turns"] == [1, 2]
    assert result["label_names"] == ["contradiction", "neutral", "entailment"]
    assert result["distributions"]["entailment"]["mean"][0] == pytest.approx(0.5)
    assert result["distributions"]["contradiction"]["se"][0] == pytest.approx(0.05 / 2**0.5)


# ---------------------------------------------------------------------------
# GroupAnalysisResult — aggregate_semantic
# ---------------------------------------------------------------------------

def test_group_result_aggregate_semantic_basic():
    r1 = _fake_result(sem={
        "self_consistency": {"Alpha": {"turns": [1, 2], "scores": [0.8, 0.6]}},
        "cross_agent_public_alignment": {"turns": [1, 2], "scores": [0.7, 0.5]},
    })
    r2 = _fake_result(sem={
        "self_consistency": {"Alpha": {"turns": [1, 2], "scores": [0.6, 0.4]}},
        "cross_agent_public_alignment": {"turns": [1, 2], "scores": [0.5, 0.3]},
    })
    group = ExperimentGroup(FP_A, {}, [])
    gr = GroupAnalysisResult(group=group, results=[r1, r2])
    agg = gr.aggregate_semantic()

    sc_alpha = agg["self_consistency"]["Alpha"]
    assert sc_alpha["turns"] == [1, 2]
    assert sc_alpha["mean"] == pytest.approx([0.7, 0.5])

    cpa = agg["cross_agent_public_alignment"]
    assert cpa["mean"] == pytest.approx([0.6, 0.4])

    # cached on second call
    agg2 = gr.aggregate_semantic()
    assert agg2 is agg


def test_group_result_aggregate_semantic_empty():
    r = _fake_result(sem={})
    gr = GroupAnalysisResult(group=ExperimentGroup(FP_A, {}, []), results=[r])
    assert gr.aggregate_semantic() == {}


# ---------------------------------------------------------------------------
# GroupAnalysisResult — aggregate_persona
# ---------------------------------------------------------------------------

def _persona_data(pub_mean, pub_score_mean, n_turns=1):
    turns = list(range(1, n_turns + 1))
    return {
        "alpha": {
            "public_per_turn_scores": {"turns": turns, "scores": {"mean": [pub_mean] * n_turns, "std": [0.0] * n_turns}},
            "private_per_turn_scores": {},
            "public_cumulative_scores": {},
            "private_cumulative_scores": {},
            "full_debate_public_score": {"mean": pub_score_mean, "std": 0.0},
            "full_debate_private_score": None,
            "computed_metrics": ["full_debate_public"],
        },
        "beta": {
            "public_per_turn_scores": {},
            "private_per_turn_scores": {},
            "public_cumulative_scores": {},
            "private_cumulative_scores": {},
            "full_debate_public_score": None,
            "full_debate_private_score": None,
            "computed_metrics": [],
        },
    }


def test_group_result_aggregate_persona_basic():
    r1 = _fake_result(pers=_persona_data(3.0, 3.0))
    r2 = _fake_result(pers=_persona_data(5.0, 5.0))
    gr = GroupAnalysisResult(group=ExperimentGroup(FP_A, {}, []), results=[r1, r2])
    agg = gr.aggregate_persona()
    assert agg["alpha"]["full_debate_public_score"]["mean"] == pytest.approx(4.0)
    assert agg["alpha"]["public_per_turn_scores"]["scores"]["mean"] == pytest.approx([4.0])

    # cached on second call
    assert gr.aggregate_persona() is agg


def test_group_result_aggregate_persona_no_data():
    r = _fake_result(pers=None)
    gr = GroupAnalysisResult(group=ExperimentGroup(FP_A, {}, []), results=[r])
    assert gr.aggregate_persona() == {}


# ---------------------------------------------------------------------------
# GroupAnalysisResult — metadata
# ---------------------------------------------------------------------------

def test_group_result_agent_names():
    a1 = MagicMock()
    a1.name = "Eisenhower"
    a2 = MagicMock()
    a2.name = "Khrushchev"
    result = MagicMock(agents=[a1, a2])
    gr = GroupAnalysisResult(group=ExperimentGroup(FP_A, {}, []), results=[result])
    assert gr.agent_names == ("Eisenhower", "Khrushchev")


def test_group_result_agent_names_empty_results():
    gr = GroupAnalysisResult(group=ExperimentGroup(FP_A, {}, []), results=[])
    assert gr.agent_names == ("alpha", "beta")


def test_group_result_agent_names_single_agent():
    a1 = MagicMock()
    a1.name = "Solo"
    result = MagicMock(agents=[a1])
    gr = GroupAnalysisResult(group=ExperimentGroup(FP_A, {}, []), results=[result])
    assert gr.agent_names == ("Solo", "")


def test_group_result_n_repeats():
    gr = GroupAnalysisResult(group=ExperimentGroup(FP_A, {}, []), results=[MagicMock(), MagicMock()])
    assert gr.n_repeats == 2


# ---------------------------------------------------------------------------
# GroupAnalysisResult — summary (smoke test via capsys)
# ---------------------------------------------------------------------------

def test_group_result_summary_smoke(capsys):
    r1 = _fake_result(sem={
        "self_consistency": {"Alpha": {"turns": [1], "scores": [0.8]}},
    })
    r2 = _fake_result(sem={
        "self_consistency": {"Alpha": {"turns": [1], "scores": [0.6]}},
    })
    gr = GroupAnalysisResult(group=ExperimentGroup(FP_A, {}, []), results=[r1, r2])
    gr.summary()
    out = capsys.readouterr().out
    assert "EXPERIMENT GROUP SUMMARY" in out
    assert "Repeats" in out
    assert "Self-Consistency" in out


# ---------------------------------------------------------------------------
# GroupAnalysisResult — plot_* smoke tests (mocked plt.show)
# ---------------------------------------------------------------------------

def test_group_result_plot_semantic_smoke():
    r1 = _fake_result(sem={
        "self_consistency": {"A": {"turns": [1, 2], "scores": [0.8, 0.7]}},
        "cross_agent_public_alignment": {"turns": [1, 2], "scores": [0.6, 0.5]},
    })
    gr = GroupAnalysisResult(group=ExperimentGroup(FP_A, {}, []), results=[r1, r1])
    with patch("agora.plotting.plt.show"):
        gr.plot_semantic()


def test_group_result_plot_semantic_empty(capsys):
    r = _fake_result(sem={})
    gr = GroupAnalysisResult(group=ExperimentGroup(FP_A, {}, []), results=[r])
    with patch("agora.plotting.plt.show"):
        gr.plot_semantic()  # should not raise


def test_group_result_plot_persona_smoke():
    r1 = _fake_result(pers=_persona_data(3.0, 3.0))
    r2 = _fake_result(pers=_persona_data(5.0, 5.0))
    gr = GroupAnalysisResult(group=ExperimentGroup(FP_A, {}, []), results=[r1, r2])
    with patch("agora.plotting.plt.show"):
        gr.plot_persona()


def test_group_result_plot_persona_no_data(capsys):
    r = _fake_result(pers=None)
    gr = GroupAnalysisResult(group=ExperimentGroup(FP_A, {}, []), results=[r])
    gr.plot_persona()
    assert "No persona adherence data" in capsys.readouterr().out


def _fake_result_with_nli_history(pub_a, priv_a, pub_b, priv_b, turn_num=1):
    """Build a mock ExperimentResult whose agora yields structured history with two agents."""
    turn = {
        "turn_num": turn_num,
        "public_speech": pub_a,
        "private_reflection": priv_a,
    }
    history = {
        "AgentA": {"debate_turns": [{"turn_num": turn_num, "public_speech": pub_a, "private_reflection": priv_a}]},
        "AgentB": {"debate_turns": [{"turn_num": turn_num, "public_speech": pub_b, "private_reflection": priv_b}]},
    }
    agora_mock = MagicMock()
    agora_mock.structured_history.return_value = {"turns": [turn]}
    res = MagicMock(agora=agora_mock)
    return res, history


def test_group_result_plot_nli_smoke():
    from agora.sweep_results import _agg_nli_by_turn

    res, debate_data = _fake_result_with_nli_history(
        "pub alpha", "priv alpha", "pub beta", "priv beta"
    )
    gr = GroupAnalysisResult(group=ExperimentGroup(FP_A, {}, []), results=[res])

    fake_dist = [0.1, 0.3, 0.6]
    id2label = {0: "contradiction", 1: "neutral", 2: "entailment"}

    mock_analyzer = MagicMock()
    mock_analyzer.debate_data = debate_data
    mock_analyzer._id2label = id2label
    mock_analyzer.model = MagicMock()

    with (
        patch("agora.semantic_similarity_analyzer.SemanticSimilarityAnalyzer", return_value=mock_analyzer),
        patch("agora.sweep_results._nli_bidirectional", return_value=fake_dist),
        patch("agora.plotting.plt.show"),
    ):
        gr.plot_nli()

    nli = gr._nli_cache
    assert "cross_agent_public" in nli
    assert "cross_agent_private" in nli


def test_group_result_plot_nli_second_repeat_uses_existing_analyzer():
    """Second repeat re-uses the analyzer (else branch) without reloading the model."""
    res1, debate_data = _fake_result_with_nli_history("pa", "qa", "pb", "qb")
    res2, _ = _fake_result_with_nli_history("pc", "qc", "pd", "qd")
    # Make structured_history return a proper dict for the else branch
    res2.agora.structured_history.return_value = {"turns": []}

    gr = GroupAnalysisResult(group=ExperimentGroup(FP_A, {}, []), results=[res1, res2])

    mock_analyzer = MagicMock()
    mock_analyzer.debate_data = debate_data
    mock_analyzer._id2label = {0: "contradiction", 1: "neutral", 2: "entailment"}
    mock_analyzer.model = MagicMock()

    with (
        patch("agora.semantic_similarity_analyzer.SemanticSimilarityAnalyzer", return_value=mock_analyzer),
        patch("agora.sweep_results._nli_bidirectional", return_value=[0.1, 0.3, 0.6]),
        patch("agora.debate_history.get_structured_debate_history", return_value=debate_data),
        patch("agora.plotting.plt.show"),
    ):
        gr.plot_nli()  # should not raise


# ---------------------------------------------------------------------------
# build_emotion_style
# ---------------------------------------------------------------------------

def test_build_emotion_style_produces_consistent_map():
    from agora.plotting import build_emotion_style

    field_results = [
        {"agent_a": {"turns": [1], "emotions": {"joy": {"mean": [0.5], "std": [0.1]}, "anger": {"mean": [0.3], "std": [0.05]}}}},
        {"agent_a": {"turns": [1], "emotions": {"joy": {"mean": [0.4], "std": [0.1]}, "sadness": {"mean": [0.2], "std": [0.05]}}}},
    ]
    style = build_emotion_style(field_results)
    assert set(style.keys()) == {"anger", "joy", "sadness"}
    # Same label always maps to the same color+marker
    assert style["joy"]["color"] == style["joy"]["color"]
    assert "marker" in style["joy"]


# ---------------------------------------------------------------------------
# plot_group_nli / plot_group_emotions smoke tests
# ---------------------------------------------------------------------------

def test_plot_group_nli_smoke():
    from agora.plotting import plot_group_nli

    _nli_dist = {
        "turns": [1, 2],
        "label_names": ["contradiction", "neutral", "entailment"],
        "distributions": {
            "contradiction": {"mean": [0.1, 0.2], "se": [0.02, 0.03]},
            "neutral": {"mean": [0.3, 0.4], "se": [0.05, 0.04]},
            "entailment": {"mean": [0.6, 0.4], "se": [0.06, 0.05]},
        },
    }
    agg = {
        "id2label": {0: "contradiction", 1: "neutral", 2: "entailment"},
        "self_consistency": {
            "AgentA": {
                "turns": [1, 2],
                "label_names": ["contradiction", "neutral", "entailment"],
                "distributions": {
                    "contradiction": {"mean": [0.1, 0.2], "se": [0.02, 0.03]},
                    "neutral": {"mean": [0.3, 0.4], "se": [0.05, 0.04]},
                    "entailment": {"mean": [0.6, 0.4], "se": [0.06, 0.05]},
                },
            }
        },
        "cross_agent_public": _nli_dist,
        "cross_agent_private": _nli_dist,
    }
    with patch("agora.plotting.plt.show"):
        plot_group_nli(agg, "Alpha", "Beta")


def test_plot_group_nli_only_private():
    from agora.plotting import plot_group_nli

    _nli_dist = {
        "turns": [1],
        "label_names": ["contradiction", "neutral", "entailment"],
        "distributions": {
            "contradiction": {"mean": [0.2], "se": [0.02]},
            "neutral": {"mean": [0.5], "se": [0.05]},
            "entailment": {"mean": [0.3], "se": [0.03]},
        },
    }
    agg = {"cross_agent_private": _nli_dist}
    with patch("agora.plotting.plt.show"):
        plot_group_nli(agg)  # should not raise


def test_plot_group_emotions_smoke():
    from agora.plotting import build_emotion_style, plot_group_emotions

    agg = {
        "AgentA": {
            "turns": [1, 2],
            "emotions": {
                "joy": {"mean": [0.4, 0.5], "se": [0.05, 0.06]},
                "anger": {"mean": [0.1, 0.2], "se": [0.02, 0.03]},
            },
        },
        "AgentB": {
            "turns": [1, 2],
            "emotions": {
                "joy": {"mean": [0.3, 0.4], "se": [0.04, 0.05]},
                "anger": {"mean": [0.2, 0.1], "se": [0.03, 0.02]},
            },
        },
    }
    style = build_emotion_style([agg])
    with patch("agora.plotting.plt.show"):
        plot_group_emotions(agg, "Public Utterances", "A", "B", emotion_style=style)


def test_plot_group_emotions_no_data(capsys):
    from agora.plotting import plot_group_emotions

    with patch("agora.plotting.plt.show"):
        plot_group_emotions({}, "Public Utterances")
    assert "No emotion data" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# GroupAnalysisResult — aggregate_survey
# ---------------------------------------------------------------------------

def _make_survey_turns(turn_num: int, pub_scores: dict, priv_scores: dict) -> list[dict]:
    """Build a structured-history turn (Alpha/Beta slot format) with survey data."""
    return [{
        "turn_num": turn_num,
        "Alpha": {"speaker_id": "uuid-a", "public_survey": pub_scores, "private_survey": priv_scores},
        "Beta": {"speaker_id": "uuid-b", "public_survey": pub_scores},
    }]


def _fake_result_with_survey(turns: list) -> object:
    agora_mock = MagicMock()
    agora_mock.structured_history.return_value = {"turns": turns}
    result = MagicMock(agora=agora_mock)
    result.survey_question_specs = []
    return result


def test_group_result_aggregate_survey_basic():
    turns1 = _make_survey_turns(1, {"Q1": 1, "Q2": -1}, {"Q1": 2})
    turns2 = _make_survey_turns(1, {"Q1": 3, "Q2": 1}, {"Q1": 0})
    r1 = _fake_result_with_survey(turns1)
    r2 = _fake_result_with_survey(turns2)
    gr = GroupAnalysisResult(group=ExperimentGroup(FP_A, {}, []), results=[r1, r2])
    agg = gr.aggregate_survey()

    assert "public" in agg
    assert "private" in agg
    assert "diff" in agg

    # Slot names, not speaker UUIDs
    assert "Alpha" in agg["public"]
    assert "Beta" in agg["public"]
    assert len(agg["public"]) == 2

    pub_a = agg["public"]["Alpha"]
    assert pub_a["Q1"]["turns"] == [1]
    assert pub_a["Q1"]["mean"] == pytest.approx([2.0])  # mean(1, 3)
    assert pub_a["Q1"]["se"] == pytest.approx([1.0 / 2**0.5])  # std(1,3)/sqrt(2)
    assert pub_a["Q2"]["mean"] == pytest.approx([0.0])  # mean(-1, 1)

    priv_a = agg["private"]["Alpha"]
    assert priv_a["Q1"]["mean"] == pytest.approx([1.0])  # mean(2, 0)

    # diff = pub - priv for Alpha/Q1: mean(1-2, 3-0) = mean(-1, 3) = 1.0
    diff_a = agg["diff"]["Alpha"]
    assert diff_a["Q1"]["mean"] == pytest.approx([1.0])


def test_group_result_aggregate_survey_empty():
    agora_mock = MagicMock()
    agora_mock.structured_history.return_value = {"turns": []}
    r = MagicMock(agora=agora_mock)
    gr = GroupAnalysisResult(group=ExperimentGroup(FP_A, {}, []), results=[r])
    assert gr.aggregate_survey() == {}


def test_group_result_aggregate_survey_caching():
    turns = _make_survey_turns(1, {"Q1": 1}, {})
    r = _fake_result_with_survey(turns)
    gr = GroupAnalysisResult(group=ExperimentGroup(FP_A, {}, []), results=[r])
    agg = gr.aggregate_survey()
    # Cache is used when no survey_questions provided
    assert gr.aggregate_survey() is agg
    # Passing survey_questions bypasses cache (always fresh)
    agg_qs = gr.aggregate_survey(survey_questions=["Q1 text"])
    assert agg_qs is not agg


def test_group_result_aggregate_survey_sentiment_abs():
    # Q1 is direct (pub 1, priv 3): signed diff = 1-3 = -2
    # Q2 is sentiment (pub 1, priv 3): abs diff = |1-3| = 2
    turns = _make_survey_turns(1, {"Q1": 1, "Q2": 1}, {"Q1": 3, "Q2": 3})
    r = _fake_result_with_survey(turns)
    gr = GroupAnalysisResult(group=ExperimentGroup(FP_A, {}, []), results=[r])
    q_specs = {"direct": ["Direct question"], "sentiment": ["Sentiment question"]}
    agg = gr.aggregate_survey(survey_questions=q_specs)

    diff_a = agg["diff"]["Alpha"]
    assert diff_a["Q1"]["mean"] == pytest.approx([-2.0])  # signed
    assert diff_a["Q2"]["mean"] == pytest.approx([2.0])   # absolute


def test_group_result_survey_question_specs_property():
    # Returns empty list when no results have specs
    turns = _make_survey_turns(1, {"Q1": 1}, {})
    r = _fake_result_with_survey(turns)
    gr = GroupAnalysisResult(group=ExperimentGroup(FP_A, {}, []), results=[r])
    assert gr.survey_question_specs == []

    # Returns specs from the first result that has them
    r_with_specs = _fake_result_with_survey(turns)
    specs = [{"text": "A question", "group": "direct"}]
    r_with_specs.survey_question_specs = specs
    gr2 = GroupAnalysisResult(group=ExperimentGroup(FP_A, {}, []), results=[r, r_with_specs])
    assert gr2.survey_question_specs is specs


def test_group_result_plot_survey_uses_stored_specs():
    """plot_survey() auto-uses survey_question_specs without explicit arg."""
    turns = _make_survey_turns(1, {"Q1": 1, "Q2": -1}, {"Q1": 2})
    r = _fake_result_with_survey(turns)
    r.survey_question_specs = [
        {"text": "Direct question", "group": "direct"},
        {"text": "Sentiment question", "group": "sentiment"},
    ]
    gr = GroupAnalysisResult(group=ExperimentGroup(FP_A, {}, []), results=[r])
    with patch("agora.plotting.plt.show"):
        gr.plot_survey()  # should not raise; uses specs automatically


def test_group_result_plot_survey_smoke():
    turns = _make_survey_turns(1, {"Q1": 1, "Q2": -1}, {"Q1": 2})
    r1 = _fake_result_with_survey(turns)
    r2 = _fake_result_with_survey(turns)
    gr = GroupAnalysisResult(group=ExperimentGroup(FP_A, {}, []), results=[r1, r2])
    with patch("agora.plotting.plt.show"):
        gr.plot_survey()


def test_group_result_plot_survey_no_data(capsys):
    agora_mock = MagicMock()
    agora_mock.structured_history.return_value = {"turns": []}
    r = MagicMock(agora=agora_mock)
    r.survey_question_specs = []
    gr = GroupAnalysisResult(group=ExperimentGroup(FP_A, {}, []), results=[r])
    gr.plot_survey()
    assert "No survey data" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Additional coverage tests
# ---------------------------------------------------------------------------

def test_nli_bidirectional_plain_list():
    """_nli_bidirectional with a plain list (no tolist, no nesting)."""
    mock_analyzer = MagicMock()
    mock_analyzer.model.predict.return_value = [0.1, 0.3, 0.6]
    result = _nli_bidirectional(mock_analyzer, "text_a", "text_b")
    assert len(result) == 3
    assert result == pytest.approx([0.1, 0.3, 0.6])


def test_nli_bidirectional_nested_list():
    """_nli_bidirectional with a nested list [[...]] (covers p[0] path)."""
    mock_analyzer = MagicMock()
    mock_analyzer.model.predict.return_value = [[0.2, 0.4, 0.4]]
    result = _nli_bidirectional(mock_analyzer, "text_a", "text_b")
    assert len(result) == 3


def test_nli_bidirectional_tolist():
    """_nli_bidirectional with an object that has a .tolist() method (numpy-like)."""
    class FakeTensor:
        def tolist(self):
            return [0.3, 0.3, 0.4]

    mock_analyzer = MagicMock()
    mock_analyzer.model.predict.return_value = FakeTensor()
    result = _nli_bidirectional(mock_analyzer, "text_a", "text_b")
    assert len(result) == 3


def test_aggregate_semantic_includes_cross_agent_private():
    """aggregate_semantic aggregates cross_agent_private_alignment when present."""
    r1 = _fake_result(sem={
        "cross_agent_private_alignment": {"turns": [1], "scores": [0.7]},
    })
    r2 = _fake_result(sem={
        "cross_agent_private_alignment": {"turns": [1], "scores": [0.5]},
    })
    gr = GroupAnalysisResult(group=ExperimentGroup(FP_A, {}, []), results=[r1, r2])
    agg = gr.aggregate_semantic()
    assert "cross_agent_private_alignment" in agg
    assert agg["cross_agent_private_alignment"]["mean"] == pytest.approx([0.6])


def test_run_nli_analysis_cache_hit():
    """Second call to run_nli_analysis returns the cached result."""
    res, debate_data = _fake_result_with_nli_history("pub", "priv", "pub2", "priv2")
    gr = GroupAnalysisResult(group=ExperimentGroup(FP_A, {}, []), results=[res])

    mock_analyzer = MagicMock()
    mock_analyzer.debate_data = debate_data
    mock_analyzer._id2label = {0: "contradiction", 1: "neutral", 2: "entailment"}

    with (
        patch("agora.semantic_similarity_analyzer.SemanticSimilarityAnalyzer", return_value=mock_analyzer),
        patch("agora.sweep_results._nli_bidirectional", return_value=[0.1, 0.3, 0.6]),
        patch("agora.plotting.plt.show"),
    ):
        result1 = gr.run_nli_analysis()
        result2 = gr.run_nli_analysis()  # cache hit

    assert result2 is result1


def test_run_nli_analysis_with_model_name():
    """Passing model_name forwards it to SemanticSimilarityAnalyzer kwargs."""
    res, debate_data = _fake_result_with_nli_history("pub", "priv", "pub2", "priv2")
    gr = GroupAnalysisResult(group=ExperimentGroup(FP_A, {}, []), results=[res])

    mock_analyzer = MagicMock()
    mock_analyzer.debate_data = debate_data
    mock_analyzer._id2label = {0: "contradiction", 1: "neutral", 2: "entailment"}
    captured_kwargs = {}

    def fake_ssa(history, **kwargs):
        captured_kwargs.update(kwargs)
        return mock_analyzer

    with (
        patch("agora.semantic_similarity_analyzer.SemanticSimilarityAnalyzer", side_effect=fake_ssa),
        patch("agora.sweep_results._nli_bidirectional", return_value=[0.1, 0.3, 0.6]),
    ):
        gr.run_nli_analysis(model_name="my-model")

    assert captured_kwargs.get("model_name") == "my-model"


def test_run_nli_analysis_second_repeat_non_dict_history():
    """Second repeat whose structured_history isn't a dict with 'turns' uses it directly."""
    res1, debate_data = _fake_result_with_nli_history("pub", "priv", "pub2", "priv2")
    res2, _ = _fake_result_with_nli_history("pa", "qa", "pb", "qb")
    # Make second result return raw debate_data (no "turns" key)
    res2.agora.structured_history.return_value = debate_data

    gr = GroupAnalysisResult(group=ExperimentGroup(FP_A, {}, []), results=[res1, res2])

    mock_analyzer = MagicMock()
    mock_analyzer.debate_data = debate_data
    mock_analyzer._id2label = {0: "contradiction", 1: "neutral", 2: "entailment"}

    with (
        patch("agora.semantic_similarity_analyzer.SemanticSimilarityAnalyzer", return_value=mock_analyzer),
        patch("agora.sweep_results._nli_bidirectional", return_value=[0.1, 0.3, 0.6]),
    ):
        gr.run_nli_analysis()  # should not raise


def test_run_emotion_analysis():
    """run_emotion_analysis aggregates emotion data across repeats.

    Three results:
    - r1: history with 'turns' key → creates the EmotionAnalyzer
    - r2: raw debate_data dict (no 'turns') → hits `ea.debate_data = structured_history` (else branch)
    - r3: history with 'turns' key → hits `ea.debate_data = get_structured_debate_history(...)` (if branch)
    """
    history1 = {"turns": [{"turn_num": 1, "public_speech": "hello"}]}
    # r2's history is raw debate_data (no 'turns' key) → triggers the else branch
    history2 = {"AgentA": {"debate_turns": [], "pre_interview": None, "post_interview": None}}
    # r3's history has 'turns' → triggers get_structured_debate_history on an existing analyzer
    history3 = {"turns": [{"turn_num": 2, "public_speech": "again"}]}

    r1 = MagicMock()
    r1.agora.structured_history.return_value = history1
    r2 = MagicMock()
    r2.agora.structured_history.return_value = history2
    r3 = MagicMock()
    r3.agora.structured_history.return_value = history3

    gr = GroupAnalysisResult(group=ExperimentGroup(FP_A, {}, []), results=[r1, r2, r3])

    mock_ea = MagicMock()
    mock_ea.classify_field.return_value = {
        "AgentA": {"turns": [1], "emotions": {"joy": [0.5], "anger": [0.2]}},
    }

    with patch("agora.emotion_analyzer.EmotionAnalyzer", return_value=mock_ea):
        result = gr.run_emotion_analysis("public_speech")

    assert "AgentA" in result
    assert "joy" in result["AgentA"]["emotions"]
    assert result["AgentA"]["turns"] == [1]

    # Cached on second call
    with patch("agora.emotion_analyzer.EmotionAnalyzer", return_value=mock_ea):
        result2 = gr.run_emotion_analysis("public_speech")
    assert result2 is result


def test_group_result_summary_no_self_consistency(capsys):
    """Summary with empty self_consistency prints 'not computed' message."""
    r = _fake_result(sem={"cross_agent_public_alignment": {"turns": [1], "scores": [0.7]}})
    gr = GroupAnalysisResult(group=ExperimentGroup(FP_A, {}, []), results=[r])
    gr.summary()
    out = capsys.readouterr().out
    assert "self_consistency: not computed" in out


def test_group_result_summary_with_cross_alignment(capsys):
    """Summary prints cross-agent alignment data when present."""
    r = _fake_result(sem={"cross_agent_public_alignment": {"turns": [1], "scores": [0.7]}})
    gr = GroupAnalysisResult(group=ExperimentGroup(FP_A, {}, []), results=[r])
    gr.summary()
    out = capsys.readouterr().out
    assert "Cross-Agent Public Alignment" in out
    assert "mean=" in out


def test_group_result_summary_with_persona_data(capsys):
    """Summary prints persona adherence scores when available."""
    r = _fake_result(pers=_persona_data(4.0, 4.0))
    gr = GroupAnalysisResult(group=ExperimentGroup(FP_A, {}, []), results=[r])
    gr.summary()
    out = capsys.readouterr().out
    assert "PERSONA ADHERENCE" in out
    assert "Full-debate public" in out


def test_group_result_plot_emotions():
    """plot_emotions calls run_emotion_analysis and forwards result to plot_group_emotions."""
    r = _fake_result()
    gr = GroupAnalysisResult(group=ExperimentGroup(FP_A, {}, []), results=[r])

    mock_ea = MagicMock()
    mock_ea.classify_field.return_value = {}

    with (
        patch("agora.emotion_analyzer.EmotionAnalyzer", return_value=mock_ea),
        patch("agora.plotting.plot_group_emotions") as mock_plot,
    ):
        gr.plot_emotions("public_speech")

    mock_plot.assert_called_once()


# ---------------------------------------------------------------------------
# _classify_decision
# ---------------------------------------------------------------------------

def test_classify_decision_matches_first_label():
    assert _classify_decision("ENDORSE the bill!", ["ENDORSE", "DO NOT ENDORSE"]) == 0


def test_classify_decision_matches_second_label():
    assert _classify_decision("DO NOT ENDORSE based on the analysis", ["ENDORSE", "DO NOT ENDORSE"]) == 1


def test_classify_decision_no_match():
    assert _classify_decision("Abstain from voting.", ["ENDORSE", "DO NOT ENDORSE"]) is None


def test_classify_decision_longest_label_wins():
    # "DO NOT ENDORSE" starts with "DO NOT", so should match index 1, not index 0 fallback.
    result = _classify_decision("DO NOT ENDORSE the bill", ["ENDORSE", "DO NOT ENDORSE"])
    assert result == 1


def test_classify_decision_case_insensitive():
    assert _classify_decision("endorse it", ["ENDORSE", "DO NOT ENDORSE"]) == 0


# ---------------------------------------------------------------------------
# aggregate_response_decisions
# ---------------------------------------------------------------------------

def _fake_result_with_decisions(pub_alpha, priv_alpha, pub_beta, priv_beta, turn_num=1):
    """Build a mock ExperimentResult with structured history containing decisions."""
    agora_mock = MagicMock()
    agora_mock.structured_history.return_value = {
        "turns": [
            {
                "turn_num": turn_num,
                "Alpha": {
                    "public_utterance": pub_alpha,
                    "private_utterance": priv_alpha,
                },
                "Beta": {
                    "public_utterance": pub_beta,
                    "private_utterance": priv_beta,
                },
            }
        ]
    }
    return MagicMock(agora=agora_mock)


def _write_decision_catalog(path: Path, scenario_id: str, decision_labels: list[str]) -> None:
    catalog = {
        "version": "test",
        "scenarios": [{"scenario_id": scenario_id, "decision_labels": decision_labels}],
    }
    path.write_text(json.dumps(catalog), encoding="utf-8")


def test_aggregate_response_decisions_basic(tmp_path: Path):
    catalog = tmp_path / "catalog.json"
    _write_decision_catalog(catalog, "s1", ["ENDORSE", "DO NOT ENDORSE"])

    res = _fake_result_with_decisions("ENDORSE the bill", "DO NOT ENDORSE privately", "DO NOT ENDORSE", "ENDORSE")
    gr = GroupAnalysisResult(
        group=ExperimentGroup(FP_A, {"scenario_id": "s1"}, []),
        results=[res],
    )

    agg = gr.aggregate_response_decisions(catalog_path=catalog)

    assert agg["decision_label"] == "ENDORSE"
    alpha = agg["by_slot"]["Alpha"]
    # public: matched ENDORSE (idx=0) → value 1.0
    assert alpha["public"]["turns"] == [1]
    assert alpha["public"]["mean"] == [1.0]
    # private: matched DO NOT ENDORSE (idx=1) → value 0.0
    assert alpha["private"]["mean"] == [0.0]


def test_aggregate_response_decisions_caches_result(tmp_path: Path):
    catalog = tmp_path / "catalog.json"
    _write_decision_catalog(catalog, "s1", ["YES", "NO"])

    res = _fake_result_with_decisions("YES go ahead", "NO way", "NO way", "YES")
    gr = GroupAnalysisResult(
        group=ExperimentGroup(FP_A, {"scenario_id": "s1"}, []),
        results=[res],
    )

    agg1 = gr.aggregate_response_decisions(catalog_path=catalog)
    # Second call without path should still return cache (None path means use default)
    gr._decision_cache = agg1
    agg2 = gr.aggregate_response_decisions()
    assert agg2 is agg1


def test_aggregate_response_decisions_sets_cache_on_default_path_call(tmp_path: Path):
    """When called with no explicit params, result is stored in _decision_cache."""
    catalog = tmp_path / "catalog.json"
    _write_decision_catalog(catalog, "s_real", ["YES", "NO"])

    res = _fake_result_with_decisions("YES indeed", "NO way", "NO way", "YES")
    gr = GroupAnalysisResult(
        group=ExperimentGroup(FP_A, {"scenario_id": "s_real"}, []),
        results=[res],
    )

    with patch("agora.experiment.DEFAULT_CATALOG_PATH", catalog):
        agg = gr.aggregate_response_decisions()

    assert gr._decision_cache is agg


def test_aggregate_response_decisions_explicit_scenario_id_bypasses_cache(tmp_path: Path):
    catalog = tmp_path / "catalog.json"
    _write_decision_catalog(catalog, "s_other", ["A", "B"])

    res = _fake_result_with_decisions("A first", "B second", "A first", "A first")
    gr = GroupAnalysisResult(
        group=ExperimentGroup(FP_A, {"scenario_id": "s_other"}, []),
        results=[res],
    )
    gr._decision_cache = {"decision_label": "stale"}

    # Passing scenario_id explicitly must NOT use the cache
    agg = gr.aggregate_response_decisions(scenario_id="s_other", catalog_path=catalog)
    assert agg["decision_label"] == "A"


def test_aggregate_response_decisions_missing_scenario_id_raises():
    gr = GroupAnalysisResult(group=ExperimentGroup(FP_A, {}, []), results=[])
    with pytest.raises(ValueError, match="scenario_id"):
        gr.aggregate_response_decisions()


def test_aggregate_response_decisions_scenario_not_in_catalog_raises(tmp_path: Path):
    catalog = tmp_path / "catalog.json"
    _write_decision_catalog(catalog, "other_scenario", ["X", "Y"])

    gr = GroupAnalysisResult(group=ExperimentGroup(FP_A, {"scenario_id": "missing"}, []), results=[])
    with pytest.raises(KeyError, match="missing"):
        gr.aggregate_response_decisions(catalog_path=catalog)


def test_aggregate_response_decisions_too_few_labels_raises(tmp_path: Path):
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps({"scenarios": [{"scenario_id": "s1", "decision_labels": ["ONLY_ONE"]}]}),
        encoding="utf-8",
    )
    gr = GroupAnalysisResult(group=ExperimentGroup(FP_A, {"scenario_id": "s1"}, []), results=[])
    with pytest.raises(ValueError, match="decision_labels"):
        gr.aggregate_response_decisions(catalog_path=catalog)


def test_aggregate_response_decisions_skips_unmatched_text(tmp_path: Path):
    catalog = tmp_path / "catalog.json"
    _write_decision_catalog(catalog, "s1", ["ENDORSE", "DO NOT ENDORSE"])

    res = _fake_result_with_decisions("Unclear response", "", "ENDORSE clearly", "")
    gr = GroupAnalysisResult(
        group=ExperimentGroup(FP_A, {"scenario_id": "s1"}, []),
        results=[res],
    )
    agg = gr.aggregate_response_decisions(catalog_path=catalog)
    # Alpha had unmatched public and empty private → no Alpha entry, only Beta public
    assert "Alpha" not in agg["by_slot"]
    assert "Beta" in agg["by_slot"]
    assert "private" not in agg["by_slot"]["Beta"]


def test_aggregate_response_decisions_se_formula(tmp_path: Path):
    """SE is sqrt(p*(1-p)/n) for binary proportions."""
    catalog = tmp_path / "catalog.json"
    _write_decision_catalog(catalog, "s1", ["YES", "NO"])

    # Two repeats: both chose YES → p=1.0, SE=0.0
    r1 = _fake_result_with_decisions("YES", None, None, None)
    r2 = _fake_result_with_decisions("YES", None, None, None)
    gr = GroupAnalysisResult(
        group=ExperimentGroup(FP_A, {"scenario_id": "s1"}, []),
        results=[r1, r2],
    )
    agg = gr.aggregate_response_decisions(catalog_path=catalog)
    assert agg["by_slot"]["Alpha"]["public"]["mean"] == [1.0]
    assert agg["by_slot"]["Alpha"]["public"]["se"] == [0.0]


# ---------------------------------------------------------------------------
# plot_response_decisions
# ---------------------------------------------------------------------------

def test_plot_response_decisions_calls_plot_function(tmp_path: Path):
    catalog = tmp_path / "catalog.json"
    _write_decision_catalog(catalog, "s1", ["YES", "NO"])

    res = _fake_result_with_decisions("YES", "NO", "NO", "YES")
    gr = GroupAnalysisResult(
        group=ExperimentGroup(FP_A, {"scenario_id": "s1"}, []),
        results=[res],
    )
    with patch("agora.plotting.plot_group_response_decisions") as mock_plot:
        gr.plot_response_decisions(catalog_path=catalog)
    mock_plot.assert_called_once()


def test_plot_response_decisions_empty_prints_message(capsys, tmp_path: Path):
    catalog = tmp_path / "catalog.json"
    _write_decision_catalog(catalog, "s1", ["YES", "NO"])

    # No turns → by_slot is empty
    agora_mock = MagicMock()
    agora_mock.structured_history.return_value = {"turns": []}
    res = MagicMock(agora=agora_mock)
    gr = GroupAnalysisResult(
        group=ExperimentGroup(FP_A, {"scenario_id": "s1"}, []),
        results=[res],
    )
    gr.plot_response_decisions(catalog_path=catalog)
    assert "No response decision data" in capsys.readouterr().out


# ============================================================ aggregate_response_decisions_all_repeats


def test_aggregate_response_decisions_all_repeats_basic(tmp_path: Path):
    catalog = tmp_path / "catalog.json"
    _write_decision_catalog(catalog, "s1", ["ENDORSE", "DO NOT ENDORSE"])

    res = _fake_result_with_decisions(
        "ENDORSE the bill", "DO NOT ENDORSE privately", "DO NOT ENDORSE", "ENDORSE"
    )
    gr = GroupAnalysisResult(
        group=ExperimentGroup(FP_A, {"scenario_id": "s1"}, []),
        results=[res],
    )

    data = gr.aggregate_response_decisions_all_repeats(catalog_path=catalog)

    assert data["decision_label"] == "ENDORSE"
    assert len(data["repeats"]) == 1
    repeat = data["repeats"][0]
    # public Alpha → ENDORSE (idx 0) → value 1
    assert repeat["Alpha"]["public"]["turns"] == [1]
    assert repeat["Alpha"]["public"]["decisions"] == [1]
    # private Alpha → DO NOT ENDORSE (idx 1) → value 0
    assert repeat["Alpha"]["private"]["decisions"] == [0]
    # public Beta → DO NOT ENDORSE → value 0
    assert repeat["Beta"]["public"]["decisions"] == [0]
    # private Beta → ENDORSE → value 1
    assert repeat["Beta"]["private"]["decisions"] == [1]


def test_aggregate_response_decisions_all_repeats_multiple_repeats(tmp_path: Path):
    catalog = tmp_path / "catalog.json"
    _write_decision_catalog(catalog, "s1", ["YES", "NO"])

    res1 = _fake_result_with_decisions("YES", "NO", "YES", "YES")
    res2 = _fake_result_with_decisions("NO", "YES", "NO", "NO")
    gr = GroupAnalysisResult(
        group=ExperimentGroup(FP_A, {"scenario_id": "s1"}, []),
        results=[res1, res2],
    )

    data = gr.aggregate_response_decisions_all_repeats(catalog_path=catalog)

    assert len(data["repeats"]) == 2
    assert data["repeats"][0]["Alpha"]["public"]["decisions"] == [1]
    assert data["repeats"][1]["Alpha"]["public"]["decisions"] == [0]


def test_aggregate_response_decisions_all_repeats_caches_result(tmp_path: Path):
    catalog = tmp_path / "catalog.json"
    _write_decision_catalog(catalog, "s1", ["YES", "NO"])

    res = _fake_result_with_decisions("YES", "NO", "YES", "NO")
    gr = GroupAnalysisResult(
        group=ExperimentGroup(FP_A, {"scenario_id": "s1"}, []),
        results=[res],
    )

    first = gr.aggregate_response_decisions_all_repeats(catalog_path=catalog)
    # explicit path → no automatic caching
    assert gr._decision_per_repeat_cache is None
    # manually set cache and confirm it is used
    gr._decision_per_repeat_cache = first
    second = gr.aggregate_response_decisions_all_repeats()
    assert second is first


def test_aggregate_response_decisions_all_repeats_sets_cache_on_default_path(tmp_path: Path):
    catalog = tmp_path / "catalog.json"
    _write_decision_catalog(catalog, "s_real", ["YES", "NO"])

    res = _fake_result_with_decisions("YES", "NO", "NO", "YES")
    gr = GroupAnalysisResult(
        group=ExperimentGroup(FP_A, {"scenario_id": "s_real"}, []),
        results=[res],
    )

    with patch("agora.experiment.DEFAULT_CATALOG_PATH", catalog):
        data = gr.aggregate_response_decisions_all_repeats()

    assert gr._decision_per_repeat_cache is data


def test_aggregate_response_decisions_all_repeats_explicit_params_bypass_cache(tmp_path: Path):
    catalog = tmp_path / "catalog.json"
    _write_decision_catalog(catalog, "s1", ["A", "B"])

    res = _fake_result_with_decisions("A first", "B second", "A first", "A first")
    gr = GroupAnalysisResult(
        group=ExperimentGroup(FP_A, {"scenario_id": "s1"}, []),
        results=[res],
    )
    gr._decision_per_repeat_cache = {"decision_label": "stale", "repeats": []}

    data = gr.aggregate_response_decisions_all_repeats(scenario_id="s1", catalog_path=catalog)
    assert data["decision_label"] == "A"


def test_aggregate_response_decisions_all_repeats_missing_scenario_id_raises():
    gr = GroupAnalysisResult(group=ExperimentGroup(FP_A, {}, []), results=[])
    with pytest.raises(ValueError, match="scenario_id"):
        gr.aggregate_response_decisions_all_repeats()


def test_aggregate_response_decisions_all_repeats_scenario_not_found_raises(tmp_path: Path):
    catalog = tmp_path / "catalog.json"
    _write_decision_catalog(catalog, "other", ["X", "Y"])

    gr = GroupAnalysisResult(
        group=ExperimentGroup(FP_A, {"scenario_id": "missing"}, []), results=[]
    )
    with pytest.raises(KeyError, match="missing"):
        gr.aggregate_response_decisions_all_repeats(catalog_path=catalog)


def test_aggregate_response_decisions_all_repeats_too_few_labels_raises(tmp_path: Path):
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps({"scenarios": [{"scenario_id": "s1", "decision_labels": ["ONLY"]}]}),
        encoding="utf-8",
    )

    gr = GroupAnalysisResult(
        group=ExperimentGroup(FP_A, {"scenario_id": "s1"}, []), results=[]
    )
    with pytest.raises(ValueError, match="Expected 2"):
        gr.aggregate_response_decisions_all_repeats(catalog_path=catalog)


def test_aggregate_response_decisions_all_repeats_empty_turns(tmp_path: Path):
    catalog = tmp_path / "catalog.json"
    _write_decision_catalog(catalog, "s1", ["YES", "NO"])

    agora_mock = MagicMock()
    agora_mock.structured_history.return_value = {"turns": []}
    res = MagicMock(agora=agora_mock)
    gr = GroupAnalysisResult(
        group=ExperimentGroup(FP_A, {"scenario_id": "s1"}, []),
        results=[res],
    )

    data = gr.aggregate_response_decisions_all_repeats(catalog_path=catalog)
    assert data["repeats"] == [{}]


def test_aggregate_response_decisions_all_repeats_empty_utterance_skipped(tmp_path: Path):
    """Empty public/private utterance strings are skipped (covers 'if not text: continue')."""
    catalog = tmp_path / "catalog.json"
    _write_decision_catalog(catalog, "s1", ["YES", "NO"])

    agora_mock = MagicMock()
    agora_mock.structured_history.return_value = {
        "turns": [
            {
                "turn_num": 1,
                "Alpha": {"public_utterance": "", "private_utterance": ""},
                "Beta":  {"public_utterance": "", "private_utterance": ""},
            }
        ]
    }
    res = MagicMock(agora=agora_mock)
    gr = GroupAnalysisResult(
        group=ExperimentGroup(FP_A, {"scenario_id": "s1"}, []),
        results=[res],
    )

    data = gr.aggregate_response_decisions_all_repeats(catalog_path=catalog)
    # No utterances recorded — repeat entry should be empty
    assert data["repeats"] == [{}]


def test_aggregate_response_decisions_all_repeats_unmatched_text_skipped(tmp_path: Path):
    """Text matching neither label returns None from _classify_decision (covers 'if idx is None')."""
    catalog = tmp_path / "catalog.json"
    _write_decision_catalog(catalog, "s1", ["YES", "NO"])

    agora_mock = MagicMock()
    agora_mock.structured_history.return_value = {
        "turns": [
            {
                "turn_num": 1,
                "Alpha": {
                    "public_utterance": "NEITHER label appears here at all",
                    "private_utterance": "NEITHER label",
                },
                "Beta": {
                    "public_utterance": "NEITHER",
                    "private_utterance": "NEITHER",
                },
            }
        ]
    }
    res = MagicMock(agora=agora_mock)
    gr = GroupAnalysisResult(
        group=ExperimentGroup(FP_A, {"scenario_id": "s1"}, []),
        results=[res],
    )

    data = gr.aggregate_response_decisions_all_repeats(catalog_path=catalog)
    # No matches recorded
    assert data["repeats"] == [{}]


# ============================================================ plot_response_decisions_all_repeats


def test_plot_response_decisions_all_repeats_calls_plot(tmp_path: Path):
    catalog = tmp_path / "catalog.json"
    _write_decision_catalog(catalog, "s1", ["YES", "NO"])

    res = _fake_result_with_decisions("YES", "NO", "NO", "YES")
    gr = GroupAnalysisResult(
        group=ExperimentGroup(FP_A, {"scenario_id": "s1"}, []),
        results=[res],
    )
    with patch("agora.plotting.plot_group_response_decisions_all_repeats") as mock_plot:
        gr.plot_response_decisions_all_repeats(catalog_path=catalog)
    mock_plot.assert_called_once()


def test_plot_response_decisions_all_repeats_empty_prints_message(capsys, tmp_path: Path):
    catalog = tmp_path / "catalog.json"
    _write_decision_catalog(catalog, "s1", ["YES", "NO"])

    agora_mock = MagicMock()
    agora_mock.structured_history.return_value = {"turns": []}
    res = MagicMock(agora=agora_mock)
    gr = GroupAnalysisResult(
        group=ExperimentGroup(FP_A, {"scenario_id": "s1"}, []),
        results=[res],
    )
    gr.plot_response_decisions_all_repeats(catalog_path=catalog)
    assert "No response decision data" in capsys.readouterr().out

