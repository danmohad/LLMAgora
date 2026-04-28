import sys
from types import SimpleNamespace

import pytest

from agora import eval_aggregate
from agora.sweep_results import SweepCase


class _FakeGroupResult:
    def __init__(self):
        structured_history = {
            "turns": [
                {
                    "turn_num": 1,
                    "Alpha": {
                        "public_utterance": "PROMOTE now",
                        "private_utterance": "PROMOTE privately",
                        "public_survey": {"Q1": 1.0},
                        "private_survey": {"Q2": 0.5},
                    },
                    "Beta": {
                        "public_utterance": "DO NOT PROMOTE now",
                        "private_utterance": "DO NOT PROMOTE privately",
                        "public_survey": {"Q1": 3.0},
                        "private_survey": {"Q2": 0.6},
                    },
                },
                {
                    "turn_num": 2,
                    "Alpha": {
                        "public_utterance": "DO NOT PROMOTE later",
                        "private_utterance": "PROMOTE still",
                        "public_survey": {"Q1": 2.0},
                    },
                    "Beta": {
                        "public_utterance": "PROMOTE later",
                        "private_utterance": "DO NOT PROMOTE still",
                        "public_survey": {"Q1": 4.0},
                    },
                },
            ]
        }
        self.results = [
            SimpleNamespace(
                agents=[
                    SimpleNamespace(id="agent_alpha", name="Alpha Name"),
                    SimpleNamespace(id="agent_beta", name="Beta Name"),
                ],
                eval_data={
                    "semantic_similarity": {
                        "self_consistency": {
                            "agent_alpha": {"turns": [1, 2], "scores": [0.11, 0.22]},
                            "agent_beta": {"turns": [1, 2], "scores": [0.33, 0.44]},
                        },
                        "cross_agent_public_alignment": {"turns": [1, 2], "scores": [0.55, 0.66]},
                        "cross_agent_private_alignment": {"turns": [1, 2], "scores": [0.77, 0.88]},
                    },
                    "persona_adherence": {
                        "alpha": {
                            "public_per_turn_scores": {"turns": [1, 2], "scores": {"mean": [1.1, 2.1]}},
                            "private_per_turn_scores": {"turns": [1, 2], "scores": {"mean": [1.6, 2.6]}},
                            "public_cumulative_scores": {"turns": [1, 2], "scores": {"mean": [2.1, 3.1]}},
                            "private_cumulative_scores": {"turns": [1, 2], "scores": {"mean": [2.6, 3.6]}},
                            "full_debate_public_score": {"mean": 4.1},
                            "full_debate_private_score": {"mean": 4.6},
                        },
                        "beta": {
                            "public_per_turn_scores": {"turns": [1, 2], "scores": {"mean": [2.1, 3.1]}},
                            "private_per_turn_scores": {"turns": [1, 2], "scores": {"mean": [2.6, 3.6]}},
                            "public_cumulative_scores": {"turns": [1, 2], "scores": {"mean": [3.1, 4.1]}},
                            "private_cumulative_scores": {"turns": [1, 2], "scores": {"mean": [3.6, 4.6]}},
                            "full_debate_public_score": {"mean": 5.1},
                            "full_debate_private_score": {"mean": 5.6},
                        },
                    },
                },
                agora=SimpleNamespace(structured_history=lambda: structured_history),
                survey_question_specs=[
                    {"text": "How confident are you?", "group": "deliberative"},
                    {"text": "How positive do you feel?", "group": "incentive"},
                ],
            )
        ]
        self.survey_question_specs = [
            {"text": "How confident are you?", "group": "deliberative"},
            {"text": "How positive do you feel?", "group": "incentive"},
        ]

    def aggregate_semantic(self):
        return {
            "self_consistency": {
                "agent_alpha": {"turns": [1, 2], "mean": [0.1, 0.2], "se": [0.01, 0.02]},
                "agent_beta": {"turns": [1, 2], "mean": [0.3, 0.4], "se": [0.03, 0.04]},
            },
            "cross_agent_public_alignment": {
                "turns": [1, 2],
                "mean": [0.5, 0.6],
                "se": [0.05, 0.06],
            },
            "cross_agent_private_alignment": {
                "turns": [1, 2],
                "mean": [0.7, 0.8],
                "se": [0.07, 0.08],
            },
        }

    def aggregate_persona(self):
        role = lambda offset: {
            "public_per_turn_scores": {"turns": [1, 2], "scores": {"mean": [1.0 + offset, 2.0 + offset], "se": [0.1, 0.2]}},
            "private_per_turn_scores": {"turns": [1, 2], "scores": {"mean": [1.5 + offset, 2.5 + offset], "se": [0.15, 0.25]}},
            "public_cumulative_scores": {"turns": [1, 2], "scores": {"mean": [2.0 + offset, 3.0 + offset], "se": [0.2, 0.3]}},
            "private_cumulative_scores": {"turns": [1, 2], "scores": {"mean": [2.5 + offset, 3.5 + offset], "se": [0.25, 0.35]}},
            "full_debate_public_score": {"mean": 4.0 + offset, "se": 0.4},
            "full_debate_private_score": {"mean": 4.5 + offset, "se": 0.45},
        }
        return {"alpha": role(0.0), "beta": role(1.0)}

    def run_nli_analysis(self, model_name=None, device=None):
        assert model_name == "nli-model"
        assert device == "cpu"
        distributions = {
            "contradiction": {"mean": [0.1, 0.2], "se": [0.01, 0.02]},
            "neutral": {"mean": [0.3, 0.4], "se": [0.03, 0.04]},
            "entailment": {"mean": [0.6, 0.4], "se": [0.06, 0.04]},
        }
        return {
            "self_consistency": {
                "agent_alpha": {"turns": [1, 2], "label_names": ["contradiction", "neutral", "entailment"], "distributions": distributions},
                "agent_beta": {"turns": [1, 2], "label_names": ["contradiction", "neutral", "entailment"], "distributions": distributions},
            },
            "cross_agent_public": {"turns": [1, 2], "label_names": ["contradiction", "neutral", "entailment"], "distributions": distributions},
            "cross_agent_private": {"turns": [1, 2], "label_names": ["contradiction", "neutral", "entailment"], "distributions": distributions},
        }

    def run_emotion_analysis(self, field, model_name=None, device=None):
        assert model_name == "emotion-model"
        assert device == "cpu"
        return {
            "agent_alpha": {
                "turns": [1, 2],
                "emotions": {
                    "joy": {"mean": [0.7, 0.6], "se": [0.07, 0.06]},
                    "sadness": {"mean": [0.3, 0.4], "se": [0.03, 0.04]},
                },
            },
            "agent_beta": {
                "turns": [1, 2],
                "emotions": {
                    "joy": {"mean": [0.2, 0.1], "se": [0.02, 0.01]},
                    "sadness": {"mean": [0.8, 0.9], "se": [0.08, 0.09]},
                },
            },
        }

    def aggregate_survey(self, survey_questions=None):
        assert survey_questions == self.survey_question_specs
        return {
            "public": {
                "Alpha": {"Q1": {"turns": [1, 2], "mean": [1.0, 2.0], "se": [0.1, 0.2]}},
                "Beta": {"Q1": {"turns": [1, 2], "mean": [3.0, 4.0], "se": [0.3, 0.4]}},
            },
            "private": {
                "Alpha": {"Q2": {"turns": [1], "mean": [0.5], "se": [0.05]}},
                "Beta": {"Q2": {"turns": [1], "mean": [0.6], "se": [0.06]}},
            },
            "diff": {
                "Alpha": {"Q1": {"turns": [1, 2], "mean": [0.2, 0.3], "se": [0.02, 0.03]}},
                "Beta": {"Q1": {"turns": [1, 2], "mean": [0.1, 0.2], "se": [0.01, 0.02]}},
            },
        }

    def aggregate_response_decisions(self):
        return {
            "decision_label": "PROMOTE",
            "by_slot": {
                "Alpha": {
                    "public": {"turns": [1, 2], "mean": [1.0, 0.0], "se": [0.0, 0.1]},
                    "private": {"turns": [1, 2], "mean": [0.5, 0.5], "se": [0.2, 0.2]},
                },
                "Beta": {
                    "public": {"turns": [1, 2], "mean": [0.0, 1.0], "se": [0.1, 0.0]},
                    "private": {"turns": [1, 2], "mean": [0.4, 0.6], "se": [0.2, 0.2]},
                },
            },
        }

    def aggregate_response_decisions_all_repeats(self):
        return {
            "decision_label": "PROMOTE",
            "repeats": [
                {
                    "Alpha": {
                        "public": {"turns": [1, 2], "decisions": [1, 0]},
                        "private": {"turns": [1, 2], "decisions": [1, 1]},
                    },
                    "Beta": {
                        "public": {"turns": [1, 2], "decisions": [0, 1]},
                        "private": {"turns": [1, 2], "decisions": [0, 0]},
                    },
                }
            ],
        }


class _FakeGroup:
    def __init__(self):
        self.config_fingerprint = "fingerprint-123"
        self.repeat_count = 2
        self.sweep_values = {
            "model": "m1",
            "incentive_type": "future",
            "incentive_direction": "positive",
            "scenario_id": "scenario-1",
        }
        self.cases = [SimpleNamespace(case_id="case-a"), SimpleNamespace(case_id="case-b")]

    def run_analysis(self, sweep_root, **analysis_kwargs):
        assert str(sweep_root).endswith("sweeps_5")
        assert analysis_kwargs == {"semantic_analysis_metrics": ["self_consistency"]}
        return _FakeGroupResult()


def test_build_experiment_analysis_record_serializes_requested_sections(monkeypatch):
    fake_nli = {
        "self_consistency": {
            "agent_alpha": {
                "turns": [1, 2],
                "label_names": ["contradiction", "neutral", "entailment"],
                "distributions": {
                    "contradiction": {"mean": [0.1, 0.2], "se": [0.01, 0.02]},
                    "neutral": {"mean": [0.3, 0.4], "se": [0.03, 0.04]},
                    "entailment": {"mean": [0.6, 0.4], "se": [0.06, 0.04]},
                },
            },
            "agent_beta": {
                "turns": [1, 2],
                "label_names": ["contradiction", "neutral", "entailment"],
                "distributions": {
                    "contradiction": {"mean": [0.1, 0.2], "se": [0.01, 0.02]},
                    "neutral": {"mean": [0.3, 0.4], "se": [0.03, 0.04]},
                    "entailment": {"mean": [0.6, 0.4], "se": [0.06, 0.04]},
                },
            },
        },
        "cross_agent_public": {
            "turns": [1, 2],
            "label_names": ["contradiction", "neutral", "entailment"],
            "distributions": {
                "contradiction": {"mean": [0.1, 0.2], "se": [0.01, 0.02]},
                "neutral": {"mean": [0.3, 0.4], "se": [0.03, 0.04]},
                "entailment": {"mean": [0.6, 0.4], "se": [0.06, 0.04]},
            },
        },
        "cross_agent_private": {
            "turns": [1, 2],
            "label_names": ["contradiction", "neutral", "entailment"],
            "distributions": {
                "contradiction": {"mean": [0.1, 0.2], "se": [0.01, 0.02]},
                "neutral": {"mean": [0.3, 0.4], "se": [0.03, 0.04]},
                "entailment": {"mean": [0.6, 0.4], "se": [0.06, 0.04]},
            },
        },
    }
    fake_emotions = {
        "agent_alpha": {
            "turns": [1, 2],
            "emotions": {
                "joy": {"mean": [0.7, 0.6], "se": [0.07, 0.06]},
                "sadness": {"mean": [0.3, 0.4], "se": [0.03, 0.04]},
            },
        },
        "agent_beta": {
            "turns": [1, 2],
            "emotions": {
                "joy": {"mean": [0.2, 0.1], "se": [0.02, 0.01]},
                "sadness": {"mean": [0.8, 0.9], "se": [0.08, 0.09]},
            },
        },
    }

    monkeypatch.setattr(
        eval_aggregate.GroupAnalysisResult,
        "run_nli_analysis",
        lambda self, model_name=None, device=None: fake_nli,
    )
    monkeypatch.setattr(
        eval_aggregate.GroupAnalysisResult,
        "run_emotion_analysis",
        lambda self, field, model_name=None, device=None: fake_emotions,
    )

    row = eval_aggregate.build_experiment_analysis_record(
        _FakeGroup(),
        "/tmp/outputs/sweeps_5",
        experiment_index=3,
        analysis_kwargs={"semantic_analysis_metrics": ["self_consistency"]},
        include_nli=True,
        nli_model_name="nli-model",
        include_emotions=True,
        emotion_model_name="emotion-model",
        device="cpu",
    )

    assert row["experiment_index"] == 3
    assert row["case_ids"] == ["case-a", "case-b"]
    assert row["model"] == "m1"
    assert row["cosine-similarity-self-consistency"]["alpha"]["cosine_similarity"] == [0.1, 0.2]
    assert row["cosine-similarity-cross-agent-alignment"]["private alignment"]["standard_error"] == [0.07, 0.08]
    assert row["cosine-similarity-self-consistency-all-repeats"]["repeats"][0]["alpha"]["cosine_similarity"] == [0.11, 0.22]
    assert row["cosine-similarity-cross-agent-alignment-all-repeats"]["repeats"][0]["private alignment"]["cosine_similarity"] == [0.77, 0.88]
    assert row["persona-individual-turn-scores"]["beta"]["private"]["persona_score"] == [2.5, 3.5]
    assert row["persona-full-debate-scores"]["alpha"]["public"] == {"score": 4.0, "standard_error": 0.4}
    assert row["persona-individual-turn-scores-all-repeats"]["repeats"][0]["alpha"]["public"]["persona_score"] == [1.1, 2.1]
    assert row["persona-full-debate-scores-all-repeats"]["repeats"][0]["beta"]["private"] == {"score": 5.6}
    assert row["nli-self-consistency"]["alpha"]["nli_tuple_ordering"] == ("entailment", "neutral", "contradiction")
    assert row["nli-cross-agent-alignment"]["public utterances"]["nli_probabilities"][0] == (0.6, 0.3, 0.1)
    assert row["nli-self-consistency-all-repeats"]["repeats"][0]["alpha"]["nli_probabilities"][0] == (0.6, 0.3, 0.1)
    assert row["nli-cross-agent-alignment-all-repeats"]["repeats"][0]["private reflections"]["nli_probabilities"][1] == (0.4, 0.4, 0.2)
    assert row["emotion-public-utterances"]["alpha"]["emotion_tuple_ordering"] == ("joy", "sadness")
    assert row["emotion-public-utterances-all-repeats"]["repeats"][0]["alpha"]["emotion_probabilities"][0] == (0.7, 0.3)
    assert row["survey-public"]["Q1"]["question"] == "How confident are you?"
    assert row["survey-private"]["Q2"]["alpha"]["response_score"] == [0.5]
    assert row["survey-public-all-repeats"]["repeats"][0]["Q1"]["alpha"]["response_score"] == [1.0, 2.0]
    assert row["survey-private-all-repeats"]["repeats"][0]["Q2"]["beta"]["response_score"] == [0.6]
    assert row["decision-self-consistency"]["decision"] == "PROMOTE"
    assert row["decision-self-consistency"]["alpha"]["prob_decision"][0] == (1.0, 0.5)
    assert row["decision-cross-agent-alignment"]["public"]["prob_decision_standard_error"][1] == (0.1, 0.0)
    assert row["decision-self-consistency-all-repeats"]["repeats"][0]["alpha"]["public"]["decisions"] == [1, 0]
    assert row["decision-cross-agent-alignment-all-repeats"]["repeats"][0]["private"]["beta"]["decisions"] == [0, 0]


def test_build_records_and_dataframe_support_disabled_optional_sections(monkeypatch):
    fake_manifest = SimpleNamespace(
        sweep_root="/tmp/outputs/sweeps_5",
        __iter__=lambda self: iter([_FakeGroup()]),
    )

    class _IterableManifest:
        sweep_root = "/tmp/outputs/sweeps_5"

        def __iter__(self):
            return iter([_FakeGroup()])

    monkeypatch.setattr(eval_aggregate.SweepManifest, "from_path", lambda path: _IterableManifest())

    records = eval_aggregate.build_experiment_analysis_records(
        _IterableManifest(),
        analysis_kwargs={"semantic_analysis_metrics": ["self_consistency"]},
        include_nli=False,
        include_emotions=False,
    )
    assert len(records) == 1
    assert "nli-self-consistency" not in records[0]
    assert "emotion-public-utterances" not in records[0]

    class _FakePandas:
        @staticmethod
        def DataFrame(records_arg):
            return {"kind": "dataframe", "records": records_arg}

    monkeypatch.setitem(sys.modules, "pandas", _FakePandas())
    dataframe = eval_aggregate.build_experiment_analysis_dataframe(
        "/tmp/outputs/sweeps_5/manifest.json",
        analysis_kwargs={"semantic_analysis_metrics": ["self_consistency"]},
        include_nli=False,
        include_emotions=False,
    )
    assert dataframe["kind"] == "dataframe"
    assert dataframe["records"][0]["experiment_index"] == 0


def test_build_records_skips_empty_groups_and_uses_analyzed_case_metadata(tmp_path, capsys):
    class _PartiallySkippedGroup(_FakeGroup):
        def run_analysis(self, sweep_root, **analysis_kwargs):
            result = _FakeGroupResult()
            result.results = [result.results[0]]
            result.analyzed_cases = [SimpleNamespace(case_id="case-b")]
            return result

    class _FullySkippedGroup(_FakeGroup):
        def __init__(self):
            super().__init__()
            self.config_fingerprint = "fingerprint-empty"
            self.cases = [
                SweepCase(
                    case_id="case-empty",
                    case_dir=eval_aggregate.Path("cases/case-empty"),
                    config_path=eval_aggregate.Path("cases/case-empty/config.json"),
                    label="label",
                    repeat_number=1,
                    repeat_count=1,
                    sweep_values={},
                )
            ]

        def run_analysis(self, sweep_root, **analysis_kwargs):
            return SimpleNamespace(results=[], analyzed_cases=[])

    class _IterableManifest:
        sweep_root = tmp_path / "sweeps_5"

        def __iter__(self):
            return iter([_PartiallySkippedGroup(), _FullySkippedGroup()])

    config_dir = _IterableManifest.sweep_root / "cases" / "case-empty"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.json").write_text(
        """{
  "model": "m-empty",
  "scenario_id": "scenario-empty",
  "incentive_direction": "negative",
  "incentive_type": "historical"
}""",
        encoding="utf-8",
    )

    records = eval_aggregate.build_experiment_analysis_records(
        _IterableManifest(),
        analysis_kwargs={"semantic_analysis_metrics": ["self_consistency"]},
        include_nli=False,
        include_emotions=False,
    )

    captured = capsys.readouterr()
    assert len(records) == 1
    assert records[0]["repeat_count"] == 1
    assert records[0]["case_ids"] == ["case-b"]
    assert "Warning: skipping experiment group fingerprint-empty" in captured.out
    assert "model='m-empty'" in captured.out
    assert "scenario_id='scenario-empty'" in captured.out
    assert "incentive_direction='negative'" in captured.out
    assert "incentive_type='historical'" in captured.out


def test_build_record_skips_empty_group_without_cases(capsys):
    class _EmptyGroup:
        config_fingerprint = "fp-no-cases"
        cases = []
        sweep_values = {}

        def run_analysis(self, sweep_root, **analysis_kwargs):
            return SimpleNamespace(results=[], analyzed_cases=[])

    record = eval_aggregate.build_experiment_analysis_record(
        _EmptyGroup(),
        "/tmp/outputs/sweeps_5",
        experiment_index=0,
        analysis_kwargs={},
        include_nli=False,
        include_emotions=False,
    )

    captured = capsys.readouterr()
    assert record == {}
    assert "Warning: skipping experiment group fp-no-cases because no analyzable cases remained." in captured.out


def test_internal_helpers_cover_fallback_and_sparse_paths():
    assert eval_aggregate._agent_slot_lookup(SimpleNamespace(results=[]))["Alpha"] == "alpha"

    semantic_result = SimpleNamespace(
        results=[],
        aggregate_semantic=lambda: {
            "self_consistency": {"unknown-agent": {"turns": [1], "mean": [0.9], "se": [0.09]}},
        },
    )
    semantic_self, semantic_cross = eval_aggregate._serialize_semantic(semantic_result)
    assert semantic_self == {"alpha": {}, "beta": {}}
    assert semantic_cross["public alignment"]["debate_turn"] == []

    nli_result = SimpleNamespace(
        results=[],
        run_nli_analysis=lambda model_name=None, device=None: {
            "self_consistency": {
                "unknown-agent": {
                    "turns": [1],
                    "label_names": ["neutral"],
                    "distributions": {"neutral": {"mean": [1.0], "se": [0.0]}},
                }
            }
        },
    )
    nli_self, nli_cross = eval_aggregate._serialize_nli(nli_result, model_name=None, device=None)
    assert nli_self == {"alpha": {}, "beta": {}}
    assert nli_cross == {}

    emotion_result = SimpleNamespace(
        results=[],
        run_emotion_analysis=lambda field, model_name=None, device=None: {
            "unknown-agent": {
                "turns": [1],
                "emotions": {"calm": {"mean": [1.0], "se": [0.0]}},
            }
        },
    )
    emotions_public, emotions_private = eval_aggregate._serialize_emotions(
        emotion_result,
        model_name=None,
        device=None,
    )
    assert emotions_public == {"alpha": {}, "beta": {}}
    assert emotions_private == {"alpha": {}, "beta": {}}

    paired = eval_aggregate._pair_series(
        {"turns": [1], "mean": [0.2], "se": [0.02]},
        {"turns": [2], "mean": [0.8], "se": [0.08]},
        value_key="prob_decision",
    )
    assert paired["prob_decision"] == [(0.2, None), (None, 0.8)]


def test_all_repeat_helpers_cover_unknown_agents_and_case_id_metadata(monkeypatch):
    semantic_group_result = SimpleNamespace(
        results=[
            SimpleNamespace(
                eval_data={
                    "semantic_similarity": {
                        "self_consistency": {
                            "unknown-agent": {"turns": [1], "scores": [0.9]},
                        }
                    }
                }
            )
        ],
        analyzed_cases=[SimpleNamespace(case_id="case-1")],
    )
    semantic_self, _ = eval_aggregate._serialize_semantic_all_repeats(semantic_group_result)
    assert semantic_self["repeats"][0]["case_id"] == "case-1"
    assert semantic_self["repeats"][0]["alpha"] == {}
    assert semantic_self["repeats"][0]["beta"] == {}

    fake_nli = {
        "self_consistency": {
            "agent_alpha": {
                "turns": [1],
                "label_names": ["neutral"],
                "distributions": {"neutral": {"mean": [1.0], "se": [0.0]}},
            }
        }
    }
    fake_emotions = {
        "agent_alpha": {
            "turns": [1],
            "emotions": {"calm": {"mean": [1.0], "se": [0.0]}},
        }
    }
    monkeypatch.setattr(
        eval_aggregate.GroupAnalysisResult,
        "run_nli_analysis",
        lambda self, model_name=None, device=None: fake_nli,
    )
    monkeypatch.setattr(
        eval_aggregate.GroupAnalysisResult,
        "run_emotion_analysis",
        lambda self, field, model_name=None, device=None: fake_emotions,
    )

    repeat_group_result = SimpleNamespace(
        group=None,
        results=[SimpleNamespace()],
        analyzed_cases=[SimpleNamespace(case_id="case-2")],
    )
    nli_self, _ = eval_aggregate._serialize_nli_all_repeats(
        repeat_group_result,
        model_name="ignored",
        device="cpu",
    )
    emotions_public, _ = eval_aggregate._serialize_emotions_all_repeats(
        repeat_group_result,
        model_name="ignored",
        device="cpu",
    )
    assert nli_self["repeats"][0]["case_id"] == "case-2"
    assert emotions_public["repeats"][0]["case_id"] == "case-2"
