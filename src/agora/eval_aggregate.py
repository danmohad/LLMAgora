"""Helpers for building one-row-per-experiment aggregate analysis tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .emotion_analyzer import PRIVATE_NARRATIVE_FIELD, PUBLIC_NARRATIVE_FIELD
from .semantic_similarity_analyzer import DEFAULT_NLI_MODEL_NAME
from .sweep_results import GroupAnalysisResult, SweepManifest, format_case_warning_context


def _agent_slot_lookup(group_result: Any) -> dict[str, str]:
    lookup = {
        "alpha": "alpha",
        "beta": "beta",
        "Alpha": "alpha",
        "Beta": "beta",
    }
    results = getattr(group_result, "results", [])
    if not results:
        return lookup
    agents = getattr(results[0], "agents", [])
    if len(agents) >= 1:
        lookup[getattr(agents[0], "id", "alpha")] = "alpha"
        lookup[getattr(agents[0], "name", "alpha")] = "alpha"
    if len(agents) >= 2:
        lookup[getattr(agents[1], "id", "beta")] = "beta"
        lookup[getattr(agents[1], "name", "beta")] = "beta"
    return lookup


def _line_payload(
    turns: list[int] | None,
    values: list[float] | None,
    errors: list[float] | None,
    *,
    x_key: str,
    y_key: str,
    error_key: str,
) -> dict[str, list[float] | list[int]]:
    return {
        x_key: list(turns or []),
        y_key: list(values or []),
        error_key: list(errors or []),
    }


def _tuple_series_payload(
    turns: list[int] | None,
    values_by_name: dict[str, dict[str, list[float]]],
    ordering: list[str],
    *,
    tuple_key: str,
    error_key: str,
) -> dict[str, Any]:
    turn_list = list(turns or [])
    tuples = [
        tuple(float(values_by_name[name]["mean"][i]) for name in ordering)
        for i in range(len(turn_list))
    ]
    error_tuples = [
        tuple(float(values_by_name[name]["se"][i]) for name in ordering)
        for i in range(len(turn_list))
    ]
    return {
        "debate_turns": turn_list,
        tuple_key: tuples,
        error_key: error_tuples,
    }


def _series_from_turn_scores(series: dict[str, Any], *, value_key: str) -> dict[str, Any]:
    scores = series.get("scores", {}) if isinstance(series, dict) else {}
    return _line_payload(
        series.get("turns", []),
        scores.get("mean", []),
        scores.get("se", []),
        x_key="debate_turn",
        y_key=value_key,
        error_key="standard_error",
    )


def _scalar_payload(score: dict[str, Any]) -> dict[str, float | None]:
    return {
        "score": None if score.get("mean") is None else float(score.get("mean")),
        "standard_error": None if score.get("se") is None else float(score.get("se")),
    }


def _scalar_repeat_payload(score: dict[str, Any]) -> dict[str, float | None]:
    value = None
    if isinstance(score, dict):
        raw_value = score.get("mean", score.get("score"))
        if raw_value is not None:
            value = float(raw_value)
    return {"score": value}


def _serialize_semantic(group_result: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    semantic = group_result.aggregate_semantic()
    slot_lookup = _agent_slot_lookup(group_result)

    self_consistency = {"alpha": {}, "beta": {}}
    for agent_key, payload in (semantic.get("self_consistency") or {}).items():
        slot = slot_lookup.get(agent_key)
        if slot is None:
            continue
        self_consistency[slot] = _line_payload(
            payload.get("turns", []),
            payload.get("mean", []),
            payload.get("se", []),
            x_key="debate_turn",
            y_key="cosine_similarity",
            error_key="standard_error",
        )

    cross_agent = {
        "public alignment": _line_payload(
            (semantic.get("cross_agent_public_alignment") or {}).get("turns", []),
            (semantic.get("cross_agent_public_alignment") or {}).get("mean", []),
            (semantic.get("cross_agent_public_alignment") or {}).get("se", []),
            x_key="debate_turn",
            y_key="cosine_similarity",
            error_key="standard_error",
        ),
        "private alignment": _line_payload(
            (semantic.get("cross_agent_private_alignment") or {}).get("turns", []),
            (semantic.get("cross_agent_private_alignment") or {}).get("mean", []),
            (semantic.get("cross_agent_private_alignment") or {}).get("se", []),
            x_key="debate_turn",
            y_key="cosine_similarity",
            error_key="standard_error",
        ),
    }
    return self_consistency, cross_agent


def _serialize_semantic_all_repeats(group_result: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    slot_lookup = _agent_slot_lookup(group_result)
    self_consistency = {"repeats": []}
    cross_agent = {"repeats": []}

    analyzed_cases = list(getattr(group_result, "analyzed_cases", []) or [])
    for repeat_index, res in enumerate(getattr(group_result, "results", []), start=1):
        repeat_meta = {"repeat_number": repeat_index}
        if repeat_index - 1 < len(analyzed_cases):
            repeat_meta["case_id"] = analyzed_cases[repeat_index - 1].case_id
        sem = (getattr(res, "eval_data", {}) or {}).get("semantic_similarity") or {}

        repeat_self = {"alpha": {}, "beta": {}, **repeat_meta}
        for agent_key, payload in (sem.get("self_consistency") or {}).items():
            slot = slot_lookup.get(agent_key)
            if slot is None:
                continue
            repeat_self[slot] = _line_payload(
                payload.get("turns", []),
                payload.get("scores", []),
                None,
                x_key="debate_turn",
                y_key="cosine_similarity",
                error_key="standard_error",
            )
        self_consistency["repeats"].append(repeat_self)

        repeat_cross = {
            "public alignment": _line_payload(
                (sem.get("cross_agent_public_alignment") or {}).get("turns", []),
                (sem.get("cross_agent_public_alignment") or {}).get("scores", []),
                None,
                x_key="debate_turn",
                y_key="cosine_similarity",
                error_key="standard_error",
            ),
            "private alignment": _line_payload(
                (sem.get("cross_agent_private_alignment") or {}).get("turns", []),
                (sem.get("cross_agent_private_alignment") or {}).get("scores", []),
                None,
                x_key="debate_turn",
                y_key="cosine_similarity",
                error_key="standard_error",
            ),
            **repeat_meta,
        }
        cross_agent["repeats"].append(repeat_cross)
    return self_consistency, cross_agent


def _serialize_persona(group_result: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    persona = group_result.aggregate_persona()
    individual = {
        "alpha": {
            "public": _series_from_turn_scores(
                (persona.get("alpha") or {}).get("public_per_turn_scores", {}),
                value_key="persona_score",
            ),
            "private": _series_from_turn_scores(
                (persona.get("alpha") or {}).get("private_per_turn_scores", {}),
                value_key="persona_score",
            ),
        },
        "beta": {
            "public": _series_from_turn_scores(
                (persona.get("beta") or {}).get("public_per_turn_scores", {}),
                value_key="persona_score",
            ),
            "private": _series_from_turn_scores(
                (persona.get("beta") or {}).get("private_per_turn_scores", {}),
                value_key="persona_score",
            ),
        },
    }
    cumulative = {
        "alpha": {
            "public": _series_from_turn_scores(
                (persona.get("alpha") or {}).get("public_cumulative_scores", {}),
                value_key="persona_score",
            ),
            "private": _series_from_turn_scores(
                (persona.get("alpha") or {}).get("private_cumulative_scores", {}),
                value_key="persona_score",
            ),
        },
        "beta": {
            "public": _series_from_turn_scores(
                (persona.get("beta") or {}).get("public_cumulative_scores", {}),
                value_key="persona_score",
            ),
            "private": _series_from_turn_scores(
                (persona.get("beta") or {}).get("private_cumulative_scores", {}),
                value_key="persona_score",
            ),
        },
    }
    full_debate = {
        "alpha": {
            "public": _scalar_payload((persona.get("alpha") or {}).get("full_debate_public_score", {})),
            "private": _scalar_payload((persona.get("alpha") or {}).get("full_debate_private_score", {})),
        },
        "beta": {
            "public": _scalar_payload((persona.get("beta") or {}).get("full_debate_public_score", {})),
            "private": _scalar_payload((persona.get("beta") or {}).get("full_debate_private_score", {})),
        },
    }
    return individual, cumulative, full_debate


def _serialize_persona_all_repeats(group_result: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    analyzed_cases = list(getattr(group_result, "analyzed_cases", []) or [])
    individual = {"repeats": []}
    cumulative = {"repeats": []}
    full_debate = {"repeats": []}

    for repeat_index, res in enumerate(getattr(group_result, "results", []), start=1):
        repeat_meta = {"repeat_number": repeat_index}
        if repeat_index - 1 < len(analyzed_cases):
            repeat_meta["case_id"] = analyzed_cases[repeat_index - 1].case_id
        persona = (getattr(res, "eval_data", {}) or {}).get("persona_adherence") or {}
        individual["repeats"].append(
            {
                "alpha": {
                    "public": _series_from_turn_scores(
                        (persona.get("alpha") or {}).get("public_per_turn_scores", {}),
                        value_key="persona_score",
                    ),
                    "private": _series_from_turn_scores(
                        (persona.get("alpha") or {}).get("private_per_turn_scores", {}),
                        value_key="persona_score",
                    ),
                },
                "beta": {
                    "public": _series_from_turn_scores(
                        (persona.get("beta") or {}).get("public_per_turn_scores", {}),
                        value_key="persona_score",
                    ),
                    "private": _series_from_turn_scores(
                        (persona.get("beta") or {}).get("private_per_turn_scores", {}),
                        value_key="persona_score",
                    ),
                },
                **repeat_meta,
            }
        )
        cumulative["repeats"].append(
            {
                "alpha": {
                    "public": _series_from_turn_scores(
                        (persona.get("alpha") or {}).get("public_cumulative_scores", {}),
                        value_key="persona_score",
                    ),
                    "private": _series_from_turn_scores(
                        (persona.get("alpha") or {}).get("private_cumulative_scores", {}),
                        value_key="persona_score",
                    ),
                },
                "beta": {
                    "public": _series_from_turn_scores(
                        (persona.get("beta") or {}).get("public_cumulative_scores", {}),
                        value_key="persona_score",
                    ),
                    "private": _series_from_turn_scores(
                        (persona.get("beta") or {}).get("private_cumulative_scores", {}),
                        value_key="persona_score",
                    ),
                },
                **repeat_meta,
            }
        )
        full_debate["repeats"].append(
            {
                "alpha": {
                    "public": _scalar_repeat_payload((persona.get("alpha") or {}).get("full_debate_public_score", {})),
                    "private": _scalar_repeat_payload((persona.get("alpha") or {}).get("full_debate_private_score", {})),
                },
                "beta": {
                    "public": _scalar_repeat_payload((persona.get("beta") or {}).get("full_debate_public_score", {})),
                    "private": _scalar_repeat_payload((persona.get("beta") or {}).get("full_debate_private_score", {})),
                },
                **repeat_meta,
            }
        )
    return individual, cumulative, full_debate


def _reordered_nli_payload(payload: dict[str, Any]) -> dict[str, Any]:
    distributions = payload.get("distributions", {})
    available = list(payload.get("label_names", []))
    preferred = ["entailment", "neutral", "contradiction"]
    ordering = [label for label in preferred if label in available] or available
    result = _tuple_series_payload(
        payload.get("turns", []),
        distributions,
        ordering,
        tuple_key="nli_probabilities",
        error_key="nli_probabilities_standard_error",
    )
    result["nli_tuple_ordering"] = tuple(ordering)
    return result


def _serialize_nli(group_result: Any, *, model_name: str | None, device: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
    nli = group_result.run_nli_analysis(model_name=model_name, device=device)
    slot_lookup = _agent_slot_lookup(group_result)

    self_consistency = {"alpha": {}, "beta": {}}
    for agent_key, payload in (nli.get("self_consistency") or {}).items():
        slot = slot_lookup.get(agent_key)
        if slot is None:
            continue
        self_consistency[slot] = _reordered_nli_payload(payload)

    cross_agent = {}
    if nli.get("cross_agent_public"):
        cross_agent["public utterances"] = _reordered_nli_payload(nli["cross_agent_public"])
    if nli.get("cross_agent_private"):
        cross_agent["private reflections"] = _reordered_nli_payload(nli["cross_agent_private"])
    return self_consistency, cross_agent


def _serialize_nli_all_repeats(
    group_result: Any,
    *,
    model_name: str | None,
    device: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    analyzed_cases = list(getattr(group_result, "analyzed_cases", []) or [])
    self_consistency = {"repeats": []}
    cross_agent = {"repeats": []}
    base_group = getattr(group_result, "group", None)

    for repeat_index, res in enumerate(getattr(group_result, "results", []), start=1):
        repeat_meta = {"repeat_number": repeat_index}
        if repeat_index - 1 < len(analyzed_cases):
            repeat_meta["case_id"] = analyzed_cases[repeat_index - 1].case_id
        single = GroupAnalysisResult(
            group=base_group,
            results=[res],
            analyzed_cases=analyzed_cases[repeat_index - 1:repeat_index],
        )
        repeat_self, repeat_cross = _serialize_nli(single, model_name=model_name, device=device)
        self_consistency["repeats"].append({**repeat_self, **repeat_meta})
        cross_agent["repeats"].append({**repeat_cross, **repeat_meta})
    return self_consistency, cross_agent


def _serialize_emotions(group_result: Any, *, model_name: str | None, device: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
    slot_lookup = _agent_slot_lookup(group_result)

    def _one(field: str) -> dict[str, Any]:
        payload = group_result.run_emotion_analysis(field=field, model_name=model_name, device=device)
        result = {"alpha": {}, "beta": {}}
        for agent_key, agent_payload in (payload or {}).items():
            slot = slot_lookup.get(agent_key)
            if slot is None:
                continue
            ordering = sorted((agent_payload.get("emotions") or {}).keys())
            series = _tuple_series_payload(
                agent_payload.get("turns", []),
                agent_payload.get("emotions", {}),
                ordering,
                tuple_key="emotion_probabilities",
                error_key="emotion_probabilities_standard_error",
            )
            series["emotion_tuple_ordering"] = tuple(ordering)
            result[slot] = series
        return result

    return _one(PUBLIC_NARRATIVE_FIELD), _one(PRIVATE_NARRATIVE_FIELD)


def _serialize_emotions_all_repeats(
    group_result: Any,
    *,
    model_name: str | None,
    device: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    analyzed_cases = list(getattr(group_result, "analyzed_cases", []) or [])
    public = {"repeats": []}
    private = {"repeats": []}
    base_group = getattr(group_result, "group", None)

    for repeat_index, res in enumerate(getattr(group_result, "results", []), start=1):
        repeat_meta = {"repeat_number": repeat_index}
        if repeat_index - 1 < len(analyzed_cases):
            repeat_meta["case_id"] = analyzed_cases[repeat_index - 1].case_id
        single = GroupAnalysisResult(
            group=base_group,
            results=[res],
            analyzed_cases=analyzed_cases[repeat_index - 1:repeat_index],
        )
        repeat_public, repeat_private = _serialize_emotions(
            single,
            model_name=model_name,
            device=device,
        )
        public["repeats"].append({**repeat_public, **repeat_meta})
        private["repeats"].append({**repeat_private, **repeat_meta})
    return public, private


def _question_text_map(group_result: Any) -> dict[str, dict[str, Any]]:
    specs = getattr(group_result, "survey_question_specs", []) or []
    return {
        f"Q{index}": {
            "question": spec.get("text"),
            "question_group": spec.get("group"),
        }
        for index, spec in enumerate(specs, start=1)
        if isinstance(spec, dict)
    }


def _serialize_survey_channel(channel_payload: dict[str, Any], question_meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for q_key in sorted({q for slot_payload in channel_payload.values() for q in slot_payload}):
        result[q_key] = {
            "question": question_meta.get(q_key, {}).get("question"),
            "question_group": question_meta.get(q_key, {}).get("question_group"),
            "alpha": _line_payload([], [], [], x_key="debate_turn", y_key="response_score", error_key="standard_error"),
            "beta": _line_payload([], [], [], x_key="debate_turn", y_key="response_score", error_key="standard_error"),
        }
        for slot_name, slot_key in (("Alpha", "alpha"), ("Beta", "beta")):
            series = (channel_payload.get(slot_name) or {}).get(q_key, {})
            result[q_key][slot_key] = _line_payload(
                series.get("turns", []),
                series.get("mean", []),
                series.get("se", []),
                x_key="debate_turn",
                y_key="response_score",
                error_key="standard_error",
            )
    return result


def _serialize_survey(group_result: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    survey_questions = group_result.survey_question_specs or None
    aggregated = group_result.aggregate_survey(survey_questions=survey_questions)
    question_meta = _question_text_map(group_result)
    return (
        _serialize_survey_channel(aggregated.get("public", {}), question_meta),
        _serialize_survey_channel(aggregated.get("private", {}), question_meta),
        _serialize_survey_channel(aggregated.get("diff", {}), question_meta),
    )


def _serialize_survey_all_repeats(group_result: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    analyzed_cases = list(getattr(group_result, "analyzed_cases", []) or [])
    public = {"repeats": []}
    private = {"repeats": []}
    diff = {"repeats": []}
    base_group = getattr(group_result, "group", None)

    for repeat_index, res in enumerate(getattr(group_result, "results", []), start=1):
        repeat_meta = {"repeat_number": repeat_index}
        if repeat_index - 1 < len(analyzed_cases):
            repeat_meta["case_id"] = analyzed_cases[repeat_index - 1].case_id
        single = GroupAnalysisResult(
            group=base_group,
            results=[res],
            analyzed_cases=analyzed_cases[repeat_index - 1:repeat_index],
        )
        repeat_public, repeat_private, repeat_diff = _serialize_survey(single)
        public["repeats"].append({**repeat_public, **repeat_meta})
        private["repeats"].append({**repeat_private, **repeat_meta})
        diff["repeats"].append({**repeat_diff, **repeat_meta})
    return public, private, diff


def _pair_series(left: dict[str, Any], right: dict[str, Any], *, value_key: str) -> dict[str, Any]:
    turns = sorted(set(left.get("turns", [])) | set(right.get("turns", [])))

    def _value_at(series: dict[str, Any], key: str, turn: int) -> float | None:
        series_turns = list(series.get("turns", []))
        if turn not in series_turns:
            return None
        idx = series_turns.index(turn)
        values = list(series.get(key, []))
        return float(values[idx])

    return {
        "debate_turns": turns,
        value_key: [(_value_at(left, "mean", turn), _value_at(right, "mean", turn)) for turn in turns],
        f"{value_key}_standard_error": [(_value_at(left, "se", turn), _value_at(right, "se", turn)) for turn in turns],
    }


def _serialize_decisions(group_result: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    decisions = group_result.aggregate_response_decisions()
    by_slot = decisions.get("by_slot", {})
    self_consistency = {
        "decision": decisions.get("decision_label"),
        "channel_tuple_ordering": ("public", "private"),
        "alpha": _pair_series(
            (by_slot.get("Alpha") or {}).get("public", {}),
            (by_slot.get("Alpha") or {}).get("private", {}),
            value_key="prob_decision",
        ),
        "beta": _pair_series(
            (by_slot.get("Beta") or {}).get("public", {}),
            (by_slot.get("Beta") or {}).get("private", {}),
            value_key="prob_decision",
        ),
    }
    cross_agent = {
        "decision": decisions.get("decision_label"),
        "agent_tuple_ordering": ("alpha", "beta"),
        "public": _pair_series(
            (by_slot.get("Alpha") or {}).get("public", {}),
            (by_slot.get("Beta") or {}).get("public", {}),
            value_key="prob_decision",
        ),
        "private": _pair_series(
            (by_slot.get("Alpha") or {}).get("private", {}),
            (by_slot.get("Beta") or {}).get("private", {}),
            value_key="prob_decision",
        ),
    }
    return self_consistency, cross_agent


def _serialize_decisions_all_repeats(group_result: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    decisions = group_result.aggregate_response_decisions_all_repeats()
    self_consistency = {
        "decision": decisions.get("decision_label"),
        "channel_tuple_ordering": ("public", "private"),
        "repeats": [],
    }
    cross_agent = {
        "decision": decisions.get("decision_label"),
        "agent_tuple_ordering": ("alpha", "beta"),
        "repeats": [],
    }
    for repeat_index, repeat_payload in enumerate(decisions.get("repeats", []), start=1):
        alpha_payload = repeat_payload.get("Alpha", {})
        beta_payload = repeat_payload.get("Beta", {})
        self_consistency["repeats"].append(
            {
                "repeat_number": repeat_index,
                "alpha": {
                    "public": alpha_payload.get("public", {}),
                    "private": alpha_payload.get("private", {}),
                },
                "beta": {
                    "public": beta_payload.get("public", {}),
                    "private": beta_payload.get("private", {}),
                },
            }
        )
        cross_agent["repeats"].append(
            {
                "repeat_number": repeat_index,
                "public": {
                    "alpha": alpha_payload.get("public", {}),
                    "beta": beta_payload.get("public", {}),
                },
                "private": {
                    "alpha": alpha_payload.get("private", {}),
                    "beta": beta_payload.get("private", {}),
                },
            }
        )
    return self_consistency, cross_agent


def build_experiment_analysis_record(
    group: Any,
    sweep_root: Path | str,
    *,
    experiment_index: int,
    analysis_kwargs: dict[str, Any] | None = None,
    include_nli: bool = True,
    nli_model_name: str | None = DEFAULT_NLI_MODEL_NAME,
    include_emotions: bool = True,
    emotion_model_name: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Build a single dataframe-ready row for one experiment group."""
    group_result = group.run_analysis(Path(sweep_root), **(analysis_kwargs or {}))
    analyzed_cases = list(getattr(group_result, "analyzed_cases", []) or group.cases)
    if not group_result.results:
        context = ""
        if getattr(group, "cases", None):
            context = format_case_warning_context(group.cases[0], Path(sweep_root))
        print(
            "Warning: skipping experiment group "
            f"{group.config_fingerprint}{context} because no analyzable cases remained."
        )
        return {}

    cosine_self, cosine_cross = _serialize_semantic(group_result)
    cosine_self_repeats, cosine_cross_repeats = _serialize_semantic_all_repeats(group_result)
    persona_individual, persona_cumulative, persona_full = _serialize_persona(group_result)
    persona_individual_repeats, persona_cumulative_repeats, persona_full_repeats = _serialize_persona_all_repeats(group_result)
    survey_public, survey_private, survey_diff = _serialize_survey(group_result)
    survey_public_repeats, survey_private_repeats, survey_diff_repeats = _serialize_survey_all_repeats(group_result)
    decision_self, decision_cross = _serialize_decisions(group_result)
    decision_self_repeats, decision_cross_repeats = _serialize_decisions_all_repeats(group_result)

    row = {
        "experiment_index": int(experiment_index),
        "config_fingerprint": group.config_fingerprint,
        "repeat_count": int(len(analyzed_cases)),
        "case_ids": [case.case_id for case in analyzed_cases],
        **dict(group.sweep_values or {}),
        "cosine-similarity-self-consistency": cosine_self,
        "cosine-similarity-cross-agent-alignment": cosine_cross,
        "cosine-similarity-self-consistency-all-repeats": cosine_self_repeats,
        "cosine-similarity-cross-agent-alignment-all-repeats": cosine_cross_repeats,
        "persona-individual-turn-scores": persona_individual,
        "persona-cumulative-scores": persona_cumulative,
        "persona-full-debate-scores": persona_full,
        "persona-individual-turn-scores-all-repeats": persona_individual_repeats,
        "persona-cumulative-scores-all-repeats": persona_cumulative_repeats,
        "persona-full-debate-scores-all-repeats": persona_full_repeats,
        "survey-public": survey_public,
        "survey-private": survey_private,
        "survey-diff-public-minus-private": survey_diff,
        "survey-public-all-repeats": survey_public_repeats,
        "survey-private-all-repeats": survey_private_repeats,
        "survey-diff-public-minus-private-all-repeats": survey_diff_repeats,
        "decision-self-consistency": decision_self,
        "decision-cross-agent-alignment": decision_cross,
        "decision-self-consistency-all-repeats": decision_self_repeats,
        "decision-cross-agent-alignment-all-repeats": decision_cross_repeats,
    }
    if include_nli:
        nli_self, nli_cross = _serialize_nli(
            group_result,
            model_name=nli_model_name,
            device=device,
        )
        nli_self_repeats, nli_cross_repeats = _serialize_nli_all_repeats(
            group_result,
            model_name=nli_model_name,
            device=device,
        )
        row["nli-self-consistency"] = nli_self
        row["nli-cross-agent-alignment"] = nli_cross
        row["nli-self-consistency-all-repeats"] = nli_self_repeats
        row["nli-cross-agent-alignment-all-repeats"] = nli_cross_repeats
    if include_emotions:
        emotions_public, emotions_private = _serialize_emotions(
            group_result,
            model_name=emotion_model_name,
            device=device,
        )
        emotions_public_repeats, emotions_private_repeats = _serialize_emotions_all_repeats(
            group_result,
            model_name=emotion_model_name,
            device=device,
        )
        row["emotion-public-utterances"] = emotions_public
        row["emotion-private-reflections"] = emotions_private
        row["emotion-public-utterances-all-repeats"] = emotions_public_repeats
        row["emotion-private-reflections-all-repeats"] = emotions_private_repeats
    return row


def build_experiment_analysis_records(
    manifest: SweepManifest,
    *,
    analysis_kwargs: dict[str, Any] | None = None,
    include_nli: bool = True,
    nli_model_name: str | None = DEFAULT_NLI_MODEL_NAME,
    include_emotions: bool = True,
    emotion_model_name: str | None = None,
    device: str | None = None,
) -> list[dict[str, Any]]:
    """Build one aggregate row per experiment group in a sweep manifest."""
    records = [
        build_experiment_analysis_record(
            group,
            manifest.sweep_root,
            experiment_index=index,
            analysis_kwargs=analysis_kwargs,
            include_nli=include_nli,
            nli_model_name=nli_model_name,
            include_emotions=include_emotions,
            emotion_model_name=emotion_model_name,
            device=device,
        )
        for index, group in enumerate(manifest)
    ]
    return [record for record in records if record]


def build_experiment_analysis_dataframe(
    manifest_path: Path | str,
    *,
    analysis_kwargs: dict[str, Any] | None = None,
    include_nli: bool = True,
    nli_model_name: str | None = DEFAULT_NLI_MODEL_NAME,
    include_emotions: bool = True,
    emotion_model_name: str | None = None,
    device: str | None = None,
) -> Any:
    """Build the aggregate dataframe for a sweep manifest.

    Pandas is imported lazily so the base package does not require it.
    """
    import pandas as pd

    manifest = SweepManifest.from_path(manifest_path)
    records = build_experiment_analysis_records(
        manifest,
        analysis_kwargs=analysis_kwargs,
        include_nli=include_nli,
        nli_model_name=nli_model_name,
        include_emotions=include_emotions,
        emotion_model_name=emotion_model_name,
        device=device,
    )
    return pd.DataFrame(records)
