"""Helpers for building one-row-per-experiment aggregate analysis tables."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

from .emotion_analyzer import PRIVATE_NARRATIVE_FIELD, PUBLIC_NARRATIVE_FIELD
from .semantic_similarity_analyzer import (
    DEFAULT_NLI_MODEL_NAME,
    SEMANTIC_SIMILARITY_METHOD_COSINE,
    SemanticSimilarityAnalyzer,
)
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


def _strip_leading_stance(text: str, decision_labels: list[str]) -> str:
    stripped = text.strip()
    for label in sorted(decision_labels, key=len, reverse=True):
        if stripped.upper().startswith(label.upper()):
            remainder = stripped[len(label):].lstrip()
            while remainder.startswith(("-", ":", ".", ",")):
                remainder = remainder[1:].lstrip()
            return remainder
    return text


def _resolve_optional_config_path(value: Any, *, base_dir: Path | None) -> Path | None:
    if value in (None, ""):
        return None
    path = value if isinstance(value, Path) else Path(str(value))
    if path.is_absolute() or base_dir is None:
        return path
    return (base_dir / path).resolve()


def _first_case_config_payload(
    group: Any,
    sweep_root: Path,
) -> tuple[dict[str, Any], Path | None]:
    cases = list(getattr(group, "cases", []) or [])
    if not cases:
        return {}, None
    config_path = getattr(cases[0], "config_path", None)
    if config_path is None:
        return {}, None
    abs_config_path = sweep_root / config_path
    payload = json.loads(abs_config_path.read_text(encoding="utf-8"))
    return payload, abs_config_path.parent


def _group_decision_context(
    group: Any,
    sweep_root: Path,
    analysis_kwargs: dict[str, Any],
) -> tuple[str | None, Path | None]:
    payload, config_dir = _first_case_config_payload(group, sweep_root)
    scenario_id = (getattr(group, "sweep_values", {}) or {}).get("scenario_id")
    if not scenario_id:
        scenario_id = payload.get("scenario_id")
    catalog_path = _resolve_optional_config_path(
        analysis_kwargs.get("catalog_path", payload.get("catalog_path")),
        base_dir=config_dir,
    )
    return scenario_id, catalog_path


def _decision_labels_for_group(
    group: Any,
    sweep_root: Path,
    *,
    scenario_id: str | None = None,
    catalog_path: Path | str | None = None,
) -> list[str]:
    payload, config_dir = _first_case_config_payload(group, sweep_root)
    effective_scenario_id = scenario_id or (
        getattr(group, "sweep_values", {}) or {}
    ).get("scenario_id")
    if not effective_scenario_id:
        effective_scenario_id = payload.get("scenario_id")
    if not effective_scenario_id:
        return []

    effective_catalog = _resolve_optional_config_path(catalog_path, base_dir=config_dir)
    if effective_catalog is None:
        effective_catalog = _resolve_optional_config_path(
            payload.get("catalog_path"),
            base_dir=config_dir,
        )
    if effective_catalog is None:
        from .experiment import DEFAULT_CATALOG_PATH

        effective_catalog = DEFAULT_CATALOG_PATH

    catalog = json.loads(effective_catalog.read_text(encoding="utf-8"))
    scenario = next(
        (
            item
            for item in catalog.get("scenarios", [])
            if item.get("scenario_id") == effective_scenario_id
        ),
        None,
    )
    labels = scenario.get("decision_labels", []) if isinstance(scenario, dict) else []
    return [label for label in labels if isinstance(label, str) and label.strip()]


def _strip_stance_from_structured_history(history: dict[str, Any], decision_labels: list[str]) -> dict[str, Any]:
    if not decision_labels:
        return history
    stripped_history = copy.deepcopy(history)
    for turn in stripped_history.get("turns", []):
        if not isinstance(turn, dict):
            continue
        for slot in ("Alpha", "Beta"):
            subturn = turn.get(slot)
            if not isinstance(subturn, dict):
                continue
            for field in ("public_utterance", "private_utterance"):
                value = subturn.get(field)
                if isinstance(value, str):
                    subturn[field] = _strip_leading_stance(value, decision_labels)
    return stripped_history


class _HistoryView:
    def __init__(self, history: dict[str, Any]) -> None:
        self._history = history

    def structured_history(self) -> dict[str, Any]:
        return self._history


def _no_stance_group_result(group_result: Any, decision_labels: list[str]) -> GroupAnalysisResult:
    stripped_results = []
    for res in getattr(group_result, "results", []):
        history = res.agora.structured_history()
        stripped_results.append(
            copy.copy(res)
        )
        stripped_results[-1].agora = _HistoryView(
            _strip_stance_from_structured_history(history, decision_labels)
        )
    return GroupAnalysisResult(
        group=getattr(group_result, "group", None),
        results=stripped_results,
        analyzed_cases=list(getattr(group_result, "analyzed_cases", []) or []),
    )


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


def _aggregate_turn_scores(series_list: list[dict[str, Any]]) -> dict[str, Any]:
    by_turn: dict[int, list[float]] = {}
    for series in series_list:
        for turn, score in zip(series.get("turns", []), series.get("scores", [])):
            by_turn.setdefault(int(turn), []).append(float(score))
    turns = sorted(by_turn)
    return {
        "turns": turns,
        "mean": [sum(by_turn[turn]) / len(by_turn[turn]) for turn in turns],
        "se": [
            math.sqrt(
                sum((value - (sum(by_turn[turn]) / len(by_turn[turn]))) ** 2 for value in by_turn[turn])
                / len(by_turn[turn])
            )
            / math.sqrt(len(by_turn[turn]))
            for turn in turns
        ],
    }


def _semantic_scores_for_history(
    history: dict[str, Any],
    *,
    model_name: str,
    device: str | None,
) -> dict[str, Any]:
    analyzer = SemanticSimilarityAnalyzer(
        history,
        method=SEMANTIC_SIMILARITY_METHOD_COSINE,
        model_name=model_name,
        device=device,
    )
    return {
        "self_consistency": analyzer.compute_self_consistency_scores(),
        "cross_agent_public_alignment": analyzer.compute_cross_agent_alignment_scores(
            PUBLIC_NARRATIVE_FIELD,
            PUBLIC_NARRATIVE_FIELD,
        ),
        "cross_agent_private_alignment": analyzer.compute_cross_agent_alignment_scores(
            PRIVATE_NARRATIVE_FIELD,
            PRIVATE_NARRATIVE_FIELD,
        ),
    }


def _serialize_semantic_no_stance(
    group_result: Any,
    decision_labels: list[str],
    *,
    model_name: str | None,
    device: str | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    analyzed_cases = list(getattr(group_result, "analyzed_cases", []) or [])
    slot_lookup = _agent_slot_lookup(group_result)
    sc_by_slot: dict[str, list[dict[str, Any]]] = {}
    public_series: list[dict[str, Any]] = []
    private_series: list[dict[str, Any]] = []
    self_repeats = {"repeats": []}
    cross_repeats = {"repeats": []}

    for repeat_index, res in enumerate(getattr(group_result, "results", []), start=1):
        history = _strip_stance_from_structured_history(
            res.agora.structured_history(),
            decision_labels,
        )
        semantic = _semantic_scores_for_history(history, model_name=model_name, device=device)
        repeat_meta = {"repeat_number": repeat_index}
        if repeat_index - 1 < len(analyzed_cases):
            repeat_meta["case_id"] = analyzed_cases[repeat_index - 1].case_id

        repeat_self = {"alpha": {}, "beta": {}, **repeat_meta}
        for agent_key, payload in (semantic.get("self_consistency") or {}).items():
            slot = slot_lookup.get(agent_key)
            if slot is None:
                continue
            sc_by_slot.setdefault(slot, []).append(payload)
            repeat_self[slot] = _line_payload(
                payload.get("turns", []),
                payload.get("scores", []),
                None,
                x_key="debate_turn",
                y_key="cosine_similarity",
                error_key="standard_error",
            )
        self_repeats["repeats"].append(repeat_self)

        public_payload = semantic.get("cross_agent_public_alignment") or {}
        private_payload = semantic.get("cross_agent_private_alignment") or {}
        if public_payload:
            public_series.append(public_payload)
        if private_payload:
            private_series.append(private_payload)
        cross_repeats["repeats"].append(
            {
                "public alignment": _line_payload(
                    public_payload.get("turns", []),
                    public_payload.get("scores", []),
                    None,
                    x_key="debate_turn",
                    y_key="cosine_similarity",
                    error_key="standard_error",
                ),
                "private alignment": _line_payload(
                    private_payload.get("turns", []),
                    private_payload.get("scores", []),
                    None,
                    x_key="debate_turn",
                    y_key="cosine_similarity",
                    error_key="standard_error",
                ),
                **repeat_meta,
            }
        )

    self_aggregate = {"alpha": {}, "beta": {}}
    for slot, series_list in sc_by_slot.items():
        payload = _aggregate_turn_scores(series_list)
        self_aggregate[slot] = _line_payload(
            payload.get("turns", []),
            payload.get("mean", []),
            payload.get("se", []),
            x_key="debate_turn",
            y_key="cosine_similarity",
            error_key="standard_error",
        )
    public_aggregate = _aggregate_turn_scores(public_series)
    private_aggregate = _aggregate_turn_scores(private_series)
    cross_aggregate = {
        "public alignment": _line_payload(
            public_aggregate.get("turns", []),
            public_aggregate.get("mean", []),
            public_aggregate.get("se", []),
            x_key="debate_turn",
            y_key="cosine_similarity",
            error_key="standard_error",
        ),
        "private alignment": _line_payload(
            private_aggregate.get("turns", []),
            private_aggregate.get("mean", []),
            private_aggregate.get("se", []),
            x_key="debate_turn",
            y_key="cosine_similarity",
            error_key="standard_error",
        ),
    }
    return self_aggregate, cross_aggregate, self_repeats, cross_repeats


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


def _serialize_nli_payload(nli: dict[str, Any], slot_lookup: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
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


def _serialize_nli(group_result: Any, *, model_name: str | None, device: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
    nli = group_result.run_nli_analysis(model_name=model_name, device=device)
    return _serialize_nli_payload(nli, _agent_slot_lookup(group_result))


def _serialize_nli_all_repeats(
    group_result: Any,
    *,
    model_name: str | None,
    device: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    analyzed_cases = list(getattr(group_result, "analyzed_cases", []) or [])
    self_consistency = {"repeats": []}
    cross_agent = {"repeats": []}
    slot_lookup = _agent_slot_lookup(group_result)
    repeat_payloads = group_result.run_nli_analysis_all_repeats(
        model_name=model_name,
        device=device,
    )

    for repeat_index, nli in enumerate(repeat_payloads, start=1):
        repeat_meta = {"repeat_number": repeat_index}
        if repeat_index - 1 < len(analyzed_cases):
            repeat_meta["case_id"] = analyzed_cases[repeat_index - 1].case_id
        repeat_self, repeat_cross = _serialize_nli_payload(nli, slot_lookup)
        self_consistency["repeats"].append({**repeat_self, **repeat_meta})
        cross_agent["repeats"].append({**repeat_cross, **repeat_meta})
    return self_consistency, cross_agent


def _serialize_emotion_payload(payload: dict[str, Any], slot_lookup: dict[str, str]) -> dict[str, Any]:
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


def _serialize_emotions(group_result: Any, *, model_name: str | None, device: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
    slot_lookup = _agent_slot_lookup(group_result)
    public = group_result.run_emotion_analysis(
        field=PUBLIC_NARRATIVE_FIELD,
        model_name=model_name,
        device=device,
    )
    private = group_result.run_emotion_analysis(
        field=PRIVATE_NARRATIVE_FIELD,
        model_name=model_name,
        device=device,
    )
    return (
        _serialize_emotion_payload(public, slot_lookup),
        _serialize_emotion_payload(private, slot_lookup),
    )


def _serialize_emotions_all_repeats(
    group_result: Any,
    *,
    model_name: str | None,
    device: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    analyzed_cases = list(getattr(group_result, "analyzed_cases", []) or [])
    public = {"repeats": []}
    private = {"repeats": []}
    slot_lookup = _agent_slot_lookup(group_result)
    public_payloads = group_result.run_emotion_analysis_all_repeats(
        field=PUBLIC_NARRATIVE_FIELD,
        model_name=model_name,
        device=device,
    )
    private_payloads = group_result.run_emotion_analysis_all_repeats(
        field=PRIVATE_NARRATIVE_FIELD,
        model_name=model_name,
        device=device,
    )

    for repeat_index, repeat_payload in enumerate(public_payloads, start=1):
        repeat_meta = {"repeat_number": repeat_index}
        if repeat_index - 1 < len(analyzed_cases):
            repeat_meta["case_id"] = analyzed_cases[repeat_index - 1].case_id
        repeat_public = _serialize_emotion_payload(repeat_payload, slot_lookup)
        public["repeats"].append({**repeat_public, **repeat_meta})

    for repeat_index, repeat_payload in enumerate(private_payloads, start=1):
        repeat_meta = {"repeat_number": repeat_index}
        if repeat_index - 1 < len(analyzed_cases):
            repeat_meta["case_id"] = analyzed_cases[repeat_index - 1].case_id
        repeat_private = _serialize_emotion_payload(repeat_payload, slot_lookup)
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


def _serialize_decisions(
    group_result: Any,
    *,
    scenario_id: str | None,
    catalog_path: Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    decisions = group_result.aggregate_response_decisions(
        scenario_id=scenario_id,
        catalog_path=catalog_path,
    )
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


def _serialize_decisions_all_repeats(
    group_result: Any,
    *,
    scenario_id: str | None,
    catalog_path: Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    decisions = group_result.aggregate_response_decisions_all_repeats(
        scenario_id=scenario_id,
        catalog_path=catalog_path,
    )
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
    include_no_stance: bool = True,
    no_stance_only: bool = False,
) -> dict[str, Any]:
    """Build a single dataframe-ready row for one experiment group."""
    sweep_root_path = Path(sweep_root)
    effective_analysis_kwargs = analysis_kwargs or {}
    decision_scenario_id, decision_catalog_path = _group_decision_context(
        group,
        sweep_root_path,
        effective_analysis_kwargs,
    )
    group_result = group.run_analysis(sweep_root_path, **effective_analysis_kwargs)
    analyzed_cases = list(getattr(group_result, "analyzed_cases", []) or group.cases)
    if not group_result.results:
        context = ""
        if getattr(group, "cases", None):
            context = format_case_warning_context(group.cases[0], sweep_root_path)
        print(
            "Warning: skipping experiment group "
            f"{group.config_fingerprint}{context} because no analyzable cases remained."
        )
        return {}

    row = {
        "experiment_index": int(experiment_index),
        "config_fingerprint": group.config_fingerprint,
        "repeat_count": int(len(analyzed_cases)),
        "case_ids": [case.case_id for case in analyzed_cases],
        **dict(group.sweep_values or {}),
    }
    if not no_stance_only:
        cosine_self, cosine_cross = _serialize_semantic(group_result)
        cosine_self_repeats, cosine_cross_repeats = _serialize_semantic_all_repeats(group_result)
        persona_individual, persona_cumulative, persona_full = _serialize_persona(group_result)
        persona_individual_repeats, persona_cumulative_repeats, persona_full_repeats = _serialize_persona_all_repeats(group_result)
        survey_public, survey_private, survey_diff = _serialize_survey(group_result)
        survey_public_repeats, survey_private_repeats, survey_diff_repeats = _serialize_survey_all_repeats(group_result)
        decision_self, decision_cross = _serialize_decisions(
            group_result,
            scenario_id=decision_scenario_id,
            catalog_path=decision_catalog_path,
        )
        decision_self_repeats, decision_cross_repeats = _serialize_decisions_all_repeats(
            group_result,
            scenario_id=decision_scenario_id,
            catalog_path=decision_catalog_path,
        )
        row.update(
            {
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
        )

    needs_no_stance_labels = include_no_stance or no_stance_only or include_nli
    no_stance_labels = (
        _decision_labels_for_group(
            group,
            sweep_root_path,
            scenario_id=decision_scenario_id,
            catalog_path=decision_catalog_path,
        )
        if needs_no_stance_labels
        else []
    )
    semantic_kwargs = effective_analysis_kwargs
    semantic_model_name = semantic_kwargs.get("semantic_similarity_model")
    semantic_method = semantic_kwargs.get("semantic_similarity_method")
    if include_no_stance or no_stance_only:
        if semantic_method in {None, SEMANTIC_SIMILARITY_METHOD_COSINE}:
            (
                cosine_self_no_stance,
                cosine_cross_no_stance,
                cosine_self_repeats_no_stance,
                cosine_cross_repeats_no_stance,
            ) = _serialize_semantic_no_stance(
                group_result,
                no_stance_labels,
                model_name=semantic_model_name,
                device=device or semantic_kwargs.get("semantic_similarity_device"),
            )
            row["cosine-similarity-self-consistency-no_stance"] = cosine_self_no_stance
            row["cosine-similarity-cross-agent-alignment-no_stance"] = cosine_cross_no_stance
            row["cosine-similarity-self-consistency-all-repeats-no_stance"] = cosine_self_repeats_no_stance
            row["cosine-similarity-cross-agent-alignment-all-repeats-no_stance"] = cosine_cross_repeats_no_stance

    if include_nli:
        if not no_stance_only:
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
        no_stance_result = _no_stance_group_result(group_result, no_stance_labels)
        nli_self_no_stance, nli_cross_no_stance = _serialize_nli(
            no_stance_result,
            model_name=nli_model_name,
            device=device,
        )
        nli_self_repeats_no_stance, nli_cross_repeats_no_stance = _serialize_nli_all_repeats(
            no_stance_result,
            model_name=nli_model_name,
            device=device,
        )
        row["nli-self-consistency-no_stance"] = nli_self_no_stance
        row["nli-cross-agent-alignment-no_stance"] = nli_cross_no_stance
        row["nli-self-consistency-all-repeats-no_stance"] = nli_self_repeats_no_stance
        row["nli-cross-agent-alignment-all-repeats-no_stance"] = nli_cross_repeats_no_stance
    if include_emotions:
        if not no_stance_only:
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
    include_no_stance: bool = True,
    no_stance_only: bool = False,
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
            include_no_stance=include_no_stance,
            no_stance_only=no_stance_only,
        )
        for index, group in enumerate(manifest)
    ]
    return [record for record in records if record]


def _aggregate_json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")  # pragma: no cover


def write_experiment_analysis_records(
    records: list[dict[str, Any]],
    output_path: Path | str,
) -> Path:
    """Write aggregate records as portable JSON and return the output path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(records, indent=2, default=_aggregate_json_default) + "\n",
        encoding="utf-8",
    )
    return path


def aggregate_sweep_analysis(
    manifest_path: Path | str,
    *,
    output_path: Path | str,
    analysis_kwargs: dict[str, Any] | None = None,
    include_nli: bool = False,
    nli_model_name: str | None = None,
    include_emotions: bool = False,
    emotion_model_name: str | None = None,
    device: str | None = None,
    include_no_stance: bool = False,
    no_stance_only: bool = False,
) -> dict[str, Any]:
    """Build and persist one aggregate JSON record per experiment group."""
    manifest = SweepManifest.from_path(manifest_path)
    records = build_experiment_analysis_records(
        manifest,
        analysis_kwargs=analysis_kwargs,
        include_nli=include_nli,
        nli_model_name=nli_model_name,
        include_emotions=include_emotions,
        emotion_model_name=emotion_model_name,
        device=device,
        include_no_stance=include_no_stance,
        no_stance_only=no_stance_only,
    )
    written_path = write_experiment_analysis_records(records, output_path)
    return {"output_path": written_path, "row_count": len(records)}


def build_experiment_analysis_dataframe(
    manifest_path: Path | str,
    *,
    analysis_kwargs: dict[str, Any] | None = None,
    include_nli: bool = True,
    nli_model_name: str | None = DEFAULT_NLI_MODEL_NAME,
    include_emotions: bool = True,
    emotion_model_name: str | None = None,
    device: str | None = None,
    include_no_stance: bool = True,
    no_stance_only: bool = False,
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
        include_no_stance=include_no_stance,
        no_stance_only=no_stance_only,
    )
    return pd.DataFrame(records)
