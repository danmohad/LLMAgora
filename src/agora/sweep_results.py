"""Read-only view over a completed sweep, grouped by config fingerprint."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields as dc_fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np

from .emotion_analyzer import DEFAULT_EMOTION_MODEL
from .semantic_similarity_analyzer import DEFAULT_NLI_MODEL_NAME

if TYPE_CHECKING:
    from .experiment import ExperimentResult


_WARNING_CONFIG_FIELDS = (
    "model",
    "scenario_id",
    "incentive_direction",
    "incentive_type",
)


def format_case_warning_context(case: SweepCase, sweep_root: Path) -> str:
    """Return a compact warning suffix with key config identifiers when available."""
    abs_config_path = sweep_root / case.config_path
    if not abs_config_path.exists():
        return ""

    try:
        payload = json.loads(abs_config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""

    parts = [
        f"{field}={payload[field]!r}"
        for field in _WARNING_CONFIG_FIELDS
        if payload.get(field) is not None
    ]
    return f" ({', '.join(parts)})" if parts else ""


@dataclass(slots=True)
class SweepCase:
    """One executed case (a single repeat of an experiment)."""

    case_id: str
    case_dir: Path
    config_path: Path
    label: str
    repeat_number: int
    repeat_count: int
    sweep_values: dict[str, Any]

    def run_analysis(self, sweep_root: Path, **postpro: Any) -> ExperimentResult:
        """Run offline post-processing on this case's debate snapshot.

        Loads ``config.json`` from ``case_dir``, sets the experiment to
        offline/replay mode (``num_turns=0``, ``load_snapshot=True``,
        ``reuse_load_dir_for_outputs=True``), merges any *postpro* keyword
        overrides (e.g. ``semantic_analysis_metrics``, ``persona_analysis_metrics``),
        and delegates to :func:`~agora.experiment.run_persona_experiment`.

        Parameters
        ----------
        sweep_root:
            Absolute root of the sweep directory (i.e. ``SweepManifest.sweep_root``).
        **postpro:
            Any :class:`~agora.experiment.ExperimentConfig` fields you want to
            override for the analysis pass (analysis metrics, scoring model, …).
        """
        from .experiment import ExperimentConfig, build_experiment_config, run_persona_experiment

        abs_case_dir = sweep_root / self.case_dir
        abs_config_path = sweep_root / self.config_path

        base_raw = json.loads(abs_config_path.read_text(encoding="utf-8"))
        allowed = {f.name for f in dc_fields(ExperimentConfig)}
        # Strip fields we override explicitly, plus any machine-specific paths from the
        # original run (output_dir, outputs_root, index_csv, run_name).
        _OFFLINE_OVERRIDE_FIELDS = {
            "num_turns", "load_snapshot", "load_dir", "save_snapshot",
            "reuse_load_dir_for_outputs", "indexed_output",
            "output_dir", "outputs_root", "index_csv", "run_name",
            "catalog_path", "prompts_path",
        }
        base_config = {
            k: v for k, v in base_raw.items()
            if k in allowed and k not in _OFFLINE_OVERRIDE_FIELDS
        }

        offline_payload: dict[str, Any] = {
            **base_config,
            "num_turns": 0,
            "load_snapshot": True,
            "load_dir": abs_case_dir,
            "save_snapshot": False,
            "reuse_load_dir_for_outputs": True,
            "indexed_output": False,
            **postpro,
        }

        return run_persona_experiment(build_experiment_config(offline_payload))


# ---------------------------------------------------------------------------
# Aggregation helpers (module-level so tests can import them directly)
# ---------------------------------------------------------------------------

def _agg_turn_scores(series_list: list[dict]) -> dict:
    """Aggregate per-turn score series across repeats.

    Input:  list of ``{"turns": [...], "scores": [...]}``
    Output: ``{"turns": [...], "mean": [...], "se": [...]}``
    """
    by_turn: dict[int, list[float]] = {}
    for series in series_list:
        for t, s in zip(series.get("turns", []), series.get("scores", [])):
            by_turn.setdefault(int(t), []).append(float(s))
    turns = sorted(by_turn)
    return {
        "turns": turns,
        "mean": [float(np.mean(by_turn[t])) for t in turns],
        "se": [float(np.std(by_turn[t]) / np.sqrt(max(len(by_turn[t]), 1))) for t in turns],
    }


def _agg_persona_per_turn(series_list: list[dict]) -> dict:
    """Aggregate persona per-turn score dicts across repeats.

    Input:  list of ``{"turns": [...], "scores": {"mean": [...]}}``
    Output: ``{"turns": [...], "scores": {"mean": [...], "se": [...]}}``
    """
    by_turn: dict[int, list[float]] = {}
    for pt in series_list:
        for t, m in zip(pt.get("turns", []), pt.get("scores", {}).get("mean", [])):
            by_turn.setdefault(int(t), []).append(float(m))
    turns = sorted(by_turn)
    return {
        "turns": turns,
        "scores": {
            "mean": [float(np.mean(by_turn[t])) for t in turns],
            "se": [float(np.std(by_turn[t]) / np.sqrt(max(len(by_turn[t]), 1))) for t in turns],
        },
    }


def _agg_persona_role(role_list: list[dict]) -> dict:
    """Aggregate one role's persona adherence data across repeats."""
    result: dict[str, Any] = {}
    for key in (
        "public_per_turn_scores",
        "private_per_turn_scores",
        "public_cumulative_scores",
        "private_cumulative_scores",
    ):
        series = [d[key] for d in role_list if d.get(key)]
        result[key] = _agg_persona_per_turn(series) if series else {}
    for key in ("full_debate_public_score", "full_debate_private_score"):
        scalars = [
            d[key]["mean"]
            for d in role_list
            if d.get(key) and d[key].get("mean") is not None
        ]
        if scalars:
            result[key] = {
                "mean": float(np.mean(scalars)),
                "se": float(np.std(scalars) / np.sqrt(max(len(scalars), 1))),
            }
    result["computed_metrics"] = sorted(
        {m for d in role_list for m in d.get("computed_metrics", [])}
    )
    return result


def _nli_bidirectional(analyzer: Any, text_a: str, text_b: str) -> list[float]:
    """Average forward + backward NLI class distributions."""

    def _to_list(p: Any) -> list[float]:
        if hasattr(p, "tolist"):
            p = p.tolist()
        return p[0] if (isinstance(p, list) and isinstance(p[0], (list, tuple))) else p

    ab = _to_list(analyzer.model.predict([(text_a, text_b)], apply_softmax=True))
    ba = _to_list(analyzer.model.predict([(text_b, text_a)], apply_softmax=True))
    return [(ab[i] + ba[i]) / 2.0 for i in range(len(ab))]


def _agg_nli_by_turn(
    turn_dict: dict[int, list[list[float]]],
    id2label: dict,
) -> dict:
    """Aggregate NLI 3-class distributions per turn across repeats.

    Output: ``{"turns": [...], "label_names": [...], "distributions":
    {name: {"mean": [...], "se": [...]}}}``
    """
    n_classes = len(id2label)
    label_names = [
        str(id2label.get(i, id2label.get(str(i), f"class{i}")))
        for i in range(n_classes)
    ]
    turns = sorted(turn_dict)
    distributions = {
        name: {
            "mean": [float(np.mean([d[i] for d in turn_dict[t]])) for t in turns],
            "se": [
                float(np.std([d[i] for d in turn_dict[t]]) / np.sqrt(max(len(turn_dict[t]), 1)))
                for t in turns
            ],
        }
        for i, name in enumerate(label_names)
    }
    return {"turns": turns, "label_names": label_names, "distributions": distributions}


def _nli_result_from_buckets(
    sc_by_agent: dict[str, dict[int, list[list[float]]]],
    cpa_by_turn: dict[int, list[list[float]]],
    cpriva_by_turn: dict[int, list[list[float]]],
    id2label: dict,
) -> dict[str, Any]:
    result: dict[str, Any] = {"id2label": id2label}
    if sc_by_agent:
        result["self_consistency"] = {
            aid: _agg_nli_by_turn(td, id2label) for aid, td in sc_by_agent.items()
        }
    if cpa_by_turn:
        result["cross_agent_public"] = _agg_nli_by_turn(cpa_by_turn, id2label)
    if cpriva_by_turn:
        result["cross_agent_private"] = _agg_nli_by_turn(cpriva_by_turn, id2label)
    return result


def _classify_decision(text: str, decision_labels: list[str]) -> int | None:
    """Return index of the matching decision label at the start of *text*, or *None*.

    Labels are checked longest-first so that a label like "DO NOT ENDORSE" is
    not overshadowed by a shorter label "ENDORSE".
    """
    text_upper = text.strip().upper()
    for idx, label in sorted(enumerate(decision_labels), key=lambda x: -len(x[1])):
        if text_upper.startswith(label.upper()):
            return idx
    return None


# ---------------------------------------------------------------------------
# GroupAnalysisResult
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class GroupAnalysisResult:
    """Aggregated analysis across all repeats of one experiment configuration.

    Returned by :meth:`ExperimentGroup.run_analysis`.  Provides methods to
    aggregate metrics across repeats (mean ± std) and produce group-level
    plots with error bars / shaded bands.
    """

    group: ExperimentGroup
    results: list  # list[ExperimentResult]
    analyzed_cases: list[SweepCase] = field(default_factory=list)
    _semantic_cache: dict | None = field(default=None, repr=False, compare=False, init=False)
    _persona_cache: dict | None = field(default=None, repr=False, compare=False, init=False)
    _nli_cache: dict | None = field(default=None, repr=False, compare=False, init=False)
    _nli_repeat_cache: dict | None = field(default=None, repr=False, compare=False, init=False)
    _emotion_caches: dict = field(default_factory=dict, repr=False, compare=False, init=False)
    _survey_cache: dict | None = field(default=None, repr=False, compare=False, init=False)
    _decision_cache: dict | None = field(default=None, repr=False, compare=False, init=False)
    _decision_per_repeat_cache: dict | None = field(default=None, repr=False, compare=False, init=False)

    # ------------------------------------------------------------------ metadata

    @property
    def n_repeats(self) -> int:
        """Number of experimental repeats."""
        return len(self.results)

    @property
    def agent_names(self) -> tuple[str, str]:
        """``(alpha_name, beta_name)`` taken from the first repeat's agents."""
        if not self.results:
            return ("alpha", "beta")
        agents = self.results[0].agents
        return (
            (agents[0].name, agents[1].name) if len(agents) >= 2 else (agents[0].name, "")
        )

    @property
    def survey_question_specs(self) -> list:
        """Merged survey question specs from the first repeat that has them.

        Each spec is a ``{"text": ..., "group": ...}`` dict as produced by
        :func:`~agora.survey.merge_survey_question_configs`.  Returns an empty
        list when no repeat stored question specs (e.g. survey was disabled).
        """
        for res in self.results:
            specs = getattr(res, "survey_question_specs", [])
            if specs:
                return specs
        return []

    # ------------------------------------------------------------------ aggregation

    def aggregate_semantic(self) -> dict:
        """Aggregate semantic similarity metrics across repeats (mean ± std per turn).

        Returns a dict with any of the following keys (only those computed):

        - ``"self_consistency"``: ``{agent_id: {"turns": [...], "mean": [...], "std": [...]}}``
        - ``"cross_agent_public_alignment"``: ``{"turns": [...], "mean": [...], "std": [...]}``
        - ``"cross_agent_private_alignment"``: ``{"turns": [...], "mean": [...], "std": [...]}``
        """
        if self._semantic_cache is not None:
            return self._semantic_cache
        sc_by_agent: dict[str, list[dict]] = {}
        cpa_series: list[dict] = []
        cpriva_series: list[dict] = []
        for res in self.results:
            sem = res.eval_data.get("semantic_similarity") or {}
            if not isinstance(sem, dict):
                continue
            sc_payload = sem.get("self_consistency") or {}
            if isinstance(sc_payload, dict):
                for agent_id, data in sc_payload.items():
                    if isinstance(data, dict):
                        sc_by_agent.setdefault(agent_id, []).append(data)
            cpa = sem.get("cross_agent_public_alignment")
            if isinstance(cpa, dict):
                cpa_series.append(cpa)
            cpriva = sem.get("cross_agent_private_alignment")
            if isinstance(cpriva, dict):
                cpriva_series.append(cpriva)
        result: dict[str, Any] = {}
        if sc_by_agent:
            result["self_consistency"] = {
                aid: _agg_turn_scores(sl) for aid, sl in sc_by_agent.items()
            }
        if cpa_series:
            result["cross_agent_public_alignment"] = _agg_turn_scores(cpa_series)
        if cpriva_series:
            result["cross_agent_private_alignment"] = _agg_turn_scores(cpriva_series)
        self._semantic_cache = result
        return result

    def aggregate_persona(self) -> dict:
        """Aggregate persona adherence metrics across repeats.

        Returns the same dict structure expected by
        :func:`~agora.plotting.plot_persona_adherence`, with scores aggregated
        (mean ± std) across repeats.  Returns an empty dict if no persona
        adherence data was computed.
        """
        if self._persona_cache is not None:
            return self._persona_cache
        per_role: dict[str, list[dict]] = {}
        for res in self.results:
            pers = res.eval_data.get("persona_adherence")
            if not pers:
                continue
            for role in ("alpha", "beta"):
                if role in pers:
                    per_role.setdefault(role, []).append(pers[role])
        result = {role: _agg_persona_role(rl) for role, rl in per_role.items()}
        self._persona_cache = result
        return result

    def aggregate_survey(self, survey_questions=None) -> dict:
        """Aggregate survey responses across repeats (mean ± SE per turn per question).

        Returns a dict with keys ``"public"``, ``"private"``, and ``"diff"``
        (only those for which data exists).  Each maps **slot names** (``"Alpha"`` /
        ``"Beta"``) to per-question aggregated series::

            {
                "public":  {"Alpha": {q_key: {"turns": [...], "mean": [...], "se": [...]}}, ...},
                "private": {"Alpha": {q_key: ...}, ...},
                "diff":    {"Alpha": {q_key: ...}, ...},
            }

        For questions in the ``"incentive"`` group, the diff is computed as
        ``abs(public − private)`` so that the aggregated mean reflects the
        average *magnitude* of the gap (matching :func:`~agora.plotting.plot_survey_distance`).
        Deliberative/evaluative questions keep the signed difference.

        Parameters
        ----------
        survey_questions:
            Optional question specs (same format accepted by
            :func:`~agora.plotting.plot_survey_responses`).  When supplied,
            incentive questions are identified and their diffs are
            absolute-valued before aggregation.  When *None* every diff is
            signed and results are cached on the instance.

        Returns an empty dict if no survey responses are found.
        """
        if survey_questions is None and self._survey_cache is not None:
            return self._survey_cache

        from .survey import (
            SURVEY_GROUP_DELIBERATIVE,
            SURVEY_GROUP_INCENTIVE,
            normalize_survey_questions,
        )

        incentive_q_keys: set[str] = set()
        if survey_questions is not None:
            specs = normalize_survey_questions(
                survey_questions, default_group=SURVEY_GROUP_DELIBERATIVE
            )
            for i, spec in enumerate(specs, start=1):
                if spec["group"] == SURVEY_GROUP_INCENTIVE:
                    incentive_q_keys.add(f"Q{i}")

        def _parse_slot(turns: list, event_key: str) -> dict:
            """Returns {slot: {turn_num: {q_key: score}}}, keyed by slot name."""
            result: dict[str, dict[int, dict[str, float]]] = {}
            for turn in turns:
                turn_num = int(turn.get("turn_num", 0))
                for slot in ("Alpha", "Beta"):
                    subturn = turn.get(slot, {})
                    scores = subturn.get(event_key)
                    if scores is None:
                        continue
                    result.setdefault(slot, {})[turn_num] = scores
            return result

        def _accumulate(by_slot: dict, slot_data: dict) -> None:
            for slot, slot_turns in slot_data.items():
                for turn_num, questions in slot_turns.items():
                    for q_key, score in questions.items():
                        (
                            by_slot
                            .setdefault(slot, {})
                            .setdefault(int(turn_num), {})
                            .setdefault(q_key, [])
                            .append(float(score))
                        )

        def _agg_by_question(by_slot: dict) -> dict:
            out: dict[str, dict] = {}
            for slot, turn_map in by_slot.items():
                all_q_keys = sorted({qk for qd in turn_map.values() for qk in qd})
                out[slot] = {}
                for q_key in all_q_keys:
                    turns = sorted(t for t in turn_map if q_key in turn_map[t])
                    out[slot][q_key] = {
                        "turns": turns,
                        "mean": [float(np.mean(turn_map[t][q_key])) for t in turns],
                        "se": [
                            float(
                                np.std(turn_map[t][q_key])
                                / np.sqrt(max(len(turn_map[t][q_key]), 1))
                            )
                            for t in turns
                        ],
                    }
            return out

        pub_by_slot: dict[str, dict[int, dict[str, list[float]]]] = {}
        priv_by_slot: dict[str, dict[int, dict[str, list[float]]]] = {}
        diff_by_slot: dict[str, dict[int, dict[str, list[float]]]] = {}

        for res in self.results:
            history = res.agora.structured_history()
            turns = history.get("turns", [])
            pub = _parse_slot(turns, "public_survey")
            priv = _parse_slot(turns, "private_survey")

            _accumulate(pub_by_slot, pub)
            _accumulate(priv_by_slot, priv)

            # diff: (public − private) per slot/turn/question where both present.
            # For incentive questions, abs() is applied before aggregation.
            for slot in set(pub.keys()) | set(priv.keys()):
                pub_slot = pub.get(slot, {})
                priv_slot = priv.get(slot, {})
                for turn_num in set(pub_slot.keys()) & set(priv_slot.keys()):
                    pub_q = pub_slot[turn_num]
                    priv_q = priv_slot[turn_num]
                    for q_key in set(pub_q.keys()) & set(priv_q.keys()):
                        raw_diff = float(pub_q[q_key]) - float(priv_q[q_key])
                        score = abs(raw_diff) if q_key in incentive_q_keys else raw_diff
                        (
                            diff_by_slot
                            .setdefault(slot, {})
                            .setdefault(int(turn_num), {})
                            .setdefault(q_key, [])
                            .append(score)
                        )

        result: dict[str, Any] = {}
        pub_agg = _agg_by_question(pub_by_slot)
        if pub_agg:
            result["public"] = pub_agg
        priv_agg = _agg_by_question(priv_by_slot)
        if priv_agg:
            result["private"] = priv_agg
        diff_agg = _agg_by_question(diff_by_slot)
        if diff_agg:
            result["diff"] = diff_agg
        if survey_questions is None:
            self._survey_cache = result
        return result

    def aggregate_response_decisions(
        self,
        scenario_id: str | None = None,
        catalog_path: "Path | str | None" = None,
    ) -> dict:
        """Aggregate binary response decisions across repeats per turn.

        Reads the ``decision_labels`` for the scenario from the catalog, then
        for each repeat classifies each agent's public and private utterance at
        every turn as one of the two labels.  Returns the fraction of repeats
        in which the agent chose ``decision_labels[0]`` (the "primary" label),
        with ±SE, for each (agent slot, channel, turn) combination.

        Parameters
        ----------
        scenario_id:
            The scenario identifier used to look up ``decision_labels`` in the
            catalog.  When *None*, the value is taken from
            ``self.group.sweep_values["scenario_id"]``.
        catalog_path:
            Path to the scenarios catalog JSON file.  Defaults to
            :data:`~agora.experiment.DEFAULT_CATALOG_PATH`.

        Returns
        -------
        dict with keys:

        - ``"decision_label"``: the primary label being tracked (``decision_labels[0]``).
        - ``"by_slot"``: ``{slot: {"public": series, "private": series}}`` where
          each *series* is ``{"turns": [...], "mean": [...], "se": [...]}``.
        """
        if self._decision_cache is not None and scenario_id is None and catalog_path is None:
            return self._decision_cache

        import json as _json

        from .experiment import DEFAULT_CATALOG_PATH

        effective_scenario_id = scenario_id or self.group.sweep_values.get("scenario_id")
        if not effective_scenario_id:
            raise ValueError(
                "Could not auto-detect scenario_id from sweep_values; pass scenario_id= explicitly."
            )

        effective_catalog = Path(catalog_path) if catalog_path is not None else DEFAULT_CATALOG_PATH
        catalog = _json.loads(effective_catalog.read_text(encoding="utf-8"))
        scenario = next(
            (s for s in catalog.get("scenarios", []) if s.get("scenario_id") == effective_scenario_id),
            None,
        )
        if scenario is None:
            raise KeyError(f"Scenario '{effective_scenario_id}' not found in catalog")
        decision_labels: list[str] = scenario.get("decision_labels", [])
        if len(decision_labels) < 2:
            raise ValueError(
                f"Expected 2 decision_labels for '{effective_scenario_id}', got {decision_labels}"
            )

        primary_label = decision_labels[0]

        # {slot: {channel: {turn_num: [0.0 or 1.0, ...]}}}
        by_slot: dict[str, dict[str, dict[int, list[float]]]] = {}

        for res in self.results:
            history = res.agora.structured_history()
            for turn in history.get("turns", []):
                turn_num = int(turn.get("turn_num", 0))
                for slot in ("Alpha", "Beta"):
                    subturn = turn.get(slot, {})
                    pub = subturn.get("public_utterance") or ""
                    priv = subturn.get("private_utterance") or ""
                    for channel, text in (("public", pub), ("private", priv)):
                        if not text:
                            continue
                        idx = _classify_decision(text, decision_labels)
                        if idx is None:
                            continue
                        (
                            by_slot
                            .setdefault(slot, {})
                            .setdefault(channel, {})
                            .setdefault(turn_num, [])
                            .append(1.0 if idx == 0 else 0.0)
                        )

        def _agg_series(turn_dict: dict[int, list[float]]) -> dict:
            turns = sorted(turn_dict)
            agg: dict[str, Any] = {"turns": turns, "mean": [], "se": []}
            for t in turns:
                vals = turn_dict[t]
                n = len(vals)
                p = float(np.mean(vals))
                agg["mean"].append(p)
                agg["se"].append(float(np.sqrt(p * (1.0 - p) / max(n, 1))))
            return agg

        result: dict[str, Any] = {
            "decision_label": primary_label,
            "by_slot": {
                slot: {channel: _agg_series(td) for channel, td in channels.items()}
                for slot, channels in by_slot.items()
            },
        }

        if scenario_id is None and catalog_path is None:
            self._decision_cache = result
        return result

    def aggregate_response_decisions_all_repeats(
        self,
        scenario_id: str | None = None,
        catalog_path: "Path | str | None" = None,
    ) -> dict:
        """Per-repeat, per-turn binary decision values for each agent and channel.

        Unlike :meth:`aggregate_response_decisions` which averages across repeats,
        this preserves each repeat as an individual entry so that every repeat
        can be plotted as a separate line.

        Parameters
        ----------
        scenario_id, catalog_path:
            Same semantics as :meth:`aggregate_response_decisions`.

        Returns
        -------
        dict with keys:

        - ``"decision_label"``: the primary label treated as 1.
        - ``"repeats"``: list of per-repeat dicts, each
          ``{slot: {channel: {"turns": [...], "decisions": [0|1, ...]}}}``,
          one entry per repeat in the same order as ``self.results``.
        """
        if self._decision_per_repeat_cache is not None and scenario_id is None and catalog_path is None:
            return self._decision_per_repeat_cache

        import json as _json

        from .experiment import DEFAULT_CATALOG_PATH

        effective_scenario_id = scenario_id or self.group.sweep_values.get("scenario_id")
        if not effective_scenario_id:
            raise ValueError(
                "Could not auto-detect scenario_id from sweep_values; pass scenario_id= explicitly."
            )

        effective_catalog = Path(catalog_path) if catalog_path is not None else DEFAULT_CATALOG_PATH
        catalog = _json.loads(effective_catalog.read_text(encoding="utf-8"))
        scenario = next(
            (s for s in catalog.get("scenarios", []) if s.get("scenario_id") == effective_scenario_id),
            None,
        )
        if scenario is None:
            raise KeyError(f"Scenario '{effective_scenario_id}' not found in catalog")
        decision_labels: list[str] = scenario.get("decision_labels", [])
        if len(decision_labels) < 2:
            raise ValueError(
                f"Expected 2 decision_labels for '{effective_scenario_id}', got {decision_labels}"
            )

        primary_label = decision_labels[0]
        repeats_data: list[dict] = []

        for res in self.results:
            history = res.agora.structured_history()
            repeat_by_slot: dict[str, dict[str, dict[int, int]]] = {}
            for turn in history.get("turns", []):
                turn_num = int(turn.get("turn_num", 0))
                for slot in ("Alpha", "Beta"):
                    subturn = turn.get(slot, {})
                    pub = subturn.get("public_utterance") or ""
                    priv = subturn.get("private_utterance") or ""
                    for channel, text in (("public", pub), ("private", priv)):
                        if not text:
                            continue
                        idx = _classify_decision(text, decision_labels)
                        if idx is None:
                            continue
                        (
                            repeat_by_slot
                            .setdefault(slot, {})
                            .setdefault(channel, {})
                        )[turn_num] = 1 if idx == 0 else 0

            repeat_entry: dict[str, dict] = {}
            for slot, channels in repeat_by_slot.items():
                repeat_entry[slot] = {
                    channel: {
                        "turns": sorted(turn_map),
                        "decisions": [turn_map[t] for t in sorted(turn_map)],
                    }
                    for channel, turn_map in channels.items()
                }
            repeats_data.append(repeat_entry)

        result: dict[str, Any] = {
            "decision_label": primary_label,
            "repeats": repeats_data,
        }
        if scenario_id is None and catalog_path is None:
            self._decision_per_repeat_cache = result
        return result

    # ------------------------------------------------------------------ on-demand analyses

    def run_nli_analysis(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ) -> dict:
        """Run bidirectional NLI analysis across all repeats and return aggregated distributions.

        Computes self-consistency (private→public) and cross-agent public alignment NLI
        for every repeat, then aggregates to mean ± std per turn.

        Structure: ``{metric: {key: {"turns": [...], "label_names": [...],
        "distributions": {name: {"mean": [...], "std": [...]}}}}}``

        Results are cached; repeated calls are free.
        """
        resolved_model_name = model_name or DEFAULT_NLI_MODEL_NAME
        cache_key = (resolved_model_name, device)
        if self._nli_cache is not None and cache_key in self._nli_cache:
            return self._nli_cache[cache_key]

        from .debate_history import get_structured_debate_history
        from .semantic_similarity_analyzer import (
            PRIVATE_NARRATIVE_FIELD,
            PUBLIC_NARRATIVE_FIELD,
            SEMANTIC_SIMILARITY_METHOD_NLI,
            SemanticSimilarityAnalyzer,
        )

        analyzer: Any = None
        id2label: dict | None = None
        sc_by_agent: dict[str, dict[int, list[list[float]]]] = {}
        cpa_by_turn: dict[int, list[list[float]]] = {}
        cpriva_by_turn: dict[int, list[list[float]]] = {}
        repeat_results: list[dict[str, Any]] = []

        for res in self.results:
            structured_history = res.agora.structured_history()
            if analyzer is None:
                kwargs: dict[str, Any] = {
                    "method": SEMANTIC_SIMILARITY_METHOD_NLI,
                    "model_name": resolved_model_name,
                }
                if device is not None:
                    kwargs["device"] = device
                analyzer = SemanticSimilarityAnalyzer(structured_history, **kwargs)
                _ = analyzer.model  # force load
                id2label = analyzer._id2label or {
                    0: "contradiction",
                    1: "neutral",
                    2: "entailment",
                }
            else:
                if isinstance(structured_history, dict) and "turns" in structured_history:
                    analyzer.debate_data = get_structured_debate_history(structured_history)
                else:
                    analyzer.debate_data = structured_history

            debate_data = analyzer.debate_data
            agent_ids = list(debate_data.keys())
            repeat_sc_by_agent: dict[str, dict[int, list[list[float]]]] = {}
            repeat_cpa_by_turn: dict[int, list[list[float]]] = {}
            repeat_cpriva_by_turn: dict[int, list[list[float]]] = {}

            for agent_id in agent_ids:
                for i, turn in enumerate(debate_data[agent_id]["debate_turns"]):
                    turn_num = int(turn.get("turn_num", i + 1))
                    private = turn.get(PRIVATE_NARRATIVE_FIELD, "")
                    public_ = turn.get(PUBLIC_NARRATIVE_FIELD, "")
                    if private and public_:
                        dist = _nli_bidirectional(analyzer, private, public_)
                        sc_by_agent.setdefault(agent_id, {}).setdefault(turn_num, []).append(dist)
                        repeat_sc_by_agent.setdefault(agent_id, {}).setdefault(turn_num, []).append(dist)

            if len(agent_ids) >= 2:
                a0, a1 = agent_ids[0], agent_ids[1]
                t0 = {
                    int(t.get("turn_num", i + 1)): t
                    for i, t in enumerate(debate_data[a0]["debate_turns"])
                }
                t1 = {
                    int(t.get("turn_num", i + 1)): t
                    for i, t in enumerate(debate_data[a1]["debate_turns"])
                }
                for tn in sorted(set(t0) & set(t1)):
                    ta = t0[tn].get(PUBLIC_NARRATIVE_FIELD, "")
                    tb = t1[tn].get(PUBLIC_NARRATIVE_FIELD, "")
                    if ta and tb:
                        dist = _nli_bidirectional(analyzer, ta, tb)
                        cpa_by_turn.setdefault(tn, []).append(dist)
                        repeat_cpa_by_turn.setdefault(tn, []).append(dist)
                    ta_priv = t0[tn].get(PRIVATE_NARRATIVE_FIELD, "")
                    tb_priv = t1[tn].get(PRIVATE_NARRATIVE_FIELD, "")
                    if ta_priv and tb_priv:
                        dist = _nli_bidirectional(analyzer, ta_priv, tb_priv)
                        cpriva_by_turn.setdefault(tn, []).append(dist)
                        repeat_cpriva_by_turn.setdefault(tn, []).append(dist)

            repeat_results.append(
                _nli_result_from_buckets(
                    repeat_sc_by_agent,
                    repeat_cpa_by_turn,
                    repeat_cpriva_by_turn,
                    id2label
                    if id2label is not None
                    else {0: "contradiction", 1: "neutral", 2: "entailment"},
                )
            )

        agg_id2label: dict = id2label if id2label is not None else {
            0: "contradiction", 1: "neutral", 2: "entailment"
        }
        nli_result = _nli_result_from_buckets(
            sc_by_agent,
            cpa_by_turn,
            cpriva_by_turn,
            agg_id2label,
        )
        if self._nli_cache is None:
            self._nli_cache = {}
        if self._nli_repeat_cache is None:
            self._nli_repeat_cache = {}
        self._nli_cache[cache_key] = nli_result
        self._nli_repeat_cache[cache_key] = repeat_results
        return nli_result

    def run_nli_analysis_all_repeats(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ) -> list[dict]:
        """Return per-repeat NLI distributions, reusing the aggregate NLI cache."""
        resolved_model_name = model_name or DEFAULT_NLI_MODEL_NAME
        cache_key = (resolved_model_name, device)
        if self._nli_repeat_cache is not None and cache_key in self._nli_repeat_cache:
            return self._nli_repeat_cache[cache_key]
        self.run_nli_analysis(model_name=model_name, device=device)
        return self._nli_repeat_cache[cache_key] if self._nli_repeat_cache is not None else []

    def run_emotion_analysis(
        self,
        field: str,
        model_name: str | None = None,
        device: str | None = None,
    ) -> dict:
        """Classify emotions turn-by-turn across all repeats, return aggregated probs.

        Returns ``{agent_id: {"turns": [...], "emotions": {label: {"mean": [...],
        "std": [...]}}}}``

        The model is loaded once and reused across repeats.
        Results are cached per ``(field, model_name, device)``; repeated calls are free.
        """
        resolved_model_name = model_name or DEFAULT_EMOTION_MODEL
        cache_key = (field, resolved_model_name, device)
        if cache_key in self._emotion_caches:
            return self._emotion_caches[cache_key]

        from .debate_history import get_structured_debate_history
        from .emotion_analyzer import EmotionAnalyzer

        ea: Any = None
        by_agent: dict[str, dict[int, dict[str, list[float]]]] = {}

        for res in self.results:
            structured_history = res.agora.structured_history()
            if ea is None:
                ea = EmotionAnalyzer(
                    structured_history,
                    model_name=resolved_model_name,
                    device=device,
                )
                _ = ea.pipeline  # force load
            else:
                if isinstance(structured_history, dict) and "turns" in structured_history:
                    ea.debate_data = get_structured_debate_history(structured_history)
                else:
                    ea.debate_data = structured_history

            field_result = ea.classify_field(field)
            for agent_id, data in field_result.items():
                turns = data["turns"]
                emotions = data["emotions"]
                for i, turn_num in enumerate(turns):
                    for label, vals in emotions.items():
                        (
                            by_agent.setdefault(agent_id, {})
                            .setdefault(int(turn_num), {})
                            .setdefault(label, [])
                            .append(float(vals[i]))
                        )

        agg_result: dict[str, Any] = {}
        for agent_id, turn_dict in by_agent.items():
            turns_sorted = sorted(turn_dict)
            all_labels = sorted({lbl for t in turn_dict.values() for lbl in t})
            agg_result[agent_id] = {
                "turns": turns_sorted,
                "emotions": {
                    lbl: {
                        "mean": [float(np.mean(turn_dict[t].get(lbl, [0.0]))) for t in turns_sorted],
                        "se": [
                            float(
                                np.std(turn_dict[t].get(lbl, [0.0]))
                                / np.sqrt(max(len(turn_dict[t].get(lbl, [0.0])), 1))
                            )
                            for t in turns_sorted
                        ],
                    }
                    for lbl in all_labels
                },
            }
        self._emotion_caches[cache_key] = agg_result
        return agg_result

    # ------------------------------------------------------------------ summary

    def summary(self) -> None:
        """Print a concise aggregated summary across all repeats."""
        alpha_name, beta_name = self.agent_names
        sem = self.aggregate_semantic()
        pers = self.aggregate_persona()

        def _fmt(v: float) -> str:
            return f"{v:.3f}"

        print("=" * 64)
        print("EXPERIMENT GROUP SUMMARY")
        print("=" * 64)
        print(f"  Fingerprint : {self.group.config_fingerprint[:16]}…")
        print(f"  Repeats     : {self.n_repeats}")
        print(f"  Agent α     : {alpha_name}")
        print(f"  Agent β     : {beta_name}")
        print()

        print("-" * 64)
        print("SEMANTIC SIMILARITY  (mean ± SE across repeats, averaged over turns)")
        print("-" * 64)
        sc = sem.get("self_consistency")
        if sc:
            print("\n● Self-Consistency")
            for agent_id, data in sc.items():
                m = float(np.mean(data["mean"]))
                s = float(np.mean(data["se"]))
                print(f"    {agent_id:30s}  mean={_fmt(m)}  ±{_fmt(s)}")
        else:
            print("\n  self_consistency: not computed")
        for key, label in (
            ("cross_agent_public_alignment", "Cross-Agent Public Alignment"),
            ("cross_agent_private_alignment", "Cross-Agent Private Alignment"),
        ):
            data = sem.get(key)
            if data:
                m = float(np.mean(data["mean"]))
                s = float(np.mean(data["se"]))
                print(f"\n● {label}")
                print(f"    mean={_fmt(m)}  ±{_fmt(s)}")
            else:
                print(f"\n  {key}: not computed")

        print()
        print("-" * 64)
        print("PERSONA ADHERENCE  (mean ± SE across repeats)")
        print("-" * 64)
        if pers:
            for role, rname in (("alpha", alpha_name), ("beta", beta_name)):
                rdata = pers.get(role, {})
                computed = rdata.get("computed_metrics", [])
                print(f"\n● {rname}  (metrics: {computed})")
                for key, label in (
                    ("full_debate_public_score", "Full-debate public"),
                    ("full_debate_private_score", "Full-debate private"),
                ):
                    score = rdata.get(key)
                    if score and score.get("mean") is not None:
                        print(f"    {label:25s}: mean={_fmt(score['mean'])}  ±{_fmt(score['se'])}")
        else:
            print("  persona_adherence: not computed")

        print()
        print("=" * 64)

    # ------------------------------------------------------------------ plots

    def plot_semantic(self) -> None:
        """Plot semantic similarity metrics with error bars across repeats."""
        from .plotting import plot_group_semantic_similarity

        alpha_name, beta_name = self.agent_names
        plot_group_semantic_similarity(self.aggregate_semantic(), alpha_name, beta_name)

    def plot_persona(self) -> None:
        """Plot persona adherence with error bars, aggregated across repeats."""
        from .plotting import plot_persona_adherence

        pers = self.aggregate_persona()
        if not pers:
            print("No persona adherence data to plot.")
            return
        alpha_name, beta_name = self.agent_names
        plot_persona_adherence(pers, alpha_name, beta_name, show_plot=True)

    def plot_nli(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        """Plot NLI class distributions with ±1σ bands, aggregated across repeats."""
        from .plotting import plot_group_nli

        agg = self.run_nli_analysis(model_name=model_name, device=device)
        alpha_name, beta_name = self.agent_names
        plot_group_nli(agg, alpha_name, beta_name)

    def plot_emotions(
        self,
        field: str,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        """Plot emotion probabilities with ±1σ bands, aggregated across repeats.

        Always runs both public and private analyses so the colour/marker legend
        is consistent across both ``plot_emotions`` calls.
        """
        from .emotion_analyzer import PRIVATE_NARRATIVE_FIELD, PUBLIC_NARRATIVE_FIELD
        from .plotting import build_emotion_style, plot_group_emotions

        pub = self.run_emotion_analysis(
            PUBLIC_NARRATIVE_FIELD,
            model_name=model_name,
            device=device,
        )
        priv = self.run_emotion_analysis(
            PRIVATE_NARRATIVE_FIELD,
            model_name=model_name,
            device=device,
        )
        style = build_emotion_style([pub, priv])
        data = pub if field == PUBLIC_NARRATIVE_FIELD else priv
        alpha_name, beta_name = self.agent_names
        label = (
            "Public Utterances"
            if field == PUBLIC_NARRATIVE_FIELD
            else "Private Reflections"
        )
        plot_group_emotions(data, label, alpha_name, beta_name, emotion_style=style)

    def plot_survey(
        self,
        survey_questions=None,
    ) -> None:
        """Plot aggregated survey responses with error bars across repeats.

        Parameters
        ----------
        survey_questions:
            Optional question specs used to label panels and group incentive questions.
            Accepts the same format as :func:`~agora.plotting.plot_survey_responses`.
            When *None*, the specs stored on the result object are used automatically
            (populated from the experiment run), falling back to bare Q-key labels.
        """
        from .plotting import plot_group_survey

        effective_questions = survey_questions if survey_questions is not None else self.survey_question_specs
        agg = self.aggregate_survey(survey_questions=effective_questions or None)
        if not agg:
            print("No survey data to plot.")
            return
        alpha_name, beta_name = self.agent_names
        plot_group_survey(agg, alpha_name, beta_name, survey_questions=effective_questions or None)

    def plot_response_decisions(
        self,
        scenario_id: str | None = None,
        catalog_path: "Path | str | None" = None,
    ) -> None:
        """Plot binary response decision fractions over turns, aggregated across repeats.

        Produces two figures matching the NLI plot layout:

        * **Figure 1** — public vs. private per agent (left = α, right = β).
        * **Figure 2** — cross-agent comparison: public (left) and private (right).

        Parameters
        ----------
        scenario_id:
            Passed verbatim to :meth:`aggregate_response_decisions`.  Omit to
            use the value from ``sweep_values``.
        catalog_path:
            Path to the scenarios catalog.  Omit to use the default.
        """
        from .plotting import plot_group_response_decisions

        agg = self.aggregate_response_decisions(scenario_id=scenario_id, catalog_path=catalog_path)
        if not agg.get("by_slot"):
            print("No response decision data to plot.")
            return
        alpha_name, beta_name = self.agent_names
        plot_group_response_decisions(agg, alpha_name, beta_name)

    def plot_response_decisions_all_repeats(
        self,
        scenario_id: str | None = None,
        catalog_path: "Path | str | None" = None,
    ) -> None:
        """Plot per-repeat binary response decisions as line plots over turns.

        Each repeat is rendered as a separate line with a distinct marker shape;
        colour encodes public vs. private (Figure 1) or alpha vs. beta (Figure 2).

        Parameters
        ----------
        scenario_id, catalog_path:
            Passed verbatim to :meth:`aggregate_response_decisions_all_repeats`.
        """
        from .plotting import plot_group_response_decisions_all_repeats

        data = self.aggregate_response_decisions_all_repeats(
            scenario_id=scenario_id, catalog_path=catalog_path
        )
        if not any(data.get("repeats", [])):
            print("No response decision data to plot.")
            return
        alpha_name, beta_name = self.agent_names
        plot_group_response_decisions_all_repeats(data, alpha_name, beta_name)


# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ExperimentGroup:
    """All repeats that share the same config fingerprint (i.e. the same experiment)."""

    config_fingerprint: str
    sweep_values: dict[str, Any]
    cases: list[SweepCase]

    @property
    def repeat_count(self) -> int:
        return len(self.cases)

    @property
    def case_ids(self) -> list[str]:
        return [c.case_id for c in self.cases]

    def abs_case_dirs(self, sweep_root: Path) -> list[Path]:
        """Absolute paths to each case directory."""
        return [sweep_root / c.case_dir for c in self.cases]

    def abs_config_paths(self, sweep_root: Path) -> list[Path]:
        """Absolute paths to each case config.json."""
        return [sweep_root / c.config_path for c in self.cases]

    def run_analysis(self, sweep_root: Path, **postpro: Any) -> GroupAnalysisResult:
        """Run offline post-processing for every repeat and return a grouped result.

        Parameters
        ----------
        sweep_root:
            Absolute root of the sweep directory.
        **postpro:
            Forwarded verbatim to :meth:`SweepCase.run_analysis` for each repeat.
        """
        results = []
        analyzed_cases = []
        for case in self.cases:
            try:
                results.append(case.run_analysis(sweep_root, **postpro))
                analyzed_cases.append(case)
            except FileNotFoundError as exc:
                context = format_case_warning_context(case, sweep_root)
                print(f"Warning: skipping case {case.case_id}{context}: {exc}")
        return GroupAnalysisResult(group=self, results=results, analyzed_cases=analyzed_cases)


@dataclass(slots=True)
class SweepManifest:
    """
    Structured, read-only view of a sweep's manifest.json.

    Groups all cases by their ``config_fingerprint`` so that each
    :class:`ExperimentGroup` represents one unique experiment with one or more
    repeats.

    Typical usage::

        manifest = SweepManifest.from_path("outputs/sweeps/manifest.json")
        for group in manifest:
            print(group.config_fingerprint, group.repeat_count)
    """

    schema_version: int
    generated_at: str
    sweep_root: Path
    notes: str | None
    total_cases: int
    groups: list[ExperimentGroup]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_path(cls, path: Path | str) -> SweepManifest:
        """Load from a ``manifest.json`` file or the directory that contains it."""
        manifest_path = Path(path)
        if manifest_path.is_dir():
            manifest_path = manifest_path / "manifest.json"
        sweep_root = manifest_path.resolve().parent

        data: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))

        groups_by_fp: dict[str, ExperimentGroup] = {}
        for raw in data.get("cases", []):
            fp = raw["config_fingerprint"]
            if fp not in groups_by_fp:
                groups_by_fp[fp] = ExperimentGroup(
                    config_fingerprint=fp,
                    sweep_values=dict(raw.get("sweep_values", {})),
                    cases=[],
                )
            groups_by_fp[fp].cases.append(
                SweepCase(
                    case_id=raw["case_id"],
                    case_dir=Path(raw["case_dir"]),
                    config_path=Path(raw["config_path"]),
                    label=raw["label"],
                    repeat_number=raw["repeat_number"],
                    repeat_count=raw["repeat_count"],
                    sweep_values=dict(raw.get("sweep_values", {})),
                )
            )

        return cls(
            schema_version=data["schema_version"],
            generated_at=data["generated_at"],
            sweep_root=sweep_root,
            notes=data.get("notes"),
            total_cases=data["total_cases"],
            groups=list(groups_by_fp.values()),
        )

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def experiment_count(self) -> int:
        """Number of unique experiments (distinct config fingerprints)."""
        return len(self.groups)

    def __len__(self) -> int:
        return len(self.groups)

    def __iter__(self) -> Iterator[ExperimentGroup]:
        return iter(self.groups)

    def __getitem__(self, key: int | str) -> ExperimentGroup:
        """Access a group by integer index or config fingerprint string."""
        if isinstance(key, int):
            return self.groups[key]
        for group in self.groups:
            if group.config_fingerprint == key:
                return group
        raise KeyError(key)
