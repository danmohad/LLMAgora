"""Read-only view over a completed sweep, grouped by config fingerprint."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields as dc_fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np

if TYPE_CHECKING:
    from .experiment import ExperimentResult


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
    _semantic_cache: dict | None = field(default=None, repr=False, compare=False, init=False)
    _persona_cache: dict | None = field(default=None, repr=False, compare=False, init=False)
    _nli_cache: dict | None = field(default=None, repr=False, compare=False, init=False)
    _emotion_caches: dict = field(default_factory=dict, repr=False, compare=False, init=False)

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
            sem = res.eval_data.get("semantic_similarity", {})
            for agent_id, data in sem.get("self_consistency", {}).items():
                sc_by_agent.setdefault(agent_id, []).append(data)
            cpa = sem.get("cross_agent_public_alignment")
            if cpa:
                cpa_series.append(cpa)
            cpriva = sem.get("cross_agent_private_alignment")
            if cpriva:
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

    # ------------------------------------------------------------------ on-demand analyses

    def run_nli_analysis(self, model_name: str | None = None) -> dict:
        """Run bidirectional NLI analysis across all repeats and return aggregated distributions.

        Computes self-consistency (private→public) and cross-agent public alignment NLI
        for every repeat, then aggregates to mean ± std per turn.

        Structure: ``{metric: {key: {"turns": [...], "label_names": [...],
        "distributions": {name: {"mean": [...], "std": [...]}}}}}``

        Results are cached; repeated calls are free.
        """
        if self._nli_cache is not None:
            return self._nli_cache

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

        for res in self.results:
            structured_history = res.agora.structured_history()
            if analyzer is None:
                kwargs: dict[str, Any] = {"method": SEMANTIC_SIMILARITY_METHOD_NLI}
                if model_name:
                    kwargs["model_name"] = model_name
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

            for agent_id in agent_ids:
                for i, turn in enumerate(debate_data[agent_id]["debate_turns"]):
                    turn_num = int(turn.get("turn_num", i + 1))
                    private = turn.get(PRIVATE_NARRATIVE_FIELD, "")
                    public_ = turn.get(PUBLIC_NARRATIVE_FIELD, "")
                    if private and public_:
                        dist = _nli_bidirectional(analyzer, private, public_)
                        sc_by_agent.setdefault(agent_id, {}).setdefault(turn_num, []).append(dist)

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

        agg_id2label: dict = id2label if id2label is not None else {
            0: "contradiction", 1: "neutral", 2: "entailment"
        }
        nli_result: dict[str, Any] = {"id2label": agg_id2label}
        if sc_by_agent:
            nli_result["self_consistency"] = {
                aid: _agg_nli_by_turn(td, agg_id2label) for aid, td in sc_by_agent.items()
            }
        if cpa_by_turn:
            nli_result["cross_agent_public"] = _agg_nli_by_turn(cpa_by_turn, agg_id2label)
        self._nli_cache = nli_result
        return nli_result

    def run_emotion_analysis(
        self, field: str, model_name: str = "cirimus/modernbert-base-emotions"
    ) -> dict:
        """Classify emotions turn-by-turn across all repeats, return aggregated probs.

        Returns ``{agent_id: {"turns": [...], "emotions": {label: {"mean": [...],
        "std": [...]}}}}``

        The model is loaded once and reused across repeats.
        Results are cached per ``(field, model_name)``; repeated calls are free.
        """
        cache_key = (field, model_name)
        if cache_key in self._emotion_caches:
            return self._emotion_caches[cache_key]

        from .debate_history import get_structured_debate_history
        from .emotion_analyzer import EmotionAnalyzer

        ea: Any = None
        by_agent: dict[str, dict[int, dict[str, list[float]]]] = {}

        for res in self.results:
            structured_history = res.agora.structured_history()
            if ea is None:
                ea = EmotionAnalyzer(structured_history, model_name=model_name)
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

    def plot_nli(self, model_name: str | None = None) -> None:
        """Plot NLI class distributions with ±1σ bands, aggregated across repeats."""
        from .plotting import plot_group_nli

        agg = self.run_nli_analysis(model_name=model_name)
        alpha_name, beta_name = self.agent_names
        plot_group_nli(agg, alpha_name, beta_name)

    def plot_emotions(
        self, field: str, model_name: str = "cirimus/modernbert-base-emotions"
    ) -> None:
        """Plot emotion probabilities with ±1σ bands, aggregated across repeats.

        Always runs both public and private analyses so the colour/marker legend
        is consistent across both ``plot_emotions`` calls.
        """
        from .emotion_analyzer import PRIVATE_NARRATIVE_FIELD, PUBLIC_NARRATIVE_FIELD
        from .plotting import build_emotion_style, plot_group_emotions

        pub = self.run_emotion_analysis(PUBLIC_NARRATIVE_FIELD, model_name)
        priv = self.run_emotion_analysis(PRIVATE_NARRATIVE_FIELD, model_name)
        style = build_emotion_style([pub, priv])
        data = pub if field == PUBLIC_NARRATIVE_FIELD else priv
        alpha_name, beta_name = self.agent_names
        label = (
            "Public Utterances"
            if field == PUBLIC_NARRATIVE_FIELD
            else "Private Reflections"
        )
        plot_group_emotions(data, label, alpha_name, beta_name, emotion_style=style)


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
        results = [case.run_analysis(sweep_root, **postpro) for case in self.cases]
        return GroupAnalysisResult(group=self, results=results)


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
