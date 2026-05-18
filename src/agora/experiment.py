"""High-level persona experiment workflow for notebooks and CLI."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from uuid import uuid4

import json5
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

from .agent import Agent
from .agora import ALLOWED_SUBTURN_EVENTS, Agora
from .persona_adherence_evaluator import (
    PERSONA_ANALYSIS_METRICS,
    PersonaEvaluator,
)
from .semantic_similarity_analyzer import (
    SEMANTIC_SIMILARITY_METHOD_COSINE,
    SEMANTIC_SIMILARITY_METHOD_NLI,
    SEMANTIC_SIMILARITY_METHODS,
    PRIVATE_NARRATIVE_FIELD,
    PUBLIC_NARRATIVE_FIELD,
    SemanticSimilarityAnalyzer,
)
from .llm import OpenRouterClient
from .plotting import (
    plot_persona_adherence,
    plot_survey_distance,
    plot_survey_responses,
)
from .survey import (
    merge_survey_question_configs,
    survey_question_groups,
    survey_question_texts,
)
from .workflows import (
    build_scenario_agent_configs,
    load_debate_construction,
    load_prompt_catalog,
    load_prompt_templates,
    run_debate_session,
)

DEFAULT_CATALOG_PATH = Path("data/scenarios.json")
DEFAULT_PROMPTS_PATH = Path("data/prompts.json")
DEFAULT_OUTPUTS_ROOT = Path("outputs")
DEFAULT_INDEX_CSV = DEFAULT_OUTPUTS_ROOT / "index.csv"
EXPERIMENT_PATH_FIELDS = frozenset(
    {
        "outputs_root",
        "index_csv",
        "load_dir",
        "output_dir",
        "catalog_path",
        "prompts_path",
    }
)
_PRESERVE_EXPLICIT_NONE_FIELDS = frozenset()

SEMANTIC_METRIC_SELF_CONSISTENCY = "self_consistency"
SEMANTIC_METRIC_CROSS_AGENT_PUBLIC_ALIGNMENT = "cross_agent_public_alignment"
SEMANTIC_METRIC_CROSS_AGENT_PRIVATE_ALIGNMENT = "cross_agent_private_alignment"
SEMANTIC_ANALYSIS_METRICS: tuple[str, ...] = (
    SEMANTIC_METRIC_SELF_CONSISTENCY,
    SEMANTIC_METRIC_CROSS_AGENT_PUBLIC_ALIGNMENT,
    SEMANTIC_METRIC_CROSS_AGENT_PRIVATE_ALIGNMENT,
)


@dataclass(slots=True)
class ExperimentConfig:
    """Single-source experiment configuration for notebooks and CLI."""

    scenario_id: str
    incentive_direction: Optional[str] = None
    incentive_type: str = "historical"
    prompt_set: str = "default"
    model: str = "openai/gpt-4o-mini"
    num_turns: int = 2
    subturn_event_order: list[str] = field(
        default_factory=lambda: ["public_utterance"]
    )
    verbose: bool = False

    keep_private_reflection: bool = False

    keep_pre_interview: bool = False

    keep_post_interview: bool = False

    keep_public_survey: bool = False
    keep_private_survey: bool = False

    semantic_analysis_metrics: list[str] = field(default_factory=list)
    semantic_similarity_method: Optional[str] = None
    semantic_similarity_model: Optional[str] = None
    semantic_similarity_device: Optional[str] = None
    persona_analysis_metrics: list[str] = field(default_factory=list)
    persona_scoring_model: Optional[str] = None
    persona_scoring_verbose: bool = False
    persona_score_samples: Optional[int] = None

    save_plots: bool = False
    show_plots: bool = False

    load_snapshot: bool = False
    load_dir: Optional[Path] = None
    save_snapshot: bool = False
    reuse_load_dir_for_outputs: bool = False

    output_dir: Optional[Path] = None
    outputs_root: Path = DEFAULT_OUTPUTS_ROOT
    run_name: Optional[str] = None
    indexed_output: bool = False
    index_csv: Optional[Path] = None

    catalog_path: Path = DEFAULT_CATALOG_PATH
    prompts_path: Path = DEFAULT_PROMPTS_PATH

    @property
    def enable_private_reflection(self) -> bool:
        return "private_utterance" in self.subturn_event_order

    @property
    def enable_public_survey(self) -> bool:
        return "public_survey" in self.subturn_event_order

    @property
    def enable_private_survey(self) -> bool:
        return "private_survey" in self.subturn_event_order


@dataclass(slots=True)
class ExperimentResult:
    """Artifacts produced by one experiment run."""

    agora: Agora
    agents: list[Agent]
    eval_data: dict[str, Any]
    run_dir: Optional[Path]
    run_id: Optional[str]
    semantic_analyzer: Optional[SemanticSimilarityAnalyzer]
    persona_adherence_eval: Optional[dict[str, Any]]
    survey_question_specs: list = field(default_factory=list)


def _coerce_path(value: Any, *, fallback: Path) -> Path:
    if value is None:
        return fallback
    if isinstance(value, Path):
        return value
    return Path(str(value))


def _coerce_optional_path(value: Any) -> Optional[Path]:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    return Path(str(value))


def _resolve_relative_path(path: Path, *, base_dir: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def resolve_experiment_payload_paths(
    payload: Mapping[str, Any], *, base_dir: Path
) -> dict[str, Any]:
    resolved = dict(payload)
    for field_name in EXPERIMENT_PATH_FIELDS & resolved.keys():
        value = resolved[field_name]
        if value is None:
            continue
        path_value = value if isinstance(value, Path) else Path(str(value))
        resolved[field_name] = _resolve_relative_path(path_value, base_dir=base_dir)
    return resolved


def _coerce_event_order(value: Any) -> Optional[list[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        return items
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    raise ValueError("subturn_event_order must be a list/tuple of event names")


def _coerce_metric_list(value: Any, *, field_name: str) -> Optional[list[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    raise ValueError(f"{field_name} must be a list/tuple of metric names")


def _validate_metric_list(
    *,
    values: Sequence[str],
    allowed: Sequence[str],
    field_name: str,
) -> None:
    unknown = [metric for metric in values if metric not in allowed]
    if unknown:
        raise ValueError(
            f"{field_name} contains unknown metrics: {sorted(set(unknown))}. "
            f"Allowed: {list(allowed)}"
        )
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicates")


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return slug.strip("_") or "run"


def _prompt_set_payload(prompt_catalog: dict[str, Any], prompt_set: str) -> dict[str, Any]:
    return load_prompt_templates(prompt_set, prompt_catalog=prompt_catalog)


def _resolve_run_dir(cfg: ExperimentConfig) -> tuple[Path, Optional[str]]:
    outputs_root = cfg.outputs_root
    outputs_root.mkdir(parents=True, exist_ok=True)

    if cfg.indexed_output:
        run_id = uuid4().hex[:6]
        run_dir = outputs_root / run_id
        while run_dir.exists():
            run_id = uuid4().hex[:6]
            run_dir = outputs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir, run_id

    incentive_label = (
        "no_incentive"
        if cfg.incentive_direction is None
        else f"{cfg.incentive_direction}_{cfg.incentive_type}"
    )
    base_name = cfg.run_name or f"{_slug(cfg.scenario_id)}_{_slug(incentive_label)}"

    run_dir = outputs_root / base_name
    if not run_dir.exists():
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir, None

    index = 2
    while True:
        candidate = outputs_root / f"{base_name}_{index}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate, None
        index += 1


def _save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_index_row(
    *,
    cfg: ExperimentConfig,
    run_id: str,
    run_dir: Path,
    index_csv: Path,
) -> None:
    row = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **{k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
    }

    index_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = index_csv.exists()

    with index_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _resolve_index_csv(cfg: ExperimentConfig) -> Path:
    if cfg.index_csv is not None:
        return cfg.index_csv
    return cfg.outputs_root / "index.csv"


def _plot_intra_scores(
    intra_scores: dict[str, Any],
    label_map: dict[str, str],
    output_path: Path,
    title: str,
    show_plot: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    all_turns: set[int] = set()
    for speaker, data in intra_scores.items():
        turns = [int(turn) for turn in data["turns"]]
        all_turns.update(turns)
        ax.plot(
            turns,
            data["scores"],
            marker="o",
            label=label_map.get(speaker, speaker),
        )
    ax.set_title(title)
    ax.set_xlabel("Debate Turn")
    ax.set_ylabel("Semantic Similarity Score")
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
    if all_turns:
        ax.set_xticks(sorted(all_turns))
    ax.legend()
    ax.grid()
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)


def _plot_inter_scores(
    public_alignment_scores: Optional[dict[str, Any]],
    private_alignment_scores: Optional[dict[str, Any]],
    output_path: Path,
    title: str,
    show_plot: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    all_turns: set[int] = set()
    if public_alignment_scores:
        public_turns = [int(turn) for turn in public_alignment_scores["turns"]]
        all_turns.update(public_turns)
        ax.plot(
            public_turns,
            public_alignment_scores["scores"],
            marker="o",
            label="Cross-Agent Public Alignment",
        )
    if private_alignment_scores:
        private_turns = [int(turn) for turn in private_alignment_scores["turns"]]
        all_turns.update(private_turns)
        ax.plot(
            private_turns,
            private_alignment_scores["scores"],
            marker="o",
            label="Cross-Agent Private Alignment",
        )
    ax.set_title(title)
    ax.set_xlabel("Debate Turn")
    ax.set_ylabel("Semantic Similarity Score")
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
    if all_turns:
        ax.set_xticks(sorted(all_turns))
    if ax.has_data():
        ax.legend()
    ax.grid()
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)


def _scenario_entry(catalog: dict[str, Any], scenario_id: str) -> dict[str, Any]:
    scenarios = catalog.get("scenarios", [])
    scenario = next((s for s in scenarios if s.get("scenario_id") == scenario_id), None)
    if scenario is None:
        raise KeyError(f"Scenario '{scenario_id}' not found")
    return scenario


def _ensure_required(config: dict[str, Any], required: str) -> Any:
    value = config.get(required)
    if value in (None, ""):
        raise ValueError(f"Missing required experiment setting: {required}")
    return value


def build_experiment_config(payload: Mapping[str, Any]) -> ExperimentConfig:
    """Create a validated ``ExperimentConfig`` from dictionary data."""

    data = {
        key: value
        for key, value in dict(payload).items()
        if value is not None or key in _PRESERVE_EXPLICIT_NONE_FIELDS
    }
    data["scenario_id"] = _ensure_required(data, "scenario_id")

    path_fields: tuple[tuple[str, Path], ...] = (
        ("outputs_root", DEFAULT_OUTPUTS_ROOT),
        ("catalog_path", DEFAULT_CATALOG_PATH),
        ("prompts_path", DEFAULT_PROMPTS_PATH),
    )
    for field_name, fallback in path_fields:
        data[field_name] = _coerce_path(data.get(field_name), fallback=fallback)
    data["index_csv"] = _coerce_optional_path(data.get("index_csv"))
    data["load_dir"] = _coerce_optional_path(data.get("load_dir"))
    data["output_dir"] = _coerce_optional_path(data.get("output_dir"))
    coerced_event_order = _coerce_event_order(data.get("subturn_event_order"))
    if coerced_event_order is not None:
        data["subturn_event_order"] = coerced_event_order
    else:
        data.pop("subturn_event_order", None)
    semantic_metrics = _coerce_metric_list(
        data.get("semantic_analysis_metrics"),
        field_name="semantic_analysis_metrics",
    )
    if semantic_metrics is not None:
        data["semantic_analysis_metrics"] = semantic_metrics
    persona_metrics = _coerce_metric_list(
        data.get("persona_analysis_metrics"),
        field_name="persona_analysis_metrics",
    )
    if persona_metrics is not None:
        data["persona_analysis_metrics"] = persona_metrics

    cfg = ExperimentConfig(**data)
    if cfg.num_turns < 0:
        raise ValueError("num_turns must be non-negative")
    if cfg.num_turns == 0 and not cfg.load_snapshot:
        raise ValueError("num_turns can be zero only when load_snapshot is enabled")
    if cfg.persona_score_samples is not None and cfg.persona_score_samples <= 0:
        raise ValueError("persona_score_samples must be positive")
    if cfg.semantic_analysis_metrics:
        if cfg.semantic_similarity_method is None:
            raise ValueError(
                "semantic_similarity_method is required when semantic_analysis_metrics is enabled"
            )
        if cfg.semantic_similarity_model in {None, ""}:
            raise ValueError(
                "semantic_similarity_model is required when semantic_analysis_metrics is enabled"
            )
    if cfg.incentive_direction not in {None, "positive", "negative"}:
        raise ValueError(
            "incentive_direction must be one of: positive, negative, or None"
        )
    if cfg.incentive_type not in {"historical", "future"}:
        raise ValueError("incentive_type must be 'historical' or 'future'")
    if cfg.show_plots and not cfg.save_plots:
        raise ValueError("show_plots requires save_plots=True")
    if (
        cfg.semantic_similarity_method is not None
        and cfg.semantic_similarity_method not in SEMANTIC_SIMILARITY_METHODS
    ):
        raise ValueError(
            "semantic_similarity_method must be one of "
            f"{list(SEMANTIC_SIMILARITY_METHODS)}"
        )
    # Normalize to these strings for the semantic backend; cuda is valid when CUDA is available.
    if cfg.semantic_similarity_device not in {None, "cpu", "mps", "cuda"}:
        raise ValueError("semantic_similarity_device must be one of: cpu, mps, cuda")
    if cfg.keep_private_reflection and not cfg.enable_private_reflection:
        raise ValueError(
            "keep_private_reflection requires private_utterance in subturn_event_order"
        )
    if cfg.keep_public_survey and not cfg.enable_public_survey:
        raise ValueError("keep_public_survey requires public_survey in subturn_event_order")
    if cfg.keep_private_survey and not cfg.enable_private_survey:
        raise ValueError("keep_private_survey requires private_survey in subturn_event_order")
    _validate_metric_list(
        values=cfg.semantic_analysis_metrics,
        allowed=SEMANTIC_ANALYSIS_METRICS,
        field_name="semantic_analysis_metrics",
    )
    _validate_metric_list(
        values=cfg.persona_analysis_metrics,
        allowed=PERSONA_ANALYSIS_METRICS,
        field_name="persona_analysis_metrics",
    )
    if cfg.persona_analysis_metrics and cfg.persona_scoring_model in {None, ""}:
        raise ValueError(
            "persona_scoring_model is required when persona_analysis_metrics is enabled"
        )
    if cfg.persona_analysis_metrics and cfg.persona_score_samples is None:
        raise ValueError(
            "persona_score_samples is required when persona_analysis_metrics is enabled"
        )
    if cfg.persona_scoring_verbose and not cfg.persona_analysis_metrics:
        raise ValueError(
            "persona_scoring_verbose requires at least one persona_analysis_metrics value"
        )
    if cfg.load_snapshot and cfg.load_dir is None:
        raise ValueError("load_dir must be provided when load_snapshot is enabled")
    if not cfg.load_snapshot and cfg.load_dir is not None:
        raise ValueError("load_dir must be None when load_snapshot is disabled")
    if cfg.reuse_load_dir_for_outputs and not cfg.load_snapshot:
        raise ValueError("reuse_load_dir_for_outputs requires load_snapshot=True")
    if cfg.reuse_load_dir_for_outputs and cfg.num_turns != 0:
        raise ValueError("reuse_load_dir_for_outputs requires num_turns=0")
    if cfg.reuse_load_dir_for_outputs and cfg.indexed_output:
        raise ValueError("reuse_load_dir_for_outputs cannot be combined with indexed_output")
    if cfg.output_dir is not None and cfg.indexed_output:
        raise ValueError("output_dir cannot be combined with indexed_output")
    if cfg.output_dir is not None and cfg.run_name is not None:
        raise ValueError("output_dir cannot be combined with run_name")
    if cfg.output_dir is not None and cfg.index_csv is not None:
        raise ValueError("output_dir cannot be combined with index_csv")
    if cfg.output_dir is not None and cfg.reuse_load_dir_for_outputs:
        raise ValueError("output_dir cannot be combined with reuse_load_dir_for_outputs")
    if not cfg.subturn_event_order:
        raise ValueError("subturn_event_order must not be empty")
    unknown_events = [
        event for event in cfg.subturn_event_order if event not in ALLOWED_SUBTURN_EVENTS
    ]
    if unknown_events:
        raise ValueError(
            "subturn_event_order contains unknown events: "
            + ", ".join(sorted(set(unknown_events)))
        )
    if len(cfg.subturn_event_order) != len(set(cfg.subturn_event_order)):
        raise ValueError("subturn_event_order must not contain duplicates")
    if "public_utterance" not in cfg.subturn_event_order:
        raise ValueError("subturn_event_order must include public_utterance")
    return cfg


def load_experiment_config(path: Path | str) -> ExperimentConfig:
    """Load an experiment config JSON or JSONC file."""

    config_path = Path(path)
    raw_text = config_path.read_text(encoding="utf-8")
    payload = (
        json5.loads(raw_text)
        if config_path.suffix == ".jsonc"
        else json.loads(raw_text)
    )
    if not isinstance(payload, dict):
        raise ValueError("Experiment config must be a JSON object")
    resolved_payload = resolve_experiment_payload_paths(
        payload, base_dir=config_path.resolve().parent
    )
    return build_experiment_config(resolved_payload)


def _merge_config(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> ExperimentConfig:
    # ``overrides`` is expected to contain only explicitly provided CLI fields.
    # Keep explicit ``None`` values so callers can clear config values.
    merged = dict(base)
    for key, value in overrides.items():
        merged[key] = value
    return build_experiment_config(merged)


def _should_write_outputs(cfg: ExperimentConfig) -> bool:
    """Return True when this run should persist artifacts to disk."""

    if cfg.output_dir is not None:
        return True

    return any(
        [
            cfg.save_plots,
            bool(cfg.semantic_analysis_metrics),
            bool(cfg.persona_analysis_metrics),
            cfg.enable_public_survey,
            cfg.enable_private_survey,
            cfg.save_snapshot,
            cfg.indexed_output,
        ]
    )


def _survey_responses_by_agent(
    turns: list[dict], event_key: str
) -> dict[str, dict[int, dict[str, int]]]:
    responses: dict[str, dict[int, dict[str, int]]] = {}
    for turn in turns:
        turn_num = int(turn.get("turn_num", 0))
        for slot in ("Alpha", "Beta"):
            subturn = turn.get(slot, {})
            scores = subturn.get(event_key)
            if scores is None:
                continue
            speaker_id = subturn.get("speaker_id")
            if not speaker_id:
                continue
            responses.setdefault(speaker_id, {})[turn_num] = scores
    return responses


def run_persona_experiment(
    config: ExperimentConfig | Mapping[str, Any],
    *,
    emit_progress_markers: bool = False,
) -> ExperimentResult:
    """Run one scenario-based experiment and persist all artifacts in one folder."""

    cfg = (
        build_experiment_config(asdict(config))
        if isinstance(config, ExperimentConfig)
        else build_experiment_config(config)
    )

    # Runs with all optional outputs disabled execute in-memory only.
    write_outputs = _should_write_outputs(cfg)
    run_dir: Optional[Path] = None
    run_id: Optional[str] = None
    if write_outputs:
        if cfg.output_dir is not None:
            run_dir = cfg.output_dir
            run_dir.mkdir(parents=True, exist_ok=True)
        elif cfg.reuse_load_dir_for_outputs:
            run_dir = cfg.load_dir
            assert run_dir is not None  # validated by build_experiment_config
            run_dir.mkdir(parents=True, exist_ok=True)
        else:
            run_dir, run_id = _resolve_run_dir(cfg)

    catalog = load_debate_construction(cfg.catalog_path)
    prompt_catalog = load_prompt_catalog(cfg.prompts_path)
    prompt_payload = _prompt_set_payload(prompt_catalog, cfg.prompt_set)
    scenario = _scenario_entry(catalog, cfg.scenario_id)

    question_label = scenario.get("question", {}).get("topic", "question")
    sides = scenario.get("sides")
    if not isinstance(sides, dict) or len(sides) != 2:
        raise ValueError(
            f"Scenario '{cfg.scenario_id}' must define exactly two sides in scenario.sides"
        )
    side_values = [side for _, side in sides.items()]
    alpha_persona, beta_persona = side_values[0], side_values[1]

    incentive_label = (
        "none"
        if cfg.incentive_direction is None
        else f"{cfg.incentive_direction}:{cfg.incentive_type}"
    )

    survey_question_specs = []
    survey_question_group_map = {}
    survey_questions = []
    if cfg.enable_public_survey or cfg.enable_private_survey:
        scenario_survey = scenario.get("survey_questions") or {}
        survey_question_specs = merge_survey_question_configs(
            prompt_payload.get("survey_questions", []),
            scenario_survey,
        )
        survey_question_group_map = survey_question_groups(survey_question_specs)
        survey_questions = survey_question_texts(survey_question_specs)
        if not survey_questions:
            raise ValueError(
                "Survey is enabled but no survey questions are configured in prompts/scenario data."
            )

    agent_configs = build_scenario_agent_configs(
        scenario_id=cfg.scenario_id,
        catalog=catalog,
        model=cfg.model,
        incentive_direction=cfg.incentive_direction,
        incentive_type=cfg.incentive_type,
        prompt_set=cfg.prompt_set,
        prompt_catalog=prompt_catalog,
        private_response_keep=cfg.keep_private_reflection,
        pre_interview_keep=cfg.keep_pre_interview,
        post_interview_keep=cfg.keep_post_interview,
        public_survey_keep=cfg.keep_public_survey,
        private_survey_keep=cfg.keep_private_survey,
        enable_public_survey=cfg.enable_public_survey,
        enable_private_survey=cfg.enable_private_survey,
        survey_questions=survey_questions,
        survey_question_groups=survey_question_group_map,
    )

    for agent_cfg in agent_configs:
        if not cfg.enable_private_reflection:
            agent_cfg["private_response"] = {"instruction": None, "keep": False}
        if not cfg.enable_public_survey and not cfg.enable_private_survey:
            agent_cfg["survey"] = {
                "survey_questions": [],
                "survey_public_prompt": None,
                "survey_private_prompt": None,
                "survey_scale": None,
                "survey_scale_prompt": None,
                "survey_scale_value_prompt": None,
                "survey_question_prompt": None,
                "enable_public_survey": False,
                "enable_private_survey": False,
                "public_survey_keep": False,
                "private_survey_keep": False,
            }
        else:
            if not cfg.enable_public_survey:
                agent_cfg["survey"]["survey_public_prompt"] = None
                agent_cfg["survey"]["public_survey_keep"] = False
            if not cfg.enable_private_survey:
                agent_cfg["survey"]["survey_private_prompt"] = None
                agent_cfg["survey"]["private_survey_keep"] = False
            agent_cfg["survey"]["enable_public_survey"] = cfg.enable_public_survey
            agent_cfg["survey"]["enable_private_survey"] = cfg.enable_private_survey

    snapshot_path: Optional[Path] = None
    if cfg.load_snapshot:
        # Explicit source of truth for resume behavior.
        snapshot_path = cfg.load_dir / "debate_snapshot.json"
    elif cfg.save_snapshot and run_dir is not None:
        snapshot_path = run_dir / "debate_snapshot.json"

    agora, agents = run_debate_session(
        agent_configs,
        num_turns=cfg.num_turns,
        event_order=cfg.subturn_event_order,
        verbose=cfg.verbose,
        emit_progress_markers=emit_progress_markers,
        snapshot_path=snapshot_path,
        load_snapshot_flag=cfg.load_snapshot,
        save_snapshot_flag=cfg.save_snapshot,
    )
    structured_history = agora.structured_history()
    turns = structured_history.get("turns", [])
    public_survey_responses = _survey_responses_by_agent(turns, "public_survey")
    private_survey_responses = _survey_responses_by_agent(turns, "private_survey")

    selected_semantic_metrics = set(cfg.semantic_analysis_metrics)
    semantic_analyzer = None
    self_consistency_scores = None
    cross_agent_public_alignment = None
    cross_agent_private_alignment = None

    if selected_semantic_metrics:
        assert cfg.semantic_similarity_method is not None
        assert cfg.semantic_similarity_model is not None
        semantic_analyzer = SemanticSimilarityAnalyzer(
            structured_history,
            method=cfg.semantic_similarity_method,
            model_name=cfg.semantic_similarity_model,
            device=cfg.semantic_similarity_device,
        )
        if SEMANTIC_METRIC_SELF_CONSISTENCY in selected_semantic_metrics:
            self_consistency_scores = semantic_analyzer.compute_self_consistency_scores()
        if (
            SEMANTIC_METRIC_CROSS_AGENT_PUBLIC_ALIGNMENT
            in selected_semantic_metrics
        ):
            cross_agent_public_alignment = (
                semantic_analyzer.compute_cross_agent_alignment_scores(
                    PUBLIC_NARRATIVE_FIELD,
                    PUBLIC_NARRATIVE_FIELD,
                )
            )
        if (
            SEMANTIC_METRIC_CROSS_AGENT_PRIVATE_ALIGNMENT
            in selected_semantic_metrics
        ):
            cross_agent_private_alignment = (
                semantic_analyzer.compute_cross_agent_alignment_scores(
                    PRIVATE_NARRATIVE_FIELD,
                    PRIVATE_NARRATIVE_FIELD,
                )
            )

    persona_adherence_eval = None
    if cfg.persona_analysis_metrics:
        assert cfg.persona_scoring_model is not None
        assert cfg.persona_score_samples is not None
        eval_client = OpenRouterClient()
        try:
            evaluator = PersonaEvaluator(
                llm_client=eval_client,
                personas={
                    "personas": {
                        alpha_persona["id"]: alpha_persona,
                        beta_persona["id"]: beta_persona,
                    }
                },
                model=cfg.persona_scoring_model,
                scoring_prompt_template=prompt_payload["persona_scoring_prompt"],
            )
            persona_eval = evaluator.evaluate_debate_from_history(
                memory_turns=structured_history,
                alpha_persona_id=alpha_persona["id"],
                beta_persona_id=beta_persona["id"],
                verbose=cfg.persona_scoring_verbose,
                n_samples=cfg.persona_score_samples,
                metrics=cfg.persona_analysis_metrics,
            )
            persona_adherence_eval = persona_eval.to_dict()
        finally:
            if hasattr(eval_client, "close"):
                eval_client.close()

    alpha_name = alpha_persona.get("name", "Alpha")
    beta_name = beta_persona.get("name", "Beta")

    if cfg.save_plots and run_dir is not None:
        label_map = {"Alpha": f"Alpha: {alpha_name}", "Beta": f"Beta: {beta_name}"}

        if self_consistency_scores is not None:
            _plot_intra_scores(
                self_consistency_scores,
                label_map,
                run_dir / "semantic_self_consistency.png",
                (
                    "Semantic Self-Consistency"
                    f" | {question_label}"
                    f" | incentive={incentive_label}"
                ),
                cfg.show_plots,
            )
        if (
            cross_agent_public_alignment is not None
            or cross_agent_private_alignment is not None
        ):
            _plot_inter_scores(
                cross_agent_public_alignment,
                cross_agent_private_alignment,
                run_dir / "semantic_cross_agent_alignment.png",
                (
                    "Cross-Agent Semantic Alignment"
                    f" | {question_label}"
                    f" | incentive={incentive_label}"
                ),
                cfg.show_plots,
            )

        if persona_adherence_eval is not None:
            plot_persona_adherence(
                eval_dict=persona_adherence_eval,
                alpha_persona_name=alpha_name,
                beta_persona_name=beta_name,
                save_path=str(run_dir / "persona_adherence.png"),
                show_plot=cfg.show_plots,
            )

        if cfg.enable_public_survey or cfg.enable_private_survey:
            survey_title = (
                f"Survey Responses | {question_label}"
                f" | incentive={incentive_label}"
            )
            if cfg.enable_public_survey:
                plot_survey_responses(
                    responses=public_survey_responses,
                    agents=agents,
                    survey_questions=survey_question_specs,
                    title=f"Public {survey_title}",
                    output_path=run_dir / "public_survey.png",
                )
            if cfg.enable_private_survey:
                plot_survey_responses(
                    responses=private_survey_responses,
                    agents=agents,
                    survey_questions=survey_question_specs,
                    title=f"Private {survey_title}",
                    output_path=run_dir / "private_survey.png",
                )
            if cfg.enable_private_survey and cfg.enable_public_survey:
                plot_survey_distance(
                    public_responses=public_survey_responses,
                    private_responses=private_survey_responses,
                    agents=agents,
                    survey_questions=survey_question_specs,
                    title=f"Public vs Private {survey_title}",
                    output_path=run_dir / "diff_survey.png",
                )

    eval_data: dict[str, Any] = {
        "semantic_similarity": {
            SEMANTIC_METRIC_SELF_CONSISTENCY: self_consistency_scores,
            SEMANTIC_METRIC_CROSS_AGENT_PUBLIC_ALIGNMENT: cross_agent_public_alignment,
            SEMANTIC_METRIC_CROSS_AGENT_PRIVATE_ALIGNMENT: cross_agent_private_alignment,
        },
        "persona_adherence": persona_adherence_eval,
    }

    should_write_eval_data = any(
        [
            bool(cfg.semantic_analysis_metrics),
            bool(cfg.persona_analysis_metrics),
        ]
    )

    if run_dir is not None:
        effective_config = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()}
        if not cfg.reuse_load_dir_for_outputs:
            _save_json(run_dir / "config.json", effective_config)
        if should_write_eval_data:
            _save_json(run_dir / "eval_data.json", eval_data)

    if cfg.indexed_output and run_id is not None and run_dir is not None:
        _append_index_row(
            cfg=cfg,
            run_id=run_id,
            run_dir=run_dir,
            index_csv=_resolve_index_csv(cfg),
        )

    return ExperimentResult(
        agora=agora,
        agents=agents,
        eval_data=eval_data,
        run_dir=run_dir,
        run_id=run_id,
        semantic_analyzer=semantic_analyzer,
        persona_adherence_eval=persona_adherence_eval,
        survey_question_specs=survey_question_specs,
    )


__all__ = [
    "DEFAULT_CATALOG_PATH",
    "DEFAULT_INDEX_CSV",
    "DEFAULT_OUTPUTS_ROOT",
    "DEFAULT_PROMPTS_PATH",
    "SEMANTIC_ANALYSIS_METRICS",
    "SEMANTIC_METRIC_SELF_CONSISTENCY",
    "SEMANTIC_METRIC_CROSS_AGENT_PUBLIC_ALIGNMENT",
    "SEMANTIC_METRIC_CROSS_AGENT_PRIVATE_ALIGNMENT",
    "SEMANTIC_SIMILARITY_METHODS",
    "SEMANTIC_SIMILARITY_METHOD_COSINE",
    "SEMANTIC_SIMILARITY_METHOD_NLI",
    "PERSONA_ANALYSIS_METRICS",
    "ExperimentConfig",
    "ExperimentResult",
    "build_experiment_config",
    "load_experiment_config",
    "run_persona_experiment",
    "_merge_config",
]
