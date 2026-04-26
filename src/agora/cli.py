"""Command line interface for running Agora persona experiments."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .experiment import (
    DEFAULT_CATALOG_PATH,
    DEFAULT_INDEX_CSV,
    DEFAULT_OUTPUTS_ROOT,
    DEFAULT_PROMPTS_PATH,
    PERSONA_ANALYSIS_METRICS,
    SEMANTIC_ANALYSIS_METRICS,
    SEMANTIC_SIMILARITY_METHODS,
    _merge_config,
    build_experiment_config,
    load_experiment_config,
    run_persona_experiment,
)
from .sweep import (
    RUN_SELECTION_MODES,
    generate_sweep,
    load_sweep_config,
    run_sweep,
)
from .workflows import print_agent_histories


def _run(args: argparse.Namespace) -> None:
    base: dict[str, Any] = {}
    if args.config:
        base = asdict(load_experiment_config(args.config))

    raw_overrides = {
        "scenario_id": args.scenario_id,
        "incentive_type": args.incentive_type,
        "prompt_set": args.prompt_set,
        "model": args.model,
        "num_turns": args.num_turns,
        "subturn_event_order": args.subturn_event_order,
        "verbose": args.verbose,
        "keep_private_reflection": args.keep_private_reflection,
        "keep_pre_interview": args.keep_pre_interview,
        "keep_post_interview": args.keep_post_interview,
        "keep_public_survey": args.keep_public_survey,
        "keep_private_survey": args.keep_private_survey,
        "semantic_analysis_metrics": args.semantic_analysis_metrics,
        "semantic_similarity_method": args.semantic_similarity_method,
        "semantic_similarity_model": args.semantic_similarity_model,
        "semantic_similarity_device": args.semantic_similarity_device,
        "persona_analysis_metrics": args.persona_analysis_metrics,
        "persona_scoring_model": args.persona_scoring_model,
        "persona_scoring_verbose": args.persona_scoring_verbose,
        "persona_score_samples": args.persona_score_samples,
        "save_plots": args.save_plots,
        "show_plots": args.show_plots,
        "load_snapshot": args.load_snapshot,
        "load_dir": args.load_dir,
        "save_snapshot": args.save_snapshot,
        "outputs_root": args.outputs_root,
        "run_name": args.run_name,
        "indexed_output": args.indexed_output,
        "index_csv": args.index_csv,
        "catalog_path": args.catalog_path,
        "prompts_path": args.prompts_path,
    }
    # Only pass explicitly provided CLI flags to config merging.
    # This preserves explicit clears when a flag semantically maps to None.
    overrides = {key: value for key, value in raw_overrides.items() if value is not None}
    if args.incentive_direction is not None:
        overrides["incentive_direction"] = (
            None if args.incentive_direction == "none" else args.incentive_direction
        )

    cfg = _merge_config(base, overrides) if base else build_experiment_config(overrides)
    result = run_persona_experiment(
        cfg,
        emit_progress_markers=args.emit_progress_markers,
    )

    if result.run_dir is not None:
        print(f"Run directory: {result.run_dir}")
    else:
        print("Run directory: <none> (outputs disabled by config)")
    if result.run_id:
        print(f"Run ID: {result.run_id}")

    if args.print_histories:
        print_agent_histories(result.agents)


def _add_bool(parser: argparse.ArgumentParser, name: str, help_text: str) -> None:
    parser.add_argument(
        f"--{name}",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=help_text,
    )


def _sweep_generate(args: argparse.Namespace) -> None:
    generate_sweep(args.config, force=args.force)


def _infer_sweep_root_from_single_jsonc(search_root: Path | None = None) -> Path:
    root = Path.cwd() if search_root is None else search_root
    jsonc_paths = sorted(
        path
        for path in root.rglob("*.jsonc")
        if path.is_file() and path.name != "master_config.jsonc"
    )
    if len(jsonc_paths) != 1:
        raise ValueError(
            "--root must be specified when there is not exactly one .jsonc sweep config in the current working tree"
        )
    master, _ = load_sweep_config(jsonc_paths[0])
    return Path(master["sweep_root"])


def _sweep_run(args: argparse.Namespace) -> None:
    root = args.root
    if root is None:
        root = _infer_sweep_root_from_single_jsonc()
    exit_code = run_sweep(
        root,
        max_parallel_jobs=args.max_parallel_jobs,
        mode=args.mode,
        case_ids=args.cases,
        stop_on_error=args.stop_on_error,
        persistent=args.persistent,
    )
    if exit_code:
        raise SystemExit(exit_code)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LLM Agora experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_cmd = subparsers.add_parser("run", help="Run one experiment from config and/or CLI flags.")
    run_cmd.add_argument("--config", type=Path, help="Path to JSON config file (for example: data/config_example.json)")
    run_cmd.add_argument("--scenario-id", help="Scenario ID from debate catalog")
    run_cmd.add_argument("--incentive-direction", choices=["positive", "negative", "none"])
    run_cmd.add_argument("--incentive-type", choices=["historical", "future"])
    run_cmd.add_argument("--prompt-set")
    run_cmd.add_argument("--model")
    run_cmd.add_argument("--num-turns", type=int)
    run_cmd.add_argument(
        "--subturn-event-order",
        nargs="+",
        choices=["public_utterance", "private_utterance", "public_survey", "private_survey"],
        help=(
            "Ordered events inside each sub-turn. Always include public_utterance. "
            "Include private_utterance/public_survey/private_survey to enable those features."
        ),
    )

    _add_bool(run_cmd, "verbose", "Print turn-by-turn output")
    run_cmd.add_argument(
        "--emit-progress-markers",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )

    _add_bool(run_cmd, "keep-private-reflection", "Keep private reflections in local history")

    _add_bool(run_cmd, "keep-pre-interview", "Keep pre-interview notes in local history")

    _add_bool(run_cmd, "keep-post-interview", "Keep post-interview notes in local history")

    _add_bool(run_cmd, "keep-public-survey", "Reserved survey retention flag (does not modify public_speech)")
    _add_bool(run_cmd, "keep-private-survey", "Reserved survey retention flag for private survey responses")

    run_cmd.add_argument(
        "--semantic-analysis-metrics",
        nargs="*",
        choices=list(SEMANTIC_ANALYSIS_METRICS),
        help=(
            "Select semantic similarity metrics. "
            f"Choices: {', '.join(SEMANTIC_ANALYSIS_METRICS)}. "
            "Pass with no values to clear metrics from --config."
        ),
    )
    run_cmd.add_argument(
        "--semantic-similarity-method",
        choices=list(SEMANTIC_SIMILARITY_METHODS),
        help="Semantic backend: cosine embeddings or NLI entailment scoring.",
    )
    run_cmd.add_argument(
        "--semantic-similarity-model",
        help=(
            "Override semantic model name. "
            "Defaults: all-mpnet-base-v2 (cosine), dleemiller/finecat-nli-l (nli)."
        ),
    )
    run_cmd.add_argument(
        "--semantic-similarity-device",
        choices=["cpu", "mps"],
        help="Device for semantic model inference.",
    )
    run_cmd.add_argument(
        "--persona-analysis-metrics",
        nargs="*",
        choices=list(PERSONA_ANALYSIS_METRICS),
        help=(
            "Select persona adherence metrics. "
            f"Choices: {', '.join(PERSONA_ANALYSIS_METRICS)}. "
            "Pass with no values to clear metrics from --config."
        ),
    )
    run_cmd.add_argument("--persona-scoring-model")
    _add_bool(run_cmd, "persona-scoring-verbose", "Verbose persona adherence scoring progress")
    run_cmd.add_argument("--persona-score-samples", type=int)

    _add_bool(run_cmd, "save-plots", "Save plots for enabled analyses")
    _add_bool(run_cmd, "show-plots", "Display plots while running")

    _add_bool(run_cmd, "load-snapshot", "Load existing snapshot from --load-dir")
    run_cmd.add_argument("--load-dir", type=Path, default=None, help="Directory containing debate_snapshot.json to load")
    _add_bool(run_cmd, "save-snapshot", "Save snapshot (to load dir when loading, otherwise to current run dir)")

    run_cmd.add_argument("--outputs-root", type=Path, default=None, help=f"Root outputs directory (default: {DEFAULT_OUTPUTS_ROOT})")
    run_cmd.add_argument("--run-name", help="Explicit output directory name (human-readable mode)")
    _add_bool(run_cmd, "indexed-output", "Use short unique run IDs and write index CSV")
    run_cmd.add_argument(
        "--index-csv",
        type=Path,
        default=None,
        help=(
            "Index CSV path for indexed output "
            f"(default when omitted in indexed mode: {DEFAULT_INDEX_CSV})"
        ),
    )

    run_cmd.add_argument("--catalog-path", type=Path, default=None, help=f"Scenarios catalog JSON (default: {DEFAULT_CATALOG_PATH})")
    run_cmd.add_argument("--prompts-path", type=Path, default=None, help=f"Prompt catalog JSON (default: {DEFAULT_PROMPTS_PATH})")

    _add_bool(run_cmd, "print-histories", "Print agent-visible histories after the run")

    run_cmd.set_defaults(func=_run)

    sweep_cmd = subparsers.add_parser(
        "sweep",
        help="Generate and run parameter sweeps.",
    )
    sweep_subparsers = sweep_cmd.add_subparsers(dest="sweep_command", required=True)

    sweep_generate_cmd = sweep_subparsers.add_parser(
        "generate",
        help="Expand a master sweep config into per-case run directories.",
    )
    sweep_generate_cmd.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to JSONC master sweep config.",
    )
    _add_bool(
        sweep_generate_cmd,
        "force",
        "Replace an existing sweep root before generation.",
    )
    sweep_generate_cmd.set_defaults(func=_sweep_generate)

    sweep_run_cmd = sweep_subparsers.add_parser(
        "run",
        help="Execute generated sweep cases and render the live dashboard in this terminal.",
    )
    sweep_run_cmd.add_argument(
        "--root",
        type=Path,
        default=None,
        help=(
            "Generated sweep root containing manifest.json. "
            "When omitted, inferred from the only .jsonc sweep config in the current working tree."
        ),
    )
    sweep_run_cmd.add_argument(
        "--max-parallel-jobs",
        type=int,
        default=None,
        help="Override manifest parallelism for this invocation.",
    )
    sweep_run_cmd.add_argument(
        "--mode",
        choices=list(RUN_SELECTION_MODES),
        default="resume",
        help="Case selection mode.",
    )
    sweep_run_cmd.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="Optional explicit case IDs to run.",
    )
    _add_bool(
        sweep_run_cmd,
        "stop-on-error",
        "Stop scheduling new cases after the first failure.",
    )
    sweep_run_cmd.add_argument(
        "--persistent",
        action="store_true",
        default=False,
        help=(
            "Retry failed cases until every selected case succeeds. "
            "This overrides manifest stop-on-error defaults and cannot be combined "
            "with --stop-on-error."
        ),
    )
    sweep_run_cmd.set_defaults(func=_sweep_run)

    return parser


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
