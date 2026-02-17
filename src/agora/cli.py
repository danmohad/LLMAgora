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
    _merge_config,
    build_experiment_config,
    load_experiment_config,
    run_persona_experiment,
)
from .workflows import print_agent_histories


def _run(args: argparse.Namespace) -> None:
    base: dict[str, Any] = {}
    if args.config:
        base = asdict(load_experiment_config(args.config))

    overrides = {
        "scenario_id": args.scenario_id,
        "question_variant": args.question_variant,
        "side_order": args.side_order,
        "prompt_set": args.prompt_set,
        "alpha_model": args.alpha_model,
        "beta_model": args.beta_model,
        "num_turns": args.num_turns,
        "subturn_event_order": args.subturn_event_order,
        "verbose": args.verbose,
        "use_neutral_arena": args.use_neutral_arena,
        "enable_private_reflection": args.enable_private_reflection,
        "keep_private_reflection": args.keep_private_reflection,
        "enable_pre_interview": args.enable_pre_interview,
        "keep_pre_interview": args.keep_pre_interview,
        "enable_post_interview": args.enable_post_interview,
        "keep_post_interview": args.keep_post_interview,
        "enable_public_survey": args.enable_public_survey,
        "enable_private_survey": args.enable_private_survey,
        "keep_public_survey": args.keep_public_survey,
        "keep_private_survey": args.keep_private_survey,
        "semantic_analysis_metrics": args.semantic_analysis_metrics,
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

    cfg = _merge_config(base, overrides) if base else build_experiment_config(overrides)
    result = run_persona_experiment(cfg)

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LLM Agora experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_cmd = subparsers.add_parser("run", help="Run one experiment from config and/or CLI flags.")
    run_cmd.add_argument("--config", type=Path, help="Path to JSON config file (for example: data/example.json)")
    run_cmd.add_argument("--scenario-id", help="Scenario ID from debate catalog")
    run_cmd.add_argument("--question-variant", choices=["agreeable", "controversial"])
    run_cmd.add_argument("--side-order", choices=["12", "21"])
    run_cmd.add_argument("--prompt-set")
    run_cmd.add_argument("--alpha-model")
    run_cmd.add_argument("--beta-model")
    run_cmd.add_argument("--num-turns", type=int)
    run_cmd.add_argument(
        "--subturn-event-order",
        nargs="+",
        choices=["public_utterance", "private_utterance", "public_survey", "private_survey"],
        help=(
            "Ordered events inside each sub-turn; must match enabled events exactly, "
            "for example: public_utterance private_utterance public_survey private_survey"
        ),
    )

    _add_bool(run_cmd, "verbose", "Print turn-by-turn output")
    _add_bool(run_cmd, "use-neutral-arena", "Use neutral arena prompt instead of alpha persona arena")

    _add_bool(run_cmd, "enable-private-reflection", "Enable private reflections")
    _add_bool(run_cmd, "keep-private-reflection", "Keep private reflections in local history")

    _add_bool(run_cmd, "enable-pre-interview", "Enable pre-interview stage")
    _add_bool(run_cmd, "keep-pre-interview", "Keep pre-interview notes in local history")

    _add_bool(run_cmd, "enable-post-interview", "Enable post-interview stage")
    _add_bool(run_cmd, "keep-post-interview", "Keep post-interview notes in local history")

    _add_bool(run_cmd, "enable-public-survey", "Enable public survey rounds")
    _add_bool(run_cmd, "enable-private-survey", "Enable private survey rounds")
    _add_bool(run_cmd, "keep-public-survey", "Reserved survey retention flag (does not modify public_speech)")
    _add_bool(run_cmd, "keep-private-survey", "Reserved survey retention flag for private survey responses")

    run_cmd.add_argument(
        "--semantic-analysis-metrics",
        nargs="+",
        choices=list(SEMANTIC_ANALYSIS_METRICS),
        help=(
            "Select semantic similarity metrics. "
            f"Choices: {', '.join(SEMANTIC_ANALYSIS_METRICS)}"
        ),
    )
    run_cmd.add_argument(
        "--persona-analysis-metrics",
        nargs="+",
        choices=list(PERSONA_ANALYSIS_METRICS),
        help=(
            "Select persona adherence metrics. "
            f"Choices: {', '.join(PERSONA_ANALYSIS_METRICS)}"
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
    return parser


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
