"""Command line interface for running Agora debates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

from .workflows import (
    DEFAULT_PROMPT_PATH,
    DEFAULT_PROMPT_SET,
    build_scenario_agent_configs,
    load_debate_construction,
    load_prompt_catalog,
    print_agent_histories,
    run_debate_session,
)


def _load_agent_payload(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Agent config file must contain a JSON object")
    return payload


def _run_from_config(args: argparse.Namespace) -> None:
    payload = _load_agent_payload(args.config)
    agent_configs = payload.get("agent_configs", payload.get("agents")) or payload.get("agents", [])

    if not agent_configs:
        scenario_id = payload.get("scenario_id")
        if not scenario_id:
            raise ValueError(
                "Config must include an 'agent_configs' array or a scenario_id"
            )

        catalog_path = payload.get("catalog_path", "data/debate_construction.json")
        catalog = load_debate_construction(catalog_path)
        prompt_catalog = load_prompt_catalog(payload.get("prompts_path", DEFAULT_PROMPT_PATH))
        alpha_model = payload.get("alpha_model", "openai/gpt-4o-mini")
        beta_model = payload.get("beta_model", "anthropic/claude-3-haiku")
        agent_configs = build_scenario_agent_configs(
            scenario_id=scenario_id,
            catalog=catalog,
            alpha_model=alpha_model,
            beta_model=beta_model,
            question_variant=payload.get("question_variant", "controversial"),
            side_order=payload.get("side_order", "12"),
            prompt_set=payload.get("prompt_set", DEFAULT_PROMPT_SET),
            prompt_catalog=prompt_catalog,
            private_response_keep=payload.get("private_response_keep", True),
            pre_interview_keep=payload.get("pre_interview_keep", False),
            post_interview_keep=payload.get("post_interview_keep", False),
        )

    turns = args.turns or payload.get("turns_per_agent")
    if not turns:
        raise ValueError("Config must include 'turns_per_agent' or --turns")

    skip_first = args.skip_first_reflection or payload.get("skip_first_agent_first_reflection", False)
    snapshot = args.snapshot or payload.get("snapshot_path")
    load_snapshot_flag = args.load_snapshot or payload.get("load_snapshot_flag", False)
    save_snapshot_flag = args.save_snapshot or payload.get("save_snapshot_flag", False)

    agora, agents = run_debate_session(
        agent_configs,
        turns_per_agent=turns,
        verbose=args.verbose or payload.get("verbose", False),
        skip_first_agent_first_reflection=skip_first,
        snapshot_path=snapshot,
        load_snapshot_flag=load_snapshot_flag,
        save_snapshot_flag=save_snapshot_flag,
    )
    print_agent_histories(agents)


def _run_persona(args: argparse.Namespace) -> None:
    catalog = load_debate_construction(args.catalog)
    prompt_catalog = load_prompt_catalog(args.prompts)
    agent_configs = build_scenario_agent_configs(
        scenario_id=args.scenario_id,
        catalog=catalog,
        alpha_model=args.alpha_model,
        beta_model=args.beta_model,
        question_variant=args.question_variant,
        side_order=args.side_order,
        prompt_set=args.prompt_set,
        prompt_catalog=prompt_catalog,
        private_response_keep=args.keep_private_response,
        pre_interview_keep=args.keep_pre_interview,
        post_interview_keep=args.keep_post_interview,
    )

    agora, agents = run_debate_session(
        agent_configs,
        turns_per_agent=args.turns,
        verbose=args.verbose,
        skip_first_agent_first_reflection=args.skip_first_reflection,
        snapshot_path=args.snapshot,
        load_snapshot_flag=args.load_snapshot,
        save_snapshot_flag=args.save_snapshot,
    )
    print_agent_histories(agents)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LLM Agora debates from the command line.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_cmd = subparsers.add_parser("run", help="Run a debate from a JSON config file.")
    run_cmd.add_argument("--config", type=Path, required=True, help="Path to agent configuration JSON file")
    run_cmd.add_argument("--turns", type=int, help="Override turns per agent defined in the config")
    run_cmd.add_argument("--snapshot", type=Path, help="Path to snapshot file")
    run_cmd.add_argument("--load-snapshot", action="store_true", help="Load debate state from snapshot before running")
    run_cmd.add_argument("--save-snapshot", action="store_true", help="Persist snapshot after the run finishes")
    run_cmd.add_argument(
        "--skip-first-reflection",
        action="store_true",
        help="Skip the first reflection for the first agent (useful when pre-interviews cover it)",
    )
    run_cmd.add_argument("--verbose", action="store_true", help="Print turn-by-turn output while running")
    run_cmd.set_defaults(func=_run_from_config)

    persona_cmd = subparsers.add_parser(
        "persona", help="Run a persona-driven debate using the bundled datasets."
    )
    persona_cmd.add_argument("--scenario-id", required=True, help="Scenario ID from the debate catalog")
    persona_cmd.add_argument("--alpha-model", default="openai/gpt-4o-mini", help="Model identifier for Alpha")
    persona_cmd.add_argument("--beta-model", default="anthropic/claude-3-haiku", help="Model identifier for Beta")
    persona_cmd.add_argument(
        "--question-variant",
        default="controversial",
        choices=["agreeable", "controversial"],
        help="Question variant to use for the scenario",
    )
    persona_cmd.add_argument(
        "--side-order",
        default="12",
        choices=["12", "21"],
        help="Which scenario side maps to Alpha/Beta (12=side_1->Alpha, 21=side_2->Alpha)",
    )
    persona_cmd.add_argument("--turns", type=int, default=5, help="Number of turns each agent should take")
    persona_cmd.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/debate_construction.json"),
        help="Path to the combined debate catalog JSON",
    )
    persona_cmd.add_argument(
        "--prompts", type=Path, default=DEFAULT_PROMPT_PATH, help="Path to prompt catalog JSON"
    )
    persona_cmd.add_argument(
        "--prompt-set",
        default=DEFAULT_PROMPT_SET,
        help="Name of the prompt template set to load (key within the prompt catalog)",
    )
    persona_cmd.add_argument("--snapshot", type=Path, help="Path to snapshot file")
    persona_cmd.add_argument("--load-snapshot", action="store_true", help="Load debate state from snapshot before running")
    persona_cmd.add_argument("--save-snapshot", action="store_true", help="Persist snapshot after the run finishes")
    persona_cmd.add_argument(
        "--skip-first-reflection",
        action="store_true",
        default=True,
        help="Skip the first reflection for the first agent (useful when pre-interviews cover it)",
    )
    persona_cmd.add_argument(
        "--no-skip-first-reflection",
        action="store_false",
        dest="skip_first_reflection",
        help="Allow the first reflection to run",
    )
    persona_cmd.add_argument("--verbose", action="store_true", help="Print turn-by-turn output while running")
    persona_cmd.add_argument(
        "--keep-private-response",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to retain private reflections in local history",
    )
    persona_cmd.add_argument(
        "--keep-pre-interview",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to retain pre-interview notes in local history",
    )
    persona_cmd.add_argument(
        "--keep-post-interview",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to retain post-interview notes in local history",
    )
    persona_cmd.set_defaults(func=_run_persona)

    return parser


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
