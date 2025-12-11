"""Command line interface for running Agora debates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

from .workflows import (
    DEFAULT_PROMPT_SET,
    build_persona_agent_configs,
    load_persona_catalog,
    load_question_catalog,
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
        agent_configs = payload.get("agent_configs")
    if not agent_configs:
        raise ValueError("Config must include an 'agent_configs' array or top-level agent objects")

    turns = args.turns or payload.get("turns_per_agent")
    if not turns:
        raise ValueError("Config must include 'turns_per_agent' or --turns")

    agora, agents = run_debate_session(
        agent_configs,
        turns_per_agent=turns,
        verbose=args.verbose,
        skip_first_agent_first_reflection=args.skip_first_reflection,
        snapshot_path=args.snapshot,
        load_snapshot_flag=args.load_snapshot,
        save_snapshot_flag=args.save_snapshot,
    )
    print_agent_histories(agents)


def _run_persona(args: argparse.Namespace) -> None:
    personas = load_persona_catalog(args.personas)
    questions = load_question_catalog(args.questions)
    agent_configs = build_persona_agent_configs(
        alpha_persona_id=args.alpha_id,
        beta_persona_id=args.beta_id,
        question_id=args.question_id,
        personas=personas,
        questions=questions,
        alpha_model=args.alpha_model,
        beta_model=args.beta_model,
        prompt_set=args.prompt_set,
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
    persona_cmd.add_argument("--alpha-id", required=True, help="Persona ID for the Alpha speaker")
    persona_cmd.add_argument("--beta-id", required=True, help="Persona ID for the Beta speaker")
    persona_cmd.add_argument("--question-id", required=True, help="Question ID from the dataset")
    persona_cmd.add_argument("--alpha-model", default="openai/gpt-4o-mini", help="Model identifier for Alpha")
    persona_cmd.add_argument("--beta-model", default="anthropic/claude-3-haiku", help="Model identifier for Beta")
    persona_cmd.add_argument("--turns", type=int, default=5, help="Number of turns each agent should take")
    persona_cmd.add_argument(
        "--personas", type=Path, default=Path("data/personas.json"), help="Path to persona catalog JSON"
    )
    persona_cmd.add_argument(
        "--questions", type=Path, default=Path("data/questions.json"), help="Path to questions catalog JSON"
    )
    persona_cmd.add_argument(
        "--prompt-set",
        default=DEFAULT_PROMPT_SET,
        help="Name of the prompt template set to load (matches YAML filename in agora/prompts)",
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
    persona_cmd.set_defaults(func=_run_persona)

    return parser


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
