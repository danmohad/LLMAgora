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
    load_prompt_catalog,
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
        alpha_id = payload.get("alpha_persona_id")
        beta_id = payload.get("beta_persona_id")
        question_id = payload.get("question_id")
        if not all([alpha_id, beta_id, question_id]):
            raise ValueError(
                "Config must include an 'agent_configs' array or persona identifiers alpha_persona_id, beta_persona_id, question_id"
            )

        personas = load_persona_catalog(payload.get("personas_path", "data/personas.json"))
        questions = load_question_catalog(payload.get("questions_path", "data/questions.json"))
        prompt_catalog = load_prompt_catalog(payload.get("prompts_path", "data/prompts.json"))
        alpha_model = payload.get("alpha_model", "openai/gpt-4o-mini")
        beta_model = payload.get("beta_model", "anthropic/claude-3-haiku")
        agent_configs = build_persona_agent_configs(
            alpha_persona_id=alpha_id,
            beta_persona_id=beta_id,
            question_id=question_id,
            personas=personas,
            questions=questions,
            alpha_model=alpha_model,
            beta_model=beta_model,
            prompt_set=payload.get("prompt_set", DEFAULT_PROMPT_SET),
            prompt_catalog=prompt_catalog,
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
    personas = load_persona_catalog(args.personas)
    questions = load_question_catalog(args.questions)
    prompt_catalog = load_prompt_catalog(args.prompts)
    agent_configs = build_persona_agent_configs(
        alpha_persona_id=args.alpha_id,
        beta_persona_id=args.beta_id,
        question_id=args.question_id,
        personas=personas,
        questions=questions,
        alpha_model=args.alpha_model,
        beta_model=args.beta_model,
        prompt_set=args.prompt_set,
        prompt_catalog=prompt_catalog,
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
        "--prompts", type=Path, default=Path("data/prompts.json"), help="Path to prompt catalog JSON"
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
    persona_cmd.set_defaults(func=_run_persona)

    return parser


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
