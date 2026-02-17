"""High-level workflows for running Agora debates programmatically or via CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

from .agora import Agora
from .agent import Agent, build_system_prompt
from .llm import LLMClient, OpenRouterClient
from .persistence import load_snapshot, save_snapshot


def extract_instruction(config: dict, key: str) -> Tuple[Optional[str], bool]:
    """Parse an instruction entry that may be absent, a string, or a dict.

    The notebooks historically accepted values like:
    - None (instruction unused)
    - "do something" (instruction string, keep defaults to True)
    - {"instruction": "...", "keep": False}

    This helper normalizes that shape for runtime consumption.
    """

    entry = config.get(key)
    if entry is None:
        return None, True
    if isinstance(entry, str):
        return entry, True
    if isinstance(entry, dict):
        return entry.get("instruction"), bool(entry.get("keep", True))
    raise ValueError(f"Invalid entry for {key}: {entry}")


def extract_survey_instructions(
    config: dict,
) -> Tuple[List[str], Optional[str], Optional[str], bool, bool, bool, bool]:
    """Parse survey instructions from an agent configuration dict."""
    entry = config.get("survey")
    if not isinstance(entry, dict):
        return [], None, None, False, False, False, False

    return (
        entry.get("survey_questions") or [],
        entry.get("survey_public_prompt"),
        entry.get("survey_private_prompt"),
        bool(entry.get("public_survey_keep", False)),
        bool(entry.get("private_survey_keep", False)),
        bool(entry.get("enable_public_survey", True)),
        bool(entry.get("enable_private_survey", True)),
    )


def build_agents_from_configs(
    agent_configs: Sequence[dict], llm_client: LLMClient
) -> List[Agent]:
    """Instantiate ``Agent`` objects from notebook-style configuration dicts."""

    total_agents = len(agent_configs)
    agents: List[Agent] = []
    for cfg in agent_configs:
        system_prompt = build_system_prompt(cfg, total_agents=total_agents)
        private_instr, private_keep = extract_instruction(cfg, "private_response")
        pre_instr, pre_keep = extract_instruction(cfg, "pre_interview")
        post_instr, post_keep = extract_instruction(cfg, "post_interview")

        (
            survey_questions,
            survey_public_prompt,
            survey_private_prompt,
            public_survey_keep,
            private_survey_keep,
            enable_public_survey,
            enable_private_survey,
        ) = extract_survey_instructions(cfg)

        agent = Agent(
            name=cfg["name"],
            model=cfg["model"],
            llm_client=llm_client,
            system_prompt=system_prompt,
            response_instruction=cfg["response_instruction"],
            opening_instruction=cfg.get("opening_instruction"),
            private_response_instruction=private_instr,
            private_response_keep=private_keep,
            pre_interview_instruction=pre_instr,
            pre_interview_keep=pre_keep,
            post_interview_instruction=post_instr,
            post_interview_keep=post_keep,
            survey_questions=survey_questions,
            survey_public_prompt=survey_public_prompt,
            survey_private_prompt=survey_private_prompt,
            enable_public_survey=enable_public_survey,
            enable_private_survey=enable_private_survey,
            public_survey_keep=public_survey_keep,
            private_survey_keep=private_survey_keep,
        )
        agents.append(agent)
    return agents


def run_debate_session(
    agent_configs: Sequence[dict],
    *,
    num_turns: int,
    event_order: Optional[Sequence[str]] = None,
    verbose: bool = False,
    skip_first_agent_first_reflection: bool = False,
    snapshot_path: Optional[Path | str] = None,
    load_snapshot_flag: bool = False,
    save_snapshot_flag: bool = False,
    llm_client: Optional[LLMClient] = None,
    client_factory: Callable[[], LLMClient] = OpenRouterClient,
) -> Tuple[Agora, List[Agent]]:
    """Run an Agora session from configuration dictionaries.

    When ``load_snapshot_flag`` is True, ``snapshot_path`` must exist and the
    Agora is restored from disk instead of constructing new agents. Snapshots
    are saved back to the same path when ``save_snapshot_flag`` is True.
    """

    managed_client: Optional[LLMClient] = None
    if llm_client is None:
        managed_client = client_factory()
        llm_client = managed_client

    try:
        snapshot_file = Path(snapshot_path) if snapshot_path else None
        if load_snapshot_flag and snapshot_file is None:
            raise ValueError("snapshot_path is required when load_snapshot_flag is True")
        if load_snapshot_flag and snapshot_file is not None and not snapshot_file.exists():
            raise FileNotFoundError(f"Snapshot not found at {snapshot_file}")

        # Snapshot loading bypasses config-based agent construction.
        if load_snapshot_flag:
            assert snapshot_file is not None  # guarded by validation above
            agora = load_snapshot(snapshot_file, lambda _state: llm_client)
            agents = list(agora.agents)
        else:
            agents = build_agents_from_configs(agent_configs, llm_client)
            agora = Agora(agents, event_order=event_order)

        # Offline post-processing mode: when resuming from a snapshot with
        # num_turns=0, return the loaded debate without generating new turns.
        if num_turns != 0:
            agora.run(
                num_turns=num_turns,
                verbose=verbose,
                skip_first_agent_first_reflection=skip_first_agent_first_reflection,
            )

        if save_snapshot_flag and snapshot_file:
            save_snapshot(snapshot_file, agora)
        return agora, agents
    finally:
        if managed_client and hasattr(managed_client, "close"):
            managed_client.close()


def format_history_for_agent(agent: Agent) -> str:
    """Return a human-readable string of the turns visible to ``agent``."""

    lines: List[str] = []
    for turn in agent.view_history():
        speaker = turn.metadata.get("speaker_name", turn.speaker_id)
        note = ""
        if (
            turn.role in {
                "reflection",
                "pre_interview",
                "post_interview",
                "public_survey",
                "private_survey",
            }
            and not turn.keep
        ):
            note = " (excluded)"
        if turn.role == "reflection":
            lines.append(
                f"Turn {turn.turn_id:02d} | {speaker} (private){note}: {turn.private_reflection}"
            )
        elif turn.role in {"pre_interview", "post_interview"}:
            label = (
                "pre-interview" if turn.role == "pre_interview" else "post-interview"
            )
            lines.append(
                f"Turn {turn.turn_id:02d} | {speaker} ({label}){note}: {turn.private_reflection}"
            )
        elif turn.role == "public_survey":
            lines.append(
                f"Turn {turn.turn_id:02d} | {speaker} (public survey){note}: {turn.public_speech}"
            )
        elif turn.role == "private_survey":
            lines.append(
                f"Turn {turn.turn_id:02d} | {speaker} (private survey){note}: {turn.private_reflection}"
            )
        else:
            lines.append(f"Turn {turn.turn_id:02d} | {speaker}: {turn.public_speech}")
    return "\n".join(lines)


def print_agent_histories(agents: Iterable[Agent]) -> None:
    """Print each agent's view of the history to stdout."""

    for agent in agents:
        print(f"\n### Full history visible to {agent.name}")
        print(format_history_for_agent(agent))


def load_debate_construction(debate_construction_path: Path | str) -> dict:
    """Load the debate scenario catalog from disk."""

    return json.loads(Path(debate_construction_path).read_text())


DEFAULT_PROMPT_SET = "default"
DEFAULT_PROMPT_PATH = Path(__file__).resolve().parents[2] / "data" / "prompts.json"


class _SafePromptTokens(dict):
    """Allow partial template formatting while preserving unknown placeholders."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _variant_language_tokens(prompts: dict, question_variant: str) -> dict[str, str]:
    language_by_variant = prompts.get("question_variant_language")
    if not isinstance(language_by_variant, dict):
        raise KeyError(
            "Prompt set is missing required object: question_variant_language"
        )

    tokens = language_by_variant.get(question_variant)
    if not isinstance(tokens, dict):
        available = ", ".join(sorted(language_by_variant)) or "<none>"
        raise ValueError(
            f"Unsupported question_variant '{question_variant}' for prompt set language mapping. "
            f"Expected one of: {available}"
        )
    for required in ("interaction_noun", "counterpart_noun"):
        if required not in tokens:
            raise KeyError(
                f"question_variant_language['{question_variant}'] missing required key '{required}'"
            )
    return tokens


def _apply_variant_language(template: Optional[str], tokens: dict[str, str]) -> Optional[str]:
    if template is None:
        return None
    return template.format_map(_SafePromptTokens(tokens))


def _apply_variant_language_many(
    templates: Optional[Sequence[str]], tokens: dict[str, str]
) -> Optional[list[str]]:
    if templates is None:
        return None
    return [item.format_map(_SafePromptTokens(tokens)) for item in templates]


def load_prompt_catalog(prompt_path: Path | str | None = None) -> dict:
    """Load prompt template catalog from disk or the repo default."""

    prompt_resource = (
        Path(prompt_path) if prompt_path is not None else DEFAULT_PROMPT_PATH
    )

    if not prompt_resource.exists():
        raise FileNotFoundError(f"Prompt catalog not found at {prompt_resource}")

    return json.loads(prompt_resource.read_text(encoding="utf-8"))


def load_prompt_templates(
    name: str = DEFAULT_PROMPT_SET,
    *,
    prompt_catalog: Optional[dict] = None,
    prompt_path: Path | str | None = None,
) -> dict:
    """Load a prompt template set from JSON catalog data."""

    if prompt_catalog is None:
        prompt_catalog = load_prompt_catalog(prompt_path)

    prompt_sets = prompt_catalog.get("prompt_sets", prompt_catalog)
    payload = prompt_sets.get(name)
    if payload is None:
        available = ", ".join(sorted(prompt_sets)) or "<none>"
        raise KeyError(
            f"Prompt template '{name}' not found; available sets: {available}"
        )

    required_keys = [
        "question_variant_language",
        "base_prompt",
        "perceived_prompt",
        "debate_arena_prompt",
        "public_instruction",
        "private_instruction",
        "pre_interview_instruction",
        "post_interview_instruction",
        "survey_public_prompt",
        "survey_private_prompt",
    ]
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise KeyError(
            f"Prompt template '{name}' missing required fields: {', '.join(sorted(missing))}"
        )
    if "opening_instruction" not in payload:
        payload = dict(payload)
        payload["opening_instruction"] = payload["public_instruction"]
    return payload


def build_scenario_agent_configs(
    *,
    scenario_id: str,
    catalog: dict,
    alpha_model: str,
    beta_model: str,
    question_variant: str = "controversial",
    side_order: str = "12",
    debate_arena_override: Optional[str] = None,
    base_prompt: Optional[str] = None,
    perceived_prompt: Optional[str] = None,
    debate_arena_prompt: Optional[str] = None,
    public_instruction: Optional[str] = None,
    opening_instruction: Optional[str] = None,
    private_instruction: Optional[str] = None,
    pre_interview_instruction: Optional[str] = None,
    post_interview_instruction: Optional[str] = None,
    survey_public_prompt: Optional[str] = None,
    survey_private_prompt: Optional[str] = None,
    enable_public_survey: bool = True,
    enable_private_survey: bool = True,
    public_survey_keep: bool = False,
    private_survey_keep: bool = False,
    prompt_set: str = DEFAULT_PROMPT_SET,
    private_response_keep: bool = True,
    pre_interview_keep: bool = False,
    post_interview_keep: bool = False,
    survey_questions: Optional[list[str]] = None,
    prompt_templates: Optional[dict] = None,
    prompt_catalog: Optional[dict] = None,
    prompt_path: Path | str | None = None,
) -> List[dict]:
    """Construct agent configs for a scenario-driven debate.

    Args:
        catalog: Debate catalog containing embedded scenarios.
        question_variant: Which version of the question to use - "agreeable" or "controversial".
                          Defaults to "controversial".
        side_order: "12" uses side_1 as Alpha and side_2 as Beta; "21" swaps them.
        debate_arena_override: If provided, use this arena text instead of alpha's persona arena.
                               Pass NEUTRAL_DEBATE_ARENA for a neutral setting.
    """

    prompts = prompt_templates or load_prompt_templates(
        prompt_set, prompt_catalog=prompt_catalog, prompt_path=prompt_path
    )

    base_prompt = base_prompt or prompts["base_prompt"]
    perceived_prompt = perceived_prompt or prompts["perceived_prompt"]
    debate_arena_prompt = debate_arena_prompt or prompts["debate_arena_prompt"]
    public_instruction = public_instruction or prompts["public_instruction"]
    opening_instruction = opening_instruction or prompts["opening_instruction"]
    private_instruction = private_instruction or prompts["private_instruction"]
    pre_interview_instruction = (
        pre_interview_instruction or prompts["pre_interview_instruction"]
    )
    post_interview_instruction = (
        post_interview_instruction or prompts["post_interview_instruction"]
    )
    survey_public_prompt = survey_public_prompt or prompts["survey_public_prompt"]
    survey_private_prompt = survey_private_prompt or prompts["survey_private_prompt"]

    scenarios = catalog.get("scenarios", [])
    scenario = next((item for item in scenarios if item.get("id") == scenario_id), None)
    if scenario is None:
        raise KeyError(f"Unknown scenario id: {scenario_id}")
    if side_order not in {"12", "21"}:
        raise ValueError(f"Invalid side order: {side_order}")

    question_entry = scenario.get("question", {})
    if question_variant not in question_entry:
        available = [k for k in question_entry if k not in ("id", "topic")]
        raise KeyError(
            f"Question variant '{question_variant}' not found for scenario '{scenario_id}'; "
            f"available variants: {', '.join(sorted(available))}"
        )
    question_text = question_entry[question_variant]
    variant_tokens = _variant_language_tokens(prompts, question_variant)

    public_instruction = _apply_variant_language(public_instruction, variant_tokens)
    opening_instruction = _apply_variant_language(opening_instruction, variant_tokens)
    private_instruction = _apply_variant_language(private_instruction, variant_tokens)
    pre_interview_instruction = _apply_variant_language(
        pre_interview_instruction, variant_tokens
    )
    post_interview_instruction = _apply_variant_language(
        post_interview_instruction, variant_tokens
    )
    survey_public_prompt = _apply_variant_language(
        survey_public_prompt, variant_tokens
    )
    survey_private_prompt = _apply_variant_language(
        survey_private_prompt, variant_tokens
    )
    survey_questions = _apply_variant_language_many(survey_questions, variant_tokens)

    side_1 = scenario.get("side_1", {})
    side_2 = scenario.get("side_2", {})
    if not side_1 or not side_2:
        raise KeyError(f"Scenario '{scenario_id}' missing side definitions")

    if side_order == "12":
        alpha_persona = side_1
        beta_persona = side_2
    else:
        alpha_persona = side_2
        beta_persona = side_1

    # Determine debate arena: use override if provided, otherwise alpha's arena
    if debate_arena_override is not None:
        arena_text = debate_arena_override
    else:
        arena_text = alpha_persona.get("debate_arena", "")

    arena_context = ""
    if arena_text:
        arena_context = debate_arena_prompt.format(
            debate_arena=arena_text, **variant_tokens
        )

    alpha_self_role = base_prompt.format(
        speaker_id="A",
        question=question_text,
        persona=alpha_persona["actual_persona"],
        **variant_tokens,
    )
    beta_self_role = base_prompt.format(
        speaker_id="B",
        question=question_text,
        persona=beta_persona["actual_persona"],
        **variant_tokens,
    )

    # Append debate arena context to both agents' self_role
    if arena_context:
        alpha_self_role = alpha_self_role + arena_context
        beta_self_role = beta_self_role + arena_context

    alpha_perceives_beta = perceived_prompt.format(
        perceived_persona=beta_persona["perceived_persona"], **variant_tokens
    )
    beta_perceives_alpha = perceived_prompt.format(
        perceived_persona=alpha_persona["perceived_persona"], **variant_tokens
    )

    return [
        {
            "name": "Alpha",
            "model": alpha_model,
            "self_role": alpha_self_role,
            "perceived_nonself_roles": [{"name": "Beta", "role": alpha_perceives_beta}],
            "response_instruction": public_instruction,
            "opening_instruction": opening_instruction,
            "private_response": {
                "instruction": private_instruction,
                "keep": private_response_keep,
            },
            "pre_interview": {
                "instruction": pre_interview_instruction,
                "keep": pre_interview_keep,
            },
            "post_interview": {
                "instruction": post_interview_instruction,
                "keep": post_interview_keep,
            },
            "survey": {
                "survey_questions": survey_questions,
                "survey_public_prompt": survey_public_prompt,
                "survey_private_prompt": survey_private_prompt,
                "enable_public_survey": enable_public_survey,
                "enable_private_survey": enable_private_survey,
                "public_survey_keep": public_survey_keep,
                "private_survey_keep": private_survey_keep,
            },
        },
        {
            "name": "Beta",
            "model": beta_model,
            "self_role": beta_self_role,
            "perceived_nonself_roles": [
                {"name": "Alpha", "role": beta_perceives_alpha}
            ],
            "response_instruction": public_instruction,
            "opening_instruction": opening_instruction,
            "private_response": {
                "instruction": private_instruction,
                "keep": private_response_keep,
            },
            "pre_interview": {
                "instruction": pre_interview_instruction,
                "keep": pre_interview_keep,
            },
            "post_interview": {
                "instruction": post_interview_instruction,
                "keep": post_interview_keep,
            },
            "survey": {
                "survey_questions": survey_questions,
                "survey_public_prompt": survey_public_prompt,
                "survey_private_prompt": survey_private_prompt,
                "enable_public_survey": enable_public_survey,
                "enable_private_survey": enable_private_survey,
                "public_survey_keep": public_survey_keep,
                "private_survey_keep": private_survey_keep,
            },
        },
    ]

__all__ = [
    "build_agents_from_configs",
    "build_scenario_agent_configs",
    "DEFAULT_PROMPT_PATH",
    "DEFAULT_PROMPT_SET",
    "load_prompt_catalog",
    "extract_instruction",
    "format_history_for_agent",
    "load_prompt_templates",
    "load_debate_construction",
    "print_agent_histories",
    "run_debate_session",
]
