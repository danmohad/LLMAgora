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
) -> Tuple[
    List[str],
    dict[str, str],
    Optional[str],
    Optional[str],
    bool,
    bool,
    bool,
    bool,
]:
    """Parse survey instructions from an agent configuration dict."""
    entry = config.get("survey")
    if not isinstance(entry, dict):
        return [], {}, None, None, False, False, False, False

    return (
        entry.get("survey_questions") or [],
        entry.get("survey_question_groups") or {},
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
            survey_question_groups,
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
            survey_question_groups=survey_question_groups,
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
    emit_progress_markers: bool = False,
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
                emit_progress_markers=emit_progress_markers,
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
ALLOWED_INCENTIVE_DIRECTIONS = {"positive", "negative"}
ALLOWED_INCENTIVE_TYPES = {"historical", "future"}


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
        "base_prompt",
        "perceived_prompt",
        "decision_format",
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
    if "incentive_prompt" not in payload:
        payload = dict(payload)
        payload["incentive_prompt"] = "\n\n# Incentive context:\n{incentive}"
    return payload


def _decision_format_for_scenario(
    *,
    scenario_id: str,
    scenario: dict,
    template: str,
) -> str:
    labels = scenario.get("decision_labels")
    if not isinstance(labels, list) or len(labels) != 2:
        raise KeyError(
            f"Scenario '{scenario_id}' must define exactly two decision_labels"
        )

    normalized_labels = []
    for label in labels:
        if not isinstance(label, str) or not label.strip():
            raise KeyError(
                f"Scenario '{scenario_id}' decision_labels must contain two non-empty strings"
            )
        normalized_labels.append(label.strip())

    first_label, second_label = normalized_labels
    return template.format(
        decision_label_1=first_label,
        decision_label_2=second_label,
    )


def _incentive_text_for_side(
    *,
    scenario_id: str,
    scenario: dict,
    side_label: str,
    incentive_direction: Optional[str],
    incentive_type: str,
) -> Optional[str]:
    if incentive_direction is None:
        return None
    if incentive_direction not in ALLOWED_INCENTIVE_DIRECTIONS:
        raise ValueError(
            "incentive_direction must be one of: positive, negative, or None"
        )
    if incentive_type not in ALLOWED_INCENTIVE_TYPES:
        raise ValueError("incentive_type must be one of: historical, future")

    modules = scenario.get("incentive_modules")
    if not isinstance(modules, dict):
        raise KeyError(
            f"Scenario '{scenario_id}' missing required object: incentive_modules"
        )
    direction_block = modules.get(incentive_direction)
    if not isinstance(direction_block, dict):
        raise KeyError(
            f"Scenario '{scenario_id}' missing incentive module '{incentive_direction}'"
        )
    type_block = direction_block.get(incentive_type)
    if not isinstance(type_block, dict):
        raise KeyError(
            f"Scenario '{scenario_id}' incentive '{incentive_direction}' missing type '{incentive_type}'"
        )
    views = type_block.get("views")
    if not isinstance(views, dict):
        raise KeyError(
            f"Scenario '{scenario_id}' incentive '{incentive_direction}.{incentive_type}' missing views"
        )
    text = views.get(side_label)
    if not isinstance(text, str) or not text.strip():
        raise KeyError(
            f"Scenario '{scenario_id}' incentive '{incentive_direction}.{incentive_type}' missing view for side '{side_label}'"
        )
    return text


def build_scenario_agent_configs(
    *,
    scenario_id: str,
    catalog: dict,
    model: str,
    incentive_direction: Optional[str] = None,
    incentive_type: str = "historical",
    base_prompt: Optional[str] = None,
    perceived_prompt: Optional[str] = None,
    incentive_prompt: Optional[str] = None,
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
    survey_question_groups: Optional[dict[str, str]] = None,
    prompt_templates: Optional[dict] = None,
    prompt_catalog: Optional[dict] = None,
    prompt_path: Path | str | None = None,
) -> List[dict]:
    """Construct agent configs for a scenario-driven debate.

    Args:
        catalog: Debate catalog containing embedded scenarios.
        incentive_direction: Optional incentive module key ('positive' or 'negative').
        incentive_type: Incentive subtype ('historical' or 'future').
    """

    prompts = prompt_templates or load_prompt_templates(
        prompt_set, prompt_catalog=prompt_catalog, prompt_path=prompt_path
    )

    base_prompt = base_prompt or prompts["base_prompt"]
    perceived_prompt = perceived_prompt or prompts["perceived_prompt"]
    incentive_prompt = incentive_prompt or prompts["incentive_prompt"]
    decision_format_template = prompts["decision_format"]
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
    scenario = next(
        (item for item in scenarios if item.get("scenario_id") == scenario_id),
        None,
    )
    if scenario is None:
        raise KeyError(f"Unknown scenario id: {scenario_id}")

    question = scenario.get("question")
    if not isinstance(question, dict):
        raise KeyError(f"Scenario '{scenario_id}' missing required object: question")
    question_text = question.get("prompt")
    if not isinstance(question_text, str) or not question_text.strip():
        raise KeyError(f"Scenario '{scenario_id}' missing question.prompt")

    sides = scenario.get("sides")
    if not isinstance(sides, dict) or len(sides) != 2:
        raise KeyError(
            f"Scenario '{scenario_id}' must define exactly two sides in scenario.sides"
        )
    side_items = list(sides.items())
    decision_format = _decision_format_for_scenario(
        scenario_id=scenario_id,
        scenario=scenario,
        template=decision_format_template,
    )
    public_instruction = public_instruction.replace(
        "{decision_format}", decision_format
    )
    opening_instruction = opening_instruction.replace(
        "{decision_format}", decision_format
    )
    if private_instruction is not None:
        private_instruction = private_instruction.replace(
            "{decision_format}", decision_format
        )

    alpha_label, alpha_persona = side_items[0]
    beta_label, beta_persona = side_items[1]

    alpha_incentive = _incentive_text_for_side(
        scenario_id=scenario_id,
        scenario=scenario,
        side_label=alpha_label,
        incentive_direction=incentive_direction,
        incentive_type=incentive_type,
    )
    beta_incentive = _incentive_text_for_side(
        scenario_id=scenario_id,
        scenario=scenario,
        side_label=beta_label,
        incentive_direction=incentive_direction,
        incentive_type=incentive_type,
    )

    alpha_self_role = base_prompt.format(
        speaker_id="A",
        question=question_text,
        persona=alpha_persona["actual_persona"],
    )
    beta_self_role = base_prompt.format(
        speaker_id="B",
        question=question_text,
        persona=beta_persona["actual_persona"],
    )

    if alpha_incentive:
        alpha_self_role = alpha_self_role + incentive_prompt.format(
            incentive=alpha_incentive
        )
    if beta_incentive:
        beta_self_role = beta_self_role + incentive_prompt.format(
            incentive=beta_incentive
        )

    alpha_perceives_beta = perceived_prompt.format(
        perceived_persona=beta_persona["perceived_persona_base"]
    )
    beta_perceives_alpha = perceived_prompt.format(
        perceived_persona=alpha_persona["perceived_persona_base"]
    )

    return [
        {
            "name": "Alpha",
            "model": model,
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
                "survey_question_groups": survey_question_groups or {},
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
            "model": model,
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
                "survey_question_groups": survey_question_groups or {},
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
