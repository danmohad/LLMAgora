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
        agent = Agent(
            name=cfg["name"],
            model=cfg["model"],
            llm_client=llm_client,
            system_prompt=system_prompt,
            response_instruction=cfg["response_instruction"],
            private_response_instruction=private_instr,
            private_response_keep=private_keep,
            pre_interview_instruction=pre_instr,
            pre_interview_keep=pre_keep,
            post_interview_instruction=post_instr,
            post_interview_keep=post_keep,
        )
        agents.append(agent)
    return agents


def run_debate_session(
    agent_configs: Sequence[dict],
    *,
    turns_per_agent: int,
    verbose: bool = False,
    skip_first_agent_first_reflection: bool = False,
    snapshot_path: Optional[Path | str] = None,
    load_snapshot_flag: bool = False,
    save_snapshot_flag: bool = False,
    llm_client: Optional[LLMClient] = None,
    client_factory: Callable[[], LLMClient] = OpenRouterClient,
) -> Tuple[Agora, List[Agent]]:
    """Run an Agora session from configuration dictionaries.

    When ``snapshot_path`` exists and ``load_snapshot_flag`` is True, the Agora
    is restored from disk instead of constructing new agents. Snapshots are
    saved back to the same path when ``save_snapshot_flag`` is True.
    """

    managed_client: Optional[LLMClient] = None
    if llm_client is None:
        managed_client = client_factory()
        llm_client = managed_client

    try:
        snapshot_file = Path(snapshot_path) if snapshot_path else None
        if load_snapshot_flag and snapshot_file and snapshot_file.exists():
            agora = load_snapshot(snapshot_file, lambda _state: llm_client)
            agents = list(agora.agents)
        else:
            agents = build_agents_from_configs(agent_configs, llm_client)
            agora = Agora(agents)

        agora.run(
            max_turns_per_agent=turns_per_agent,
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
        if turn.role in {"reflection", "pre_interview", "post_interview"} and not turn.keep:
            note = " (excluded)"
        if turn.role == "reflection":
            lines.append(
                f"Turn {turn.turn_id:02d} | {speaker} (private){note}: {turn.private_reflection}"
            )
        elif turn.role in {"pre_interview", "post_interview"}:
            label = "pre-interview" if turn.role == "pre_interview" else "post-interview"
            lines.append(
                f"Turn {turn.turn_id:02d} | {speaker} ({label}){note}: {turn.private_reflection}"
            )
        else:
            lines.append(f"Turn {turn.turn_id:02d} | {speaker}: {turn.public_speech}")
    return "\n".join(lines)


def print_agent_histories(agents: Iterable[Agent]) -> None:
    """Print each agent's view of the history to stdout."""

    for agent in agents:
        print(f"\n### Full history visible to {agent.name}")
        print(format_history_for_agent(agent))


def load_persona_catalog(persona_path: Path | str) -> dict:
    """Load persona definitions from disk."""

    return json.loads(Path(persona_path).read_text())


def load_question_catalog(question_path: Path | str) -> dict:
    """Load question definitions from disk."""

    return json.loads(Path(question_path).read_text())


DEFAULT_PROMPT_SET = "default"
DEFAULT_PROMPT_PATH = Path(__file__).resolve().parents[2] / "data" / "prompts.json"


def load_prompt_catalog(prompt_path: Path | str | None = None) -> dict:
    """Load prompt template catalog from disk or the repo default."""

    prompt_resource = Path(prompt_path) if prompt_path is not None else DEFAULT_PROMPT_PATH

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
        raise KeyError(f"Prompt template '{name}' not found; available sets: {available}")

    required_keys = [
        "base_prompt",
        "perceived_prompt",
        "public_instruction",
        "private_instruction",
        "pre_interview_instruction",
        "post_interview_instruction",
    ]
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise KeyError(
            f"Prompt template '{name}' missing required fields: {', '.join(sorted(missing))}"
        )
    return payload


DEFAULT_PROMPTS = load_prompt_templates(DEFAULT_PROMPT_SET)
DEFAULT_BASE_PROMPT = DEFAULT_PROMPTS["base_prompt"]
DEFAULT_PERCEIVED_PROMPT = DEFAULT_PROMPTS["perceived_prompt"]
DEFAULT_PUBLIC_INSTRUCTION = DEFAULT_PROMPTS["public_instruction"]
DEFAULT_PRIVATE_INSTRUCTION = DEFAULT_PROMPTS["private_instruction"]
DEFAULT_PRE_INTERVIEW_INSTRUCTION = DEFAULT_PROMPTS["pre_interview_instruction"]
DEFAULT_POST_INTERVIEW_INSTRUCTION = DEFAULT_PROMPTS["post_interview_instruction"]


def build_persona_agent_configs(
    *,
    alpha_persona_id: str,
    beta_persona_id: str,
    question_id: str,
    personas: dict,
    questions: dict,
    alpha_model: str,
    beta_model: str,
    base_prompt: Optional[str] = None,
    perceived_prompt: Optional[str] = None,
    public_instruction: Optional[str] = None,
    private_instruction: Optional[str] = None,
    pre_interview_instruction: Optional[str] = None,
    post_interview_instruction: Optional[str] = None,
    prompt_set: str = DEFAULT_PROMPT_SET,
    private_response_keep: bool = True,
    pre_interview_keep: bool = False,
    post_interview_keep: bool = False,
    prompt_templates: Optional[dict] = None,
    prompt_catalog: Optional[dict] = None,
    prompt_path: Path | str | None = None,
) -> List[dict]:
    """Construct agent configs for the persona-driven debate notebook."""

    prompts = prompt_templates
    if prompts is None:
        if prompt_set == DEFAULT_PROMPT_SET and prompt_catalog is None and prompt_path is None:
            prompts = DEFAULT_PROMPTS
        else:
            prompts = load_prompt_templates(
                prompt_set, prompt_catalog=prompt_catalog, prompt_path=prompt_path
            )

    base_prompt = base_prompt or prompts["base_prompt"]
    perceived_prompt = perceived_prompt or prompts["perceived_prompt"]
    public_instruction = public_instruction or prompts["public_instruction"]
    private_instruction = private_instruction or prompts["private_instruction"]
    pre_interview_instruction = pre_interview_instruction or prompts["pre_interview_instruction"]
    post_interview_instruction = post_interview_instruction or prompts["post_interview_instruction"]

    personas_data = personas.get("personas", {})
    questions_data = questions.get("questions", {})

    if question_id not in questions_data:
        raise KeyError(f"Unknown question id: {question_id}")
    if alpha_persona_id not in personas_data:
        raise KeyError(f"Unknown persona id: {alpha_persona_id}")
    if beta_persona_id not in personas_data:
        raise KeyError(f"Unknown persona id: {beta_persona_id}")

    question_text = questions_data[question_id]["question"]
    alpha_persona = personas_data[alpha_persona_id]
    beta_persona = personas_data[beta_persona_id]

    alpha_self_role = base_prompt.format(
        speaker_id="A", question=question_text, persona=alpha_persona["actual_persona"]
    )
    beta_self_role = base_prompt.format(
        speaker_id="B", question=question_text, persona=beta_persona["actual_persona"]
    )

    alpha_perceives_beta = perceived_prompt.format(
        perceived_persona=personas_data[beta_persona_id]["perceived_persona"]
    )
    beta_perceives_alpha = perceived_prompt.format(
        perceived_persona=personas_data[alpha_persona_id]["perceived_persona"]
    )

    return [
        {
            "name": "Alpha",
            "model": alpha_model,
            "self_role": alpha_self_role,
            "perceived_nonself_roles": [{"name": "Beta", "role": alpha_perceives_beta}],
            "response_instruction": public_instruction,
            "private_response": {"instruction": private_instruction, "keep": private_response_keep},
            "pre_interview": {"instruction": pre_interview_instruction, "keep": pre_interview_keep},
            "post_interview": {"instruction": post_interview_instruction, "keep": post_interview_keep},
        },
        {
            "name": "Beta",
            "model": beta_model,
            "self_role": beta_self_role,
            "perceived_nonself_roles": [{"name": "Alpha", "role": beta_perceives_alpha}],
            "response_instruction": public_instruction,
            "private_response": {"instruction": private_instruction, "keep": private_response_keep},
            "pre_interview": {"instruction": pre_interview_instruction, "keep": pre_interview_keep},
            "post_interview": {"instruction": post_interview_instruction, "keep": post_interview_keep},
        },
    ]


__all__ = [
    "build_agents_from_configs",
    "build_persona_agent_configs",
    "DEFAULT_PROMPT_PATH",
    "DEFAULT_PROMPT_SET",
    "DEFAULT_BASE_PROMPT",
    "DEFAULT_PERCEIVED_PROMPT",
    "DEFAULT_PUBLIC_INSTRUCTION",
    "DEFAULT_PRIVATE_INSTRUCTION",
    "DEFAULT_PRE_INTERVIEW_INSTRUCTION",
    "DEFAULT_POST_INTERVIEW_INSTRUCTION",
    "load_prompt_catalog",
    "extract_instruction",
    "format_history_for_agent",
    "load_prompt_templates",
    "load_persona_catalog",
    "load_question_catalog",
    "print_agent_histories",
    "run_debate_session",
]
