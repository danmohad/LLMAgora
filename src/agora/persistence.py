"""Utilities for saving and loading Agora state."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from .agent import Agent
from .agora import Agora
from .llm import LLMClient
from .memory import MemoryTurn


@dataclass
class AgentState:
    """Persistable representation of an agent's prompts."""

    id: str
    name: str
    model: str
    system_prompt: str
    response_instruction: str
    opening_instruction: Optional[str] = None
    private_response_instruction: Optional[str] = None
    private_response_keep: bool = True
    pre_interview_instruction: Optional[str] = None
    pre_interview_keep: bool = True
    post_interview_instruction: Optional[str] = None
    post_interview_keep: bool = True
    survey_base_prompt: Optional[str] = None

    @classmethod
    def from_agent(cls, agent: Agent) -> "AgentState":
        config = agent.export_configuration()
        return cls(
            id=config["id"] or agent.id,
            name=config["name"] or agent.name,
            model=config["model"] or agent.model,
            system_prompt=config.get("system_prompt", "") or "",
            response_instruction=config.get("response_instruction", "") or "",
            opening_instruction=config.get("opening_instruction"),
            private_response_instruction=config.get("private_response_instruction"),
            private_response_keep=config.get("private_response_keep", True),
            pre_interview_instruction=config.get("pre_interview_instruction"),
            pre_interview_keep=config.get("pre_interview_keep", True),
            post_interview_instruction=config.get("post_interview_instruction"),
            post_interview_keep=config.get("post_interview_keep", True),
            survey_base_prompt=config.get("survey_base_prompt"),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "response_instruction": self.response_instruction,
            "opening_instruction": self.opening_instruction,
            "private_response_instruction": self.private_response_instruction,
            "private_response_keep": self.private_response_keep,
            "pre_interview_instruction": self.pre_interview_instruction,
            "pre_interview_keep": self.pre_interview_keep,
            "post_interview_instruction": self.post_interview_instruction,
            "post_interview_keep": self.post_interview_keep,
            "survey_base_prompt": self.survey_base_prompt,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "AgentState":
        return cls(
            id=payload["id"],
            name=payload["name"],
            model=payload["model"],
            system_prompt=payload.get("system_prompt", ""),
            response_instruction=payload.get("response_instruction", ""),
            opening_instruction=payload.get("opening_instruction"),
            private_response_instruction=payload.get("private_response_instruction"),
            private_response_keep=payload.get("private_response_keep", True),
            pre_interview_instruction=payload.get("pre_interview_instruction"),
            pre_interview_keep=payload.get("pre_interview_keep", True),
            post_interview_instruction=payload.get("post_interview_instruction"),
            post_interview_keep=payload.get("post_interview_keep", True),
            survey_base_prompt=payload.get("survey_base_prompt"),
        )


@dataclass
class AgoraSnapshot:
    """Combination of agent definitions and turn history."""

    agents: List[AgentState]
    turns: List[MemoryTurn]

    @classmethod
    def from_agora(cls, agora: Agora) -> "AgoraSnapshot":
        agent_states = [AgentState.from_agent(agent) for agent in agora.agents]
        turns = [MemoryTurn.from_dict(turn.to_dict()) for turn in agora.history()]
        return cls(agent_states, turns)

    def to_dict(self) -> dict:
        return {
            "agents": [agent.to_dict() for agent in self.agents],
            "turns": [turn.to_dict() for turn in self.turns],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "AgoraSnapshot":
        agents = [AgentState.from_dict(item) for item in payload.get("agents", [])]
        turns = [MemoryTurn.from_dict(item) for item in payload.get("turns", [])]
        return cls(agents, turns)

    def save(self, path: Path | str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path | str) -> "AgoraSnapshot":
        payload = json.loads(Path(path).read_text())
        return cls.from_dict(payload)

    def instantiate(self, llm_factory: Callable[[AgentState], LLMClient]) -> Agora:
        agents: List[Agent] = []
        for agent_state in self.agents:
            client = llm_factory(agent_state)
            agent = Agent.from_configuration(agent_state.to_dict(), llm_client=client)
            agents.append(agent)

        agora = Agora(agents)
        agora.load_history(self.turns)
        return agora


def save_snapshot(path: Path | str, agora: Agora) -> None:
    """Persist the entire Agora to ``path``."""

    AgoraSnapshot.from_agora(agora).save(path)


def load_snapshot(
    path: Path | str, llm_factory: Callable[[AgentState], LLMClient]
) -> Agora:
    """Restore an Agora from disk, instantiating LLM clients via ``llm_factory``."""

    snapshot = AgoraSnapshot.load(path)
    return snapshot.instantiate(llm_factory)


__all__ = [
    "AgentState",
    "AgoraSnapshot",
    "save_snapshot",
    "load_snapshot",
]
