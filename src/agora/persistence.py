"""Utilities for saving and loading Agora state."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Sequence

from .agent import Agent
from .agora import Agora
from .llm import LLMClient


@dataclass
class AgentState:
    """Persistable representation of an agent's prompts."""

    id: str
    name: str
    model: str
    system_prompt: str
    response_instruction: str
    opening_instruction: str | None = None
    private_response_instruction: str | None = None
    private_response_keep: bool = True
    pre_interview_instruction: str | None = None
    pre_interview_keep: bool = True
    post_interview_instruction: str | None = None
    post_interview_keep: bool = True
    survey_questions: list[str] | None = None
    survey_question_groups: dict[str, str] | None = None
    survey_public_prompt: str | None = None
    survey_private_prompt: str | None = None
    enable_public_survey: bool = True
    enable_private_survey: bool = True
    public_survey_keep: bool = False
    private_survey_keep: bool = False

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
            survey_questions=config.get("survey_questions"),
            survey_question_groups=config.get("survey_question_groups"),
            survey_public_prompt=config.get("survey_public_prompt"),
            survey_private_prompt=config.get("survey_private_prompt"),
            enable_public_survey=config.get("enable_public_survey", True),
            enable_private_survey=config.get("enable_private_survey", True),
            public_survey_keep=config.get("public_survey_keep", False),
            private_survey_keep=config.get("private_survey_keep", False),
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
            "survey_questions": self.survey_questions,
            "survey_question_groups": self.survey_question_groups,
            "survey_public_prompt": self.survey_public_prompt,
            "survey_private_prompt": self.survey_private_prompt,
            "enable_public_survey": self.enable_public_survey,
            "enable_private_survey": self.enable_private_survey,
            "public_survey_keep": self.public_survey_keep,
            "private_survey_keep": self.private_survey_keep,
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
            survey_questions=payload.get("survey_questions"),
            survey_question_groups=payload.get("survey_question_groups"),
            survey_public_prompt=payload.get("survey_public_prompt"),
            survey_private_prompt=payload.get("survey_private_prompt"),
            enable_public_survey=payload.get("enable_public_survey", True),
            enable_private_survey=payload.get("enable_private_survey", True),
            public_survey_keep=payload.get("public_survey_keep", False),
            private_survey_keep=payload.get("private_survey_keep", False),
        )


@dataclass
class AgoraSnapshot:
    """Combination of agent definitions and canonical turn structure."""

    agent_states: List[AgentState]
    event_order: list[str]
    pre_interviews: dict
    turns: list[dict]
    post_interviews: dict
    llm_receipts: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_agora(cls, agora: Agora) -> "AgoraSnapshot":
        agent_states = [AgentState.from_agent(agent) for agent in agora.agents]
        history = agora.structured_history()
        return cls(
            agent_states=agent_states,
            event_order=list(history.get("event_order", [])),
            llm_receipts=list(history.get("llm_receipts", [])),
            pre_interviews=history.get("pre_interviews", {}),
            turns=list(history.get("turns", [])),
            post_interviews=history.get("post_interviews", {}),
        )

    def to_dict(self) -> dict:
        return {
            "llm_receipts": self.llm_receipts,
            "event_order": self.event_order,
            "agent_states": [agent.to_dict() for agent in self.agent_states],
            "pre_interviews": self.pre_interviews,
            "turns": self.turns,
            "post_interviews": self.post_interviews,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "AgoraSnapshot":
        agent_states = [
            AgentState.from_dict(item) for item in payload["agent_states"]
        ]
        return cls(
            agent_states=agent_states,
            event_order=list(payload.get("event_order", [])),
            pre_interviews=payload.get("pre_interviews", {}),
            turns=list(payload.get("turns", [])),
            post_interviews=payload.get("post_interviews", {}),
            llm_receipts=list(payload.get("llm_receipts", [])),
        )

    def save(self, path: Path | str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "AgoraSnapshot":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    def instantiate(self, llm_factory: Callable[[AgentState], LLMClient]) -> Agora:
        agents: List[Agent] = []
        for agent_state in self.agent_states:
            client = llm_factory(agent_state)
            agent = Agent.from_configuration(agent_state.to_dict(), llm_client=client)
            agents.append(agent)

        agora = Agora(agents, event_order=self.event_order)
        agora.load_structured_history(
            event_order=self.event_order,
            llm_receipts=self.llm_receipts,
            pre_interviews=self.pre_interviews,
            turns=self.turns,
            post_interviews=self.post_interviews,
        )
        return agora


def save_snapshot(path: Path | str, agora: Agora) -> None:
    """Persist the entire Agora to ``path``."""
    AgoraSnapshot.from_agora(agora).save(path)


def load_snapshot(path: Path | str, llm_factory: Callable[[AgentState], LLMClient]) -> Agora:
    """Restore an Agora from disk, instantiating LLM clients via ``llm_factory``."""
    snapshot = AgoraSnapshot.load(path)
    return snapshot.instantiate(llm_factory)


__all__ = [
    "AgentState",
    "AgoraSnapshot",
    "load_snapshot",
    "save_snapshot",
]
