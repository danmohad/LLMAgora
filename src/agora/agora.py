"""The Agora orchestrator for agent interactions."""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence

from agora.survey import parse_survey_response_str

from .agent import Agent
from .memory import MemoryTurn

ALLOWED_SUBTURN_EVENTS = (
    "public_utterance",
    "private_utterance",
    "public_survey",
    "private_survey",
)


class Agora:
    """Coordinates agents, enforces event order, and records structured turns."""

    def __init__(
        self,
        agents: Sequence[Agent],
        *,
        event_order: Optional[Sequence[str]] = None,
    ) -> None:
        self._agents: List[Agent] = list(agents)
        if len(self._agents) != 2:
            raise ValueError("Agora requires exactly two agents")
        self._agent_lookup: Dict[str, Agent] = {agent.id: agent for agent in self._agents}
        if len(self._agent_lookup) != len(self._agents):
            raise ValueError("Agent identifiers must be unique")
        for agent in self._agents:
            agent.attach_agora(self)

        self._event_log: List[MemoryTurn] = []
        self._event_counter = 0
        self._turns: List[dict] = []
        self._pre_interviews = self._empty_interview_stage(stage="pre")
        self._post_interviews = self._empty_interview_stage(stage="post")
        self._event_order = self._resolve_event_order(event_order)

    def _empty_interview_stage(self, *, stage: str) -> dict:
        if stage not in {"pre", "post"}:
            raise ValueError(f"Unknown interview stage: {stage}")
        return {
            "Alpha": {
                "speaker_id": self._agents[0].id,
                "speaker_name": self._agents[0].name,
                "response": None,
                "keep": (
                    self._agents[0].pre_interview_keep
                    if stage == "pre"
                    else self._agents[0].post_interview_keep
                ),
            },
            "Beta": {
                "speaker_id": self._agents[1].id,
                "speaker_name": self._agents[1].name,
                "response": None,
                "keep": (
                    self._agents[1].pre_interview_keep
                    if stage == "pre"
                    else self._agents[1].post_interview_keep
                ),
            },
        }

    def _enabled_subturn_events(self) -> list[str]:
        enabled = ["public_utterance"]
        if any(agent.supports_private_reflection for agent in self._agents):
            enabled.append("private_utterance")
        if any(
            agent.do_survey_eval and agent.enable_public_survey
            for agent in self._agents
        ):
            enabled.append("public_survey")
        if any(
            agent.do_survey_eval and agent.enable_private_survey
            for agent in self._agents
        ):
            enabled.append("private_survey")
        return enabled

    def _resolve_event_order(self, event_order: Optional[Sequence[str]]) -> list[str]:
        enabled = self._enabled_subturn_events()
        if event_order is None:
            return list(enabled)

        order = list(event_order)
        if not order:
            raise ValueError("event_order must not be empty")
        unknown = [event for event in order if event not in ALLOWED_SUBTURN_EVENTS]
        if unknown:
            raise ValueError(
                f"event_order contains unknown events: {', '.join(unknown)}"
            )
        if len(order) != len(set(order)):
            raise ValueError("event_order must not contain duplicates")
        if set(order) != set(enabled):
            raise ValueError(
                "event_order must match enabled events 1:1. "
                f"Enabled: {enabled}. Provided: {order}."
            )
        return order

    def _next_event_id(self) -> int:
        self._event_counter += 1
        return self._event_counter

    def _slot_name(self, index: int) -> str:
        return "Alpha" if index == 0 else "Beta"

    def _blank_subturn(self, agent: Agent) -> dict:
        return {
            "speaker_id": agent.id,
            "speaker_name": agent.name,
            "public_utterance": None,
            "private_utterance": None,
            "public_survey": None,
            "private_survey": None,
        }

    def _append_event_turn(self, turn: MemoryTurn, recipients: Sequence[Agent]) -> None:
        self._event_log.append(turn)
        for agent in recipients:
            agent.observe_turn(turn)

    def _clear_post_interviews_for_continuation(self) -> None:
        """Drop previously finalized post-interviews before adding more turns."""
        if not any(turn.role == "post_interview" for turn in self._event_log):
            return

        self._post_interviews = self._empty_interview_stage(stage="post")
        self._event_log = [turn for turn in self._event_log if turn.role != "post_interview"]

        for agent in self._agents:
            agent.reset_memory()
        for agent in self._agents:
            for turn in self.history_for_agent(agent.id):
                agent.observe_turn(turn)

    def run(
        self,
        *,
        num_turns: int,
        verbose: bool = False,
        skip_first_agent_first_reflection: bool = False,
    ) -> List[MemoryTurn]:
        """Run periodic macro-turns; each turn contains Alpha then Beta sub-turns."""

        if num_turns <= 0:
            raise ValueError("num_turns must be positive")

        first_reflection_skipped = False
        starting_turn_num = len(self._turns)
        self._clear_post_interviews_for_continuation()

        # Turn 0 special case: optional pre-interviews, once per conversation.
        if starting_turn_num == 0:
            for index, agent in enumerate(self._agents):
                slot = self._slot_name(index)
                if not agent.pre_interview_instruction:
                    continue
                response = agent.generate_interview_response(agent.pre_interview_instruction)
                self._pre_interviews[slot]["response"] = response
                self._pre_interviews[slot]["keep"] = agent.pre_interview_keep
                pre_turn = MemoryTurn(
                    turn_id=self._next_event_id(),
                    speaker_id=agent.id,
                    role="pre_interview",
                    private_reflection=response,
                    metadata={
                        "speaker_name": agent.name,
                        "turn_num": 0,
                        "subturn": slot,
                        "event_type": "pre_interview",
                    },
                    keep=agent.pre_interview_keep,
                )
                recipients = [agent] if agent.pre_interview_keep else []
                self._append_event_turn(pre_turn, recipients)
                if verbose:
                    suffix = " (excluded)" if not agent.pre_interview_keep else ""
                    print(f"Turn 0 | {slot} (pre-interview){suffix}: {response}")

        for offset in range(1, num_turns + 1):
            turn_num = starting_turn_num + offset
            turn_entry = {
                "turn_num": turn_num,
                "Alpha": self._blank_subturn(self._agents[0]),
                "Beta": self._blank_subturn(self._agents[1]),
            }

            for index, agent in enumerate(self._agents):
                slot = self._slot_name(index)
                subturn = turn_entry[slot]
                opening = turn_num == 1 and slot == "Alpha"

                for event_name in self._event_order:
                    if event_name == "public_utterance":
                        speech = agent.generate_public_speech(opening=opening)
                        subturn["public_utterance"] = speech
                        public_turn = MemoryTurn(
                            turn_id=self._next_event_id(),
                            speaker_id=agent.id,
                            role="assistant",
                            public_speech=speech,
                            metadata={
                                "speaker_name": agent.name,
                                "turn_num": turn_num,
                                "subturn": slot,
                                "event_type": "public_utterance",
                            },
                        )
                        self._append_event_turn(public_turn, self._agents)
                        if verbose:
                            print(f"Turn {turn_num} | {slot} (public): {speech}")

                    elif event_name == "private_utterance":
                        if not agent.supports_private_reflection:
                            continue
                        if (
                            skip_first_agent_first_reflection
                            and slot == "Alpha"
                            and turn_num == 1
                            and not first_reflection_skipped
                        ):
                            first_reflection_skipped = True
                            continue
                        reflection = agent.generate_private_reflection()
                        subturn["private_utterance"] = reflection
                        reflection_turn = MemoryTurn(
                            turn_id=self._next_event_id(),
                            speaker_id=agent.id,
                            role="reflection",
                            private_reflection=reflection,
                            metadata={
                                "speaker_name": agent.name,
                                "turn_num": turn_num,
                                "subturn": slot,
                                "event_type": "private_utterance",
                            },
                            keep=agent.private_keep,
                        )
                        recipients = [agent] if agent.private_keep else []
                        self._append_event_turn(reflection_turn, recipients)
                        if verbose:
                            suffix = " (excluded)" if not agent.private_keep else ""
                            print(
                                f"Turn {turn_num} | {slot} (private){suffix}: {reflection}"
                            )

                    elif event_name == "public_survey":
                        if not (agent.do_survey_eval and agent.enable_public_survey):
                            continue
                        response = agent.generate_public_survey_response(
                            agent.survey_questions
                        )
                        parsed = parse_survey_response_str(
                            response,
                            agent.survey_question_groups,
                        )
                        subturn["public_survey"] = parsed
                        survey_turn = MemoryTurn(
                            turn_id=self._next_event_id(),
                            speaker_id=agent.id,
                            role="public_survey",
                            public_speech=json.dumps(parsed),
                            metadata={
                                "speaker_name": agent.name,
                                "turn_num": turn_num,
                                "subturn": slot,
                                "event_type": "public_survey",
                                "survey_scores": parsed,
                            },
                            keep=agent.public_survey_keep,
                        )
                        recipients = list(self._agents) if agent.public_survey_keep else []
                        self._append_event_turn(survey_turn, recipients)
                        if verbose:
                            suffix = " (excluded)" if not agent.public_survey_keep else ""
                            print(
                                f"Turn {turn_num} | {slot} (public survey){suffix}: {parsed}"
                            )

                    elif event_name == "private_survey":
                        if not (agent.do_survey_eval and agent.enable_private_survey):
                            continue
                        response_private = agent.generate_private_survey_response(
                            agent.survey_questions
                        )
                        parsed_private = parse_survey_response_str(
                            response_private,
                            agent.survey_question_groups,
                        )
                        subturn["private_survey"] = parsed_private
                        survey_turn = MemoryTurn(
                            turn_id=self._next_event_id(),
                            speaker_id=agent.id,
                            role="private_survey",
                            private_reflection=json.dumps(parsed_private),
                            metadata={
                                "speaker_name": agent.name,
                                "turn_num": turn_num,
                                "subturn": slot,
                                "event_type": "private_survey",
                                "survey_scores": parsed_private,
                            },
                            keep=agent.private_survey_keep,
                        )
                        recipients = [agent] if agent.private_survey_keep else []
                        self._append_event_turn(survey_turn, recipients)
                        if verbose:
                            suffix = " (excluded)" if not agent.private_survey_keep else ""
                            print(
                                f"Turn {turn_num} | {slot} (private survey){suffix}: {parsed_private}"
                            )

            self._turns.append(turn_entry)

        # Turn N+1 special case: optional post-interviews.
        final_turn_num = len(self._turns) + 1
        for index, agent in enumerate(self._agents):
            slot = self._slot_name(index)
            if not agent.post_interview_instruction:
                continue
            response = agent.generate_interview_response(agent.post_interview_instruction)
            self._post_interviews[slot] = {
                "speaker_id": agent.id,
                "speaker_name": agent.name,
                "response": response,
                "keep": agent.post_interview_keep,
            }
            post_turn = MemoryTurn(
                turn_id=self._next_event_id(),
                speaker_id=agent.id,
                role="post_interview",
                private_reflection=response,
                metadata={
                    "speaker_name": agent.name,
                    "turn_num": final_turn_num,
                    "subturn": slot,
                    "event_type": "post_interview",
                },
                keep=agent.post_interview_keep,
            )
            recipients = [agent] if agent.post_interview_keep else []
            self._append_event_turn(post_turn, recipients)
            if verbose:
                suffix = " (excluded)" if not agent.post_interview_keep else ""
                print(f"Turn {final_turn_num} | {slot} (post-interview){suffix}: {response}")

        return list(self._event_log)

    def history(self) -> List[MemoryTurn]:
        """Return the event log used for model context and diagnostics."""
        return list(self._event_log)

    def structured_history(self) -> dict:
        """Return canonical turn-structured history for outputs and analytics."""
        return {
            "event_order": list(self._event_order),
            "pre_interviews": json.loads(json.dumps(self._pre_interviews)),
            "turns": json.loads(json.dumps(self._turns)),
            "post_interviews": json.loads(json.dumps(self._post_interviews)),
        }

    @property
    def agents(self) -> Sequence[Agent]:
        return tuple(self._agents)

    def load_structured_history(
        self,
        *,
        event_order: Sequence[str],
        pre_interviews: dict,
        turns: Sequence[dict],
        post_interviews: dict,
    ) -> None:
        """Load canonical structured history and rebuild event-log memory."""

        self._event_order = self._resolve_event_order(event_order)
        self._pre_interviews = json.loads(json.dumps(pre_interviews))
        self._turns = json.loads(json.dumps(list(turns)))
        self._post_interviews = json.loads(json.dumps(post_interviews))

        for agent in self._agents:
            agent.reset_memory()
        self._event_log = []
        self._event_counter = 0

        # Rebuild event log from canonical representation.
        for slot in ("Alpha", "Beta"):
            stage = self._pre_interviews.get(slot, {})
            response = stage.get("response")
            if response is None:
                continue
            speaker_id = stage.get("speaker_id")
            speaker_name = stage.get("speaker_name", slot)
            keep = bool(stage.get("keep", False))
            turn = MemoryTurn(
                turn_id=self._next_event_id(),
                speaker_id=speaker_id,
                role="pre_interview",
                private_reflection=response,
                metadata={"speaker_name": speaker_name, "turn_num": 0, "subturn": slot, "event_type": "pre_interview"},
                keep=keep,
            )
            recipients = [self._agent_lookup[speaker_id]] if keep and speaker_id in self._agent_lookup else []
            self._append_event_turn(turn, recipients)

        for turn_entry in self._turns:
            turn_num = int(turn_entry.get("turn_num", 0))
            for slot_index, slot in enumerate(("Alpha", "Beta")):
                agent = self._agents[slot_index]
                subturn = turn_entry.get(slot, {})
                speaker_id = subturn.get("speaker_id", agent.id)
                speaker_name = subturn.get("speaker_name", agent.name)
                for event_name in self._event_order:
                    if event_name == "public_utterance":
                        speech = subturn.get("public_utterance")
                        if speech is None:
                            continue
                        turn = MemoryTurn(
                            turn_id=self._next_event_id(),
                            speaker_id=speaker_id,
                            role="assistant",
                            public_speech=speech,
                            metadata={
                                "speaker_name": speaker_name,
                                "turn_num": turn_num,
                                "subturn": slot,
                                "event_type": "public_utterance",
                            },
                        )
                        self._append_event_turn(turn, self._agents)

                    elif event_name == "private_utterance":
                        reflection = subturn.get("private_utterance")
                        if reflection is None:
                            continue
                        turn = MemoryTurn(
                            turn_id=self._next_event_id(),
                            speaker_id=speaker_id,
                            role="reflection",
                            private_reflection=reflection,
                            metadata={
                                "speaker_name": speaker_name,
                                "turn_num": turn_num,
                                "subturn": slot,
                                "event_type": "private_utterance",
                            },
                            keep=agent.private_keep,
                        )
                        recipients = [agent] if agent.private_keep else []
                        self._append_event_turn(turn, recipients)

                    elif event_name == "public_survey":
                        scores = subturn.get("public_survey")
                        if scores is None:
                            continue
                        turn = MemoryTurn(
                            turn_id=self._next_event_id(),
                            speaker_id=speaker_id,
                            role="public_survey",
                            public_speech=json.dumps(scores),
                            metadata={
                                "speaker_name": speaker_name,
                                "turn_num": turn_num,
                                "subturn": slot,
                                "event_type": "public_survey",
                                "survey_scores": scores,
                            },
                            keep=agent.public_survey_keep,
                        )
                        recipients = list(self._agents) if agent.public_survey_keep else []
                        self._append_event_turn(turn, recipients)

                    elif event_name == "private_survey":
                        scores = subturn.get("private_survey")
                        if scores is None:
                            continue
                        turn = MemoryTurn(
                            turn_id=self._next_event_id(),
                            speaker_id=speaker_id,
                            role="private_survey",
                            private_reflection=json.dumps(scores),
                            metadata={
                                "speaker_name": speaker_name,
                                "turn_num": turn_num,
                                "subturn": slot,
                                "event_type": "private_survey",
                                "survey_scores": scores,
                            },
                            keep=agent.private_survey_keep,
                        )
                        recipients = [agent] if agent.private_survey_keep else []
                        self._append_event_turn(turn, recipients)

        for slot in ("Alpha", "Beta"):
            stage = self._post_interviews.get(slot, {})
            response = stage.get("response")
            if response is None:
                continue
            speaker_id = stage.get("speaker_id")
            speaker_name = stage.get("speaker_name", slot)
            keep = bool(stage.get("keep", False))
            turn = MemoryTurn(
                turn_id=self._next_event_id(),
                speaker_id=speaker_id,
                role="post_interview",
                private_reflection=response,
                metadata={
                    "speaker_name": speaker_name,
                    "turn_num": len(self._turns) + 1,
                    "subturn": slot,
                    "event_type": "post_interview",
                },
                keep=keep,
            )
            recipients = [self._agent_lookup[speaker_id]] if keep and speaker_id in self._agent_lookup else []
            self._append_event_turn(turn, recipients)

    def history_for_agent(self, agent_id: str) -> List[MemoryTurn]:
        """Return the history view appropriate for a particular agent."""

        if agent_id not in self._agent_lookup:
            raise KeyError(f"Unknown agent id: {agent_id}")
        visible: List[MemoryTurn] = []
        for turn in self._event_log:
            if turn.role in {"reflection", "pre_interview", "post_interview", "private_survey"}:
                if turn.speaker_id != agent_id or not turn.keep:
                    continue
            elif turn.role == "public_survey":
                if not turn.keep:
                    continue
            visible.append(turn)
        return visible


__all__ = ["ALLOWED_SUBTURN_EVENTS", "Agora"]
