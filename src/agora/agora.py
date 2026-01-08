"""The Agora orchestrator for agent interactions."""

from typing import Dict, List, Sequence

from agora.survey import parse_survey_response_str

from .agent import Agent
from .memory import MemoryTurn


class Agora:
    """Coordinates agents, enforces rules, and records public history."""

    def __init__(self, agents: Sequence[Agent]) -> None:
        """Attach agents to the arena and initialize shared bookkeeping."""

        if not agents:
            raise ValueError("Agora requires at least one agent")
        self._agents: List[Agent] = list(agents)
        self._agent_lookup: Dict[str, Agent] = {
            agent.id: agent for agent in self._agents
        }
        if len(self._agent_lookup) != len(self._agents):
            raise ValueError("Agent identifiers must be unique")
        for agent in self._agents:
            agent.attach_agora(self)
        self._turn_log: List[MemoryTurn] = []
        self._turn_counter = 0
        self.survey_respose = {}

    def run(
        self,
        *,
        max_turns_per_agent: int,
        verbose: bool = False,
        skip_first_agent_first_reflection: bool = False,
    ) -> List[MemoryTurn]:
        """
        Run the Agora until each agent has taken the specified number of turns.

        Args:
            max_turns_per_agent: Stop once every agent has produced this many public turns.
            verbose: When True, print turn-by-turn diagnostics for debugging.
            skip_first_agent_first_reflection: If True, suppress the very first
                reflection from the first agent (useful when pre-interviews already
                cover initial state).
        """

        if max_turns_per_agent <= 0:
            raise ValueError("max_turns_per_agent must be positive")

        # First round of survey
        for agent in self._agents:
            if agent.do_survey_eval:
                response = agent.generate_survey_response(agent.survey_questions)
                self.survey_respose[agent.id] = {0: parse_survey_response_str(response)}

                if verbose:
                    print(f"Survey reponse from {agent.name}:")
                    print(response)

        # Optional pre-interviews
        for agent in self._agents:
            if not agent.pre_interview_instruction:
                continue
            response = agent.generate_interview_response(
                agent.pre_interview_instruction
            )
            self._turn_counter += 1
            pre_turn = MemoryTurn(
                turn_id=self._turn_counter,
                speaker_id=agent.id,
                role="pre_interview",
                private_reflection=response,
                metadata={"speaker_name": agent.name},
                keep=agent.pre_interview_keep,
            )
            self._turn_log.append(pre_turn)
            if agent.pre_interview_keep:
                agent.observe_turn(pre_turn)
            if verbose:
                suffix = " (excluded)" if not agent.pre_interview_keep else ""
                print(
                    f"Turn {self._turn_counter} | {agent.name} (pre-interview){suffix}: {response}"
                )

        # Track how many turns each agent has already taken.
        turns_taken: Dict[str, int] = {agent.id: 0 for agent in self._agents}
        agent_index = 0
        opening_turn = not any(turn.role == "assistant" for turn in self._turn_log)

        first_reflection_skipped = False

        while True:
            if all(count >= max_turns_per_agent for count in turns_taken.values()):
                break

            agent = self._agents[agent_index % len(self._agents)]
            agent_index += 1

            if turns_taken[agent.id] >= max_turns_per_agent:
                continue

            # Allow the agent to privately reflect before speaking publicly.
            if agent.supports_private_reflection:
                if skip_first_agent_first_reflection and not first_reflection_skipped:
                    first_reflection_skipped = True
                else:
                    reflection = agent.generate_private_reflection()
                    self._turn_counter += 1
                    reflection_turn = MemoryTurn(
                        turn_id=self._turn_counter,
                        speaker_id=agent.id,
                        role="reflection",
                        private_reflection=reflection,
                        metadata={"speaker_name": agent.name},
                        keep=agent.private_keep,
                    )
                    self._turn_log.append(reflection_turn)
                    if agent.private_keep:
                        agent.observe_turn(reflection_turn)
                    if verbose:
                        suffix = " (excluded)" if not agent.private_keep else ""
                        print(
                            f"Turn {self._turn_counter} | {agent.name} (reflection){suffix}: {reflection}"
                        )

            # Ask the selected agent for its next public utterance.
            speech = agent.generate_public_speech(opening=opening_turn)
            opening_turn = False
            self._turn_counter += 1
            turn = MemoryTurn(
                turn_id=self._turn_counter,
                speaker_id=agent.id,
                role="assistant",
                public_speech=speech,
                metadata={"speaker_name": agent.name},
            )
            self._turn_log.append(turn)
            # Broadcast the public turn into every agent's local memory.
            for recipient in self._agents:
                recipient.observe_turn(turn)
            turns_taken[agent.id] += 1
            if verbose:
                print(f"Turn {self._turn_counter} | {agent.name} (public): {speech}")

            if agent.do_survey_eval:
                response = agent.generate_survey_response(agent.survey_questions)
                self.survey_respose[agent.id][self._turn_counter] = (
                    parse_survey_response_str(response)
                )

                if verbose:
                    print(f"Survey reponse from {agent.name}:")
                    print(response)

        # Optional post-interviews
        for agent in self._agents:
            if not agent.post_interview_instruction:
                continue
            response = agent.generate_interview_response(
                agent.post_interview_instruction
            )
            self._turn_counter += 1
            post_turn = MemoryTurn(
                turn_id=self._turn_counter,
                speaker_id=agent.id,
                role="post_interview",
                private_reflection=response,
                metadata={"speaker_name": agent.name},
                keep=agent.post_interview_keep,
            )
            self._turn_log.append(post_turn)
            if agent.post_interview_keep:
                agent.observe_turn(post_turn)
            if verbose:
                suffix = " (excluded)" if not agent.post_interview_keep else ""
                print(
                    f"Turn {self._turn_counter} | {agent.name} (post-interview){suffix}: {response}"
                )

        return list(self._turn_log)

    def history(self) -> List[MemoryTurn]:
        """Return the full history, including private reflections."""

        return list(self._turn_log)

    @property
    def agents(self) -> Sequence[Agent]:
        """Expose the list of agents participating in this Agora."""

        return tuple(self._agents)

    def load_history(self, turns: Sequence[MemoryTurn]) -> None:
        """
        Replace the Agora's history with the provided turns.

        Intended for snapshots; assumes the agents are freshly attached.
        """

        for agent in self._agents:
            agent.reset_memory()
        self._turn_log = []
        self._turn_counter = 0

        for turn in turns:
            self._turn_log.append(turn)
            if turn.role in {"reflection", "pre_interview", "post_interview"}:
                speaker = self._agent_lookup.get(turn.speaker_id)
                if speaker and turn.keep:
                    speaker.observe_turn(turn)
            elif turn.role == "assistant":
                for agent in self._agents:
                    agent.observe_turn(turn)
            self._turn_counter = max(self._turn_counter, turn.turn_id)

    def agent_count(self) -> int:
        """Return the number of agents currently participating in the Agora."""

        return len(self._agents)

    def history_public(self) -> List[MemoryTurn]:
        """Return only the publicly visible portion of the history."""

        return [turn for turn in self._turn_log if turn.role == "assistant"]

    def history_for_agent(self, agent_id: str) -> List[MemoryTurn]:
        """Return the history view appropriate for a particular agent."""

        if agent_id not in self._agent_lookup:
            raise KeyError(f"Unknown agent id: {agent_id}")
        visible: List[MemoryTurn] = []
        for turn in self._turn_log:
            if turn.role in {"reflection", "pre_interview", "post_interview"}:
                if turn.speaker_id != agent_id:
                    continue
                if not turn.keep:
                    continue
            visible.append(turn)
        return visible


__all__ = ["Agora"]
