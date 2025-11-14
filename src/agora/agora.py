"""The Agora orchestrator for agent interactions."""

from typing import Dict, List, Sequence

from .agent import Agent
from .memory import MemoryTurn


class Agora:
    """Coordinates agents, enforces rules, and records public history."""

    def __init__(self, agents: Sequence[Agent]) -> None:
        """Attach agents to the arena and initialize shared bookkeeping."""

        if not agents:
            raise ValueError("Agora requires at least one agent")
        self._agents: List[Agent] = list(agents)
        self._agent_lookup: Dict[str, Agent] = {agent.id: agent for agent in self._agents}
        if len(self._agent_lookup) != len(self._agents):
            raise ValueError("Agent identifiers must be unique")
        for agent in self._agents:
            agent.attach_agora(self)
        self._turn_log: List[MemoryTurn] = []
        self._turn_counter = 0

    def run(self, *, max_turns_per_agent: int) -> List[MemoryTurn]:
        """Run the Agora until each agent has taken the specified number of turns."""

        if max_turns_per_agent <= 0:
            raise ValueError("max_turns_per_agent must be positive")

        # Track how many turns each agent has already taken.
        turns_taken: Dict[str, int] = {agent.id: 0 for agent in self._agents}
        agent_index = 0

        while True:
            if all(count >= max_turns_per_agent for count in turns_taken.values()):
                break

            agent = self._agents[agent_index % len(self._agents)]
            agent_index += 1

            if turns_taken[agent.id] >= max_turns_per_agent:
                continue

            # Ask the selected agent for its next public utterance.
            speech = agent.generate_public_speech()
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

        return list(self._turn_log)

    def history(self) -> List[MemoryTurn]:
        """Return the full public history of the Agora."""

        return list(self._turn_log)

    def agent_count(self) -> int:
        """Return the number of agents currently participating in the Agora."""

        return len(self._agents)

    def history_public(self) -> List[MemoryTurn]:
        """Return only the publicly visible portion of the history."""

        return self.history()

    def history_for_agent(self, agent_id: str) -> List[MemoryTurn]:
        """Return the history view appropriate for a particular agent."""

        if agent_id not in self._agent_lookup:
            raise KeyError(f"Unknown agent id: {agent_id}")
        return self.history_public()


__all__ = ["Agora"]
