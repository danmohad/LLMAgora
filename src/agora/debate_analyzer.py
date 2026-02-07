"""Semantic similarity metrics for a two-agent debate history."""

from __future__ import annotations

from typing import Any, Optional


class DebateAnalyzer:
    """Compute honesty/alignment metrics from a structured debate history."""

    def __init__(self, memory_turns: Any, model_name: str = "all-mpnet-base-v2"):
        # Accept either raw MemoryTurn history or pre-structured debate data.
        if isinstance(memory_turns, dict):
            self.debate_data = memory_turns
        else:
            from agora.persona_evaluator import get_structured_debate_history

            self.debate_data = get_structured_debate_history(memory_turns)

        self.model_name = model_name
        self._model: Optional[Any] = None
        self._util: Optional[Any] = None
        self._intra_agent_honesty: Optional[dict[str, dict[str, list[float]]]] = None
        self._inter_agent_alignment: dict[tuple[str, str], dict[str, list[float]]] = {}

    @property
    def model(self) -> Any:
        """Lazy-load the sentence-transformers model only if needed."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self) -> Any:
        try:
            from sentence_transformers import SentenceTransformer, util
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "sentence-transformers is required for DebateAnalyzer. "
                "Install it to compute similarity metrics."
            ) from exc
        self._util = util
        print(f"Loading model: {self.model_name}...")
        return SentenceTransformer(self.model_name)

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Return cosine similarity between two text snippets."""
        embedding1 = self.model.encode(text1, convert_to_tensor=True)
        embedding2 = self.model.encode(text2, convert_to_tensor=True)
        if self._util is None:
            self._model = self._load_model()
        cosine_score = self._util.cos_sim(embedding1, embedding2)
        return float(cosine_score.item())

    def compute_intra_agent_honesty(self, force_recompute: bool = False) -> dict[str, dict[str, list[float]]]:
        """Score per-turn similarity between each agent's private and public text."""
        if self._intra_agent_honesty is not None and not force_recompute:
            return self._intra_agent_honesty

        self._intra_agent_honesty = {
            speaker_name: {
                "turns": list(range(len(speaker_data["debate_turns"]))),
                "scores": [
                    self.calculate_similarity(
                        turn["private_reflection"],
                        turn["public_speech"],
                    )
                    for turn in speaker_data["debate_turns"]
                ],
            }
            for speaker_name, speaker_data in self.debate_data.items()
        }
        return self._intra_agent_honesty

    def compute_inter_agent_alignment(
        self,
        agent_a_narrative: str = "public_speech",
        agent_b_narrative: str = "public_speech",
        force_recompute: bool = False,
    ) -> dict[str, list[float]]:
        """Score turn-by-turn similarity between Alpha/Beta narratives."""
        cache_key = (agent_a_narrative, agent_b_narrative)
        if cache_key in self._inter_agent_alignment and not force_recompute:
            return self._inter_agent_alignment[cache_key]

        agent_ids = list(self.debate_data.keys())
        if len(agent_ids) < 2:
            raise ValueError("Requires at least two agents")

        agent_a, agent_b = agent_ids[:2]
        turns_a = self.debate_data[agent_a]["debate_turns"]
        turns_b = self.debate_data[agent_b]["debate_turns"]
        num_turns = min(len(turns_a), len(turns_b))
        turns = list(range(num_turns))

        scores = [
            self.calculate_similarity(
                turns_a[t][agent_a_narrative],
                turns_b[t][agent_b_narrative],
            )
            for t in turns
        ]

        result = {"turns": turns, "scores": scores}
        self._inter_agent_alignment[cache_key] = result
        return result
