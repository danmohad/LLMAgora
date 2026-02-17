"""Semantic similarity analysis for debate transcripts.

The analyzer compares text streams across turns and agents using sentence-level
embeddings:
- self-consistency: each agent's private reflection vs their public statement
- cross-agent alignment: one agent narrative stream vs the other agent stream
"""

from __future__ import annotations

from typing import Any, Optional

from .debate_history import get_structured_debate_history

PUBLIC_NARRATIVE_FIELD = "public_speech"
PRIVATE_NARRATIVE_FIELD = "private_reflection"


class SemanticSimilarityAnalyzer:
    """Compute semantic similarity metrics from structured debate history."""

    def __init__(self, memory_turns: Any, model_name: str = "all-mpnet-base-v2"):
        # Accept raw memory turns, canonical Agora structured history,
        # or already-normalized debate data keyed by speaker.
        if isinstance(memory_turns, dict) and "turns" in memory_turns:
            self.debate_data = get_structured_debate_history(memory_turns)
        elif isinstance(memory_turns, dict):
            self.debate_data = memory_turns
        else:
            self.debate_data = get_structured_debate_history(memory_turns)

        self.model_name = model_name
        self._model: Optional[Any] = None
        self._util: Optional[Any] = None
        self._self_consistency_cache: Optional[dict[str, dict[str, list[float]]]] = None
        self._cross_agent_alignment_cache: dict[
            tuple[str, str], dict[str, list[float]]
        ] = {}

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
                "sentence-transformers is required for SemanticSimilarityAnalyzer. "
                "Install it to compute similarity metrics."
            ) from exc
        self._util = util
        print(f"Loading model: {self.model_name}...")
        return SentenceTransformer(self.model_name)

    def cosine_similarity(self, text_a: str, text_b: str) -> float:
        """Return cosine similarity between two text snippets."""
        embedding_a = self.model.encode(text_a, convert_to_tensor=True)
        embedding_b = self.model.encode(text_b, convert_to_tensor=True)
        if self._util is None:
            self._model = self._load_model()
        cosine_score = self._util.cos_sim(embedding_a, embedding_b)
        return float(cosine_score.item())

    def compute_self_consistency_scores(
        self, force_recompute: bool = False
    ) -> dict[str, dict[str, list[float]]]:
        """Score private-vs-public similarity for each agent on each turn."""
        if self._self_consistency_cache is not None and not force_recompute:
            return self._self_consistency_cache

        self._self_consistency_cache = {
            speaker_name: {
                "turns": [
                    turn.get("turn_num", idx + 1)
                    for idx, turn in enumerate(speaker_data["debate_turns"])
                ],
                "scores": [
                    self.cosine_similarity(
                        turn[PRIVATE_NARRATIVE_FIELD],
                        turn[PUBLIC_NARRATIVE_FIELD],
                    )
                    for turn in speaker_data["debate_turns"]
                ],
            }
            for speaker_name, speaker_data in self.debate_data.items()
        }
        return self._self_consistency_cache

    def compute_cross_agent_alignment_scores(
        self,
        agent_a_field: str = PUBLIC_NARRATIVE_FIELD,
        agent_b_field: str = PUBLIC_NARRATIVE_FIELD,
        force_recompute: bool = False,
    ) -> dict[str, list[float]]:
        """Score turn-by-turn similarity between two agents' selected text fields."""
        cache_key = (agent_a_field, agent_b_field)
        if cache_key in self._cross_agent_alignment_cache and not force_recompute:
            return self._cross_agent_alignment_cache[cache_key]

        agent_ids = list(self.debate_data.keys())
        if len(agent_ids) < 2:
            raise ValueError("Requires at least two agents")

        agent_a, agent_b = agent_ids[:2]
        turns_a = self.debate_data[agent_a]["debate_turns"]
        turns_b = self.debate_data[agent_b]["debate_turns"]
        by_turn_a = {
            turn.get("turn_num", index + 1): turn for index, turn in enumerate(turns_a)
        }
        by_turn_b = {
            turn.get("turn_num", index + 1): turn for index, turn in enumerate(turns_b)
        }
        turns = sorted(set(by_turn_a) & set(by_turn_b))

        scores = []
        for turn_num in turns:
            entry_a = by_turn_a[turn_num]
            entry_b = by_turn_b[turn_num]
            scores.append(
                self.cosine_similarity(
                    entry_a[agent_a_field],
                    entry_b[agent_b_field],
                )
            )

        result = {"turns": turns, "scores": scores}
        self._cross_agent_alignment_cache[cache_key] = result
        return result


__all__ = [
    "SemanticSimilarityAnalyzer",
    "PUBLIC_NARRATIVE_FIELD",
    "PRIVATE_NARRATIVE_FIELD",
]
