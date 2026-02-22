"""Semantic similarity analysis for debate transcripts.

The analyzer compares text streams across turns and agents using one of:
- cosine similarity over sentence embeddings
- NLI entailment probability over text pairs
"""

from __future__ import annotations

from typing import Any, Optional

from .debate_history import get_structured_debate_history

PUBLIC_NARRATIVE_FIELD = "public_speech"
PRIVATE_NARRATIVE_FIELD = "private_reflection"
SEMANTIC_SIMILARITY_METHOD_COSINE = "cosine"
SEMANTIC_SIMILARITY_METHOD_NLI = "nli"
SEMANTIC_SIMILARITY_METHODS: tuple[str, ...] = (
    SEMANTIC_SIMILARITY_METHOD_COSINE,
    SEMANTIC_SIMILARITY_METHOD_NLI,
)
DEFAULT_COSINE_MODEL_NAME = "all-mpnet-base-v2"
DEFAULT_NLI_MODEL_NAME = "dleemiller/finecat-nli-l"


class SemanticSimilarityAnalyzer:
    """Compute semantic similarity metrics from structured debate history."""

    def __init__(
        self,
        memory_turns: Any,
        method: str = SEMANTIC_SIMILARITY_METHOD_COSINE,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        # Accept raw memory turns, canonical Agora structured history,
        # or already-normalized debate data keyed by speaker.
        if isinstance(memory_turns, dict) and "turns" in memory_turns:
            self.debate_data = get_structured_debate_history(memory_turns)
        elif isinstance(memory_turns, dict):
            self.debate_data = memory_turns
        else:
            self.debate_data = get_structured_debate_history(memory_turns)

        if method not in SEMANTIC_SIMILARITY_METHODS:
            raise ValueError(
                "Unknown semantic similarity method: "
                f"{method}. Allowed values: {', '.join(SEMANTIC_SIMILARITY_METHODS)}"
            )

        self.method = method
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = (
                DEFAULT_NLI_MODEL_NAME
                if self.method == SEMANTIC_SIMILARITY_METHOD_NLI
                else DEFAULT_COSINE_MODEL_NAME
            )
        self.device = device
        self._model: Optional[Any] = None
        self._util: Optional[Any] = None
        self._id2label: Optional[dict[Any, str]] = None
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
            from sentence_transformers import CrossEncoder, SentenceTransformer, util
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "sentence-transformers is required for SemanticSimilarityAnalyzer. "
                "Install it to compute similarity metrics."
            ) from exc

        model_kwargs = {}
        if self.device is not None:
            model_kwargs["device"] = self.device

        print(f"Loading {self.method} model: {self.model_name}...")
        if self.method == SEMANTIC_SIMILARITY_METHOD_COSINE:
            self._util = util
            self._id2label = None
            return SentenceTransformer(self.model_name, **model_kwargs)

        cross_encoder = CrossEncoder(self.model_name, **model_kwargs)
        self._util = None
        self._id2label = self._extract_id2label(cross_encoder)
        return cross_encoder

    @staticmethod
    def _extract_id2label(model: Any) -> Optional[dict[Any, str]]:
        config = getattr(getattr(model, "model", None), "config", None)
        if config is None:
            return None
        id2label = getattr(config, "id2label", None)
        if not isinstance(id2label, dict):
            return None
        return {key: str(value) for key, value in id2label.items()}

    def _nli_entailment_index(self, num_labels: int) -> int:
        if self._id2label:
            for label_idx, label_name in self._id2label.items():
                if "entail" in label_name.lower():
                    try:
                        return int(label_idx)
                    except (TypeError, ValueError):
                        continue
        if num_labels == 3:
            return 2
        raise ValueError(
            "Could not infer entailment index from NLI labels. "
            "Model must expose an entailment label or use 3-way NLI logits."
        )

    def cosine_similarity(self, text_a: str, text_b: str) -> float:
        """Return cosine similarity between two text snippets."""
        if self.method != SEMANTIC_SIMILARITY_METHOD_COSINE:
            raise ValueError("cosine_similarity requires method='cosine'")
        embedding_a = self.model.encode(text_a, convert_to_tensor=True)
        embedding_b = self.model.encode(text_b, convert_to_tensor=True)
        if self._util is None:
            self._model = self._load_model()
        cosine_score = self._util.cos_sim(embedding_a, embedding_b)
        return float(cosine_score.item())

    def _nli_entailment_probability(self, premise: str, hypothesis: str) -> float:
        """Return one-way entailment probability for (premise, hypothesis)."""
        if self.method != SEMANTIC_SIMILARITY_METHOD_NLI:
            raise ValueError("nli_similarity requires method='nli'")
        probs = self.model.predict([(premise, hypothesis)], apply_softmax=True)
        if hasattr(probs, "tolist"):
            probs = probs.tolist()
        if isinstance(probs, tuple):
            probs = list(probs)
        if isinstance(probs, list) and probs and isinstance(probs[0], list | tuple):
            scores = list(probs[0])
        elif isinstance(probs, list):
            scores = probs
        else:
            raise ValueError("NLI model returned an unexpected probability shape")
        entailment_index = self._nli_entailment_index(len(scores))
        return float(scores[entailment_index])

    def nli_similarity(self, premise: str, hypothesis: str) -> float:
        """Return symmetric NLI similarity via averaged bidirectional entailment."""
        forward = self._nli_entailment_probability(premise, hypothesis)
        reverse = self._nli_entailment_probability(hypothesis, premise)
        return (forward + reverse) / 2.0

    def _score_text_pair(self, text_a: str, text_b: str) -> float:
        if self.method == SEMANTIC_SIMILARITY_METHOD_COSINE:
            return self.cosine_similarity(text_a, text_b)
        return self.nli_similarity(text_a, text_b)

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
                    self._score_text_pair(
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
                self._score_text_pair(
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
    "SEMANTIC_SIMILARITY_METHOD_COSINE",
    "SEMANTIC_SIMILARITY_METHOD_NLI",
    "SEMANTIC_SIMILARITY_METHODS",
    "DEFAULT_COSINE_MODEL_NAME",
    "DEFAULT_NLI_MODEL_NAME",
]
