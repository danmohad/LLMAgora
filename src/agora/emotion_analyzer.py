"""Emotion classification for debate transcripts.

Uses a pretrained HuggingFace text-classification pipeline to score the
probability of each emotion label for every agent turn.

Default model: ``j-hartmann/emotion-english-distilroberta-base``
Emotions: anger, disgust, fear, joy, neutral, sadness, surprise
"""

from __future__ import annotations

from typing import Any, Optional

from .debate_history import get_structured_debate_history

PUBLIC_NARRATIVE_FIELD = "public_speech"
PRIVATE_NARRATIVE_FIELD = "private_reflection"

DEFAULT_EMOTION_MODEL = "SamLowe/roberta-base-go_emotions"

# Shape returned by classify_turns / classify_field:
#   {
#     agent_name: {
#       "turns": [1, 2, 3, ...],
#       "emotions": {
#         "joy":     [0.12, 0.34, ...],
#         "anger":   [0.05, 0.08, ...],
#         ...
#       }
#     }
#   }
EmotionResult = dict[str, dict[str, Any]]


class EmotionAnalyzer:
    """Classify emotions in debate speech turn-by-turn.

    Parameters
    ----------
    memory_turns:
        Raw Agora structured history (``{"turns": [...]}``), already-normalized
        debate data keyed by speaker name, or any format accepted by
        :func:`~agora.debate_history.get_structured_debate_history`.
    model_name:
        HuggingFace model identifier for a text-classification model that
        returns per-label probabilities.  Defaults to
        ``j-hartmann/emotion-english-distilroberta-base``.
    device:
        Optional device string passed to the HuggingFace pipeline
        (e.g. ``"cpu"``, ``"cuda"``).  ``None`` lets the library choose.
    """

    def __init__(
        self,
        memory_turns: Any,
        model_name: str = DEFAULT_EMOTION_MODEL,
        device: Optional[str] = None,
    ) -> None:
        if isinstance(memory_turns, dict) and "turns" in memory_turns:
            self.debate_data = get_structured_debate_history(memory_turns)
        elif isinstance(memory_turns, dict):
            self.debate_data = memory_turns
        else:
            self.debate_data = get_structured_debate_history(memory_turns)

        self.model_name = model_name
        self.device = device
        self._pipeline: Optional[Any] = None

    @property
    def pipeline(self) -> Any:
        """Lazy-load the HuggingFace pipeline on first use."""
        if self._pipeline is None:
            self._pipeline = self._load_pipeline()
        return self._pipeline

    def _load_pipeline(self) -> Any:
        try:
            from transformers import pipeline
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "transformers is required for EmotionAnalyzer. "
                "Install it with: pip install transformers"
            ) from exc

        kwargs: dict[str, Any] = {
            "task": "text-classification",
            "model": self.model_name,
            "top_k": None,  # return all labels
        }
        if self.device is not None:
            kwargs["device"] = self.device

        print(f"Loading emotion model: {self.model_name}...")
        return pipeline(**kwargs)

    def classify_text(self, text: str) -> dict[str, float]:
        """Return a dict mapping each emotion label to its probability.

        Parameters
        ----------
        text:
            A single string to classify.

        Returns
        -------
        dict
            ``{"joy": 0.45, "anger": 0.12, ...}`` — probabilities sum to ~1.
        """
        raw = self.pipeline(text)
        # HuggingFace pipelines with top_k=None return a list of
        # [{"label": ..., "score": ...}, ...] (possibly nested in a list).
        if isinstance(raw, list) and raw and isinstance(raw[0], list):
            raw = raw[0]
        return {item["label"]: float(item["score"]) for item in raw}

    def classify_field(self, field: str) -> EmotionResult:
        """Classify all agents' turns for the given text field.

        Parameters
        ----------
        field:
            Either :data:`PUBLIC_NARRATIVE_FIELD` (``"public_speech"``) or
            :data:`PRIVATE_NARRATIVE_FIELD` (``"private_reflection"``).

        Returns
        -------
        EmotionResult
            Nested dict: ``{agent_name: {"turns": [...], "emotions": {label: [...]}}}``
        """
        result: EmotionResult = {}
        for agent_name, agent_data in self.debate_data.items():
            turns_data = agent_data.get("debate_turns", [])
            turn_nums: list[int] = []
            per_turn_dicts: list[dict[str, float]] = []

            for i, turn in enumerate(turns_data):
                text = turn.get(field, "")
                if not text:
                    continue
                turn_nums.append(int(turn.get("turn_num", i + 1)))
                per_turn_dicts.append(self.classify_text(text))

            if not per_turn_dicts:
                result[agent_name] = {"turns": [], "emotions": {}}
                continue

            # Transpose list-of-dicts → dict-of-lists
            all_labels = list(per_turn_dicts[0].keys())
            emotions: dict[str, list[float]] = {
                label: [d.get(label, 0.0) for d in per_turn_dicts]
                for label in all_labels
            }
            result[agent_name] = {"turns": turn_nums, "emotions": emotions}

        return result
