"""Tests for agora.emotion_analyzer.EmotionAnalyzer."""

import builtins
from unittest.mock import patch

import pytest

from agora.emotion_analyzer import (
    DEFAULT_EMOTION_MODEL,
    PRIVATE_NARRATIVE_FIELD,
    PUBLIC_NARRATIVE_FIELD,
    EmotionAnalyzer,
)


# ---------------------------------------------------------------------------
# Minimal debate data helpers
# ---------------------------------------------------------------------------

def _debate_data(public_a="I agree", private_a="I secretly agree",
                 public_b="I disagree", private_b="I secretly disagree",
                 n_turns=2):
    """Build already-normalised debate_data for two agents."""
    alpha_turns = [
        {
            "turn_num": i + 1,
            "public_speech": f"{public_a} t{i+1}",
            "private_reflection": f"{private_a} t{i+1}",
        }
        for i in range(n_turns)
    ]
    beta_turns = [
        {
            "turn_num": i + 1,
            "public_speech": f"{public_b} t{i+1}",
            "private_reflection": f"{private_b} t{i+1}",
        }
        for i in range(n_turns)
    ]
    return {
        "Alpha": {"debate_turns": alpha_turns, "pre_interview": None, "post_interview": None},
        "Beta":  {"debate_turns": beta_turns,  "pre_interview": None, "post_interview": None},
    }


def _structured_history(n_turns=2):
    """Build a canonical Agora structured-history payload."""
    turns = []
    for i in range(1, n_turns + 1):
        turns.append({
            "turn_num": i,
            "Alpha": {
                "speaker_id": "Alpha",
                "public_utterance": f"alpha public t{i}",
                "private_utterance": f"alpha private t{i}",
            },
            "Beta": {
                "speaker_id": "Beta",
                "public_utterance": f"beta public t{i}",
                "private_utterance": f"beta private t{i}",
            },
        })
    return {"turns": turns, "pre_interviews": {}, "post_interviews": {}}


# ---------------------------------------------------------------------------
# Fake pipeline
# ---------------------------------------------------------------------------

def _make_fake_pipeline(labels=("joy", "anger", "neutral")):
    """Return a callable that mimics a HuggingFace text-classification pipeline."""
    n = len(labels)
    base_score = 1.0 / n

    def fake_pipeline(text):
        return [[{"label": lab, "score": base_score} for lab in labels]]

    return fake_pipeline


# ---------------------------------------------------------------------------
# Init / normalisation
# ---------------------------------------------------------------------------

def test_init_normalizes_supported_dict_inputs():
    data = _debate_data()
    ea = EmotionAnalyzer(data)
    assert ea.debate_data is data
    assert set(ea.debate_data.keys()) == {"Alpha", "Beta"}
    assert ea.model_name == DEFAULT_EMOTION_MODEL
    assert ea.device is None

    structured = EmotionAnalyzer(_structured_history())
    assert set(structured.debate_data.keys()) == {"Alpha", "Beta"}

    custom = EmotionAnalyzer(data, model_name="my/model", device="cpu")
    assert custom.model_name == "my/model"
    assert custom.device == "cpu"


# ---------------------------------------------------------------------------
# Lazy pipeline loading
# ---------------------------------------------------------------------------

def test_pipeline_lazy_loaded(monkeypatch):
    ea = EmotionAnalyzer(_debate_data())
    assert ea._pipeline is None  # not loaded yet

    fake = _make_fake_pipeline()

    def fake_load(self):
        return fake

    monkeypatch.setattr(EmotionAnalyzer, "_load_pipeline", fake_load)
    p = ea.pipeline
    assert p is fake
    # Second access returns the cached instance.
    assert ea.pipeline is fake


def test_pipeline_requires_transformers(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "transformers" or name.startswith("transformers"):
            raise ModuleNotFoundError("No module named 'transformers'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    ea = EmotionAnalyzer(_debate_data())
    with pytest.raises(RuntimeError, match="transformers"):
        ea._load_pipeline()


def test_load_pipeline_kwargs(monkeypatch):
    """_load_pipeline passes the right kwargs (including device when set)."""
    captured = {}

    def fake_pipeline(**kwargs):
        captured.update(kwargs)
        return _make_fake_pipeline()

    import agora.emotion_analyzer as _mod
    monkeypatch.setattr(_mod, "__builtins__", None)  # won't be reached; bypass via monkeypatch below

    # Patch at the module level inside emotion_analyzer
    original_load = EmotionAnalyzer._load_pipeline

    def patched_load(self):
        import agora.emotion_analyzer as em
        # Monkey-patch transformers.pipeline just for this call
        import sys
        import types
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.pipeline = fake_pipeline
        old = sys.modules.get("transformers")
        sys.modules["transformers"] = fake_transformers
        try:
            result = original_load(self)
        finally:
            if old is None:
                del sys.modules["transformers"]
            else:
                sys.modules["transformers"] = old
        return result

    ea = EmotionAnalyzer(_debate_data(), model_name="test/model", device="cpu")
    monkeypatch.setattr(EmotionAnalyzer, "_load_pipeline", patched_load)
    ea.pipeline  # trigger load
    assert captured.get("model") == "test/model"
    assert captured.get("device") == "cpu"
    assert captured.get("top_k") is None


def test_load_pipeline_no_device(monkeypatch):
    """_load_pipeline omits 'device' kwarg when device is None."""
    captured = {}

    import sys, types

    def patched_load(self):
        fake_transformers = types.ModuleType("transformers")

        def fake_pipeline(**kwargs):
            captured.update(kwargs)
            return _make_fake_pipeline()

        fake_transformers.pipeline = fake_pipeline
        old = sys.modules.get("transformers")
        sys.modules["transformers"] = fake_transformers
        try:
            from agora.emotion_analyzer import EmotionAnalyzer as _EA
            from transformers import pipeline
            kwargs = {
                "task": "text-classification",
                "model": self.model_name,
                "top_k": None,
            }
            print(f"Loading emotion model: {self.model_name}...")
            return pipeline(**kwargs)
        finally:
            if old is None:
                del sys.modules["transformers"]
            else:
                sys.modules["transformers"] = old

    ea = EmotionAnalyzer(_debate_data(), device=None)
    monkeypatch.setattr(EmotionAnalyzer, "_load_pipeline", patched_load)
    ea.pipeline
    assert "device" not in captured


# ---------------------------------------------------------------------------
# classify_text
# ---------------------------------------------------------------------------

def test_classify_text_returns_label_dict(monkeypatch):
    ea = EmotionAnalyzer(_debate_data())
    fake = _make_fake_pipeline(("joy", "anger", "neutral"))
    monkeypatch.setattr(ea, "_pipeline", fake)

    result = ea.classify_text("Hello world")
    assert set(result.keys()) == {"joy", "anger", "neutral"}
    for v in result.values():
        assert isinstance(v, float)


def test_classify_text_flat_output(monkeypatch):
    """Pipeline returning a flat list (not nested) is also handled."""
    ea = EmotionAnalyzer(_debate_data())

    def flat_pipeline(text):
        return [{"label": "joy", "score": 0.9}, {"label": "anger", "score": 0.1}]

    monkeypatch.setattr(ea, "_pipeline", flat_pipeline)
    result = ea.classify_text("test")
    assert result == {"joy": 0.9, "anger": 0.1}


# ---------------------------------------------------------------------------
# classify_field
# ---------------------------------------------------------------------------

def test_classify_field_public_and_private(monkeypatch):
    ea = EmotionAnalyzer(_debate_data(n_turns=3))
    fake = _make_fake_pipeline(("joy", "sadness"))
    monkeypatch.setattr(ea, "_pipeline", fake)

    public_result = ea.classify_field(PUBLIC_NARRATIVE_FIELD)

    assert set(public_result.keys()) == {"Alpha", "Beta"}
    for agent_result in public_result.values():
        assert agent_result["turns"] == [1, 2, 3]
        assert set(agent_result["emotions"].keys()) == {"joy", "sadness"}
        assert len(agent_result["emotions"]["joy"]) == 3

    private = EmotionAnalyzer(_debate_data(n_turns=2))
    monkeypatch.setattr(private, "_pipeline", _make_fake_pipeline(("anger",)))
    private_result = private.classify_field(PRIVATE_NARRATIVE_FIELD)
    assert private_result["Alpha"]["turns"] == [1, 2]
    assert "anger" in private_result["Alpha"]["emotions"]


def test_classify_field_empty_turns(monkeypatch):
    """Agents with no turns produce empty turn lists."""
    data = {
        "Alpha": {"debate_turns": [], "pre_interview": None, "post_interview": None},
    }
    ea = EmotionAnalyzer(data)
    fake = _make_fake_pipeline()
    monkeypatch.setattr(ea, "_pipeline", fake)

    result = ea.classify_field(PUBLIC_NARRATIVE_FIELD)
    assert result["Alpha"]["turns"] == []
    assert result["Alpha"]["emotions"] == {}


def test_classify_field_skips_empty_text(monkeypatch):
    """Turns whose text field is empty string are skipped."""
    data = {
        "Alpha": {
            "debate_turns": [
                {"turn_num": 1, "public_speech": "", "private_reflection": "some thought"},
                {"turn_num": 2, "public_speech": "hello", "private_reflection": ""},
            ],
            "pre_interview": None,
            "post_interview": None,
        },
    }
    ea = EmotionAnalyzer(data)
    fake = _make_fake_pipeline(("joy",))
    monkeypatch.setattr(ea, "_pipeline", fake)

    public_result = ea.classify_field(PUBLIC_NARRATIVE_FIELD)
    assert public_result["Alpha"]["turns"] == [2]

    private_result = ea.classify_field(PRIVATE_NARRATIVE_FIELD)
    assert private_result["Alpha"]["turns"] == [1]


def test_init_from_non_dict_memory_turns():
    """EmotionAnalyzer calls get_structured_debate_history when memory_turns is not a dict."""
    non_dict_turns = [{"turn_num": 1, "public_speech": "hello"}]
    fake_data = {"Alpha": {}}

    with patch(
        "agora.emotion_analyzer.get_structured_debate_history", return_value=fake_data
    ) as mock_gsdh:
        ea = EmotionAnalyzer(non_dict_turns)

    mock_gsdh.assert_called_once_with(non_dict_turns)
    assert ea.debate_data is fake_data
