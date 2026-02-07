import builtins
import sys
import types

import pytest

from agora.debate_analyzer import DebateAnalyzer
from agora.memory import MemoryTurn


class DummyModel:
    def encode(self, text, convert_to_tensor=True):
        return f"emb:{text}"


class DummyUtil:
    @staticmethod
    def cos_sim(_a, _b):
        class Score:
            def item(self):
                return 0.42
        return Score()


def test_model_requires_dependency(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sentence_transformers" or name.startswith("sentence_transformers"):
            raise ModuleNotFoundError("No module named 'sentence_transformers'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    analyzer = DebateAnalyzer({"A": {"debate_turns": [], "pre_interview": None, "post_interview": None}})
    with pytest.raises(RuntimeError):
        _ = analyzer.model


def test_similarity_and_alignment(monkeypatch):
    def fake_load(self):
        self._util = DummyUtil
        return DummyModel()

    monkeypatch.setattr(DebateAnalyzer, "_load_model", fake_load)

    debate_data = {
        "Alpha": {
            "debate_turns": [
                {"public_speech": "Hello", "private_reflection": "Hi"}
            ],
            "pre_interview": None,
            "post_interview": None,
        },
        "Beta": {
            "debate_turns": [
                {"public_speech": "Yo", "private_reflection": "Yo"}
            ],
            "pre_interview": None,
            "post_interview": None,
        },
    }

    analyzer = DebateAnalyzer(debate_data)
    honesty = analyzer.compute_intra_agent_honesty()
    assert honesty["Alpha"]["scores"][0] == 0.42

    cached = analyzer.compute_intra_agent_honesty()
    assert cached is honesty

    alignment = analyzer.compute_inter_agent_alignment()
    assert alignment["scores"][0] == 0.42

    cached_alignment = analyzer.compute_inter_agent_alignment()
    assert cached_alignment is alignment

    analyzer._util = None
    assert analyzer.calculate_similarity("a", "b") == 0.42


def test_compute_alignment_requires_two_agents(monkeypatch):
    def fake_load(self):
        self._util = DummyUtil
        return DummyModel()

    monkeypatch.setattr(DebateAnalyzer, "_load_model", fake_load)

    debate_data = {
        "Solo": {
            "debate_turns": [
                {"public_speech": "Hello", "private_reflection": "Hi"}
            ],
            "pre_interview": None,
            "post_interview": None,
        }
    }

    analyzer = DebateAnalyzer(debate_data)
    with pytest.raises(ValueError):
        analyzer.compute_inter_agent_alignment()


def test_structured_history_path(monkeypatch):
    def fake_load(self):
        self._util = DummyUtil
        return DummyModel()

    monkeypatch.setattr(DebateAnalyzer, "_load_model", fake_load)

    turns = [
        MemoryTurn(
            turn_id=1,
            speaker_id="a",
            role="assistant",
            public_speech="Hi",
            metadata={"speaker_name": "Alpha"},
        ),
        MemoryTurn(
            turn_id=2,
            speaker_id="b",
            role="assistant",
            public_speech="Hello",
            metadata={"speaker_name": "Beta"},
        ),
    ]

    analyzer = DebateAnalyzer(turns)
    result = analyzer.compute_intra_agent_honesty(force_recompute=True)
    assert "Alpha" in result


def test_load_model_with_fake_dependency(monkeypatch, capsys):
    class FakeTransformer:
        def __init__(self, name):
            self.name = name

        def encode(self, text, convert_to_tensor=True):
            return f"emb:{text}"

    fake_module = types.SimpleNamespace(SentenceTransformer=FakeTransformer, util=DummyUtil)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    analyzer = DebateAnalyzer({"A": {"debate_turns": [], "pre_interview": None, "post_interview": None}})
    model = analyzer._load_model()
    assert isinstance(model, FakeTransformer)
    assert analyzer._util is DummyUtil
    assert "Loading model" in capsys.readouterr().out
