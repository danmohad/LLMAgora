import builtins
import sys
import types

import pytest

from agora.semantic_similarity_analyzer import (
    DEFAULT_COSINE_MODEL_NAME,
    DEFAULT_NLI_MODEL_NAME,
    SEMANTIC_SIMILARITY_METHOD_NLI,
    SemanticSimilarityAnalyzer,
)


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


class DummyNLIModel:
    def __init__(self):
        self.calls = []

    def predict(self, pairs, apply_softmax=True):
        self.calls.append((pairs, apply_softmax))
        return [[0.1, 0.2, 0.7]]


def test_model_requires_dependency(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sentence_transformers" or name.startswith("sentence_transformers"):
            raise ModuleNotFoundError("No module named 'sentence_transformers'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    analyzer = SemanticSimilarityAnalyzer({"A": {"debate_turns": [], "pre_interview": None, "post_interview": None}})
    with pytest.raises(RuntimeError):
        _ = analyzer.model


def test_init_defaults_and_invalid_method():
    cosine = SemanticSimilarityAnalyzer(
        {"A": {"debate_turns": [], "pre_interview": None, "post_interview": None}}
    )
    nli = SemanticSimilarityAnalyzer(
        {"A": {"debate_turns": [], "pre_interview": None, "post_interview": None}},
        method=SEMANTIC_SIMILARITY_METHOD_NLI,
    )
    assert cosine.model_name == DEFAULT_COSINE_MODEL_NAME
    assert nli.model_name == DEFAULT_NLI_MODEL_NAME
    assert SemanticSimilarityAnalyzer(
        {"A": {"debate_turns": [], "pre_interview": None, "post_interview": None}},
        model_name="custom-cosine-model",
    ).model_name == "custom-cosine-model"
    with pytest.raises(ValueError):
        SemanticSimilarityAnalyzer(
            {"A": {"debate_turns": [], "pre_interview": None, "post_interview": None}},
            method="bad",
        )


def test_similarity_and_alignment(monkeypatch):
    def fake_load(self):
        self._util = DummyUtil
        return DummyModel()

    monkeypatch.setattr(SemanticSimilarityAnalyzer, "_load_model", fake_load)

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

    analyzer = SemanticSimilarityAnalyzer(debate_data)
    honesty = analyzer.compute_self_consistency_scores()
    assert honesty["Alpha"]["scores"][0] == 0.42

    cached = analyzer.compute_self_consistency_scores()
    assert cached is honesty

    alignment = analyzer.compute_cross_agent_alignment_scores()
    assert alignment["scores"][0] == 0.42

    cached_alignment = analyzer.compute_cross_agent_alignment_scores()
    assert cached_alignment is alignment

    analyzer._util = None
    assert analyzer.cosine_similarity("a", "b") == 0.42


def test_nli_similarity_and_alignment(monkeypatch):
    def fake_load(self):
        self._id2label = {0: "contradiction", 1: "neutral", 2: "entailment"}
        return DummyNLIModel()

    monkeypatch.setattr(SemanticSimilarityAnalyzer, "_load_model", fake_load)

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

    analyzer = SemanticSimilarityAnalyzer(debate_data, method="nli", device="mps")
    assert analyzer.nli_similarity("premise", "hypothesis") == 0.7

    honesty = analyzer.compute_self_consistency_scores()
    assert honesty["Alpha"]["scores"][0] == 0.7

    alignment = analyzer.compute_cross_agent_alignment_scores()
    assert alignment["scores"][0] == 0.7

    with pytest.raises(ValueError):
        analyzer.cosine_similarity("a", "b")


def test_similarity_method_specific_guards(monkeypatch):
    def fake_load(self):
        self._util = DummyUtil
        return DummyModel()

    monkeypatch.setattr(SemanticSimilarityAnalyzer, "_load_model", fake_load)
    analyzer = SemanticSimilarityAnalyzer(
        {"A": {"debate_turns": [], "pre_interview": None, "post_interview": None}}
    )
    with pytest.raises(ValueError):
        analyzer.nli_similarity("p", "h")


def test_nli_score_shape_and_label_resolution(monkeypatch):
    class TupleScoresNLI:
        def predict(self, _pairs, apply_softmax=True):
            assert apply_softmax is True
            return (0.2, 0.3, 0.5)

    def fake_load(self):
        self._id2label = {"bad-key": "entailment"}
        return TupleScoresNLI()

    monkeypatch.setattr(SemanticSimilarityAnalyzer, "_load_model", fake_load)
    analyzer = SemanticSimilarityAnalyzer(
        {"A": {"debate_turns": [], "pre_interview": None, "post_interview": None}},
        method="nli",
    )
    assert analyzer.nli_similarity("premise", "hypothesis") == 0.5
    with pytest.raises(ValueError):
        analyzer._nli_entailment_index(2)


def test_nli_similarity_with_tolist_scores(monkeypatch):
    class ToListScores:
        def tolist(self):
            return [[0.1, 0.2, 0.7]]

    class ToListNLI:
        def predict(self, _pairs, apply_softmax=True):
            assert apply_softmax is True
            return ToListScores()

    def fake_load(self):
        self._id2label = {0: "contradiction", 1: "neutral", 2: "entailment"}
        return ToListNLI()

    monkeypatch.setattr(SemanticSimilarityAnalyzer, "_load_model", fake_load)
    analyzer = SemanticSimilarityAnalyzer(
        {"A": {"debate_turns": [], "pre_interview": None, "post_interview": None}},
        method="nli",
    )
    assert analyzer.nli_similarity("premise", "hypothesis") == 0.7


def test_nli_similarity_averages_bidirectional_entailment(monkeypatch):
    class DirectionalNLI:
        def __init__(self):
            self.calls = []

        def predict(self, pairs, apply_softmax=True):
            self.calls.append((pairs, apply_softmax))
            premise, hypothesis = pairs[0]
            if (premise, hypothesis) == ("a", "b"):
                return [[0.2, 0.2, 0.6]]
            if (premise, hypothesis) == ("b", "a"):
                return [[0.2, 0.3, 0.5]]
            return [[0.1, 0.1, 0.8]]

    model_holder = {}

    def fake_load(self):
        self._id2label = {0: "contradiction", 1: "neutral", 2: "entailment"}
        model = DirectionalNLI()
        model_holder["model"] = model
        return model

    monkeypatch.setattr(SemanticSimilarityAnalyzer, "_load_model", fake_load)
    analyzer = SemanticSimilarityAnalyzer(
        {"A": {"debate_turns": [], "pre_interview": None, "post_interview": None}},
        method="nli",
    )
    assert analyzer.nli_similarity("a", "b") == pytest.approx(0.55)
    assert model_holder["model"].calls == [
        ([("a", "b")], True),
        ([("b", "a")], True),
    ]


def test_nli_similarity_invalid_shapes(monkeypatch):
    class BadScoresNLI:
        def __init__(self, value):
            self.value = value

        def predict(self, _pairs, apply_softmax=True):
            return self.value

    analyzer = SemanticSimilarityAnalyzer(
        {"A": {"debate_turns": [], "pre_interview": None, "post_interview": None}},
        method="nli",
    )

    def fake_load_scalar(self):
        self._id2label = {0: "contradiction", 1: "neutral", 2: "entailment"}
        return BadScoresNLI(0.9)

    monkeypatch.setattr(SemanticSimilarityAnalyzer, "_load_model", fake_load_scalar)
    with pytest.raises(ValueError):
        analyzer.nli_similarity("premise", "hypothesis")

    analyzer2 = SemanticSimilarityAnalyzer(
        {"A": {"debate_turns": [], "pre_interview": None, "post_interview": None}},
        method="nli",
    )

    def fake_load_two_scores(self):
        self._id2label = None
        return BadScoresNLI([0.4, 0.6])

    monkeypatch.setattr(SemanticSimilarityAnalyzer, "_load_model", fake_load_two_scores)
    with pytest.raises(ValueError):
        analyzer2.nli_similarity("premise", "hypothesis")


def test_extract_id2label_missing_config_cases():
    assert SemanticSimilarityAnalyzer._extract_id2label(object()) is None

    class NoDictMapping:
        model = types.SimpleNamespace(config=types.SimpleNamespace(id2label="bad"))

    assert SemanticSimilarityAnalyzer._extract_id2label(NoDictMapping()) is None


def test_compute_alignment_requires_two_agents(monkeypatch):
    def fake_load(self):
        self._util = DummyUtil
        return DummyModel()

    monkeypatch.setattr(SemanticSimilarityAnalyzer, "_load_model", fake_load)

    debate_data = {
        "Solo": {
            "debate_turns": [
                {"public_speech": "Hello", "private_reflection": "Hi"}
            ],
            "pre_interview": None,
            "post_interview": None,
        }
    }

    analyzer = SemanticSimilarityAnalyzer(debate_data)
    with pytest.raises(ValueError):
        analyzer.compute_cross_agent_alignment_scores()


def test_structured_history_path(monkeypatch):
    def fake_load(self):
        self._util = DummyUtil
        return DummyModel()

    monkeypatch.setattr(SemanticSimilarityAnalyzer, "_load_model", fake_load)

    structured_history = {
        "pre_interviews": {"Alpha": {"response": None}, "Beta": {"response": None}},
        "turns": [
            {
                "turn_num": 1,
                "Alpha": {
                    "public_utterance": "Hi",
                    "private_utterance": None,
                },
                "Beta": {
                    "public_utterance": "Hello",
                    "private_utterance": None,
                },
            }
        ],
        "post_interviews": {"Alpha": {"response": None}, "Beta": {"response": None}},
    }

    analyzer = SemanticSimilarityAnalyzer(structured_history)
    result = analyzer.compute_self_consistency_scores(force_recompute=True)
    assert "Alpha" in result


def test_structured_history_rejects_raw_turn_list():
    with pytest.raises(ValueError, match="canonical structured history"):
        SemanticSimilarityAnalyzer([])


def test_load_model_with_fake_dependency(monkeypatch, capsys):
    class FakeTransformer:
        def __init__(self, name, device=None):
            self.name = name
            self.device = device

        def encode(self, text, convert_to_tensor=True):
            return f"emb:{text}"

    class FakeCrossEncoder:
        def __init__(self, name, device=None):
            self.name = name
            self.device = device
            self.model = types.SimpleNamespace(
                config=types.SimpleNamespace(id2label={0: "contradiction", 1: "neutral", 2: "entailment"})
            )

    fake_module = types.SimpleNamespace(
        SentenceTransformer=FakeTransformer,
        CrossEncoder=FakeCrossEncoder,
        util=DummyUtil,
    )
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    analyzer = SemanticSimilarityAnalyzer(
        {"A": {"debate_turns": [], "pre_interview": None, "post_interview": None}},
        device="cpu",
    )
    model = analyzer._load_model()
    assert isinstance(model, FakeTransformer)
    assert model.device == "cpu"
    assert analyzer._util is DummyUtil
    assert "Loading cosine model" in capsys.readouterr().out


def test_load_nli_model_with_fake_dependency(monkeypatch):
    class FakeTransformer:
        def __init__(self, name, device=None):
            self.name = name
            self.device = device

    class FakeCrossEncoder:
        def __init__(self, name, device=None):
            self.name = name
            self.device = device
            self.model = types.SimpleNamespace(
                config=types.SimpleNamespace(id2label={0: "contradiction", 1: "neutral", 2: "entailment"})
            )

    fake_module = types.SimpleNamespace(
        SentenceTransformer=FakeTransformer,
        CrossEncoder=FakeCrossEncoder,
        util=DummyUtil,
    )
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    analyzer = SemanticSimilarityAnalyzer(
        {"A": {"debate_turns": [], "pre_interview": None, "post_interview": None}},
        method="nli",
        device="mps",
    )
    model = analyzer._load_model()
    assert isinstance(model, FakeCrossEncoder)
    assert model.device == "mps"
    assert analyzer._util is None
    assert analyzer._id2label == {0: "contradiction", 1: "neutral", 2: "entailment"}


def test_canonical_turn_structure_path(monkeypatch):
    def fake_load(self):
        self._util = DummyUtil
        return DummyModel()

    monkeypatch.setattr(SemanticSimilarityAnalyzer, "_load_model", fake_load)

    canonical = {
        "pre_interviews": {
            "Alpha": {"response": None},
            "Beta": {"response": None},
        },
        "turns": [
            {
                "turn_num": 1,
                "Alpha": {
                    "public_utterance": "alpha-public",
                    "private_utterance": "alpha-private",
                },
                "Beta": {
                    "public_utterance": "beta-public",
                    "private_utterance": "beta-private",
                },
            }
        ],
        "post_interviews": {
            "Alpha": {"response": None},
            "Beta": {"response": None},
        },
    }

    analyzer = SemanticSimilarityAnalyzer(canonical)
    honesty = analyzer.compute_self_consistency_scores()
    assert honesty["Alpha"]["turns"] == [1]
    assert honesty["Alpha"]["scores"][0] == 0.42
