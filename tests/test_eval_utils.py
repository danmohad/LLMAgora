from agora.eval_utils import get_structured_debate_history
from agora.memory import MemoryTurn


def test_eval_utils_structured_debate_history_pairs_turns_and_interviews():
    turns = [
        MemoryTurn(
            turn_id=1,
            speaker_id="a",
            role="pre_interview",
            private_reflection="pre",
            metadata={"speaker_name": "Alpha"},
        ),
        MemoryTurn(
            turn_id=2,
            speaker_id="a",
            role="reflection",
            private_reflection="think",
            metadata={"speaker_name": "Alpha"},
        ),
        MemoryTurn(
            turn_id=3,
            speaker_id="a",
            role="assistant",
            public_speech="hello",
            metadata={"speaker_name": "Alpha"},
        ),
        MemoryTurn(
            turn_id=4,
            speaker_id="b",
            role="assistant",
            public_speech="hi",
            metadata={},
        ),
        MemoryTurn(
            turn_id=5,
            speaker_id="b",
            role="reflection",
            private_reflection="b-think",
            metadata={},
        ),
        MemoryTurn(
            turn_id=6,
            speaker_id="b",
            role="assistant",
            public_speech="bye",
            metadata={},
        ),
        MemoryTurn(
            turn_id=7,
            speaker_id="a",
            role="post_interview",
            private_reflection="post",
            metadata={"speaker_name": "Alpha"},
        ),
    ]

    structured = get_structured_debate_history(turns)

    assert set(structured.keys()) == {"Alpha", "b"}
    assert structured["Alpha"]["pre_interview"] == "pre"
    assert structured["Alpha"]["post_interview"] == "post"
    assert structured["Alpha"]["debate_turns"] == [
        {
            "private_reflection": "think",
            "public_speech": "hello",
            "turn_id": 3,
        }
    ]

    assert structured["b"]["pre_interview"] is None
    assert structured["b"]["post_interview"] is None
    assert structured["b"]["debate_turns"][0]["private_reflection"] == ""
    assert structured["b"]["debate_turns"][0]["public_speech"] == "hi"
    assert structured["b"]["debate_turns"][1]["private_reflection"] == "b-think"
    assert structured["b"]["debate_turns"][1]["public_speech"] == "bye"
