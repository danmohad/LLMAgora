from agora.memory import MemoryTurn


def test_to_chat_message_respects_keep_flag():
    turn = MemoryTurn(turn_id=1, speaker_id="a", role="assistant", public_speech="Hi", keep=False)
    assert turn.to_chat_message(viewer_id="a") is None


def test_to_chat_message_hidden_reflection():
    turn = MemoryTurn(
        turn_id=2,
        speaker_id="a",
        role="reflection",
        private_reflection="secret",
        keep=True,
    )
    assert turn.to_chat_message(viewer_id="b") is None


def test_to_chat_message_multi_party_prefixes_user_content():
    turn = MemoryTurn(
        turn_id=3,
        speaker_id="a",
        role="assistant",
        public_speech="Hello",
        metadata={"speaker_name": "Alpha"},
        keep=True,
    )
    rendered = turn.to_chat_message(viewer_id="b", multi_party=True)
    assert rendered is not None
    assert rendered["role"] == "user"
    assert rendered["content"] == "Alpha: Hello"
