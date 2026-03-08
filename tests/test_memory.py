from agora.memory import MemoryTurn, strip_transcript_label_prefix


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


def test_to_chat_message_labels_public_history_for_self_and_other():
    other_turn = MemoryTurn(
        turn_id=3,
        speaker_id="b",
        role="assistant",
        public_speech="Hello",
        metadata={"event_type": "public_utterance"},
    )
    self_turn = MemoryTurn(
        turn_id=4,
        speaker_id="a",
        role="assistant",
        public_speech="Hi back",
        metadata={"event_type": "public_utterance"},
    )

    assert other_turn.to_chat_message(viewer_id="a") == {
        "role": "user",
        "content": "[Other speaker | public statement]\nHello",
    }
    assert self_turn.to_chat_message(viewer_id="a") == {
        "role": "assistant",
        "content": "[You | public statement]\nHi back",
    }


def test_to_chat_message_labels_private_history_for_self():
    turn = MemoryTurn(
        turn_id=5,
        speaker_id="a",
        role="reflection",
        private_reflection="secret",
        metadata={"event_type": "private_utterance"},
    )

    assert turn.to_chat_message(viewer_id="a") == {
        "role": "assistant",
        "content": "[You | private note]\nsecret",
    }


def test_to_chat_message_labels_survey_and_interview_history():
    public_survey = MemoryTurn(
        turn_id=6,
        speaker_id="b",
        role="public_survey",
        public_speech='{"Q1": 1}',
        metadata={"event_type": "public_survey"},
    )
    private_survey = MemoryTurn(
        turn_id=7,
        speaker_id="a",
        role="private_survey",
        private_reflection='{"Q1": 2}',
        metadata={"event_type": "private_survey"},
    )
    pre_interview = MemoryTurn(
        turn_id=8,
        speaker_id="a",
        role="pre_interview",
        private_reflection="plan",
        metadata={"event_type": "pre_interview"},
    )
    post_interview = MemoryTurn(
        turn_id=9,
        speaker_id="a",
        role="post_interview",
        private_reflection="summary",
        metadata={"event_type": "post_interview"},
    )

    assert public_survey.to_chat_message(viewer_id="a") == {
        "role": "user",
        "content": '[Other speaker | public survey response]\n{"Q1": 1}',
    }
    assert private_survey.to_chat_message(viewer_id="a") == {
        "role": "assistant",
        "content": '[You | private survey response]\n{"Q1": 2}',
    }
    assert pre_interview.to_chat_message(viewer_id="a") == {
        "role": "assistant",
        "content": "[You | pre-interview note]\nplan",
    }
    assert post_interview.to_chat_message(viewer_id="a") == {
        "role": "assistant",
        "content": "[You | post-interview note]\nsummary",
    }


def test_strip_transcript_label_prefix_removes_known_labels():
    assert strip_transcript_label_prefix("[Current instruction]\nPROMOTE") == "PROMOTE"
    assert strip_transcript_label_prefix("[You | private note]: DO NOT PROMOTE") == "DO NOT PROMOTE"
