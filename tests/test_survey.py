import pytest

from agora import survey


def test_build_likert_schema():
    schema = survey.build_likert_survey_schema(2)
    assert schema["schema"]["required"] == ["Q1", "Q2"]
    assert schema["schema"]["properties"]["Q1"]["enum"] == survey.LIKERT_VALUES


def test_parse_survey_response():
    payload = '{"Q1": "Agree", "Q2": "Neutral"}'
    result = survey.parse_survey_response_str(payload)
    assert result == {"Q1": 1, "Q2": 0}


def test_parse_survey_invalid_json():
    with pytest.raises(ValueError):
        survey.parse_survey_response_str("not json")


def test_parse_survey_invalid_answer():
    with pytest.raises(KeyError):
        survey.parse_survey_response_str('{"Q1": "maybe"}')


def test_merge_survey_question_configs_assigns_groups():
    merged = survey.merge_survey_question_configs(
        {"default": ["default q"]},
        {"direct": ["direct q"], "sentiment": ["sentiment q"]},
    )

    assert merged == [
        {"text": "default q", "group": survey.SURVEY_GROUP_DEFAULT},
        {"text": "direct q", "group": survey.SURVEY_GROUP_DIRECT},
        {"text": "sentiment q", "group": survey.SURVEY_GROUP_SENTIMENT},
    ]
    assert survey.survey_question_texts(merged) == [
        "default q",
        "direct q",
        "sentiment q",
    ]
    assert survey.survey_question_groups(merged) == {
        "Q1": survey.SURVEY_GROUP_DEFAULT,
        "Q2": survey.SURVEY_GROUP_DIRECT,
        "Q3": survey.SURVEY_GROUP_SENTIMENT,
    }


def test_normalize_survey_questions_rejects_unknown_group():
    with pytest.raises(ValueError, match="Unknown survey group"):
        survey.normalize_survey_questions({"invalid": ["q"]}, default_group="default")


def test_normalize_survey_questions_accepts_none():
    assert survey.normalize_survey_questions(None, default_group="default") == []


def test_normalize_survey_questions_rejects_unknown_default_group():
    with pytest.raises(ValueError, match="Unknown survey group"):
        survey.normalize_survey_questions([], default_group="invalid")


def test_normalize_survey_questions_rejects_non_list_group_entries():
    with pytest.raises(ValueError, match="must contain a list"):
        survey.normalize_survey_questions({"default": "q1"}, default_group="default")


def test_normalize_survey_questions_rejects_invalid_top_level_type():
    with pytest.raises(ValueError, match="list or dict"):
        survey.normalize_survey_questions("q1", default_group="default")


def test_normalize_survey_questions_rejects_invalid_entry_group():
    with pytest.raises(ValueError, match="Unknown survey group"):
        survey.normalize_survey_questions(
            [{"text": "q1", "group": "invalid"}],
            default_group="default",
        )


def test_normalize_survey_questions_rejects_missing_entry_text():
    with pytest.raises(ValueError, match="non-empty text"):
        survey.normalize_survey_questions(
            [{"group": "default"}],
            default_group="default",
        )


def test_normalize_survey_questions_rejects_invalid_entry_type():
    with pytest.raises(ValueError, match="strings or dicts"):
        survey.normalize_survey_questions([1], default_group="default")


def test_normalize_survey_questions_accepts_dict_entry_with_explicit_group():
    assert survey.normalize_survey_questions(
        [{"text": "q1", "group": "sentiment"}],
        default_group="default",
    ) == [{"text": "q1", "group": "sentiment"}]
