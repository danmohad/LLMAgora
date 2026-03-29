import pytest

from agora import survey


def test_build_likert_schema():
    schema = survey.build_likert_survey_schema(2)
    assert schema["schema"]["required"] == ["Q1", "Q2"]
    assert schema["schema"]["properties"]["Q1"]["enum"] == survey.LIKERT_VALUES


def test_build_grouped_likert_schema():
    schema = survey.build_survey_response_schema(
        {
            "Q1": survey.SURVEY_GROUP_DELIBERATIVE,
            "Q2": survey.SURVEY_GROUP_EVALUATIVE,
            "Q3": survey.SURVEY_GROUP_INCENTIVE,
        }
    )
    assert schema["schema"]["properties"]["Q1"]["enum"] == survey.LIKERT_VALUES
    assert schema["schema"]["properties"]["Q2"]["enum"] == survey.LIKERT_VALUES
    assert schema["schema"]["properties"]["Q3"]["enum"] == survey.LIKERT_VALUES


def test_build_grouped_likert_schema_rejects_unknown_group():
    with pytest.raises(ValueError, match="Unknown survey group"):
        survey.build_survey_response_schema({"Q1": "invalid"})


def test_parse_survey_response():
    payload = '{"Q1": "Agree", "Q2": "Neutral"}'
    result = survey.parse_survey_response_str(payload)
    assert result == {"Q1": 1, "Q2": 0}


def test_parse_survey_response_supports_named_groups():
    payload = '{"Q1": "Agree", "Q2": "Neutral", "Q3": "Disagree"}'
    result = survey.parse_survey_response_str(
        payload,
        {
            "Q1": survey.SURVEY_GROUP_DELIBERATIVE,
            "Q2": survey.SURVEY_GROUP_EVALUATIVE,
            "Q3": survey.SURVEY_GROUP_INCENTIVE,
        },
    )
    assert result == {"Q1": 1, "Q2": 0, "Q3": -1}


def test_parse_survey_invalid_json():
    with pytest.raises(ValueError):
        survey.parse_survey_response_str("not json")


def test_parse_survey_invalid_answer():
    with pytest.raises(KeyError):
        survey.parse_survey_response_str('{"Q1": "maybe"}')


def test_parse_survey_rejects_unknown_group_mapping():
    with pytest.raises(ValueError, match="Unknown survey group"):
        survey.parse_survey_response_str('{"Q1": "Agree"}', {"Q1": "invalid"})


def test_merge_survey_question_configs_assigns_groups():
    merged = survey.merge_survey_question_configs(
        {"deliberative": ["deliberative q"]},
        {"evaluative": ["evaluative q"], "incentive": ["incentive q"]},
    )

    assert merged == [
        {"text": "deliberative q", "group": survey.SURVEY_GROUP_DELIBERATIVE},
        {"text": "evaluative q", "group": survey.SURVEY_GROUP_EVALUATIVE},
        {"text": "incentive q", "group": survey.SURVEY_GROUP_INCENTIVE},
    ]
    assert survey.survey_question_texts(merged) == [
        "deliberative q",
        "evaluative q",
        "incentive q",
    ]
    assert survey.survey_question_groups(merged) == {
        "Q1": survey.SURVEY_GROUP_DELIBERATIVE,
        "Q2": survey.SURVEY_GROUP_EVALUATIVE,
        "Q3": survey.SURVEY_GROUP_INCENTIVE,
    }


def test_build_survey_scale_prompt_defaults_to_likert():
    prompt = survey.build_survey_scale_prompt({})
    assert "Likert scale" in prompt
    assert "Strongly disagree" in prompt


def test_build_survey_scale_prompt_rejects_unknown_group():
    with pytest.raises(ValueError, match="Unknown survey group"):
        survey.build_survey_scale_prompt({"Q1": "invalid"})


def test_build_survey_scale_prompt_uses_single_likert_scale_for_all_groups():
    prompt = survey.build_survey_scale_prompt(
        {
            "Q1": survey.SURVEY_GROUP_DELIBERATIVE,
            "Q2": survey.SURVEY_GROUP_INCENTIVE,
            "Q3": survey.SURVEY_GROUP_EVALUATIVE,
            "Q4": survey.SURVEY_GROUP_EVALUATIVE,
        }
    )
    assert "Likert scale" in prompt
    assert "Strongly agree" in prompt
    assert "No / Yes" not in prompt


def test_normalize_survey_questions_rejects_unknown_group():
    with pytest.raises(ValueError, match="Unknown survey group"):
        survey.normalize_survey_questions({"invalid": ["q"]}, default_group="deliberative")


def test_normalize_survey_questions_accepts_none():
    assert survey.normalize_survey_questions(None, default_group="deliberative") == []


def test_normalize_survey_questions_rejects_unknown_default_group():
    with pytest.raises(ValueError, match="Unknown survey group"):
        survey.normalize_survey_questions([], default_group="invalid")


def test_normalize_survey_questions_rejects_non_list_group_entries():
    with pytest.raises(ValueError, match="must contain a list"):
        survey.normalize_survey_questions({"deliberative": "q1"}, default_group="deliberative")


def test_normalize_survey_questions_rejects_invalid_top_level_type():
    with pytest.raises(ValueError, match="list or dict"):
        survey.normalize_survey_questions("q1", default_group="deliberative")


def test_normalize_survey_questions_rejects_invalid_entry_group():
    with pytest.raises(ValueError, match="Unknown survey group"):
        survey.normalize_survey_questions(
            [{"text": "q1", "group": "invalid"}],
            default_group="deliberative",
        )


def test_normalize_survey_questions_rejects_missing_entry_text():
    with pytest.raises(ValueError, match="non-empty text"):
        survey.normalize_survey_questions(
            [{"group": "deliberative"}],
            default_group="deliberative",
        )


def test_normalize_survey_questions_rejects_invalid_entry_type():
    with pytest.raises(ValueError, match="strings or dicts"):
        survey.normalize_survey_questions([1], default_group="deliberative")


def test_normalize_survey_questions_accepts_dict_entry_with_explicit_group():
    assert survey.normalize_survey_questions(
        [{"text": "q1", "group": "incentive"}],
        default_group="deliberative",
    ) == [{"text": "q1", "group": "incentive"}]
