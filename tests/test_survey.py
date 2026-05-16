import pytest

from agora import survey


def test_build_numbered_schema():
    schema = survey.build_numbered_survey_schema(2)
    assert schema["schema"]["required"] == ["Q1", "Q2"]
    assert schema["schema"]["properties"]["Q1"]["enum"] == survey.default_survey_scale_values()


def test_default_survey_prompt_helpers_read_prompt_catalog():
    assert survey.default_survey_question_prompt() == "Q{question_number}. {question_text}\n"
    assert survey.default_survey_scale_scores()["Agree"] == 1


def test_build_grouped_likert_schema():
    schema = survey.build_survey_response_schema(
        {
            "Q1": survey.SURVEY_GROUP_DELIBERATIVE,
            "Q2": survey.SURVEY_GROUP_EVALUATIVE,
            "Q3": survey.SURVEY_GROUP_INCENTIVE,
        }
    )
    scale_values = survey.default_survey_scale_values()
    assert schema["schema"]["properties"]["Q1"]["enum"] == scale_values
    assert schema["schema"]["properties"]["Q2"]["enum"] == scale_values
    assert schema["schema"]["properties"]["Q3"]["enum"] == scale_values


def test_build_grouped_likert_schema_rejects_unknown_group():
    with pytest.raises(ValueError, match="Unknown survey group"):
        survey.build_survey_response_schema({"Q1": "invalid"})


def test_parse_survey_response():
    payload = '{"Q1": "Agree", "Q2": "Neutral"}'
    result = survey.parse_survey_response_str(payload)
    assert result == {"Q1": 1, "Q2": 0}


def test_parse_survey_response_extracts_embedded_json_object():
    payload = 'Here is the survey:\n```json\n{"Q1": "Agree", "Q2": "Neutral"}\n```'
    result = survey.parse_survey_response_str(payload)
    assert result == {"Q1": 1, "Q2": 0}


def test_parse_survey_response_recovers_truncated_json_object():
    payload = '{"Q1":"Agree","Q2":"Neutral'
    result = survey.parse_survey_response_str(
        payload,
        {
            "Q1": survey.SURVEY_GROUP_DELIBERATIVE,
            "Q2": survey.SURVEY_GROUP_DELIBERATIVE,
        },
    )
    assert result == {"Q1": 1, "Q2": 0}


def test_parse_survey_response_recovers_truncated_json_object_without_expected_keys():
    payload = '{"Q2":"Neutral","Q1":"Agree'
    result = survey.parse_survey_response_str(payload)
    assert result == {"Q1": 1, "Q2": 0}


def test_parse_survey_response_rejects_truncated_json_with_missing_answer():
    payload = '{"Q1":"Agree","Q2'
    with pytest.raises(ValueError, match="missing answers for: Q2"):
        survey.parse_survey_response_str(
            payload,
            {
                "Q1": survey.SURVEY_GROUP_DELIBERATIVE,
                "Q2": survey.SURVEY_GROUP_DELIBERATIVE,
            },
        )


def test_parse_survey_response_recovers_numbered_markdown_answers():
    payload = """# Likert Scale Responses

**Q1. I agree with the other participant's overall position.**
Disagree - my assessment differs.

Q2. **Strongly agree** - their points deserve weight.
Q3. Neutral

---
Summary reflection that should be ignored.
"""
    result = survey.parse_survey_response_str(
        payload,
        {
            "Q1": survey.SURVEY_GROUP_DELIBERATIVE,
            "Q2": survey.SURVEY_GROUP_EVALUATIVE,
            "Q3": survey.SURVEY_GROUP_INCENTIVE,
        },
    )
    assert result == {"Q1": -1, "Q2": 2, "Q3": 0}


def test_parse_survey_response_recovers_numbered_answers_after_bad_brace():
    payload = "{not json}\nQ2. Agree\nQ1. Strongly disagree"
    result = survey.parse_survey_response_str(payload)
    assert result == {"Q1": -2, "Q2": 1}


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


def test_parse_survey_response_rejects_json_with_missing_expected_answer():
    with pytest.raises(ValueError, match="missing answers for: Q2"):
        survey.parse_survey_response_str(
            '{"Q1": "Agree"}',
            {
                "Q1": survey.SURVEY_GROUP_DELIBERATIVE,
                "Q2": survey.SURVEY_GROUP_DELIBERATIVE,
            },
        )


def test_parse_survey_response_rejects_embedded_json_with_missing_expected_answer():
    with pytest.raises(ValueError, match="missing answers for: Q2"):
        survey.parse_survey_response_str(
            '```json\n{"Q1": "Agree"}\n```',
            {
                "Q1": survey.SURVEY_GROUP_DELIBERATIVE,
                "Q2": survey.SURVEY_GROUP_DELIBERATIVE,
            },
        )


def test_parse_survey_invalid_json():
    with pytest.raises(ValueError, match="No numbered survey answers"):
        survey.parse_survey_response_str("not json")


def test_parse_survey_invalid_answer():
    with pytest.raises(ValueError, match="Unknown survey answer"):
        survey.parse_survey_response_str('{"Q1": "maybe"}')


def test_parse_survey_rejects_json_non_object():
    with pytest.raises(ValueError, match="must be an object"):
        survey.parse_survey_response_str("1.0")


def test_parse_survey_rejects_json_non_string_answer():
    with pytest.raises(ValueError, match="must be strings"):
        survey.parse_survey_response_str('{"Q1": 1}')


def test_parse_survey_rejects_numbered_response_with_missing_expected_answer():
    with pytest.raises(ValueError, match="missing answers for: Q2"):
        survey.parse_survey_response_str(
            "Q1. Agree",
            {
                "Q1": survey.SURVEY_GROUP_DELIBERATIVE,
                "Q2": survey.SURVEY_GROUP_DELIBERATIVE,
            },
        )


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


def test_survey_scale_can_be_configured_from_json_data():
    scale = {
        "name": "Binary",
        "values": [
            {"label": "No", "score": 0},
            {"label": "Yes", "score": 1},
        ],
    }
    prompt = survey.build_survey_scale_prompt(
        {},
        scale_config=scale,
    )
    schema = survey.build_survey_response_schema(
        {"Q1": survey.SURVEY_GROUP_DELIBERATIVE},
        scale_config=scale,
    )
    parsed = survey.parse_survey_response_str(
        '{"Q1": "Yes"}',
        {"Q1": survey.SURVEY_GROUP_DELIBERATIVE},
        scale_config=scale,
    )

    assert "Binary scale" in prompt
    assert "- Yes" in prompt
    assert schema["schema"]["properties"]["Q1"]["enum"] == ["No", "Yes"]
    assert parsed == {"Q1": 1}


def test_survey_scale_rejects_invalid_config():
    with pytest.raises(ValueError, match="survey_scale.values"):
        survey.build_survey_scale_prompt({}, scale_config={"name": "Bad", "values": []})


@pytest.mark.parametrize(
    "scale_config, message",
    [
        ([], "survey_scale must be a JSON object"),
        (
            {"name": "", "values": [{"label": "Yes", "score": 1}]},
            "survey_scale.name",
        ),
        (
            {"name": "Bad", "values": [1]},
            "entries must be JSON objects",
        ),
        (
            {"name": "Bad", "values": [{"label": "", "score": 1}]},
            "labels must be non-empty",
        ),
        (
            {"name": "Bad", "values": [{"label": "Yes", "score": "1"}]},
            "scores must be integers",
        ),
        (
            {
                "name": "Bad",
                "values": [
                    {"label": "Yes", "score": 1},
                    {"label": "yes", "score": 2},
                ],
            },
            "labels must be unique",
        ),
    ],
)
def test_survey_scale_rejects_malformed_entries(scale_config, message):
    with pytest.raises(ValueError, match=message):
        survey.build_survey_scale_prompt({}, scale_config=scale_config)


def test_default_prompt_set_requires_default_object(tmp_path, monkeypatch):
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_text('{"prompt_sets": {"default": []}}', encoding="utf-8")
    monkeypatch.setattr(survey, "_DEFAULT_PROMPTS_PATH", prompt_path)
    survey._default_prompt_set.cache_clear()

    with pytest.raises(KeyError, match="Default prompt set missing"):
        survey.default_survey_scale()

    survey._default_prompt_set.cache_clear()


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
