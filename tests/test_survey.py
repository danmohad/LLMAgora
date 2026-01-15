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
