import json

LIKERT_VALUES = [
    "Strongly disagree",
    "Disagree",
    "Neutral",
    "Agree",
    "Strongly agree",
]

LIKERT_TO_SCORE = {
    "Strongly disagree": -2,
    "Disagree": -1,
    "Neutral": 0,
    "Agree": 1,
    "Strongly agree": 2,
}

METRIC_SURVEYS = {
    "public": {
        "name": "public_metrics",
        "title": "Public response metrics",
        "fields": {
            "Public_STANCE_SHIFT": (-2, 2),
            "Public_CONFIDENCE": (0, 100),
            "Public_RESPECT": (0, 100),
            "Public_INTEREST_IN_OPPONENT_RESPONSE": (0, 100),
            "Public_TENSION_WITH_OPPONENT_RESPONSE": (0, 100),
        },
    },
    "off_record": {
        "name": "off_record_metrics",
        "title": "Off-record reflection metrics",
        "fields": {
            "Off_Record_STANCE_SHIFT": (-2, 2),
            "Off_Record_CONFIDENCE": (0, 100),
            "Off_Record_RESPECT": (0, 100),
            "Off_Record_INTEREST_IN_OPPONENT_RESPONSE": (0, 100),
            "Off_Record_TENSION_WITH_OPPONENT_RESPONSE": (0, 100),
        },
    },
}


def build_likert_survey_schema(num_questions: int):
    """
    Build a strict JSON Schema for a Likert survey with Q1..Q{num_q}.
    """
    properties = {
        f"Q{i}": {
            "type": "string",
            "enum": LIKERT_VALUES,
        }
        for i in range(1, num_questions + 1)
    }

    return {
        "name": f"likert_survey_{num_questions}_questions",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": properties,
            "required": list(properties.keys()),
            "additionalProperties": False,
        },
    }


def build_numeric_survey_schema(name: str, fields: dict[str, tuple[int, int]]):
    """Build a strict JSON Schema for numeric survey responses."""
    properties = {
        field: {
            "type": "integer",
            "minimum": minimum,
            "maximum": maximum,
        }
        for field, (minimum, maximum) in fields.items()
    }
    return {
        "name": name,
        "strict": True,
        "schema": {
            "type": "object",
            "properties": properties,
            "required": list(properties.keys()),
            "additionalProperties": False,
        },
    }


def parse_survey_response_str(response_str: str) -> dict[str, int]:
    """
    Parse and validate a Likert survey response provided as a JSON string.
    """

    # --- Deserialize JSON ---
    try:
        survey_answers = json.loads(response_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON returned by model: {response_str}") from e

    numeric_scores = {q: LIKERT_TO_SCORE[a] for q, a in survey_answers.items()}

    return numeric_scores


def parse_numeric_survey_response_str(
    response_str: str, *, fields: dict[str, tuple[int, int]]
) -> dict[str, int]:
    """Parse and validate a numeric survey response."""
    try:
        survey_answers = json.loads(response_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON returned by model: {response_str}") from e

    parsed: dict[str, int] = {}
    for key, (minimum, maximum) in fields.items():
        if key not in survey_answers:
            raise ValueError(f"Missing required survey key: {key}")
        value = survey_answers[key]
        if not isinstance(value, int):
            raise ValueError(f"Expected integer for {key}, got {value!r}")
        if value < minimum or value > maximum:
            raise ValueError(
                f"Value {value} for {key} out of range [{minimum}, {maximum}]"
            )
        parsed[key] = value

    return parsed
