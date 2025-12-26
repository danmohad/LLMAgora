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

def parse_survey_response_str(response_str: str) -> dict[str, int]:
    """
    Parse and validate a Likert survey response provided as a JSON string.
    """

    # --- Deserialize JSON ---
    try:
        survey_answers = json.loads(response_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON returned by model: {response_str}") from e
    
    numeric_scores = {
        q: LIKERT_TO_SCORE[a]
        for q, a in survey_answers.items()
    }
    
    return numeric_scores