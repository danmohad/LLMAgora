import json
import re

from .constant import LIKERT_TO_SCORE, LIKERT_VALUES

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

def parse_survey_response_str(
    response_str: str,
    agent_id: str | None = None,
) -> dict[str, int]:
    """
    Parse and validate a Likert survey response provided as a JSON string.
    """

    # --- Deserialize JSON ---
    
    text = response_str.strip()
    # Remove opening fence: ```json or ```
    text = re.sub(
        r"^\s*```(?:json)?\s*\n?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove closing fence
    text = re.sub(
        r"\n?\s*```\s*$",
        "",
        text,
    )
    print(text)
    try:
        response = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response: {e}") from e

    if not isinstance(response, dict):
        raise ValueError("Parsed JSON must be an object/dict")

    # --- Build stored record ---
    record =  {
            q: LIKERT_TO_SCORE[ans]
            for q, ans in response.items()
        }

    return record