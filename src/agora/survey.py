import json
from typing import Any

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

SURVEY_GROUP_DEFAULT = "default"
SURVEY_GROUP_DIRECT = "direct"
SURVEY_GROUP_SENTIMENT = "sentiment"
VALID_SURVEY_GROUPS = {
    SURVEY_GROUP_DEFAULT,
    SURVEY_GROUP_DIRECT,
    SURVEY_GROUP_SENTIMENT,
}


def normalize_survey_questions(
    question_config: Any,
    *,
    default_group: str,
) -> list[dict[str, str]]:
    """
    Normalize survey question config into ordered ``{"text", "group"}`` entries.
    """
    if question_config is None:
        return []
    if default_group not in VALID_SURVEY_GROUPS:
        raise ValueError(f"Unknown survey group: {default_group}")

    if isinstance(question_config, list):
        return [
            _normalize_survey_question_entry(entry, default_group=default_group)
            for entry in question_config
        ]

    if isinstance(question_config, dict):
        normalized: list[dict[str, str]] = []
        for group, entries in question_config.items():
            if group not in VALID_SURVEY_GROUPS:
                raise ValueError(f"Unknown survey group: {group}")
            if not isinstance(entries, list):
                raise ValueError(
                    f"Survey group '{group}' must contain a list of questions"
                )
            normalized.extend(
                _normalize_survey_question_entry(entry, default_group=group)
                for entry in entries
            )
        return normalized

    raise ValueError("Survey questions must be configured as a list or dict")


def merge_survey_question_configs(
    default_questions: Any,
    scenario_questions: Any,
) -> list[dict[str, str]]:
    """Merge prompt-level and scenario-level survey questions into one ordered list."""
    return normalize_survey_questions(
        default_questions, default_group=SURVEY_GROUP_DEFAULT
    ) + normalize_survey_questions(
        scenario_questions, default_group=SURVEY_GROUP_DIRECT
    )


def survey_question_texts(question_specs: list[dict[str, str]]) -> list[str]:
    """Extract plain text questions in order."""
    return [entry["text"] for entry in question_specs]


def survey_question_groups(question_specs: list[dict[str, str]]) -> dict[str, str]:
    """Map survey question ids (Q1, Q2, ...) to group names."""
    return {
        f"Q{index}": entry["group"] for index, entry in enumerate(question_specs, start=1)
    }


def _normalize_survey_question_entry(
    entry: Any,
    *,
    default_group: str,
) -> dict[str, str]:
    if isinstance(entry, str):
        return {"text": entry, "group": default_group}
    if isinstance(entry, dict):
        text = entry.get("text")
        group = entry.get("group", default_group)
        if group not in VALID_SURVEY_GROUPS:
            raise ValueError(f"Unknown survey group: {group}")
        if not isinstance(text, str) or not text:
            raise ValueError("Survey question entries must include non-empty text")
        return {"text": text, "group": group}
    raise ValueError("Survey question entries must be strings or dicts")


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

    numeric_scores = {q: LIKERT_TO_SCORE[a] for q, a in survey_answers.items()}

    return numeric_scores
