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

SURVEY_GROUP_DELIBERATIVE = "deliberative"
SURVEY_GROUP_EVALUATIVE = "evaluative"
SURVEY_GROUP_INCENTIVE = "incentive"
VALID_SURVEY_GROUPS = {
    SURVEY_GROUP_DELIBERATIVE,
    SURVEY_GROUP_EVALUATIVE,
    SURVEY_GROUP_INCENTIVE,
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
        default_questions, default_group=SURVEY_GROUP_DELIBERATIVE
    ) + normalize_survey_questions(
        scenario_questions, default_group=SURVEY_GROUP_EVALUATIVE
    )


def survey_question_texts(question_specs: list[dict[str, str]]) -> list[str]:
    """Extract plain text questions in order."""
    return [entry["text"] for entry in question_specs]


def survey_question_groups(question_specs: list[dict[str, str]]) -> dict[str, str]:
    """Map survey question ids (Q1, Q2, ...) to group names."""
    return {
        f"Q{index}": entry["group"] for index, entry in enumerate(question_specs, start=1)
    }


def survey_group_scale_label(group: str) -> str:
    """Return a short human-readable label for the response scale."""

    if group not in VALID_SURVEY_GROUPS:
        raise ValueError(f"Unknown survey group: {group}")
    return "Likert"


def build_survey_scale_prompt(question_groups: dict[str, str]) -> str:
    """Render survey scale instructions for the configured question groups."""

    if question_groups:
        for group in question_groups.values():
            if group not in VALID_SURVEY_GROUPS:
                raise ValueError(f"Unknown survey group: {group}")
    return _single_scale_prompt(LIKERT_VALUES, scale_name="Likert")


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
    question_groups = {
        f"Q{i}": SURVEY_GROUP_DELIBERATIVE for i in range(1, num_questions + 1)
    }
    return build_survey_response_schema(question_groups)


def build_survey_response_schema(question_groups: dict[str, str]):
    """Build a strict JSON Schema for a mixed-scale survey."""

    properties = {
        q_key: {
            "type": "string",
            "enum": _response_values_for_group(group),
        }
        for q_key, group in _sorted_question_groups(question_groups)
    }

    return {
        "name": f"survey_{len(properties)}_questions",
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
    question_groups: dict[str, str] | None = None,
) -> dict[str, int]:
    """
    Parse and validate a survey response provided as a JSON string.
    """

    # --- Deserialize JSON ---
    try:
        survey_answers = json.loads(response_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON returned by model: {response_str}") from e

    numeric_scores = {
        q: _answer_to_score(
            answer,
            group=question_groups.get(q, SURVEY_GROUP_DELIBERATIVE)
            if question_groups
            else SURVEY_GROUP_DELIBERATIVE,
        )
        for q, answer in survey_answers.items()
    }

    return numeric_scores


def _response_values_for_group(group: str) -> list[str]:
    if group in VALID_SURVEY_GROUPS:
        return LIKERT_VALUES
    raise ValueError(f"Unknown survey group: {group}")


def _answer_to_score(answer: str, *, group: str) -> int:
    if group in VALID_SURVEY_GROUPS:
        return LIKERT_TO_SCORE[answer]
    raise ValueError(f"Unknown survey group: {group}")


def _sorted_question_groups(question_groups: dict[str, str]) -> list[tuple[str, str]]:
    return sorted(
        question_groups.items(),
        key=lambda item: _question_sort_key(item[0]),
    )


def _question_sort_key(question_key: str) -> int:
    return int(question_key.removeprefix("Q"))


def _single_scale_prompt(values: list[str], *, scale_name: str) -> str:
    return "\n".join(
        [
            f"Use the following {scale_name} scale:",
            *[f"- {value}" for value in values],
        ]
    )
