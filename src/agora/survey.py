import json
import re
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
_QUESTION_LINE_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:[*_`]+\s*)?Q(?P<number>\d+)\s*"
    r"(?:[.):]|\s+-)\s*(?P<rest>.*?)\s*(?:[*_`]+)?\s*$",
    re.IGNORECASE,
)
_JSON_STRING_PAIR_RE = re.compile(r'"(?P<key>Q\d+)"\s*:\s*"(?P<value>[^"]*)')


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

    try:
        survey_answers = _load_json_survey_answers(response_str)
    except json.JSONDecodeError:
        expected_keys = _expected_question_keys(question_groups)
        try:
            survey_answers = _parse_json_like_survey_object(
                response_str,
                expected_keys=expected_keys,
            )
        except ValueError as json_like_error:
            try:
                survey_answers = _parse_numbered_likert_response(
                    response_str,
                    expected_keys=expected_keys,
                )
            except ValueError as fallback_error:
                raise ValueError(
                    "Invalid survey response returned by model: expected a JSON "
                    f"object or numbered Likert answers ({json_like_error}; "
                    f"{fallback_error}). Raw response: {response_str}"
                ) from fallback_error

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


def _load_json_survey_answers(response_str: str) -> dict[str, Any]:
    try:
        survey_answers = json.loads(response_str)
    except json.JSONDecodeError:
        embedded = _extract_embedded_json_object(response_str)
        if embedded is None:
            raise
        survey_answers = embedded

    if not isinstance(survey_answers, dict):
        raise ValueError(
            "Survey response JSON must be an object mapping question IDs to answers; "
            f"got {type(survey_answers).__name__}."
        )
    return survey_answers


def _extract_embedded_json_object(response_str: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(response_str):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(response_str[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _parse_numbered_likert_response(
    response_str: str,
    *,
    expected_keys: list[str],
) -> dict[str, str]:
    answers: dict[str, str] = {}
    pending_key: str | None = None

    for raw_line in response_str.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = _QUESTION_LINE_RE.match(line)
        if match:
            key = f"Q{int(match.group('number'))}"
            answer = _extract_likert_answer(match.group("rest"))
            if answer is None:
                pending_key = key
                continue
            answers[key] = answer
            pending_key = None
            continue

        if pending_key is None:
            continue
        answer = _extract_likert_answer(line)
        if answer is not None:
            answers[pending_key] = answer
            pending_key = None

    if expected_keys:
        missing = [key for key in expected_keys if key not in answers]
        if missing:
            raise ValueError(
                "Numbered survey response missing answers for: "
                + ", ".join(missing)
            )
        return {key: answers[key] for key in expected_keys}

    if not answers:
        raise ValueError("No numbered Likert answers found in survey response")
    return dict(sorted(answers.items(), key=lambda item: _question_sort_key(item[0])))


def _parse_json_like_survey_object(
    response_str: str,
    *,
    expected_keys: list[str],
) -> dict[str, str]:
    pairs = {
        match.group("key"): match.group("value")
        for match in _JSON_STRING_PAIR_RE.finditer(response_str)
    }
    if not pairs:
        raise ValueError("No JSON-like question answers found in survey response")

    if expected_keys:
        missing = [key for key in expected_keys if key not in pairs]
        if missing:
            raise ValueError(
                "JSON-like survey response missing answers for: "
                + ", ".join(missing)
            )
        return {key: pairs[key] for key in expected_keys}

    return dict(sorted(pairs.items(), key=lambda item: _question_sort_key(item[0])))


def _expected_question_keys(question_groups: dict[str, str] | None) -> list[str]:
    if not question_groups:
        return []
    return [key for key, _group in _sorted_question_groups(question_groups)]


def _extract_likert_answer(candidate: str) -> str | None:
    cleaned = re.sub(r"[*_`]+", "", candidate).strip()
    for value in sorted(LIKERT_VALUES, key=len, reverse=True):
        answer_pattern = rf"^{re.escape(value)}(?=$|[\s.,;:!?\-\u2013\u2014])"
        if re.match(answer_pattern, cleaned, re.IGNORECASE):
            return value
    return None


def _response_values_for_group(group: str) -> list[str]:
    if group in VALID_SURVEY_GROUPS:
        return LIKERT_VALUES
    raise ValueError(f"Unknown survey group: {group}")


def _answer_to_score(answer: Any, *, group: str) -> int:
    if group in VALID_SURVEY_GROUPS:
        if not isinstance(answer, str):
            raise ValueError(
                "Survey answer values must be strings from the configured scale"
            )
        normalized = _extract_likert_answer(answer)
        if normalized is not None:
            return LIKERT_TO_SCORE[normalized]
        raise ValueError(f"Unknown survey answer: {answer}")
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
