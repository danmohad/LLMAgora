"""Survey configuration, prompt-scale rendering, and response parsing."""

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

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
_DEFAULT_PROMPTS_PATH = Path(__file__).resolve().parents[2] / "data" / "prompts.json"


@lru_cache(maxsize=1)
def _default_prompt_set() -> dict[str, Any]:
    catalog = json.loads(_DEFAULT_PROMPTS_PATH.read_text(encoding="utf-8"))
    prompt_sets = catalog.get("prompt_sets", catalog)
    payload = prompt_sets.get("default")
    if not isinstance(payload, dict):
        raise KeyError("Default prompt set missing from data/prompts.json")
    return payload


def default_survey_question_prompt() -> str:
    """Return the default question template from the prompt JSON catalog."""
    return str(_default_prompt_set()["survey_question_prompt"])


def default_survey_scale_prompt() -> str:
    """Return the default scale block template from the prompt JSON catalog."""
    return str(_default_prompt_set()["survey_scale_prompt"])


def default_survey_scale_value_prompt() -> str:
    """Return the default scale value template from the prompt JSON catalog."""
    return str(_default_prompt_set()["survey_scale_value_prompt"])


def default_survey_scale() -> dict[str, Any]:
    """Return the default survey scale config from the prompt JSON catalog."""
    return _normalize_survey_scale(_default_prompt_set()["survey_scale"])


def default_survey_scale_values() -> list[str]:
    """Return default survey response labels in prompt order."""
    return _survey_scale_values(default_survey_scale())


def default_survey_scale_scores() -> dict[str, int]:
    """Return the default label-to-score map."""
    return _survey_scale_scores(default_survey_scale())


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


def build_survey_scale_prompt(
    question_groups: dict[str, str],
    *,
    scale_prompt_template: str | None = None,
    scale_value_template: str | None = None,
    scale_config: Any | None = None,
) -> str:
    """Render survey scale instructions for the configured question groups."""

    if question_groups:
        for group in question_groups.values():
            if group not in VALID_SURVEY_GROUPS:
                raise ValueError(f"Unknown survey group: {group}")
    if scale_prompt_template is None:
        scale_prompt_template = default_survey_scale_prompt()
    if scale_value_template is None:
        scale_value_template = default_survey_scale_value_prompt()
    scale = default_survey_scale() if scale_config is None else scale_config
    normalized_scale = _normalize_survey_scale(scale)
    return _single_scale_prompt(
        _survey_scale_values(normalized_scale),
        scale_name=normalized_scale["name"],
        scale_prompt_template=scale_prompt_template,
        scale_value_template=scale_value_template,
    )


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


def build_numbered_survey_schema(
    num_questions: int,
    *,
    scale_config: Any | None = None,
):
    """Build a strict JSON Schema for Q1..Q{num_questions} survey answers."""

    question_groups = {
        f"Q{i}": SURVEY_GROUP_DELIBERATIVE for i in range(1, num_questions + 1)
    }
    return build_survey_response_schema(question_groups, scale_config=scale_config)


def build_survey_response_schema(
    question_groups: dict[str, str],
    *,
    scale_config: Any | None = None,
):
    """Build a strict JSON Schema for a mixed-scale survey."""

    scale = default_survey_scale() if scale_config is None else scale_config
    scale_values = _survey_scale_values(_normalize_survey_scale(scale))
    properties = {
        q_key: {
            "type": "string",
            "enum": _response_values_for_group(group, scale_values=scale_values),
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
    *,
    scale_config: Any | None = None,
) -> dict[str, int]:
    """
    Parse and validate a survey response provided as a JSON string.
    """
    expected_keys = _expected_question_keys(question_groups)
    scale = _normalize_survey_scale(
        default_survey_scale() if scale_config is None else scale_config
    )
    scale_values = _survey_scale_values(scale)
    scale_scores = _survey_scale_scores(scale)

    try:
        survey_answers = _require_expected_survey_answers(
            _load_json_survey_answers(response_str),
            expected_keys=expected_keys,
            response_kind="Survey response JSON",
        )
    except json.JSONDecodeError:
        try:
            survey_answers = _parse_json_like_survey_object(
                response_str,
                expected_keys=expected_keys,
            )
        except ValueError as json_like_error:
            try:
                survey_answers = _parse_numbered_survey_response(
                    response_str,
                    expected_keys=expected_keys,
                    scale_values=scale_values,
                )
            except ValueError as fallback_error:
                raise ValueError(
                    "Invalid survey response returned by model: expected a JSON "
                    f"object or numbered survey answers ({json_like_error}; "
                    f"{fallback_error}). Raw response: {response_str}"
                ) from fallback_error

    numeric_scores = {
        q: _answer_to_score(
            answer,
            group=question_groups.get(q, SURVEY_GROUP_DELIBERATIVE)
            if question_groups
            else SURVEY_GROUP_DELIBERATIVE,
            scale_values=scale_values,
            scale_scores=scale_scores,
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


def _parse_numbered_survey_response(
    response_str: str,
    *,
    expected_keys: list[str],
    scale_values: list[str],
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
            answer = _extract_scale_answer(match.group("rest"), scale_values)
            if answer is None:
                pending_key = key
                continue
            answers[key] = answer
            pending_key = None
            continue

        if pending_key is None:
            continue
        answer = _extract_scale_answer(line, scale_values)
        if answer is not None:
            answers[pending_key] = answer
            pending_key = None

    if expected_keys:
        return _require_expected_survey_answers(
            answers,
            expected_keys=expected_keys,
            response_kind="Numbered survey response",
        )

    if not answers:
        raise ValueError("No numbered survey answers found in survey response")
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
        return _require_expected_survey_answers(
            pairs,
            expected_keys=expected_keys,
            response_kind="JSON-like survey response",
        )

    return dict(sorted(pairs.items(), key=lambda item: _question_sort_key(item[0])))


def _expected_question_keys(question_groups: dict[str, str] | None) -> list[str]:
    if not question_groups:
        return []
    return [key for key, _group in _sorted_question_groups(question_groups)]


def _require_expected_survey_answers(
    survey_answers: dict[str, Any],
    *,
    expected_keys: list[str],
    response_kind: str,
) -> dict[str, Any]:
    if not expected_keys:
        return survey_answers
    missing = [key for key in expected_keys if key not in survey_answers]
    if missing:
        raise ValueError(f"{response_kind} missing answers for: " + ", ".join(missing))
    return {key: survey_answers[key] for key in expected_keys}


def _extract_scale_answer(candidate: str, scale_values: list[str]) -> str | None:
    cleaned = re.sub(r"[*_`]+", "", candidate).strip()
    for value in sorted(scale_values, key=len, reverse=True):
        answer_pattern = rf"^{re.escape(value)}(?=$|[\s.,;:!?\-\u2013\u2014])"
        if re.match(answer_pattern, cleaned, re.IGNORECASE):
            return value
    return None


def _response_values_for_group(group: str, *, scale_values: list[str]) -> list[str]:
    if group in VALID_SURVEY_GROUPS:
        return list(scale_values)
    raise ValueError(f"Unknown survey group: {group}")


def _answer_to_score(
    answer: Any,
    *,
    group: str,
    scale_values: list[str],
    scale_scores: dict[str, int],
) -> int:
    if group in VALID_SURVEY_GROUPS:
        if not isinstance(answer, str):
            raise ValueError(
                "Survey answer values must be strings from the configured scale"
            )
        normalized = _extract_scale_answer(answer, scale_values)
        if normalized is not None:
            return scale_scores[normalized]
        raise ValueError(f"Unknown survey answer: {answer}")
    raise ValueError(f"Unknown survey group: {group}")


def _sorted_question_groups(question_groups: dict[str, str]) -> list[tuple[str, str]]:
    return sorted(
        question_groups.items(),
        key=lambda item: _question_sort_key(item[0]),
    )


def _question_sort_key(question_key: str) -> int:
    return int(question_key.removeprefix("Q"))


def _normalize_survey_scale(scale_config: Any) -> dict[str, Any]:
    if not isinstance(scale_config, dict):
        raise ValueError("survey_scale must be a JSON object")
    name = scale_config.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("survey_scale.name must be a non-empty string")
    values = scale_config.get("values")
    if not isinstance(values, list) or not values:
        raise ValueError("survey_scale.values must be a non-empty list")

    normalized_values: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    for entry in values:
        if not isinstance(entry, dict):
            raise ValueError("survey_scale.values entries must be JSON objects")
        label = entry.get("label")
        score = entry.get("score")
        if not isinstance(label, str) or not label.strip():
            raise ValueError("survey_scale value labels must be non-empty strings")
        if not isinstance(score, int):
            raise ValueError("survey_scale value scores must be integers")
        label_key = label.casefold()
        if label_key in seen_labels:
            raise ValueError("survey_scale value labels must be unique")
        seen_labels.add(label_key)
        normalized_values.append({"label": label.strip(), "score": score})

    return {"name": name.strip(), "values": normalized_values}


def _survey_scale_values(scale_config: dict[str, Any]) -> list[str]:
    return [entry["label"] for entry in scale_config["values"]]


def _survey_scale_scores(scale_config: dict[str, Any]) -> dict[str, int]:
    return {entry["label"]: entry["score"] for entry in scale_config["values"]}


def _single_scale_prompt(
    values: list[str],
    *,
    scale_name: str,
    scale_prompt_template: str,
    scale_value_template: str,
) -> str:
    scale_values = "\n".join(
        scale_value_template.format(value=value) for value in values
    )
    return scale_prompt_template.format(
        scale_name=scale_name,
        scale_values=scale_values,
    )
