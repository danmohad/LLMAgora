import json

import pytest

from agora.agent import Agent
from agora.agora import Agora
from agora.memory import MemoryTurn
from agora.workflows import (
    DEFAULT_PROMPT_SET,
    build_scenario_agent_configs,
    extract_instruction,
    extract_survey_instructions,
    format_history_for_agent,
    load_debate_construction,
    load_prompt_catalog,
    load_prompt_templates,
    print_agent_histories,
)


class QueueLLM:
    def __init__(self, responses):
        self._responses = list(responses)

    def complete(
        self,
        *,
        messages,
        model,
        survey_questions=None,
        survey_question_groups=None,
    ):
        return self._responses.pop(0)


def test_extract_instruction_rejects_invalid_type():
    with pytest.raises(ValueError):
        extract_instruction({"private_response": 123}, "private_response")


def test_extract_survey_instructions_handles_missing_and_present():
    (
        questions,
        question_groups,
        public_prompt,
        private_prompt,
        keep,
        keep_private,
        enable_public,
        enable_private,
    ) = extract_survey_instructions({})
    assert questions == []
    assert question_groups == {}
    assert public_prompt is None
    assert private_prompt is None
    assert keep is False
    assert keep_private is False
    assert enable_public is False
    assert enable_private is False

    (
        questions,
        question_groups,
        public_prompt,
        private_prompt,
        keep,
        keep_private,
        enable_public,
        enable_private,
    ) = extract_survey_instructions(
        {
            "survey": {
                "survey_questions": ["q1", "q2"],
                "survey_question_groups": {"Q1": "default", "Q2": "direct"},
                "survey_public_prompt": "public",
                "survey_private_prompt": "private",
                "public_survey_keep": True,
                "private_survey_keep": False,
                "enable_public_survey": False,
                "enable_private_survey": True,
            }
        }
    )
    assert questions == ["q1", "q2"]
    assert question_groups == {"Q1": "default", "Q2": "direct"}
    assert public_prompt == "public"
    assert private_prompt == "private"
    assert keep is True
    assert keep_private is False
    assert enable_public is False
    assert enable_private is True


def test_format_and_print_history_includes_exclusions(capsys):
    class DummyAgent:
        def __init__(self, turns):
            self._turns = turns

        def view_history(self):
            return self._turns

    rendered = format_history_for_agent(
        DummyAgent(
            [
                MemoryTurn(
                    turn_id=1,
                    speaker_id="a",
                    role="pre_interview",
                    private_reflection="pre",
                    metadata={"speaker_name": "Alpha"},
                    keep=False,
                ),
                MemoryTurn(
                    turn_id=2,
                    speaker_id="a",
                    role="reflection",
                    private_reflection="reflect",
                    metadata={"speaker_name": "Alpha"},
                    keep=False,
                ),
                MemoryTurn(
                    turn_id=3,
                    speaker_id="a",
                    role="assistant",
                    public_speech="public",
                    metadata={"speaker_name": "Alpha"},
                ),
            ]
        )
    )
    assert "(excluded)" in rendered
    assert "pre-interview" in rendered

    llm_client = QueueLLM(["public"])
    agent = Agent(
        name="Alpha",
        model="demo",
        llm_client=llm_client,
        response_instruction="respond",
    )
    beta = Agent(
        name="Beta",
        model="demo",
        llm_client=QueueLLM(["public beta"]),
        response_instruction="respond",
    )
    Agora([agent, beta]).run(num_turns=1)

    print_agent_histories([agent])
    output = capsys.readouterr().out
    assert "Full history visible" in output


def test_format_history_for_agent_includes_survey_roles():
    class DummyAgent:
        def __init__(self, turns):
            self._turns = turns

        def view_history(self):
            return self._turns

    rendered = format_history_for_agent(
        DummyAgent(
            [
                MemoryTurn(
                    turn_id=1,
                    speaker_id="a",
                    role="public_survey",
                    public_speech='{"Q1": 0}',
                    metadata={"speaker_name": "Alpha"},
                    keep=True,
                ),
                MemoryTurn(
                    turn_id=2,
                    speaker_id="a",
                    role="private_survey",
                    private_reflection='{"Q1": 1}',
                    metadata={"speaker_name": "Alpha"},
                    keep=False,
                ),
            ]
        )
    )
    assert "public survey" in rendered
    assert "private survey" in rendered


def test_load_prompt_catalog_missing_file(tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        load_prompt_catalog(missing)


def test_load_debate_construction_reads_json(tmp_path):
    payload = {"debates": [{"id": "d1", "label": "demo"}]}
    path = tmp_path / "debates.json"
    path.write_text(json.dumps(payload))
    assert load_debate_construction(path) == payload


def test_load_prompt_templates_missing_set():
    with pytest.raises(KeyError):
        load_prompt_templates("missing", prompt_catalog={"prompt_sets": {}})


def test_load_prompt_templates_missing_keys():
    catalog = {"prompt_sets": {"custom": {"base_prompt": "x", "perceived_prompt": "y", "public_instruction": "pub"}}}
    with pytest.raises(KeyError):
        load_prompt_templates("custom", prompt_catalog=catalog)


def test_load_prompt_templates_adds_opening_instruction_and_incentive_prompt():
    catalog = {
        "prompt_sets": {
            "custom": {
                "base_prompt": "{persona}",
                "perceived_prompt": "{perceived_persona}",
                "public_instruction": "public",
                "private_instruction": "private",
                "pre_interview_instruction": "pre",
                "post_interview_instruction": "post",
                "survey_public_prompt": "survey",
                "survey_private_prompt": "survey",
            }
        }
    }
    prompts = load_prompt_templates("custom", prompt_catalog=catalog)
    assert prompts["opening_instruction"] == "public"
    assert "{incentive}" in prompts["incentive_prompt"]


def _catalog_for_builder():
    return {
        "scenarios": [
            {
                "scenario_id": "s1",
                "question": {"topic": "Topic", "prompt": "Q"},
                "sides": {
                    "Side A": {
                        "id": "a",
                        "name": "Side A",
                        "actual_persona": "A persona",
                        "perceived_persona_base": "A sees B",
                    },
                    "Side B": {
                        "id": "b",
                        "name": "Side B",
                        "actual_persona": "B persona",
                        "perceived_persona_base": "B sees A",
                    },
                },
                "incentive_modules": {
                    "positive": {
                        "historical": {
                            "views": {"Side A": "A pos hist", "Side B": "B pos hist"}
                        },
                        "future": {
                            "views": {"Side A": "A pos future", "Side B": "B pos future"}
                        },
                    },
                    "negative": {
                        "historical": {
                            "views": {"Side A": "A neg hist", "Side B": "B neg hist"}
                        },
                        "future": {
                            "views": {"Side A": "A neg future", "Side B": "B neg future"}
                        },
                    },
                },
            }
        ]
    }


def _prompt_catalog_for_builder(private_instruction="priv"):
    return {
        "prompt_sets": {
            "custom": {
                "base_prompt": "{speaker_id}:{question}:{persona}",
                "perceived_prompt": "{perceived_persona}",
                "incentive_prompt": "|INC:{incentive}|",
                "public_instruction": "pub",
                "private_instruction": private_instruction,
                "pre_interview_instruction": "pre",
                "post_interview_instruction": "post",
                "survey_public_prompt": "survey",
                "survey_private_prompt": "survey",
            }
        }
    }


def test_build_scenario_agent_configs_applies_incentive_and_persona_fields():
    configs = build_scenario_agent_configs(
        scenario_id="s1",
        catalog=_catalog_for_builder(),
        alpha_model="alpha",
        beta_model="beta",
        incentive_direction="positive",
        incentive_type="historical",
        prompt_set="custom",
        prompt_catalog=_prompt_catalog_for_builder(),
    )
    assert configs[0]["self_role"].startswith("A:Q:A persona")
    assert "|INC:A pos hist|" in configs[0]["self_role"]
    assert "|INC:B pos hist|" in configs[1]["self_role"]
    assert configs[0]["perceived_nonself_roles"][0]["role"] == "B sees A"
    assert configs[0]["opening_instruction"] == "pub"


def test_build_scenario_agent_configs_accepts_optional_private_instruction_and_no_incentive():
    configs = build_scenario_agent_configs(
        scenario_id="s1",
        catalog=_catalog_for_builder(),
        alpha_model="alpha",
        beta_model="beta",
        prompt_set="custom",
        prompt_catalog=_prompt_catalog_for_builder(private_instruction=None),
        survey_questions=["q1"],
        survey_question_groups={"Q1": "direct"},
    )
    assert "|INC:" not in configs[0]["self_role"]
    assert configs[0]["private_response"]["instruction"] is None
    assert configs[0]["survey"]["survey_questions"] == ["q1"]
    assert configs[0]["survey"]["survey_question_groups"] == {"Q1": "direct"}


def test_build_scenario_agent_configs_rejects_unknown_scenario():
    with pytest.raises(KeyError):
        build_scenario_agent_configs(
            scenario_id="missing",
            catalog=_catalog_for_builder(),
            alpha_model="alpha",
            beta_model="beta",
            prompt_set=DEFAULT_PROMPT_SET,
        )


def test_build_scenario_agent_configs_requires_question_and_two_sides():
    bad_catalog_no_question = {
        "scenarios": [{"scenario_id": "s1", "sides": {}, "incentive_modules": {}}]
    }
    with pytest.raises(KeyError, match="required object: question"):
        build_scenario_agent_configs(
            scenario_id="s1",
            catalog=bad_catalog_no_question,
            alpha_model="alpha",
            beta_model="beta",
            prompt_set="custom",
            prompt_catalog=_prompt_catalog_for_builder(),
        )

    bad_catalog_no_prompt = {
        "scenarios": [
            {"scenario_id": "s1", "question": {"topic": "t"}, "sides": {}, "incentive_modules": {}}
        ]
    }
    with pytest.raises(KeyError, match="question.prompt"):
        build_scenario_agent_configs(
            scenario_id="s1",
            catalog=bad_catalog_no_prompt,
            alpha_model="alpha",
            beta_model="beta",
            prompt_set="custom",
            prompt_catalog=_prompt_catalog_for_builder(),
        )

    bad_catalog_one_side = {
        "scenarios": [
            {
                "scenario_id": "s1",
                "question": {"prompt": "Q"},
                "sides": {"Side A": {"actual_persona": "A", "perceived_persona_base": "X"}},
            }
        ]
    }
    with pytest.raises(KeyError, match="exactly two sides"):
        build_scenario_agent_configs(
            scenario_id="s1",
            catalog=bad_catalog_one_side,
            alpha_model="alpha",
            beta_model="beta",
            prompt_set="custom",
            prompt_catalog=_prompt_catalog_for_builder(),
        )


def test_build_scenario_agent_configs_rejects_invalid_incentive_selector():
    with pytest.raises(ValueError, match="incentive_direction"):
        build_scenario_agent_configs(
            scenario_id="s1",
            catalog=_catalog_for_builder(),
            alpha_model="alpha",
            beta_model="beta",
            incentive_direction="bad",
            prompt_set="custom",
            prompt_catalog=_prompt_catalog_for_builder(),
        )
    with pytest.raises(ValueError, match="incentive_type"):
        build_scenario_agent_configs(
            scenario_id="s1",
            catalog=_catalog_for_builder(),
            alpha_model="alpha",
            beta_model="beta",
            incentive_direction="positive",
            incentive_type="bad",
            prompt_set="custom",
            prompt_catalog=_prompt_catalog_for_builder(),
        )


def test_build_scenario_agent_configs_requires_incentive_module_parts():
    catalog_missing_modules = _catalog_for_builder()
    del catalog_missing_modules["scenarios"][0]["incentive_modules"]
    with pytest.raises(KeyError, match="incentive_modules"):
        build_scenario_agent_configs(
            scenario_id="s1",
            catalog=catalog_missing_modules,
            alpha_model="alpha",
            beta_model="beta",
            incentive_direction="positive",
            prompt_set="custom",
            prompt_catalog=_prompt_catalog_for_builder(),
        )

    catalog_missing_direction = _catalog_for_builder()
    del catalog_missing_direction["scenarios"][0]["incentive_modules"]["positive"]
    with pytest.raises(KeyError, match="module 'positive'"):
        build_scenario_agent_configs(
            scenario_id="s1",
            catalog=catalog_missing_direction,
            alpha_model="alpha",
            beta_model="beta",
            incentive_direction="positive",
            prompt_set="custom",
            prompt_catalog=_prompt_catalog_for_builder(),
        )

    catalog_missing_type = _catalog_for_builder()
    del catalog_missing_type["scenarios"][0]["incentive_modules"]["positive"]["future"]
    with pytest.raises(KeyError, match="missing type 'future'"):
        build_scenario_agent_configs(
            scenario_id="s1",
            catalog=catalog_missing_type,
            alpha_model="alpha",
            beta_model="beta",
            incentive_direction="positive",
            incentive_type="future",
            prompt_set="custom",
            prompt_catalog=_prompt_catalog_for_builder(),
        )

    catalog_missing_views = _catalog_for_builder()
    del catalog_missing_views["scenarios"][0]["incentive_modules"]["positive"]["historical"]["views"]
    with pytest.raises(KeyError, match="missing views"):
        build_scenario_agent_configs(
            scenario_id="s1",
            catalog=catalog_missing_views,
            alpha_model="alpha",
            beta_model="beta",
            incentive_direction="positive",
            incentive_type="historical",
            prompt_set="custom",
            prompt_catalog=_prompt_catalog_for_builder(),
        )

    catalog_missing_side_view = _catalog_for_builder()
    del catalog_missing_side_view["scenarios"][0]["incentive_modules"]["positive"]["historical"]["views"]["Side B"]
    with pytest.raises(KeyError, match="missing view for side 'Side B'"):
        build_scenario_agent_configs(
            scenario_id="s1",
            catalog=catalog_missing_side_view,
            alpha_model="alpha",
            beta_model="beta",
            incentive_direction="positive",
            incentive_type="historical",
            prompt_set="custom",
            prompt_catalog=_prompt_catalog_for_builder(),
        )
