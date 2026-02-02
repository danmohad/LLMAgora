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

    def complete(self, *, messages, model, survey_questions=None):
        return self._responses.pop(0)


def test_extract_instruction_rejects_invalid_type():
    with pytest.raises(ValueError):
        extract_instruction({"private_response": 123}, "private_response")


def test_extract_survey_instructions_handles_missing_and_present():
    questions, public_prompt, private_prompt, keep = extract_survey_instructions({})
    assert questions == []
    assert public_prompt is None
    assert private_prompt is None
    assert keep is False

    questions, public_prompt, private_prompt, keep = extract_survey_instructions(
        {
            "survey": {
                "survey_questions": ["q1", "q2"],
                "survey_public_prompt": "public",
                "survey_private_prompt": "private",
                "public_survey_keep": True,
            }
        }
    )
    assert questions == ["q1", "q2"]
    assert public_prompt == "public"
    assert private_prompt == "private"
    assert keep is True


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
    Agora([agent]).run(max_turns_per_agent=1)

    print_agent_histories([agent])
    output = capsys.readouterr().out
    assert "Full history visible" in output


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


def test_load_prompt_templates_adds_opening_instruction():
    catalog = {
        "prompt_sets": {
            "custom": {
                "base_prompt": "{persona}", 
                "debate_arena_prompt": "what's up",
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


def test_build_scenario_agent_configs_errors_and_custom_prompts():
    catalog = {
        "scenarios": [
            {
                "id": "s1",
                "question": {"controversial": "Q"},
                "side_1": {"actual_persona": "A", "perceived_persona": "PA"},
                "side_2": {"actual_persona": "B", "perceived_persona": "PB"},
            }
        ]
    }
    prompt_catalog = {
        "prompt_sets": {
            "custom": {
                "base_prompt": "{speaker_id}:{question}:{persona}",
                "debate_arena_prompt": "Arena: {debate_arena}",
                "perceived_prompt": "{perceived_persona}",
                "public_instruction": "pub",
                "private_instruction": "priv",
                "pre_interview_instruction": "pre",
                "post_interview_instruction": "post",
                "survey_public_prompt": "survey",
                "survey_private_prompt": "survey",
            }
        }
    }

    configs = build_scenario_agent_configs(
        scenario_id="s1",
        catalog=catalog,
        alpha_model="alpha",
        beta_model="beta",
        prompt_set="custom",
        prompt_catalog=prompt_catalog,
    )
    assert configs[0]["opening_instruction"] == "pub"

    with pytest.raises(KeyError):
        build_scenario_agent_configs(
            scenario_id="missing",
            catalog=catalog,
            alpha_model="alpha",
            beta_model="beta",
            prompt_set=DEFAULT_PROMPT_SET,
        )

    with pytest.raises(KeyError):
        build_scenario_agent_configs(
            scenario_id="s1",
            catalog={"scenarios": [{"id": "s1", "question": {"controversial": "Q"}, "side_1": {}}]},
            alpha_model="alpha",
            beta_model="beta",
            prompt_set=DEFAULT_PROMPT_SET,
        )

    questions_missing_variant = {
        "scenarios": [
            {
                "id": "s1",
                "question": {"id": "q1", "topic": "t"},
                "side_1": {"actual_persona": "A", "perceived_persona": "PA"},
                "side_2": {"actual_persona": "B", "perceived_persona": "PB"},
            }
        ]
    }
    with pytest.raises(KeyError):
        build_scenario_agent_configs(
            scenario_id="s1",
            question_variant="agreeable",
            catalog=questions_missing_variant,
            alpha_model="alpha",
            beta_model="beta",
            prompt_set="custom",
            prompt_catalog=prompt_catalog,
        )

    with pytest.raises(ValueError):
        build_scenario_agent_configs(
            scenario_id="s1",
            catalog=catalog,
            alpha_model="alpha",
            beta_model="beta",
            side_order="bad",
            prompt_set="custom",
            prompt_catalog=prompt_catalog,
        )

    swapped = build_scenario_agent_configs(
        scenario_id="s1",
        catalog=catalog,
        alpha_model="alpha",
        beta_model="beta",
        side_order="21",
        prompt_set="custom",
        prompt_catalog=prompt_catalog,
    )
    assert swapped[0]["self_role"].startswith("A:Q:B")

    configs = build_scenario_agent_configs(
        scenario_id="s1",
        catalog=catalog,
        alpha_model="alpha",
        beta_model="beta",
        debate_arena_override="Custom arena",
        prompt_set="custom",
        prompt_catalog=prompt_catalog,
    )
    assert "Custom arena" in configs[0]["self_role"]
