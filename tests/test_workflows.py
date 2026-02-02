import json

import pytest

from agora.agent import Agent
from agora.agora import Agora
from agora.workflows import (
    build_agents_from_configs,
    build_scenario_agent_configs,
    extract_instruction,
    format_history_for_agent,
    load_prompt_catalog,
    load_prompt_templates,
    load_debate_construction,
    run_debate_session,
)


class CloseableStub:
    """Stub client that records closes and reuses the pytest fixture responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.closed = False

    def complete(self, *, messages, model):
        self.calls.append({"messages": list(messages), "model": model})
        return self._responses.pop(0)

    def close(self):
        self.closed = True


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"private_response": None}, (None, True)),
        ({"private_response": "think"}, ("think", True)),
        ({"private_response": {"instruction": "think", "keep": False}}, ("think", False)),
    ],
)
def test_extract_instruction_normalizes_config(payload, expected):
    assert extract_instruction(payload, "private_response") == expected


def test_build_agents_from_configs_applies_prompts(stub_llm_factory):
    configs = [
        {
            "name": "Alpha",
            "model": "demo",
            "self_role": "You are Alpha",
            "perceived_nonself_roles": [{"name": "Beta", "role": "Beta role"}],
            "response_instruction": "Respond",
            "private_response": "Private",
            "pre_interview": {"instruction": "Pre", "keep": False},
            "post_interview": {"instruction": "Post", "keep": True},
        },
        {
            "name": "Beta",
            "model": "demo",
            "self_role": "You are Beta",
            "perceived_nonself_roles": [{"name": "Alpha", "role": "Alpha role"}],
            "response_instruction": "Reply",
        },
    ]
    agents = build_agents_from_configs(configs, stub_llm_factory(["hi", "there"]))
    assert len(agents) == 2
    assert agents[0].private_response_instruction == "Private"
    assert agents[0].pre_interview_instruction == "Pre"
    assert agents[0].post_interview_instruction == "Post"


def test_run_debate_session_handles_snapshots(tmp_path):
    agent_configs = [
        {"name": "Alpha", "model": "demo", "self_role": "Alpha", "response_instruction": "Say"},
        {"name": "Beta", "model": "demo", "self_role": "Beta", "response_instruction": "Say"},
    ]
    snapshot = tmp_path / "snap.json"

    clients = []

    def factory():
        client = CloseableStub(["alpha1", "beta1"])
        clients.append(client)
        return client

    agora, agents = run_debate_session(
        agent_configs,
        turns_per_agent=1,
        snapshot_path=snapshot,
        save_snapshot_flag=True,
        client_factory=factory,
    )
    assert snapshot.exists()
    assert clients[0].closed is True

    def second_factory():
        client = CloseableStub(["alpha2", "beta2"])
        clients.append(client)
        return client

    resumed, resumed_agents = run_debate_session(
        agent_configs,
        turns_per_agent=1,
        snapshot_path=snapshot,
        load_snapshot_flag=True,
        client_factory=second_factory,
    )
    assert len(resumed.history()) > len(agora.history())
    assert clients[-1].closed is True
    assert len(resumed_agents) == 2


def test_format_history_for_agent_renders_turns(stub_llm_factory):
    agent_a = Agent(
        name="Alpha",
        model="demo",
        llm_client=stub_llm_factory(["a1", "a2"]),
        response_instruction="public",
    )
    agent_b = Agent(
        name="Beta",
        model="demo",
        llm_client=stub_llm_factory(["b1", "b2"]),
        response_instruction="public",
    )
    agora = Agora([agent_a, agent_b])
    agora.run(max_turns_per_agent=1)

    rendered = format_history_for_agent(agent_a)
    assert "Turn 01" in rendered
    assert "Alpha" in rendered


def test_persona_builder_uses_catalogues():
    catalog = load_debate_construction("data/debate_construction.json")

    configs = build_scenario_agent_configs(
        scenario_id="hier_account_1",
        catalog=catalog,
        alpha_model="alpha-model",
        beta_model="beta-model",
    )

    assert configs[0]["model"] == "alpha-model"
    assert "Persona" in configs[0]["self_role"] or "persona" in configs[0]["self_role"].lower()

    with pytest.raises(KeyError):
        build_scenario_agent_configs(
            scenario_id="missing",
            catalog=catalog,
            alpha_model="alpha-model",
            beta_model="beta-model",
        )


def test_persona_builder_honors_keep_flags():
    catalog = load_debate_construction("data/debate_construction.json")

    configs = build_scenario_agent_configs(
        scenario_id="hier_account_1",
        catalog=catalog,
        alpha_model="alpha-model",
        beta_model="beta-model",
        private_response_keep=False,
        pre_interview_keep=True,
        post_interview_keep=True,
    )

    alpha_cfg = configs[0]
    assert alpha_cfg["private_response"]["keep"] is False
    assert alpha_cfg["pre_interview"]["keep"] is True
    assert alpha_cfg["post_interview"]["keep"] is True


def test_load_prompt_templates_reads_default_json():
    prompts = load_prompt_templates()

    assert "{persona}" in prompts["base_prompt"]
    assert "{perceived_persona}" in prompts["perceived_prompt"]


def test_load_prompt_catalog_supports_external_path(tmp_path):
    catalog = {"prompt_sets": {"custom": {"base_prompt": "{question}", "perceived_prompt": "{perceived_persona}", "public_instruction": "pub", "private_instruction": "priv", "pre_interview_instruction": "pre", "post_interview_instruction": "post"}}}
    prompt_file = tmp_path / "prompts.json"
    prompt_file.write_text(json.dumps(catalog))

    loaded_catalog = load_prompt_catalog(prompt_file)
    assert loaded_catalog["prompt_sets"]["custom"]["base_prompt"] == "{question}"
