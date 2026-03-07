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

    def complete(
        self,
        *,
        messages,
        model,
        survey_questions=None,
        survey_question_groups=None,
    ):
        self.calls.append(
            {
                "messages": list(messages),
                "model": model,
                "survey_questions": survey_questions,
                "survey_question_groups": survey_question_groups,
            }
        )
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
        num_turns=1,
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
        num_turns=1,
        snapshot_path=snapshot,
        load_snapshot_flag=True,
        client_factory=second_factory,
    )
    assert len(resumed.history()) > len(agora.history())
    assert clients[-1].closed is True
    assert len(resumed_agents) == 2


def test_run_debate_session_requires_snapshot_path_when_loading():
    agent_configs = [
        {"name": "Alpha", "model": "demo", "self_role": "Alpha", "response_instruction": "Say"},
        {"name": "Beta", "model": "demo", "self_role": "Beta", "response_instruction": "Say"},
    ]
    clients = []

    def factory():
        client = CloseableStub([])
        clients.append(client)
        return client

    with pytest.raises(ValueError, match="snapshot_path is required"):
        run_debate_session(
            agent_configs,
            num_turns=0,
            load_snapshot_flag=True,
            snapshot_path=None,
            client_factory=factory,
        )

    assert clients[0].closed is True


def test_run_debate_session_errors_when_snapshot_file_missing(tmp_path):
    agent_configs = [
        {"name": "Alpha", "model": "demo", "self_role": "Alpha", "response_instruction": "Say"},
        {"name": "Beta", "model": "demo", "self_role": "Beta", "response_instruction": "Say"},
    ]
    missing_snapshot = tmp_path / "missing.json"
    clients = []

    def factory():
        client = CloseableStub([])
        clients.append(client)
        return client

    with pytest.raises(FileNotFoundError, match="Snapshot not found at"):
        run_debate_session(
            agent_configs,
            num_turns=0,
            load_snapshot_flag=True,
            snapshot_path=missing_snapshot,
            client_factory=factory,
        )

    assert clients[0].closed is True


def test_run_debate_session_resuming_replaces_old_post_interviews(tmp_path):
    agent_configs = [
        {
            "name": "Alpha",
            "model": "demo",
            "self_role": "Alpha",
            "response_instruction": "Say",
            "post_interview": {"instruction": "Post", "keep": True},
        },
        {
            "name": "Beta",
            "model": "demo",
            "self_role": "Beta",
            "response_instruction": "Say",
            "post_interview": {"instruction": "Post", "keep": True},
        },
    ]
    snapshot = tmp_path / "snap.json"
    clients = []

    def first_factory():
        client = CloseableStub(["alpha1", "beta1", "alpha post1", "beta post1"])
        clients.append(client)
        return client

    run_debate_session(
        agent_configs,
        num_turns=1,
        snapshot_path=snapshot,
        save_snapshot_flag=True,
        client_factory=first_factory,
    )

    def second_factory():
        client = CloseableStub(["alpha2", "beta2", "alpha post2", "beta post2"])
        clients.append(client)
        return client

    resumed, _ = run_debate_session(
        agent_configs,
        num_turns=1,
        snapshot_path=snapshot,
        load_snapshot_flag=True,
        client_factory=second_factory,
    )

    post_turns = [turn for turn in resumed.history() if turn.role == "post_interview"]
    assert len(post_turns) == 2
    assert {turn.metadata["turn_num"] for turn in post_turns} == {3}
    assert not any(
        msg["content"] in {"alpha post1", "beta post1"}
        for msg in clients[-1].calls[0]["messages"]
    )


def test_run_debate_session_loads_snapshot_without_generating_new_turns(tmp_path):
    agent_configs = [
        {"name": "Alpha", "model": "demo", "self_role": "Alpha", "response_instruction": "Say"},
        {"name": "Beta", "model": "demo", "self_role": "Beta", "response_instruction": "Say"},
    ]
    snapshot = tmp_path / "snap.json"

    first_clients = []

    def first_factory():
        client = CloseableStub(["alpha1", "beta1"])
        first_clients.append(client)
        return client

    first_agora, _ = run_debate_session(
        agent_configs,
        num_turns=1,
        snapshot_path=snapshot,
        save_snapshot_flag=True,
        client_factory=first_factory,
    )
    assert snapshot.exists()
    assert first_clients[0].closed is True

    second_clients = []

    def second_factory():
        client = CloseableStub([])
        second_clients.append(client)
        return client

    resumed, resumed_agents = run_debate_session(
        agent_configs,
        num_turns=0,
        snapshot_path=snapshot,
        load_snapshot_flag=True,
        client_factory=second_factory,
    )

    assert len(resumed.history()) == len(first_agora.history())
    assert len(resumed_agents) == 2
    assert second_clients[0].calls == []
    assert second_clients[0].closed is True


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
    agora.run(num_turns=1)

    rendered = format_history_for_agent(agent_a)
    assert "Turn 01" in rendered
    assert "Alpha" in rendered


def test_persona_builder_uses_catalogues():
    catalog = load_debate_construction("data/scenarios.json")

    configs = build_scenario_agent_configs(
        scenario_id="promotion_committee_max_divergence",
        catalog=catalog,
        model="shared-model",
        incentive_direction="positive",
        incentive_type="historical",
    )

    assert configs[0]["model"] == "shared-model"
    assert configs[1]["model"] == "shared-model"
    assert "Persona" in configs[0]["self_role"] or "persona" in configs[0]["self_role"].lower()
    assert "Additional scenario context" in configs[0]["self_role"]

    with pytest.raises(KeyError):
        build_scenario_agent_configs(
            scenario_id="missing",
            catalog=catalog,
            model="shared-model",
        )


def test_persona_builder_honors_keep_flags():
    catalog = load_debate_construction("data/scenarios.json")

    configs = build_scenario_agent_configs(
        scenario_id="promotion_committee_max_divergence",
        catalog=catalog,
        model="shared-model",
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
