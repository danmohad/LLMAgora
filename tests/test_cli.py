import json
from types import SimpleNamespace

import pytest

from agora import cli


def test_build_parser_registers_subcommands():
    parser = cli.build_parser()
    run_args = parser.parse_args(["run", "--config", "config.json", "--turns", "2"])
    assert run_args.func is cli._run_from_config

    persona_args = parser.parse_args([
        "persona",
        "--alpha-id",
        "alpha",
        "--beta-id",
        "beta",
        "--question-id",
        "question",
    ])
    assert persona_args.func is cli._run_persona


def test_load_agent_payload_requires_object(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(ValueError):
        cli._load_agent_payload(path)


def test_run_from_config_uses_agent_configs(tmp_path, monkeypatch):
    config = {
        "agent_configs": [
            {
                "name": "Alpha",
                "model": "demo",
                "self_role": "role",
                "response_instruction": "respond",
            }
        ],
        "turns_per_agent": 3,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    calls = {}

    def fake_run_debate_session(
        agent_configs,
        *,
        turns_per_agent,
        verbose,
        skip_first_agent_first_reflection,
        snapshot_path,
        load_snapshot_flag,
        save_snapshot_flag,
    ):
        calls["agent_configs"] = agent_configs
        calls["turns_per_agent"] = turns_per_agent
        calls["verbose"] = verbose
        calls["skip_first"] = skip_first_agent_first_reflection
        calls["snapshot_path"] = snapshot_path
        calls["load_snapshot"] = load_snapshot_flag
        calls["save_snapshot"] = save_snapshot_flag
        return object(), ["alpha"]

    def fake_print_agent_histories(agents):
        calls["printed"] = list(agents)

    monkeypatch.setattr(cli, "run_debate_session", fake_run_debate_session)
    monkeypatch.setattr(cli, "print_agent_histories", fake_print_agent_histories)

    args = SimpleNamespace(
        config=config_path,
        turns=None,
        verbose=True,
        skip_first_reflection=True,
        snapshot=tmp_path / "snap.json",
        load_snapshot=True,
        save_snapshot=True,
    )

    cli._run_from_config(args)

    assert calls["turns_per_agent"] == 3
    assert calls["printed"] == ["alpha"]
    assert calls["snapshot_path"] == args.snapshot
    assert calls["load_snapshot"] is True
    assert calls["save_snapshot"] is True


def test_run_from_config_persona_path(tmp_path, monkeypatch):
    config = {
        "alpha_persona_id": "alpha",
        "beta_persona_id": "beta",
        "question_id": "question",
        "turns_per_agent": 2,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    calls = {}

    def fake_load_persona_catalog(path):
        calls["personas"] = path
        return {"personas": {"alpha": {}, "beta": {}}}

    def fake_load_question_catalog(path):
        calls["questions"] = path
        return {"questions": {"question": {"question": "Q"}}}

    def fake_load_prompt_catalog(path):
        calls["prompts"] = path
        return {"prompt_sets": {"default": {}}}

    def fake_build_persona_agent_configs(**kwargs):
        calls["persona_args"] = kwargs
        return [
            {
                "name": "Alpha",
                "model": "demo",
                "self_role": "role",
                "response_instruction": "respond",
            }
        ]

    def fake_run_debate_session(agent_configs, *, turns_per_agent, **kwargs):
        calls["turns_per_agent"] = turns_per_agent
        calls["agent_configs"] = agent_configs
        return object(), ["alpha"]

    def fake_print_agent_histories(agents):
        calls["printed"] = list(agents)

    monkeypatch.setattr(cli, "load_persona_catalog", fake_load_persona_catalog)
    monkeypatch.setattr(cli, "load_question_catalog", fake_load_question_catalog)
    monkeypatch.setattr(cli, "load_prompt_catalog", fake_load_prompt_catalog)
    monkeypatch.setattr(cli, "build_persona_agent_configs", fake_build_persona_agent_configs)
    monkeypatch.setattr(cli, "run_debate_session", fake_run_debate_session)
    monkeypatch.setattr(cli, "print_agent_histories", fake_print_agent_histories)

    args = SimpleNamespace(
        config=config_path,
        turns=5,
        verbose=False,
        skip_first_reflection=False,
        snapshot=None,
        load_snapshot=False,
        save_snapshot=False,
    )

    cli._run_from_config(args)

    assert calls["turns_per_agent"] == 5
    assert calls["printed"] == ["alpha"]
    assert calls["persona_args"]["alpha_persona_id"] == "alpha"


def test_run_from_config_requires_turns(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "agent_configs": [
                    {
                        "name": "Alpha",
                        "model": "demo",
                        "self_role": "role",
                        "response_instruction": "respond",
                    }
                ]
            }
        )
    )

    args = SimpleNamespace(
        config=config_path,
        turns=None,
        verbose=False,
        skip_first_reflection=False,
        snapshot=None,
        load_snapshot=False,
        save_snapshot=False,
    )

    with pytest.raises(ValueError):
        cli._run_from_config(args)


def test_run_from_config_requires_persona_ids(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"agent_configs": []}))

    args = SimpleNamespace(
        config=config_path,
        turns=1,
        verbose=False,
        skip_first_reflection=False,
        snapshot=None,
        load_snapshot=False,
        save_snapshot=False,
    )

    with pytest.raises(ValueError):
        cli._run_from_config(args)


def test_run_persona_happy_path(monkeypatch):
    calls = {}

    def fake_load_persona_catalog(path):
        calls["personas"] = path
        return {"personas": {}}

    def fake_load_question_catalog(path):
        calls["questions"] = path
        return {"questions": {}}

    def fake_load_prompt_catalog(path):
        calls["prompts"] = path
        return {"prompt_sets": {"default": {}}}

    def fake_build_persona_agent_configs(**kwargs):
        calls["persona_args"] = kwargs
        return [
            {
                "name": "Alpha",
                "model": "demo",
                "self_role": "role",
                "response_instruction": "respond",
            }
        ]

    def fake_run_debate_session(agent_configs, *, turns_per_agent, **kwargs):
        calls["turns_per_agent"] = turns_per_agent
        calls["agent_configs"] = agent_configs
        return object(), ["alpha"]

    def fake_print_agent_histories(agents):
        calls["printed"] = list(agents)

    monkeypatch.setattr(cli, "load_persona_catalog", fake_load_persona_catalog)
    monkeypatch.setattr(cli, "load_question_catalog", fake_load_question_catalog)
    monkeypatch.setattr(cli, "load_prompt_catalog", fake_load_prompt_catalog)
    monkeypatch.setattr(cli, "build_persona_agent_configs", fake_build_persona_agent_configs)
    monkeypatch.setattr(cli, "run_debate_session", fake_run_debate_session)
    monkeypatch.setattr(cli, "print_agent_histories", fake_print_agent_histories)

    args = SimpleNamespace(
        alpha_id="alpha",
        beta_id="beta",
        question_id="question",
        alpha_model="alpha-model",
        beta_model="beta-model",
        turns=2,
        personas="data/personas.json",
        questions="data/questions.json",
        prompts=cli.DEFAULT_PROMPT_PATH,
        prompt_set=cli.DEFAULT_PROMPT_SET,
        snapshot=None,
        load_snapshot=False,
        save_snapshot=False,
        skip_first_reflection=False,
        verbose=False,
        keep_private_response=True,
        keep_pre_interview=False,
        keep_post_interview=False,
    )

    cli._run_persona(args)

    assert calls["turns_per_agent"] == 2
    assert calls["printed"] == ["alpha"]
    assert calls["persona_args"]["alpha_persona_id"] == "alpha"


def test_main_dispatches(monkeypatch):
    calls = {}

    class DummyParser:
        def parse_args(self, argv):
            calls["argv"] = argv
            return SimpleNamespace(func=lambda _: calls.setdefault("called", True))

    monkeypatch.setattr(cli, "build_parser", lambda: DummyParser())
    monkeypatch.setattr(cli, "load_dotenv", lambda: calls.setdefault("dotenv", True))

    cli.main(["run"])

    assert calls["dotenv"] is True
    assert calls["called"] is True
