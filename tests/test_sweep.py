import io
import json
import os
import threading
from pathlib import Path

import pytest

from agora import sweep


def _master_payload(tmp_path, **overrides):
    payload = {
        "sweep_root": str(tmp_path / "sweeps" / "demo"),
        "max_parallel_jobs": 1,
        "stop_on_error": False,
        "notes": "test sweep",
        "base": {
            "scenario_id": "s1",
        },
        "sweep": {
            "incentive_type": ["historical", "future"],
        },
    }
    payload.update(overrides)
    return payload


def _write_master(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class _Stream(io.StringIO):
    def __init__(self, *, tty: bool):
        super().__init__()
        self._tty = tty

    def isatty(self):
        return self._tty


def test_jsonc_helpers_parse_comments_and_strings():
    text = (
        '{\n'
        '  // line comment\n'
        '  "message": "keep // inside string",\n'
        '  /* block comment */\n'
        '  "value": 3\n'
        '}\n'
    )
    payload = sweep._load_jsonc_object(text)
    assert payload == {"message": "keep // inside string", "value": 3}
    stripped = sweep._strip_jsonc_comments(
        '{\n  "escaped": "backslash\\\\quote\\\"",\n  /* multi\n     line */\n  "done": true\n}\n'
    )
    assert '"escaped": "backslash\\\\quote\\\""' in stripped
    assert "\n\n" in stripped

    with pytest.raises(ValueError):
        sweep._strip_jsonc_comments("/* bad")

    with pytest.raises(ValueError):
        sweep._load_jsonc_object("[]")


def test_load_json_object_rejects_non_object(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError):
        sweep._load_json_object(path)


def test_normalize_master_config_rejects_invalid_shapes(tmp_path):
    normalized = sweep._normalize_master_config(
        {
            "sweep_root": str(tmp_path / "single"),
            "base": {"scenario_id": "s1"},
            "sweep": None,
        }
    )
    assert normalized["number_of_repeats"] == 1
    assert normalized["sweep"] == {}
    assert sweep._expand_cases(normalized)[0]["label"] == "base"

    with pytest.raises(ValueError):
        sweep._normalize_master_config({"base": {}})
    with pytest.raises(ValueError):
        sweep._normalize_master_config({"sweep_root": str(tmp_path / "x")})
    with pytest.raises(ValueError):
        sweep._normalize_master_config({"sweep_root": str(tmp_path / "x"), "base": {}, "extra": 1})
    with pytest.raises(ValueError):
        sweep._normalize_master_config({"sweep_root": str(tmp_path / "x"), "base": [], "sweep": {}})
    with pytest.raises(ValueError):
        sweep._normalize_master_config({"sweep_root": str(tmp_path / "x"), "base": {}, "sweep": []})
    with pytest.raises(ValueError):
        sweep._normalize_master_config({"sweep_root": str(tmp_path / "x"), "base": {}, "notes": 1})
    with pytest.raises(ValueError):
        sweep._normalize_master_config(
            {"sweep_root": str(tmp_path / "x"), "base": {}, "max_parallel_jobs": 0}
        )
    with pytest.raises(ValueError):
        sweep._normalize_master_config(
            {"sweep_root": str(tmp_path / "x"), "base": {}, "stop_on_error": "yes"}
        )
    with pytest.raises(ValueError):
        sweep._normalize_master_config(
            {"sweep_root": str(tmp_path / "x"), "base": {}, "number_of_repeats": 0}
        )
    with pytest.raises(ValueError):
        sweep._normalize_master_config(
            {"sweep_root": str(tmp_path / "x"), "base": {}, "number_of_repeats": "2"}
        )
    with pytest.raises(ValueError):
        sweep._normalize_master_config(
            {
                "sweep_root": str(tmp_path / "x"),
                "base": {"not_a_field": 1},
                "sweep": {},
            }
        )
    with pytest.raises(ValueError):
        sweep._normalize_master_config(
            {
                "sweep_root": str(tmp_path / "x"),
                "base": {"scenario_id": "s1", "outputs_root": "oops"},
                "sweep": {},
            }
        )
    with pytest.raises(ValueError):
        sweep._normalize_master_config(
            {
                "sweep_root": str(tmp_path / "x"),
                "base": {"scenario_id": "s1"},
                "sweep": {"incentive_type": "future"},
            }
        )
    with pytest.raises(ValueError):
        sweep._normalize_master_config(
            {
                "sweep_root": str(tmp_path / "x"),
                "base": {"scenario_id": "s1"},
                "sweep": {"incentive_type": []},
            }
        )
    with pytest.raises(ValueError):
        sweep._normalize_master_config(
            {
                "sweep_root": str(tmp_path / "x"),
                "base": {"scenario_id": "s1"},
                "sweep": {"incentive_type": ["future", "future"]},
            }
        )


def test_load_sweep_config_and_expand_cases(tmp_path):
    master_path = tmp_path / "master.jsonc"
    sweep_root = tmp_path / "sweeps" / "demo"
    text = (
        "{\n"
        "  // config with comments\n"
        f'  "sweep_root": "{sweep_root}",\n'
        '  "number_of_repeats": 2,\n'
        '  "base": {"scenario_id": "s1"},\n'
        '  "sweep": {\n'
        '    "subturn_event_order": [\n'
        '      ["public_utterance"],\n'
        '      ["public_utterance", "private_utterance"]\n'
        "    ],\n"
        '    "semantic_analysis_metrics": [[], ["self_consistency"]]\n'
        "  }\n"
        "}\n"
    )
    master_path.write_text(text, encoding="utf-8")

    master, raw_text = sweep.load_sweep_config(master_path)
    cases = sweep._expand_cases(master)

    assert "// config with comments" in raw_text
    assert len(cases) == 8
    assert cases[0]["label"].endswith("(repeat 1/2)")
    assert len(cases[0]["case_id"]) == 12
    assert cases[0]["case_id"].isalnum()
    assert cases[0]["repeat_number"] == 1
    assert cases[0]["repeat_count"] == 2
    assert cases[1]["repeat_number"] == 2
    assert cases[0]["case_id"] != cases[1]["case_id"]
    assert cases[0]["config_fingerprint"] == cases[1]["config_fingerprint"]
    assert Path(cases[0]["config_payload"]["output_dir"]).is_absolute()


def test_load_sweep_config_resolves_relative_path_fields(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    master_path = config_dir / "master.jsonc"
    master_path.write_text(
        (
            "{\n"
            '  "sweep_root": "../sweeps/demo",\n'
            '  "base": {\n'
            '    "scenario_id": "s1",\n'
            '    "catalog_path": "../data/catalog.json"\n'
            "  },\n"
            '  "sweep": {\n'
            '    "prompts_path": ["../data/prompts-a.json", "../data/prompts-b.json"]\n'
            "  }\n"
            "}\n"
        ),
        encoding="utf-8",
    )

    master, _ = sweep.load_sweep_config(master_path)

    assert master["sweep_root"] == tmp_path / "sweeps" / "demo"
    assert master["base"]["catalog_path"] == tmp_path / "data" / "catalog.json"
    assert master["sweep"]["prompts_path"] == [
        tmp_path / "data" / "prompts-a.json",
        tmp_path / "data" / "prompts-b.json",
    ]


def test_generate_sweep_preserves_explicit_nulls_and_avoids_unspecified_defaults(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    master_path = config_dir / "master.jsonc"
    master_path.write_text(
        (
            "{\n"
            '  "sweep_root": "../sweeps/demo",\n'
            '  "base": {\n'
            '    "scenario_id": "s1",\n'
            '    "catalog_path": "../data/catalog.json",\n'
            '    "prompts_path": "../data/prompts.json",\n'
            '    "semantic_similarity_method": null\n'
            "  },\n"
            '  "sweep": {\n'
            '    "incentive_type": ["historical"]\n'
            "  }\n"
            "}\n"
        ),
        encoding="utf-8",
    )

    manifest = sweep.generate_sweep(master_path)
    root = Path(manifest["sweep_root"])
    case_id = manifest["cases"][0]["case_id"]
    config_payload = json.loads(
        (root / "cases" / case_id / "config.json").read_text(encoding="utf-8")
    )

    assert config_payload["catalog_path"] == str(tmp_path / "data" / "catalog.json")
    assert config_payload["prompts_path"] == str(tmp_path / "data" / "prompts.json")
    assert config_payload["semantic_similarity_method"] is None
    assert config_payload["incentive_type"] == "historical"
    assert Path(config_payload["output_dir"]).is_absolute()
    assert "semantic_similarity_model" not in config_payload
    assert "subturn_event_order" not in config_payload


def test_expand_cases_rejects_duplicates_and_hash_collisions(tmp_path, monkeypatch):
    master = sweep._normalize_master_config(
        {
            "sweep_root": str(tmp_path / "sweeps" / "demo"),
            "base": {"scenario_id": "s1"},
            "sweep": {"prompt_set": ["default", "default-alt"]},
        }
    )
    duplicate_master = dict(master)
    duplicate_master["sweep"] = {"prompt_set": ["default", "default"]}
    with pytest.raises(ValueError):
        sweep._expand_cases(duplicate_master)

    digests = iter(
        [
            type("Digest", (), {"hexdigest": lambda self: "aaaaaaaaaaaa" + "0" * 52})(),
            type("Digest", (), {"hexdigest": lambda self: "aaaaaaaaaaaa" + "1" * 52})(),
        ]
    )
    monkeypatch.setattr(sweep.hashlib, "sha256", lambda _value: next(digests))

    with pytest.raises(ValueError):
        sweep._expand_cases(master)


def test_generate_sweep_writes_tree_and_force_replaces(tmp_path):
    master_path = tmp_path / "master.jsonc"
    payload = _master_payload(tmp_path, number_of_repeats=3)
    _write_master(master_path, payload)

    manifest = sweep.generate_sweep(master_path)
    root = Path(manifest["sweep_root"])
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    assert manifest["number_of_repeats"] == 3
    assert manifest["total_cases"] == 6
    assert status["aggregate_counts"]["pending"] == 6

    first_case = manifest["cases"][0]["case_id"]
    assert manifest["cases"][0]["repeat_number"] == 1
    assert manifest["cases"][0]["repeat_count"] == 3
    assert (root / "cases" / first_case / "config.json").exists()
    (root / "junk.txt").write_text("remove me", encoding="utf-8")

    with pytest.raises(ValueError):
        sweep.generate_sweep(master_path)

    sweep.generate_sweep(master_path, force=True)
    assert not (root / "junk.txt").exists()


def test_generate_sweep_fails_before_writing_on_invalid_case(tmp_path):
    master_path = tmp_path / "bad_master.jsonc"
    payload = _master_payload(tmp_path)
    payload["base"] = {"num_turns": 1}
    _write_master(master_path, payload)

    with pytest.raises(ValueError):
        sweep.generate_sweep(master_path)

    assert not (tmp_path / "sweeps" / "demo").exists()


def test_generate_sweep_handles_existing_file_root(tmp_path):
    master_path = tmp_path / "master.jsonc"
    root_file = tmp_path / "occupied"
    root_file.write_text("x", encoding="utf-8")
    payload = _master_payload(tmp_path, sweep_root=str(root_file))
    _write_master(master_path, payload)

    with pytest.raises(ValueError):
        sweep.generate_sweep(master_path)

    sweep.generate_sweep(master_path, force=True)
    assert root_file.is_dir()


def test_status_helpers_and_selection(tmp_path):
    manifest = {
        "total_cases": 3,
        "cases": [
            {"case_id": "aaa111aaa111", "case_dir": "cases/aaa111aaa111", "label": "a"},
            {"case_id": "bbb222bbb222", "case_dir": "cases/bbb222bbb222", "label": "b"},
            {"case_id": "ccc333ccc333", "case_dir": "cases/ccc333ccc333", "label": "c"},
        ],
        "runner_defaults": {"max_parallel_jobs": 2, "stop_on_error": False},
    }
    status = {
        "aggregate_counts": {},
        "run_session": {"is_active": False},
        "cases": {
            "aaa111aaa111": {
                "last_status": "succeeded",
                "attempt_count": 1,
                "last_return_code": 0,
                "last_started_at": None,
                "last_finished_at": None,
                "last_duration_seconds": None,
                "log_path": "cases/aaa111aaa111/run.log",
                "last_error_summary": None,
            },
            "bbb222bbb222": {
                "last_status": "failed",
                "attempt_count": 1,
                "last_return_code": 1,
                "last_started_at": None,
                "last_finished_at": None,
                "last_duration_seconds": None,
                "log_path": "cases/bbb222bbb222/run.log",
                "last_error_summary": "bad",
            },
            "ccc333ccc333": {
                "last_status": "pending",
                "attempt_count": 0,
                "last_return_code": None,
                "last_started_at": None,
                "last_finished_at": None,
                "last_duration_seconds": None,
                "log_path": "cases/ccc333ccc333/run.log",
                "last_error_summary": None,
            },
        },
    }

    counts = sweep._status_counts(status)
    assert counts["succeeded"] == 1
    assert counts["failed"] == 1
    assert counts["pending"] == 1
    assert sweep._render_duration(None) == "-"
    assert sweep._duration_seconds(None, None) is None
    assert sweep._truncate("abcdef", 0) == ""
    assert sweep._truncate("abcdef", 3) == "abc"
    assert sweep._truncate("abcdef", 4) == "a..."
    assert sweep._progress_bar(0, 0, 10) == "[" + ("-" * 10) + "]"
    assert sweep._progress_bar(1, 2, 10).startswith("[")

    snapshot = sweep._render_status_snapshot(tmp_path, manifest, status)
    assert "Next: agora sweep run --root" in snapshot
    assert "Failed: bbb222bbb222" in snapshot
    failed_only_status = dict(status)
    failed_only_status["aggregate_counts"] = {
        "pending": 0,
        "queued": 0,
        "running": 0,
        "succeeded": 2,
        "failed": 1,
        "skipped": 0,
        "interrupted": 0,
    }
    failed_only_status["cases"] = {
        **status["cases"],
        "ccc333ccc333": {
            **status["cases"]["ccc333ccc333"],
            "last_status": "succeeded",
        },
    }
    assert "--mode failed" in sweep._render_status_snapshot(tmp_path, manifest, failed_only_status)
    complete_status = dict(failed_only_status)
    complete_status["aggregate_counts"] = {
        "pending": 0,
        "queued": 0,
        "running": 0,
        "succeeded": 3,
        "failed": 0,
        "skipped": 0,
        "interrupted": 0,
    }
    complete_status["cases"] = {
        case_id: {**record, "last_status": "succeeded"}
        for case_id, record in status["cases"].items()
    }
    assert "Next: Sweep complete" in sweep._render_status_snapshot(
        tmp_path, manifest, complete_status
    )

    assert sweep._select_case_ids(manifest, status, mode="resume", explicit_case_ids=None) == [
        "bbb222bbb222",
        "ccc333ccc333",
    ]
    assert sweep._select_case_ids(manifest, status, mode="all", explicit_case_ids=None) == [
        "aaa111aaa111",
        "bbb222bbb222",
        "ccc333ccc333",
    ]
    assert sweep._select_case_ids(manifest, status, mode="failed", explicit_case_ids=None) == [
        "bbb222bbb222"
    ]
    assert sweep._select_case_ids(manifest, status, mode="pending", explicit_case_ids=None) == [
        "ccc333ccc333"
    ]
    assert sweep._select_case_ids(
        manifest,
        status,
        mode="all",
        explicit_case_ids=["ccc333ccc333"],
    ) == ["ccc333ccc333"]
    with pytest.raises(ValueError):
        sweep._select_case_ids(
            manifest,
            status,
            mode="all",
            explicit_case_ids=["missing"],
        )


def test_render_dashboard_layout(tmp_path):
    root = tmp_path / "sweep"
    root.mkdir()
    manifest = {
        "total_cases": 2,
        "runner_defaults": {"max_parallel_jobs": 1, "stop_on_error": False},
        "cases": [
            {"case_id": "aaa111aaa111", "case_dir": "cases/aaa111aaa111", "label": "alpha"},
            {"case_id": "bbb222bbb222", "case_dir": "cases/bbb222bbb222", "label": "beta"},
        ],
    }
    status = {
        "aggregate_counts": {
            "pending": 0,
            "queued": 0,
            "running": 1,
            "succeeded": 1,
            "failed": 0,
            "skipped": 0,
            "interrupted": 0,
        },
        "run_session": {
            "is_active": True,
            "mode": "resume",
            "worker_count": 1,
            "stop_on_error": False,
            "active_case_ids": ["bbb222bbb222"],
        },
        "cases": {
            "aaa111aaa111": {
                "last_status": "succeeded",
                "attempt_count": 1,
                "last_return_code": 0,
                "last_started_at": "2024-01-01T00:00:00+00:00",
                "last_finished_at": "2024-01-01T00:00:02+00:00",
                "last_duration_seconds": 2.0,
                "log_path": "cases/aaa111aaa111/run.log",
                "last_error_summary": None,
            },
            "bbb222bbb222": {
                "last_status": "running",
                "attempt_count": 2,
                "last_return_code": None,
                "last_started_at": "2024-01-01T00:00:01+00:00",
                "last_finished_at": None,
                "last_duration_seconds": None,
                "log_path": "cases/bbb222bbb222/run.log",
                "last_error_summary": None,
            },
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "status.json").write_text(json.dumps(status), encoding="utf-8")

    dashboard = sweep.render_status_dashboard(
        root,
        manifest,
        status,
        width=80,
        height=16,
    )
    assert "Progress:" in dashboard
    assert "Running:" in dashboard
    assert "bbb222bbb222" in dashboard

    idle_status = {
        "aggregate_counts": {
            "pending": 1,
            "queued": 0,
            "running": 0,
            "succeeded": 0,
            "failed": 0,
            "skipped": 0,
            "interrupted": 0,
        },
        "run_session": {
            "is_active": False,
            "active_case_ids": ["missing"],
        },
        "cases": {
            "aaa111aaa111": {
                "last_status": "pending",
                "attempt_count": 0,
                "last_return_code": None,
                "last_started_at": None,
                "last_finished_at": None,
                "last_duration_seconds": None,
                "log_path": "cases/aaa111aaa111/run.log",
                "last_error_summary": None,
            }
        },
    }
    idle_manifest = {
        "total_cases": 1,
        "runner_defaults": {"max_parallel_jobs": 1, "stop_on_error": False},
        "cases": [
            {"case_id": "aaa111aaa111", "case_dir": "cases/aaa111aaa111", "label": "alpha"},
        ],
    }
    idle_dashboard = sweep.render_status_dashboard(
        root,
        idle_manifest,
        idle_status,
        width=20,
        height=12,
    )
    assert "Session: idle" in idle_dashboard
    assert "Running:\n  <none>" in idle_dashboard
    assert "Recent:\n  <none>" in idle_dashboard


def test_status_store_and_run_case_subprocess_exception(tmp_path, monkeypatch):
    root = tmp_path / "sweep"
    root.mkdir()
    manifest = {
        "schema_version": 1,
        "generated_at": "now",
        "sweep_root": str(root),
        "runner_defaults": {"max_parallel_jobs": 1, "stop_on_error": False},
        "notes": None,
        "total_cases": 1,
        "cases": [
            {
                "case_id": "aaa111aaa111",
                "case_dir": "cases/aaa111aaa111",
                "config_path": "cases/aaa111aaa111/config.json",
                "label": "base",
                "sweep_values": {},
                "config_fingerprint": "f" * 64,
            }
        ],
    }
    status = sweep._initial_status(manifest)
    (root / "cases" / "aaa111aaa111").mkdir(parents=True)
    (root / "cases" / "aaa111aaa111" / "config.json").write_text("{}", encoding="utf-8")
    store = sweep._StatusStore(root, status)
    store.start_session(mode="resume", worker_count=1, stop_on_error=False, selected_case_ids=["aaa111aaa111"])
    with pytest.raises(ValueError):
        store.start_session(mode="resume", worker_count=1, stop_on_error=False, selected_case_ids=[])

    monkeypatch.setattr(sweep.subprocess, "Popen", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("boom")))
    result = sweep._run_case_subprocess(
        root,
        manifest["cases"][0],
        store=store,
        process_lock=threading.Lock(),
        active_processes={},
        interrupted_case_ids=set(),
    )
    store.finish_session()

    updated = json.loads((root / "status.json").read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert updated["cases"]["aaa111aaa111"]["last_error_summary"] == "boom"


def test_run_case_subprocess_marks_interrupted_result(tmp_path, monkeypatch):
    root = tmp_path / "sweep"
    root.mkdir()
    manifest = {
        "schema_version": 1,
        "generated_at": "now",
        "sweep_root": str(root),
        "runner_defaults": {"max_parallel_jobs": 1, "stop_on_error": False},
        "notes": None,
        "total_cases": 1,
        "cases": [
            {
                "case_id": "aaa111aaa111",
                "case_dir": "cases/aaa111aaa111",
                "config_path": "cases/aaa111aaa111/config.json",
                "label": "base",
                "sweep_values": {},
                "config_fingerprint": "f" * 64,
            }
        ],
    }
    status = sweep._initial_status(manifest)
    (root / "cases" / "aaa111aaa111").mkdir(parents=True)
    (root / "cases" / "aaa111aaa111" / "config.json").write_text("{}", encoding="utf-8")
    store = sweep._StatusStore(root, status)
    store.start_session(
        mode="resume",
        worker_count=1,
        stop_on_error=False,
        selected_case_ids=["aaa111aaa111"],
    )

    class SuccessPopen:
        def __init__(self, cmd, stdout, stderr):
            self.returncode = 0

        def wait(self, timeout=None):
            return self.returncode

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(sweep.subprocess, "Popen", SuccessPopen)
    result = sweep._run_case_subprocess(
        root,
        manifest["cases"][0],
        store=store,
        process_lock=threading.Lock(),
        active_processes={},
        interrupted_case_ids={"aaa111aaa111"},
    )
    store.finish_session()

    updated = json.loads((root / "status.json").read_text(encoding="utf-8"))
    assert result["status"] == "interrupted"
    assert updated["cases"]["aaa111aaa111"]["last_error_summary"] == "Interrupted by user"


def test_terminate_active_processes_kills_stuck_children():
    class HangingProcess:
        def __init__(self):
            self._returncode = None
            self.terminated = False
            self.killed = False

        def poll(self):
            return self._returncode

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            if self._returncode is None:
                raise sweep.subprocess.TimeoutExpired(cmd="x", timeout=timeout)
            return self._returncode

        def kill(self):
            self.killed = True
            self._returncode = -9

    process = HangingProcess()
    interrupted_case_ids = set()
    sweep._terminate_active_processes(
        {"aaa111aaa111": process},
        interrupted_case_ids=interrupted_case_ids,
        process_lock=threading.Lock(),
    )

    assert "aaa111aaa111" in interrupted_case_ids
    assert process.terminated is True
    assert process.killed is True


def test_terminate_active_processes_ignores_finished_children():
    class FinishedProcess:
        def __init__(self):
            self.terminated = False
            self.wait_called = False

        def poll(self):
            return 0

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            self.wait_called = True
            return 0

        def kill(self):
            raise AssertionError("kill should not be called")

    process = FinishedProcess()
    sweep._terminate_active_processes(
        {"aaa111aaa111": process},
        interrupted_case_ids=set(),
        process_lock=threading.Lock(),
    )

    assert process.terminated is False
    assert process.wait_called is False


def test_run_sweep_resume_and_stop_on_error(tmp_path, monkeypatch):
    master_path = tmp_path / "master.jsonc"
    payload = _master_payload(
        tmp_path,
        sweep={
            "incentive_direction": [None, "positive", "negative"],
        },
    )
    _write_master(master_path, payload)
    manifest = sweep.generate_sweep(master_path)
    root = Path(manifest["sweep_root"])

    launched = []
    outcomes = {
        manifest["cases"][0]["case_id"]: 1,
        manifest["cases"][1]["case_id"]: 0,
        manifest["cases"][2]["case_id"]: 0,
    }

    class ImmediatePopen:
        def __init__(self, cmd, stdout, stderr):
            self.case_id = Path(cmd[-1]).parent.name
            launched.append(self.case_id)
            self.returncode = outcomes[self.case_id]
            stdout.write(f"log for {self.case_id}\n")

        def wait(self, timeout=None):
            return self.returncode

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(sweep.subprocess, "Popen", ImmediatePopen)

    exit_code = sweep.run_sweep(root, stop_on_error=True)
    assert exit_code == 1
    assert launched == [manifest["cases"][0]["case_id"]]

    status_after_first = json.loads((root / "status.json").read_text(encoding="utf-8"))
    assert status_after_first["cases"][manifest["cases"][0]["case_id"]]["last_status"] == "failed"
    assert status_after_first["cases"][manifest["cases"][1]["case_id"]]["last_status"] == "skipped"
    assert status_after_first["cases"][manifest["cases"][2]["case_id"]]["last_status"] == "skipped"

    launched.clear()
    outcomes[manifest["cases"][0]["case_id"]] = 0

    exit_code = sweep.run_sweep(root)
    assert exit_code == 0
    assert launched == [
        manifest["cases"][0]["case_id"],
        manifest["cases"][1]["case_id"],
        manifest["cases"][2]["case_id"],
    ]

    status_after_second = json.loads((root / "status.json").read_text(encoding="utf-8"))
    assert status_after_second["cases"][manifest["cases"][0]["case_id"]]["attempt_count"] == 2
    assert status_after_second["cases"][manifest["cases"][0]["case_id"]]["last_status"] == "succeeded"
    assert status_after_second["cases"][manifest["cases"][1]["case_id"]]["last_status"] == "succeeded"
    assert status_after_second["cases"][manifest["cases"][2]["case_id"]]["last_status"] == "succeeded"
    assert (root / "cases" / manifest["cases"][0]["case_id"] / "run.log").exists()
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert summary["run"]["selected_case_count"] == 3


def test_run_sweep_stop_on_error_does_not_schedule_after_mixed_done_batch(
    tmp_path, monkeypatch
):
    master_path = tmp_path / "master.jsonc"
    payload = _master_payload(
        tmp_path,
        max_parallel_jobs=2,
        sweep={
            "incentive_direction": [None, "positive", "negative"],
        },
    )
    _write_master(master_path, payload)
    manifest = sweep.generate_sweep(master_path)
    root = Path(manifest["sweep_root"])

    launched = []
    case_ids = [case["case_id"] for case in manifest["cases"]]
    outcomes = {
        case_ids[0]: 1,
        case_ids[1]: 0,
        case_ids[2]: 0,
    }

    class ImmediatePopen:
        def __init__(self, cmd, stdout, stderr):
            self.case_id = Path(cmd[-1]).parent.name
            launched.append(self.case_id)
            self.returncode = outcomes[self.case_id]
            stdout.write(f"log for {self.case_id}\n")

        def wait(self, timeout=None):
            return self.returncode

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    original_wait = sweep.wait
    call_count = {"value": 0}

    def fake_wait(futures, timeout=None, return_when=None):
        call_count["value"] += 1
        if call_count["value"] == 1:
            future_list = list(futures)
            return set(future_list[:2]), set()
        return original_wait(futures, timeout=timeout, return_when=return_when)

    monkeypatch.setattr(sweep.subprocess, "Popen", ImmediatePopen)
    monkeypatch.setattr(sweep, "wait", fake_wait)

    exit_code = sweep.run_sweep(root, stop_on_error=True)

    assert exit_code == 1
    assert launched == case_ids[:2]

    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    assert status["cases"][case_ids[0]]["last_status"] == "failed"
    assert status["cases"][case_ids[1]]["last_status"] == "succeeded"
    assert status["cases"][case_ids[2]]["last_status"] == "skipped"


def test_run_sweep_respects_explicit_no_stop_on_error_override(tmp_path, monkeypatch):
    master_path = tmp_path / "master.jsonc"
    payload = _master_payload(
        tmp_path,
        stop_on_error=True,
        sweep={
            "incentive_direction": [None, "positive", "negative"],
        },
    )
    _write_master(master_path, payload)
    manifest = sweep.generate_sweep(master_path)
    root = Path(manifest["sweep_root"])

    launched = []
    case_ids = [case["case_id"] for case in manifest["cases"]]
    outcomes = {
        case_ids[0]: 1,
        case_ids[1]: 0,
        case_ids[2]: 0,
    }

    class ImmediatePopen:
        def __init__(self, cmd, stdout, stderr):
            self.case_id = Path(cmd[-1]).parent.name
            launched.append(self.case_id)
            self.returncode = outcomes[self.case_id]
            stdout.write(f"log for {self.case_id}\n")

        def wait(self, timeout=None):
            return self.returncode

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(sweep.subprocess, "Popen", ImmediatePopen)

    exit_code = sweep.run_sweep(root, stop_on_error=False)

    assert exit_code == 1
    assert launched == case_ids

    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    assert status["cases"][case_ids[0]]["last_status"] == "failed"
    assert status["cases"][case_ids[1]]["last_status"] == "succeeded"
    assert status["cases"][case_ids[2]]["last_status"] == "succeeded"
    assert status["run_session"]["stop_on_error"] is False


def test_run_sweep_modes_noop_and_invalid_mode(tmp_path, monkeypatch):
    master_path = tmp_path / "master.jsonc"
    payload = _master_payload(tmp_path)
    _write_master(master_path, payload)
    manifest = sweep.generate_sweep(master_path)
    root = Path(manifest["sweep_root"])

    class SuccessPopen:
        def __init__(self, cmd, stdout, stderr):
            self.returncode = 0
            stdout.write("ok\n")

        def wait(self, timeout=None):
            return self.returncode

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(sweep.subprocess, "Popen", SuccessPopen)
    assert sweep.run_sweep(root, mode="all") == 0
    live_stream = _Stream(tty=True)
    original_wait = sweep.wait
    calls = {"count": 0}

    def fake_wait(futures, timeout=None, return_when=None):
        calls["count"] += 1
        if calls["count"] == 1:
            return set(), set(futures)
        return original_wait(futures, timeout=timeout, return_when=return_when)

    monkeypatch.setattr(sweep, "wait", fake_wait)
    assert sweep.run_sweep(
        root,
        mode="all",
        case_ids=[manifest["cases"][0]["case_id"]],
        max_parallel_jobs=2,
        stream=live_stream,
        dashboard_refresh_interval=0.01,
        terminal_size_getter=lambda fallback: os.terminal_size((80, 16)),
    ) == 0
    assert "\x1b[2J\x1b[H" in live_stream.getvalue()
    assert calls["count"] >= 2
    assert sweep.run_sweep(
        root,
        mode="all",
        case_ids=[manifest["cases"][0]["case_id"]],
        max_parallel_jobs=2,
    ) == 0
    no_work_stream = _Stream(tty=True)
    assert sweep.run_sweep(root, mode="pending") == 0
    assert sweep.run_sweep(
        root,
        mode="pending",
        stream=no_work_stream,
        terminal_size_getter=lambda fallback: os.terminal_size((80, 16)),
    ) == 0
    assert "No matching cases to run." in no_work_stream.getvalue()

    with pytest.raises(ValueError):
        sweep.run_sweep(root, mode="bad")
    with pytest.raises(ValueError):
        sweep.run_sweep(root, max_parallel_jobs=0)


def test_run_sweep_handles_keyboard_interrupt(tmp_path, monkeypatch):
    master_path = tmp_path / "master.jsonc"
    payload = _master_payload(tmp_path, sweep={"incentive_type": ["historical"]})
    _write_master(master_path, payload)
    manifest = sweep.generate_sweep(master_path)
    root = Path(manifest["sweep_root"])
    captured = {}

    class FakeFuture:
        def __init__(self):
            self.cancelled = False

        def cancel(self):
            self.cancelled = True
            return True

    class FakeExecutor:
        def __init__(self, max_workers):
            captured["max_workers"] = max_workers
            self.future = FakeFuture()

        def submit(self, *args, **kwargs):
            return self.future

        def shutdown(self, wait=True):
            captured["shutdown_wait"] = wait

    monkeypatch.setattr(sweep, "ThreadPoolExecutor", FakeExecutor)
    monkeypatch.setattr(
        sweep,
        "wait",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    assert sweep.run_sweep(root) == 1
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    case_id = manifest["cases"][0]["case_id"]
    assert status["cases"][case_id]["last_status"] == "interrupted"
    assert captured["max_workers"] == 1
    assert captured["shutdown_wait"] is True
