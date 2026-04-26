"""Sweep generation, execution, and live status display for Agora."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import fields
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
import re
from threading import Lock
from typing import IO, Any, Mapping, Sequence

from .agora import SWEEP_PROGRESS_PREFIX
from .experiment import (
    EXPERIMENT_PATH_FIELDS,
    ExperimentConfig,
    build_experiment_config,
    resolve_experiment_payload_paths,
)

SWEEP_SCHEMA_VERSION = 1
CASE_STATUSES: tuple[str, ...] = (
    "pending",
    "queued",
    "running",
    "succeeded",
    "failed",
    "skipped",
    "interrupted",
)
TERMINAL_CASE_STATUSES: tuple[str, ...] = (
    "succeeded",
    "failed",
    "skipped",
    "interrupted",
)
RUN_SELECTION_MODES: tuple[str, ...] = ("resume", "all", "failed", "pending")
_TRACEBACK_PREFIX = "Traceback (most recent call last):"
_TRACEBACK_EXIT_GRACE_SECONDS = 5.0
_PROCESS_POLL_INTERVAL_SECONDS = 0.25
_PROGRESS_RE = re.compile(rf"^{re.escape(SWEEP_PROGRESS_PREFIX)} Turn (\d+)/(\d+)$")
MASTER_ALLOWED_KEYS = frozenset(
    {
        "sweep_root",
        "max_parallel_jobs",
        "stop_on_error",
        "number_of_repeats",
        "base",
        "sweep",
        "notes",
    }
)
MASTER_FORBIDDEN_FIELDS = frozenset(
    {"output_dir", "outputs_root", "run_name", "indexed_output", "index_csv"}
)


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _strip_jsonc_comments(text: str) -> str:
    pieces: list[str] = []
    in_string = False
    escaped = False
    in_line_comment = False
    in_block_comment = False
    index = 0

    while index < len(text):
        char = text[index]
        nxt = text[index + 1] if index + 1 < len(text) else ""

        if in_line_comment:
            if char == "\n":
                in_line_comment = False
                pieces.append(char)
            index += 1
            continue

        if in_block_comment:
            if char == "*" and nxt == "/":
                in_block_comment = False
                index += 2
            else:
                if char == "\n":
                    pieces.append("\n")
                index += 1
            continue

        if in_string:
            pieces.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue

        if char == '"':
            in_string = True
            pieces.append(char)
            index += 1
            continue

        if char == "/" and nxt == "/":
            in_line_comment = True
            index += 2
            continue

        if char == "/" and nxt == "*":
            in_block_comment = True
            index += 2
            continue

        pieces.append(char)
        index += 1

    if in_block_comment:
        raise ValueError("Unterminated block comment in JSONC input")

    return "".join(pieces)


def _load_jsonc_object(text: str) -> dict[str, Any]:
    payload = json.loads(_strip_jsonc_comments(text))
    if not isinstance(payload, dict):
        raise ValueError("Sweep config must be a JSON object")
    return payload


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must be a JSON object")
    return payload


def _read_log_delta(
    path: Path,
    *,
    offset: int,
) -> tuple[int, list[str]]:
    if not path.exists():
        return offset, []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(offset)
        chunk = handle.read()
        new_offset = handle.tell()
    if not chunk:
        return new_offset, []
    return new_offset, chunk.splitlines()


def _error_summary_from_log_lines(
    lines: Sequence[str],
    *,
    fallback: str,
) -> str:
    ignored_prefixes = (
        "Traceback ",
        'File "',
        "[agora sweep]",
    )
    ignored_lines = {
        "The above exception was the direct cause of the following exception:",
        "During handling of the above exception, another exception occurred:",
    }
    for line in reversed([item.strip() for item in lines if item.strip()]):
        if line in ignored_lines:
            continue
        if line.startswith(ignored_prefixes):
            continue
        return line
    return fallback


def _read_text_tail(path: Path, *, max_bytes: int = 8192) -> str:
    if not path.exists():
        return ""
    size = path.stat().st_size
    with path.open("rb") as handle:
        if size > max_bytes:
            handle.seek(-max_bytes, os.SEEK_END)
        chunk = handle.read()
    return chunk.decode("utf-8", errors="replace")


@lru_cache(maxsize=None)
def _case_total_turns(config_path: str) -> int:
    payload = _load_json_object(Path(config_path))
    return int(build_experiment_config(payload).num_turns)


def _case_turn_progress(
    root: Path | str,
    case: Mapping[str, Any],
) -> str:
    root_path = Path(root)
    config_path = case.get("config_path")
    if config_path is None:
        config_path = Path(case["case_dir"]) / "config.json"
    total_turns = _case_total_turns(str(root_path / config_path))
    log_text = _read_text_tail(root_path / case["case_dir"] / "run.log")

    last_match: tuple[str, str] | None = None
    for line in log_text.splitlines():
        match = _PROGRESS_RE.match(line.strip())
        if match is not None:
            last_match = match.groups()

    if last_match is not None:
        current_turn, total_from_log = last_match
        return f"Turn {current_turn}/{total_from_log}"
    return f"Turn 0/{total_turns}"


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(_json_ready(payload), indent=2) + "\n")


def _experiment_field_names() -> set[str]:
    return {field.name for field in fields(ExperimentConfig)}


def _normalize_master_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    unknown_keys = set(payload) - MASTER_ALLOWED_KEYS
    if unknown_keys:
        raise ValueError(f"Unknown sweep config keys: {sorted(unknown_keys)}")

    if "sweep_root" not in payload:
        raise ValueError("Missing required sweep setting: sweep_root")
    if "base" not in payload:
        raise ValueError("Missing required sweep setting: base")

    base = payload["base"]
    if not isinstance(base, dict):
        raise ValueError("base must be a JSON object")
    sweep = payload.get("sweep", {})
    if sweep is None:
        sweep = {}
    if not isinstance(sweep, dict):
        raise ValueError("sweep must be a JSON object")

    notes = payload.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise ValueError("notes must be a string when provided")

    max_parallel_jobs = payload.get("max_parallel_jobs", 1)
    if not isinstance(max_parallel_jobs, int) or max_parallel_jobs <= 0:
        raise ValueError("max_parallel_jobs must be a positive integer")

    stop_on_error = payload.get("stop_on_error", False)
    if not isinstance(stop_on_error, bool):
        raise ValueError("stop_on_error must be a boolean")

    number_of_repeats = payload.get("number_of_repeats", 1)
    if not isinstance(number_of_repeats, int) or number_of_repeats <= 0:
        raise ValueError("number_of_repeats must be a positive integer")

    valid_fields = _experiment_field_names()
    for section_name, section_payload in (("base", base), ("sweep", sweep)):
        unknown_fields = set(section_payload) - valid_fields
        if unknown_fields:
            raise ValueError(
                f"Unknown {section_name} fields: {sorted(unknown_fields)}"
            )
        forbidden_fields = set(section_payload) & MASTER_FORBIDDEN_FIELDS
        if forbidden_fields:
            raise ValueError(
                f"{section_name} cannot set generator-managed fields: {sorted(forbidden_fields)}"
            )

    normalized_sweep: dict[str, list[Any]] = {}
    for field_name, values in sweep.items():
        if not isinstance(values, list):
            raise ValueError(f"sweep field '{field_name}' must be a list of candidate values")
        if not values:
            raise ValueError(f"sweep field '{field_name}' must not be empty")
        seen_tokens: set[str] = set()
        normalized_values: list[Any] = []
        for value in values:
            token = _canonical_json(value)
            if token in seen_tokens:
                raise ValueError(f"sweep field '{field_name}' contains duplicate values")
            seen_tokens.add(token)
            normalized_values.append(value)
        normalized_sweep[field_name] = normalized_values

    return {
        "sweep_root": Path(str(payload["sweep_root"])),
        "max_parallel_jobs": max_parallel_jobs,
        "stop_on_error": stop_on_error,
        "number_of_repeats": number_of_repeats,
        "notes": notes,
        "base": dict(base),
        "sweep": normalized_sweep,
    }


def _resolve_sweep_master_paths(
    payload: Mapping[str, Any], *, base_dir: Path
) -> dict[str, Any]:
    resolved = dict(payload)

    sweep_root = resolved.get("sweep_root")
    if sweep_root is not None:
        sweep_root_path = (
            sweep_root if isinstance(sweep_root, Path) else Path(str(sweep_root))
        )
        if not sweep_root_path.is_absolute():
            resolved["sweep_root"] = (base_dir / sweep_root_path).resolve()
        else:
            resolved["sweep_root"] = sweep_root_path.resolve()

    base_payload = resolved.get("base")
    if isinstance(base_payload, Mapping):
        resolved["base"] = resolve_experiment_payload_paths(
            base_payload, base_dir=base_dir
        )

    sweep_payload = resolved.get("sweep")
    if isinstance(sweep_payload, Mapping):
        resolved_sweep: dict[str, Any] = {}
        for field_name, values in sweep_payload.items():
            if field_name not in EXPERIMENT_PATH_FIELDS or not isinstance(values, list):
                resolved_sweep[field_name] = values
                continue
            resolved_sweep[field_name] = [
                resolve_experiment_payload_paths(
                    {field_name: value}, base_dir=base_dir
                )[field_name]
                for value in values
            ]
        resolved["sweep"] = resolved_sweep

    return resolved


def load_sweep_config(path: Path | str) -> tuple[dict[str, Any], str]:
    config_path = Path(path)
    raw_text = config_path.read_text(encoding="utf-8")
    payload = _resolve_sweep_master_paths(
        _load_jsonc_object(raw_text),
        base_dir=config_path.resolve().parent,
    )
    return _normalize_master_config(payload), raw_text


def _case_label(
    sweep_values: Mapping[str, Any], *, repeat_number: int = 1, repeat_count: int = 1
) -> str:
    if not sweep_values:
        label = "base"
    else:
        parts = [
            f"{field_name}={json.dumps(_json_ready(value), sort_keys=True)}"
            for field_name, value in sorted(sweep_values.items())
        ]
        label = ", ".join(parts)
    if repeat_count > 1:
        return f"{label} (repeat {repeat_number}/{repeat_count})"
    return label


def _case_identity(
    config_fingerprint: str, *, repeat_number: int, repeat_count: int
) -> tuple[str, str]:
    if repeat_count == 1:
        return config_fingerprint[:12], config_fingerprint
    case_fingerprint = hashlib.sha256(
        f"{config_fingerprint}:{repeat_number}:{repeat_count}".encode("utf-8")
    ).hexdigest()
    return case_fingerprint[:12], case_fingerprint


def _normalize_generated_case(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    # A no-incentive run must not carry a historical/future subtype.
    if normalized.get("incentive_direction") is None and "incentive_type" in normalized:
        normalized["incentive_type"] = None
    return normalized


def _expand_cases(master: Mapping[str, Any]) -> list[dict[str, Any]]:
    base = dict(master["base"])
    sweep = dict(master["sweep"])
    repeat_count = master["number_of_repeats"]
    sweep_fields = list(sweep)
    value_sets = [sweep[field_name] for field_name in sweep_fields]
    combinations = itertools.product(*value_sets) if sweep_fields else [()]

    seen_fingerprints: dict[str, str] = {}
    seen_case_ids: dict[str, str] = {}
    cases: list[dict[str, Any]] = []
    sweep_root = Path(master["sweep_root"])

    for combination in combinations:
        sweep_values = {
            field_name: value for field_name, value in zip(sweep_fields, combination)
        }
        merged = dict(base)
        merged.update(sweep_values)
        merged = _normalize_generated_case(merged)
        normalized_sweep_values = {
            field_name: merged.get(field_name) for field_name in sweep_fields
        }

        raw_combination = _canonical_json({**base, **sweep_values})
        canonical_config = _canonical_json(merged)
        prior_combination = seen_fingerprints.get(canonical_config)
        if prior_combination is not None:
            if prior_combination == raw_combination:
                raise ValueError("Sweep expansion produced duplicate cases")
            continue
        seen_fingerprints[canonical_config] = raw_combination

        config_fingerprint = hashlib.sha256(canonical_config.encode("utf-8")).hexdigest()
        for repeat_number in range(1, repeat_count + 1):
            case_id, case_fingerprint = _case_identity(
                config_fingerprint,
                repeat_number=repeat_number,
                repeat_count=repeat_count,
            )
            prior_fingerprint = seen_case_ids.get(case_id)
            if prior_fingerprint is not None and prior_fingerprint != case_fingerprint:
                raise ValueError(f"Hash collision detected for case_id {case_id}")
            seen_case_ids[case_id] = case_fingerprint

            case_dir = (sweep_root / "cases" / case_id).resolve()
            config_payload = {**merged, "output_dir": case_dir}
            build_experiment_config(config_payload)

            cases.append(
                {
                    "case_id": case_id,
                    "case_dir": Path("cases") / case_id,
                    "config_path": Path("cases") / case_id / "config.json",
                    "label": _case_label(
                        normalized_sweep_values,
                        repeat_number=repeat_number,
                        repeat_count=repeat_count,
                    ),
                    "repeat_number": repeat_number,
                    "repeat_count": repeat_count,
                    "sweep_values": normalized_sweep_values,
                    "config_fingerprint": config_fingerprint,
                    "config_payload": _json_ready(config_payload),
                }
            )

    return cases


def _manifest_from_master(master: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SWEEP_SCHEMA_VERSION,
        "generated_at": _now_utc_iso(),
        "sweep_root": str(master["sweep_root"]),
        "runner_defaults": {
            "max_parallel_jobs": master["max_parallel_jobs"],
            "stop_on_error": master["stop_on_error"],
        },
        "number_of_repeats": master["number_of_repeats"],
        "notes": master["notes"],
        "total_cases": len(cases),
        "cases": [
            {
                "case_id": case["case_id"],
                "case_dir": str(case["case_dir"]),
                "config_path": str(case["config_path"]),
                "label": case["label"],
                "repeat_number": case["repeat_number"],
                "repeat_count": case["repeat_count"],
                "sweep_values": _json_ready(case["sweep_values"]),
                "config_fingerprint": case["config_fingerprint"],
            }
            for case in cases
        ],
    }


def _counts_for_cases(case_records: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    counts = {status: 0 for status in CASE_STATUSES}
    for record in case_records.values():
        status = record["last_status"]
        counts[status] += 1
    return counts


def _initial_status(manifest: Mapping[str, Any]) -> dict[str, Any]:
    cases = {
        case["case_id"]: {
            "last_status": "pending",
            "attempt_count": 0,
            "last_return_code": None,
            "last_started_at": None,
            "last_finished_at": None,
            "last_duration_seconds": None,
            "log_path": f"{case['case_dir']}/run.log",
            "last_error_summary": None,
        }
        for case in manifest["cases"]
    }
    return {
        "schema_version": SWEEP_SCHEMA_VERSION,
        "updated_at": _now_utc_iso(),
        "aggregate_counts": _counts_for_cases(cases),
        "run_session": {
            "is_active": False,
            "started_at": None,
            "finished_at": None,
            "mode": None,
            "worker_count": None,
            "stop_on_error": None,
            "persistent": None,
            "selected_case_ids": [],
            "active_case_ids": [],
        },
        "cases": cases,
    }


def _prepare_sweep_root(root: Path, *, force: bool) -> None:
    if root.exists():
        if not force:
            if root.is_dir():
                if any(root.iterdir()):
                    raise ValueError(
                        f"Sweep root already exists and is not empty: {root}"
                    )
            else:
                raise ValueError(f"Sweep root already exists and is not a directory: {root}")
        else:
            if root.is_dir():
                shutil.rmtree(root)
            else:
                root.unlink()
    root.mkdir(parents=True, exist_ok=True)


def generate_sweep(config_path: Path | str, *, force: bool = False) -> dict[str, Any]:
    master, raw_text = load_sweep_config(config_path)
    cases = _expand_cases(master)
    manifest = _manifest_from_master(master, cases)
    status = _initial_status(manifest)

    root = Path(master["sweep_root"])
    _prepare_sweep_root(root, force=force)

    _write_text_atomic(root / "master_config.jsonc", raw_text)
    _write_json_atomic(root / "manifest.json", manifest)
    _write_json_atomic(root / "status.json", status)

    for case in cases:
        case_dir = root / case["case_dir"]
        case_dir.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(case_dir / "config.json", case["config_payload"])

    print(f"Generated {len(cases)} case(s) under {root}")
    return manifest


def _load_manifest(root: Path | str) -> dict[str, Any]:
    return _load_json_object(Path(root) / "manifest.json")


def _load_status(root: Path | str) -> dict[str, Any]:
    return _load_json_object(Path(root) / "status.json")


def _status_counts(status: Mapping[str, Any]) -> dict[str, int]:
    counts = status.get("aggregate_counts")
    if isinstance(counts, dict):
        expected = {key: int(counts.get(key, 0)) for key in CASE_STATUSES}
        if sum(expected.values()) == len(status.get("cases", {})):
            return expected
    return _counts_for_cases(status["cases"])


def _parse_iso(value: str | None) -> datetime | None:
    if value in (None, ""):
        return None
    return datetime.fromisoformat(value)


def _duration_seconds(started_at: str | None, finished_at: str | None) -> float | None:
    start = _parse_iso(started_at)
    finish = _parse_iso(finished_at)
    if start is None or finish is None:
        return None
    return round((finish - start).total_seconds(), 3)


def _render_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _truncate(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def _terminate_process(
    process: subprocess.Popen[Any],
    *,
    grace_seconds: float = 2.0,
) -> int | None:
    return_code = process.poll()
    if return_code is not None:
        return return_code

    process_id = getattr(process, "pid", None)
    use_process_group = process_id is not None and hasattr(os, "killpg")

    try:
        if use_process_group:
            os.killpg(process_id, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        return process.poll()

    deadline = time.time() + grace_seconds
    while process.poll() is None:
        remaining = max(0.0, deadline - time.time())
        if remaining == 0.0:
            break
        try:
            return process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            break

    if process.poll() is None:
        try:
            if use_process_group:
                os.killpg(process_id, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            return process.poll()

    return process.wait()


def _progress_bar(completed: int, total: int, width: int) -> str:
    width = max(10, width)
    if total <= 0:
        return "[" + ("-" * width) + "]"
    filled = int(round((completed / total) * width))
    filled = max(0, min(width, filled))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _case_map(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {case["case_id"]: dict(case) for case in manifest["cases"]}


def _render_status_snapshot(
    root: Path | str,
    manifest: Mapping[str, Any],
    status: Mapping[str, Any],
) -> str:
    counts = _status_counts(status)
    total_cases = manifest["total_cases"]
    run_session = status.get("run_session", {})
    failed_case_ids = [
        case_id
        for case_id, record in status["cases"].items()
        if record["last_status"] == "failed"
    ]
    interrupted_case_ids = [
        case_id
        for case_id, record in status["cases"].items()
        if record["last_status"] == "interrupted"
    ]

    if counts["pending"] or counts["skipped"]:
        next_action = f"agora sweep run --root {root}"
    elif counts["failed"] or counts["interrupted"]:
        next_action = f"agora sweep run --root {root} --mode failed"
    else:
        next_action = "Sweep complete"

    lines = [
        f"Sweep root: {root}",
        f"Total cases: {total_cases}",
        (
            "Counts: "
            + ", ".join(f"{status_name}={counts[status_name]}" for status_name in CASE_STATUSES)
        ),
        f"Run active: {bool(run_session.get('is_active'))}",
        f"Failed: {', '.join(failed_case_ids) if failed_case_ids else '<none>'}",
        (
            "Interrupted: "
            + (", ".join(interrupted_case_ids) if interrupted_case_ids else "<none>")
        ),
        f"Next: {next_action}",
    ]
    return "\n".join(lines)


def render_status_dashboard(
    root: Path | str,
    manifest: Mapping[str, Any],
    status: Mapping[str, Any],
    *,
    width: int,
    height: int,
    now: datetime | None = None,
) -> str:
    now = now or datetime.now(timezone.utc)
    counts = _status_counts(status)
    total_cases = manifest["total_cases"]
    completed = sum(counts[name] for name in TERMINAL_CASE_STATUSES)
    bar_width = max(10, min(40, width - 24))
    case_lookup = _case_map(manifest)
    run_session = status.get("run_session", {})

    active_rows: list[tuple[str, str, str, str]] = []
    for case_id in run_session.get("active_case_ids", []):
        record = status["cases"].get(case_id)
        if record is None:
            continue
        started_at = _parse_iso(record.get("last_started_at"))
        elapsed = None if started_at is None else (now - started_at).total_seconds()
        active_rows.append(
            (
                case_id,
                str(record.get("attempt_count", 0)),
                _render_duration(elapsed),
                _case_turn_progress(root, case_lookup.get(case_id, {})),
                case_lookup.get(case_id, {}).get("label", "unknown"),
            )
        )

    recent_rows: list[tuple[str, str, str]] = []
    for case_id, record in status["cases"].items():
        if record["last_status"] not in TERMINAL_CASE_STATUSES:
            continue
        recent_rows.append(
            (
                record.get("last_finished_at") or "",
                case_id,
                record["last_status"],
            )
        )
    recent_rows.sort(reverse=True)

    header_lines = [
        f"Sweep: {root}",
        (
            "Counts: "
            + ", ".join(f"{status_name}={counts[status_name]}" for status_name in CASE_STATUSES)
        ),
        (
            f"Progress: {_progress_bar(completed, total_cases, bar_width)} "
            f"{completed}/{total_cases}"
        ),
    ]

    if run_session.get("is_active"):
        header_lines.append(
            "Session: active"
            f" | mode={run_session.get('mode')}"
            f" | workers={run_session.get('worker_count')}"
            f" | stop_on_error={run_session.get('stop_on_error')}"
        )
    else:
        header_lines.append("Session: idle")

    remaining_lines = max(6, height - len(header_lines) - 4)
    active_slots = min(len(active_rows), max(1, remaining_lines // 2))
    recent_slots = max(1, remaining_lines - active_slots - 2)

    lines = list(header_lines)
    lines.append("")
    lines.append("Running:")
    if active_rows:
        for case_id, attempt, elapsed, turn_progress, label in active_rows[:active_slots]:
            row = (
                f"  {case_id} | attempt {attempt} | {elapsed} | "
                f"{turn_progress} | {label}"
            )
            lines.append(_truncate(row, width))
    else:
        lines.append("  <none>")

    lines.append("")
    lines.append("Recent:")
    if recent_rows:
        for _, case_id, final_status in recent_rows[:recent_slots]:
            record = status["cases"][case_id]
            duration = _render_duration(record.get("last_duration_seconds"))
            row = f"  {case_id} | {final_status} | {duration}"
            lines.append(_truncate(row, width))
    else:
        lines.append("  <none>")

    failed_case_ids = [
        case_id
        for case_id, record in status["cases"].items()
        if record["last_status"] == "failed"
    ]
    interrupted_case_ids = [
        case_id
        for case_id, record in status["cases"].items()
        if record["last_status"] == "interrupted"
    ]
    lines.append("")
    lines.append(
        _truncate(
            (
                f"Pending={counts['pending']} | Failed="
                f"{', '.join(failed_case_ids) if failed_case_ids else '<none>'} | "
                f"Interrupted="
                f"{', '.join(interrupted_case_ids) if interrupted_case_ids else '<none>'}"
            ),
            width,
        )
    )
    lines.append("Ctrl-C to exit dashboard. Use --once for snapshot mode.")

    return "\n".join(lines[: max(1, height)])


def _is_tty(stream: IO[str]) -> bool:
    return bool(getattr(stream, "isatty", lambda: False)())


def _write_line(stream: IO[str], line: str) -> None:
    stream.write(f"{line}\n")
    stream.flush()


def _render_live_dashboard(
    root: Path,
    *,
    stream: IO[str],
    terminal_size_getter: Any,
) -> None:
    manifest = _load_manifest(root)
    status = _load_status(root)
    size = terminal_size_getter(fallback=(120, 40))
    dashboard = render_status_dashboard(
        root,
        manifest,
        status,
        width=size.columns,
        height=size.lines,
    )
    stream.write("\x1b[2J\x1b[H")
    stream.write(dashboard)
    stream.flush()


def _select_case_ids(
    manifest: Mapping[str, Any],
    status: Mapping[str, Any],
    *,
    mode: str,
    explicit_case_ids: Sequence[str] | None,
) -> list[str]:
    ordered_case_ids = [case["case_id"] for case in manifest["cases"]]
    known_case_ids = set(ordered_case_ids)

    if explicit_case_ids:
        unknown_case_ids = sorted(set(explicit_case_ids) - known_case_ids)
        if unknown_case_ids:
            raise ValueError(f"Unknown case IDs: {unknown_case_ids}")
        case_filter = set(explicit_case_ids)
    else:
        case_filter = None

    def matches(case_id: str) -> bool:
        record = status["cases"][case_id]
        current_status = record["last_status"]
        if mode == "all":
            return True
        if mode == "failed":
            return current_status in {"failed", "interrupted"}
        if mode == "pending":
            return current_status == "pending"
        return current_status != "succeeded"

    return [
        case_id
        for case_id in ordered_case_ids
        if matches(case_id) and (case_filter is None or case_id in case_filter)
    ]


class _StatusStore:
    def __init__(self, root: Path, status: Mapping[str, Any]) -> None:
        self.root = root
        self.path = root / "status.json"
        self.data = json.loads(json.dumps(_json_ready(status)))
        self._lock = Lock()

    def _save_locked(self) -> None:
        self.data["updated_at"] = _now_utc_iso()
        self.data["aggregate_counts"] = _counts_for_cases(self.data["cases"])
        _write_json_atomic(self.path, self.data)

    def start_session(
        self,
        *,
        mode: str,
        worker_count: int,
        stop_on_error: bool,
        selected_case_ids: Sequence[str],
        persistent: bool = False,
    ) -> None:
        with self._lock:
            if self.data["run_session"]["is_active"]:
                raise ValueError("A sweep run is already marked active for this root")
            self.data["run_session"] = {
                "is_active": True,
                "started_at": _now_utc_iso(),
                "finished_at": None,
                "mode": mode,
                "worker_count": worker_count,
                "stop_on_error": stop_on_error,
                "persistent": persistent,
                "selected_case_ids": list(selected_case_ids),
                "active_case_ids": [],
            }
            self._save_locked()

    def finish_session(self) -> None:
        with self._lock:
            self.data["run_session"]["is_active"] = False
            self.data["run_session"]["finished_at"] = _now_utc_iso()
            self.data["run_session"]["active_case_ids"] = []
            self._save_locked()

    def mark_case_queued(self, case_id: str) -> None:
        with self._lock:
            self.data["cases"][case_id]["last_status"] = "queued"
            self._save_locked()

    def mark_case_running(self, case_id: str) -> int:
        with self._lock:
            case = self.data["cases"][case_id]
            case["attempt_count"] += 1
            case["last_status"] = "running"
            case["last_return_code"] = None
            case["last_started_at"] = _now_utc_iso()
            case["last_finished_at"] = None
            case["last_duration_seconds"] = None
            case["last_error_summary"] = None
            active_case_ids = self.data["run_session"]["active_case_ids"]
            if case_id not in active_case_ids:
                active_case_ids.append(case_id)
            self._save_locked()
            return int(case["attempt_count"])

    def mark_case_finished(
        self,
        case_id: str,
        *,
        final_status: str,
        return_code: int | None,
        error_summary: str | None,
    ) -> None:
        with self._lock:
            case = self.data["cases"][case_id]
            finished_at = _now_utc_iso()
            case["last_status"] = final_status
            case["last_return_code"] = return_code
            case["last_finished_at"] = finished_at
            case["last_duration_seconds"] = _duration_seconds(
                case.get("last_started_at"), finished_at
            )
            case["last_error_summary"] = error_summary
            active_case_ids = self.data["run_session"]["active_case_ids"]
            if case_id in active_case_ids:
                active_case_ids.remove(case_id)
            self._save_locked()

    def mark_case_skipped(self, case_id: str, reason: str) -> None:
        with self._lock:
            case = self.data["cases"][case_id]
            case["last_status"] = "skipped"
            case["last_return_code"] = None
            case["last_error_summary"] = reason
            self._save_locked()


def _run_case_subprocess(
    root: Path,
    case: Mapping[str, Any],
    *,
    store: _StatusStore,
    process_lock: Lock,
    active_processes: dict[str, subprocess.Popen[Any]],
    interrupted_case_ids: set[str],
) -> dict[str, Any]:
    case_id = case["case_id"]
    case_dir = root / case["case_dir"]
    config_path = root / case["config_path"]
    log_path = case_dir / "run.log"

    store.mark_case_running(case_id)

    try:
        with log_path.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "agora.cli",
                    "run",
                    "--emit-progress-markers",
                    "--config",
                    str(config_path),
                ],
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            with process_lock:
                active_processes[case_id] = process
            log_offset = 0
            log_lines: deque[str] = deque(maxlen=50)
            traceback_started_at: float | None = None
            terminated_after_traceback = False

            while True:
                handle.flush()
                log_offset, new_lines = _read_log_delta(log_path, offset=log_offset)
                if new_lines:
                    log_lines.extend(new_lines)
                    if (
                        traceback_started_at is None
                        and any(line.startswith(_TRACEBACK_PREFIX) for line in new_lines)
                    ):
                        traceback_started_at = time.monotonic()

                return_code = process.poll()
                if return_code is not None:
                    break

                if (
                    traceback_started_at is not None
                    and time.monotonic() - traceback_started_at
                    >= _TRACEBACK_EXIT_GRACE_SECONDS
                ):
                    handle.write(
                        "\n[agora sweep] Traceback detected but process did not exit; terminating.\n"
                    )
                    handle.flush()
                    return_code = _terminate_process(process)
                    terminated_after_traceback = True
                    break

                time.sleep(_PROCESS_POLL_INTERVAL_SECONDS)

            handle.flush()
            log_offset, new_lines = _read_log_delta(log_path, offset=log_offset)
            if new_lines:
                log_lines.extend(new_lines)
    except Exception as exc:
        store.mark_case_finished(
            case_id,
            final_status="failed",
            return_code=None,
            error_summary=str(exc),
        )
        return {"case_id": case_id, "status": "failed", "return_code": None}
    finally:
        with process_lock:
            active_processes.pop(case_id, None)

    if case_id in interrupted_case_ids:
        final_status = "interrupted"
        error_summary = "Interrupted by user"
    elif return_code == 0:
        final_status = "succeeded"
        error_summary = None
    else:
        final_status = "failed"
        fallback = (
            "Process hung after traceback and was terminated"
            if terminated_after_traceback
            else f"Process exited with code {return_code}"
        )
        error_summary = _error_summary_from_log_lines(log_lines, fallback=fallback)

    store.mark_case_finished(
        case_id,
        final_status=final_status,
        return_code=return_code,
        error_summary=error_summary,
    )
    return {"case_id": case_id, "status": final_status, "return_code": return_code}


def _terminate_active_processes(
    active_processes: Mapping[str, subprocess.Popen[Any]],
    *,
    interrupted_case_ids: set[str],
    process_lock: Lock,
) -> None:
    with process_lock:
        processes = list(active_processes.items())
        interrupted_case_ids.update(case_id for case_id, _ in processes)

    for _, process in processes:
        _terminate_process(process)


def _build_summary(
    manifest: Mapping[str, Any],
    status: Mapping[str, Any],
    *,
    mode: str,
    worker_count: int,
    selected_case_ids: Sequence[str],
    persistent: bool,
    started_at: str,
    finished_at: str,
) -> dict[str, Any]:
    selected_statuses = {
        case_id: status["cases"][case_id]["last_status"] for case_id in selected_case_ids
    }
    failed_case_ids = [
        case_id
        for case_id, record in status["cases"].items()
        if record["last_status"] == "failed"
    ]
    skipped_case_ids = [
        case_id
        for case_id, record in status["cases"].items()
        if record["last_status"] == "skipped"
    ]
    interrupted_case_ids = [
        case_id
        for case_id, record in status["cases"].items()
        if record["last_status"] == "interrupted"
    ]
    success = all(case_status == "succeeded" for case_status in selected_statuses.values())

    return {
        "schema_version": SWEEP_SCHEMA_VERSION,
        "created_at": finished_at,
        "run": {
            "started_at": started_at,
            "finished_at": finished_at,
            "mode": mode,
            "worker_count": worker_count,
            "persistent": persistent,
            "selected_case_ids": list(selected_case_ids),
            "selected_case_count": len(selected_case_ids),
            "success": success,
            "elapsed_seconds": _duration_seconds(started_at, finished_at),
        },
        "total_cases": manifest["total_cases"],
        "counts": _status_counts(status),
        "failed_case_ids": failed_case_ids,
        "skipped_case_ids": skipped_case_ids,
        "interrupted_case_ids": interrupted_case_ids,
    }


def run_sweep(
    root: Path | str,
    *,
    max_parallel_jobs: int | None = None,
    mode: str = "resume",
    case_ids: Sequence[str] | None = None,
    stop_on_error: bool | None = None,
    persistent: bool = False,
    stream: IO[str] | None = None,
    dashboard_refresh_interval: float = 0.25,
    terminal_size_getter: Any = shutil.get_terminal_size,
) -> int:
    if mode not in RUN_SELECTION_MODES:
        raise ValueError(f"mode must be one of: {list(RUN_SELECTION_MODES)}")
    if persistent and stop_on_error is True:
        raise ValueError("persistent cannot be combined with --stop-on-error")

    root_path = Path(root)
    stream = sys.stdout if stream is None else stream
    live_dashboard = _is_tty(stream)
    manifest = _load_manifest(root_path)
    status = _load_status(root_path)
    worker_count = (
        max_parallel_jobs
        if max_parallel_jobs is not None
        else manifest["runner_defaults"]["max_parallel_jobs"]
    )
    if worker_count <= 0:
        raise ValueError("max_parallel_jobs must be positive")
    selected_case_ids = _select_case_ids(
        manifest,
        status,
        mode=mode,
        explicit_case_ids=case_ids,
    )

    if not selected_case_ids:
        summary = _build_summary(
            manifest,
            status,
            mode=mode,
            worker_count=worker_count,
            selected_case_ids=[],
            persistent=persistent,
            started_at=_now_utc_iso(),
            finished_at=_now_utc_iso(),
        )
        _write_json_atomic(root_path / "summary.json", summary)
        if live_dashboard:
            _render_live_dashboard(
                root_path,
                stream=stream,
                terminal_size_getter=terminal_size_getter,
            )
            stream.write("\n")
            stream.flush()
        _write_line(stream, "No matching cases to run.")
        return 0

    if persistent:
        effective_stop_on_error = False
    elif stop_on_error is None:
        effective_stop_on_error = bool(manifest["runner_defaults"]["stop_on_error"])
    else:
        effective_stop_on_error = stop_on_error

    store = _StatusStore(root_path, status)
    store.start_session(
        mode=mode,
        worker_count=worker_count,
        stop_on_error=effective_stop_on_error,
        persistent=persistent,
        selected_case_ids=selected_case_ids,
    )

    case_lookup = _case_map(manifest)
    started_at = store.data["run_session"]["started_at"]
    process_lock = Lock()
    active_processes: dict[str, subprocess.Popen[Any]] = {}
    interrupted_case_ids: set[str] = set()
    scheduled_case_ids: list[str] = []
    failed_this_run = False
    interrupted = False

    pending_case_ids = deque(selected_case_ids)
    futures: dict[Any, str] = {}
    executor = ThreadPoolExecutor(max_workers=worker_count)

    def submit_next_case() -> bool:
        if not pending_case_ids:
            return False
        case_id = pending_case_ids.popleft()
        scheduled_case_ids.append(case_id)
        store.mark_case_queued(case_id)
        futures[
            executor.submit(
                _run_case_subprocess,
                root_path,
                case_lookup[case_id],
                store=store,
                process_lock=process_lock,
                active_processes=active_processes,
                interrupted_case_ids=interrupted_case_ids,
            )
        ] = case_id
        if not live_dashboard:
            _write_line(stream, f"Started {case_id}")
        return True

    try:
        while len(futures) < worker_count and submit_next_case():
            pass

        if live_dashboard:
            _render_live_dashboard(
                root_path,
                stream=stream,
                terminal_size_getter=terminal_size_getter,
            )

        while futures:
            done, _ = wait(
                tuple(futures),
                timeout=max(0.05, dashboard_refresh_interval) if live_dashboard else None,
                return_when=FIRST_COMPLETED,
            )
            if live_dashboard:
                _render_live_dashboard(
                    root_path,
                    stream=stream,
                    terminal_size_getter=terminal_size_getter,
                )
            if not done:
                continue
            completed_slots = 0
            for future in done:
                completed_slots += 1
                case_id = futures.pop(future)
                result = future.result()
                final_status = result["status"]
                if final_status in {"failed", "interrupted"}:
                    failed_this_run = True
                if persistent and final_status == "failed":
                    pending_case_ids.append(case_id)
                if not live_dashboard:
                    _write_line(stream, f"{final_status.capitalize()}: {case_id}")

            if effective_stop_on_error and failed_this_run:
                continue

            for _ in range(completed_slots):
                if not submit_next_case():
                    break
    except KeyboardInterrupt:
        interrupted = True
        for future, case_id in list(futures.items()):
            if future.cancel():
                interrupted_case_ids.add(case_id)
                store.mark_case_finished(
                    case_id,
                    final_status="interrupted",
                    return_code=None,
                    error_summary="Interrupted before launch",
                )
                futures.pop(future, None)
        _terminate_active_processes(
            active_processes,
            interrupted_case_ids=interrupted_case_ids,
            process_lock=process_lock,
        )
    finally:
        executor.shutdown(wait=True)
        if failed_this_run and effective_stop_on_error:
            scheduled_set = set(scheduled_case_ids)
            for case_id in selected_case_ids:
                if case_id not in scheduled_set:
                    store.mark_case_skipped(
                        case_id,
                        "Skipped because stop_on_error halted scheduling",
                    )
                    if not live_dashboard:
                        _write_line(stream, f"Skipped: {case_id}")

        store.finish_session()

    final_status = _load_status(root_path)
    finished_at = _now_utc_iso()
    summary = _build_summary(
        manifest,
        final_status,
        mode=mode,
        worker_count=worker_count,
        selected_case_ids=selected_case_ids,
        persistent=persistent,
        started_at=started_at,
        finished_at=finished_at,
    )
    _write_json_atomic(root_path / "summary.json", summary)

    if live_dashboard:
        _render_live_dashboard(
            root_path,
            stream=stream,
            terminal_size_getter=terminal_size_getter,
        )
        stream.write("\n")
        stream.flush()

    final_counts = summary["counts"]
    if interrupted:
        _write_line(stream, "Interrupted sweep run.")
    _write_line(
        stream,
        "Summary: "
        + ", ".join(f"{name}={final_counts[name]}" for name in CASE_STATUSES),
    )

    if interrupted:
        return 1
    if any(
        final_status["cases"][case_id]["last_status"] in {"failed", "interrupted"}
        for case_id in selected_case_ids
    ):
        return 1
    return 0


__all__ = [
    "CASE_STATUSES",
    "RUN_SELECTION_MODES",
    "SWEEP_SCHEMA_VERSION",
    "_expand_cases",
    "_load_jsonc_object",
    "_normalize_master_config",
    "_render_status_snapshot",
    "_select_case_ids",
    "_strip_jsonc_comments",
    "generate_sweep",
    "load_sweep_config",
    "render_status_dashboard",
    "run_sweep",
]
