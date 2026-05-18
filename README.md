# LLM Agora

[![Tests](https://github.com/danmohad/LLMAgora/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/danmohad/LLMAgora/actions/workflows/tests.yml)

A minimal arena where two LLM-backed agents discuss a scenario through configurable sub-turn events: public utterances, optional private reflections, optional public or private surveys, optional pre- and post-interviews. 

Agent personas, scenario topics, incentive modules, and prompt templates are defined JSON files in `data/`. Optional semantic analysis and parameter sweep workflows are provided.


## Requirements

- Python >=3.12
- An OpenRouter API key (`OPENROUTER_API_KEY`)

## Setup

1. Create a virtualenv via [uv](https://github.com/astral-sh/uv) (or your tool of choice):
   ```bash
   uv venv --python 3.12.4
   source .venv/bin/activate
   uv pip install -e .
   ```
2. Install optional analysis dependencies only if you will run semantic similarity or aggregate NLI/emotion analysis:
   ```bash
   uv pip install -e ".[analysis]"
   ```
3. Add your OpenRouter API key to the shell environment or to a `.env` file in the repo root:
   ```bash
   OPENROUTER_API_KEY=sk-...
   ```

### Running tests

```bash
.venv/bin/python3 -m pytest --cov=agora
```

Tests monkeypatch the `LLMClient`, so no external calls occur. Running `uv pip install -e .` keeps pytest in sync with local code.
Tests assume the package is installed (editable install recommended).

### CI

CI runs `pytest` with coverage and enforces 100% coverage through `pyproject.toml`.

## Running debates

### Notebook demo

Run `notebooks/run_demo.ipynb` for a single configurable run that can save a `debate_snapshot.json`.
The notebook is intentionally thin and calls the high-level workflow in `agora.experiment`.

### CLI

Install the package in editable mode (`uv pip install -e .`) to expose the `agora` command.

Canonical single-run config lives at `data/config_example.jsonc`. Relative paths are resolved relative to the file that contains them. CLI-only paths are resolved from the current working directory.

Optional retention, output, and analysis features are disabled by default (`false` flags and empty analysis metric lists). Analysis backends must be configured explicitly when their metric lists are non-empty; otherwise config validation fails. Prompt templates live in `data/prompts.json`, and sweep generation uses the commented master template at `data/sweep_example.jsonc`.

```bash
# Run with config: this is the recommended way to run the code for a single debate
# The example enables survey events, so it writes run output under outputs/.
agora run --config data/config_example.jsonc
```

The CLI allows any setting to be overridden using flags, e.g.:

```bash
agora run --config data/config_example.jsonc \
  --scenario-id ngo_climate_endorsement \
  --incentive-direction positive \
  --incentive-type future \
  --semantic-analysis-metrics self_consistency cross_agent_public_alignment \
  --semantic-similarity-method cosine \
  --semantic-similarity-model all-mpnet-base-v2 \
  --save-plots
```

### Sweep Workflows

Use `agora sweep` to expand one master config into generated case configs, run those cases, and aggregate completed results. Edit `data/sweep_example.jsonc`: put shared single-run settings in `base`, candidate values in `sweep`, and aggregate defaults in `aggregation`.

```bash
# Expand the master config into manifest/status plus cases/<case_id>/config.json
agora sweep generate --config data/sweep_example.jsonc

# Run all cases not already marked succeeded; this owns the terminal dashboard.
agora sweep run

# Re-run failed or interrupted cases only
agora sweep run --root outputs/sweeps/example --mode failed

# Keep retrying selected failed attempts until every selected case succeeds
agora sweep run --root outputs/sweeps/example --mode failed --persistent

# Build one aggregate JSON record for the parameter sweep.
agora sweep aggregate
```

When `--root` is omitted, `agora sweep run` and `agora sweep aggregate` infer it from the only non-generated `.jsonc` sweep config in the current working tree; pass `--root` explicitly when there is more than one. Generated files live under `sweep_root`, with per-case artifacts in `cases/<case_id>/`. Field meanings and sweep-specific rules are documented inline in `data/sweep_example.jsonc`.
