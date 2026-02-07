# LLM Agora

A minimal arena where LLM-backed agents interact by taking public turns (optionally preceded by private reflections, succeeded by public and private surveys) until each reaches a configurable quota.
Pre- and post-interviews can be optionally included. Agent personas, topics of the interaction and the arena in which the interaction takes place are all specifiable. An analysis and plotting suite is also included.


## Requirements

- Python >=3.12
- An OpenRouter API key stored in `.env`


## Setup

1. Create a virtualenv via [uv](https://github.com/astral-sh/uv) (or your tool of choice):
   ```bash
   uv venv --python 3.12.4
   source .venv/bin/activate
   uv pip install -e ".[analysis]"
   ```
2. Add your OpenRouter API key to a `.env` file in the repo root:
   ```bash
   OPENROUTER_API_KEY=sk-...
   ```

## Running tests

```bash
.venv/bin/python3 -m pytest --cov=agora
```

Tests monkeypatch the `LLMClient`, so no external calls occur. Running `uv pip install -e .` keeps pytest in sync with local code.
Tests assume the package is installed (editable install recommended).

## GitHub PR pipeline

Runs `pytest` with coverage report, and comments the coverage percentage on the PR. Note that it doesn't install the package in 'analysis' mode, because the `sentence-transformers` package is too large for GitHub free nodes.

## Notebook demo

There is one canonical notebook: `notebooks/demo.ipynb`.
It is intentionally thin and calls the high-level source workflow in `agora.experiment`.

## CLI

Install the package in editable mode (`uv pip install -e .`) to expose the `agora` command.

Canonical config lives at `data/example.json` and matches the notebook and CLI arguments exactly.
All optional features are `false` by default and must be explicitly enabled.

```bash
# Run with config
agora run --config data/example.json

# Override specific fields from config
agora run --config data/example.json \
  --scenario-id peer_collab_1 \
  --question-variant agreeable \
  --enable-analyzer \
  --enable-plots

# Run with no config file (all args via CLI)
agora run \
  --scenario-id hier_account_1 \
  --question-variant controversial \
  --side-order 12 \
  --turns-per-agent 2

# Enable surveys independently
agora run --config data/example.json --enable-public-survey --enable-private-survey

# Optional retention toggles for survey streams
agora run --config data/example.json --enable-private-survey --keep-private-survey

# Resume from an existing snapshot directory
agora run --config data/example.json --load-snapshot --load-dir outputs/hier_account_1_controversial_12_biased

# Indexed output mode: run folder is a short unique ID and index row is appended
agora run --config data/example.json --indexed-output
```

Output behavior:
- default mode writes to a readable folder name under `outputs/` (for example `outputs/peer_collab_1_agreeable_12_neutral`)
- indexed mode writes to `outputs/<run_id>` and appends one row per run to `outputs/index.csv`
- if output-related features are all disabled, no output directory is created
- `load_snapshot=true` requires `load_dir` (directory containing `debate_snapshot.json`)
- `show_plots=true` requires `enable_plots=true`
- when outputs are enabled, each run folder contains run artifacts (`config.json`, plots, optional snapshot)
- `eval_data.json` is written only when at least one evaluation stream is enabled (`enable_analyzer`, `enable_persona_evaluation`, `enable_public_survey`, or `enable_private_survey`)
