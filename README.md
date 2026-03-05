# LLM Agora

A minimal arena where LLM-backed agents interact by taking public turns (optionally preceded by private reflections, succeeded by public and private surveys) until each reaches a configurable quota.
Pre- and post-interviews run by default; `keep_pre_interview` and `keep_post_interview` only control whether agents retain those notes in local memory. Agent personas, topics of the interaction, and scenario incentive modules are configurable. An analysis and plotting suite is also included.


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

Canonical notebooks:
- `notebooks/run_demo.ipynb` for on-the-fly/online runs.
- `notebooks/offline_analysis_demo.ipynb` for post-processing from an existing `debate_snapshot.json` (offline analysis mode).
  It loads baseline settings from the source run's `config.json`, so you only choose post-processing overrides.
  It writes post-processing artifacts back into that same source run directory.

Both are intentionally thin and call the high-level workflow in `agora.experiment`.

## CLI

Install the package in editable mode (`uv pip install -e .`) to expose the `agora` command.

Canonical config lives at `data/config_example.json` and matches the notebook and CLI arguments exactly.
Optional retention and analysis features are disabled by default (`false` flags and empty analysis metric lists).
Sweep generation uses the commented master template at `data/sweep_example.jsonc`.

```bash
# Run with config
agora run --config data/config_example.json

# Override specific fields from config
agora run --config data/config_example.json \
  --scenario-id ngo_climate_endorsement \
  --incentive-direction positive \
  --incentive-type future \
  --semantic-analysis-metrics self_consistency cross_agent_public_alignment \
  --save-plots

# Use NLI for semantic scoring (instead of cosine embeddings)
agora run --config data/config_example.json \
  --semantic-analysis-metrics self_consistency \
  --semantic-similarity-method nli \
  --semantic-similarity-model dleemiller/finecat-nli-l \
  --semantic-similarity-device mps  # or cpu

# Run persona adherence analysis for selected metric slices only
agora run --config data/config_example.json \
  --persona-analysis-metrics public_per_turn full_debate_public \
  --persona-score-samples 3

# Run with no config file (all args via CLI)
agora run \
  --scenario-id promotion_committee_max_divergence \
  --incentive-direction none \
  --incentive-type historical \
  --num-turns 2

# Enable surveys via sub-turn events
agora run --config data/config_example.json \
  --subturn-event-order public_utterance private_survey

# Optional retention toggles for survey streams
agora run --config data/config_example.json \
  --keep-private-survey \
  --subturn-event-order public_utterance private_survey

# Resume from an existing snapshot directory
agora run --config data/config_example.json --load-snapshot --load-dir outputs/promotion_committee_max_divergence_no_incentive

# Indexed output mode: run folder is a short unique ID and index row is appended
agora run --config data/config_example.json --indexed-output
```

### Sweep Workflows

Use `agora sweep` to generate and run parameter sweeps. `agora sweep run` owns the terminal and renders the live dashboard until the batch completes.

```bash
# Expand the master config into manifest/status plus cases/<case_id>/config.json
agora sweep generate --config data/sweep_example.jsonc

# Run all cases that are not already marked succeeded.
# This command renders the live in-place dashboard in the current TTY.
agora sweep run

# Re-run failed or interrupted cases only
agora sweep run --root outputs/sweeps/example --mode failed
```

When `--root` is omitted, `agora sweep run` infers it from the only non-generated
`.jsonc` sweep config in the current working tree; if there is not exactly one,
pass `--root` explicitly.

Sweep directory layout:
- `master_config.jsonc` stores the original commented sweep input
- `manifest.json` stores the immutable case list and 12-character case IDs
- `status.json` stores mutable per-case state for resume and live monitoring
- `summary.json` stores the latest run summary
- `cases/<case_id>/config.json` stores the strict JSON config for that case
- `cases/<case_id>/run.log` stores the latest combined stdout/stderr for that case

Sweep config rules:
- `base` uses the normal single-run config fields
- `sweep` maps fields to candidate-value lists
- multiple sweep axes expand as a Cartesian product
- `number_of_repeats` controls how many separate generated runs are created for each Cartesian case (default `1`)
- list-valued run fields use a list of candidate full values (for example `subturn_event_order` as a list of lists)
- `output_dir`, `outputs_root`, `run_name`, `indexed_output`, and `index_csv` are generator-managed and not allowed in the master file
- generated leaf configs include a fixed absolute `output_dir`, so each run writes into its own `cases/<case_id>` directory

Output behavior:
- default mode writes to a readable folder name under `outputs/` (for example `outputs/promotion_committee_max_divergence_no_incentive`)
- indexed mode writes to `outputs/<run_id>` and appends one row per run to `outputs/index.csv`
- if output-related features are all disabled, no output directory is created
- `load_snapshot=true` requires `load_dir` (directory containing `debate_snapshot.json`)
- `show_plots=true` requires `save_plots=true`
- `subturn_event_order` drives event enablement directly:
- must include `public_utterance`
- include `private_utterance` to enable private reflection
- include `public_survey` to enable public survey
- include `private_survey` to enable private survey
- `keep_private_reflection`, `keep_public_survey`, and `keep_private_survey` require their respective events in `subturn_event_order`
- when outputs are enabled, each run folder contains run artifacts (`config.json`, plots, optional snapshot)
- `eval_data.json` is written only when at least one analysis stream is enabled (`semantic_analysis_metrics` or `persona_analysis_metrics`)
