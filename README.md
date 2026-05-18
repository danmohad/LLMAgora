# LLM Agora

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

## Use

### Notebook demo

Run `notebooks/run_demo.ipynb` for a single configurable run that can save a `debate_snapshot.json`.
The notebook is intentionally thin and calls the high-level workflow in `agora.experiment`.

### CLI

Install the package in editable mode (`uv pip install -e .`) to expose the `agora` command.

Canonical single-run config lives at `data/config_example.json`. Relative paths inside JSON config files are resolved relative to the file that contains them, so the checked-in config uses `scenarios.json`, `prompts.json`, and `../outputs`. CLI-only paths are resolved from the current working directory.

Optional retention, output, and analysis features are disabled by default (`false` flags and empty analysis metric lists). Analysis backends must be configured explicitly when their metric lists are non-empty; otherwise config validation fails. Prompt templates live in `data/prompts.json`, and sweep generation uses the commented master template at `data/sweep_example.jsonc`.

```bash
# Run with config: this is the recommended way to run the code
# With the default config, no output directory is created unless you enable
# snapshots, surveys, indexed output, plots, or analysis.
agora run --config data/config_example.json
```

The CLI allows any setting to be overridden using flags:

```bash
# Override specific fields from config
# Semantic similarity requires the analysis extra.
agora run --config data/config_example.json \
  --scenario-id ngo_climate_endorsement \
  --incentive-direction positive \
  --incentive-type future \
  --semantic-analysis-metrics self_consistency cross_agent_public_alignment \
  --semantic-similarity-method cosine \
  --semantic-similarity-model all-mpnet-base-v2 \
  --save-plots

# Use NLI for semantic scoring
agora run --config data/config_example.json \
  --semantic-analysis-metrics self_consistency \
  --semantic-similarity-method nli \
  --semantic-similarity-model dleemiller/finecat-nli-l \
  --semantic-similarity-device mps

# Run persona adherence analysis for selected metric slices only
agora run --config data/config_example.json \
  --persona-analysis-metrics public_per_turn full_debate_public \
  --persona-scoring-model anthropic/claude-sonnet-4 \
  --persona-score-samples 3

# Run with no config file (all args via CLI)
agora run \
  --scenario-id promotion_committee \
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

# Save a snapshot that can be resumed later
agora run --config data/config_example.json \
  --save-snapshot \
  --run-name snapshot_demo

# Resume from that snapshot directory
agora run --config data/config_example.json \
  --load-snapshot \
  --load-dir outputs/snapshot_demo

# Indexed output mode: run folder is a short unique ID and index row is appended
agora run --config data/config_example.json --indexed-output
```

### Sweep Workflows

Use `agora sweep` to generate and run parameter sweeps. The checked-in sweep example writes to `outputs/sweeps/example` from the repo root and generates one case for each `incentive_type` value. `agora sweep run` owns the terminal and renders the live dashboard until the batch completes.

```bash
# Expand the master config into manifest/status plus cases/<case_id>/config.json
agora sweep generate --config data/sweep_example.jsonc

# Run all cases that are not already marked succeeded.
# This command renders the live in-place dashboard in the current TTY.
agora sweep run

# Re-run failed or interrupted cases only
agora sweep run --root outputs/sweeps/example --mode failed

# Keep retrying selected failed attempts until every selected case succeeds
agora sweep run --root outputs/sweeps/example --mode failed --persistent

# Build one aggregate JSON record for the parameter sweep.
agora sweep aggregate
```

When `--root` is omitted, `agora sweep run` and `agora sweep aggregate` infer it from the only non-generated
`.jsonc` sweep config in the current working tree; if there is not exactly one,
pass `--root` explicitly.

Sweep directory layout:
- `master_config.jsonc` stores the original commented sweep input
- `manifest.json` stores the immutable case list and 12-character case IDs
- `status.json` stores mutable per-case state for resume and live monitoring
- `summary.json` stores the latest run summary
- `cases/<case_id>/config.json` stores the strict JSON config for that case
- `cases/<case_id>/run.log` stores the latest combined stdout/stderr for that case
- `aggregate_analysis.json` stores the default aggregate table produced by `agora sweep aggregate`

Sweep config rules:
- `base` uses the normal single-run config fields
- `sweep` maps fields to candidate-value lists
- `aggregation` is optional and configures `agora sweep aggregate`
- multiple sweep axes expand as a Cartesian product
- `number_of_repeats` controls how many separate generated runs are created for each Cartesian case (default `1`)
- list-valued run fields use a list of candidate full values (for example `subturn_event_order` as a list of lists)
- `output_dir`, `outputs_root`, `run_name`, `indexed_output`, and `index_csv` are generator-managed and not allowed in the master file
- generated leaf configs include a fixed absolute `output_dir` and force `save_snapshot=true`, so each run writes replayable artifacts into its own `cases/<case_id>` directory
- aggregation defaults write `aggregate_analysis.json` under the sweep root with no extra offline post-processing
- aggregation `analysis` overrides use the normal analysis field names, such as `semantic_analysis_metrics`, `persona_analysis_metrics`, and their model/device controls
- aggregation can opt into extra aggregate-only analyses with `include_nli`, `include_emotions`, and `strip_decision_labels`
- `strip_decision_labels` controls aggregate semantic/NLI analysis after removing leading scenario decision labels from utterances: `off`, `include`, or `only`

Output behavior:
- default mode writes to a readable folder name under `outputs/` (for example `outputs/promotion_committee_no_incentive`)
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
