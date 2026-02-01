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
pytest --cov=agora
```

Tests monkeypatch the `LLMClient`, so no external calls occur. Running `uv pip install -e .` keeps pytest in sync with local code.
Tests assume the package is installed (editable install recommended).

## GitHub PR pipeline

Runs `pytest` with coverage report, and comments the coverage percentage on the PR. Note that it doesn't install the package in 'analysis' mode, because the `sentence-transformers` package is too large for GitHub free nodes.

## Notebook demos

Ready-to-run walkthroughs live in `notebooks/demo.ipynb` (basic debate flows) and `notebooks/demo_persona.ipynb` (persona-driven debates with plotting). Both notebooks now call into reusable helpers rather than embedding logic inline.
For `demo_persona_eval.ipynb` (semantic analysis), the analysis extra is required; it's included if you used the setup command above.

## CLI

Install the package in editable mode (`uv pip install -e .`) to expose the `agora` command:

```bash
# Use the persona config; the example mirrors the `demo_persona.ipynb` notebook
agora run --config data/agents_persona_example.json --verbose

# Run the persona demo directly from the datasets
agora persona \
  --alpha-id high_wealth_founder \
  --beta-id unionized_warehouse_worker \
  --question-id work \
  --prompts data/prompts.json \
  --prompt-set default \
  --keep-private-response \
  --no-keep-pre-interview \
  --no-keep-post-interview \
  --verbose
```
Prompt templates (base prompt, public and private instructions, etc.) live in JSON under `data/prompts.json`. Use `--prompt-set` to choose which template entry to load (e.g., `default`).
