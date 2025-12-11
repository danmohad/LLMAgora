# LLM Agora

A minimal arena where LLM-backed agents take public turns (optionally preceded by private reflections) until each reaches a configurable quota.


## Requirements

- Python >=3.12
- An OpenRouter API key stored in `.env`


## Setup

1. Create a virtualenv via [uv](https://github.com/astral-sh/uv) (or your tool of choice):
   ```bash
   uv venv --python 3.12.4
   source .venv/bin/activate
   uv pip install -r requirements.txt
   uv pip install -e .
   ```
2. Add your OpenRouter API key to a `.env` file in the repo root:
   ```bash
   OPENROUTER_API_KEY=sk-...
   ```

## Running tests

```bash
python -m pytest
```

Tests monkeypatch the `LLMClient`, so no external calls occur. Running `uv pip install -e .` keeps pytest in sync with local code.

## Notebook demos

Ready-to-run walkthroughs live in `notebooks/demo.ipynb` (basic debate flows) and `notebooks/demo_persona.ipynb` (persona-driven debates with plotting). Both notebooks now call into reusable helpers rather than embedding logic inline.

## CLI

Install the package in editable mode (`uv pip install -e .`) to expose the `agora` command:

```bash
# Run with a JSON agent configuration
agora run --config path/to/agents.json --turns 2 --verbose

# Run the persona demo directly from the datasets
agora persona --alpha-id high_wealth_founder --beta-id unionized_warehouse_worker --question-id work --prompt-set default --verbose
```

Prompt templates live in YAML under `src/agora/prompts`. Use `--prompt-set` to choose which template file to load (e.g., `default` reads `src/agora/prompts/default.yaml`).