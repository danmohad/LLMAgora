# LLM Agora

A minimal arena where LLM-backed agents take public turns until each reaches a configurable quota. This slice focuses on:

- Agent-specific memory via `MemoryTurn` records
- Public-only speech routed through an `LLMClient` abstraction
- An `Agora` coordinator that alternates turns and exposes history views
- Pytest coverage using a stub LLM client (no network calls during tests)

## Requirements

- Python 3.12
- An OpenRouter API key stored in `.env`

## Project layout

```
src/agora/          # Core package (agents, memory, LLM clients, Agora)
tests/              # Pytests + stub LLM helpers
requirements.txt    # Runtime + dev dependencies (install before editable mode)
setup.py            # Editable install metadata
notebooks/demo.ipynb  # Live walkthrough hitting OpenRouter
```

## Setup

1. Create a virtualenv via [uv](https://github.com/astral-sh/uv) (or your tool of choice):
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -r requirements.txt
   uv pip install -e .
   ```
2. Add your OpenRouter API key to a `.env` file in the repo root:
   ```bash
   echo "OPENROUTER_API_KEY=sk-..." > .env
   ```
3. (Optional, recommended for notebooks) Register the virtualenv’s IPython kernel once:
   ```bash
   python -m ipykernel install --user --name llm-agora
   ```

## Running tests

```bash
pytest
```

Tests monkeypatch the `LLMClient`, so no external calls occur. Running `uv pip install -e .` keeps pytest in sync with local code for tight feedback loops.

## Notebook demo

A ready-to-run walkthrough lives in `notebooks/demo.ipynb`. Launch Jupyter (e.g., `jupyter notebook notebooks/demo.ipynb`), select the `llm-agora` kernel, tweak the `agent_configs` / `turns_per_agent` cells, and run the notebook to watch live OpenRouter-powered conversations between your configured agents.
