# LLM Agora

A minimal arena where LLM-backed agents take public turns (optionally preceded by private reflections) until each reaches a configurable quota.


## Requirements

- Python 3.12
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

## Notebook demo

A ready-to-run walkthrough lives in `notebooks/demo.ipynb`. Launch Jupyter and select the local kernel (i.e., `.venv`).
