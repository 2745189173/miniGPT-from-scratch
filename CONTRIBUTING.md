# Contributing

This is primarily a learning repository, but focused fixes, tests, explanations,
and controlled experiments are welcome.

## Development Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python scripts\download_data.py
```

## Before Submitting a Change

```powershell
ruff check .
ruff format --check .
python scripts\check_data.py
python scripts\check_bpe.py
python scripts\check_tokenizer_integration.py
python scripts\check_attention.py
python scripts\check_block.py
python scripts\check_model.py
python scripts\check_gradients.py
```

Keep changes scoped and preserve legacy checkpoint behavior unless a migration is
explicitly documented.

For a new experiment, record:

- the single changed variable and all fixed conditions;
- parameter count and training configuration;
- best/final validation metrics;
- a representative generated sample;
- the conclusion and next decision.

Do not commit raw datasets, model checkpoints, generated plots, or local secrets.
