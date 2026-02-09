# LinearSteering

Research scripts for studying linear steering vectors, layerwise linearity, and
probe behavior in decoder language models.

## Requirements

- Python `>=3.13`
- `uv` for dependency and environment management
- CUDA GPU recommended for model runs (CPU is supported but slow)

## Setup

```bash
uv sync
```

## Quick start

Run scripts directly from the repository root:

```bash
uv run python src/<script>.py
```

Common workflows:

```bash
# 1) Extract concept vectors
uv run python src/extract_concepts.py --model Qwen/Qwen3-1.7B --concept_category steering_detectable_format_json_format

# 2) Run PCA linearity sweep
uv run python src/pca_linear.py --model Qwen/Qwen3-1.7B --layers 0,6,13,20,27

# 3) Plot linearity metrics
uv run python src/plot/plot_linear.py
uv run python src/plot/plot_linear_n95.py

# 4) Run steering probe experiments
uv run python src/steering_linear_probe.py
uv run python src/steering_lda_probe.py
uv run python src/steering_pca_probe.py
```

## Testing and linting

```bash
uv run ruff check src
uv run pytest
```

Single test example:

```bash
uv run pytest tests/test_smoke.py
```

## Repository layout

- `src/`: research scripts and shared utilities
- `tests/`: pytest suite for core scripts and plotting helpers
- `dataset/`: JSONL task/prompt datasets
- `assets/`: concept vectors and computed experiment artifacts
- `logs/`, `plots/`: generated outputs

## Notes

- Concept vectors are stored under `assets/concept_vectors/<model_name>/`.
- Many scripts default to `Qwen/Qwen3-1.7B`; override with `--model` as needed.
- For script-specific flags, use:

```bash
uv run python src/<script>.py --help
```
