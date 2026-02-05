# AGENTS

Purpose
- This repository contains research scripts for linear steering, concept vectors, and linearity/probing analyses.
- Primary entry points live in `src/` and are run as Python scripts.

Quick Start
- Python version: 3.13+ (see `pyproject.toml`).
- Install dependencies with `uv sync`.
- Run scripts directly with `uv run python src/<script>.py`.

Build / Lint / Test
- Lint: `uv run ruff check src`
- Tests: `uv run pytest`

Single Test
- `uv run pytest tests/test_smoke.py`

Common Commands
- Install dependencies: `uv sync`
- Concept vectors: `uv run python src/extract_concepts.py --model Qwen/Qwen3-1.7B --concept_category steering_detectable_format_json_format`
- PCA linearity sweep: `uv run python src/pca_linear.py --model Qwen/Qwen3-1.7B --layers 0,6,13,20,27`
- Plot results: `uv run python src/plot_linear.py` or `uv run python src/plot_linear_n95.py`
- Steering probe (custom): `uv run python src/steering_linear_probe.py`

Repository Layout
- `src/`: research scripts and utilities
- `assets/`: concept vectors, linearity results, and derived artifacts
- `dataset/`: prompt datasets (JSONL)
- `logs/`, `plots/`: outputs from analysis scripts

Code Style Guidelines

Imports
- Use standard library imports first, then third-party packages, then local imports.
- Keep one import per line for readability; group related imports by section.
- Prefer absolute imports within `src/` (e.g., `from utils import ...`).

Formatting
- Follow PEP 8 conventions for spacing and line breaks.
- Use 4-space indentation and keep lines reasonably short.
- Prefer explicit parentheses for multi-line calls and argument lists.

Types
- Type hints are used in utilities and data loading; keep or add hints where they clarify shapes or return types.
- Use `torch.Tensor` for tensors and `Dict`, `List`, `Tuple` from `typing` for collections.
- When returning multiple tensors, use `tuple[torch.Tensor, ...]` or `Tuple[torch.Tensor, ...]` consistently.

Naming
- Functions and variables: `snake_case`.
- Classes: `CamelCase`.
- Constants: `UPPER_SNAKE_CASE` (see `MODEL_LAYERS`, `CONCEPT_CATEGORIES`).
- File names: `snake_case.py`.

Error Handling
- Use `ValueError` for invalid user arguments or configuration.
- Use `RuntimeError` for unexpected runtime failures (e.g., missing captured tensors).
- Log errors with `loguru` where relevant, especially in multiprocess code.

Logging
- Prefer `loguru.logger` for progress and status messages.
- Avoid noisy output inside tight loops unless gated by rank or verbosity.

Reproducibility
- Use `set_seed()` from `src/utils.py` to seed Python, NumPy, and Torch.
- When sampling randomized values for experiments, derive seeds from names with `seed_from_name()`.

Torch / Transformers Practices
- Use `torch.no_grad()` for inference-only passes.
- Handle device selection via `cuda` availability; default to `cpu` if CUDA is not present.
- Use `torch.bfloat16` for GPU models where possible; fall back to `float32` on CPU.
- When modifying activations, prefer hooks and remove them reliably (try/finally).

Data Handling
- Datasets are loaded from JSONL in `dataset/steering_tasks.jsonl`.
- For concept vectors, follow `CONCEPT_CATEGORIES` mapping in `src/utils.py`.
- Avoid hardcoding dataset paths; use utilities in `src/extract_concepts.py`.

Steering Vectors
- Concept vectors are stored in `assets/concept_vectors/<model_name>/`.
- Random directions follow `random_direction_<n>.pt`.
- Per-layer concept vectors are shaped `[num_layers, d_model]`.

Script Conventions
- Use `argparse` for CLI parameters.
- Provide defaults that match common use (e.g., Qwen3-1.7B).
- Validate CLI inputs early with clear errors.

Multiprocessing
- `src/pca_linear.py` uses `torch.multiprocessing` with `spawn`.
- Avoid sharing CUDA tensors across processes without explicit handling.
- Use queues to distribute concept tasks across workers.

Assets and Outputs
- Avoid overwriting existing assets unless the script is intended to regenerate them.
- Save outputs into `assets/` or `logs/` with clear, descriptive file names.

Agentic Changes
- Keep edits minimal and aligned with existing style.
- If adding new scripts, place them in `src/` and keep CLI args explicit.
- If adding dependencies, update `pyproject.toml`.

Missing Rules
- No `.cursor/rules/`, `.cursorrules`, or `.github/copilot-instructions.md` present.
- If these files are added later, update this document to include their guidance.

Notes
- README is currently minimal; do not assume additional workflow steps beyond scripts.
- There is no formal test suite; validate changes by running the relevant scripts.
