# src

Research scripts and utilities for linear steering, concept vectors, and probe analysis.

## Run scripts

Use `uv` to run any script:

```bash
uv run python src/<script>.py
```

## Run tests

Run the full test suite:

```bash
uv run pytest
```

Run a single test file:

```bash
uv run pytest tests/test_utils.py
```

## Common modules

- `src/utils.py`: shared constants and general utilities (models, datasets, plotting).
- `src/probe_utils.py`: shared helpers for steering/probe workflows.
- `src/plot/plot_probe_utils.py`: shared helpers for plotting probe weights/components.

## Key scripts

- `src/extract_concepts.py`: build concept vectors with difference-in-means.
- `src/pca_linear.py`: compute linearity scores and n95 statistics.
- `src/plot/plot_linear.py`: plot linearity and n95 curves across models.
- `src/plot/plot_linear_n95.py`: focused n95 comparison plots.
- `src/steering_linear_probe.py`: linear probe before/after steering.
- `src/steering_lda_probe.py`: LDA probe before/after steering.
- `src/steering_pca_probe.py`: PCA probe before/after steering.
- `src/plot/plot_linear_probe.py`: grid plots of linear probe accuracy.
- `src/plot/plot_linear_probe_weights.py`: cosine heatmaps for linear probe weights.
- `src/plot/plot_lda_probe_weights.py`: cosine heatmaps for LDA probe weights.
- `src/plot/plot_pca_probe_eigenvectors.py`: max-cosine heatmaps for PCA components.
- `src/plot/plot_alpha_delta_cosine.py`: cosine similarity of mean deltas across alphas.
