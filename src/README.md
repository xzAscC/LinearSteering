# src

Research scripts and utilities for linear steering, concept vectors, and probe analysis.

## Run scripts

Use `uv` to run any script:

```bash
uv run python src/<script>.py
```

## Common modules

- `src/utils.py`: shared constants and general utilities (models, datasets, plotting).
- `src/probe_utils.py`: shared helpers for steering/probe workflows.
- `src/plot_probe_utils.py`: shared helpers for plotting probe weights/components.

## Key scripts

- `src/extract_concepts.py`: build concept vectors with difference-in-means.
- `src/pca_linear.py`: compute linearity scores and n95 statistics.
- `src/plot_linear.py`: plot linearity and n95 curves across models.
- `src/plot_linear_n95.py`: focused n95 comparison plots.
- `src/steering_linear_probe.py`: linear probe before/after steering.
- `src/steering_lda_probe.py`: LDA probe before/after steering.
- `src/steering_pca_probe.py`: PCA probe before/after steering.
- `src/plot_linear_probe.py`: grid plots of linear probe accuracy.
- `src/plot_linear_probe_weights.py`: cosine heatmaps for linear probe weights.
- `src/plot_lda_probe_weights.py`: cosine heatmaps for LDA probe weights.
- `src/plot_pca_probe_eigenvectors.py`: max-cosine heatmaps for PCA components.
- `src/plot_alpha_delta_cosine.py`: cosine similarity of mean deltas across alphas.
