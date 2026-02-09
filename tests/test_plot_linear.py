from __future__ import annotations

from plot import plot_linear


def test_plot_linearity_creates_output(tmp_path, monkeypatch) -> None:
    def fake_scores(*_args, **_kwargs):
        return [0, 1], [0.1, 0.2], [0.01, 0.02], [2.0, 3.0], [0.2, 0.3]

    monkeypatch.setattr(plot_linear, "CONCEPT_CATEGORIES", {"c1": {}, "c2": {}})
    monkeypatch.setattr(
        plot_linear,
        "MODEL_LAYERS",
        {"Qwen/Qwen3-1.7B": 4, "google/gemma-2-2b": 4},
    )
    monkeypatch.setattr(plot_linear, "load_linearity_scores", fake_scores)
    monkeypatch.chdir(tmp_path)

    plot_linear.plot_linearity()
    assert (tmp_path / "plots" / "linearity_multimodel_combined.pdf").exists()
