from __future__ import annotations

import sys

import torch

from plot import plot_pca_probe_eigenvectors


def test_main_runs_with_minimal_assets(tmp_path, monkeypatch) -> None:
    model_full = "Qwen/Qwen3-1.7B"
    model_name = "Qwen3-1.7B"
    comp_dir = (
        tmp_path
        / "assets"
        / "pca_probe"
        / model_name
        / "probe_components"
        / "concept_a"
        / "steer_0"
        / "alpha_1"
    )
    comp_dir.mkdir(parents=True)
    torch.save({"components": torch.eye(2)}, comp_dir / "layer_0.pt")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--model",
            model_full,
            "--concepts",
            "concept_a",
            "--steer_layers",
            "0",
            "--probe_layers",
            "0",
            "--output_dir",
            str(tmp_path / "plots"),
        ],
    )

    plot_pca_probe_eigenvectors.main()
    output_path = (
        tmp_path / "plots" / model_name / "concept_a" / "steer_0" / "probe_0.png"
    )
    assert output_path.exists()
