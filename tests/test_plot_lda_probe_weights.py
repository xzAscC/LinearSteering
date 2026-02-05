from __future__ import annotations

import sys

import torch

import plot_lda_probe_weights


def test_main_runs_with_minimal_assets(tmp_path, monkeypatch) -> None:
    model_full = "Qwen/Qwen3-1.7B"
    model_name = "Qwen3-1.7B"
    weight_dir = (
        tmp_path
        / "assets"
        / "lda_probe"
        / model_name
        / "probe_weights"
        / "concept_a"
        / "steer_0"
        / "alpha_1"
    )
    weight_dir.mkdir(parents=True)
    torch.save({"weight": torch.tensor([1.0, 0.0])}, weight_dir / "layer_0.pt")

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

    plot_lda_probe_weights.main()
    output_path = (
        tmp_path / "plots" / model_name / "concept_a" / "steer_0" / "probe_0.png"
    )
    assert output_path.exists()
