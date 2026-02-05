from __future__ import annotations

import pytest
import torch

import steering_pca_probe


def test_train_eval_pca_probe_outputs() -> None:
    X_before = torch.tensor([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]])
    X_after = torch.tensor([[1.0, 1.0], [1.1, 1.1], [1.2, 1.2]])
    stats, components, delta_mean = steering_pca_probe.train_eval_pca_probe(
        X_before,
        X_after,
        seed=0,
        test_ratio=0.33,
        n_components=2,
    )
    assert stats["train_size"] + stats["test_size"] == 3
    assert components.shape[1] == 2
    assert delta_mean.shape == (2,)


def test_train_eval_pca_probe_mismatch() -> None:
    X_before = torch.zeros(2, 2)
    X_after = torch.zeros(3, 2)
    with pytest.raises(RuntimeError):
        steering_pca_probe.train_eval_pca_probe(
            X_before,
            X_after,
            seed=0,
            test_ratio=0.5,
            n_components=2,
        )


def test_plot_probe_accuracy(tmp_path) -> None:
    results = {"1": {"0": {"test_acc": 0.7}}}
    steering_pca_probe.plot_probe_accuracy(
        results=results,
        probe_layers=[0],
        vector_name="concept",
        output_dir=str(tmp_path),
        steer_layer=0,
        alpha_mode="manual",
    )
    assert (tmp_path / "pca_probe_acc_concept_steer_0.png").exists()
