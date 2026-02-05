from __future__ import annotations

import pytest
import torch

import steering_lda_probe


def test_train_eval_lda_probe_outputs() -> None:
    X = torch.tensor(
        [
            [0.0, 0.0],
            [0.1, 0.2],
            [1.0, 1.0],
            [1.1, 1.2],
        ]
    )
    y = torch.tensor([0.0, 0.0, 1.0, 1.0])
    (
        stats,
        weight,
        bias,
        mean0,
        mean1,
        cov,
        prior0,
        prior1,
    ) = steering_lda_probe.train_eval_lda_probe(
        X,
        y,
        seed=0,
        test_ratio=0.5,
        reg=1e-3,
    )
    assert stats["train_size"] + stats["test_size"] == 4
    assert weight.shape == (2,)
    assert cov.shape == (2, 2)
    assert 0.0 < prior0 < 1.0
    assert 0.0 < prior1 < 1.0


def test_train_eval_lda_probe_requires_two_classes() -> None:
    X = torch.zeros(3, 2)
    y = torch.zeros(3)
    with pytest.raises(RuntimeError):
        steering_lda_probe.train_eval_lda_probe(
            X,
            y,
            seed=0,
            test_ratio=0.5,
            reg=1e-3,
        )


def test_plot_probe_accuracy(tmp_path) -> None:
    results = {"1": {"0": {"test_acc": 0.7}}}
    steering_lda_probe.plot_probe_accuracy(
        results=results,
        probe_layers=[0],
        vector_name="concept",
        output_dir=str(tmp_path),
        steer_layer=0,
        alpha_mode="manual",
    )
    assert (tmp_path / "lda_probe_acc_concept_steer_0.png").exists()
