from __future__ import annotations

import torch

import steering_linear_probe


def test_train_eval_linear_probe_outputs() -> None:
    X = torch.tensor(
        [
            [0.0, 0.0],
            [0.1, 0.2],
            [1.0, 1.0],
            [1.1, 1.2],
        ]
    )
    y = torch.tensor([0.0, 0.0, 1.0, 1.0])
    stats, weight, bias, mean, std = steering_linear_probe.train_eval_linear_probe(
        X,
        y,
        seed=42,
        epochs=10,
        lr=0.1,
        test_ratio=0.5,
    )
    assert stats["train_size"] + stats["test_size"] == 4
    assert weight.shape == (2,)
    assert mean.shape == (2,)
    assert std.shape == (2,)


def test_plot_probe_accuracy(tmp_path) -> None:
    results = {"1": {"0": {"test_acc": 0.7}}}
    steering_linear_probe.plot_probe_accuracy(
        results=results,
        probe_layers=[0],
        vector_name="concept",
        output_dir=str(tmp_path),
        steer_layer=0,
        alpha_mode="manual",
    )
    assert (tmp_path / "probe_acc_concept_steer_0.png").exists()
