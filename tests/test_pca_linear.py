from __future__ import annotations

import torch

import pca_linear


def test_pca_statistics_aggregator_update() -> None:
    aggregator = pca_linear.PCAStatisticsAggregator()
    trajectory = torch.randn(3, 4, 5)
    aggregator.update(trajectory)
    stats = aggregator.finalize()
    assert 0.0 <= stats["mean_score"] <= 1.0
    assert stats["mean_n95"] >= 1.0


def test_pca_statistics_aggregator_empty() -> None:
    aggregator = pca_linear.PCAStatisticsAggregator()
    stats = aggregator.finalize()
    assert stats["mean_score"] == 1.0
    assert stats["mean_n95"] == 1.0
