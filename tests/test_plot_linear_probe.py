from __future__ import annotations

import json

from plot import plot_linear_probe


def test_load_probe_results(tmp_path) -> None:
    payload = {"probe_layers": [0], "alpha_values": [1], "results": {"1": {}}}
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload))
    results = plot_linear_probe.load_probe_results(str(tmp_path))
    assert results and results[0]["_path"] == str(path)


def test_format_layer_label() -> None:
    assert plot_linear_probe.format_layer_label(3) == "L3"


def test_plot_layer_grid(tmp_path) -> None:
    results = [
        {
            "probe_layers": [0],
            "alpha_values": [1, 10],
            "vector": "concept",
            "results": {
                "1": {"0": {"test_acc": 0.6}},
                "10": {"0": {"test_acc": 0.8}},
            },
        }
    ]
    output_path = tmp_path / "grid.png"
    plot_linear_probe.plot_layer_grid(
        results,
        title="Test",
        output_path=str(output_path),
        ncols=1,
    )
    assert output_path.exists()


def test_resolve_alpha_axis_uses_scales_for_avg_norm() -> None:
    alpha_axis = plot_linear_probe._resolve_alpha_axis(
        ["0.01", "0.1", "1.0"],
        {
            "alpha_mode": "avg_norm",
            "alpha_values": [0.01, 0.1, 1.0],
            "legacy_alpha_values_percent": [1, 10, 100],
        },
    )
    assert alpha_axis == [0.01, 0.1, 1.0]


def test_resolve_alpha_axis_uses_legacy_percent_fallback() -> None:
    alpha_axis = plot_linear_probe._resolve_alpha_axis(
        ["12.3", "45.6"],
        {
            "alpha_mode": "avg_norm",
            "alpha_values": [],
            "legacy_alpha_values_percent": [0.01, 0.1],
        },
    )
    assert alpha_axis == [0.01, 0.1]


def test_resolve_alpha_axis_label_avg_norm() -> None:
    label = plot_linear_probe._resolve_alpha_axis_label(
        [{"alpha_mode": "avg_norm"}, {"alpha_mode": "avg_norm"}]
    )
    assert label == "Alpha scale (x avg norm)"
