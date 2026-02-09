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
