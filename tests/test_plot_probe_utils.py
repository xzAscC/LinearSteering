from __future__ import annotations

import torch

from plot import plot_probe_utils


def test_weight_label_from_payload() -> None:
    payload = {"alpha_percent": 10.0, "steer_layer": 3, "layer": 7}
    assert (
        plot_probe_utils.weight_label_from_payload(payload, "fallback")
        == "10%_steer3_layer7"
    )


def test_parse_optional_layers() -> None:
    assert plot_probe_utils.parse_optional_layers("all") is None
    assert plot_probe_utils.parse_optional_layers("1, 2") == [1, 2]


def test_find_probe_layers(tmp_path) -> None:
    steer_dir = tmp_path / "steer_0"
    (steer_dir / "alpha_1").mkdir(parents=True)
    (steer_dir / "alpha_1" / "layer_3.pt").write_bytes(b"")
    layers = plot_probe_utils.find_probe_layers(str(steer_dir))
    assert layers == [3]


def test_load_weight_tensor_dict(tmp_path) -> None:
    path = tmp_path / "weight.pt"
    payload = {"raw_weight": torch.tensor([1.0, 2.0])}
    torch.save(payload, path)
    label, weight = plot_probe_utils.load_weight_tensor(
        str(path),
        label_fallback="fallback",
        prefer_raw_weight=True,
    )
    assert label == "fallback"
    torch.testing.assert_close(weight, torch.tensor([1.0, 2.0]))


def test_load_probe_weights_for_steer_layer(tmp_path) -> None:
    steer_dir = tmp_path / "steer_1"
    alpha_dir = steer_dir / "alpha_1"
    alpha_dir.mkdir(parents=True)
    torch.save({"weight": torch.tensor([1.0, 0.0])}, alpha_dir / "layer_2.pt")
    labels, weights = plot_probe_utils.load_probe_weights_for_steer_layer(
        str(steer_dir), 2
    )
    assert labels
    assert weights.shape == (1, 2)


def test_load_components_for_steer_layer(tmp_path) -> None:
    steer_dir = tmp_path / "steer_1"
    alpha_dir = steer_dir / "alpha_1"
    alpha_dir.mkdir(parents=True)
    torch.save({"components": torch.eye(2)}, alpha_dir / "layer_2.pt")
    labels, components = plot_probe_utils.load_components_for_steer_layer(
        str(steer_dir), 2
    )
    assert labels
    assert components[0].shape == (2, 2)


def test_compute_cosine_similarity() -> None:
    weights = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    cosine = plot_probe_utils.compute_cosine_similarity(weights)
    torch.testing.assert_close(cosine, torch.eye(2))


def test_compute_max_cosine_matrix() -> None:
    components = [torch.eye(2), torch.tensor([[1.0, 0.0], [1.0, 0.0]])]
    max_cos = plot_probe_utils.compute_max_cosine_matrix(components)
    assert max_cos.shape == (2, 2)
    assert max_cos[0, 0] == 1.0
    assert max_cos[0, 1] >= 0.99


def test_plot_cosine_heatmap(tmp_path) -> None:
    output_path = tmp_path / "heatmap.png"
    cosine = torch.eye(2)
    plot_probe_utils.plot_cosine_heatmap(
        cosine,
        labels=["a", "b"],
        title="test",
        output_path=str(output_path),
    )
    assert output_path.exists()
    assert output_path.with_suffix(".pdf").exists()
