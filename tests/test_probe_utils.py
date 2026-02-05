from __future__ import annotations

import torch

import probe_utils


def test_alpha_to_slug() -> None:
    assert probe_utils.alpha_to_slug(-1.5) == "m1p5"


def test_alpha_to_percent() -> None:
    assert probe_utils.alpha_to_percent(10.0) == 0.1


def test_get_alpha_label_manual() -> None:
    assert probe_utils.get_alpha_label(10.0, "manual", []) == "0.1%"


def test_get_alpha_label_avg_norm() -> None:
    label = probe_utils.get_alpha_label(1.0, "avg_norm", [1.0, 10.0])
    assert label == "1"


def test_select_steer_layers_small() -> None:
    assert probe_utils.select_steer_layers(1) == [0]
    assert all(l <= 4 for l in probe_utils.select_steer_layers(6))


def test_select_probe_layers_edges() -> None:
    assert probe_utils.select_probe_layers(10, max_layers=5, total_layers=4) == []
    assert probe_utils.select_probe_layers(2, max_layers=6, total_layers=1) == [2]


def test_pick_segment_layers_invalid() -> None:
    assert probe_utils._pick_segment_layers(3, 1, 2) == []


def test_load_vector_tensor(tmp_path) -> None:
    path = tmp_path / "vec.pt"
    tensor = torch.ones(3)
    torch.save(tensor, path)
    loaded = probe_utils.load_vector(str(path))
    torch.testing.assert_close(loaded, tensor)


def test_load_vector_random_dict(tmp_path) -> None:
    path = tmp_path / "vec.pt"
    tensor = torch.zeros(4)
    torch.save({"random_vector": tensor}, path)
    loaded = probe_utils.load_vector(str(path))
    torch.testing.assert_close(loaded, tensor)
