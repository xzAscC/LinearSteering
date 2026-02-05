from __future__ import annotations

import torch

import utils


def test_get_model_name_for_path() -> None:
    assert utils.get_model_name_for_path("google/gemma-2-2b") == "gemma-2-2b"


def test_parse_layers_to_run_all() -> None:
    assert utils.parse_layers_to_run("all", max_layers=5) == [0, 1, 2, 3]


def test_parse_layers_to_run_percent() -> None:
    layers = utils.parse_layers_to_run("50", max_layers=20)
    assert layers == [10]


def test_parse_layers_to_run_indices() -> None:
    layers = utils.parse_layers_to_run("2,4,6", max_layers=10, is_percentage=False)
    assert layers == [2, 4, 6]


def test_build_concept_renames() -> None:
    concepts = ["steering_change_case"]
    renames = utils.build_concept_renames(
        concepts,
        replacements={"Change": "Case"},
        strip_prefix="steering_",
        title_case=True,
    )
    assert renames["steering_change_case"] == "Case Case"


def test_layer_depth_percent() -> None:
    assert utils.layer_depth_percent([0, 2], max_layers=1) == [0.0, 0.0]
    assert utils.layer_depth_percent([0, 2], max_layers=5) == [0.0, 50.0]


def test_vector_style_variants() -> None:
    concept_style = utils.vector_style("concept", False)
    random_style = utils.vector_style("random", False)
    removed_style = utils.vector_style("concept", True)
    assert concept_style["marker"] == "*"
    assert random_style["linestyle"] == "--"
    assert removed_style["hollow"]


def test_hidden_to_flat_dtype() -> None:
    h = torch.zeros(2, 3, 4)
    flat = utils.hidden_to_flat(h, target_dtype=torch.float32)
    assert flat.shape == (6, 4)
    assert flat.dtype == torch.float32


def test_get_layers_container_model_attribute() -> None:
    class DummyLayers:
        def __init__(self) -> None:
            self.layers = [object(), object()]

    class DummyModel:
        def __init__(self) -> None:
            self.model = DummyLayers()

    layers = utils._get_layers_container(DummyModel())
    assert len(layers) == 2


def test_apply_steering_output_tensor() -> None:
    output = torch.zeros(2, 3)
    steering = torch.ones(2, 3)
    updated = utils._apply_steering_output(output, steering, alpha_value=2.0)
    torch.testing.assert_close(updated, torch.full((2, 3), 2.0))


def test_load_linearity_scores(tmp_path, monkeypatch) -> None:
    payload = {
        "results": {
            0: {
                "mean_score": 0.1,
                "std_score": 0.01,
                "n_components_95_mean": 2.0,
                "n_components_95_std": 0.2,
            },
            "1": 0.2,
        }
    }
    path = tmp_path / "linear.pt"
    torch.save(payload, path)
    monkeypatch.setattr(
        utils, "_linearity_result_path", lambda *args, **kwargs: str(path)
    )
    layers, means, stds, n95_means, n95_stds = utils.load_linearity_scores(
        "model",
        "concept",
        "vector",
        False,
    )
    assert layers == [0, 1]
    assert means == [0.1, 0.2]
    assert stds == [0.01, 0.0]
    assert n95_means == [2.0, 0.0]
    assert n95_stds == [0.2, 0.0]


def test_load_linearity_n95(tmp_path, monkeypatch) -> None:
    payload = {
        "results": {
            0: {"n_components_95_mean": 3.0},
            "1": {"n_components_95": 4.0},
        }
    }
    path = tmp_path / "linear.pt"
    torch.save(payload, path)
    monkeypatch.setattr(
        utils, "_linearity_result_path", lambda *args, **kwargs: str(path)
    )
    layers, scores = utils.load_linearity_n95("model", "concept", "vector", False)
    assert layers == [0, 1]
    assert scores == [3.0, 4.0]
