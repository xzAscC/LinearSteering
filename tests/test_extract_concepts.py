from __future__ import annotations

import torch

import extract_concepts


def test_get_concept_vectors_uses_difference_in_means(monkeypatch) -> None:
    class DummyDiff:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

        def get_concept_vectors(self, save_path: str, is_save: bool = False):
            return torch.ones(2, 3)

    monkeypatch.setattr(extract_concepts, "DifferenceInMeans", DummyDiff)

    vec = extract_concepts.get_concept_vectors(
        model=None,
        positive_dataset=None,
        negative_dataset=None,
        layer=2,
        device="cpu",
        dataset_key="text",
        methods="difference-in-means",
        save_path="/tmp/unused.pt",
        max_dataset_size=2,
    )
    assert vec.shape == (2, 3)


def test_get_concept_vectors_invalid_method() -> None:
    try:
        extract_concepts.get_concept_vectors(
            model=None,
            positive_dataset=None,
            negative_dataset=None,
            layer=2,
            device="cpu",
            dataset_key="text",
            methods="unknown",
            save_path="/tmp/unused.pt",
        )
    except ValueError as exc:
        assert "Invalid method" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid method")
