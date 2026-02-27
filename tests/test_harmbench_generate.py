from __future__ import annotations

from pathlib import Path

from benchmark_eval import harmbench_generate


def test_resolve_run_mode_defaults_from_steering_mode() -> None:
    assert harmbench_generate.resolve_run_mode("no_steering", None) == "unsteering"
    assert harmbench_generate.resolve_run_mode("dim_100_tokens", None) == "steering"


def test_resolve_run_mode_allows_override() -> None:
    assert harmbench_generate.resolve_run_mode("no_steering", "steering") == "steering"


def test_resolve_steering_mode_vector_path_no_steering() -> None:
    path = harmbench_generate.resolve_steering_mode_vector_path(
        steering_mode="no_steering",
        model_name="Qwen3-1.7B",
        steer_vector="steering_safety",
        dim_100_vector=None,
        dim_all_vector=None,
        steering_required=False,
    )
    assert path is None


def test_resolve_steering_mode_vector_path_dim_all_uses_concept_name() -> None:
    path = harmbench_generate.resolve_steering_mode_vector_path(
        steering_mode="dim_all_tokens",
        model_name="Qwen3-1.7B",
        steer_vector="steering_safety",
        dim_100_vector=None,
        dim_all_vector=None,
        steering_required=True,
    )
    assert path == "assets/concept_vectors/Qwen3-1.7B/steering_safety.pt"


def test_resolve_latest_dim_100_vector_uses_newest_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_name = "Qwen3-1.7B"
    old_path = tmp_path / "pku_pos_neg_center_token_layer15_old.pt"
    new_path = tmp_path / "pku_pos_neg_center_token_layer15_new.pt"
    old_path.write_bytes(b"old")
    new_path.write_bytes(b"new")

    monkeypatch.setattr(
        harmbench_generate,
        "glob",
        lambda _pattern: [str(old_path), str(new_path)],
    )
    monkeypatch.setattr(
        harmbench_generate.os.path,
        "isfile",
        lambda _path: True,
    )
    monkeypatch.setattr(
        harmbench_generate.os.path,
        "getmtime",
        lambda path: 1.0 if path == str(old_path) else 2.0,
    )

    resolved = harmbench_generate.resolve_latest_dim_100_vector(model_name)
    assert resolved == str(new_path)


def test_resolve_latest_dim_100_vector_raises_without_candidates(
    monkeypatch,
) -> None:
    monkeypatch.setattr(harmbench_generate, "glob", lambda _pattern: [])

    try:
        harmbench_generate.resolve_latest_dim_100_vector("Qwen3-1.7B")
    except FileNotFoundError as exc:
        assert "100-token" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError when no vectors are found")
