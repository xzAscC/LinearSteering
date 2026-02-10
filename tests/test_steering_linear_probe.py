from __future__ import annotations

import json
import sys
from pathlib import Path

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


def test_main_quick_verify_qwen_and_gemma(tmp_path, monkeypatch) -> None:
    class DummyInputs:
        def __init__(self, batch_size: int, seq_len: int = 3) -> None:
            self.input_ids = torch.ones((batch_size, seq_len), dtype=torch.long)
            self.attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)

        def to(self, _device: str):
            return self

    class DummyTokenizer:
        pad_token = None
        eos_token = "<eos>"

        def __call__(self, prompts, return_tensors, padding, truncation):
            del return_tensors, padding, truncation
            return DummyInputs(batch_size=len(prompts))

    class DummyModel:
        pass

    loaded_models = []

    def fake_model_from_pretrained(model_name, **kwargs):
        del kwargs
        loaded_models.append(model_name)
        return DummyModel()

    def fake_tokenizer_from_pretrained(model_name, **kwargs):
        del model_name, kwargs
        return DummyTokenizer()

    def fake_load_prompts_for_concepts(_concepts, max_prompts):
        return ["prompt-1", "prompt-2"][:max_prompts]

    def fake_load_vector(_path):
        return torch.ones(4)

    def fake_run_model_capture_layers(
        model,
        input_ids,
        capture_layers,
        device,
        capture_hook_point,
        steering_vector,
        steer_layer,
        alpha,
    ):
        del model, device, capture_hook_point, steering_vector
        batch_size, seq_len = input_ids.shape
        hidden_dim = 4
        captured = {}
        for layer_idx in capture_layers:
            base = torch.full((batch_size, seq_len, hidden_dim), float(layer_idx))
            if layer_idx == steer_layer:
                base = base + alpha
            else:
                base = base + alpha * 0.5
            captured[layer_idx] = base
        return captured

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        steering_linear_probe.transformers.AutoModelForCausalLM,
        "from_pretrained",
        fake_model_from_pretrained,
    )
    monkeypatch.setattr(
        steering_linear_probe.transformers.AutoTokenizer,
        "from_pretrained",
        fake_tokenizer_from_pretrained,
    )
    monkeypatch.setattr(
        steering_linear_probe,
        "load_prompts_for_concepts",
        fake_load_prompts_for_concepts,
    )
    monkeypatch.setattr(steering_linear_probe, "load_vector", fake_load_vector)
    monkeypatch.setattr(
        steering_linear_probe,
        "run_model_capture_layers",
        fake_run_model_capture_layers,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "steering_linear_probe.py",
            "--model",
            "Qwen/Qwen3-1.7B,google/gemma-2-2b",
            "--alpha_mode",
            "manual",
            "--alpha_values",
            "1",
            "--max_prompts",
            "2",
            "--batch_size",
            "2",
            "--epochs",
            "2",
            "--steer_layers",
            "auto:1",
            "--probe_layers",
            "auto:1",
            "--hook_points",
            "block_out",
        ],
    )

    steering_linear_probe.main()

    assert loaded_models == ["Qwen/Qwen3-1.7B", "google/gemma-2-2b"]

    qwen_result_paths = sorted(
        Path("assets/linear_probe/Qwen3-1.7B").glob("probe_*.json")
    )
    gemma_result_paths = sorted(
        Path("assets/linear_probe/gemma-2-2b").glob("probe_*.json")
    )

    assert qwen_result_paths
    assert gemma_result_paths

    qwen_payload = json.loads(qwen_result_paths[0].read_text())
    gemma_payload = json.loads(gemma_result_paths[0].read_text())
    assert qwen_payload["model"] == "Qwen/Qwen3-1.7B"
    assert gemma_payload["model"] == "google/gemma-2-2b"


def test_main_avg_norm_saves_scales_without_duplicate_layer_field(
    tmp_path,
    monkeypatch,
) -> None:
    class DummyInputs:
        def __init__(self, batch_size: int, seq_len: int = 3) -> None:
            self.input_ids = torch.ones((batch_size, seq_len), dtype=torch.long)
            self.attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)

        def to(self, _device: str):
            return self

    class DummyTokenizer:
        pad_token = None
        eos_token = "<eos>"

        def __call__(self, prompts, return_tensors, padding, truncation):
            del return_tensors, padding, truncation
            return DummyInputs(batch_size=len(prompts))

    class DummyModel:
        pass

    def fake_model_from_pretrained(model_name, **kwargs):
        del model_name, kwargs
        return DummyModel()

    def fake_tokenizer_from_pretrained(model_name, **kwargs):
        del model_name, kwargs
        return DummyTokenizer()

    def fake_load_prompts_for_concepts(_concepts, max_prompts):
        return ["prompt-1", "prompt-2"][:max_prompts]

    def fake_load_vector(_path):
        return torch.ones(4)

    def fake_compute_avg_hidden_norm(
        model,
        tokenizer,
        prompts,
        steer_layer,
        batch_size,
        device,
        capture_hook_point,
    ):
        del (
            model,
            tokenizer,
            prompts,
            steer_layer,
            batch_size,
            device,
            capture_hook_point,
        )
        return 2.0

    def fake_run_model_capture_layers(
        model,
        input_ids,
        capture_layers,
        device,
        capture_hook_point,
        steering_vector,
        steer_layer,
        alpha,
    ):
        del model, device, capture_hook_point, steering_vector
        batch_size, seq_len = input_ids.shape
        hidden_dim = 4
        captured = {}
        for layer_idx in capture_layers:
            base = torch.full((batch_size, seq_len, hidden_dim), float(layer_idx))
            if layer_idx == steer_layer:
                base = base + alpha
            else:
                base = base + alpha * 0.5
            captured[layer_idx] = base
        return captured

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        steering_linear_probe.transformers.AutoModelForCausalLM,
        "from_pretrained",
        fake_model_from_pretrained,
    )
    monkeypatch.setattr(
        steering_linear_probe.transformers.AutoTokenizer,
        "from_pretrained",
        fake_tokenizer_from_pretrained,
    )
    monkeypatch.setattr(
        steering_linear_probe,
        "load_prompts_for_concepts",
        fake_load_prompts_for_concepts,
    )
    monkeypatch.setattr(steering_linear_probe, "load_vector", fake_load_vector)
    monkeypatch.setattr(
        steering_linear_probe,
        "compute_avg_hidden_norm",
        fake_compute_avg_hidden_norm,
    )
    monkeypatch.setattr(
        steering_linear_probe,
        "run_model_capture_layers",
        fake_run_model_capture_layers,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "steering_linear_probe.py",
            "--model",
            "Qwen/Qwen3-1.7B",
            "--alpha_mode",
            "avg_norm",
            "--alpha_scales",
            "0.1,1",
            "--max_prompts",
            "2",
            "--batch_size",
            "2",
            "--epochs",
            "2",
            "--steer_layers",
            "auto:1",
            "--probe_layers",
            "auto:1",
            "--hook_points",
            "block_out",
        ],
    )

    steering_linear_probe.main()

    result_paths = sorted(Path("assets/linear_probe/Qwen3-1.7B").glob("probe_*.json"))
    assert result_paths
    payload = json.loads(result_paths[0].read_text())

    assert payload["alpha_mode"] == "avg_norm"
    assert payload["alpha_values"] == [0.1, 1.0]
    assert payload["alpha_scales"] == [0.1, 1.0]
    assert "alpha_values_by_layer" not in payload
    assert payload["alpha_scales_by_layer"] is not None
