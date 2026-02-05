import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from plot_alpha_delta_cosine import compute_delta_means_by_alpha


class DummyBatch:
    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def to(self, device: str) -> "DummyBatch":
        return self


class DummyTokenizer:
    def __call__(
        self,
        prompts,
        return_tensors: str,
        padding: bool,
        truncation: bool,
    ) -> DummyBatch:
        batch_size = len(prompts)
        input_ids = torch.zeros((batch_size, 3), dtype=torch.long)
        attention_mask = torch.tensor(
            [[1, 1, 0], [1, 0, 0]],
            dtype=torch.long,
        )[:batch_size]
        return DummyBatch(input_ids, attention_mask)


def test_compute_delta_means_by_alpha(monkeypatch) -> None:
    def fake_run_model_capture_layers(
        model,
        input_ids,
        probe_layers,
        device,
        steering_vector=None,
        steer_layer=None,
        alpha: float = 0.0,
    ):
        batch, seq = input_ids.shape
        hidden = 4
        return {
            layer: torch.full((batch, seq, hidden), float(layer) + alpha)
            for layer in probe_layers
        }

    monkeypatch.setattr(
        "plot_alpha_delta_cosine.run_model_capture_layers",
        fake_run_model_capture_layers,
    )

    tokenizer = DummyTokenizer()
    prompts = ["a", "b"]
    steering_vector = torch.zeros(4)
    probe_layers = [1, 3]
    alpha_values = [0.5, 1.5]

    delta_means = compute_delta_means_by_alpha(
        model=None,
        tokenizer=tokenizer,
        prompts=prompts,
        steering_vector=steering_vector,
        steer_layer=0,
        probe_layers=probe_layers,
        alpha_values=alpha_values,
        batch_size=2,
        device="cpu",
    )

    for alpha in alpha_values:
        for layer in probe_layers:
            expected = torch.full((4,), alpha)
            torch.testing.assert_close(delta_means[alpha][layer], expected)
