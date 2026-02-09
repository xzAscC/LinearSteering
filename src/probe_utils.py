from __future__ import annotations

from typing import Dict, List, Optional

import os

import torch

from utils import CONCEPT_CATEGORIES
from utils import _get_layers_container
from utils import load_concept_datasets
from utils import make_steering_hook


QWEN3_DEFAULT_CAPTURE_HOOK_POINTS = [
    "input_ln",
    "attn",
    "post_attn_ln",
    "mlp",
    "block_out",
]
GEMMA2_DEFAULT_CAPTURE_HOOK_POINTS = [
    "input_ln",
    "attn",
    "post_attn_proj_ln",
    "post_attn_ln",
    "mlp",
    "post_mlp_ln",
    "block_out",
]


def _extract_hidden(output):
    return output[0] if isinstance(output, tuple) else output


def _is_qwen3_style_layer(layer_module) -> bool:
    return all(
        hasattr(layer_module, name)
        for name in (
            "input_layernorm",
            "self_attn",
            "post_attention_layernorm",
            "mlp",
        )
    )


def _is_gemma2_style_layer(layer_module) -> bool:
    return all(
        hasattr(layer_module, name)
        for name in (
            "input_layernorm",
            "self_attn",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "mlp",
            "post_feedforward_layernorm",
        )
    )


def parse_probe_hook_point(hook_point_arg: str) -> str:
    alias_map = {
        "ln1": "input_ln",
        "ln2": "post_attn_ln",
        "pre_mlp_ln": "post_attn_ln",
        "pre_ffn_ln": "post_attn_ln",
        "block": "block_out",
        "resid": "block_out",
        "post_attention_ln": "post_attn_proj_ln",
        "post_attn_norm": "post_attn_proj_ln",
        "post_ffn_ln": "post_mlp_ln",
        "post_feedforward_ln": "post_mlp_ln",
    }
    valid_points = {
        "input_ln",
        "attn",
        "post_attn_proj_ln",
        "post_attn_ln",
        "mlp",
        "post_mlp_ln",
        "block_out",
    }

    normalized = hook_point_arg.strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized:
        raise ValueError("Hook point must be non-empty")

    canonical = alias_map.get(normalized, normalized)
    if canonical not in valid_points:
        raise ValueError(
            f"Unsupported hook point '{hook_point_arg}'. Valid points: {sorted(valid_points)}"
        )
    return canonical


def parse_probe_hook_points(hook_points_arg: str) -> List[str]:
    points: List[str] = []
    for raw_point in hook_points_arg.split(","):
        raw_point = raw_point.strip()
        if not raw_point:
            continue
        canonical = parse_probe_hook_point(raw_point)
        if canonical not in points:
            points.append(canonical)

    if not points:
        raise ValueError("No valid hook points were provided")

    return points


def resolve_default_probe_hook_points(model) -> List[str]:
    layers_container = _get_layers_container(model)
    if len(layers_container) > 0 and _is_gemma2_style_layer(layers_container[0]):
        return GEMMA2_DEFAULT_CAPTURE_HOOK_POINTS.copy()
    return QWEN3_DEFAULT_CAPTURE_HOOK_POINTS.copy()


def resolve_default_probe_hook_point(model) -> str:
    return resolve_default_probe_hook_points(model)[0]


def _resolve_hook_module(layer_module, hook_point: str):
    if _is_gemma2_style_layer(layer_module):
        mapping = {
            "input_ln": layer_module.input_layernorm,
            "attn": layer_module.self_attn,
            "post_attn_proj_ln": layer_module.post_attention_layernorm,
            "post_attn_ln": layer_module.pre_feedforward_layernorm,
            "mlp": layer_module.mlp,
            "post_mlp_ln": layer_module.post_feedforward_layernorm,
            "block_out": layer_module,
        }
        if hook_point not in mapping:
            raise ValueError(
                f"Hook point '{hook_point}' not supported for Gemma-style layers."
            )
        return mapping[hook_point]

    if _is_qwen3_style_layer(layer_module):
        mapping = {
            "input_ln": layer_module.input_layernorm,
            "attn": layer_module.self_attn,
            "post_attn_ln": layer_module.post_attention_layernorm,
            "mlp": layer_module.mlp,
            "block_out": layer_module,
        }
        if hook_point not in mapping:
            raise ValueError(
                f"Hook point '{hook_point}' not supported for Qwen-style layers."
            )
        return mapping[hook_point]

    if hook_point != "block_out":
        raise ValueError(
            f"Hook point '{hook_point}' not supported for layer type "
            f"'{layer_module.__class__.__name__}'. Use 'block_out'."
        )
    return layer_module


def load_prompts_for_concepts(concepts: List[str], max_prompts: int) -> List[str]:
    prompts: List[str] = []
    for concept in concepts:
        if concept not in CONCEPT_CATEGORIES:
            raise ValueError(f"Unknown concept category: {concept}")
        pos_dataset, neg_dataset, dataset_key = load_concept_datasets(
            concept,
            CONCEPT_CATEGORIES[concept],
        )
        for dataset in (pos_dataset, neg_dataset):
            for item in dataset:
                prompts.append(item[dataset_key])
                if len(prompts) >= max_prompts:
                    return prompts
    return prompts


def run_model_capture_layers(
    model,
    input_ids: torch.Tensor,
    probe_layers: List[int],
    device: str,
    steering_vector: Optional[torch.Tensor] = None,
    steer_layer: Optional[int] = None,
    alpha: float = 0.0,
    capture_hook_point: str = "block_out",
) -> Dict[int, torch.Tensor]:
    if (steering_vector is None) != (steer_layer is None):
        raise ValueError(
            "steering_vector and steer_layer must be both set or both None"
        )

    layers_container = _get_layers_container(model)
    captured: Dict[int, torch.Tensor] = {}
    handles = []

    def make_probe_hook(layer_idx: int):
        def _hook(_module, _inputs, output):
            hidden = _extract_hidden(output)
            captured[layer_idx] = hidden.detach()
            return output

        return _hook

    if steering_vector is not None:
        handles.append(
            layers_container[steer_layer].register_forward_hook(
                make_steering_hook(steering_vector, alpha, device=device)
            )
        )

    for layer_idx in probe_layers:
        capture_module = _resolve_hook_module(
            layers_container[layer_idx], capture_hook_point
        )
        handles.append(capture_module.register_forward_hook(make_probe_hook(layer_idx)))

    try:
        with torch.no_grad():
            _ = model(input_ids)
    finally:
        for handle in handles:
            handle.remove()

    return captured


def run_model_capture_layers_no_steering(
    model,
    input_ids: torch.Tensor,
    probe_layers: List[int],
    device: str,
    capture_hook_point: str = "block_out",
) -> Dict[int, torch.Tensor]:
    return run_model_capture_layers(
        model,
        input_ids,
        probe_layers,
        device,
        capture_hook_point=capture_hook_point,
        steering_vector=None,
        steer_layer=None,
        alpha=0.0,
    )


def compute_avg_hidden_norm(
    model,
    tokenizer,
    prompts: List[str],
    steer_layer: int,
    batch_size: int,
    device: str,
    capture_hook_point: str = "block_out",
) -> float:
    total_norm = 0.0
    total_count = 0
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        captured = run_model_capture_layers_no_steering(
            model,
            inputs.input_ids,
            [steer_layer],
            device,
            capture_hook_point=capture_hook_point,
        )
        hidden = captured.get(steer_layer)
        if hidden is None:
            raise RuntimeError(f"Missing hidden state for layer {steer_layer}")
        pooled = hidden.mean(dim=1).float()
        norms = torch.linalg.vector_norm(pooled, ord=2, dim=1)
        total_norm += norms.sum().item()
        total_count += norms.numel()

    if total_count == 0:
        raise RuntimeError("No hidden states captured for norm computation")
    return total_norm / total_count


def alpha_to_slug(alpha: float) -> str:
    alpha_str = str(alpha)
    return alpha_str.replace("-", "m").replace(".", "p")


def alpha_to_percent(alpha: float) -> float:
    return float(alpha) / 100.0


def get_alpha_label(alpha: float, alpha_mode: str, alpha_scales: List[float]) -> str:
    if alpha_mode == "avg_norm":
        for scale in alpha_scales:
            if abs(alpha - scale) < 1e-6 or abs(alpha / scale - 1.0) < 0.01:
                return f"{scale:g}"
        return f"{alpha:g}"
    alpha_percent = alpha_to_percent(alpha)
    return f"{alpha_percent:g}%"


def load_vector(path: str) -> torch.Tensor:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Vector not found: {path}")
    vector = torch.load(path, map_location="cpu")
    if isinstance(vector, dict):
        vector = vector.get("random_vector", vector)
    return vector


def _pick_segment_layers(start: int, end: int, count: int) -> List[int]:
    if start > end or count <= 0:
        return []
    length = end - start + 1
    if length <= 1:
        return [start]
    if count == 1:
        return [start + length // 2]
    indices = []
    for i in range(count):
        pos = start + round(i * (length - 1) / (count - 1))
        indices.append(int(min(end, max(start, pos))))
    return sorted(set(indices))


def select_steer_layers(max_layers: int, total_layers: int = 6) -> List[int]:
    if max_layers <= 1:
        return [0]
    last_valid = max_layers - 2
    if last_valid <= 0:
        return [0]
    per_segment = max(1, total_layers // 3)
    early_end = last_valid // 3
    mid_end = (2 * last_valid) // 3
    segments = [
        (0, early_end),
        (early_end + 1, mid_end),
        (mid_end + 1, last_valid),
    ]
    selected: List[int] = []
    for start, end in segments:
        selected.extend(_pick_segment_layers(start, end, per_segment))
    selected = sorted(
        set([layer_idx for layer_idx in selected if 0 <= layer_idx <= last_valid])
    )
    if len(selected) > total_layers:
        selected = selected[:total_layers]
    return selected


def select_probe_layers(
    steer_layer: int,
    max_layers: int,
    total_layers: int = 4,
) -> List[int]:
    if total_layers <= 0:
        return []
    last_valid = max_layers - 2
    if steer_layer > last_valid:
        return []
    remaining = max(0, total_layers - 1)
    later_layers = list(range(steer_layer + 1, last_valid + 1))
    if remaining <= 0:
        return [steer_layer]
    if not later_layers:
        return [steer_layer]
    if len(later_layers) <= remaining:
        return [steer_layer] + later_layers
    selected_later = _pick_segment_layers(later_layers[0], later_layers[-1], remaining)
    return [steer_layer] + selected_later
