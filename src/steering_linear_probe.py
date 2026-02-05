import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import transformers
from loguru import logger

from utils import (
    MODEL_LAYERS,
    CONCEPT_CATEGORIES,
    get_model_name_for_path,
    seed_from_name,
    set_seed,
    _get_layers_container,
    load_concept_datasets,
    make_steering_hook,
)


def load_prompts_for_concepts(concepts: List[str], max_prompts: int) -> List[str]:
    prompts: List[str] = []
    for concept in concepts:
        if concept not in CONCEPT_CATEGORIES:
            raise ValueError(f"Unknown concept category: {concept}")
        _, _, dataset_key = load_concept_datasets(concept, CONCEPT_CATEGORIES[concept])
        pos_dataset, neg_dataset, _ = load_concept_datasets(
            concept, CONCEPT_CATEGORIES[concept]
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
    steering_vector: torch.Tensor,
    steer_layer: int,
    alpha: float,
    probe_layers: List[int],
    device: str,
) -> Dict[int, torch.Tensor]:
    layers_container = _get_layers_container(model)
    captured: Dict[int, torch.Tensor] = {}
    handles = []

    def make_probe_hook(layer_idx: int):
        def _hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hidden.detach()
            return output

        return _hook

    handles.append(
        layers_container[steer_layer].register_forward_hook(
            make_steering_hook(steering_vector, alpha, device=device)
        )
    )
    for layer_idx in probe_layers:
        handles.append(
            layers_container[layer_idx].register_forward_hook(
                make_probe_hook(layer_idx)
            )
        )

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
) -> Dict[int, torch.Tensor]:
    layers_container = _get_layers_container(model)
    captured: Dict[int, torch.Tensor] = {}
    handles = []

    def make_probe_hook(layer_idx: int):
        def _hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hidden.detach()
            return output

        return _hook

    for layer_idx in probe_layers:
        handles.append(
            layers_container[layer_idx].register_forward_hook(
                make_probe_hook(layer_idx)
            )
        )

    try:
        with torch.no_grad():
            _ = model(input_ids)
    finally:
        for handle in handles:
            handle.remove()

    return captured


def compute_avg_hidden_norm(
    model,
    tokenizer,
    prompts: List[str],
    steer_layer: int,
    batch_size: int,
    device: str,
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


def train_eval_linear_probe(
    X: torch.Tensor,
    y: torch.Tensor,
    seed: int,
    epochs: int,
    lr: float,
    test_ratio: float,
) -> Tuple[Dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    n_samples = X.shape[0]
    perm = torch.randperm(n_samples)
    X = X[perm]
    y = y[perm]

    split_idx = max(1, int(n_samples * (1.0 - test_ratio)))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    model = torch.nn.Linear(X.shape[1], 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        logits = model(X_train).squeeze(-1)
        loss = loss_fn(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        test_logits = model(X_test).squeeze(-1)
        test_probs = torch.sigmoid(test_logits)
        test_preds = (test_probs >= 0.5).float()
        test_acc = (
            (test_preds == y_test).float().mean().item() if y_test.numel() > 0 else 0.0
        )

    stats = {
        "test_acc": float(test_acc),
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
    }
    weight = model.weight.detach().cpu().squeeze(0)
    bias = model.bias.detach().cpu().squeeze(0)
    mean = mean.detach().cpu().squeeze(0)
    std = std.detach().cpu().squeeze(0)

    return stats, weight, bias, mean, std


def alpha_to_slug(alpha: float) -> str:
    alpha_str = str(alpha)
    return alpha_str.replace("-", "m").replace(".", "p")


def alpha_to_percent(alpha: float) -> float:
    return float(alpha) / 100.0


def get_alpha_label(alpha: float, alpha_mode: str, alpha_scales: List[float]) -> str:
    """Generate label for alpha value based on mode."""
    if alpha_mode == "avg_norm":
        # Find the closest scale value
        for scale in alpha_scales:
            if abs(alpha - scale) < 1e-6 or abs(alpha / scale - 1.0) < 0.01:
                return f"{scale:g}"
        return f"{alpha:g}"
    else:
        alpha_percent = alpha_to_percent(alpha)
        return f"{alpha_percent:g}%"


def plot_probe_accuracy(
    results: Dict[str, Dict[str, Dict[str, float]]],
    probe_layers: List[int],
    vector_name: str,
    output_dir: str,
    steer_layer: int,
    alpha_mode: str = "manual",
    alpha_scales: List[float] = None,
) -> None:
    if not results:
        return

    alpha_keys = sorted(results.keys(), key=lambda x: float(x))
    if not alpha_keys:
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    for alpha_key in alpha_keys:
        alpha_val = float(alpha_key)
        label = get_alpha_label(alpha_val, alpha_mode, alpha_scales or [])
        acc_vals = []
        for layer_idx in probe_layers:
            layer_stats = results.get(alpha_key, {}).get(str(layer_idx), {})
            acc_vals.append(layer_stats.get("test_acc", float("nan")))
        ax.plot(
            probe_layers,
            acc_vals,
            marker="o",
            linewidth=1.6,
            markersize=4,
            label=f"alpha={label}",
        )

    ax.set_xlabel("Probe layer")
    ax.set_ylabel("Probe accuracy")
    ax.set_title(
        f"Probe Accuracy ({vector_name}) steer={steer_layer}", fontweight="bold"
    )
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, ncol=2)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(
        output_dir, f"probe_acc_{vector_name}_steer_{steer_layer}.png"
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    fig.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def load_all_models_results(
    model_names: List[str],
    vector_name: str,
    alpha_mode: str,
) -> Dict[str, Dict]:
    """Load results from all models for a given vector."""
    all_results = {}

    for model_name_full in model_names:
        model_name = get_model_name_for_path(model_name_full)
        result_path = os.path.join(
            "assets", "linear_probe", model_name, f"probe_{vector_name}.json"
        )

        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                data = json.load(f)
                all_results[model_name_full] = data

    return all_results


def plot_multi_model_probe_accuracy(
    all_results: Dict[str, Dict],
    vector_name: str,
    output_dir: str,
    alpha_mode: str,
    alpha_scales: List[float],
) -> None:
    """Plot probe accuracy for all models, with each layer as a subplot."""
    if not all_results:
        logger.warning(f"No results found for {vector_name}")
        return

    # Collect all layers across all models
    all_layers = set()
    model_layers_map = {}

    for model_name, data in all_results.items():
        steer_layers = data.get("steer_layers", [])
        model_layers_map[model_name] = steer_layers
        all_layers.update(steer_layers)

    if not all_layers:
        logger.warning(f"No layers found for {vector_name}")
        return

    all_layers = sorted(all_layers)
    n_layers = len(all_layers)

    # Create subplots: one per layer
    n_cols = min(3, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False
    )

    # Color map for models
    colors = plt.cm.tab10.colors

    for idx, layer in enumerate(all_layers):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # Plot each model
        for model_idx, (model_name, data) in enumerate(all_results.items()):
            if layer not in data.get("steer_layers", []):
                continue

            results = data.get("results", {})
            alpha_keys = sorted(results.keys(), key=lambda x: float(x))

            acc_vals = []
            alpha_labels = []

            for alpha_key in alpha_keys:
                layer_stats = results.get(alpha_key, {}).get(str(layer), {})
                acc = layer_stats.get("test_acc", float("nan"))
                if not (acc != acc):  # not NaN
                    acc_vals.append(acc)
                    alpha_val = float(alpha_key)
                    label = get_alpha_label(alpha_val, alpha_mode, alpha_scales)
                    alpha_labels.append(label)

            if acc_vals:
                model_short = (
                    model_name.split("/")[-1] if "/" in model_name else model_name
                )
                ax.plot(
                    range(len(acc_vals)),
                    acc_vals,
                    marker="o",
                    linewidth=1.6,
                    markersize=4,
                    label=model_short,
                    color=colors[model_idx % len(colors)],
                )

        ax.set_xlabel("Alpha scale")
        ax.set_ylabel("Probe accuracy")
        ax.set_title(f"Layer {layer}", fontweight="bold")
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(range(len(alpha_labels)))
        ax.set_xticklabels(alpha_labels, rotation=45, ha="right")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_layers, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")

    fig.suptitle(f"Probe Accuracy - {vector_name}", fontweight="bold", fontsize=14)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"probe_acc_multi_model_{vector_name}.png")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    fig.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved multi-model plot to {save_path}")


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
    selected = sorted(set([l for l in selected if 0 <= l <= last_valid]))
    if len(selected) > total_layers:
        selected = selected[:total_layers]
    return selected


def select_probe_layers(
    steer_layer: int, max_layers: int, total_layers: int = 4
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Linear probe before/after steering")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-1.7B,google/gemma-2-2b",
        help="Comma-separated model names",
    )
    parser.add_argument(
        "--concepts",
        type=str,
        default="steering_detectable_format_json_format,steering_language_response_language,steering_startend_quotation",
    )
    parser.add_argument(
        "--random_directions",
        type=str,
        default="random_direction_1,random_direction_2,random_direction_3",
    )
    parser.add_argument("--alpha_values", type=str, default="1, 10,100,1000,10000")
    parser.add_argument(
        "--alpha_mode",
        type=str,
        default="avg_norm",
        choices=["manual", "avg_norm"],
        help="How to choose alpha values (manual or avg_norm)",
    )
    parser.add_argument(
        "--alpha_scales",
        type=str,
        default="0.01,0.1,1,10,100",
        help="Scales multiplied by average hidden norm when alpha_mode=avg_norm",
    )
    parser.add_argument("--max_prompts", type=int, default=8196)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_names = [m.strip() for m in args.model.split(",") if m.strip()]
    if not model_names:
        raise ValueError("No model names provided")
    unknown_models = [m for m in model_names if m not in MODEL_LAYERS]
    if unknown_models:
        raise ValueError(f"Unknown model(s): {unknown_models}")

    set_seed(args.seed)
    concepts = [c.strip() for c in args.concepts.split(",") if c.strip()]
    random_dirs = [r.strip() for r in args.random_directions.split(",") if r.strip()]
    manual_alpha_values = [
        float(a.strip()) for a in args.alpha_values.split(",") if a.strip()
    ]
    alpha_scales = [float(a.strip()) for a in args.alpha_scales.split(",") if a.strip()]

    prompts = load_prompts_for_concepts(concepts, args.max_prompts)
    if not prompts:
        raise RuntimeError("No prompts loaded for selected concepts")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    logger.info(f"Device: {device}")

    for model_name_full in model_names:
        max_layers = MODEL_LAYERS[model_name_full]
        steer_layers = select_steer_layers(max_layers, total_layers=6)
        if not steer_layers:
            raise ValueError("No valid steer layers selected")

        logger.info(f"Model: {model_name_full}")
        logger.info(f"Steer layers: {steer_layers}")
        logger.info(f"Alpha mode: {args.alpha_mode}")
        if args.alpha_mode == "manual":
            logger.info(f"Alpha values: {manual_alpha_values}")
        else:
            logger.info(f"Alpha scales: {alpha_scales}")

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name_full, device_map=device, dtype=dtype, trust_remote_code=True
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_full, use_fast=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_name = get_model_name_for_path(model_name_full)
        output_dir = os.path.join("assets", "linear_probe", model_name)
        os.makedirs(output_dir, exist_ok=True)

        alpha_values_by_layer: Dict[int, List[float]] = {}
        alpha_values_by_layer_percent: Dict[int, List[float]] = {}
        if args.alpha_mode == "avg_norm":
            for steer_layer in steer_layers:
                avg_norm = compute_avg_hidden_norm(
                    model,
                    tokenizer,
                    prompts,
                    steer_layer,
                    args.batch_size,
                    device,
                )
                alpha_values_by_layer[steer_layer] = [
                    avg_norm * scale for scale in alpha_scales
                ]
                alpha_values_by_layer_percent[steer_layer] = [
                    alpha_to_percent(alpha)
                    for alpha in alpha_values_by_layer[steer_layer]
                ]
                logger.info(
                    "Layer {} avg norm={:.4f} -> alphas={}",
                    steer_layer,
                    avg_norm,
                    alpha_values_by_layer[steer_layer],
                )
                logger.info(
                    "Layer {} alpha%={}",
                    steer_layer,
                    alpha_values_by_layer_percent[steer_layer],
                )

        vector_specs: List[Tuple[str, str]] = []
        for concept in concepts:
            vector_specs.append(
                (
                    concept,
                    os.path.join(
                        "assets", "concept_vectors", model_name, f"{concept}.pt"
                    ),
                )
            )
        for random_dir in random_dirs:
            vector_specs.append(
                (
                    random_dir,
                    os.path.join(
                        "assets", "concept_vectors", model_name, f"{random_dir}.pt"
                    ),
                )
            )

        for vector_name, vector_path in vector_specs:
            logger.info(f"Processing vector: {vector_name}")
            vector_tensor = load_vector(vector_path)
            results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}

            for steer_layer in steer_layers:
                alpha_values = manual_alpha_values
                if args.alpha_mode == "avg_norm":
                    alpha_values = alpha_values_by_layer.get(steer_layer, [])
                if not alpha_values:
                    logger.warning(
                        f"Skipping steer layer {steer_layer}: no alpha values"
                    )
                    continue
                probe_layers = select_probe_layers(
                    steer_layer, max_layers, total_layers=4
                )
                if not probe_layers:
                    logger.warning(
                        f"Skipping steer layer {steer_layer}: no probe layers"
                    )
                    continue
                logger.info(
                    f"Steer layer: {steer_layer} | Probe layers: {probe_layers}"
                )

                if vector_tensor.ndim == 1:
                    steering_vector = vector_tensor
                else:
                    steering_vector = vector_tensor[steer_layer]

                alpha_results: Dict[str, Dict[str, Dict[str, float]]] = {}

                for alpha in alpha_values:
                    alpha_percent = alpha_to_percent(alpha)
                    logger.info(f"Alpha: {alpha_percent:g}%")
                    layer_features_before: Dict[int, List[torch.Tensor]] = {
                        l: [] for l in probe_layers
                    }
                    layer_features_after: Dict[int, List[torch.Tensor]] = {
                        l: [] for l in probe_layers
                    }

                    for i in range(0, len(prompts), args.batch_size):
                        batch_prompts = prompts[i : i + args.batch_size]
                        inputs = tokenizer(
                            batch_prompts,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                        ).to(device)

                        captured_before = run_model_capture_layers(
                            model,
                            inputs.input_ids,
                            steering_vector,
                            steer_layer,
                            0.0,
                            probe_layers,
                            device,
                        )
                        captured_after = run_model_capture_layers(
                            model,
                            inputs.input_ids,
                            steering_vector,
                            steer_layer,
                            alpha,
                            probe_layers,
                            device,
                        )

                        for layer_idx in probe_layers:
                            h_before = captured_before[layer_idx]
                            h_after = captured_after[layer_idx]
                            token_mask = inputs.attention_mask.bool()
                            token_before = h_before[token_mask].float().cpu()
                            token_after = h_after[token_mask].float().cpu()
                            layer_features_before[layer_idx].append(token_before)
                            layer_features_after[layer_idx].append(token_after)

                    alpha_key = str(alpha)
                    alpha_results[alpha_key] = {}
                    for layer_idx in probe_layers:
                        X_before = torch.cat(layer_features_before[layer_idx], dim=0)
                        X_after = torch.cat(layer_features_after[layer_idx], dim=0)
                        X = torch.cat([X_before, X_after], dim=0)
                        y = torch.cat(
                            [
                                torch.zeros(X_before.shape[0]),
                                torch.ones(X_after.shape[0]),
                            ],
                            dim=0,
                        )
                        probe_seed = seed_from_name(
                            f"{vector_name}-{steer_layer}-{alpha}-{layer_idx}"
                        )
                        stats, weight, bias, mean, std = train_eval_linear_probe(
                            X,
                            y,
                            seed=probe_seed,
                            epochs=args.epochs,
                            lr=args.lr,
                            test_ratio=args.test_ratio,
                        )
                        std = std.clamp_min(1e-6)
                        raw_weight = weight / std

                        weight_dir = os.path.join(
                            output_dir,
                            "probe_weights",
                            vector_name,
                            f"steer_{steer_layer}",
                            f"alpha_{alpha_to_slug(alpha)}",
                        )
                        os.makedirs(weight_dir, exist_ok=True)
                        weight_path = os.path.join(weight_dir, f"layer_{layer_idx}.pt")
                        torch.save(
                            {
                                "weight": weight,
                                "bias": bias,
                                "mean": mean,
                                "std": std,
                                "raw_weight": raw_weight,
                                "vector": vector_name,
                                "alpha": alpha,
                                "alpha_percent": alpha_percent,
                                "steer_layer": steer_layer,
                                "layer": layer_idx,
                                "model": model_name_full,
                            },
                            weight_path,
                        )

                        stats["weight_path"] = weight_path
                        alpha_results[alpha_key][str(layer_idx)] = stats

                    plot_probe_accuracy(
                        alpha_results,
                        probe_layers,
                        vector_name,
                        os.path.join(output_dir, "probe_plots"),
                        steer_layer,
                        args.alpha_mode,
                        alpha_scales,
                    )

                results[str(steer_layer)] = {
                    "probe_layers": probe_layers,
                    "alpha_results": alpha_results,
                }

            plot_alpha_keys = set()
            plot_results: Dict[str, Dict[str, Dict[str, float]]] = {}
            for steer_layer in steer_layers:
                layer_payload = results.get(str(steer_layer), {})
                alpha_results = layer_payload.get("alpha_results", {})
                for alpha_key in alpha_results:
                    plot_alpha_keys.add(alpha_key)

            plot_alpha_keys_sorted = sorted(plot_alpha_keys, key=lambda x: float(x))
            for alpha_key in plot_alpha_keys_sorted:
                plot_results[alpha_key] = {}
                for steer_layer in steer_layers:
                    layer_payload = results.get(str(steer_layer), {})
                    alpha_results = layer_payload.get("alpha_results", {})
                    layer_stats = alpha_results.get(alpha_key, {}).get(str(steer_layer))
                    if layer_stats:
                        plot_results[alpha_key][str(steer_layer)] = layer_stats

            plot_alpha_values = [float(a) for a in plot_alpha_keys_sorted]
            plot_alpha_values_percent = [
                alpha_to_percent(alpha) for alpha in plot_alpha_values
            ]

            save_path = os.path.join(output_dir, f"probe_{vector_name}.json")
            with open(save_path, "w") as f:
                alpha_values_by_layer_str = {
                    str(k): v for k, v in alpha_values_by_layer.items()
                }
                alpha_values_by_layer_percent_str = {
                    str(k): v for k, v in alpha_values_by_layer_percent.items()
                }
                json.dump(
                    {
                        "model": model_name_full,
                        "vector": vector_name,
                        "steer_layers": steer_layers,
                        "alpha_mode": args.alpha_mode,
                        "alpha_values": plot_alpha_values,
                        "alpha_values_percent": plot_alpha_values_percent,
                        "alpha_values_manual": (
                            manual_alpha_values if args.alpha_mode == "manual" else None
                        ),
                        "alpha_values_manual_percent": (
                            [alpha_to_percent(alpha) for alpha in manual_alpha_values]
                            if args.alpha_mode == "manual"
                            else None
                        ),
                        "alpha_scales": (
                            alpha_scales if args.alpha_mode == "avg_norm" else None
                        ),
                        "alpha_values_by_layer": (
                            alpha_values_by_layer_str
                            if args.alpha_mode == "avg_norm"
                            else None
                        ),
                        "alpha_values_by_layer_percent": (
                            alpha_values_by_layer_percent_str
                            if args.alpha_mode == "avg_norm"
                            else None
                        ),
                        "max_prompts": len(prompts),
                        "probe_layers": steer_layers,
                        "results": plot_results,
                        "results_by_steer_layer": results,
                    },
                    f,
                    indent=2,
                )
            logger.info(f"Saved results to {save_path}")

    # After processing all models, create combined plots for each concept
    logger.info("Creating multi-model comparison plots...")
    all_vector_names = concepts + random_dirs
    for vector_name in all_vector_names:
        all_results = load_all_models_results(model_names, vector_name, args.alpha_mode)
        if all_results:
            plot_multi_model_probe_accuracy(
                all_results,
                vector_name,
                os.path.join("assets", "linear_probe", "combined_plots"),
                args.alpha_mode,
                alpha_scales,
            )


if __name__ == "__main__":
    main()
