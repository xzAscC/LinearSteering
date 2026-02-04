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


def plot_probe_accuracy(
    results: Dict[str, Dict[str, Dict[str, float]]],
    probe_layers: List[int],
    vector_name: str,
    output_dir: str,
    steer_layer: int,
) -> None:
    if not results:
        return

    alpha_keys = sorted(results.keys(), key=lambda x: float(x))
    if not alpha_keys:
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    for alpha_key in alpha_keys:
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
            label=f"alpha={alpha_key}",
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
    alpha_values = [float(a.strip()) for a in args.alpha_values.split(",") if a.strip()]

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
        logger.info(f"Alpha values: {alpha_values}")

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
                probe_layers = [
                    layer
                    for layer in (steer_layer, steer_layer + 1, steer_layer + 2)
                    if layer <= max_layers - 2
                ]
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
                    logger.info(f"Alpha: {alpha}")
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
                            pooled_before = h_before.mean(dim=1).float().cpu()
                            pooled_after = h_after.mean(dim=1).float().cpu()
                            layer_features_before[layer_idx].append(pooled_before)
                            layer_features_after[layer_idx].append(pooled_after)

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
                    )

                results[str(steer_layer)] = {
                    "probe_layers": probe_layers,
                    "alpha_results": alpha_results,
                }

            save_path = os.path.join(output_dir, f"probe_{vector_name}.json")
            with open(save_path, "w") as f:
                json.dump(
                    {
                        "model": model_name_full,
                        "vector": vector_name,
                        "steer_layers": steer_layers,
                        "alpha_values": alpha_values,
                        "max_prompts": len(prompts),
                        "results": results,
                    },
                    f,
                    indent=2,
                )
            logger.info(f"Saved results to {save_path}")


if __name__ == "__main__":
    main()
