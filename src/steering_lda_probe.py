import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import transformers
from loguru import logger

from probe_utils import (
    alpha_to_percent,
    alpha_to_slug,
    compute_avg_hidden_norm,
    get_alpha_label,
    load_prompts_for_concepts,
    load_vector,
    run_model_capture_layers,
    select_probe_layers,
    select_steer_layers,
)
from utils import MODEL_LAYERS
from utils import get_model_name_for_path
from utils import seed_from_name
from utils import set_seed


def train_eval_lda_probe(
    X: torch.Tensor,
    y: torch.Tensor,
    seed: int,
    test_ratio: float,
    reg: float,
) -> Tuple[
    Dict[str, float],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    float,
    float,
]:
    torch.manual_seed(seed)
    n_samples = X.shape[0]
    perm = torch.randperm(n_samples)
    X = X[perm]
    y = y[perm]

    split_idx = max(1, int(n_samples * (1.0 - test_ratio)))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    y_train = y_train.float()
    y_test = y_test.float()

    mask1 = y_train == 1
    mask0 = y_train == 0
    if mask0.sum() == 0 or mask1.sum() == 0:
        raise RuntimeError("Both classes must be present in training data")

    X0 = X_train[mask0].float()
    X1 = X_train[mask1].float()
    mean0 = X0.mean(dim=0)
    mean1 = X1.mean(dim=0)
    centered0 = X0 - mean0
    centered1 = X1 - mean1

    cov = (centered0.T @ centered0 + centered1.T @ centered1) / max(
        1, (X0.shape[0] + X1.shape[0] - 2)
    )
    cov = cov + torch.eye(cov.shape[0], device=cov.device) * reg

    diff = (mean1 - mean0).unsqueeze(1)
    try:
        weight = torch.linalg.solve(cov, diff).squeeze(1)
    except RuntimeError:
        weight = (torch.linalg.pinv(cov) @ diff).squeeze(1)

    prior0 = float(mask0.float().mean().item())
    prior1 = float(mask1.float().mean().item())
    log_prior = float(torch.log(torch.tensor(prior1 / prior0)))
    bias = -0.5 * (mean1 + mean0).dot(weight) + log_prior

    if X_test.numel() == 0:
        test_acc = 0.0
    else:
        scores = X_test.float() @ weight + bias
        preds = (scores >= 0).float()
        test_acc = (preds == y_test).float().mean().item()

    stats = {
        "test_acc": float(test_acc),
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
    }

    return (
        stats,
        weight.detach().cpu(),
        torch.tensor(bias).detach().cpu(),
        mean0.detach().cpu(),
        mean1.detach().cpu(),
        cov.detach().cpu(),
        prior0,
        prior1,
    )


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
        f"LDA Probe Accuracy ({vector_name}) steer={steer_layer}", fontweight="bold"
    )
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, ncol=2)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(
        output_dir, f"lda_probe_acc_{vector_name}_steer_{steer_layer}.png"
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    fig.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def load_all_models_results(
    model_names: List[str],
    vector_name: str,
) -> Dict[str, Dict]:
    all_results = {}

    for model_name_full in model_names:
        model_name = get_model_name_for_path(model_name_full)
        result_path = os.path.join(
            "assets", "lda_probe", model_name, f"lda_probe_{vector_name}.json"
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
    if not all_results:
        logger.warning(f"No results found for {vector_name}")
        return

    all_layers = set()

    for model_name, data in all_results.items():
        steer_layers = data.get("steer_layers", [])
        all_layers.update(steer_layers)

    if not all_layers:
        logger.warning(f"No layers found for {vector_name}")
        return

    all_layers = sorted(all_layers)
    n_layers = len(all_layers)

    n_cols = min(3, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False
    )

    colors = plt.cm.tab10.colors

    for idx, layer in enumerate(all_layers):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

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
                if not (acc != acc):
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

    for idx in range(n_layers, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")

    fig.suptitle(f"LDA Probe Accuracy - {vector_name}", fontweight="bold", fontsize=14)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"lda_probe_acc_multi_model_{vector_name}.png")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    fig.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved multi-model plot to {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LDA probe before/after steering")
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
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lda_reg", type=float, default=1e-4)
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
        output_dir = os.path.join("assets", "lda_probe", model_name)
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
                            probe_layers,
                            device,
                            steering_vector=steering_vector,
                            steer_layer=steer_layer,
                            alpha=0.0,
                        )
                        captured_after = run_model_capture_layers(
                            model,
                            inputs.input_ids,
                            probe_layers,
                            device,
                            steering_vector=steering_vector,
                            steer_layer=steer_layer,
                            alpha=alpha,
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
                        (
                            stats,
                            weight,
                            bias,
                            mean0,
                            mean1,
                            cov,
                            prior0,
                            prior1,
                        ) = train_eval_lda_probe(
                            X,
                            y,
                            seed=probe_seed,
                            test_ratio=args.test_ratio,
                            reg=args.lda_reg,
                        )

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
                                "mean0": mean0,
                                "mean1": mean1,
                                "cov": cov,
                                "cov_reg": args.lda_reg,
                                "prior0": prior0,
                                "prior1": prior1,
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

            save_path = os.path.join(output_dir, f"lda_probe_{vector_name}.json")
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
                        "lda_reg": args.lda_reg,
                    },
                    f,
                    indent=2,
                )
            logger.info(f"Saved results to {save_path}")

    logger.info("Creating multi-model comparison plots...")
    all_vector_names = concepts + random_dirs
    for vector_name in all_vector_names:
        all_results = load_all_models_results(model_names, vector_name)
        if all_results:
            plot_multi_model_probe_accuracy(
                all_results,
                vector_name,
                os.path.join("assets", "lda_probe", "combined_plots"),
                args.alpha_mode,
                alpha_scales,
            )


if __name__ == "__main__":
    main()
