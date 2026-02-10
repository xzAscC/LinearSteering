import argparse
import hashlib
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import transformers
from loguru import logger
from tqdm import tqdm

from probe_utils import (
    alpha_to_slug,
    compute_avg_hidden_norm,
    load_prompts_for_concepts,
    load_vector,
    parse_probe_hook_points,
    resolve_default_probe_hook_points,
    run_model_capture_layers,
)
from utils import CONCEPT_CATEGORIES
from utils import MODEL_LAYERS
from utils import get_model_name_for_path
from utils import load_concept_datasets
from utils import parse_layers_to_run
from utils import seed_from_name
from utils import set_seed


RANDOM_DIRECTION_CONCEPT = "steering_random_direction"
RANDOM_DIRECTION_DATASET_CONCEPT = "steering_safety"


def _parse_concepts_to_run(concepts_arg: str) -> List[str]:
    selected_concepts: List[str] = []
    for raw_concept in concepts_arg.split(","):
        concept_name = raw_concept.strip()
        if not concept_name:
            continue
        if concept_name == RANDOM_DIRECTION_CONCEPT:
            if concept_name not in selected_concepts:
                selected_concepts.append(concept_name)
            continue
        if concept_name not in CONCEPT_CATEGORIES:
            raise ValueError(
                f"Unknown concept '{concept_name}'. "
                f"Available: {sorted([*CONCEPT_CATEGORIES.keys(), RANDOM_DIRECTION_CONCEPT])}"
            )
        if concept_name not in selected_concepts:
            selected_concepts.append(concept_name)

    if not selected_concepts:
        raise ValueError("No valid concepts were provided")

    if len(selected_concepts) != 1:
        raise ValueError(
            "Please provide exactly one concept direction per run via --concepts. "
            f"Got: {selected_concepts}"
        )

    return selected_concepts


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


def _auto_select_layers(
    layer_count: int, max_layers: int, min_layer_exclusive: int = -1
) -> List[int]:
    max_valid_layer = max_layers - 2
    first_valid_layer = min_layer_exclusive + 1
    if max_valid_layer < first_valid_layer:
        return []

    if layer_count <= 0:
        raise ValueError("auto layer count must be positive")

    available_layers = list(range(first_valid_layer, max_valid_layer + 1))
    total_available = len(available_layers)
    if layer_count >= total_available:
        return available_layers

    if layer_count == 1:
        return [available_layers[0]]

    sample_positions = [
        int(round(i * (total_available - 1) / (layer_count - 1)))
        for i in range(layer_count)
    ]
    selected_layers: List[int] = []
    for position in sample_positions:
        layer_idx = available_layers[position]
        if layer_idx not in selected_layers:
            selected_layers.append(layer_idx)

    if len(selected_layers) < layer_count:
        for layer_idx in available_layers:
            if layer_idx not in selected_layers:
                selected_layers.append(layer_idx)
            if len(selected_layers) == layer_count:
                break

    return selected_layers


def _resolve_layers_to_run(
    layers_arg: str, max_layers: int, min_layer_exclusive: int = -1
) -> List[int]:
    layers_arg_normalized = layers_arg.strip().lower()
    if layers_arg_normalized.startswith("auto:"):
        count_str = layers_arg.split(":", 1)[1].strip()
        if not count_str:
            raise ValueError("Missing auto layer count. Use format: auto:<count>")
        try:
            auto_count = int(count_str)
        except ValueError as error:
            raise ValueError(
                f"Invalid auto layer count '{count_str}'. Use an integer."
            ) from error
        return _auto_select_layers(
            auto_count,
            max_layers,
            min_layer_exclusive=min_layer_exclusive,
        )

    layers = parse_layers_to_run(layers_arg, max_layers)
    return [layer for layer in layers if layer > min_layer_exclusive]


def _build_run_metadata(args: argparse.Namespace) -> Tuple[str, str, str]:
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args_json = json.dumps(vars(args), sort_keys=True, separators=(",", ":"))
    params_hash = hashlib.sha1(args_json.encode("utf-8")).hexdigest()[:8]
    run_id = f"{run_timestamp}_s{args.seed}_h{params_hash}"
    return run_id, run_timestamp, params_hash


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
        default="steering_detectable_format_json_format",
        help=(
            "Single concept category to process per run. "
            f"Use '{RANDOM_DIRECTION_CONCEPT}' for a random direction baseline."
        ),
    )
    parser.add_argument(
        "--random_direction",
        "--random_directions",
        dest="random_direction",
        type=str,
        default="random_direction_1",
        help=(
            "Random direction vector name under assets/concept_vectors/<model>/ "
            "used when concepts include steering_random_direction."
        ),
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
    parser.add_argument(
        "--steer_layers",
        type=str,
        default="auto:6",
        help=(
            "Steering layer selection. Supports: 'auto:<count>' (e.g. auto:6), "
            "comma-separated percentages, comma-separated layer indices, or 'all'."
        ),
    )
    parser.add_argument(
        "--probe_layers",
        type=str,
        default="auto:3",
        help=(
            "Probe layer selection for each steering layer. Supports: 'auto:<count>' "
            "(e.g. auto:8, sampled from layers after the steering layer), "
            "comma-separated percentages, comma-separated layer indices, or 'all'."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hook_points",
        "--hook_point",
        dest="hook_points",
        type=str,
        default="auto",
        help=(
            "Comma-separated hook positions for probe feature capture. "
            "Use 'auto' for model-specific defaults "
            "(Qwen3: input_ln,attn,post_attn_ln,mlp,block_out; "
            "Gemma2: input_ln,attn,post_attn_proj_ln,post_attn_ln,mlp,post_mlp_ln,block_out)."
        ),
    )
    args = parser.parse_args()
    logger.remove()
    run_id, run_timestamp, params_hash = _build_run_metadata(args)

    model_names = [m.strip() for m in args.model.split(",") if m.strip()]
    if not model_names:
        raise ValueError("No model names provided")
    unknown_models = [m for m in model_names if m not in MODEL_LAYERS]
    if unknown_models:
        raise ValueError(f"Unknown model(s): {unknown_models}")

    set_seed(args.seed)
    concepts = _parse_concepts_to_run(args.concepts)
    random_direction_names = [
        name.strip() for name in args.random_direction.split(",") if name.strip()
    ]
    if not random_direction_names:
        raise ValueError("No random direction provided")
    if len(random_direction_names) > 1:
        logger.warning(
            "Multiple random directions were provided; using the first one: {}",
            random_direction_names[0],
        )
    random_direction_name = random_direction_names[0]
    manual_alpha_values = [
        float(a.strip()) for a in args.alpha_values.split(",") if a.strip()
    ]
    alpha_scales = [float(a.strip()) for a in args.alpha_scales.split(",") if a.strip()]

    concept_prompts: Dict[str, List[str]] = {}
    for concept in concepts:
        if concept == RANDOM_DIRECTION_CONCEPT:
            safety_config = CONCEPT_CATEGORIES[RANDOM_DIRECTION_DATASET_CONCEPT]
            pos_dataset, neg_dataset, dataset_key = load_concept_datasets(
                RANDOM_DIRECTION_DATASET_CONCEPT,
                safety_config,
            )
            prompts: List[str] = []
            for dataset in (pos_dataset, neg_dataset):
                for item in dataset:
                    prompts.append(item[dataset_key])
                    if len(prompts) >= args.max_prompts:
                        break
                if len(prompts) >= args.max_prompts:
                    break
            concept_prompts[concept] = prompts
            continue

        prompts = load_prompts_for_concepts([concept], args.max_prompts)
        concept_prompts[concept] = prompts

    empty_concepts = [
        concept for concept, prompts in concept_prompts.items() if not prompts
    ]
    if empty_concepts:
        raise RuntimeError(f"No prompts loaded for concepts: {empty_concepts}")

    alpha_reference_prompts: List[str] = []
    for concept in concepts:
        alpha_reference_prompts.extend(concept_prompts[concept])
        if len(alpha_reference_prompts) >= args.max_prompts:
            alpha_reference_prompts = alpha_reference_prompts[: args.max_prompts]
            break

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    logger.info(f"Device: {device}")
    logger.info(f"Run ID: {run_id}")

    for model_name_full in model_names:
        max_layers = MODEL_LAYERS[model_name_full]
        steer_layers = _resolve_layers_to_run(args.steer_layers, max_layers)
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

        if args.hook_points.strip().lower() == "auto":
            hook_points = resolve_default_probe_hook_points(model)
        else:
            hook_points = parse_probe_hook_points(args.hook_points)
        logger.info(f"Probe hook points: {hook_points}")

        model_name = get_model_name_for_path(model_name_full)
        output_dir = os.path.join("assets", "linear_probe", model_name)
        os.makedirs(output_dir, exist_ok=True)

        alpha_values_by_hook_layer: Dict[str, Dict[int, List[float]]] = {
            hook_point: {} for hook_point in hook_points
        }
        alpha_entries_by_hook_layer: Dict[
            str,
            Dict[int, List[Tuple[float, float]]],
        ] = {hook_point: {} for hook_point in hook_points}
        if args.alpha_mode == "avg_norm":
            for hook_point in hook_points:
                for steer_layer in steer_layers:
                    avg_norm = compute_avg_hidden_norm(
                        model,
                        tokenizer,
                        alpha_reference_prompts,
                        steer_layer,
                        args.batch_size,
                        device,
                        capture_hook_point=hook_point,
                    )
                    alpha_values_by_hook_layer[hook_point][steer_layer] = [
                        avg_norm * scale for scale in alpha_scales
                    ]
                    alpha_entries_by_hook_layer[hook_point][steer_layer] = list(
                        zip(
                            alpha_scales,
                            alpha_values_by_hook_layer[hook_point][steer_layer],
                        )
                    )
                    logger.info(
                        "Hook {} | Layer {} avg norm={:.4f} -> alpha scales={}",
                        hook_point,
                        steer_layer,
                        avg_norm,
                        alpha_scales,
                    )

        vector_specs: List[Tuple[str, str, str, List[str]]] = []
        for concept in concepts:
            vector_file_stem = (
                random_direction_name
                if concept == RANDOM_DIRECTION_CONCEPT
                else concept
            )
            vector_specs.append(
                (
                    concept,
                    vector_file_stem,
                    os.path.join(
                        "assets",
                        "concept_vectors",
                        model_name,
                        f"{vector_file_stem}.pt",
                    ),
                    concept_prompts[concept],
                )
            )

        for concept_name, vector_name, vector_path, prompts in vector_specs:
            vector_tensor = load_vector(vector_path)
            results_by_hook_point = {}
            merged_results_by_hook_point = {}
            merged_alpha_values_by_hook_point = {}

            total_probe_evals = 0
            for hook_point in hook_points:
                for steer_layer in steer_layers:
                    alpha_entries: List[Tuple[float, float]] = [
                        (alpha, alpha) for alpha in manual_alpha_values
                    ]
                    if args.alpha_mode == "avg_norm":
                        alpha_entries = alpha_entries_by_hook_layer.get(
                            hook_point, {}
                        ).get(steer_layer, [])
                    if not alpha_entries:
                        continue
                    probe_layers = _resolve_layers_to_run(
                        args.probe_layers,
                        max_layers,
                        min_layer_exclusive=steer_layer,
                    )
                    if not probe_layers:
                        continue
                    total_probe_evals += len(alpha_entries) * len(probe_layers)

            with tqdm(
                total=total_probe_evals,
                desc=f"{model_name}:{vector_name}",
                unit="probe",
            ) as pbar:
                for hook_point in hook_points:
                    results = {}

                    for steer_layer in steer_layers:
                        alpha_entries: List[Tuple[float, float]] = [
                            (alpha, alpha) for alpha in manual_alpha_values
                        ]
                        if args.alpha_mode == "avg_norm":
                            alpha_entries = alpha_entries_by_hook_layer.get(
                                hook_point, {}
                            ).get(steer_layer, [])
                        if not alpha_entries:
                            continue
                        probe_layers = _resolve_layers_to_run(
                            args.probe_layers,
                            max_layers,
                            min_layer_exclusive=steer_layer,
                        )
                        if not probe_layers:
                            continue

                        if vector_tensor.ndim == 1:
                            steering_vector = vector_tensor
                        else:
                            steering_vector = vector_tensor[steer_layer]

                        alpha_results: Dict[str, Dict[str, Dict[str, float]]] = {}

                        for alpha_key_value, alpha in alpha_entries:
                            layer_features_before: Dict[int, List[torch.Tensor]] = {
                                layer_idx: [] for layer_idx in probe_layers
                            }
                            layer_features_after: Dict[int, List[torch.Tensor]] = {
                                layer_idx: [] for layer_idx in probe_layers
                            }

                            for i in range(0, len(prompts), args.batch_size):
                                batch_prompts = prompts[i : i + args.batch_size]
                                inputs = tokenizer(
                                    batch_prompts,
                                    return_tensors="pt",
                                    padding=True,
                                    truncation=True,
                                ).to(device)

                                capture_layers = sorted(
                                    set(probe_layers + [steer_layer])
                                )
                                captured_before = run_model_capture_layers(
                                    model,
                                    inputs.input_ids,
                                    capture_layers,
                                    device,
                                    capture_hook_point=hook_point,
                                    steering_vector=steering_vector,
                                    steer_layer=steer_layer,
                                    alpha=0.0,
                                )
                                captured_after = run_model_capture_layers(
                                    model,
                                    inputs.input_ids,
                                    capture_layers,
                                    device,
                                    capture_hook_point=hook_point,
                                    steering_vector=steering_vector,
                                    steer_layer=steer_layer,
                                    alpha=alpha,
                                )

                                token_mask = inputs.attention_mask.bool()
                                h_before_steer = captured_before[steer_layer]
                                h_after_steer = captured_after[steer_layer]
                                for layer_idx in probe_layers:
                                    h_before = captured_before[layer_idx]
                                    h_after = captured_after[layer_idx]
                                    if hook_point == "block_out":
                                        z_before = h_before - h_before_steer
                                        z_after = h_after - h_after_steer
                                    else:
                                        z_before = h_before
                                        z_after = h_after
                                    token_before = z_before[token_mask].float().cpu()
                                    token_after = z_after[token_mask].float().cpu()
                                    layer_features_before[layer_idx].append(
                                        token_before
                                    )
                                    layer_features_after[layer_idx].append(token_after)

                            alpha_key = str(alpha_key_value)
                            alpha_results[alpha_key] = {}
                            for layer_idx in probe_layers:
                                X_before = torch.cat(
                                    layer_features_before[layer_idx], dim=0
                                )
                                X_after = torch.cat(
                                    layer_features_after[layer_idx], dim=0
                                )
                                X = torch.cat([X_before, X_after], dim=0)
                                y = torch.cat(
                                    [
                                        torch.zeros(X_before.shape[0]),
                                        torch.ones(X_after.shape[0]),
                                    ],
                                    dim=0,
                                )
                                probe_seed = seed_from_name(
                                    f"{vector_name}-{hook_point}-{steer_layer}-{alpha}-{layer_idx}"
                                )
                                stats, weight, bias, mean, std = (
                                    train_eval_linear_probe(
                                        X,
                                        y,
                                        seed=probe_seed,
                                        epochs=args.epochs,
                                        lr=args.lr,
                                        test_ratio=args.test_ratio,
                                    )
                                )
                                std = std.clamp_min(1e-6)
                                raw_weight = weight / std

                                weight_dir = os.path.join(
                                    output_dir,
                                    "probe_weights",
                                    vector_name,
                                    f"run_{run_id}",
                                    f"hook_{hook_point}",
                                    f"steer_{steer_layer}",
                                    f"alpha_{alpha_to_slug(alpha_key_value if args.alpha_mode == 'avg_norm' else alpha)}",
                                )
                                os.makedirs(weight_dir, exist_ok=True)
                                weight_path = os.path.join(
                                    weight_dir, f"layer_{layer_idx}.pt"
                                )
                                torch.save(
                                    {
                                        "weight": weight,
                                        "bias": bias,
                                        "mean": mean,
                                        "std": std,
                                        "raw_weight": raw_weight,
                                        "vector": vector_name,
                                        "alpha": alpha
                                        if args.alpha_mode == "manual"
                                        else None,
                                        "alpha_scale": (
                                            alpha_key_value
                                            if args.alpha_mode == "avg_norm"
                                            else None
                                        ),
                                        "steer_layer": steer_layer,
                                        "layer": layer_idx,
                                        "hook_point": hook_point,
                                        "model": model_name_full,
                                    },
                                    weight_path,
                                )

                                alpha_results[alpha_key][str(layer_idx)] = stats
                                pbar.set_postfix_str(
                                    f"test_acc={stats['test_acc']:.4f}"
                                )
                                pbar.update(1)

                        results[str(steer_layer)] = {
                            "probe_layers": probe_layers,
                            "alpha_results": alpha_results,
                        }

                    merged_alpha_keys = set()
                    merged_results: Dict[str, Dict[str, Dict[str, float]]] = {}
                    for steer_layer in steer_layers:
                        layer_payload = results.get(str(steer_layer), {})
                        alpha_results = layer_payload.get("alpha_results", {})
                        for alpha_key in alpha_results:
                            merged_alpha_keys.add(alpha_key)

                    merged_alpha_keys_sorted = sorted(
                        merged_alpha_keys,
                        key=lambda x: float(x),
                    )
                    for alpha_key in merged_alpha_keys_sorted:
                        merged_results[alpha_key] = {}
                        for steer_layer in steer_layers:
                            layer_payload = results.get(str(steer_layer), {})
                            alpha_results = layer_payload.get("alpha_results", {})
                            layer_stats = alpha_results.get(alpha_key, {}).get(
                                str(steer_layer)
                            )
                            if layer_stats:
                                merged_results[alpha_key][str(steer_layer)] = (
                                    layer_stats
                                )

                    merged_alpha_values = [float(a) for a in merged_alpha_keys_sorted]

                    results_by_hook_point[hook_point] = results
                    merged_results_by_hook_point[hook_point] = merged_results
                    merged_alpha_values_by_hook_point[hook_point] = merged_alpha_values

            primary_hook_point = hook_points[0]

            alpha_scales_by_layer_str = {
                hook_point: {
                    str(layer_idx): [
                        scale
                        for scale, _alpha in alpha_entries_by_hook_layer.get(
                            hook_point, {}
                        ).get(layer_idx, [])
                    ]
                    for layer_idx in layer_map
                }
                for hook_point, layer_map in alpha_values_by_hook_layer.items()
            }
            hook_point_payload = {}
            for hook_point in hook_points:
                hook_point_payload[hook_point] = {
                    "alpha_values": merged_alpha_values_by_hook_point.get(
                        hook_point, []
                    ),
                    "results": merged_results_by_hook_point.get(hook_point, {}),
                    "results_by_steer_layer": results_by_hook_point.get(hook_point, {}),
                }

            payload = {
                "model": model_name_full,
                "concept": concept_name,
                "vector": vector_name,
                "run_id": run_id,
                "run_timestamp": run_timestamp,
                "params_hash": params_hash,
                "steer_layers": steer_layers,
                "alpha_mode": args.alpha_mode,
                "hook_points": hook_points,
                "hook_point": primary_hook_point if len(hook_points) == 1 else None,
                "alpha_values_manual": (
                    manual_alpha_values if args.alpha_mode == "manual" else None
                ),
                "alpha_scales": alpha_scales if args.alpha_mode == "avg_norm" else None,
                "alpha_scales_by_layer": (
                    alpha_scales_by_layer_str if args.alpha_mode == "avg_norm" else None
                ),
                "max_prompts": len(prompts),
                "hooks": hook_point_payload,
            }

            save_path = os.path.join(
                output_dir,
                f"probe_{vector_name}__run_{run_id}.json",
            )
            with open(save_path, "w") as f:
                json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
