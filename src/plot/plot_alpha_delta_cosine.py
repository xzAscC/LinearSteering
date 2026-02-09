# ruff: noqa: E402

import argparse
from pathlib import Path
import os
import json
import sys
from typing import Dict, List, Tuple

from loguru import logger
import torch
import transformers

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from plot.plot_probe_utils import compute_cosine_similarity
from plot.plot_probe_utils import plot_cosine_heatmap
from probe_utils import (
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
from utils import set_seed


def compute_delta_means_by_alpha(
    model,
    tokenizer,
    prompts: List[str],
    steering_vector: torch.Tensor,
    steer_layer: int,
    probe_layers: List[int],
    alpha_values: List[float],
    batch_size: int,
    device: str,
) -> Dict[float, Dict[int, torch.Tensor]]:
    delta_means: Dict[float, Dict[int, torch.Tensor]] = {}
    for alpha in alpha_values:
        totals: Dict[int, torch.Tensor] = {}
        counts: Dict[int, int] = {}
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
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

            token_mask = inputs.attention_mask.bool()
            for layer_idx in probe_layers:
                h_before = captured_before[layer_idx]
                h_after = captured_after[layer_idx]
                delta = (h_after - h_before)[token_mask].float().cpu()
                if layer_idx not in totals:
                    totals[layer_idx] = delta.sum(dim=0)
                    counts[layer_idx] = delta.shape[0]
                else:
                    totals[layer_idx] = totals[layer_idx] + delta.sum(dim=0)
                    counts[layer_idx] = counts[layer_idx] + delta.shape[0]

        delta_means[alpha] = {}
        for layer_idx in probe_layers:
            if counts.get(layer_idx, 0) == 0:
                raise RuntimeError(f"No delta tokens for layer {layer_idx}")
            delta_means[alpha][layer_idx] = totals[layer_idx] / counts[layer_idx]

    return delta_means


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot cosine similarity of mean deltas across alphas"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="Model name used for assets and loading",
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
    parser.add_argument(
        "--probe_layers",
        type=int,
        default=8,
        help="How many probe layers to capture (includes steer layer)",
    )
    parser.add_argument("--max_prompts", type=int, default=8196)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots/alpha_delta_cosine",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    model_name_full = args.model
    if model_name_full not in MODEL_LAYERS:
        raise ValueError(f"Unknown model: {model_name_full}")

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

    max_layers = MODEL_LAYERS[model_name_full]
    steer_layers = select_steer_layers(max_layers, total_layers=6)
    if not steer_layers:
        raise ValueError("No valid steer layers selected")

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_full, device_map=device, dtype=dtype, trust_remote_code=True
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_full, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_name = get_model_name_for_path(model_name_full)

    alpha_values_by_layer: Dict[int, List[float]] = {}
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
            logger.info(
                "Layer {} avg norm={:.4f} -> alphas={}",
                steer_layer,
                avg_norm,
                alpha_values_by_layer[steer_layer],
            )

    vector_specs: List[Tuple[str, str]] = []
    for concept in concepts:
        vector_specs.append(
            (
                concept,
                os.path.join("assets", "concept_vectors", model_name, f"{concept}.pt"),
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

        for steer_layer in steer_layers:
            alpha_values = manual_alpha_values
            if args.alpha_mode == "avg_norm":
                alpha_values = alpha_values_by_layer.get(steer_layer, [])
            if not alpha_values:
                logger.warning(f"Skipping steer layer {steer_layer}: no alpha values")
                continue

            probe_layers = select_probe_layers(
                steer_layer,
                max_layers,
                total_layers=args.probe_layers,
            )
            if not probe_layers:
                logger.warning(f"Skipping steer layer {steer_layer}: no probe layers")
                continue

            if vector_tensor.ndim == 1:
                steering_vector = vector_tensor
            else:
                steering_vector = vector_tensor[steer_layer]

            delta_means = compute_delta_means_by_alpha(
                model,
                tokenizer,
                prompts,
                steering_vector,
                steer_layer,
                probe_layers,
                alpha_values,
                args.batch_size,
                device,
            )

            alpha_labels = [
                get_alpha_label(alpha, args.alpha_mode, alpha_scales)
                for alpha in alpha_values
            ]

            for probe_layer in probe_layers:
                vectors = [delta_means[alpha][probe_layer] for alpha in alpha_values]
                cosine = compute_cosine_similarity(torch.stack(vectors, dim=0))
                output_path = os.path.join(
                    args.output_dir,
                    model_name,
                    vector_name,
                    f"steer_{steer_layer}",
                    f"probe_{probe_layer}.png",
                )
                plot_cosine_heatmap(
                    cosine,
                    alpha_labels,
                    title=(
                        f"Delta cosine ({vector_name}) steer{steer_layer} "
                        f"probe{probe_layer}"
                    ),
                    output_path=output_path,
                    xlabel="Alpha index",
                    ylabel="Alpha index",
                )

                save_path = output_path.replace(".png", ".json")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                payload = {
                    "model": model_name_full,
                    "vector": vector_name,
                    "steer_layer": steer_layer,
                    "probe_layer": probe_layer,
                    "alpha_mode": args.alpha_mode,
                    "alpha_values": alpha_values,
                    "alpha_labels": alpha_labels,
                    "alpha_values_manual": (
                        manual_alpha_values if args.alpha_mode == "manual" else None
                    ),
                    "alpha_scales": (
                        alpha_scales if args.alpha_mode == "avg_norm" else None
                    ),
                    "cosine": cosine.cpu().numpy().tolist(),
                }
                with open(save_path, "w") as f:
                    json.dump(payload, f, indent=2)
                logger.info(f"Saved cosine data to {save_path}")


if __name__ == "__main__":
    main()
