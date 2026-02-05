import argparse
import json
import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import transformer_lens
from loguru import logger

from utils import (
    CONCEPT_CATEGORIES,
    MODEL_LAYERS,
    apply_plot_style,
    build_plot_config,
    get_model_name_for_path,
    load_concept_datasets,
    parse_layers_to_run,
    seed_from_name,
    set_seed,
)


def parse_example_sizes(sizes_arg: str) -> list[int]:
    sizes = [int(x.strip()) for x in sizes_arg.split(",") if x.strip()]
    return sorted(list({s for s in sizes if s > 0}))


def parse_reference_size(reference_arg: Optional[str]) -> Optional[int]:
    if reference_arg is None:
        return None
    if reference_arg.lower() == "all":
        return -1
    return int(reference_arg)


def sample_dataset(dataset, num_examples: int, seed: int):
    num_examples = min(num_examples, len(dataset))
    if num_examples >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=num_examples, replace=False)
    return dataset.select(indices.tolist())


def compute_hidden_sums(
    model: transformer_lens.HookedTransformer,
    dataset,
    dataset_key: str,
    layers: list[int],
    max_examples: int,
    device: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, int]:
    d_model = model.cfg.d_model
    layer_count = len(layers)
    sums = torch.zeros(layer_count, d_model, device=device, dtype=dtype)
    token_count = 0
    max_layer_idx = max(layers)

    for i in range(min(max_examples, len(dataset))):
        context = dataset[i][dataset_key]
        _, cache = model.run_with_cache(context, stop_at_layer=max_layer_idx + 1)
        for layer_pos, layer_idx in enumerate(layers):
            hidden = cache[f"blocks.{layer_idx}.hook_resid_post"].reshape(-1, d_model)
            sums[layer_pos] += hidden.sum(dim=0)
            if layer_pos == 0:
                token_count += hidden.shape[0]

    return sums, token_count


def compute_concept_vector(
    model: transformer_lens.HookedTransformer,
    positive_dataset,
    negative_dataset,
    dataset_key: str,
    layers: list[int],
    max_examples: int,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    pos_sum, pos_tokens = compute_hidden_sums(
        model,
        positive_dataset,
        dataset_key,
        layers,
        max_examples,
        device,
        dtype,
    )
    neg_sum, neg_tokens = compute_hidden_sums(
        model,
        negative_dataset,
        dataset_key,
        layers,
        max_examples,
        device,
        dtype,
    )

    pos_mean = pos_sum / max(pos_tokens, 1)
    neg_mean = neg_sum / max(neg_tokens, 1)
    concept_vector = F.normalize(pos_mean - neg_mean, dim=1)
    return concept_vector


def summarize_cosines(cosines: list[torch.Tensor]) -> dict:
    cos = torch.stack(cosines)
    return {
        "per_layer_mean": cos.mean(dim=0).tolist(),
        "per_layer_std": cos.std(dim=0, unbiased=False).tolist(),
        "mean": float(cos.mean().item()),
        "std": float(cos.std(unbiased=False).item()),
        "runs": cos.tolist(),
    }


def summarize_pairwise(vectors: list[torch.Tensor]) -> Optional[dict]:
    if len(vectors) < 2:
        return None
    pair_cos = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            pair_cos.append(F.cosine_similarity(vectors[i], vectors[j], dim=1))
    pair_cos = torch.stack(pair_cos)
    return {
        "per_layer_mean": pair_cos.mean(dim=0).tolist(),
        "per_layer_std": pair_cos.std(dim=0, unbiased=False).tolist(),
        "mean": float(pair_cos.mean().item()),
        "std": float(pair_cos.std(unbiased=False).item()),
    }


def plot_stability(results: dict, model_name: str, concept_name: str) -> None:
    import matplotlib.pyplot as plt

    plot_config = build_plot_config(
        {
            "figsize": (10, 6),
            "rc_params": {"axes.titlesize": 16, "axes.labelsize": 14},
        }
    )
    apply_plot_style(plot_config["style"], plot_config["rc_params"])

    sizes = [int(s) for s in results["example_sizes"]]
    sizes = sorted(sizes)

    means = []
    stds = []
    for size in sizes:
        metrics = results["metrics"][str(size)]["cosine_to_reference"]
        means.append(metrics["mean"])
        stds.append(metrics["std"])

    os.makedirs("plots", exist_ok=True)
    fig, ax = plt.subplots(figsize=plot_config["figsize"])
    ax.plot(sizes, means, marker="o", linewidth=2, color="#1f77b4")
    ax.fill_between(
        sizes,
        np.array(means) - np.array(stds),
        np.array(means) + np.array(stds),
        alpha=0.2,
        color="#1f77b4",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Number of examples")
    ax.set_ylabel("Cosine similarity to reference")
    ax.set_title(f"Stability vs. sample size: {concept_name}")
    ax.grid(True, linestyle="--", alpha=0.6)

    plot_path = os.path.join("plots", f"stability_{model_name}_{concept_name}.png")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Study concept vector stability vs. sample size"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help=f"Model name. Choices: {list(MODEL_LAYERS.keys())}",
    )
    parser.add_argument(
        "--concept_category",
        type=str,
        default="steering_detectable_format_json_format",
        help=f"Concept category. Choices: {list(CONCEPT_CATEGORIES.keys())}",
    )
    parser.add_argument(
        "--example_sizes",
        type=str,
        default="10,20,50,100",
        help="Comma-separated example counts to evaluate",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Random resamples per example size",
    )
    parser.add_argument(
        "--reference_size",
        type=str,
        default=None,
        help="Reference example size or 'all' (default: max(example_sizes))",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Comma-separated percentages or layer indices",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save_vectors",
        action="store_true",
        help="Save mean concept vectors per size",
    )
    args = parser.parse_args()

    if args.model not in MODEL_LAYERS:
        parser.error(
            f"Invalid model: {args.model}. Must be one of {list(MODEL_LAYERS.keys())}"
        )
    if args.concept_category not in CONCEPT_CATEGORIES:
        parser.error(
            f"Invalid concept_category: {args.concept_category}. Must be one of {list(CONCEPT_CATEGORIES.keys())}"
        )

    os.makedirs("logs", exist_ok=True)
    logger.add("logs/concept_vector_stability.log")
    logger.info(f"args: {args}")

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = transformer_lens.HookedTransformer.from_pretrained(
        args.model, device=device, dtype=dtype, trust_remote_code=True
    )

    positive_dataset, negative_dataset, dataset_key = load_concept_datasets(
        args.concept_category, CONCEPT_CATEGORIES[args.concept_category]
    )

    max_layers = model.cfg.n_layers
    layers_to_run = parse_layers_to_run(args.layers, max_layers)
    if not layers_to_run:
        raise ValueError("No valid layers selected")

    example_sizes = parse_example_sizes(args.example_sizes)
    if not example_sizes:
        raise ValueError("No valid example sizes provided")

    available_examples = min(len(positive_dataset), len(negative_dataset))
    if available_examples <= 0:
        raise ValueError("No available examples in positive/negative datasets")

    clamped_sizes = sorted({min(size, available_examples) for size in example_sizes})
    if not clamped_sizes:
        raise ValueError("No valid example sizes after clamping to dataset size")
    if any(size > available_examples for size in example_sizes):
        logger.warning(
            "Some example sizes exceed dataset size; clamping to %d",
            available_examples,
        )
    example_sizes = clamped_sizes

    reference_size = parse_reference_size(args.reference_size)
    if reference_size is None:
        reference_size = max(example_sizes)
    if reference_size < 0:
        reference_size = available_examples
    if reference_size > available_examples:
        logger.warning(
            "Reference size exceeds dataset size; clamping to %d",
            available_examples,
        )
        reference_size = available_examples
    if reference_size <= 0:
        raise ValueError("Reference size must be positive")

    logger.info(
        f"Layers: {layers_to_run}, Example sizes: {example_sizes}, Reference size: {reference_size}"
    )

    reference_seed = seed_from_name(f"{args.concept_category}-reference") + args.seed
    pos_ref = sample_dataset(positive_dataset, reference_size, reference_seed)
    neg_ref = sample_dataset(negative_dataset, reference_size, reference_seed + 1)

    with torch.no_grad():
        reference_vector = compute_concept_vector(
            model,
            pos_ref,
            neg_ref,
            dataset_key,
            layers_to_run,
            reference_size,
            device,
            dtype,
        ).detach()

    model_name = get_model_name_for_path(args.model)
    save_dir = os.path.join(
        "assets", "concept_vector_stability", model_name, args.concept_category
    )
    os.makedirs(save_dir, exist_ok=True)

    torch.save(reference_vector.cpu(), os.path.join(save_dir, "concept_vector_ref.pt"))

    results = {
        "model": args.model,
        "concept_category": args.concept_category,
        "layers": layers_to_run,
        "example_sizes": example_sizes,
        "reference_size": reference_size,
        "repeats": args.repeats,
        "metrics": {},
    }

    for size in example_sizes:
        logger.info(f"Evaluating size {size}")
        vectors = []
        cosines_to_ref = []

        for r in range(args.repeats):
            seed = seed_from_name(f"{args.concept_category}-{size}-{r}") + args.seed
            pos_sample = sample_dataset(positive_dataset, size, seed)
            neg_sample = sample_dataset(negative_dataset, size, seed + 1)

            with torch.no_grad():
                vec = compute_concept_vector(
                    model,
                    pos_sample,
                    neg_sample,
                    dataset_key,
                    layers_to_run,
                    size,
                    device,
                    dtype,
                ).detach()

            vec_cpu = vec.cpu()
            vectors.append(vec_cpu)
            cosines_to_ref.append(
                F.cosine_similarity(vec_cpu, reference_vector.cpu(), dim=1)
            )

        summary = {
            "cosine_to_reference": summarize_cosines(cosines_to_ref),
        }
        pairwise = summarize_pairwise(vectors)
        if pairwise is not None:
            summary["pairwise_cosine"] = pairwise

        if args.save_vectors:
            mean_vector = torch.stack(vectors).mean(dim=0)
            mean_vector = F.normalize(mean_vector, dim=1)
            torch.save(
                mean_vector,
                os.path.join(save_dir, f"concept_vector_n{size}_mean.pt"),
            )

        results["metrics"][str(size)] = summary

    results_path = os.path.join(save_dir, "stability_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    plot_stability(
        results,
        model_name=model_name,
        concept_name=args.concept_category,
    )
    logger.info(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
