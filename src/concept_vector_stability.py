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
    get_model_name_for_path,
    load_concept_datasets,
    parse_layers_to_run,
    seed_from_name,
    set_seed,
)


def parse_example_sizes(sizes_arg: str) -> list[int]:
    sizes = [int(x.strip()) for x in sizes_arg.split(",") if x.strip()]
    return sorted({s for s in sizes if s > 0})


def parse_reference_size(reference_arg: Optional[str]) -> Optional[int]:
    if reference_arg is None:
        return None
    if reference_arg.lower() == "all":
        return -1
    return int(reference_arg)


def sample_dataset(dataset, num_examples: int, seed: int):
    num_examples = min(num_examples, len(dataset))
    rng = np.random.default_rng(seed)
    if num_examples >= len(dataset):
        indices = rng.permutation(len(dataset))
        return dataset.select(indices.tolist())
    indices = rng.choice(len(dataset), size=num_examples, replace=False)
    return dataset.select(indices.tolist())


def compute_hidden_sums(
    model: transformer_lens.HookedTransformer,
    dataset,
    dataset_key: str,
    layers: list[int],
    max_tokens: int,
    device: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, int]:
    d_model = model.cfg.d_model
    layer_count = len(layers)
    sums = torch.zeros(layer_count, d_model, device=device, dtype=dtype)
    token_count = 0
    max_layer_idx = max(layers)

    if max_tokens <= 0:
        max_tokens = -1

    for i in range(len(dataset)):
        if 0 < max_tokens <= token_count:
            break
        context = dataset[i][dataset_key]
        _, cache = model.run_with_cache(context, stop_at_layer=max_layer_idx + 1)
        first_hidden = cache[f"blocks.{layers[0]}.hook_resid_post"].reshape(-1, d_model)
        seq_len = first_hidden.shape[0]
        if max_tokens > 0:
            remaining = max_tokens - token_count
            used_tokens = min(seq_len, remaining)
        else:
            used_tokens = seq_len
        for layer_pos, layer_idx in enumerate(layers):
            hidden = cache[f"blocks.{layer_idx}.hook_resid_post"].reshape(-1, d_model)
            if used_tokens < seq_len:
                hidden = hidden[:used_tokens]
            sums[layer_pos] += hidden.sum(dim=0)
        token_count += used_tokens

    return sums, token_count


def compute_concept_vector(
    model: transformer_lens.HookedTransformer,
    positive_dataset,
    negative_dataset,
    dataset_key: str,
    layers: list[int],
    max_tokens: int,
    device: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, int, int]:
    pos_sum, pos_tokens = compute_hidden_sums(
        model,
        positive_dataset,
        dataset_key,
        layers,
        max_tokens,
        device,
        dtype,
    )
    neg_sum, neg_tokens = compute_hidden_sums(
        model,
        negative_dataset,
        dataset_key,
        layers,
        max_tokens,
        device,
        dtype,
    )

    pos_mean = pos_sum / max(pos_tokens, 1)
    neg_mean = neg_sum / max(neg_tokens, 1)
    concept_vector = F.normalize(pos_mean - neg_mean, dim=1)
    return concept_vector, pos_tokens, neg_tokens


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


def build_cosine_by_run(
    cosines: list[torch.Tensor],
    layers: list[int],
    tokens_by_run: list[dict[str, int]],
) -> list[dict]:
    records = []
    for run_idx, cosine in enumerate(cosines):
        per_layer = []
        for layer_pos, layer_idx in enumerate(layers):
            per_layer.append(
                {
                    "layer": layer_idx,
                    "cosine_similarity": float(cosine[layer_pos].item()),
                }
            )

        records.append(
            {
                "run": run_idx,
                "tokens": tokens_by_run[run_idx],
                "mean": float(cosine.mean().item()),
                "std": float(cosine.std(unbiased=False).item()),
                "per_layer": per_layer,
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Study concept vector stability vs. token budget"
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
        default="all",
        help=(f"Concept category or 'all'. Choices: {list(CONCEPT_CATEGORIES.keys())}"),
    )
    parser.add_argument(
        "--example_sizes",
        type=str,
        default="10,20,50,100,200,400",
        help="Comma-separated token counts to evaluate",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Random resamples per token size",
    )
    parser.add_argument(
        "--reference_size",
        type=str,
        default="all",
        help="Reference token count or 'all' (default: max(example_sizes))",
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
    if (
        args.concept_category != "all"
        and args.concept_category not in CONCEPT_CATEGORIES
    ):
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

    max_layers = model.cfg.n_layers
    layers_to_run = parse_layers_to_run(args.layers, max_layers)
    if not layers_to_run:
        raise ValueError("No valid layers selected")

    token_sizes = parse_example_sizes(args.example_sizes)
    if not token_sizes:
        raise ValueError("No valid token sizes provided")

    reference_size = parse_reference_size(args.reference_size)
    if reference_size is None:
        reference_size = max(token_sizes)
    if reference_size < 0:
        reference_size = -1
    if reference_size == 0:
        raise ValueError("Reference size must be positive")

    display_reference = "all" if reference_size < 0 else reference_size
    logger.info(
        f"Layers: {layers_to_run}, Token sizes: {token_sizes}, Reference size: {display_reference}"
    )

    concept_categories = (
        list(CONCEPT_CATEGORIES.keys())
        if args.concept_category == "all"
        else [args.concept_category]
    )

    model_name = get_model_name_for_path(args.model)

    for concept_category in concept_categories:
        logger.info(f"Processing concept category: {concept_category}")

        positive_dataset, negative_dataset, dataset_key = load_concept_datasets(
            concept_category, CONCEPT_CATEGORIES[concept_category]
        )

        pos_dataset_size = len(positive_dataset)
        neg_dataset_size = len(negative_dataset)
        if min(pos_dataset_size, neg_dataset_size) <= 0:
            raise ValueError("No available examples in positive/negative datasets")

        reference_seed = seed_from_name(f"{concept_category}-reference") + args.seed
        pos_ref = sample_dataset(positive_dataset, pos_dataset_size, reference_seed)
        neg_ref = sample_dataset(negative_dataset, neg_dataset_size, reference_seed + 1)

        with torch.no_grad():
            reference_vector, ref_pos_tokens, ref_neg_tokens = compute_concept_vector(
                model,
                pos_ref,
                neg_ref,
                dataset_key,
                layers_to_run,
                reference_size,
                device,
                dtype,
            )
            reference_vector = reference_vector.detach()
        reference_vector_cpu = reference_vector.cpu()
        reference_tokens_total = ref_pos_tokens + ref_neg_tokens

        save_dir = os.path.join(
            "assets", "concept_vector_stability_cosine", model_name, concept_category
        )
        os.makedirs(save_dir, exist_ok=True)

        torch.save(
            reference_vector_cpu, os.path.join(save_dir, "concept_vector_ref.pt")
        )

        results = {
            "model": args.model,
            "model_name": model_name,
            "concept_name": concept_category,
            "concept_category": concept_category,
            "layers": layers_to_run,
            "example_sizes": token_sizes,
            "token_sizes": token_sizes,
            "reference_size": reference_size,
            "reference_tokens": {
                "pos_tokens": ref_pos_tokens,
                "neg_tokens": ref_neg_tokens,
                "total_tokens": reference_tokens_total,
            },
            "repeats": args.repeats,
            "metrics": {},
        }

        for size in token_sizes:
            logger.info(f"Evaluating token size {size}")
            vectors = []
            cosines_to_ref = []
            tokens_by_run = []

            for r in range(args.repeats):
                seed = seed_from_name(f"{concept_category}-{size}-{r}") + args.seed
                pos_sample = sample_dataset(positive_dataset, pos_dataset_size, seed)
                neg_sample = sample_dataset(
                    negative_dataset, neg_dataset_size, seed + 1
                )

                with torch.no_grad():
                    vec, pos_tokens, neg_tokens = compute_concept_vector(
                        model,
                        pos_sample,
                        neg_sample,
                        dataset_key,
                        layers_to_run,
                        size,
                        device,
                        dtype,
                    )
                    vec = vec.detach()

                vec_cpu = vec.cpu()
                vectors.append(vec_cpu)
                cosines_to_ref.append(
                    F.cosine_similarity(vec_cpu, reference_vector_cpu, dim=1)
                )
                tokens_by_run.append(
                    {
                        "pos_tokens": pos_tokens,
                        "neg_tokens": neg_tokens,
                        "total_tokens": pos_tokens + neg_tokens,
                    }
                )

            summary = {
                "cosine_to_reference": summarize_cosines(cosines_to_ref),
                "cosine_by_run": build_cosine_by_run(
                    cosines_to_ref,
                    layers_to_run,
                    tokens_by_run,
                ),
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

        results_path = os.path.join(save_dir, "cosine_similarity.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
