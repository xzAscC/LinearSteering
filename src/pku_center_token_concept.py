import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformer_lens
from loguru import logger

from utils import MODEL_LAYERS, get_model_name_for_path, set_seed


def load_pos_neg_texts(
    dataset_path: Path, text_key: str
) -> tuple[list[str], list[str]]:
    pos_texts: list[str] = []
    neg_texts: list[str] = []

    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            label = item["label"]
            text = item[text_key]
            if label == "pos":
                pos_texts.append(text)
            elif label == "neg":
                neg_texts.append(text)

    return pos_texts, neg_texts


def sample_texts(
    texts: list[str], max_samples: int | None, rng: np.random.Generator
) -> list[str]:
    if max_samples is None or max_samples >= len(texts):
        return texts
    indices = rng.choice(len(texts), size=max_samples, replace=False)
    return [texts[i] for i in indices.tolist()]


def get_layer_hidden_sequence(cache: Any, layer_idx: int) -> torch.Tensor:
    hidden = cache[f"blocks.{layer_idx}.hook_resid_post"]
    if hidden.ndim == 3:
        return hidden[0]
    return hidden


def compute_center_point_for_layer(
    model: transformer_lens.HookedTransformer,
    texts: list[str],
    layer_idx: int,
    d_model: int,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    center_sum = torch.zeros(d_model, device=device, dtype=dtype)
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            _, cache = model.run_with_cache(text, stop_at_layer=layer_idx + 1)
            hidden_seq = get_layer_hidden_sequence(cache, layer_idx)
            center_sum += hidden_seq.sum(dim=0)
            total_tokens += int(hidden_seq.shape[0])

    if total_tokens == 0:
        raise ValueError("No tokens found while computing class center")
    return center_sum / total_tokens


def select_closest_token_indices(
    model: transformer_lens.HookedTransformer,
    texts: list[str],
    layer_idx: int,
    center_point: torch.Tensor,
) -> tuple[list[int], list[float]]:
    token_indices: list[int] = []
    min_distances: list[float] = []

    with torch.no_grad():
        for text in texts:
            _, cache = model.run_with_cache(text, stop_at_layer=layer_idx + 1)
            hidden_seq = get_layer_hidden_sequence(cache, layer_idx)
            distances = torch.norm(hidden_seq - center_point.unsqueeze(0), dim=1)
            closest_idx = int(torch.argmin(distances).item())
            token_indices.append(closest_idx)
            min_distances.append(
                float(distances[closest_idx].detach().float().cpu().item())
            )

    return token_indices, min_distances


def select_global_topk_token_refs(
    model: transformer_lens.HookedTransformer,
    texts: list[str],
    layer_idx: int,
    center_point: torch.Tensor,
    top_k: int,
) -> tuple[list[tuple[int, int]], list[float]]:
    token_refs: list[tuple[int, int]] = []
    token_distances: list[float] = []

    with torch.no_grad():
        for sample_idx, text in enumerate(texts):
            _, cache = model.run_with_cache(text, stop_at_layer=layer_idx + 1)
            hidden_seq = get_layer_hidden_sequence(cache, layer_idx)
            distances = torch.norm(hidden_seq - center_point.unsqueeze(0), dim=1)

            for token_idx in range(int(hidden_seq.shape[0])):
                token_refs.append((sample_idx, token_idx))
                token_distances.append(
                    float(distances[token_idx].detach().float().cpu().item())
                )

    if not token_refs:
        raise ValueError("No tokens found while selecting closest tokens")

    effective_k = min(top_k, len(token_refs))
    sorted_indices = sorted(range(len(token_refs)), key=lambda i: token_distances[i])
    selected_indices = sorted_indices[:effective_k]

    selected_refs = [token_refs[i] for i in selected_indices]
    selected_distances = [token_distances[i] for i in selected_indices]
    return selected_refs, selected_distances


def build_topk_token_literal_records(
    model: transformer_lens.HookedTransformer,
    texts: list[str],
    token_refs: list[tuple[int, int]],
    token_distances: list[float],
) -> list[dict[str, Any]]:
    token_literals_by_sample: dict[int, list[str]] = {}
    records: list[dict[str, Any]] = []

    for idx, (sample_idx, token_idx) in enumerate(token_refs):
        if sample_idx not in token_literals_by_sample:
            token_literals_by_sample[sample_idx] = model.to_str_tokens(
                texts[sample_idx]
            )

        sample_tokens = token_literals_by_sample[sample_idx]
        token_literal = ""
        if 0 <= token_idx < len(sample_tokens):
            token_literal = sample_tokens[token_idx]

        records.append(
            {
                "sample_index": sample_idx,
                "token_index": token_idx,
                "token_literal": token_literal,
                "distance_to_center": token_distances[idx],
            }
        )

    return records


def compute_concept_vectors_from_selected_tokens(
    model: transformer_lens.HookedTransformer,
    pos_texts: list[str],
    neg_texts: list[str],
    pos_token_refs: list[tuple[int, int]],
    neg_token_refs: list[tuple[int, int]],
    layers: list[int],
    d_model: int,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    pos_mean = torch.zeros(len(layers), d_model, device=device, dtype=dtype)
    neg_mean = torch.zeros(len(layers), d_model, device=device, dtype=dtype)

    max_layer_idx = max(layers)
    pos_by_sample: dict[int, list[int]] = {}
    neg_by_sample: dict[int, list[int]] = {}
    for sample_idx, token_idx in pos_token_refs:
        pos_by_sample.setdefault(sample_idx, []).append(token_idx)
    for sample_idx, token_idx in neg_token_refs:
        neg_by_sample.setdefault(sample_idx, []).append(token_idx)

    with torch.no_grad():
        for sample_idx, text in enumerate(pos_texts):
            token_indices = pos_by_sample.get(sample_idx, [])
            if not token_indices:
                continue
            _, cache = model.run_with_cache(text, stop_at_layer=max_layer_idx + 1)
            for layer_pos, layer_idx in enumerate(layers):
                hidden_seq = get_layer_hidden_sequence(cache, layer_idx)
                max_token_idx = int(hidden_seq.shape[0]) - 1
                for token_idx in token_indices:
                    clamped_idx = min(token_idx, max_token_idx)
                    pos_mean[layer_pos] += hidden_seq[clamped_idx]

        for sample_idx, text in enumerate(neg_texts):
            token_indices = neg_by_sample.get(sample_idx, [])
            if not token_indices:
                continue
            _, cache = model.run_with_cache(text, stop_at_layer=max_layer_idx + 1)
            for layer_pos, layer_idx in enumerate(layers):
                hidden_seq = get_layer_hidden_sequence(cache, layer_idx)
                max_token_idx = int(hidden_seq.shape[0]) - 1
                for token_idx in token_indices:
                    clamped_idx = min(token_idx, max_token_idx)
                    neg_mean[layer_pos] += hidden_seq[clamped_idx]

    pos_mean = pos_mean / max(len(pos_token_refs), 1)
    neg_mean = neg_mean / max(len(neg_token_refs), 1)

    concept_vectors = pos_mean - neg_mean
    return torch.nn.functional.normalize(concept_vectors, dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select closest tokens to pos/neg center points at a target layer, "
            "then recompute concept vectors for all layers."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dataset/pku_saferlhf_pos_neg_100x2_minimal.jsonl"),
        help="JSONL path with prompt/response/label fields",
    )
    parser.add_argument(
        "--text_key",
        type=str,
        default="prompt",
        choices=["prompt", "response"],
        help="Which field to use for representation extraction",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help=f"Model name. Choices: {list(MODEL_LAYERS.keys())}",
    )
    parser.add_argument(
        "--best_layer",
        type=int,
        default=15,
        help="Layer index used to select closest tokens",
    )
    parser.add_argument(
        "--max_samples_per_class",
        type=int,
        default=0,
        help="Max examples per class; <=0 means use all",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num_closest_tokens",
        type=int,
        default=100,
        help="Number of globally closest tokens per class used for concept vectors",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("assets/concept_vectors"),
        help="Directory to save concept vectors",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.model not in MODEL_LAYERS:
        raise ValueError(
            f"Invalid model: {args.model}. Must be one of {list(MODEL_LAYERS.keys())}"
        )
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    logger.info("Loading model {} on {} ({})", args.model, device, dtype)
    model = transformer_lens.HookedTransformer.from_pretrained(
        args.model,
        device=device,
        dtype=dtype,
        trust_remote_code=True,
    )

    max_layers = model.cfg.n_layers
    if args.best_layer < 0 or args.best_layer >= max_layers:
        raise ValueError(
            f"Invalid best_layer={args.best_layer}. Must be in [0, {max_layers - 1}]"
        )
    if args.num_closest_tokens <= 0:
        raise ValueError("num_closest_tokens must be > 0")

    pos_texts, neg_texts = load_pos_neg_texts(args.dataset, args.text_key)
    if len(pos_texts) == 0 or len(neg_texts) == 0:
        raise ValueError("No valid pos/neg examples found in dataset")

    max_samples = args.max_samples_per_class if args.max_samples_per_class > 0 else None
    rng = np.random.default_rng(args.seed)
    pos_texts = sample_texts(pos_texts, max_samples, rng)
    neg_texts = sample_texts(neg_texts, max_samples, rng)

    logger.info(
        "Using {} safety and {} unsafety samples",
        len(pos_texts),
        len(neg_texts),
    )

    d_model = model.cfg.d_model
    pos_center = compute_center_point_for_layer(
        model=model,
        texts=pos_texts,
        layer_idx=args.best_layer,
        d_model=d_model,
        device=device,
        dtype=dtype,
    )
    neg_center = compute_center_point_for_layer(
        model=model,
        texts=neg_texts,
        layer_idx=args.best_layer,
        d_model=d_model,
        device=device,
        dtype=dtype,
    )

    pos_token_indices, pos_min_distances = select_closest_token_indices(
        model=model,
        texts=pos_texts,
        layer_idx=args.best_layer,
        center_point=pos_center,
    )
    neg_token_indices, neg_min_distances = select_closest_token_indices(
        model=model,
        texts=neg_texts,
        layer_idx=args.best_layer,
        center_point=neg_center,
    )

    pos_token_refs_topk, pos_topk_distances = select_global_topk_token_refs(
        model=model,
        texts=pos_texts,
        layer_idx=args.best_layer,
        center_point=pos_center,
        top_k=args.num_closest_tokens,
    )
    neg_token_refs_topk, neg_topk_distances = select_global_topk_token_refs(
        model=model,
        texts=neg_texts,
        layer_idx=args.best_layer,
        center_point=neg_center,
        top_k=args.num_closest_tokens,
    )

    pos_topk_literals = build_topk_token_literal_records(
        model=model,
        texts=pos_texts,
        token_refs=pos_token_refs_topk,
        token_distances=pos_topk_distances,
    )
    neg_topk_literals = build_topk_token_literal_records(
        model=model,
        texts=neg_texts,
        token_refs=neg_token_refs_topk,
        token_distances=neg_topk_distances,
    )

    concept_vectors = compute_concept_vectors_from_selected_tokens(
        model=model,
        pos_texts=pos_texts,
        neg_texts=neg_texts,
        pos_token_refs=pos_token_refs_topk,
        neg_token_refs=neg_token_refs_topk,
        layers=list(range(max_layers)),
        d_model=d_model,
        device=device,
        dtype=dtype,
    )

    model_name = get_model_name_for_path(args.model)
    save_dir = args.out_dir / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.dataset.stem}_{args.text_key}"
    concept_path = (
        save_dir / f"pku_pos_neg_center_token_layer{args.best_layer}_{stem}.pt"
    )
    torch.save(concept_vectors.detach().float().cpu(), concept_path)

    meta_path = concept_path.with_suffix(".json")
    payload = {
        "dataset": str(args.dataset),
        "text_key": args.text_key,
        "model": args.model,
        "best_layer": args.best_layer,
        "num_closest_tokens": args.num_closest_tokens,
        "num_layers": max_layers,
        "seed": args.seed,
        "num_samples": {
            "safety": len(pos_texts),
            "unsafety": len(neg_texts),
        },
        "selected_token_index": {
            "safety": pos_token_indices,
            "unsafety": neg_token_indices,
        },
        "selected_token_distance_to_center": {
            "safety": pos_min_distances,
            "unsafety": neg_min_distances,
        },
        "selected_topk_token_ref": {
            "safety": [
                {"sample_index": sample_idx, "token_index": token_idx}
                for sample_idx, token_idx in pos_token_refs_topk
            ],
            "unsafety": [
                {"sample_index": sample_idx, "token_index": token_idx}
                for sample_idx, token_idx in neg_token_refs_topk
            ],
        },
        "selected_topk_token_distance_to_center": {
            "safety": pos_topk_distances,
            "unsafety": neg_topk_distances,
        },
        "selected_topk_token_literal": {
            "safety": pos_topk_literals,
            "unsafety": neg_topk_literals,
        },
        "output_path": str(concept_path),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info("Saved concept vectors: {}", concept_path)
    logger.info("Saved metadata: {}", meta_path)


if __name__ == "__main__":
    main()
