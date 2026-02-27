import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import torch
import transformer_lens
import umap
from loguru import logger

from utils import (
    MODEL_LAYERS,
    apply_plot_style,
    build_plot_config,
    get_model_name_for_path,
    set_seed,
)


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


def extract_selected_token_hidden_states(
    model: transformer_lens.HookedTransformer,
    texts: list[str],
    token_refs: list[tuple[int, int]],
    layer_idx: int,
) -> torch.Tensor:
    tokens_by_sample: dict[int, list[int]] = {}
    for sample_idx, token_idx in token_refs:
        tokens_by_sample.setdefault(sample_idx, []).append(token_idx)

    collected: list[torch.Tensor] = []
    with torch.no_grad():
        for sample_idx, text in enumerate(texts):
            token_indices = tokens_by_sample.get(sample_idx, [])
            if not token_indices:
                continue
            _, cache = model.run_with_cache(text, stop_at_layer=layer_idx + 1)
            hidden_seq = get_layer_hidden_sequence(cache, layer_idx)
            max_token_idx = int(hidden_seq.shape[0]) - 1
            for token_idx in token_indices:
                clamped_idx = min(token_idx, max_token_idx)
                collected.append(hidden_seq[clamped_idx].detach())

    if not collected:
        raise ValueError("No hidden states found for selected tokens")

    return torch.stack(collected)


def compute_umap_2d(features: torch.Tensor, seed: int) -> torch.Tensor:
    features_np = features.detach().to(dtype=torch.float32, device="cpu").numpy()
    n_neighbors = min(15, features_np.shape[0] - 1)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="cosine",
        random_state=seed,
    )
    embedding = reducer.fit_transform(features_np)
    return torch.from_numpy(embedding)


def extract_selected_token_features_all_layers(
    model: transformer_lens.HookedTransformer,
    texts: list[str],
    token_refs: list[tuple[int, int]],
    layers: list[int],
    d_model: int,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    max_layer_idx = max(layers)
    outputs = torch.zeros(
        len(token_refs), len(layers), d_model, device=device, dtype=dtype
    )

    tokens_by_sample: dict[int, list[tuple[int, int]]] = {}
    for out_idx, (sample_idx, token_idx) in enumerate(token_refs):
        tokens_by_sample.setdefault(sample_idx, []).append((token_idx, out_idx))

    with torch.no_grad():
        for sample_idx, token_items in tokens_by_sample.items():
            _, cache = model.run_with_cache(
                texts[sample_idx], stop_at_layer=max_layer_idx + 1
            )
            for layer_pos, layer_idx in enumerate(layers):
                hidden_seq = get_layer_hidden_sequence(cache, layer_idx)
                max_token_idx = int(hidden_seq.shape[0]) - 1
                for token_idx, out_idx in token_items:
                    clamped_idx = min(token_idx, max_token_idx)
                    outputs[out_idx, layer_pos] = hidden_seq[clamped_idx]

    return outputs


def compute_tdnv_two_tasks(
    features_a: torch.Tensor,
    features_b: torch.Tensor,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    mean_a = features_a.mean(dim=0)
    mean_b = features_b.mean(dim=0)

    var_a = ((features_a - mean_a) ** 2).sum(dim=2).mean(dim=0)
    var_b = ((features_b - mean_b) ** 2).sum(dim=2).mean(dim=0)

    between = ((mean_a - mean_b) ** 2).sum(dim=1).clamp_min(eps)
    tdnv = (var_a + var_b) / between

    return {
        "mean_a": mean_a,
        "mean_b": mean_b,
        "within_var_a": var_a,
        "within_var_b": var_b,
        "between_dist": between,
        "tdnv": tdnv,
    }


def parse_layer_list(layer_text: str) -> list[int]:
    values: list[int] = []
    for part in layer_text.split(","):
        item = part.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError("best_layers must contain at least one layer index")

    unique_values: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value in seen:
            continue
        unique_values.append(value)
        seen.add(value)
    return unique_values


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
        "--best_layers",
        type=str,
        default="0,3,6,9,12,15,18,21,24",
        help="Comma-separated layer indices used to select closest tokens",
    )
    parser.add_argument(
        "--best_layer",
        type=int,
        default=None,
        help="Backward-compatible single layer index (overrides --best_layers)",
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
    parser.add_argument(
        "--plot_dir",
        type=Path,
        default=Path("plots"),
        help="Directory to save UMAP plot",
    )
    parser.add_argument(
        "--tdnv_out_dir",
        type=Path,
        default=Path("assets/tdnv"),
        help="Directory to save TDNV results",
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
    if args.num_closest_tokens <= 0:
        raise ValueError("num_closest_tokens must be > 0")

    best_layers = parse_layer_list(args.best_layers)
    if args.best_layer is not None:
        best_layers = [args.best_layer]
    for best_layer in best_layers:
        if best_layer < 0 or best_layer >= max_layers:
            raise ValueError(
                f"Invalid best_layer={best_layer}. Must be in [0, {max_layers - 1}]"
            )

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

    logger.info("Running best layers: {}", best_layers)

    d_model = model.cfg.d_model
    model_name = get_model_name_for_path(args.model)
    save_dir = args.out_dir / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.dataset.stem}_{args.text_key}"
    layers = list(range(max_layers))

    for best_layer in best_layers:
        logger.info("Processing best_layer={} ...", best_layer)

        pos_center = compute_center_point_for_layer(
            model=model,
            texts=pos_texts,
            layer_idx=best_layer,
            d_model=d_model,
            device=device,
            dtype=dtype,
        )
        neg_center = compute_center_point_for_layer(
            model=model,
            texts=neg_texts,
            layer_idx=best_layer,
            d_model=d_model,
            device=device,
            dtype=dtype,
        )

        pos_token_indices, pos_min_distances = select_closest_token_indices(
            model=model,
            texts=pos_texts,
            layer_idx=best_layer,
            center_point=pos_center,
        )
        neg_token_indices, neg_min_distances = select_closest_token_indices(
            model=model,
            texts=neg_texts,
            layer_idx=best_layer,
            center_point=neg_center,
        )

        pos_token_refs_topk, pos_topk_distances = select_global_topk_token_refs(
            model=model,
            texts=pos_texts,
            layer_idx=best_layer,
            center_point=pos_center,
            top_k=args.num_closest_tokens,
        )
        neg_token_refs_topk, neg_topk_distances = select_global_topk_token_refs(
            model=model,
            texts=neg_texts,
            layer_idx=best_layer,
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
            layers=layers,
            d_model=d_model,
            device=device,
            dtype=dtype,
        )

        concept_path = (
            save_dir / f"pku_pos_neg_center_token_layer{best_layer}_{stem}.pt"
        )
        torch.save(concept_vectors.detach().float().cpu(), concept_path)

        pos_hidden = extract_selected_token_hidden_states(
            model=model,
            texts=pos_texts,
            token_refs=pos_token_refs_topk,
            layer_idx=best_layer,
        )
        neg_hidden = extract_selected_token_hidden_states(
            model=model,
            texts=neg_texts,
            token_refs=neg_token_refs_topk,
            layer_idx=best_layer,
        )
        all_hidden = torch.cat([pos_hidden, neg_hidden], dim=0)
        projected = compute_umap_2d(all_hidden, seed=args.seed)

        plot_config = build_plot_config(
            {"figsize": (7, 6), "rc_params": {"axes.titlesize": 14}}
        )
        apply_plot_style(plot_config["style"], plot_config["rc_params"])

        fig, ax = plt.subplots(figsize=plot_config["figsize"])
        pos_count = pos_hidden.shape[0]
        ax.scatter(
            projected[:pos_count, 0],
            projected[:pos_count, 1],
            label="pos",
            color="#1f77b4",
            alpha=0.8,
            edgecolors="white",
            linewidths=0.6,
        )
        ax.scatter(
            projected[pos_count:, 0],
            projected[pos_count:, 1],
            label="neg",
            color="#ff7f0e",
            alpha=0.8,
            edgecolors="white",
            linewidths=0.6,
        )
        ax.set_title(
            "UMAP of Closest Tokens (Layer {})\n{} | {} | top {} tokens".format(
                best_layer,
                get_model_name_for_path(args.model),
                args.text_key,
                args.num_closest_tokens,
            )
        )
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.legend(frameon=False)
        ax.grid(True, linestyle="--", alpha=0.5)

        args.plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = (
            args.plot_dir / f"pku_center_token_umap_layer{best_layer}_{stem}.pdf"
        )
        fig.tight_layout()
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)

        pos_features = extract_selected_token_features_all_layers(
            model=model,
            texts=pos_texts,
            token_refs=pos_token_refs_topk,
            layers=layers,
            d_model=d_model,
            device=device,
            dtype=dtype,
        )
        neg_features = extract_selected_token_features_all_layers(
            model=model,
            texts=neg_texts,
            token_refs=neg_token_refs_topk,
            layers=layers,
            d_model=d_model,
            device=device,
            dtype=dtype,
        )
        tdnv_stats = compute_tdnv_two_tasks(pos_features, neg_features)
        tdnv = tdnv_stats["tdnv"].detach().float().cpu().numpy()
        within_pos = tdnv_stats["within_var_a"].detach().float().cpu().numpy()
        within_neg = tdnv_stats["within_var_b"].detach().float().cpu().numpy()
        between = tdnv_stats["between_dist"].detach().float().cpu().numpy()

        tdnv_dir = args.tdnv_out_dir / model_name
        tdnv_dir.mkdir(parents=True, exist_ok=True)
        tdnv_path = tdnv_dir / f"tdnv_center_token_layer{best_layer}_{stem}.json"
        tdnv_payload = {
            "dataset": str(args.dataset),
            "text_key": args.text_key,
            "model": args.model,
            "best_layer": best_layer,
            "num_closest_tokens": args.num_closest_tokens,
            "layers": layers,
            "seed": args.seed,
            "num_selected_tokens": {
                "safety": len(pos_token_refs_topk),
                "unsafety": len(neg_token_refs_topk),
            },
            "tdnv": tdnv.tolist(),
            "within_task_variance": {
                "safety": within_pos.tolist(),
                "unsafety": within_neg.tolist(),
            },
            "between_task_distance": between.tolist(),
            "tdnv_mean": float(tdnv.mean()),
            "tdnv_std": float(tdnv.std()),
        }
        with tdnv_path.open("w", encoding="utf-8") as f:
            json.dump(tdnv_payload, f, ensure_ascii=False, indent=2)

        tdnv_plot_path = (
            args.plot_dir / f"tdnv_center_token_layer{best_layer}_{stem}.pdf"
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(layers, tdnv, color="#1f77b4", linewidth=2, label="TDNV")
        ax.set_title("TDNV from Center-Selected Tokens (Qwen3)")
        ax.set_xlabel("Layer index")
        ax.set_ylabel("TDNV")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(tdnv_plot_path, dpi=220)
        plt.close(fig)

        meta_path = concept_path.with_suffix(".json")
        payload = {
            "dataset": str(args.dataset),
            "text_key": args.text_key,
            "model": args.model,
            "best_layer": best_layer,
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
        logger.info("Saved UMAP plot: {}", plot_path)
        logger.info("Saved TDNV results: {}", tdnv_path)
        logger.info("Saved TDNV plot: {}", tdnv_plot_path)


if __name__ == "__main__":
    main()
