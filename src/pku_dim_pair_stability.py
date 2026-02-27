import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import transformer_lens
from loguru import logger

from utils import MODEL_LAYERS, get_model_name_for_path, parse_layers_to_run, set_seed


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


def compute_prompt_level_features(
    model: transformer_lens.HookedTransformer,
    texts: list[str],
    layers: list[int],
    d_model: int,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    max_layer_idx = max(layers)
    outputs = torch.zeros(len(texts), len(layers), d_model, device=device, dtype=dtype)

    with torch.no_grad():
        for idx, text in enumerate(texts):
            _, cache = model.run_with_cache(text, stop_at_layer=max_layer_idx + 1)
            for layer_pos, layer_idx in enumerate(layers):
                hidden = cache[f"blocks.{layer_idx}.hook_resid_post"].reshape(
                    -1, d_model
                )
                outputs[idx, layer_pos] = hidden.mean(dim=0)

    return outputs


def compute_dim_vector(
    features_pos: torch.Tensor, features_neg: torch.Tensor
) -> torch.Tensor:
    vec = features_pos.mean(dim=0) - features_neg.mean(dim=0)
    return F.normalize(vec, dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate how many pos/neg prompt pairs are needed for a stable "
            "Difference-in-Means concept vector."
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
        help="Which field to use for concept extraction",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help=f"Model name. Choices: {list(MODEL_LAYERS.keys())}",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Comma-separated percentages or layer indices, or 'all'",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=100,
        help="Evaluate pair counts from 1 to max_pairs",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=30,
        help="Random sampling repeats per pair count",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Cosine threshold used to report stability pair count",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("assets/concept_vector_stability_cosine"),
        help="Directory to save json results",
    )
    parser.add_argument(
        "--plot_dir",
        type=Path,
        default=Path("plots"),
        help="Directory to save stability plot",
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
    if args.max_pairs <= 0:
        raise ValueError("max_pairs must be positive")
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")

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
    layers = parse_layers_to_run(args.layers, max_layers)
    if not layers:
        raise ValueError("No valid layers selected")

    pos_texts, neg_texts = load_pos_neg_texts(args.dataset, args.text_key)
    available_pairs = min(len(pos_texts), len(neg_texts))
    if available_pairs == 0:
        raise ValueError("No valid pos/neg examples found in dataset")
    max_pairs = min(args.max_pairs, available_pairs)

    logger.info(
        "Loaded dataset pos={} neg={}, evaluating n=1..{}",
        len(pos_texts),
        len(neg_texts),
        max_pairs,
    )

    d_model = model.cfg.d_model
    pos_features = compute_prompt_level_features(
        model,
        pos_texts,
        layers,
        d_model,
        device,
        dtype,
    )
    neg_features = compute_prompt_level_features(
        model,
        neg_texts,
        layers,
        d_model,
        device,
        dtype,
    )

    ref_vector = compute_dim_vector(pos_features, neg_features)

    rng = np.random.default_rng(args.seed)
    pair_counts = np.arange(1, max_pairs + 1)
    mean_cosines = np.zeros_like(pair_counts, dtype=float)
    std_cosines = np.zeros_like(pair_counts, dtype=float)
    per_layer_means: list[list[float]] = []

    all_runs_mean = []
    for n in pair_counts:
        run_scores = []
        run_scores_per_layer = []
        for _ in range(args.repeats):
            pos_idx = rng.choice(len(pos_features), size=n, replace=False)
            neg_idx = rng.choice(len(neg_features), size=n, replace=False)
            vec_n = compute_dim_vector(pos_features[pos_idx], neg_features[neg_idx])
            cosine_per_layer = F.cosine_similarity(vec_n, ref_vector, dim=1)
            run_scores_per_layer.append(cosine_per_layer.detach().float().cpu().numpy())
            run_scores.append(float(cosine_per_layer.mean().item()))

        run_scores_np = np.asarray(run_scores, dtype=float)
        run_scores_layer_np = np.asarray(run_scores_per_layer, dtype=float)
        mean_cosines[n - 1] = float(run_scores_np.mean())
        std_cosines[n - 1] = float(run_scores_np.std(ddof=0))
        per_layer_means.append(run_scores_layer_np.mean(axis=0).tolist())
        all_runs_mean.append(run_scores)

    stable_n = None
    for n, score in zip(pair_counts, mean_cosines):
        if score >= args.threshold:
            stable_n = int(n)
            break

    model_name = get_model_name_for_path(args.model)
    stem = f"{args.dataset.stem}_{args.text_key}_{model_name}"
    result_dir = args.out_dir / model_name
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"dim_pair_stability_{stem}.json"

    payload = {
        "dataset": str(args.dataset),
        "text_key": args.text_key,
        "model": args.model,
        "layers": layers,
        "max_pairs": int(max_pairs),
        "repeats": args.repeats,
        "seed": args.seed,
        "threshold": args.threshold,
        "stable_pairs_at_threshold": stable_n,
        "pair_counts": pair_counts.tolist(),
        "mean_cosine_to_full": mean_cosines.tolist(),
        "std_cosine_to_full": std_cosines.tolist(),
        "per_layer_mean_cosine_to_full": per_layer_means,
        "all_run_mean_cosine_to_full": all_runs_mean,
    }
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    args.plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = args.plot_dir / f"dim_pair_stability_{stem}.pdf"
    plt.style.use("seaborn-v0_8-paper")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        pair_counts, mean_cosines, color="#1f77b4", linewidth=2, label="Mean cosine"
    )
    ax.fill_between(
        pair_counts,
        mean_cosines - std_cosines,
        mean_cosines + std_cosines,
        color="#1f77b4",
        alpha=0.2,
        label="±1 std",
    )
    ax.axhline(y=args.threshold, color="#d62728", linestyle="--", label="Threshold")
    ax.set_title("Difference-in-Means Stability vs Prompt Pairs")
    ax.set_xlabel("Number of pos/neg prompt pairs (n)")
    ax.set_ylabel("Cosine similarity to full concept vector")
    ax.set_xlim(1, max_pairs)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)

    logger.info("Saved results: {}", result_path)
    logger.info("Saved plot: {}", plot_path)
    logger.info("Stable pair count at threshold {}: {}", args.threshold, stable_n)


if __name__ == "__main__":
    main()
