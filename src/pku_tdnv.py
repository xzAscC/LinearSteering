import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
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


def sample_texts(
    texts: list[str], max_samples: int | None, rng: np.random.Generator
) -> list[str]:
    if max_samples is None or max_samples >= len(texts):
        return texts
    indices = rng.choice(len(texts), size=max_samples, replace=False)
    return [texts[i] for i in indices.tolist()]


def compute_last_token_features(
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
                hidden = cache[f"blocks.{layer_idx}.hook_resid_post"]
                if hidden.ndim == 3:
                    outputs[idx, layer_pos] = hidden[0, -1]
                else:
                    outputs[idx, layer_pos] = hidden[-1]

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute TDNV layer-wise for PKU safety (pos) vs unsafety (neg) "
            "using last-token hidden representations."
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
        "--layers",
        type=str,
        default="all",
        help="Comma-separated percentages or layer indices, or 'all'",
    )
    parser.add_argument(
        "--max_samples_per_class",
        type=int,
        default=0,
        help="Max examples per class; <=0 means use all",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("assets/tdnv"),
        help="Directory to save JSON results",
    )
    parser.add_argument(
        "--plot_dir",
        type=Path,
        default=Path("plots"),
        help="Directory to save TDNV plot",
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
    layers = parse_layers_to_run(args.layers, max_layers)
    if not layers:
        raise ValueError("No valid layers selected")

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
    pos_features = compute_last_token_features(
        model,
        pos_texts,
        layers,
        d_model,
        device,
        dtype,
    )
    neg_features = compute_last_token_features(
        model,
        neg_texts,
        layers,
        d_model,
        device,
        dtype,
    )

    tdnv_stats = compute_tdnv_two_tasks(pos_features, neg_features)
    tdnv = tdnv_stats["tdnv"].detach().float().cpu().numpy()
    within_pos = tdnv_stats["within_var_a"].detach().float().cpu().numpy()
    within_neg = tdnv_stats["within_var_b"].detach().float().cpu().numpy()
    between = tdnv_stats["between_dist"].detach().float().cpu().numpy()

    model_name = get_model_name_for_path(args.model)
    stem = f"{args.dataset.stem}_{args.text_key}_{model_name}"

    result_dir = args.out_dir / model_name
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"tdnv_pku_safety_unsafety_{stem}.json"

    payload = {
        "dataset": str(args.dataset),
        "text_key": args.text_key,
        "model": args.model,
        "layers": layers,
        "seed": args.seed,
        "num_samples": {
            "safety": len(pos_texts),
            "unsafety": len(neg_texts),
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
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    args.plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = args.plot_dir / f"tdnv_pku_safety_unsafety_{stem}.pdf"

    plt.style.use("seaborn-v0_8-paper")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(layers, tdnv, color="#1f77b4", linewidth=2, label="TDNV")
    ax.set_title("TDNV for Safety vs Unsafety (Qwen3)")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("TDNV")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)

    logger.info("Saved results: {}", result_path)
    logger.info("Saved plot: {}", plot_path)
    logger.info("TDNV mean={:.6f}, std={:.6f}", float(tdnv.mean()), float(tdnv.std()))


if __name__ == "__main__":
    main()
