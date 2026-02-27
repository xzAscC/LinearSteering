# ruff: noqa: E402

"""Visualize PCA of hidden states for PKU prompt pairs.

This script selects the first N pos/neg prompt pairs from the PKU SaferLHF
minimal dataset, extracts per-prompt hidden states, and saves a PCA scatter
plot for each layer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch
import transformer_lens
from loguru import logger

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

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


def extract_prompt_features(
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


def compute_pca_2d(features: torch.Tensor) -> torch.Tensor:
    centered = features - features.mean(dim=0, keepdim=True)
    centered = centered.to(dtype=torch.float32, device="cpu")
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    components = vh[:2].T
    return centered @ components


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot PCA scatter per layer for PKU prompt pairs."
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
        help="Which field to use for PCA",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help=f"Model name. Choices: {list(MODEL_LAYERS.keys())}",
    )
    parser.add_argument(
        "--pairs",
        type=int,
        default=100,
        help="Number of pos/neg prompt pairs to visualize",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Comma-separated layer indices, or 'all'",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Directory to save per-layer PCA PDFs",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_layers(layers_arg: str, max_layers: int) -> list[int]:
    if layers_arg.strip().lower() == "all":
        return list(range(max_layers))
    layers = [int(x.strip()) for x in layers_arg.split(",") if x.strip()]
    return [layer for layer in layers if 0 <= layer < max_layers]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.model not in MODEL_LAYERS:
        raise ValueError(
            f"Invalid model: {args.model}. Must be one of {list(MODEL_LAYERS.keys())}"
        )
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")
    if args.pairs <= 0:
        raise ValueError("pairs must be positive")

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
    layers = resolve_layers(args.layers, max_layers)
    if not layers:
        raise ValueError("No valid layers selected")

    pos_texts, neg_texts = load_pos_neg_texts(args.dataset, args.text_key)
    available_pairs = min(len(pos_texts), len(neg_texts))
    if available_pairs == 0:
        raise ValueError("No valid pos/neg examples found in dataset")
    pair_count = min(args.pairs, available_pairs)

    pos_texts = pos_texts[:pair_count]
    neg_texts = neg_texts[:pair_count]

    logger.info(
        "Loaded dataset pos={} neg={}, using first {} pairs",
        len(pos_texts),
        len(neg_texts),
        pair_count,
    )

    d_model = model.cfg.d_model
    pos_features = extract_prompt_features(
        model,
        pos_texts,
        layers,
        d_model,
        device,
        dtype,
    )
    neg_features = extract_prompt_features(
        model,
        neg_texts,
        layers,
        d_model,
        device,
        dtype,
    )

    model_tag = get_model_name_for_path(args.model)
    out_dir = (
        args.out_dir
        if args.out_dir is not None
        else Path("plots") / f"pku_prompt_pair_pca_{model_tag}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_config = build_plot_config(
        {
            "figsize": (7, 6),
            "rc_params": {"axes.titlesize": 14, "axes.labelsize": 12},
        }
    )
    apply_plot_style(plot_config["style"], plot_config["rc_params"])

    for layer_pos, layer_idx in enumerate(layers):
        layer_pos_features = pos_features[:, layer_pos, :]
        layer_neg_features = neg_features[:, layer_pos, :]
        all_features = torch.cat([layer_pos_features, layer_neg_features], dim=0)
        projected = compute_pca_2d(all_features)

        fig, ax = plt.subplots(figsize=plot_config["figsize"])
        ax.scatter(
            projected[:pair_count, 0],
            projected[:pair_count, 1],
            label="pos",
            color="#1f77b4",
            alpha=0.8,
            edgecolors="white",
            linewidths=0.6,
        )
        ax.scatter(
            projected[pair_count:, 0],
            projected[pair_count:, 1],
            label="neg",
            color="#ff7f0e",
            alpha=0.8,
            edgecolors="white",
            linewidths=0.6,
        )

        ax.set_title(
            f"PCA of Hidden States (Layer {layer_idx})\n"
            f"{model_tag} | {args.text_key} | {pair_count} pairs"
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(frameon=False)
        ax.grid(True, linestyle="--", alpha=0.5)

        out_path = out_dir / f"layer_{layer_idx:02d}.pdf"
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    logger.info("Saved PCA plots to {}", out_dir)


if __name__ == "__main__":
    main()
