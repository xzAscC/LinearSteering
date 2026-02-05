import argparse
import glob
import os
from typing import Dict, List, Optional, Tuple

from loguru import logger
import matplotlib.pyplot as plt
import torch

from utils import get_model_name_for_path


def _weight_label_from_payload(payload: Dict, fallback: str) -> str:
    alpha = payload.get("alpha")
    alpha_percent = payload.get("alpha_percent")
    steer_layer = payload.get("steer_layer")
    probe_layer = payload.get("layer")
    parts = []
    if alpha_percent is not None:
        parts.append(f"{alpha_percent:g}%")
    elif alpha is not None:
        parts.append(f"{alpha:g}")
    if steer_layer is not None:
        parts.append(f"steer{steer_layer}")
    if probe_layer is not None:
        parts.append(f"layer{probe_layer}")
    if parts:
        return "_".join(parts)
    return fallback


def _parse_optional_layers(value: str) -> Optional[List[int]]:
    if not value or value.strip().lower() == "all":
        return None
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _load_weight_tensor(
    path: str,
    label_fallback: str,
) -> Tuple[Optional[str], Optional[torch.Tensor]]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        weight = payload.get("raw_weight", payload.get("weight"))
        label = _weight_label_from_payload(payload, label_fallback)
    else:
        weight = payload
        label = label_fallback
    if weight is None:
        return None, None
    return label, weight.flatten().float()


def load_probe_weights_for_steer_layer(
    steer_dir: str,
    probe_layer: int,
) -> Tuple[List[str], torch.Tensor]:
    labels: List[str] = []
    weights: List[torch.Tensor] = []

    alpha_dirs = sorted(
        [d for d in glob.glob(os.path.join(steer_dir, "alpha_*")) if os.path.isdir(d)]
    )
    if alpha_dirs:
        for alpha_dir in alpha_dirs:
            path = os.path.join(alpha_dir, f"layer_{probe_layer}.pt")
            if not os.path.exists(path):
                continue
            label, weight = _load_weight_tensor(path, os.path.basename(alpha_dir))
            if weight is None:
                logger.warning("Missing weight in {}", path)
                continue
            labels.append(label)
            weights.append(weight)
    else:
        path = os.path.join(steer_dir, f"layer_{probe_layer}.pt")
        if os.path.exists(path):
            label, weight = _load_weight_tensor(path, os.path.basename(path))
            if weight is not None:
                labels.append(label)
                weights.append(weight)

    if not weights:
        return [], torch.empty(0)

    return labels, torch.stack(weights, dim=0)


def compute_cosine_similarity(weights: torch.Tensor) -> torch.Tensor:
    if weights.numel() == 0:
        return weights
    norms = torch.linalg.vector_norm(weights, ord=2, dim=1, keepdim=True).clamp_min(
        1e-8
    )
    normalized = weights / norms
    return normalized @ normalized.T


def plot_cosine_heatmap(
    cosine: torch.Tensor,
    labels: List[str],
    title: str,
    output_path: str,
) -> None:
    if cosine.numel() == 0:
        logger.warning("No cosine similarities to plot for {}", title)
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    cax = ax.imshow(cosine.cpu().numpy(), vmin=-1.0, vmax=1.0, cmap="coolwarm")
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title, fontweight="bold")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Weight index")
    ax.set_ylabel("Weight index")

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    fig.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot cosine similarity heatmaps for linear probe weights"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="Model name used in assets path",
    )
    parser.add_argument(
        "--concepts",
        type=str,
        default="",
        help="Comma-separated concept names (defaults to all in directory)",
    )
    parser.add_argument(
        "--steer_layers",
        type=str,
        default="all",
        help="Comma-separated steer layers or 'all'",
    )
    parser.add_argument(
        "--probe_layers",
        type=str,
        default="all",
        help="Comma-separated probe layers or 'all'",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots/linear_probe_weights",
    )
    args = parser.parse_args()

    model_name = get_model_name_for_path(args.model)
    weight_root = os.path.join("assets", "linear_probe", model_name, "probe_weights")
    if not os.path.isdir(weight_root):
        raise FileNotFoundError(f"Probe weights directory not found: {weight_root}")

    if args.concepts:
        concepts = [c.strip() for c in args.concepts.split(",") if c.strip()]
    else:
        concepts = sorted(
            [
                name
                for name in os.listdir(weight_root)
                if os.path.isdir(os.path.join(weight_root, name))
            ]
        )

    if not concepts:
        logger.warning("No concept directories found in {}", weight_root)
        return

    steer_layers_filter = _parse_optional_layers(args.steer_layers)
    probe_layers_filter = _parse_optional_layers(args.probe_layers)

    for concept in concepts:
        concept_dir = os.path.join(weight_root, concept)
        steer_dirs = sorted(
            [
                d
                for d in glob.glob(os.path.join(concept_dir, "steer_*"))
                if os.path.isdir(d)
            ]
        )
        if not steer_dirs:
            logger.warning("No steer layer directories found for concept {}", concept)
            continue

        for steer_dir in steer_dirs:
            steer_name = os.path.basename(steer_dir)
            try:
                steer_layer = int(steer_name.replace("steer_", ""))
            except ValueError:
                logger.warning("Invalid steer layer directory: {}", steer_dir)
                continue
            if (
                steer_layers_filter is not None
                and steer_layer not in steer_layers_filter
            ):
                continue

            probe_layers = probe_layers_filter
            if probe_layers is None:
                probe_paths = sorted(
                    glob.glob(os.path.join(steer_dir, "alpha_*", "layer_*.pt"))
                )
                if not probe_paths:
                    probe_paths = sorted(
                        glob.glob(os.path.join(steer_dir, "layer_*.pt"))
                    )
                probe_layers = sorted(
                    {
                        int(
                            os.path.basename(p).replace("layer_", "").replace(".pt", "")
                        )
                        for p in probe_paths
                    }
                )

            if not probe_layers:
                logger.warning("No probe layers found for {} {}", concept, steer_name)
                continue

            for probe_layer in probe_layers:
                labels, weights = load_probe_weights_for_steer_layer(
                    steer_dir,
                    probe_layer,
                )
                if not labels:
                    logger.warning(
                        "No weights found for concept {} steer {} probe {}",
                        concept,
                        steer_layer,
                        probe_layer,
                    )
                    continue
                cosine = compute_cosine_similarity(weights)
                output_path = os.path.join(
                    args.output_dir,
                    model_name,
                    concept,
                    steer_name,
                    f"probe_{probe_layer}.png",
                )
                plot_cosine_heatmap(
                    cosine,
                    labels,
                    title=(
                        f"Probe weight cosine ({concept}) {steer_name} "
                        f"probe{probe_layer}"
                    ),
                    output_path=output_path,
                )


if __name__ == "__main__":
    main()
