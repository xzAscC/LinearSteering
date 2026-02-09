from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from loguru import logger
import torch


def weight_label_from_payload(payload: Dict, fallback: str) -> str:
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


def parse_optional_layers(value: str) -> Optional[List[int]]:
    if not value or value.strip().lower() == "all":
        return None
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def find_probe_layers(steer_dir: str) -> List[int]:
    probe_paths = sorted(glob.glob(os.path.join(steer_dir, "alpha_*", "layer_*.pt")))
    if not probe_paths:
        probe_paths = sorted(glob.glob(os.path.join(steer_dir, "layer_*.pt")))
    return sorted(
        {
            int(os.path.basename(p).replace("layer_", "").replace(".pt", ""))
            for p in probe_paths
        }
    )


def load_weight_tensor(
    path: str,
    label_fallback: str,
    prefer_raw_weight: bool = False,
) -> Tuple[Optional[str], Optional[torch.Tensor]]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        if prefer_raw_weight and payload.get("raw_weight") is not None:
            weight = payload.get("raw_weight")
        else:
            weight = payload.get("weight", payload.get("raw_weight"))
        label = weight_label_from_payload(payload, label_fallback)
    else:
        weight = payload
        label = label_fallback
    if weight is None:
        return None, None
    return label, weight.flatten().float()


def load_probe_weights_for_steer_layer(
    steer_dir: str,
    probe_layer: int,
    prefer_raw_weight: bool = False,
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
            label, weight = load_weight_tensor(
                path,
                os.path.basename(alpha_dir),
                prefer_raw_weight=prefer_raw_weight,
            )
            if weight is None:
                logger.warning("Missing weight in {}", path)
                continue
            labels.append(label)
            weights.append(weight)
    else:
        path = os.path.join(steer_dir, f"layer_{probe_layer}.pt")
        if os.path.exists(path):
            label, weight = load_weight_tensor(
                path,
                os.path.basename(path),
                prefer_raw_weight=prefer_raw_weight,
            )
            if weight is not None:
                labels.append(label)
                weights.append(weight)

    if not weights:
        return [], torch.empty(0)

    return labels, torch.stack(weights, dim=0)


def load_components_tensor(
    path: str,
    label_fallback: str,
) -> Tuple[Optional[str], Optional[torch.Tensor]]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        components = payload.get("components")
        label = weight_label_from_payload(payload, label_fallback)
    else:
        components = payload
        label = label_fallback
    if components is None:
        return None, None
    return label, components.float()


def load_components_for_steer_layer(
    steer_dir: str,
    probe_layer: int,
) -> Tuple[List[str], List[torch.Tensor]]:
    labels: List[str] = []
    components_list: List[torch.Tensor] = []

    alpha_dirs = sorted(
        [d for d in glob.glob(os.path.join(steer_dir, "alpha_*")) if os.path.isdir(d)]
    )
    if alpha_dirs:
        for alpha_dir in alpha_dirs:
            path = os.path.join(alpha_dir, f"layer_{probe_layer}.pt")
            if not os.path.exists(path):
                continue
            label, components = load_components_tensor(
                path,
                os.path.basename(alpha_dir),
            )
            if components is None:
                logger.warning("Missing components in {}", path)
                continue
            labels.append(label)
            components_list.append(components)
    else:
        path = os.path.join(steer_dir, f"layer_{probe_layer}.pt")
        if os.path.exists(path):
            label, components = load_components_tensor(
                path,
                os.path.basename(path),
            )
            if components is not None:
                labels.append(label)
                components_list.append(components)

    return labels, components_list


def compute_cosine_similarity(weights: torch.Tensor) -> torch.Tensor:
    if weights.numel() == 0:
        return weights
    norms = torch.linalg.vector_norm(weights, ord=2, dim=1, keepdim=True).clamp_min(
        1e-8
    )
    normalized = weights / norms
    return normalized @ normalized.T


def compute_max_cosine_matrix(components_list: List[torch.Tensor]) -> torch.Tensor:
    if not components_list:
        return torch.empty(0)
    n = len(components_list)
    max_cos = torch.zeros((n, n), dtype=torch.float32)
    for i in range(n):
        comp_i = components_list[i]
        comp_i = comp_i[: min(20, comp_i.shape[0])]
        norm_i = torch.linalg.vector_norm(comp_i, ord=2, dim=1, keepdim=True).clamp_min(
            1e-8
        )
        comp_i = comp_i / norm_i
        for j in range(n):
            comp_j = components_list[j]
            comp_j = comp_j[: min(10, comp_j.shape[0])]
            norm_j = torch.linalg.vector_norm(
                comp_j,
                ord=2,
                dim=1,
                keepdim=True,
            ).clamp_min(1e-8)
            comp_j = comp_j / norm_j
            cos = comp_i @ comp_j.T
            max_cos[i, j] = cos.max().item()
    return max_cos


def plot_cosine_heatmap(
    cosine: torch.Tensor,
    labels: List[str],
    title: str,
    output_path: str,
    xlabel: str = "Weight index",
    ylabel: str = "Weight index",
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
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    fig.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
