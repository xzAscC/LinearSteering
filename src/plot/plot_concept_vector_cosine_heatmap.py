# ruff: noqa: E402

import argparse
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils import apply_plot_style
from utils import build_plot_config


def load_vector_matrix(input_path: str) -> torch.Tensor:
    tensor = torch.load(input_path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"Expected a torch.Tensor in {input_path}, got {type(tensor)}")

    if tensor.ndim == 1:
        return tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(
            f"Expected tensor shape [N, D] or [D], got {tuple(tensor.shape)}"
        )
    return tensor


def compute_cosine_similarity_heatmap(vectors: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(vectors.float(), p=2, dim=1)
    return normalized @ normalized.T


def plot_and_save_heatmap(cos_sim: torch.Tensor, title: str, output_path: str) -> None:
    plot_config = build_plot_config(
        {
            "figsize": (10, 8),
            "rc_params": {
                "font.size": 10,
                "axes.titlesize": 12,
                "axes.labelsize": 10,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
            },
        }
    )
    apply_plot_style(plot_config["style"], plot_config["rc_params"])

    fig, ax = plt.subplots(figsize=plot_config["figsize"])
    image = ax.imshow(cos_sim.numpy(), cmap="coolwarm", vmin=-1.0, vmax=1.0)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Cosine similarity")

    n_vectors = cos_sim.shape[0]
    ax.set_title(title)
    ax.set_xlabel("Vector index")
    ax.set_ylabel("Vector index")
    ax.set_xticks(range(n_vectors))
    ax.set_yticks(range(n_vectors))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute pairwise cosine similarity heatmap for a concept vector tensor"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to .pt tensor file with shape [N, D]",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PDF path (default: plots/cosine_heatmap_<stem>.pdf)",
    )
    args = parser.parse_args()

    input_path = args.input
    stem = Path(input_path).stem
    output_path = args.output or os.path.join("plots", f"cosine_heatmap_{stem}.pdf")

    vectors = load_vector_matrix(input_path)
    cos_sim = compute_cosine_similarity_heatmap(vectors)
    plot_and_save_heatmap(
        cos_sim=cos_sim,
        title=f"Pairwise Cosine Similarity Heatmap ({stem})",
        output_path=output_path,
    )
    print(f"Saved heatmap to: {output_path}")


if __name__ == "__main__":
    main()
