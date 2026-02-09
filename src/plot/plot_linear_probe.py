# ruff: noqa: E402

import argparse
import glob
import json
import math
import os
from pathlib import Path
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
from loguru import logger

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils import get_model_name_for_path


def load_probe_results(input_dir: str) -> List[Dict]:
    paths = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    results = []
    for path in paths:
        with open(path, "r") as f:
            data = json.load(f)
        data["_path"] = path
        results.append(data)
    return results


def format_layer_label(layer_idx: int) -> str:
    return f"L{layer_idx}"


def plot_layer_grid(
    results: List[Dict],
    title: str,
    output_path: str,
    ncols: int = 3,
) -> None:
    if not results:
        logger.warning(f"No results to plot for {title}")
        return

    layer_set = set()
    for item in results:
        layer_set.update(item.get("probe_layers", []))
    layers = sorted(layer_set)
    if not layers:
        logger.warning(f"No probe layers found for {title}")
        return

    n = len(layers)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows), squeeze=False
    )
    colors = plt.cm.tab20

    for idx, layer_idx in enumerate(layers):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row][col]

        has_data = False
        for j, item in enumerate(results):
            vector_name = item.get("vector", f"vector_{j}")
            alpha_values = item.get("alpha_values", [])
            result_table = item.get("results", {})
            if not alpha_values or not result_table:
                continue

            accs = []
            for alpha in alpha_values:
                alpha_key = str(alpha)
                layer_stats = result_table.get(alpha_key, {}).get(str(layer_idx), {})
                accs.append(layer_stats.get("test_acc", float("nan")))

            alpha_percent = [float(alpha) / 100.0 for alpha in alpha_values]
            color = colors(j % colors.N)
            ax.plot(
                alpha_percent,
                accs,
                marker="o",
                markersize=3.5,
                linewidth=1.4,
                color=color,
                label=vector_name,
            )
            has_data = True

        if not has_data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.axis("off")
            continue

        ax.set_xscale("log")
        ax.set_ylim(0.0, 1.0)
        ax.axhline(0.5, color="#444444", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_title(format_layer_label(layer_idx), fontsize=11, fontweight="bold")
        ax.set_xlabel("Steering strength (%)")
        ax.set_ylabel("Probe accuracy")
        ax.tick_params(axis="both", labelsize=9)

    for idx in range(n, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row][col].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(len(labels), 6),
            frameon=False,
            bbox_to_anchor=(0.5, -0.02),
        )

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    fig.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot linear probe results")
    parser.add_argument(
        "--models",
        type=str,
        default="Qwen/Qwen3-1.7B,google/gemma-2-2b",
        help="Comma-separated model names",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots/linear_probe",
    )
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        logger.error("No models provided")
        return

    for model_full in models:
        model_name = get_model_name_for_path(model_full)
        input_dir = os.path.join("assets", "linear_probe", model_name)
        output_dir = os.path.join(args.output_dir, model_name)
        results = load_probe_results(input_dir)
        if not results:
            logger.error(f"No JSON results found in {input_dir}")
            continue

        plot_layer_grid(
            results,
            title=f"Linear Probe Accuracy ({model_name})",
            output_path=os.path.join(output_dir, "concept_random_combined.png"),
        )


if __name__ == "__main__":
    main()
