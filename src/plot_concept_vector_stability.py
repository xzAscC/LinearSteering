import argparse
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np

from utils import (
    CONCEPT_CATEGORIES,
    MODEL_LAYERS,
    apply_plot_style,
    build_plot_config,
    get_model_name_for_path,
)


def _extract_layer_series(
    results: dict,
) -> tuple[list[int], list[int], np.ndarray]:
    sizes = sorted(int(s) for s in results.get("token_sizes", results["example_sizes"]))
    layers = results["layers"]

    means = np.zeros((len(layers), len(sizes)), dtype=float)
    for size_idx, size in enumerate(sizes):
        metrics = results["metrics"][str(size)]["cosine_to_reference"]
        means[:, size_idx] = np.asarray(metrics["per_layer_mean"], dtype=float)

    return sizes, layers, means


def plot_model_layer_stability(
    results_by_concept: dict,
    model_name: str,
    concept_scope: str,
) -> None:
    concepts = list(results_by_concept.keys())
    if not concepts:
        return

    plot_config = build_plot_config(
        {
            "figsize": (16, 10),
            "rc_params": {"axes.titlesize": 12, "axes.labelsize": 10},
        }
    )
    apply_plot_style(plot_config["style"], plot_config["rc_params"])

    series_by_concept = {}
    layer_ids = None
    for concept_name in concepts:
        sizes, layers, means = _extract_layer_series(results_by_concept[concept_name])
        if layer_ids is None:
            layer_ids = layers
        elif layers != layer_ids:
            raise ValueError(
                f"Layer mismatch for concept {concept_name}. Expected {layer_ids}, got {layers}"
            )
        series_by_concept[concept_name] = {
            "sizes": sizes,
            "means": means,
        }

    total_layers = len(layer_ids)
    ncols = min(4, max(1, math.ceil(math.sqrt(total_layers))))
    nrows = math.ceil(total_layers / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.5 * ncols, 3.2 * nrows),
        squeeze=False,
    )

    for layer_pos, layer_idx in enumerate(layer_ids):
        row = layer_pos // ncols
        col = layer_pos % ncols
        ax = axes[row][col]

        for concept_name in concepts:
            series = series_by_concept[concept_name]
            sizes = series["sizes"]
            means = series["means"][layer_pos]

            ax.plot(
                sizes,
                means,
                marker="o",
                linewidth=1.8,
                markersize=4,
                label=concept_name,
            )

        ax.set_xscale("log")
        ax.set_title(f"Layer {layer_idx}")
        ax.grid(True, linestyle="--", alpha=0.6)

    for idx in range(total_layers, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row][col].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(4, len(labels)),
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.suptitle(
        f"Concept vector stability by layer: {model_name}", fontsize=14, y=1.02
    )
    fig.supxlabel("Number of tokens")
    fig.supylabel("Cosine similarity to reference")

    os.makedirs("plots", exist_ok=True)
    plot_path = os.path.join(
        "plots", f"stability_{model_name}_{concept_scope}_by_layer.png"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_results(model_name: str, concept_name: str) -> dict:
    result_path = os.path.join(
        "assets",
        "concept_vector_stability_cosine",
        model_name,
        concept_name,
        "cosine_similarity.json",
    )
    with open(result_path, "r") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot concept vector stability from saved cosine results"
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
    args = parser.parse_args()

    if args.model not in MODEL_LAYERS:
        raise ValueError(
            f"Invalid model: {args.model}. Must be one of {list(MODEL_LAYERS.keys())}"
        )
    if (
        args.concept_category != "all"
        and args.concept_category not in CONCEPT_CATEGORIES
    ):
        raise ValueError(
            "Invalid concept_category: "
            f"{args.concept_category}. Must be one of {list(CONCEPT_CATEGORIES.keys())}"
        )

    model_name = get_model_name_for_path(args.model)
    concept_categories = (
        list(CONCEPT_CATEGORIES.keys())
        if args.concept_category == "all"
        else [args.concept_category]
    )

    results_by_concept = {}
    for concept_category in concept_categories:
        results_by_concept[concept_category] = load_results(
            model_name, concept_category
        )

    concept_scope = (
        "all_concepts" if args.concept_category == "all" else args.concept_category
    )
    plot_model_layer_stability(
        results_by_concept,
        model_name=model_name,
        concept_scope=concept_scope,
    )


if __name__ == "__main__":
    main()
