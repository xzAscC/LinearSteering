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


def plot_stability_on_ax(results: dict, concept_name: str, ax) -> None:
    sizes = [int(s) for s in results.get("token_sizes", results["example_sizes"])]
    sizes = sorted(sizes)

    means = []
    stds = []
    for size in sizes:
        metrics = results["metrics"][str(size)]["cosine_to_reference"]
        means.append(metrics["mean"])
        stds.append(metrics["std"])

    ax.plot(sizes, means, marker="o", linewidth=2, color="#1f77b4")
    means_arr = np.asarray(means)
    stds_arr = np.asarray(stds)
    ax.fill_between(
        sizes,
        means_arr - stds_arr,
        means_arr + stds_arr,
        alpha=0.2,
        color="#1f77b4",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Number of tokens")
    ax.set_ylabel("Cosine similarity to reference")
    ax.set_title(f"Stability vs. token budget: {concept_name}")
    ax.grid(True, linestyle="--", alpha=0.6)

    reference_size = results.get("reference_size")
    if reference_size is not None and reference_size < 0:
        reference_tokens = results.get("reference_tokens", {}).get("total_tokens")
        if reference_tokens is not None:
            ax.text(
                0.02,
                0.98,
                f"ref=all ({reference_tokens} tokens)",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                color="#444444",
            )

    max_size = sizes[-1]
    max_mean = means[-1]
    ax.scatter([max_size], [max_mean], color="#d62728", zorder=3)
    ax.annotate(
        f"max={max_size}",
        xy=(max_size, max_mean),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=10,
        color="#d62728",
    )


def plot_stability(results: dict, model_name: str, concept_name: str) -> None:
    plot_config = build_plot_config(
        {
            "figsize": (10, 6),
            "rc_params": {"axes.titlesize": 16, "axes.labelsize": 14},
        }
    )
    apply_plot_style(plot_config["style"], plot_config["rc_params"])

    os.makedirs("plots", exist_ok=True)
    fig, ax = plt.subplots(figsize=plot_config["figsize"])
    plot_stability_on_ax(results, concept_name, ax)
    plot_path = os.path.join("plots", f"stability_{model_name}_{concept_name}.png")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


def plot_stability_grid(results_by_concept: dict, model_name: str) -> None:
    plot_config = build_plot_config(
        {
            "figsize": (12, 8),
            "rc_params": {"axes.titlesize": 14, "axes.labelsize": 12},
        }
    )
    apply_plot_style(plot_config["style"], plot_config["rc_params"])

    concepts = list(results_by_concept.keys())
    total = len(concepts)
    if total == 0:
        return

    ncols = 2 if total > 1 else 1
    nrows = math.ceil(total / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6 * ncols, 4 * nrows),
        squeeze=False,
    )

    for idx, concept_name in enumerate(concepts):
        row = idx // ncols
        col = idx % ncols
        plot_stability_on_ax(
            results_by_concept[concept_name], concept_name, axes[row][col]
        )

    for idx in range(total, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row][col].axis("off")

    fig.suptitle(f"Stability vs. token budget: {model_name}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs("plots", exist_ok=True)
    plot_path = os.path.join("plots", f"stability_{model_name}_all.png")
    fig.savefig(plot_path, dpi=200)
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

    for concept_category in concept_categories:
        plot_stability(
            results_by_concept[concept_category],
            model_name=model_name,
            concept_name=concept_category,
        )

    if len(concept_categories) > 1:
        plot_stability_grid(results_by_concept, model_name=model_name)


if __name__ == "__main__":
    main()
