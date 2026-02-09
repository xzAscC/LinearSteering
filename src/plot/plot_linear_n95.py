# ruff: noqa: E402

import matplotlib.pyplot as plt
import os
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils import (
    apply_plot_style,
    build_plot_config,
    build_concept_colors,
    build_concept_renames,
    build_linearity_legend,
    configure_axis,
    layer_depth_percent,
    load_linearity_n95,
    vector_style,
)


def plot_linearity_n95():
    # Configuration
    # Order: Qwen (Left), Gemma (Right)
    models_order = [
        ("Qwen/Qwen3-1.7B", "Qwen3-1.7B"),
        ("google/gemma-2-2b", "gemma-2-2b"),
    ]
    models_config = {
        "Qwen/Qwen3-1.7B": {"max_layers": 28, "files_name": "Qwen3-1.7B"},
        "google/gemma-2-2b": {"max_layers": 26, "files_name": "gemma-2-2b"},
    }

    concepts = ["evil", "optimistic", "refusal", "sycophantic", "language_en_fr_paired"]
    concept_renames = build_concept_renames(
        concepts, replacements={"language_en_fr_paired": "translation"}
    )

    plot_config = build_plot_config(
        {
            "rc_params": {
                "font.size": 20,
                "axes.titlesize": 24,
                "axes.labelsize": 20,
                "xtick.labelsize": 18,
                "ytick.labelsize": 18,
                "legend.fontsize": 18,
            },
            "use_cycle": True,
            "figsize": (20, 8),
        }
    )
    apply_plot_style(plot_config["style"], plot_config["rc_params"])

    concept_colors = build_concept_colors(
        concepts,
        palette=plot_config["palette"],
        use_cycle=plot_config["use_cycle"],
    )

    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)

    # Create 1 row, 2 cols figure
    fig, axes = plt.subplots(1, 2, figsize=plot_config["figsize"])

    for idx, (model_full, model_name) in enumerate(models_order):
        ax = axes[idx]
        config = models_config.get(model_full)
        if not config:
            print(f"Config not found for {model_full}")
            continue

        max_layers = config["max_layers"]
        has_data = False

        for concept in concepts:
            color = concept_colors[concept]

            # 1. Standard (No Remove)
            # Random
            style = vector_style("random", False)
            layers, scores = load_linearity_n95(
                model_name, concept, "random", is_remove=False
            )
            if layers:
                x_axis = layer_depth_percent(layers, max_layers)
                ax.plot(
                    x_axis,
                    scores,
                    color=color,
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    markersize=style["markersize"],
                    alpha=style["alpha"],
                )
                has_data = True

            # Concept
            style = vector_style("concept", False)
            layers, scores = load_linearity_n95(
                model_name, concept, "concept", is_remove=False
            )
            if layers:
                x_axis = layer_depth_percent(layers, max_layers)
                ax.plot(
                    x_axis,
                    scores,
                    color=color,
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    markersize=style["markersize"],
                    alpha=style["alpha"],
                )
                has_data = True

            # 2. Remove (Concept Removed)
            # Random
            style = vector_style("random", True)
            layers, scores = load_linearity_n95(
                model_name, concept, "random", is_remove=True
            )
            if layers:
                x_axis = layer_depth_percent(layers, max_layers)
                ax.plot(
                    x_axis,
                    scores,
                    color=color,
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    markersize=style["markersize"],
                    alpha=style["alpha"],
                    markerfacecolor="white",
                    markeredgewidth=1.5,
                )
                has_data = True

            # Concept
            style = vector_style("concept", True)
            layers, scores = load_linearity_n95(
                model_name, concept, "concept", is_remove=True
            )
            if layers:
                x_axis = layer_depth_percent(layers, max_layers)
                ax.plot(
                    x_axis,
                    scores,
                    color=color,
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    markersize=style["markersize"],
                    alpha=style["alpha"],
                    markerfacecolor="white",
                    markeredgewidth=1.5,
                )
                has_data = True

        if not has_data:
            print(f"No data found for {model_name}.")

        if idx == 0:
            configure_axis(
                ax,
                xlabel="Layer Depth (%)",
                ylabel="# Components (95% Var)",
                title=f"{model_name}",
                ylim=(0, None),
            )
        else:
            configure_axis(
                ax,
                xlabel="Layer Depth (%)",
                ylabel="",
                title=f"{model_name}",
                ylim=(0, None),
            )

    legend_elements = build_linearity_legend(
        concepts,
        concept_colors,
        concept_renames,
        labels={
            "concept": "Concept (Standard)",
            "random": "Random (Standard)",
            "concept_removed": "Concept (Removed)",
            "random_removed": "Random (Removed)",
        },
    )

    fig.subplots_adjust(bottom=0.25)

    leg = fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=5,
        frameon=True,
        framealpha=0.9,
        borderaxespad=0.5,
        handletextpad=0.5,
        columnspacing=1.0,
    )

    plt.setp(leg.get_texts(), fontweight="bold")

    # Save combined plot
    save_path = os.path.join(output_dir, "linearity_n95_combined.pdf")
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Saved plot to {save_path}")
    plt.close()


if __name__ == "__main__":
    plot_linearity_n95()
