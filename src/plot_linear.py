import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.lines import Line2D

from utils import CONCEPT_CATEGORIES, MODEL_LAYERS, get_model_name_for_path
import matplotlib.cm as cm


def plot_linearity():
    # Configuration
    models = [
        "Qwen/Qwen3-1.7B",
        "google/gemma-2-2b",
    ]

    # Get concepts dynamically
    concepts = list(CONCEPT_CATEGORIES.keys())

    # Create readable names
    concept_renames = {}
    for c in concepts:
        # e.g., steering_change_case_english_capital -> Change Case: English Capital
        name = c.replace("steering_", "").replace("_", " ").title()
        # Handle specific cases for better formatting if needed
        name = name.replace("Change Case:", "Case:").replace(
            "Detectable Format:", "Format:"
        )
        concept_renames[c] = name

    # Style settings
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.titlesize": 22,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 14,
        }
    )

    # Define colors
    cmap = plt.get_cmap("tab20")
    concept_colors = {c: cmap(i / len(concepts)) for i, c in enumerate(concepts)}

    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)

    # Create N rows, 2 cols figure
    # Left: Variance (Mean + Std), Right: N95
    n_rows = len(models)
    fig, axes = plt.subplots(n_rows, 2, figsize=(20, 8 * n_rows))
    axes = np.atleast_2d(axes)

    def load_scores(model_name, concept, vector_type, is_remove):
        """
        Load mean_score, std_score, n_components_95_mean, and n_components_95_std.
        """
        suffix = "_remove" if is_remove else ""
        path = (
            f"assets/linear/{model_name}/linearity_{concept}_{vector_type}{suffix}.pt"
        )

        if not os.path.exists(path):
            return None, None, None, None, None

        try:
            data = torch.load(path)
            if "results" not in data:
                return None, None, None, None, None

            results = data["results"]
            layers = sorted(
                [
                    k
                    for k in results.keys()
                    if isinstance(k, (int, float, str)) and str(k).isdigit()
                ]
            )
            layers = [int(l) for l in layers]
            layers.sort()

            if not layers:
                return None, None, None, None, None

            means = []
            stds = []
            n95_means = []
            n95_stds = []

            for l in layers:
                val = results[l]
                if isinstance(val, dict):
                    means.append(val.get("mean_score", 0.0))
                    stds.append(val.get("std_score", 0.0))
                    n95_means.append(val.get("n_components_95_mean", 0.0))
                    n95_stds.append(val.get("n_components_95_std", 0.0))
                elif isinstance(val, (float, int)):
                    # Old format fallback
                    means.append(float(val))
                    stds.append(0.0)
                    n95_means.append(0.0)
                    n95_stds.append(0.0)
                else:
                    means.append(0.0)
                    stds.append(0.0)
                    n95_means.append(0.0)
                    n95_stds.append(0.0)

            return layers, means, stds, n95_means, n95_stds
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None, None, None, None, None

    # Plotting
    for row_idx, model_full in enumerate(models):
        model_name = get_model_name_for_path(model_full)
        max_layers = MODEL_LAYERS.get(model_full)
        if max_layers is None:
            print(f"Unknown model: {model_full}")
            continue

        ax_var = axes[row_idx, 0]
        ax_n95 = axes[row_idx, 1]

        has_data = False

        for concept in concepts:
            display_concept = concept_renames.get(concept, concept)
            color = concept_colors[concept]

            for vector_type in ["random", "concept"]:
                for is_remove in [False, True]:
                    # Define style based on type
                    if vector_type == "concept" and not is_remove:
                        # Standard Concept: Star, Solid
                        marker = "*"
                        linestyle = "-"
                        alpha = 0.9
                        hollow = False
                    elif vector_type == "random" and not is_remove:
                        # Standard Random: Circle, Dashed
                        marker = "o"
                        linestyle = "--"
                        alpha = 0.7
                        hollow = False
                    elif vector_type == "concept" and is_remove:
                        # Removed Concept: Star, Dash-Dot, Hollow
                        marker = "*"
                        linestyle = "-."
                        alpha = 0.6
                        hollow = True
                    elif vector_type == "random" and is_remove:
                        # Removed Random: Circle, Dotted, Hollow
                        marker = "o"
                        linestyle = ":"
                        alpha = 0.6
                        hollow = True

                    layers, means, stds, n95_means, n95_stds = load_scores(
                        model_name, concept, vector_type, is_remove
                    )

                    if layers:
                        x_axis = [l / (max_layers - 1) * 100 for l in layers]
                        has_data = True

                        # Common plot kwargs
                        kwargs = {
                            "color": color,
                            "marker": marker,
                            "linestyle": linestyle,
                            "linewidth": 2 if vector_type == "concept" else 1.5,
                            "markersize": 12 if vector_type == "concept" else 8,
                            "alpha": alpha,
                        }
                        if hollow:
                            kwargs["markerfacecolor"] = "white"
                            kwargs["markeredgewidth"] = 1.5

                        # Plot Variance (Left)
                        ax_var.plot(x_axis, means, **kwargs)
                        # Add shading for std
                        ax_var.fill_between(
                            x_axis,
                            np.array(means) - np.array(stds),
                            np.array(means) + np.array(stds),
                            color=color,
                            alpha=0.1,
                        )

                        # Plot N95 (Right)
                        ax_n95.plot(x_axis, n95_means, **kwargs)
                        # Add shading for std (N95)
                        ax_n95.fill_between(
                            x_axis,
                            np.array(n95_means) - np.array(n95_stds),
                            np.array(n95_means) + np.array(n95_stds),
                            color=color,
                            alpha=0.1,
                        )

        if not has_data:
            print(f"No data found for {model_name}.")

        # Axes labels and titles
        ax_var.set_xlabel("Layer Depth (%)", fontweight="bold")
        ax_var.set_ylabel("Linearity Score (Var Explained)", fontweight="bold")
        ax_var.set_title(f"{model_name} - Linearity", fontweight="bold")
        ax_var.grid(True, linestyle="--", alpha=0.7)
        ax_var.set_ylim(-0.05, 1.05)

        ax_n95.set_xlabel("Layer Depth (%)", fontweight="bold")
        ax_n95.set_ylabel("# Components (95% Var)", fontweight="bold")
        ax_n95.set_title(f"{model_name} - Complexity (N95)", fontweight="bold")
        ax_n95.grid(True, linestyle="--", alpha=0.7)
        ax_n95.set_ylim(bottom=0)

    # Legend
    legend_elements = []

    # 1. Concepts
    for c in sorted(concepts):
        display_c = concept_renames.get(c, c)
        legend_elements.append(
            Line2D([0], [0], color=concept_colors[c], lw=3, label=display_c)
        )

    # Spacer
    legend_elements.append(Line2D([0], [0], color="w", label=" ", alpha=0))

    # Styles
    legend_elements.append(
        Line2D(
            [0],
            [0],
            color="black",
            marker="*",
            linestyle="-",
            label="Concept (Full Hidden states)",
            markersize=12,
        )
    )
    legend_elements.append(
        Line2D(
            [0],
            [0],
            color="black",
            marker="o",
            linestyle="--",
            label="Random (Full Hidden states)",
            markersize=8,
        )
    )
    legend_elements.append(
        Line2D(
            [0],
            [0],
            color="black",
            marker="*",
            linestyle="-.",
            label="Concept (Mapping Only)",
            markersize=12,
            markerfacecolor="white",
            markeredgewidth=1.5,
        )
    )
    legend_elements.append(
        Line2D(
            [0],
            [0],
            color="black",
            marker="o",
            linestyle=":",
            label="Random (Mapping Only)",
            markersize=8,
            markerfacecolor="white",
            markeredgewidth=1.5,
        )
    )

    fig.subplots_adjust(bottom=0.35)

    leg = fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=5,
        frameon=True,
        framealpha=0.9,
        borderaxespad=0.5,
        handletextpad=0.5,
        columnspacing=1.0,
    )

    plt.setp(leg.get_texts(), fontweight="bold")

    save_path = os.path.join(output_dir, "linearity_multimodel_combined.pdf")
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Saved plot to {save_path}")
    plt.close()


if __name__ == "__main__":
    plot_linearity()
