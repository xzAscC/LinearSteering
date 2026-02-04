import matplotlib.pyplot as plt
import os
import numpy as np

from utils import (
    CONCEPT_CATEGORIES,
    MODEL_LAYERS,
    apply_plot_style,
    build_plot_config,
    build_concept_colors,
    build_concept_renames,
    build_linearity_legend,
    configure_axis,
    get_model_name_for_path,
    layer_depth_percent,
    load_linearity_scores,
    vector_style,
)


def plot_linearity():
    # Configuration
    models = [
        "Qwen/Qwen3-1.7B",
        "google/gemma-2-2b",
    ]

    # Get concepts dynamically
    concepts = list(CONCEPT_CATEGORIES.keys())

    concept_renames = build_concept_renames(
        concepts,
        strip_prefix="steering_",
        title_case=True,
        replacements={
            "Change Case:": "Case:",
            "Detectable Format:": "Format:",
        },
    )

    n_rows = len(models)
    plot_config = build_plot_config({"figsize": (20, 8 * n_rows)})
    apply_plot_style(plot_config["style"], plot_config["rc_params"])

    concept_colors = build_concept_colors(
        concepts,
        palette=plot_config["palette"],
        use_cycle=plot_config["use_cycle"],
    )

    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)

    # Create N rows, 2 cols figure
    # Left: Variance (Mean + Std), Right: N95
    fig, axes = plt.subplots(n_rows, 2, figsize=plot_config["figsize"])
    axes = np.atleast_2d(axes)

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
            color = concept_colors[concept]

            for vector_type in ["random", "concept"]:
                for is_remove in [False, True]:
                    style = vector_style(vector_type, is_remove)

                    layers, means, stds, n95_means, n95_stds = load_linearity_scores(
                        model_name, concept, vector_type, is_remove
                    )

                    if layers:
                        x_axis = layer_depth_percent(layers, max_layers)
                        has_data = True

                        # Common plot kwargs
                        kwargs = {
                            "color": color,
                            "marker": style["marker"],
                            "linestyle": style["linestyle"],
                            "linewidth": style["linewidth"],
                            "markersize": style["markersize"],
                            "alpha": style["alpha"],
                        }
                        if style["hollow"]:
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

        configure_axis(
            ax_var,
            xlabel="Layer Depth (%)",
            ylabel="Linearity Score (Var Explained)",
            title=f"{model_name} - Linearity",
            ylim=(-0.05, 1.05),
        )

        configure_axis(
            ax_n95,
            xlabel="Layer Depth (%)",
            ylabel="# Components (95% Var)",
            title=f"{model_name} - Complexity (N95)",
            ylim=(0, None),
        )

    legend_elements = build_linearity_legend(
        concepts,
        concept_colors,
        concept_renames,
        labels={
            "concept": "Concept (Full Hidden states)",
            "random": "Random (Full Hidden states)",
            "concept_removed": "Concept (Mapping Only)",
            "random_removed": "Random (Mapping Only)",
        },
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
