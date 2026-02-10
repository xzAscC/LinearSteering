# ruff: noqa: E402

"""Plot PCA linearity results with one figure per model.

This script reads saved PCA linearity artifacts from ``assets/linear/<model_name>/``
and generates a separate figure for each model with:
- left panel: PC1 explained variance ratio (mean +/- std)
- right panel: n_95 (mean +/- std)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils import (
    MODEL_LAYERS,
    apply_plot_style,
    build_concept_colors,
    build_concept_renames,
    build_plot_config,
    configure_axis,
    get_model_name_for_path,
)


HOOK_POSITION_ORDER = [
    "input_ln",
    "attn",
    "post_attn_ln",
    "post_attn_proj_ln",
    "mlp",
    "post_mlp_ln",
    "block_out",
    "block_in",
    "attn_in",
    "attn_out",
    "mlp_in",
    "mlp_out",
    "block_out",
]


def _parse_pca_filename(file_name: str) -> tuple[str, str] | None:
    if not file_name.startswith("pca_hooks_") or not file_name.endswith(".pt"):
        return None

    core = file_name[len("pca_hooks_") : -3]
    if core.endswith("_remove"):
        core = core[: -len("_remove")]

    for token, vector_type in (("_concept_", "concept"), ("_random_", "random")):
        if token not in core:
            continue
        concept, _rest = core.rsplit(token, 1)
        if concept:
            return concept, vector_type

    return None


def _safe_to_full_model_map() -> dict[str, str]:
    return {get_model_name_for_path(full_name): full_name for full_name in MODEL_LAYERS}


def discover_models(assets_root: str, models_arg: str) -> list[str]:
    if models_arg.strip().lower() != "auto":
        return [name.strip() for name in models_arg.split(",") if name.strip()]

    linear_root = Path(assets_root)
    if not linear_root.exists():
        return []

    model_names: list[str] = []
    for child in sorted(linear_root.iterdir()):
        if not child.is_dir():
            continue
        if any(_parse_pca_filename(p.name) is not None for p in child.iterdir()):
            model_names.append(child.name)
    return model_names


def discover_concepts(model_dir: Path) -> list[str]:
    concepts: set[str] = set()
    for pt_file in model_dir.glob("pca_hooks_*.pt"):
        parsed = _parse_pca_filename(pt_file.name)
        if parsed is None:
            continue
        concept, _vector_type = parsed
        concepts.add(concept)
    return sorted(concepts)


def _parse_layer_index(layer_key: object) -> int | None:
    if isinstance(layer_key, int):
        return layer_key
    if isinstance(layer_key, str) and layer_key.isdigit():
        return int(layer_key)
    return None


def _flatten_run_series(
    data: dict,
) -> tuple[
    list[int],
    list[str],
    list[float],
    list[float],
    list[float],
    list[float],
    list[str],
]:
    results = data.get("results")
    if not isinstance(results, dict):
        return [], [], [], [], [], [], []

    preferred_hook_points = data.get("hook_points")
    if not isinstance(preferred_hook_points, list):
        preferred_hook_points = []

    observed_hooks: set[str] = set()
    parsed_layers: list[tuple[int, dict]] = []

    for layer_key, layer_value in results.items():
        layer_idx = _parse_layer_index(layer_key)
        if layer_idx is None or not isinstance(layer_value, dict):
            continue
        parsed_layers.append((layer_idx, layer_value))

        if "mean_score" in layer_value or "n_components_95_mean" in layer_value:
            observed_hooks.add("unknown")
            continue

        for hook_point in layer_value.keys():
            if isinstance(hook_point, str):
                observed_hooks.add(hook_point)

    ordered_hooks = [
        hook
        for hook in preferred_hook_points
        if isinstance(hook, str) and hook in observed_hooks
    ]
    for hook in HOOK_POSITION_ORDER:
        if hook in observed_hooks and hook not in ordered_hooks:
            ordered_hooks.append(hook)
    for hook in sorted(observed_hooks):
        if hook not in ordered_hooks:
            ordered_hooks.append(hook)

    parsed_layers.sort(key=lambda item: item[0])

    layers: list[int] = []
    hooks: list[str] = []
    means: list[float] = []
    stds: list[float] = []
    n95_means: list[float] = []
    n95_stds: list[float] = []

    for layer_idx, layer_value in parsed_layers:
        if "mean_score" in layer_value or "n_components_95_mean" in layer_value:
            layers.append(layer_idx)
            hooks.append("unknown")
            means.append(float(layer_value.get("mean_score", 0.0)))
            stds.append(float(layer_value.get("std_score", 0.0)))
            n95_means.append(float(layer_value.get("n_components_95_mean", 0.0)))
            n95_stds.append(float(layer_value.get("n_components_95_std", 0.0)))
            continue

        for hook_point in ordered_hooks:
            hook_stats = layer_value.get(hook_point)
            if not isinstance(hook_stats, dict):
                continue
            if (
                "mean_score" not in hook_stats
                and "n_components_95_mean" not in hook_stats
            ):
                continue
            layers.append(layer_idx)
            hooks.append(hook_point)
            means.append(float(hook_stats.get("mean_score", 0.0)))
            stds.append(float(hook_stats.get("std_score", 0.0)))
            n95_means.append(float(hook_stats.get("n_components_95_mean", 0.0)))
            n95_stds.append(float(hook_stats.get("n_components_95_std", 0.0)))

    return layers, hooks, means, stds, n95_means, n95_stds, ordered_hooks


def _layer_hook_positions(
    layers: list[int],
    hooks: list[str],
    ordered_hooks: list[str],
    layer_to_rank: dict[int, int],
    layer_span: float = 0.84,
) -> list[float]:
    slots_per_layer = max(1, len(ordered_hooks))
    hook_index = {hook: idx for idx, hook in enumerate(ordered_hooks)}

    if slots_per_layer == 1:
        offsets = [0.0]
    else:
        step = layer_span / (slots_per_layer - 1)
        offsets = [(-layer_span / 2.0) + idx * step for idx in range(slots_per_layer)]

    positions: list[float] = []
    for layer_idx, hook in zip(layers, hooks):
        layer_rank = layer_to_rank.get(layer_idx)
        if layer_rank is None:
            continue
        slot_idx = hook_index.get(hook, 0)
        positions.append(float(layer_rank) + offsets[min(slot_idx, len(offsets) - 1)])
    return positions


def _vector_marker(vector_type: str) -> dict[str, object]:
    if vector_type == "concept":
        return {
            "linestyle": "-",
            "linewidth": 2.0,
            "alpha": 0.9,
            "point_size": 42,
        }
    return {
        "linestyle": "-",
        "linewidth": 1.8,
        "alpha": 0.8,
        "point_size": 34,
    }


def _hook_marker(hook_point: str) -> str:
    marker_map = {
        "input_ln": "X",
        "attn": "*",
        "post_attn_ln": "^",
        "post_attn_proj_ln": "s",
        "mlp": "v",
        "post_mlp_ln": "P",
        "block_out": "D",
        "block_in": "x",
        "attn_in": "*",
        "attn_out": "^",
        "mlp_in": "v",
        "mlp_out": "s",
        "unknown": "X",
    }
    return marker_map.get(hook_point, "P")


def _pretty_hook_name(hook_point: str) -> str:
    if hook_point == "unknown":
        return "Unknown Hook"
    pretty_map = {
        "input_ln": "Input LN",
        "post_attn_ln": "Post-Attn LN",
        "post_attn_proj_ln": "Post-Attn Proj LN",
        "post_mlp_ln": "Post-MLP LN",
    }
    if hook_point in pretty_map:
        return pretty_map[hook_point]
    return hook_point.replace("_", " ").title()


def _sort_hook_points(hooks: set[str]) -> list[str]:
    order = {hook: idx for idx, hook in enumerate(HOOK_POSITION_ORDER)}
    return sorted(hooks, key=lambda h: (order.get(h, 999), h))


def _hook_point_size(hook_point: str) -> float:
    size_map = {
        "input_ln": 82,
        "attn": 96,
        "post_attn_ln": 80,
        "post_attn_proj_ln": 74,
        "mlp": 80,
        "post_mlp_ln": 74,
        "block_out": 76,
        "block_in": 64,
        "attn_in": 80,
        "attn_out": 78,
        "mlp_in": 62,
        "mlp_out": 66,
        "unknown": 64,
    }
    return size_map.get(hook_point, 64)


def load_latest_pca_runs(model_dir: Path) -> list[tuple[str, str, Path, dict]]:
    latest: dict[tuple[str, str, tuple[str, ...]], tuple[float, Path, dict]] = {}
    for pt_file in model_dir.glob("pca_hooks_*.pt"):
        parsed = _parse_pca_filename(pt_file.name)
        if parsed is None:
            continue
        concept, vector_type = parsed
        try:
            data = torch.load(pt_file)
        except Exception:
            continue
        if not isinstance(data, dict) or not isinstance(data.get("results"), dict):
            continue

        hook_points = data.get("hook_points")
        if isinstance(hook_points, list):
            hook_key = tuple(str(point) for point in hook_points)
        else:
            hook_key = tuple()

        key = (concept, vector_type, hook_key)
        mtime = pt_file.stat().st_mtime
        previous = latest.get(key)
        if previous is None or mtime > previous[0]:
            latest[key] = (mtime, pt_file, data)

    runs: list[tuple[str, str, Path, dict]] = []
    for _key, (_mtime, file_path, data) in sorted(
        latest.items(), key=lambda item: item[1][1].name
    ):
        concept, vector_type, _hook_key = _key
        runs.append((concept, vector_type, file_path, data))
    return runs


def _infer_max_layers(model_name: str) -> int | None:
    safe_to_full = _safe_to_full_model_map()
    full_name = safe_to_full.get(model_name)
    if full_name is None:
        return None
    return MODEL_LAYERS[full_name]


def plot_one_model(model_name: str, assets_root: str, output_dir: str) -> bool:
    model_dir = Path(assets_root) / model_name
    if not model_dir.exists():
        print(f"Skip {model_name}: {model_dir} does not exist")
        return False

    concepts = discover_concepts(model_dir)
    if not concepts:
        print(f"Skip {model_name}: no pca_hooks files found")
        return False

    concept_renames = build_concept_renames(
        concepts,
        strip_prefix="steering_",
        title_case=True,
        replacements={
            "Change Case:": "Case:",
            "Detectable Format:": "Format:",
        },
    )

    plot_config = build_plot_config(
        {
            "figsize": (20, 8),
            "rc_params": {
                "font.size": 16,
                "axes.titlesize": 20,
                "axes.labelsize": 16,
                "xtick.labelsize": 14,
                "ytick.labelsize": 14,
                "legend.fontsize": 12,
            },
        }
    )
    apply_plot_style(plot_config["style"], plot_config["rc_params"])
    concept_colors = build_concept_colors(
        concepts,
        palette=plot_config["palette"],
        use_cycle=plot_config["use_cycle"],
    )

    fig, (ax_var, ax_n95) = plt.subplots(1, 2, figsize=plot_config["figsize"])
    has_data = False

    all_hook_points: set[str] = set()
    all_vector_types: set[str] = set()
    plotted_concepts: set[str] = set()
    random_legend_color = "black"

    runs = load_latest_pca_runs(model_dir)
    prepared_runs: list[
        tuple[
            str,
            str,
            list[int],
            list[str],
            list[float],
            list[float],
            list[float],
            list[float],
            list[str],
        ]
    ] = []
    all_layers_seen: set[int] = set()

    for concept, vector_type, _file_path, run_data in runs:
        (
            layers,
            hooks,
            means,
            stds,
            n95_means,
            n95_stds,
            ordered_hooks,
        ) = _flatten_run_series(run_data)
        if not layers:
            continue
        prepared_runs.append(
            (
                concept,
                vector_type,
                layers,
                hooks,
                means,
                stds,
                n95_means,
                n95_stds,
                ordered_hooks,
            )
        )
        all_layers_seen.update(layers)

    if not prepared_runs:
        print(f"Skip {model_name}: could not load linearity points from saved files")
        plt.close(fig)
        return False

    unique_layers = sorted(all_layers_seen)
    layer_to_rank = {layer_idx: rank for rank, layer_idx in enumerate(unique_layers)}

    for (
        concept,
        vector_type,
        layers,
        hooks,
        means,
        stds,
        n95_means,
        n95_stds,
        ordered_hooks,
    ) in prepared_runs:
        if concept not in concept_colors:
            continue

        color = concept_colors[concept]
        if vector_type == "concept":
            plotted_concepts.add(concept)
        if vector_type == "random":
            random_legend_color = color
        marker_style = _vector_marker(vector_type)
        all_vector_types.add(vector_type)

        all_hook_points.update(hooks)

        x_axis = _layer_hook_positions(layers, hooks, ordered_hooks, layer_to_rank)
        if len(x_axis) != len(layers):
            continue
        has_data = True

        ax_var.plot(
            x_axis,
            means,
            color=color,
            linestyle=marker_style["linestyle"],
            linewidth=marker_style["linewidth"],
            alpha=marker_style["alpha"],
        )
        ax_var.fill_between(
            x_axis,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            color=color,
            alpha=0.08,
        )

        ax_n95.plot(
            x_axis,
            n95_means,
            color=color,
            linestyle=marker_style["linestyle"],
            linewidth=marker_style["linewidth"],
            alpha=marker_style["alpha"],
        )
        ax_n95.fill_between(
            x_axis,
            np.array(n95_means) - np.array(n95_stds),
            np.array(n95_means) + np.array(n95_stds),
            color=color,
            alpha=0.08,
        )

        for hook_point in set(hooks):
            hook_indices = [idx for idx, h in enumerate(hooks) if h == hook_point]
            if not hook_indices:
                continue

            var_x = [x_axis[idx] for idx in hook_indices]
            var_y = [means[idx] for idx in hook_indices]
            n95_y = [n95_means[idx] for idx in hook_indices]
            marker = _hook_marker(hook_point)
            point_size = _hook_point_size(hook_point)

            ax_var.scatter(
                var_x,
                var_y,
                marker=marker,
                s=point_size,
                color=color,
                alpha=1.0,
                linewidths=1.0,
                zorder=4,
            )
            ax_n95.scatter(
                var_x,
                n95_y,
                marker=marker,
                s=point_size,
                color=color,
                alpha=1.0,
                linewidths=1.0,
                zorder=4,
            )

    if not has_data:
        print(f"Skip {model_name}: could not load linearity points from saved files")
        plt.close(fig)
        return False

    configure_axis(
        ax_var,
        xlabel="Layer (Hook Order Within Layer)",
        ylabel="Linearity Score (Var Explained)",
        title=f"{model_name} - Linearity",
        ylim=(-0.05, 1.05),
    )
    configure_axis(
        ax_n95,
        xlabel="Layer (Hook Order Within Layer)",
        ylabel="# Components (95% Var)",
        title=f"{model_name} - Complexity (N95)",
        ylim=(0, None),
    )

    if unique_layers:
        layer_tick_positions = list(range(len(unique_layers)))
        layer_tick_labels = [str(layer_idx) for layer_idx in unique_layers]
        ax_var.set_xticks(layer_tick_positions)
        ax_var.set_xticklabels(layer_tick_labels)
        ax_n95.set_xticks(layer_tick_positions)
        ax_n95.set_xticklabels(layer_tick_labels)
        ax_var.set_xlim(-0.6, len(unique_layers) - 0.4)
        ax_n95.set_xlim(-0.6, len(unique_layers) - 0.4)

    legend_elements = []
    legend_concepts = sorted(plotted_concepts) if plotted_concepts else sorted(concepts)
    for concept in legend_concepts:
        if concept not in concept_colors:
            continue
        display_c = concept_renames.get(concept, concept)
        legend_elements.append(
            Line2D([0], [0], color=concept_colors[concept], lw=3, label=display_c)
        )

    legend_elements.append(Line2D([0], [0], color="w", label=" ", alpha=0))

    if "random" in all_vector_types:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color=random_legend_color,
                linestyle="-",
                label="Random Direction",
                linewidth=1.8,
            )
        )

    for hook_point in _sort_hook_points(all_hook_points):
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="None",
                marker=_hook_marker(hook_point),
                label=f"Hook: {_pretty_hook_name(hook_point)}",
                markersize=9,
                markeredgewidth=1.5,
            )
        )

    fig.subplots_adjust(bottom=0.34)
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        frameon=True,
        framealpha=0.9,
        borderaxespad=0.5,
        handletextpad=0.5,
        columnspacing=1.0,
    )

    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f"linearity_pca_{model_name}.pdf")
    png_path = os.path.join(output_dir, f"linearity_pca_{model_name}.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot PCA linearity results and save one figure per model."
    )
    parser.add_argument(
        "--assets_root",
        type=str,
        default="assets/linear",
        help="Directory containing per-model PCA linearity artifacts.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="auto",
        help=(
            "Comma-separated safe model names (e.g., Qwen3-1.7B,gemma-2-2b), "
            "or 'auto' to detect from assets_root."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots",
        help="Directory to save generated figures.",
    )
    args = parser.parse_args()

    models = discover_models(args.assets_root, args.models)
    if not models:
        print(f"No models found under {args.assets_root}")
        return

    success = 0
    for model_name in models:
        if plot_one_model(model_name, args.assets_root, args.output_dir):
            success += 1

    print(f"Done. Generated {success}/{len(models)} model figures.")


if __name__ == "__main__":
    main()
