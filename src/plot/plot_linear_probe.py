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

from utils import apply_plot_style
from utils import build_concept_colors
from utils import build_concept_renames
from utils import build_plot_config
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


def discover_models(input_root: str, models_arg: str) -> List[str]:
    if models_arg.strip().lower() != "auto":
        return [m.strip() for m in models_arg.split(",") if m.strip()]

    root = Path(input_root)
    if not root.exists():
        return []

    models: List[str] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if any(child.glob("*.json")):
            models.append(child.name)
    return models


def _pick_latest_results(results: List[Dict]) -> List[Dict]:
    latest: Dict[str, Dict] = {}
    latest_with_data: Dict[str, Dict] = {}
    for item in results:
        vector_name = str(item.get("vector", "unknown"))
        _update_latest_entry(latest, vector_name, item)
        if _has_plottable_data(item):
            _update_latest_entry(latest_with_data, vector_name, item)

    selected = latest_with_data if latest_with_data else latest
    return [selected[key] for key in sorted(selected.keys())]


def _update_latest_entry(latest: Dict[str, Dict], key: str, item: Dict) -> None:
    existing = latest.get(key)
    if existing is None:
        latest[key] = item
        return

    current_path = Path(str(item.get("_path", "")))
    existing_path = Path(str(existing.get("_path", "")))
    current_time = current_path.stat().st_mtime if current_path.exists() else -1
    existing_time = existing_path.stat().st_mtime if existing_path.exists() else -1
    if current_time >= existing_time:
        latest[key] = item


def _get_result_payload(item: Dict) -> Dict:
    results = item.get("results")
    if isinstance(results, dict) and _result_table_has_stats(results):
        return {
            "results": results,
            "alpha_values": item.get("alpha_values", []),
            "alpha_values_percent": item.get("alpha_values_percent", []),
            "probe_layers": item.get("probe_layers", []),
        }

    top_results_by_steer = item.get("results_by_steer_layer", {})
    if isinstance(top_results_by_steer, dict) and top_results_by_steer:
        merged_results = _merge_results_by_steer_layer(top_results_by_steer)
        if merged_results:
            alpha_keys = _sorted_alpha_keys(merged_results)
            return {
                "results": merged_results,
                "alpha_values": [float(alpha_key) for alpha_key in alpha_keys],
                "alpha_values_percent": item.get("alpha_values_percent", []),
                "probe_layers": _collect_probe_layers_by_steer(top_results_by_steer),
            }

    hooks = item.get("hooks", {})
    if not isinstance(hooks, dict) or not hooks:
        return {
            "results": {},
            "alpha_values": [],
            "alpha_values_percent": [],
            "probe_layers": [],
        }

    preferred_hook = item.get("hook_point")
    if isinstance(preferred_hook, str):
        preferred_payload = hooks.get(preferred_hook, {})
        preferred_results = (
            preferred_payload.get("results")
            if isinstance(preferred_payload, dict)
            else None
        )
        if isinstance(preferred_results, dict) and _result_table_has_stats(
            preferred_results
        ):
            return {
                "results": preferred_results,
                "alpha_values": preferred_payload.get("alpha_values", []),
                "alpha_values_percent": preferred_payload.get(
                    "alpha_values_percent", []
                ),
                "probe_layers": item.get("probe_layers", []),
            }

        preferred_results_by_steer = preferred_payload.get("results_by_steer_layer", {})
        if isinstance(preferred_results_by_steer, dict) and preferred_results_by_steer:
            merged_results = _merge_results_by_steer_layer(preferred_results_by_steer)
            if merged_results:
                alpha_keys = _sorted_alpha_keys(merged_results)
                return {
                    "results": merged_results,
                    "alpha_values": [float(alpha_key) for alpha_key in alpha_keys],
                    "alpha_values_percent": preferred_payload.get(
                        "alpha_values_percent", []
                    ),
                    "probe_layers": _collect_probe_layers_by_steer(
                        preferred_results_by_steer
                    ),
                }

    for hook_payload in hooks.values():
        if not isinstance(hook_payload, dict):
            continue
        hook_results = hook_payload.get("results")
        if isinstance(hook_results, dict) and _result_table_has_stats(hook_results):
            return {
                "results": hook_results,
                "alpha_values": hook_payload.get("alpha_values", []),
                "alpha_values_percent": hook_payload.get("alpha_values_percent", []),
                "probe_layers": item.get("probe_layers", []),
            }

        hook_results_by_steer = hook_payload.get("results_by_steer_layer", {})
        if isinstance(hook_results_by_steer, dict) and hook_results_by_steer:
            merged_results = _merge_results_by_steer_layer(hook_results_by_steer)
            if merged_results:
                alpha_keys = _sorted_alpha_keys(merged_results)
                return {
                    "results": merged_results,
                    "alpha_values": [float(alpha_key) for alpha_key in alpha_keys],
                    "alpha_values_percent": hook_payload.get(
                        "alpha_values_percent", []
                    ),
                    "probe_layers": _collect_probe_layers_by_steer(
                        hook_results_by_steer
                    ),
                }

    return {
        "results": {},
        "alpha_values": [],
        "alpha_values_percent": [],
        "probe_layers": [],
    }


def _derive_probe_layers(result_payload: Dict) -> List[int]:
    configured_layers = result_payload.get("probe_layers", [])
    if isinstance(configured_layers, list) and configured_layers:
        return [int(layer) for layer in configured_layers]

    layers = set()
    table = result_payload.get("results", {})
    if not isinstance(table, dict):
        return []
    for alpha_payload in table.values():
        if not isinstance(alpha_payload, dict):
            continue
        for layer_key in alpha_payload:
            if str(layer_key).isdigit():
                layers.add(int(layer_key))
    return sorted(layers)


def _collect_probe_layers_by_steer(results_by_steer: Dict) -> List[int]:
    layers = set()
    for steer_payload in results_by_steer.values():
        if not isinstance(steer_payload, dict):
            continue
        probe_layers = steer_payload.get("probe_layers", [])
        if not isinstance(probe_layers, list):
            continue
        for layer in probe_layers:
            if str(layer).isdigit():
                layers.add(int(layer))
    return sorted(layers)


def _merge_results_by_steer_layer(results_by_steer: Dict) -> Dict[str, Dict]:
    accum: Dict[str, Dict[str, List[float]]] = {}
    train_sizes: Dict[str, Dict[str, List[float]]] = {}
    test_sizes: Dict[str, Dict[str, List[float]]] = {}

    for steer_payload in results_by_steer.values():
        if not isinstance(steer_payload, dict):
            continue
        alpha_results = steer_payload.get("alpha_results", {})
        if not isinstance(alpha_results, dict):
            continue

        for alpha_key, per_layer in alpha_results.items():
            if not isinstance(per_layer, dict):
                continue

            alpha_bucket = accum.setdefault(str(alpha_key), {})
            train_bucket = train_sizes.setdefault(str(alpha_key), {})
            test_bucket = test_sizes.setdefault(str(alpha_key), {})

            for layer_key, stats in per_layer.items():
                if not isinstance(stats, dict) or "test_acc" not in stats:
                    continue
                layer_name = str(layer_key)
                alpha_bucket.setdefault(layer_name, []).append(float(stats["test_acc"]))

                if "train_size" in stats:
                    train_bucket.setdefault(layer_name, []).append(
                        float(stats["train_size"])
                    )
                if "test_size" in stats:
                    test_bucket.setdefault(layer_name, []).append(
                        float(stats["test_size"])
                    )

    merged: Dict[str, Dict] = {}
    for alpha_key in _sorted_alpha_keys(accum):
        merged[alpha_key] = {}
        for layer_key, scores in accum[alpha_key].items():
            if not scores:
                continue

            layer_stats: Dict[str, float] = {
                "test_acc": float(sum(scores) / len(scores)),
            }

            layer_train_sizes = train_sizes.get(alpha_key, {}).get(layer_key, [])
            if layer_train_sizes:
                layer_stats["train_size"] = float(
                    sum(layer_train_sizes) / len(layer_train_sizes)
                )

            layer_test_sizes = test_sizes.get(alpha_key, {}).get(layer_key, [])
            if layer_test_sizes:
                layer_stats["test_size"] = float(
                    sum(layer_test_sizes) / len(layer_test_sizes)
                )

            merged[alpha_key][layer_key] = layer_stats

    return merged


def _sorted_alpha_keys(result_table: Dict) -> List[str]:
    sortable = []
    for key in result_table.keys():
        try:
            sortable.append((float(key), str(key)))
        except (TypeError, ValueError):
            continue
    sortable.sort(key=lambda item: item[0])
    return [key for _value, key in sortable]


def _result_table_has_stats(result_table: Dict) -> bool:
    for per_layer in result_table.values():
        if not isinstance(per_layer, dict):
            continue
        for stats in per_layer.values():
            if isinstance(stats, dict) and "test_acc" in stats:
                return True
    return False


def _has_plottable_data(item: Dict) -> bool:
    payload = _get_result_payload(item)
    result_table = payload.get("results", {})
    if not isinstance(result_table, dict) or not result_table:
        return False

    for alpha_key in _sorted_alpha_keys(result_table):
        per_layer = result_table.get(alpha_key, {})
        if not isinstance(per_layer, dict):
            continue
        for stats in per_layer.values():
            if isinstance(stats, dict) and "test_acc" in stats:
                return True
    return False


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
        result_payload = _get_result_payload(item)
        layer_set.update(_derive_probe_layers(result_payload))
    layers = sorted(layer_set)
    if not layers:
        logger.warning(f"No probe layers found for {title}")
        return

    n = len(layers)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)

    concepts = [
        str(item.get("concept", item.get("vector", "unknown"))) for item in results
    ]
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
            "style": "seaborn-v0_8-paper",
            "figsize": (5.2 * ncols, 3.8 * nrows),
            "rc_params": {
                "font.size": 12,
                "axes.titlesize": 12,
                "axes.labelsize": 11,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.fontsize": 10,
            },
        }
    )
    apply_plot_style(plot_config["style"], plot_config["rc_params"])
    concept_colors = build_concept_colors(
        concepts,
        palette=plot_config["palette"],
        use_cycle=plot_config["use_cycle"],
    )

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=plot_config["figsize"],
        squeeze=False,
    )

    handles_by_label = {}

    for idx, layer_idx in enumerate(layers):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row][col]

        has_data = False
        for j, item in enumerate(results):
            vector_name = str(item.get("vector", f"vector_{j}"))
            concept_name = str(item.get("concept", vector_name))
            result_payload = _get_result_payload(item)
            alpha_values = result_payload.get("alpha_values", [])
            alpha_values_percent = result_payload.get("alpha_values_percent", [])
            result_table = result_payload.get("results", {})
            if not result_table:
                continue

            alpha_keys = _sorted_alpha_keys(result_table)
            if not alpha_keys:
                continue

            accs = []
            for alpha_key in alpha_keys:
                layer_stats = result_table.get(alpha_key, {}).get(str(layer_idx), {})
                accs.append(layer_stats.get("test_acc", float("nan")))

            if all(math.isnan(value) for value in accs):
                continue

            if isinstance(alpha_values_percent, list) and len(
                alpha_values_percent
            ) == len(alpha_keys):
                alpha_axis = [float(alpha) for alpha in alpha_values_percent]
            elif isinstance(alpha_values, list) and len(alpha_values) == len(
                alpha_keys
            ):
                alpha_axis = [float(alpha) / 100.0 for alpha in alpha_values]
            else:
                alpha_axis = [float(alpha_key) / 100.0 for alpha_key in alpha_keys]

            color = concept_colors.get(concept_name, plt.cm.tab20(j % plt.cm.tab20.N))
            line = ax.plot(
                alpha_axis,
                accs,
                marker="o",
                markersize=3.5,
                linewidth=1.4,
                color=color,
                label=concept_renames.get(concept_name, vector_name),
            )
            label = concept_renames.get(concept_name, vector_name)
            if label not in handles_by_label:
                handles_by_label[label] = line[0]
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

    if handles_by_label:
        fig.legend(
            list(handles_by_label.values()),
            list(handles_by_label.keys()),
            loc="lower center",
            ncol=min(len(handles_by_label), 6),
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
        "--input_root",
        type=str,
        default="assets/linear_probe",
        help="Directory containing per-model linear probe JSON files",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="auto",
        help=(
            "Comma-separated model names (full or safe), or 'auto' to detect under "
            "input_root"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots/linear_probe",
    )
    args = parser.parse_args()

    models = discover_models(args.input_root, args.models)
    if not models:
        logger.error("No models provided")
        return

    for model_input in models:
        model_name = (
            get_model_name_for_path(model_input) if "/" in model_input else model_input
        )
        input_dir = os.path.join(args.input_root, model_name)
        if not os.path.isdir(input_dir) and "/" in model_input:
            input_dir = os.path.join(args.input_root, model_input)

        output_dir = os.path.join(args.output_dir, model_name)
        all_results = load_probe_results(input_dir)
        results = _pick_latest_results(all_results)
        if not results:
            logger.error(f"No JSON results found in {input_dir}")
            continue

        plot_layer_grid(
            results,
            title=f"Linear Probe Accuracy ({model_name})",
            output_path=os.path.join(output_dir, f"linear_probe_{model_name}.png"),
        )


if __name__ == "__main__":
    main()
