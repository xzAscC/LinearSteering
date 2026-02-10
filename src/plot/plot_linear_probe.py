# ruff: noqa: E402

import argparse
import glob
import json
import math
import os
from pathlib import Path
import sys
from typing import Dict
from typing import List

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from loguru import logger

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils import apply_plot_style
from utils import build_concept_colors
from utils import build_concept_renames
from utils import build_plot_config
from utils import get_model_name_for_path


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
]


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
    alpha_mode = item.get("alpha_mode")
    results = item.get("results")
    if isinstance(results, dict) and _result_table_has_stats(results):
        return {
            "results": results,
            "alpha_values": item.get("alpha_values", []),
            "legacy_alpha_values_percent": item.get("alpha_values_percent", []),
            "probe_layers": item.get("probe_layers", []),
            "alpha_mode": alpha_mode,
        }

    top_results_by_steer = item.get("results_by_steer_layer", {})
    if isinstance(top_results_by_steer, dict) and top_results_by_steer:
        merged_results = _merge_results_by_steer_layer(top_results_by_steer)
        if merged_results:
            alpha_keys = _sorted_alpha_keys(merged_results)
            return {
                "results": merged_results,
                "alpha_values": [float(alpha_key) for alpha_key in alpha_keys],
                "legacy_alpha_values_percent": item.get("alpha_values_percent", []),
                "probe_layers": _collect_probe_layers_by_steer(top_results_by_steer),
                "alpha_mode": alpha_mode,
            }

    hooks = item.get("hooks", {})
    if not isinstance(hooks, dict) or not hooks:
        return {
            "results": {},
            "alpha_values": [],
            "legacy_alpha_values_percent": [],
            "probe_layers": [],
            "alpha_mode": alpha_mode,
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
                "legacy_alpha_values_percent": preferred_payload.get(
                    "alpha_values_percent", []
                ),
                "probe_layers": item.get("probe_layers", []),
                "alpha_mode": alpha_mode,
            }

        preferred_results_by_steer = preferred_payload.get("results_by_steer_layer", {})
        if isinstance(preferred_results_by_steer, dict) and preferred_results_by_steer:
            merged_results = _merge_results_by_steer_layer(preferred_results_by_steer)
            if merged_results:
                alpha_keys = _sorted_alpha_keys(merged_results)
                return {
                    "results": merged_results,
                    "alpha_values": [float(alpha_key) for alpha_key in alpha_keys],
                    "legacy_alpha_values_percent": preferred_payload.get(
                        "alpha_values_percent", []
                    ),
                    "probe_layers": _collect_probe_layers_by_steer(
                        preferred_results_by_steer
                    ),
                    "alpha_mode": alpha_mode,
                }

    for hook_payload in hooks.values():
        if not isinstance(hook_payload, dict):
            continue
        hook_results = hook_payload.get("results")
        if isinstance(hook_results, dict) and _result_table_has_stats(hook_results):
            return {
                "results": hook_results,
                "alpha_values": hook_payload.get("alpha_values", []),
                "legacy_alpha_values_percent": hook_payload.get(
                    "alpha_values_percent", []
                ),
                "probe_layers": item.get("probe_layers", []),
                "alpha_mode": alpha_mode,
            }

        hook_results_by_steer = hook_payload.get("results_by_steer_layer", {})
        if isinstance(hook_results_by_steer, dict) and hook_results_by_steer:
            merged_results = _merge_results_by_steer_layer(hook_results_by_steer)
            if merged_results:
                alpha_keys = _sorted_alpha_keys(merged_results)
                return {
                    "results": merged_results,
                    "alpha_values": [float(alpha_key) for alpha_key in alpha_keys],
                    "legacy_alpha_values_percent": hook_payload.get(
                        "alpha_values_percent", []
                    ),
                    "probe_layers": _collect_probe_layers_by_steer(
                        hook_results_by_steer
                    ),
                    "alpha_mode": alpha_mode,
                }

    return {
        "results": {},
        "alpha_values": [],
        "legacy_alpha_values_percent": [],
        "probe_layers": [],
        "alpha_mode": alpha_mode,
    }


def _resolve_alpha_axis(alpha_keys: List[str], result_payload: Dict) -> List[float]:
    alpha_values = result_payload.get("alpha_values", [])
    if isinstance(alpha_values, list) and len(alpha_values) == len(alpha_keys):
        return [float(alpha) for alpha in alpha_values]

    alpha_mode = str(result_payload.get("alpha_mode", "")).strip().lower()
    legacy_alpha_values_percent = result_payload.get("legacy_alpha_values_percent", [])
    if (
        alpha_mode == "avg_norm"
        and isinstance(legacy_alpha_values_percent, list)
        and len(legacy_alpha_values_percent) == len(alpha_keys)
    ):
        return [float(alpha) for alpha in legacy_alpha_values_percent]

    return [float(alpha_key) for alpha_key in alpha_keys]


def _resolve_alpha_axis_label(results: List[Dict]) -> str:
    modes = {
        str(item.get("alpha_mode", "")).strip().lower()
        for item in results
        if isinstance(item, dict)
    }
    if modes == {"avg_norm"}:
        return "Alpha scale (x avg norm)"
    if modes == {"manual"}:
        return "Alpha value"
    return "Alpha"


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
                metric_value = _extract_eval_metric(stats)
                if metric_value is None:
                    continue
                layer_name = str(layer_key)
                alpha_bucket.setdefault(layer_name, []).append(metric_value)

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


def _extract_eval_metric(stats: Dict) -> float | None:
    if not isinstance(stats, dict):
        return None
    if "test_acc" in stats:
        return float(stats["test_acc"])
    if "test_auc" in stats:
        return float(stats["test_auc"])
    return None


def _result_table_has_stats(result_table: Dict) -> bool:
    for per_layer in result_table.values():
        if not isinstance(per_layer, dict):
            continue
        for stats in per_layer.values():
            if _extract_eval_metric(stats) is not None:
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
            if _extract_eval_metric(stats) is not None:
                return True
    return False


def _iter_stats_payloads(item: Dict):
    hooks = item.get("hooks", {})
    if isinstance(hooks, dict) and hooks:
        for hook_payload in hooks.values():
            if not isinstance(hook_payload, dict):
                continue
            by_steer = hook_payload.get("results_by_steer_layer", {})
            if isinstance(by_steer, dict):
                for steer_payload in by_steer.values():
                    if not isinstance(steer_payload, dict):
                        continue
                    alpha_results = steer_payload.get("alpha_results", {})
                    if not isinstance(alpha_results, dict):
                        continue
                    for per_probe in alpha_results.values():
                        if not isinstance(per_probe, dict):
                            continue
                        for stats in per_probe.values():
                            if isinstance(stats, dict):
                                yield stats

            hook_results = hook_payload.get("results", {})
            if isinstance(hook_results, dict):
                for per_probe in hook_results.values():
                    if not isinstance(per_probe, dict):
                        continue
                    for stats in per_probe.values():
                        if isinstance(stats, dict):
                            yield stats

    top_by_steer = item.get("results_by_steer_layer", {})
    if isinstance(top_by_steer, dict):
        for steer_payload in top_by_steer.values():
            if not isinstance(steer_payload, dict):
                continue
            alpha_results = steer_payload.get("alpha_results", {})
            if not isinstance(alpha_results, dict):
                continue
            for per_probe in alpha_results.values():
                if not isinstance(per_probe, dict):
                    continue
                for stats in per_probe.values():
                    if isinstance(stats, dict):
                        yield stats


def _average_curve(curves: List[List[float]]) -> List[float]:
    if not curves:
        return []
    max_len = max(len(curve) for curve in curves)
    sums = [0.0] * max_len
    counts = [0] * max_len
    for curve in curves:
        for idx, value in enumerate(curve):
            sums[idx] += float(value)
            counts[idx] += 1
    return [
        (sums[idx] / counts[idx]) if counts[idx] > 0 else float("nan")
        for idx in range(max_len)
    ]


def plot_training_curves(results: List[Dict], title: str, output_path: str) -> None:
    if not results:
        logger.warning(f"No results to plot training curves for {title}")
        return

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
            "figsize": (9.0, 5.2),
            "rc_params": {
                "font.size": 12,
                "axes.titlesize": 12,
                "axes.labelsize": 11,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
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

    fig, ax = plt.subplots(figsize=plot_config["figsize"])
    has_data = False

    for idx, item in enumerate(results):
        vector_name = str(item.get("vector", f"vector_{idx}"))
        concept_name = str(item.get("concept", vector_name))
        concept_label = concept_renames.get(concept_name, vector_name)
        color = concept_colors.get(
            concept_name,
            plt.cm.tab20(idx % plt.cm.tab20.N),
        )

        curves: List[List[float]] = []
        for stats in _iter_stats_payloads(item):
            curve = stats.get("train_loss_curve")
            if not isinstance(curve, list) or not curve:
                continue
            curves.append([float(v) for v in curve])

        if not curves:
            continue

        mean_curve = _average_curve(curves)
        if not mean_curve:
            continue

        epochs = list(range(1, len(mean_curve) + 1))
        ax.plot(
            epochs,
            mean_curve,
            label=concept_label,
            color=color,
            linewidth=1.6,
            alpha=0.9,
        )
        has_data = True

    if not has_data:
        logger.warning(f"No train_loss_curve found for {title}")
        plt.close(fig)
        return

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss (BCE)")
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    fig.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


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
        "unknown": "o",
    }
    return marker_map.get(hook_point, "o")


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


def _sort_hook_points(hooks: set[str]) -> List[str]:
    order = {hook: idx for idx, hook in enumerate(HOOK_POSITION_ORDER)}
    return sorted(hooks, key=lambda h: (order.get(h, 999), h))


def _ordered_hooks_for_item(
    preferred_hook_points: List[str],
    observed_hooks: set[str],
) -> List[str]:
    ordered_hooks = [hook for hook in preferred_hook_points if hook in observed_hooks]
    for hook in HOOK_POSITION_ORDER:
        if hook in observed_hooks and hook not in ordered_hooks:
            ordered_hooks.append(hook)
    for hook in sorted(observed_hooks):
        if hook not in ordered_hooks:
            ordered_hooks.append(hook)
    return ordered_hooks


def _layer_hook_positions(
    layers: List[int],
    hooks: List[str],
    ordered_hooks: List[str],
    layer_to_rank: Dict[int, int],
    layer_span: float = 0.84,
) -> List[float]:
    slots_per_layer = max(1, len(ordered_hooks))
    hook_index = {hook: idx for idx, hook in enumerate(ordered_hooks)}

    if slots_per_layer == 1:
        offsets = [0.0]
    else:
        step = layer_span / (slots_per_layer - 1)
        offsets = [(-layer_span / 2.0) + idx * step for idx in range(slots_per_layer)]

    positions: List[float] = []
    for layer_idx, hook in zip(layers, hooks):
        layer_rank = layer_to_rank.get(layer_idx)
        if layer_rank is None:
            continue
        slot_idx = hook_index.get(hook, 0)
        positions.append(float(layer_rank) + offsets[min(slot_idx, len(offsets) - 1)])
    return positions


def _build_hook_payload(
    hook_payload: Dict,
    alpha_mode: str,
    default_probe_layers: List[int],
) -> Dict:
    if not isinstance(hook_payload, dict):
        return {
            "results": {},
            "alpha_values": [],
            "legacy_alpha_values_percent": [],
            "probe_layers": default_probe_layers,
            "alpha_mode": alpha_mode,
        }

    hook_results = hook_payload.get("results")
    if isinstance(hook_results, dict) and _result_table_has_stats(hook_results):
        return {
            "results": hook_results,
            "alpha_values": hook_payload.get("alpha_values", []),
            "legacy_alpha_values_percent": hook_payload.get("alpha_values_percent", []),
            "probe_layers": default_probe_layers,
            "alpha_mode": alpha_mode,
        }

    hook_results_by_steer = hook_payload.get("results_by_steer_layer", {})
    if isinstance(hook_results_by_steer, dict) and hook_results_by_steer:
        merged_results = _merge_results_by_steer_layer(hook_results_by_steer)
        if merged_results:
            alpha_keys = _sorted_alpha_keys(merged_results)
            return {
                "results": merged_results,
                "alpha_values": [float(alpha_key) for alpha_key in alpha_keys],
                "legacy_alpha_values_percent": hook_payload.get(
                    "alpha_values_percent", []
                ),
                "probe_layers": _collect_probe_layers_by_steer(hook_results_by_steer),
                "alpha_mode": alpha_mode,
            }

    return {
        "results": {},
        "alpha_values": [],
        "legacy_alpha_values_percent": [],
        "probe_layers": default_probe_layers,
        "alpha_mode": alpha_mode,
    }


def _get_hook_payloads(item: Dict) -> Dict[str, Dict]:
    alpha_mode = item.get("alpha_mode")
    hooks = item.get("hooks", {})
    default_probe_layers = item.get("probe_layers", [])
    payloads: Dict[str, Dict] = {}

    if isinstance(hooks, dict) and hooks:
        for hook_point, hook_payload in hooks.items():
            payloads[str(hook_point)] = _build_hook_payload(
                hook_payload,
                alpha_mode=alpha_mode,
                default_probe_layers=default_probe_layers,
            )

    if payloads:
        return payloads

    fallback_payload = _get_result_payload(item)
    return {"unknown": fallback_payload}


def _find_alpha_key(result_table: Dict, alpha_target: float) -> str | None:
    for alpha_key in result_table.keys():
        try:
            if math.isclose(
                float(alpha_key), alpha_target, rel_tol=1e-9, abs_tol=1e-12
            ):
                return str(alpha_key)
        except (TypeError, ValueError):
            continue
    return None


def _get_hook_steer_payloads(item: Dict) -> Dict[str, Dict[str, Dict]]:
    alpha_mode = item.get("alpha_mode")
    hooks = item.get("hooks", {})
    default_probe_layers = item.get("probe_layers", [])
    payloads: Dict[str, Dict[str, Dict]] = {}

    if isinstance(hooks, dict) and hooks:
        for hook_point, hook_payload in hooks.items():
            if not isinstance(hook_payload, dict):
                continue

            hook_point_name = str(hook_point)
            by_steer = hook_payload.get("results_by_steer_layer", {})
            if isinstance(by_steer, dict) and by_steer:
                steer_payloads: Dict[str, Dict] = {}
                for steer_layer, steer_payload in by_steer.items():
                    if not isinstance(steer_payload, dict):
                        continue
                    alpha_results = steer_payload.get("alpha_results", {})
                    if not isinstance(alpha_results, dict):
                        continue

                    alpha_keys = _sorted_alpha_keys(alpha_results)
                    steer_payloads[str(steer_layer)] = {
                        "results": alpha_results,
                        "alpha_values": [float(alpha_key) for alpha_key in alpha_keys],
                        "legacy_alpha_values_percent": steer_payload.get(
                            "alpha_values_percent", []
                        ),
                        "probe_layers": steer_payload.get(
                            "probe_layers", default_probe_layers
                        ),
                        "alpha_mode": alpha_mode,
                    }

                if steer_payloads:
                    payloads[hook_point_name] = steer_payloads
                    continue

            fallback_payload = _build_hook_payload(
                hook_payload,
                alpha_mode=alpha_mode,
                default_probe_layers=default_probe_layers,
            )
            if fallback_payload.get("results"):
                payloads[hook_point_name] = {"unknown": fallback_payload}

    if payloads:
        return payloads

    fallback_payload = _get_result_payload(item)
    return {"unknown": {"unknown": fallback_payload}}


def _sort_layer_keys(layer_keys: set[str]) -> List[str]:
    numeric = sorted(int(key) for key in layer_keys if key.isdigit())
    non_numeric = sorted(key for key in layer_keys if not key.isdigit())
    return [str(key) for key in numeric] + non_numeric


def plot_alpha_grid(
    results: List[Dict],
    title: str,
    output_path: str,
    ncols: int = 3,
) -> None:
    if not results:
        logger.warning(f"No results to plot for {title}")
        return

    steer_layer_set: set[str] = set()
    alpha_values_set: set[float] = set()
    probe_layers_by_steer: Dict[str, set[int]] = {}
    all_hook_points: set[str] = set()
    prepared_items = []

    for j, item in enumerate(results):
        vector_name = str(item.get("vector", f"vector_{j}"))
        concept_name = str(item.get("concept", vector_name))
        hook_steer_payloads = _get_hook_steer_payloads(item)
        preferred_hook_points = item.get("hook_points", [])
        if not isinstance(preferred_hook_points, list):
            preferred_hook_points = []
        prepared_items.append(
            (vector_name, concept_name, hook_steer_payloads, preferred_hook_points)
        )

        for hook_point, steer_payloads in hook_steer_payloads.items():
            all_hook_points.add(hook_point)
            for steer_layer, steer_payload in steer_payloads.items():
                steer_layer_set.add(steer_layer)
                probe_layers_by_steer.setdefault(steer_layer, set())

                configured_probe_layers = steer_payload.get("probe_layers", [])
                if isinstance(configured_probe_layers, list):
                    for layer in configured_probe_layers:
                        if str(layer).isdigit():
                            probe_layers_by_steer[steer_layer].add(int(layer))

                result_table = steer_payload.get("results", {})
                if not isinstance(result_table, dict):
                    continue
                alpha_keys = _sorted_alpha_keys(result_table)
                for alpha_key in alpha_keys:
                    try:
                        alpha_values_set.add(float(alpha_key))
                    except (TypeError, ValueError):
                        continue

                for per_probe in result_table.values():
                    if not isinstance(per_probe, dict):
                        continue
                    for probe_layer in per_probe.keys():
                        if str(probe_layer).isdigit():
                            probe_layers_by_steer[steer_layer].add(int(probe_layer))

    steer_layers = _sort_layer_keys(steer_layer_set)
    if not steer_layers:
        logger.warning(f"No steering layers found for {title}")
        return

    alpha_values = sorted(alpha_values_set)
    if not alpha_values:
        logger.warning(f"No alpha values found for {title}")
        return

    alpha_per_row = max(1, ncols)
    alpha_rows_per_steer = math.ceil(len(alpha_values) / alpha_per_row)
    nrows = len(steer_layers) * alpha_rows_per_steer
    ncols = alpha_per_row

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
            "figsize": (5.6 * ncols, 3.2 * nrows),
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
    hook_handles_by_label = {}
    alpha_label = _resolve_alpha_axis_label(results)

    for steer_idx, steer_layer in enumerate(steer_layers):
        probe_layers = sorted(probe_layers_by_steer.get(steer_layer, set()))
        if not probe_layers:
            continue

        for alpha_idx, alpha_value in enumerate(alpha_values):
            panel_idx = steer_idx * alpha_rows_per_steer * ncols + alpha_idx
            row = panel_idx // ncols
            col = panel_idx % ncols
            ax = axes[row][col]

            has_data = False
            layer_to_rank = {
                layer_idx: rank for rank, layer_idx in enumerate(sorted(probe_layers))
            }

            for j, (
                vector_name,
                concept_name,
                hook_steer_payloads,
                preferred_hook_points,
            ) in enumerate(prepared_items):
                concept_label = concept_renames.get(concept_name, vector_name)
                color = concept_colors.get(
                    concept_name,
                    plt.cm.tab20(j % plt.cm.tab20.N),
                )

                observed_hooks_for_line: set[str] = set()
                points: List[tuple[int, str, float]] = []
                for hook_point, steer_payloads in hook_steer_payloads.items():
                    steer_payload = steer_payloads.get(steer_layer)
                    if steer_payload is None:
                        continue

                    result_table = steer_payload.get("results", {})
                    if not isinstance(result_table, dict) or not result_table:
                        continue

                    alpha_key = _find_alpha_key(result_table, alpha_value)
                    if alpha_key is None:
                        continue

                    per_probe = result_table.get(alpha_key, {})
                    if not isinstance(per_probe, dict):
                        continue

                    for probe_layer in probe_layers:
                        layer_stats = per_probe.get(str(probe_layer), {})
                        metric_value = _extract_eval_metric(layer_stats)
                        if metric_value is None:
                            continue
                        points.append(
                            (
                                int(probe_layer),
                                str(hook_point),
                                float(metric_value),
                            )
                        )
                        observed_hooks_for_line.add(str(hook_point))

                if not points:
                    continue

                ordered_hooks = _ordered_hooks_for_item(
                    [str(hook) for hook in preferred_hook_points],
                    observed_hooks_for_line,
                )
                hook_rank = {hook: idx for idx, hook in enumerate(ordered_hooks)}
                points.sort(
                    key=lambda item: (item[0], hook_rank.get(item[1], 999), item[1])
                )

                point_layers = [item[0] for item in points]
                point_hooks = [item[1] for item in points]
                point_accs = [item[2] for item in points]
                x_positions = _layer_hook_positions(
                    point_layers,
                    point_hooks,
                    ordered_hooks,
                    layer_to_rank,
                )
                if len(x_positions) != len(point_accs):
                    continue

                line = ax.plot(
                    x_positions,
                    point_accs,
                    linewidth=1.3,
                    color=color,
                    alpha=0.9,
                    label=concept_label,
                )
                if concept_label not in handles_by_label:
                    handles_by_label[concept_label] = line[0]

                for hook_point in set(point_hooks):
                    marker = _hook_marker(hook_point)
                    hook_idx = [
                        idx for idx, h in enumerate(point_hooks) if h == hook_point
                    ]
                    if not hook_idx:
                        continue
                    ax.scatter(
                        [x_positions[idx] for idx in hook_idx],
                        [point_accs[idx] for idx in hook_idx],
                        marker=marker,
                        s=30,
                        color=color,
                        alpha=1.0,
                        linewidths=0.8,
                        zorder=4,
                    )

                    hook_label = f"Hook: {_pretty_hook_name(hook_point)}"
                    if hook_label not in hook_handles_by_label:
                        hook_handles_by_label[hook_label] = Line2D(
                            [0],
                            [0],
                            color="black",
                            linestyle="None",
                            marker=marker,
                            markersize=8,
                            markeredgewidth=1.2,
                            label=hook_label,
                        )

                has_data = True

            if not has_data:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.axis("off")
                continue

            ax.set_ylim(0.0, 1.0)
            ax.axhline(0.5, color="#444444", linestyle="--", linewidth=1, alpha=0.7)
            ax.set_title(
                f"Steer L{steer_layer} | {alpha_label}: {alpha_value:g}",
                fontsize=10,
                fontweight="bold",
            )
            ax.set_xlabel("Probe Layer (Hook Order Within Layer)")
            ax.set_ylabel("Probe score (Acc/AUC)")
            probe_ranks = list(range(len(probe_layers)))
            probe_labels = [str(layer_idx) for layer_idx in probe_layers]
            ax.set_xticks(probe_ranks)
            ax.set_xticklabels(probe_labels)
            ax.set_xlim(-0.6, len(probe_layers) - 0.4)
            ax.tick_params(axis="both", labelsize=9)

    total_panels = len(steer_layers) * len(alpha_values)
    for idx in range(total_panels, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row][col].axis("off")

    legend_handles = []
    legend_labels = []
    for label in sorted(handles_by_label.keys()):
        legend_handles.append(handles_by_label[label])
        legend_labels.append(label)
    for hook_point in _sort_hook_points(all_hook_points):
        hook_label = f"Hook: {_pretty_hook_name(hook_point)}"
        hook_handle = hook_handles_by_label.get(hook_label)
        if hook_handle is None:
            continue
        legend_handles.append(hook_handle)
        legend_labels.append(hook_label)

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=min(len(legend_labels), 5),
            frameon=False,
            bbox_to_anchor=(0.5, -0.02),
        )

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
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

        plot_alpha_grid(
            results,
            title=f"Linear Probe Accuracy by Alpha ({model_name})",
            output_path=os.path.join(output_dir, f"linear_probe_{model_name}.png"),
        )
        plot_training_curves(
            results,
            title=f"Linear Probe Training Curves ({model_name})",
            output_path=os.path.join(
                output_dir,
                f"linear_probe_training_curve_{model_name}.png",
            ),
        )


if __name__ == "__main__":
    main()
