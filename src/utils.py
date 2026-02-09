import hashlib
import json
import os
import random
from glob import glob
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    import datasets

MODEL_LAYERS = {
    "Qwen/Qwen3-1.7B": 28,
    "google/gemma-2-2b": 26,
}

CONCEPT_CATEGORIES = {
    f"steering_{subtype.replace(':', '_')}": {
        "base_path": "dataset/steering_tasks.jsonl",
        "dataset_key": "prompt",
        "loader_type": "jsonl",
        "sub_type": subtype,
    }
    for subtype in [
        "change_case:capital_word_frequency",
        "change_case:english_capital",
        "change_case:english_lowercase",
        "detectable_format:constrained_response",
        "detectable_format:json_format",
        "detectable_format:multiple_sections",
        "detectable_format:number_bullet_lists",
        "detectable_format:number_highlighted_sections",
        "detectable_format:title",
        "exclude",
        "include",
        "language:response_language",
        "punctuation:no_comma",
        "startend:end_checker",
        "startend:quotation",
    ]
}

CONCEPT_CATEGORIES["steering_safety"] = {
    "base_path": "dataset/safety_tasks.jsonl",
    "dataset_key": "prompt",
    "loader_type": "jsonl",
    "sub_type": "safety",
}


def set_seed(seed: int) -> None:
    """
    Set the seed for the random number generator
    Args:
        seed: the seed to use
    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return None


def seed_from_name(name: str) -> int:
    """
    Get a seed from a string
    Args:
        name: the string to get a seed from
    Returns:
        the seed
    """
    h = hashlib.md5(name.encode()).hexdigest()
    return int(h, 16) % (2**31)


def get_model_name_for_path(model_name: str) -> str:
    """
    Extract a safe model name for use in file paths.
    Handles model names with slashes (e.g., "google/gemma-2-2b" -> "gemma-2-2b").

    Args:
        model_name: Full model name (e.g., "google/gemma-2-2b")

    Returns:
        Safe name for file paths (e.g., "gemma-2-2b")
    """
    return model_name.split("/")[-1]


def parse_layers_to_run(
    layers_arg: str, max_layers: int, is_percentage: bool = True
) -> list[int]:
    """
    Parse the layers argument to determine which layers to run.

    Args:
        layers_arg: String containing layer specification. Can be:
            - "all": run all layers
            - Comma-separated percentages (0-100): e.g., "25,50,75"
            - Comma-separated layer indices: e.g., "5,10,15"
        max_layers: Total number of layers in the model

    Returns:
        List of layer indices to run

    Examples:
        >>> parse_layers_to_run("all", 26)
        [0, 1, 2, ..., 24]
        >>> parse_layers_to_run("25,50,75", 26)
        [6, 13, 19]
        >>> parse_layers_to_run("5,10,15", 26)
        [5, 10, 15]
    """
    if layers_arg.lower() == "all":
        return list(range(max_layers - 1))

    # Parse comma-separated values
    layer_values = [float(x.strip()) for x in layers_arg.split(",")]

    # Disambiguate between percentages and layer indices
    # If all values are <= 100, they could be either percentages or indices
    # Use heuristic: if all values are < max_layers, prefer layer indices
    if all(0 <= v <= 100 for v in layer_values):
        if not is_percentage:
            # All values are valid layer indices, treat as indices
            layers_to_run = [int(v) for v in layer_values]
        else:
            # At least one value >= max_layers, treat as percentages
            layers_to_run = [int(max_layers * (pct / 100.0)) for pct in layer_values]
    else:
        # At least one value > 100, treat as direct layer indices
        layers_to_run = [int(v) for v in layer_values]

    # Validate layer indices
    layers_to_run = [
        layer_idx for layer_idx in layers_to_run if 0 <= layer_idx < max_layers - 1
    ]

    return layers_to_run


def _load_separate_files_dataset(
    base_path: str, positive_file: str, negative_file: str
) -> Tuple["datasets.Dataset", "datasets.Dataset"]:
    """Load datasets from separate positive and negative files.

    Args:
        base_path: Base directory path containing the files
        positive_file: Name of the positive examples file
        negative_file: Name of the negative examples file

    Returns:
        Tuple of (positive_dataset, negative_dataset)
    """
    import datasets

    positive_dataset_path = os.path.join(base_path, positive_file)
    negative_dataset_path = os.path.join(base_path, negative_file)

    positive_dataset = datasets.load_dataset(
        "json", data_files=positive_dataset_path, split="train"
    )
    negative_dataset = datasets.load_dataset(
        "json", data_files=negative_dataset_path, split="train"
    )

    return positive_dataset, negative_dataset


def _load_single_file_with_pos_neg(
    file_path: str, instruction_key: str, dataset_key: str
) -> Tuple["datasets.Dataset", "datasets.Dataset"]:
    """Load datasets from a single file containing positive and negative examples.

    Args:
        file_path: Path to the JSON file
        instruction_key: Key in the JSON file containing the instruction array
        dataset_key: Key name to use when creating the dataset

    Returns:
        Tuple of (positive_dataset, negative_dataset)
    """
    import datasets

    dataset_file = datasets.load_dataset("json", data_files=file_path, split="train")

    instructions = dataset_file[0][instruction_key]
    positive_prompts = [item["pos"] for item in instructions]
    negative_prompts = [item["neg"] for item in instructions]

    positive_dataset = datasets.Dataset.from_dict({dataset_key: positive_prompts})
    negative_dataset = datasets.Dataset.from_dict({dataset_key: negative_prompts})

    return positive_dataset, negative_dataset


def _load_jsonl_dataset(
    file_path: str, dataset_key: str, filter_subtype: Optional[str] = None
) -> Tuple["datasets.Dataset", "datasets.Dataset"]:
    """Load datasets from a JSONL file with 'pos' and 'neg' keys per line.

    Args:
        file_path: Path to the JSONL file
        dataset_key: Key name to use when creating the dataset
        filter_subtype: Optional subtype to filter the dataset by

    Returns:
        Tuple of (positive_dataset, negative_dataset)
    """
    import datasets

    positive_prompts = []
    negative_prompts = []

    with open(file_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)

            if filter_subtype is not None and item.get("sub_type") != filter_subtype:
                continue

            if "pos" in item and "neg" in item:
                positive_prompts.append(item["pos"])
                negative_prompts.append(item["neg"])

    positive_dataset = datasets.Dataset.from_dict({dataset_key: positive_prompts})
    negative_dataset = datasets.Dataset.from_dict({dataset_key: negative_prompts})

    return positive_dataset, negative_dataset


def load_concept_datasets(
    concept_category_name: str, concept_category_config: Dict[str, Any]
) -> Tuple["datasets.Dataset", "datasets.Dataset", str]:
    """Load positive and negative datasets for a concept category.

    Args:
        concept_category_name: Name of the concept category
        concept_category_config: Configuration dictionary for the concept category

    Returns:
        Tuple of (positive_dataset, negative_dataset, dataset_key)
    """
    loader_type = concept_category_config["loader_type"]
    dataset_key = concept_category_config["dataset_key"]
    filter_subtype = concept_category_config.get("sub_type")

    if loader_type == "separate_files":
        base_path = concept_category_config["base_path"]
        positive_file = concept_category_config["positive_file"]
        negative_file = concept_category_config["negative_file"]
        positive_dataset, negative_dataset = _load_separate_files_dataset(
            base_path, positive_file, negative_file
        )
    elif loader_type == "single_file_with_pos_neg":
        file_path = concept_category_config["base_path"]
        instruction_key = concept_category_config["instruction_key"]
        positive_dataset, negative_dataset = _load_single_file_with_pos_neg(
            file_path, instruction_key, dataset_key
        )
    elif loader_type == "jsonl":
        file_path = concept_category_config["base_path"]
        positive_dataset, negative_dataset = _load_jsonl_dataset(
            file_path, dataset_key, filter_subtype=filter_subtype
        )
    else:
        raise ValueError(
            f"Unknown loader_type '{loader_type}' for concept category '{concept_category_name}'"
        )

    return positive_dataset, negative_dataset, dataset_key


def _get_layers_container(hf_model):
    """used to get the layers container of the model
    Args:
        hf_model (transformers.PreTrainedModel): the model to get the layers container
    Returns:
        layers (list): the layers container of the model
    """
    # Common containers across HF architectures
    candidates = [
        (hf_model, "gpt_neox", "layers"),
        (hf_model, "model", "layers"),
        (hf_model, "transformer", "layers"),
        (hf_model, "transformer", "h"),
    ]
    for root_obj, root_attr, layers_attr in candidates:
        root = getattr(root_obj, root_attr, None)
        if root is None:
            continue
        layers = getattr(root, layers_attr, None)
        if layers is not None:
            return layers
    raise AttributeError("Unable to locate transformer layers container on model")


def _extract_hidden(output: torch.Tensor | tuple) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def _replace_hidden(
    output: torch.Tensor | tuple, hidden: torch.Tensor
) -> torch.Tensor | tuple:
    if isinstance(output, tuple):
        return (hidden,) + output[1:]
    return hidden


def _apply_steering_output(
    output: torch.Tensor | tuple,
    steering_vector: torch.Tensor,
    alpha_value: float,
    device=None,
) -> torch.Tensor | tuple:
    if isinstance(output, tuple):
        hidden = output[0]
        target_device = device if device is not None else hidden.device
        vec = steering_vector.to(device=target_device, dtype=hidden.dtype)
        hidden = hidden + (alpha_value * vec)
        return (hidden,) + output[1:]
    target_device = device if device is not None else output.device
    vec = steering_vector.to(device=target_device, dtype=output.dtype)
    return output + (alpha_value * vec)


def make_steering_hook(steering_vector: torch.Tensor, alpha_value: float, device=None):
    def _hook(_module, _inputs, output):
        return _apply_steering_output(output, steering_vector, alpha_value, device)

    return _hook


def run_model_with_steering(
    model,
    input_ids,
    steering_vector: torch.Tensor,
    layer_idx: int,
    alpha_value: float,
    device,
) -> torch.Tensor:
    """
    Run model with steering applied at a specific layer and capture last layer hidden states.

    Args:
        model: The transformer model
        input_ids: Input token IDs
        steering_vector: The vector to add at the specified layer
        layer_idx: Which layer to apply steering at
        alpha_value: The scaling factor for steering (alpha * steering_vector)
        device: Device to run on

    Returns:
        Hidden states from the last layer [batch, seq, d]
    """
    layers_container = _get_layers_container(model)
    target_layer_module = layers_container[layer_idx]
    last_layer_module = layers_container[len(layers_container) - 1]

    captured: dict[str, torch.Tensor] = {}

    def _last_layer_forward_hook(_module, _inputs, output):
        captured["h"] = _extract_hidden(output).detach()
        return output

    last_handle = last_layer_module.register_forward_hook(_last_layer_forward_hook)
    steer_handle = target_layer_module.register_forward_hook(
        make_steering_hook(steering_vector, alpha_value, device=device)
    )
    try:
        _ = model(input_ids, output_hidden_states=True)
    finally:
        steer_handle.remove()
        last_handle.remove()

    h = captured.get("h", None)
    if h is None:
        raise RuntimeError("Failed to capture hidden states")
    return h


def run_model_with_steering_and_ablation(
    model,
    input_ids,
    steering_vector: torch.Tensor,
    layer_idx: int,
    alpha_value: float,
    device,
    remove_at_last_layer: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run model with steering applied at a specific layer, optionally removing it at the last layer.
    Returns both hidden states and logits for comparison.

    Args:
        model: The transformer model
        input_ids: Input token IDs
        steering_vector: The vector to add at the specified layer
        layer_idx: Which layer to apply steering at
        alpha_value: The scaling factor for steering (alpha * steering_vector)
        device: Device to run on
        remove_at_last_layer: If True, subtract steering vector at the last layer

    Returns:
        Tuple of (hidden_states, logits) from the last layer
        - hidden_states: [batch, seq, d]
        - logits: [batch, seq, vocab_size]
    """
    layers_container = _get_layers_container(model)
    target_layer_module = layers_container[layer_idx]
    last_layer_module = layers_container[len(layers_container) - 1]

    captured: dict[str, torch.Tensor] = {}

    def _last_layer_forward_hook(_module, _inputs, output):
        hidden = _extract_hidden(output)

        if remove_at_last_layer:
            vec = steering_vector.to(device=hidden.device, dtype=hidden.dtype)
            hidden = hidden - (alpha_value * vec)
            captured["h"] = hidden.detach()
            return _replace_hidden(output, hidden)

        captured["h"] = hidden.detach()
        return output

    last_handle = last_layer_module.register_forward_hook(_last_layer_forward_hook)
    steer_handle = target_layer_module.register_forward_hook(
        make_steering_hook(steering_vector, alpha_value, device=device)
    )

    try:
        outputs = model(input_ids, output_hidden_states=True)
    finally:
        steer_handle.remove()
        last_handle.remove()

    logits = outputs.logits

    h = captured.get("h", None)
    if h is None:
        raise RuntimeError("Failed to capture hidden states")

    return h, logits


def hidden_to_flat(h: torch.Tensor, target_dtype=torch.bfloat16) -> torch.Tensor:
    """
    Flatten hidden states from [batch, seq, d] to [batch*seq, d].

    Args:
        h: Hidden states tensor of shape [batch, seq, d]
        target_dtype: Target dtype for the output (default: bfloat16)

    Returns:
        Flattened tensor of shape [batch*seq, d]
    """
    hs_dim = h.shape[-1]
    return h.reshape(-1, hs_dim).to(target_dtype)


def _linearity_result_path(
    model_name: str, concept: str, vector_type: str, is_remove: bool
) -> str:
    suffix = "_remove" if is_remove else ""
    return f"assets/linear/{model_name}/linearity_{concept}_{vector_type}{suffix}.pt"


def _pca_hooks_result_path(
    model_name: str, concept: str, vector_type: str, is_remove: bool
) -> str:
    suffix = "_remove" if is_remove else ""
    return f"assets/linear/{model_name}/pca_hooks_{concept}_{vector_type}{suffix}.pt"


def _pca_hooks_result_candidates(
    model_name: str, concept: str, vector_type: str, is_remove: bool
) -> list[str]:
    suffix = "_remove" if is_remove else ""
    base_dir = f"assets/linear/{model_name}"

    tagged_pattern = f"{base_dir}/pca_hooks_{concept}_{vector_type}_cfg_*{suffix}.pt"
    tagged_paths = [path for path in glob(tagged_pattern) if os.path.isfile(path)]
    tagged_paths.sort(key=os.path.getmtime, reverse=True)

    legacy_path = _pca_hooks_result_path(model_name, concept, vector_type, is_remove)
    if os.path.isfile(legacy_path):
        tagged_paths.append(legacy_path)

    return tagged_paths


def _load_result_data(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        data = torch.load(path)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _extract_layer_stats(
    value: Any, preferred_hook_points: Optional[list[str]] = None
) -> Optional[Dict[str, Any]]:
    if not isinstance(value, dict):
        return None

    if "mean_score" in value or "n_components_95_mean" in value:
        return value

    hook_points = preferred_hook_points or []
    remaining_hook_points = sorted(k for k in value.keys() if isinstance(k, str))
    ordered_hook_points = hook_points + [
        hp for hp in remaining_hook_points if hp not in hook_points
    ]

    for hook_point in ordered_hook_points:
        hook_stats = value.get(hook_point)
        if isinstance(hook_stats, dict) and (
            "mean_score" in hook_stats or "n_components_95_mean" in hook_stats
        ):
            return hook_stats

    return None


def _load_compatible_linearity_results(
    model_name: str, concept: str, vector_type: str, is_remove: bool
) -> Optional[Dict[str, Any]]:
    old_data = _load_result_data(
        _linearity_result_path(model_name, concept, vector_type, is_remove)
    )
    if old_data and isinstance(old_data.get("results"), dict):
        return old_data["results"]

    hooks_data = None
    for hooks_path in _pca_hooks_result_candidates(
        model_name, concept, vector_type, is_remove
    ):
        candidate = _load_result_data(hooks_path)
        if candidate and isinstance(candidate.get("results"), dict):
            hooks_data = candidate
            break

    if not hooks_data:
        return None

    preferred_hook_points = hooks_data.get("hook_points")
    if not isinstance(preferred_hook_points, list):
        preferred_hook_points = None

    normalized: Dict[str, Any] = {}
    for layer_key, value in hooks_data["results"].items():
        layer_stats = _extract_layer_stats(value, preferred_hook_points)
        if layer_stats is not None:
            normalized[layer_key] = layer_stats

    return normalized if normalized else None


def _extract_linearity_layers(results: Dict[str, Any]) -> list[int]:
    layers = sorted(
        [
            k
            for k in results.keys()
            if isinstance(k, (int, float, str)) and str(k).isdigit()
        ]
    )
    layers = [int(l) for l in layers]
    layers.sort()
    return layers


def load_linearity_scores(
    model_name: str, concept: str, vector_type: str, is_remove: bool
) -> tuple[
    Optional[list[int]],
    Optional[list[float]],
    Optional[list[float]],
    Optional[list[float]],
    Optional[list[float]],
]:
    """Load linearity scores and n95 stats for plotting.

    Returns:
        layers, means, stds, n95_means, n95_stds
    """
    results = _load_compatible_linearity_results(
        model_name, concept, vector_type, is_remove
    )
    if not results:
        return None, None, None, None, None

    layers = _extract_linearity_layers(results)
    if not layers:
        return None, None, None, None, None

    means = []
    stds = []
    n95_means = []
    n95_stds = []

    for l in layers:
        val = results[l]
        if isinstance(val, dict):
            means.append(float(val.get("mean_score", 0.0)))
            stds.append(float(val.get("std_score", 0.0)))
            n95_means.append(float(val.get("n_components_95_mean", 0.0)))
            n95_stds.append(float(val.get("n_components_95_std", 0.0)))
        elif isinstance(val, (float, int)):
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


def load_linearity_n95(
    model_name: str, concept: str, vector_type: str, is_remove: bool
) -> tuple[Optional[list[int]], Optional[list[float]]]:
    """Load n95 scores for plotting."""
    results = _load_compatible_linearity_results(
        model_name, concept, vector_type, is_remove
    )
    if not results:
        return None, None

    layers = _extract_linearity_layers(results)
    if not layers:
        return None, None

    scores = []
    for l in layers:
        val = results[l]
        if isinstance(val, dict):
            if "n_components_95_mean" in val:
                scores.append(float(val.get("n_components_95_mean", 0.0)))
            else:
                scores.append(float(val.get("n_components_95", 0.0)))
        else:
            scores.append(0.0)

    return layers, scores


def apply_plot_style(style: str, rc_params: Dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    plt.style.use(style)
    plt.rcParams.update(rc_params)


def build_plot_config(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "style": "seaborn-v0_8-paper",
        "rc_params": {
            "font.size": 18,
            "axes.titlesize": 22,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 14,
        },
        "figsize": (20, 8),
        "palette": "tab20",
        "use_cycle": False,
    }

    if overrides:
        rc_overrides = overrides.get("rc_params")
        if rc_overrides:
            config["rc_params"] = {**config["rc_params"], **rc_overrides}
        for key, value in overrides.items():
            if key != "rc_params":
                config[key] = value

    return config


def build_concept_renames(
    concepts: list[str],
    replacements: Optional[Dict[str, str]] = None,
    strip_prefix: Optional[str] = None,
    title_case: bool = False,
) -> Dict[str, str]:
    renames: Dict[str, str] = {}
    for concept in concepts:
        name = concept
        if strip_prefix and name.startswith(strip_prefix):
            name = name[len(strip_prefix) :]
        if title_case:
            name = name.replace("_", " ").title()
        if replacements:
            for old, new in replacements.items():
                name = name.replace(old, new)
        renames[concept] = name
    return renames


def build_concept_colors(
    concepts: list[str], palette: str = "tab20", use_cycle: bool = False
) -> Dict[str, Any]:
    import matplotlib.pyplot as plt

    if use_cycle:
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors_list = prop_cycle.by_key()["color"]
        return {c: colors_list[i % len(colors_list)] for i, c in enumerate(concepts)}
    cmap = plt.get_cmap(palette)
    return {c: cmap(i / max(1, len(concepts))) for i, c in enumerate(concepts)}


def layer_depth_percent(layers: list[int], max_layers: int) -> list[float]:
    if max_layers <= 1:
        return [0.0 for _ in layers]
    return [l / (max_layers - 1) * 100 for l in layers]


def vector_style(vector_type: str, is_remove: bool) -> Dict[str, Any]:
    if vector_type == "concept" and not is_remove:
        return {
            "marker": "*",
            "linestyle": "-",
            "alpha": 0.9,
            "hollow": False,
            "linewidth": 2,
            "markersize": 12,
        }
    if vector_type == "random" and not is_remove:
        return {
            "marker": "o",
            "linestyle": "--",
            "alpha": 0.7,
            "hollow": False,
            "linewidth": 1.5,
            "markersize": 8,
        }
    if vector_type == "concept" and is_remove:
        return {
            "marker": "*",
            "linestyle": "-.",
            "alpha": 0.6,
            "hollow": True,
            "linewidth": 2,
            "markersize": 12,
        }
    return {
        "marker": "o",
        "linestyle": ":",
        "alpha": 0.6,
        "hollow": True,
        "linewidth": 1.5,
        "markersize": 8,
    }


def build_linearity_legend(
    concepts: list[str],
    concept_colors: Dict[str, Any],
    concept_renames: Dict[str, str],
    labels: Dict[str, str],
) -> list[Any]:
    from matplotlib.lines import Line2D

    legend_elements = []
    for c in sorted(concepts):
        display_c = concept_renames.get(c, c)
        legend_elements.append(
            Line2D([0], [0], color=concept_colors[c], lw=3, label=display_c)
        )

    legend_elements.append(Line2D([0], [0], color="w", label=" ", alpha=0))

    legend_elements.append(
        Line2D(
            [0],
            [0],
            color="black",
            marker="*",
            linestyle="-",
            label=labels["concept"],
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
            label=labels["random"],
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
            label=labels["concept_removed"],
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
            label=labels["random_removed"],
            markersize=8,
            markerfacecolor="white",
            markeredgewidth=1.5,
        )
    )

    return legend_elements


def configure_axis(
    ax,
    xlabel: str,
    ylabel: str,
    title: str,
    ylim: Optional[tuple[Optional[float], Optional[float]]] = None,
) -> None:
    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.7)
    if ylim is not None:
        ax.set_ylim(*ylim)
