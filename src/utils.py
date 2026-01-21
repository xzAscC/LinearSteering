import random
import torch
import hashlib
import numpy as np

MODEL_LAYERS = {
    "Qwen/Qwen3-1.7B": 28,
    # "Qwen/Qwen3-14B": 40,
    # "meta-llama/Llama-3.1-8B-Instruct": 32,
    # "google/gemma-2-2b": 26,
    # "allenai/Olmo-3-1025-7B": 32,
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


def parse_layers_to_run(layers_arg: str, max_layers: int, is_percentage: bool = True) -> list[int]:
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
            layers_to_run = [
                int(max_layers * (pct / 100.0)) for pct in layer_values
            ]
    else:
        # At least one value > 100, treat as direct layer indices
        layers_to_run = [int(v) for v in layer_values]
    
    # Validate layer indices
    layers_to_run = [
        layer_idx
        for layer_idx in layers_to_run
        if 0 <= layer_idx < max_layers - 1
    ]
    
    return layers_to_run


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
        hidden = output[0] if isinstance(output, tuple) else output
        captured["h"] = hidden.detach()
        return output

    def _steer_hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            vec = steering_vector.to(device=hidden.device, dtype=hidden.dtype)
            hidden = hidden + (alpha_value * vec)
            return (hidden,) + output[1:]
        vec = steering_vector.to(device=output.device, dtype=output.dtype)
        return output + (alpha_value * vec)

    last_handle = last_layer_module.register_forward_hook(_last_layer_forward_hook)
    steer_handle = target_layer_module.register_forward_hook(_steer_hook)
    _ = model(input_ids, output_hidden_states=True)
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
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        
        # Optionally remove steering vector at last layer
        if remove_at_last_layer:
            vec = steering_vector.to(device=hidden.device, dtype=hidden.dtype)
            hidden_ablated = hidden - (alpha_value * vec)
            captured["h"] = hidden_ablated.detach()
            # Return the ablated hidden states
            if isinstance(output, tuple):
                return (hidden_ablated,) + output[1:]
            else:
                return hidden_ablated
        else:
            captured["h"] = hidden.detach()
            return output

    def _steer_hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            vec = steering_vector.to(device=hidden.device, dtype=hidden.dtype)
            hidden = hidden + (alpha_value * vec)
            return (hidden,) + output[1:]
        vec = steering_vector.to(device=output.device, dtype=output.dtype)
        return output + (alpha_value * vec)

    last_handle = last_layer_module.register_forward_hook(_last_layer_forward_hook)
    steer_handle = target_layer_module.register_forward_hook(_steer_hook)
    
    # Run model and get logits
    outputs = model(input_ids, output_hidden_states=True)
    logits = outputs.logits
    
    steer_handle.remove()
    last_handle.remove()

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
