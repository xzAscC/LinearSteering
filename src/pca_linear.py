import argparse
import os
import torch
import transformers
from loguru import logger
from tqdm import tqdm

from utils import (
    CONCEPT_CATEGORIES,
    get_model_name_for_path,
    hidden_to_flat,
    load_concept_datasets,
    make_steering_hook,
    MODEL_LAYERS,
    parse_layers_to_run,
    set_seed,
)


class PCAStatisticsAggregator:
    """
    Aggregates PCA statistics (linearity score and n_95) across batches of tokens.
    Computes global mean and std from online updates of sum and sum_sq.
    """

    def __init__(self):
        self.score_sum = 0.0
        self.score_sq_sum = 0.0
        self.n95_sum = 0.0
        self.n95_sq_sum = 0.0
        self.count = 0

    def update(self, trajectory_data):
        """
        Compute stats for the current batch of trajectories and update aggregators.

        Args:
            trajectory_data (torch.Tensor): [num_steps, num_samples, hidden_dim]
        """
        # [Steps, N_tokens, Hidden] -> [N_tokens, Steps, Hidden]
        X = trajectory_data.permute(1, 0, 2).float()

        # Center each trajectory independently
        X_mean = X.mean(dim=1, keepdim=True)
        X_centered = X - X_mean

        # Compute SVD for each sample
        # S has shape [N, min(T, D)]
        S = torch.linalg.svdvals(X_centered)

        eigenvalues = S**2
        total_variance = eigenvalues.sum(dim=-1)

        # Avoid division by zero
        epsilon = 1e-12
        valid_mask = total_variance > epsilon

        # 1. Linearity Score (PC1 / Total)
        scores = torch.ones_like(total_variance)  # Default 1.0
        if valid_mask.any():
            scores[valid_mask] = (
                eigenvalues[valid_mask][:, 0] / total_variance[valid_mask]
            )

        # 2. n_95
        n_95 = torch.ones_like(total_variance)  # Default 1.0
        if valid_mask.any():
            valid_eigenvars = eigenvalues[valid_mask]
            valid_total = total_variance[valid_mask].unsqueeze(-1)
            ratios = valid_eigenvars / valid_total

            cumsum = torch.cumsum(ratios, dim=-1)
            # Count components needed to reach >= 0.95
            current_n95 = (cumsum < 0.95).sum(dim=-1) + 1
            n_95[valid_mask] = current_n95.float()

        # Update aggregators
        self.score_sum += scores.sum().item()
        self.score_sq_sum += (scores**2).sum().item()
        self.n95_sum += n_95.sum().item()
        self.n95_sq_sum += (n_95**2).sum().item()
        self.count += scores.shape[0]

    def finalize(self):
        if self.count == 0:
            return {
                "mean_score": 1.0,
                "std_score": 0.0,
                "mean_n95": 1.0,
                "std_n95": 0.0,
                "count": 0,
            }

        mean_score = self.score_sum / self.count
        var_score = max(0.0, self.score_sq_sum / self.count - mean_score**2)

        mean_n95 = self.n95_sum / self.count
        var_n95 = max(0.0, self.n95_sq_sum / self.count - mean_n95**2)

        return {
            "mean_score": mean_score,
            "std_score": var_score**0.5,
            "mean_n95": mean_n95,
            "std_n95": var_n95**0.5,
            "count": self.count,
        }


def _get_layers_container(hf_model):
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


def _extract_hidden(output):
    return output[0] if isinstance(output, tuple) else output


def _is_qwen3_style_layer(layer_module) -> bool:
    return all(
        hasattr(layer_module, name)
        for name in (
            "input_layernorm",
            "self_attn",
            "post_attention_layernorm",
            "mlp",
        )
    )


def _parse_hook_points(hook_points_arg: str) -> list[str]:
    alias_map = {
        "ln1": "input_ln",
        "ln2": "post_attn_ln",
        "block": "block_out",
        "resid": "block_out",
    }
    valid_points = {"input_ln", "attn", "post_attn_ln", "mlp", "block_out"}

    parsed_points: list[str] = []
    for raw_point in hook_points_arg.split(","):
        point = raw_point.strip().lower()
        if not point:
            continue
        canonical_point = alias_map.get(point, point)
        if canonical_point not in valid_points:
            raise ValueError(
                f"Unsupported hook point '{raw_point}'. "
                f"Valid points: {sorted(valid_points)}"
            )
        if canonical_point not in parsed_points:
            parsed_points.append(canonical_point)

    if not parsed_points:
        raise ValueError("No valid hook points were provided.")

    return parsed_points


def _parse_concepts_to_run(concepts_arg: str) -> list[tuple[str, dict]]:
    if concepts_arg.strip().lower() == "all":
        return list(CONCEPT_CATEGORIES.items())

    selected_concepts: list[str] = []
    for raw_concept in concepts_arg.split(","):
        concept_name = raw_concept.strip()
        if not concept_name:
            continue
        if concept_name not in CONCEPT_CATEGORIES:
            raise ValueError(
                f"Unknown concept '{concept_name}'. "
                f"Available: {sorted(CONCEPT_CATEGORIES.keys())}"
            )
        if concept_name not in selected_concepts:
            selected_concepts.append(concept_name)

    if not selected_concepts:
        raise ValueError("No valid concepts were provided.")

    return [(name, CONCEPT_CATEGORIES[name]) for name in selected_concepts]


def _resolve_hook_module(layer_module, hook_point: str):
    if _is_qwen3_style_layer(layer_module):
        mapping = {
            "input_ln": layer_module.input_layernorm,
            "attn": layer_module.self_attn,
            "post_attn_ln": layer_module.post_attention_layernorm,
            "mlp": layer_module.mlp,
            "block_out": layer_module,
        }
        return mapping[hook_point]

    if hook_point != "block_out":
        raise ValueError(
            f"Hook point '{hook_point}' not supported for layer type "
            f"'{layer_module.__class__.__name__}'. Use 'block_out'."
        )
    return layer_module


def _auto_select_layers(layer_count: int, max_layers: int) -> list[int]:
    """Select approximately uniform layer indices from depth percentages.

    Example: layer_count=8 -> 12.5%, 25%, ..., 100% mapped to valid indices.
    """
    max_valid_layer = max_layers - 2
    if max_valid_layer < 0:
        return []

    if layer_count <= 0:
        raise ValueError("auto layer count must be positive")

    total_available = max_valid_layer + 1
    if layer_count >= total_available:
        return list(range(total_available))

    percentages = [100.0 * i / layer_count for i in range(1, layer_count + 1)]
    selected_layers: list[int] = []
    for pct in percentages:
        layer_idx = int((pct / 100.0) * (max_valid_layer + 1))
        layer_idx = min(layer_idx, max_valid_layer)
        if layer_idx not in selected_layers:
            selected_layers.append(layer_idx)

    # Fill missing slots if integer rounding produced duplicates.
    if len(selected_layers) < layer_count:
        for layer_idx in range(total_available):
            if layer_idx not in selected_layers:
                selected_layers.append(layer_idx)
            if len(selected_layers) == layer_count:
                break
        selected_layers.sort()

    return selected_layers


def _resolve_layers_to_run(layers_arg: str, max_layers: int) -> list[int]:
    layers_arg_normalized = layers_arg.strip().lower()
    if layers_arg_normalized.startswith("auto:"):
        count_str = layers_arg.split(":", 1)[1].strip()
        if not count_str:
            raise ValueError("Missing auto layer count. Use format: auto:<count>")
        try:
            auto_count = int(count_str)
        except ValueError as error:
            raise ValueError(
                f"Invalid auto layer count '{count_str}'. Use an integer."
            ) from error
        return _auto_select_layers(auto_count, max_layers)

    return parse_layers_to_run(layers_arg, max_layers)


def _capture_occurs_after_steering(
    target_layer_idx: int, hook_point: str, steer_layer_idx: int
) -> bool:
    if target_layer_idx > steer_layer_idx:
        return True
    if target_layer_idx < steer_layer_idx:
        return False
    return hook_point == "block_out"


def _run_with_steering_and_capture(
    model,
    input_ids: torch.Tensor,
    steering_vector: torch.Tensor,
    alpha_value: float,
    steer_layer_idx: int,
    target_layer_idx: int,
    hook_point: str,
    device,
) -> torch.Tensor:
    layers_container = _get_layers_container(model)
    steer_layer_module = layers_container[steer_layer_idx]
    target_layer_module = layers_container[target_layer_idx]
    capture_module = _resolve_hook_module(target_layer_module, hook_point)

    captured: dict[str, torch.Tensor] = {}
    handles = []

    def _capture_hook(_module, _inputs, output):
        captured["h"] = _extract_hidden(output).detach()
        return output

    steering_hook = make_steering_hook(steering_vector, alpha_value, device=device)
    if capture_module is steer_layer_module:
        # Ensure capture observes the steered activation when both hooks are on
        # the same module (e.g., hook_point="block_out" at steer_layer_idx).
        handles.append(steer_layer_module.register_forward_hook(steering_hook))
        handles.append(capture_module.register_forward_hook(_capture_hook))
    else:
        handles.append(capture_module.register_forward_hook(_capture_hook))
        handles.append(steer_layer_module.register_forward_hook(steering_hook))

    try:
        with torch.no_grad():
            _ = model(input_ids)
    finally:
        for handle in handles:
            handle.remove()

    hidden = captured.get("h")
    if hidden is None:
        raise RuntimeError(
            f"Failed to capture hook activation at layer {target_layer_idx}, "
            f"hook point '{hook_point}'."
        )
    return hidden


def run_model_analysis(model_full_name, args, max_layers):
    """Run PCA linearity analysis for one model on a single device."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Started analysis on device: {device}")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Load model
    logger.info(f"Loading model {model_full_name}")
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_full_name, device_map=device, dtype=dtype, trust_remote_code=True
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_full_name, use_fast=True, device=device, dtype=dtype
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        logger.error(f"Failed to load model {model_full_name}: {e}")
        return

    model_name = get_model_name_for_path(model_full_name)
    os.makedirs(f"assets/linear/{model_name}", exist_ok=True)

    layers_to_run = _resolve_layers_to_run(args.layers, max_layers)
    steer_layer_idx = args.steer_layer
    hook_points = _parse_hook_points(args.hook_points)

    logger.info(f"Layers to run: {layers_to_run}")

    if steer_layer_idx < 0 or steer_layer_idx >= max_layers:
        raise ValueError(
            f"Invalid steer layer index {steer_layer_idx}. "
            f"Expected in [0, {max_layers - 1}]"
        )

    layers_container = _get_layers_container(model)
    valid_targets = []
    for layer_idx in layers_to_run:
        if layer_idx >= len(layers_container):
            logger.warning(f"Layer {layer_idx} out of range for model {model_name}.")
            continue
        for hook_point in hook_points:
            try:
                _resolve_hook_module(layers_container[layer_idx], hook_point)
                valid_targets.append((layer_idx, hook_point))
            except ValueError:
                logger.warning(
                    f"Skip unsupported hook point '{hook_point}' at layer {layer_idx}."
                )

    if not valid_targets:
        logger.error(f"No valid (layer, hook_point) targets for model {model_name}.")
        return

    concept_items = _parse_concepts_to_run(args.concepts)

    for concept_category_name, concept_category_config in concept_items:
        logger.info(f"Processing {concept_category_name}")

        concept_vectors_path = (
            f"assets/concept_vectors/{model_name}/{concept_category_name}.pt"
        )
        if not os.path.exists(concept_vectors_path):
            logger.warning(
                f"Concept vectors not found for {concept_category_name} in {model_name}. Skipping."
            )
            continue

        concept_vectors = torch.load(concept_vectors_path)  # Load to CPU initially

        # Load dataset and select prompts
        positive_dataset, _, dataset_key = load_concept_datasets(
            concept_category_name, concept_category_config
        )

        selected_prompts = []
        total_tokens = 0
        for i in range(min(args.test_size, len(positive_dataset))):
            prompt = positive_dataset[i][dataset_key]
            tokens = tokenizer(prompt, return_tensors="pt", truncation=False)
            prompt_length = tokens.input_ids.shape[1]
            if (
                total_tokens + prompt_length > args.max_tokens
                and len(selected_prompts) > 0
            ):
                break
            selected_prompts.append(prompt)
            total_tokens += prompt_length
            if total_tokens >= args.max_tokens:
                break

        input_ids = (
            tokenizer(
                selected_prompts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=args.max_tokens,
            )
            .to(device)
            .input_ids
        )

        # Generate random vector for comparison
        random_vector_dir = f"assets/linear/{model_name}/random_vectors"
        os.makedirs(random_vector_dir, exist_ok=True)
        random_vector_path = f"{random_vector_dir}/{concept_category_name}.pt"

        vector_dim = concept_vectors.shape[1]

        if os.path.exists(random_vector_path):
            random_vector_data = torch.load(random_vector_path)
            if isinstance(random_vector_data, dict):
                random_vector = random_vector_data["random_vector"]
            else:
                random_vector = random_vector_data
        else:
            random_vector = torch.randn(vector_dim, dtype=torch.float32)
            random_vector = random_vector / random_vector.norm()
            torch.save({"random_vector": random_vector}, random_vector_path)

        # Run analysis
        for vector_type, _vector_source in [
            ("concept", concept_vectors),
            ("random", random_vector),
        ]:
            results = {}
            alphas = torch.logspace(
                float(torch.log10(torch.tensor(args.alpha_min))),
                float(torch.log10(torch.tensor(args.alpha_max))),
                steps=args.alpha_points,
            ).tolist()

            iterator = tqdm(
                valid_targets,
                desc=f"Linearity ({concept_category_name}, {vector_type})",
            )

            for target in iterator:
                layer_idx, hook_point = target
                if vector_type == "concept":
                    steering_vector = concept_vectors[steer_layer_idx, :]
                else:
                    steering_vector = random_vector

                aggregator = PCAStatisticsAggregator()

                # Process in batches of prompts to save memory
                batch_size = 1  # Process one prompt (or small batch) at a time

                # Split input_ids into chunks
                total_prompts = input_ids.shape[0]

                for i in range(0, total_prompts, batch_size):
                    input_ids_batch = input_ids[i : i + batch_size]

                    batch_collected_deltas = []

                    h_ref_batch = _run_with_steering_and_capture(
                        model=model,
                        input_ids=input_ids_batch,
                        steering_vector=steering_vector,
                        alpha_value=0.0,
                        steer_layer_idx=steer_layer_idx,
                        target_layer_idx=layer_idx,
                        hook_point=hook_point,
                        device=device,
                    )
                    h_ref_flat_batch = hidden_to_flat(
                        h_ref_batch, target_dtype=torch.float32
                    )

                    if 0.0 not in alphas:
                        batch_collected_deltas.append(
                            torch.zeros_like(h_ref_flat_batch)
                        )

                    for alpha in alphas:
                        h_batch = _run_with_steering_and_capture(
                            model=model,
                            input_ids=input_ids_batch,
                            steering_vector=steering_vector,
                            alpha_value=alpha,
                            steer_layer_idx=steer_layer_idx,
                            target_layer_idx=layer_idx,
                            hook_point=hook_point,
                            device=device,
                        )

                        if (
                            args.remove_concept_vector
                            and _capture_occurs_after_steering(
                                target_layer_idx=layer_idx,
                                hook_point=hook_point,
                                steer_layer_idx=steer_layer_idx,
                            )
                        ):
                            steering_vec_device = steering_vector.to(
                                device=h_batch.device, dtype=h_batch.dtype
                            )
                            h_batch = h_batch - alpha * steering_vec_device

                        h_flat = hidden_to_flat(h_batch, target_dtype=torch.float32)
                        delta = h_flat - h_ref_flat_batch

                        batch_collected_deltas.append(delta)

                    # Stack deltas for this batch [N_steps, N_tokens_in_batch, Hidden]
                    batch_trajectory = torch.stack(batch_collected_deltas)

                    # Update aggregator
                    aggregator.update(batch_trajectory)

                    del (
                        batch_trajectory,
                        batch_collected_deltas,
                        h_ref_batch,
                        h_ref_flat_batch,
                    )

                stats = aggregator.finalize()

                if layer_idx not in results:
                    results[layer_idx] = {}

                results[layer_idx][hook_point] = {
                    "explained_variance_ratio_pc1_mean": stats["mean_score"],
                    "explained_variance_ratio_pc1_std": stats["std_score"],
                    "n_components_95_mean": stats["mean_n95"],
                    "n_components_95_std": stats["std_n95"],
                    "count": stats["count"],
                    "mean_score": stats["mean_score"],
                    "std_score": stats["std_score"],
                    "alphas": alphas,
                }
                logger.info(
                    f"PCA for {concept_category_name} ({vector_type}) "
                    f"layer {layer_idx}, hook {hook_point}: "
                    f"pc1={stats['mean_score']:.4f}, "
                    f"n95={stats['mean_n95']:.2f}, "
                    f"count={stats['count']}"
                )

            # Save results
            suffix = "_remove" if args.remove_concept_vector else ""
            save_path = (
                f"assets/linear/{model_name}/"
                f"pca_hooks_{concept_category_name}_{vector_type}{suffix}.pt"
            )
            torch.save(
                {
                    "model": model_full_name,
                    "concept_category": concept_category_name,
                    "vector_type": vector_type,
                    "steer_layer_idx": steer_layer_idx,
                    "hook_points": hook_points,
                    "layers": layers_to_run,
                    "analysis_targets": valid_targets,
                    "results": results,
                },
                save_path,
            )

    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help=(
            "Model name to process. Use 'all' to process all models. "
            f"Available: {list(MODEL_LAYERS.keys())}"
        ),
    )
    parser.add_argument(
        "--concepts",
        type=str,
        default="steering_safety,steering_detectable_format_json_format",
        help=(
            "Comma-separated concept categories to process. "
            "Use 'all' to process all concept categories."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=int, default=16)
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=500,
        help="Maximum number of tokens to use from the dataset",
    )
    # Trajectory sweep parameters
    parser.add_argument("--alpha_min", type=float, default=1)
    parser.add_argument("--alpha_max", type=float, default=1e5)
    parser.add_argument(
        "--alpha_points", type=int, default=200
    )  # Fewer points than smoothness

    parser.add_argument(
        "--layers",
        type=str,
        default="auto:8",
        help=(
            "Layer selection. Supports: 'auto:<count>' (e.g. auto:8), "
            "comma-separated percentages, comma-separated layer indices, or 'all'."
        ),
    )
    parser.add_argument(
        "--steer_layer",
        type=int,
        default=0,
        help="Layer index where linear steering is injected (default: first layer).",
    )
    parser.add_argument(
        "--hook_points",
        type=str,
        default="input_ln,attn,post_attn_ln,mlp",
        help=(
            "Comma-separated hook points for PCA capture. "
            "Supported: input_ln,attn,post_attn_ln,mlp,block_out"
        ),
    )
    parser.add_argument(
        "--remove_concept_vector",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove concept vector from hidden states (default: true)",
    )
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    logger.add("logs/linear.log")
    logger.info(f"args: {args}")
    set_seed(args.seed)

    if args.model.lower() == "all":
        models_to_process = list(MODEL_LAYERS.items())
    else:
        if args.model not in MODEL_LAYERS:
            raise ValueError(
                f"Unknown model '{args.model}'. Available: {list(MODEL_LAYERS.keys())}"
            )
        models_to_process = [(args.model, MODEL_LAYERS[args.model])]

    for model_full_name, max_layers in models_to_process:
        logger.info(f"Processing model: {model_full_name}")
        run_model_analysis(model_full_name, args, max_layers)


if __name__ == "__main__":
    main()
