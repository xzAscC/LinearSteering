import argparse
import faulthandler
import hashlib
import json
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

    def __init__(self, token_chunk_size: int = 32):
        if token_chunk_size <= 0:
            raise ValueError("token_chunk_size must be positive")
        self.score_sum = 0.0
        self.score_sq_sum = 0.0
        self.n95_sum = 0.0
        self.n95_sq_sum = 0.0
        self.count = 0
        self.degenerate_count = 0
        self.token_chunk_size = token_chunk_size

    def update(self, trajectory_data):
        """
        Compute stats for the current batch of trajectories and update aggregators.

        Args:
            trajectory_data (torch.Tensor): [num_steps, num_samples, hidden_dim]
        """
        # [Steps, N_tokens, Hidden] -> [N_tokens, Steps, Hidden]
        X = trajectory_data.permute(1, 0, 2).float()

        epsilon = 1e-12
        total_tokens = X.shape[0]

        for start in range(0, total_tokens, self.token_chunk_size):
            end = min(start + self.token_chunk_size, total_tokens)
            X_chunk = X[start:end]

            # Center each trajectory independently
            X_mean = X_chunk.mean(dim=1, keepdim=True)
            X_centered = X_chunk - X_mean

            # Compute SVD for each sample
            # S has shape [N_chunk, min(T, D)]
            S = torch.linalg.svdvals(X_centered)

            eigenvalues = S**2
            total_variance = eigenvalues.sum(dim=-1)

            # Avoid division by zero
            valid_mask = total_variance > epsilon

            valid_count = int(valid_mask.sum().item())
            self.degenerate_count += int((~valid_mask).sum().item())
            if valid_count == 0:
                continue

            valid_eigenvars = eigenvalues[valid_mask]
            valid_total = total_variance[valid_mask]

            # 1. Linearity score (PC1 / Total)
            scores = valid_eigenvars[:, 0] / valid_total

            # 2. n_95
            ratios = valid_eigenvars / valid_total.unsqueeze(-1)
            cumsum = torch.cumsum(ratios, dim=-1)
            # Count components needed to reach >= 0.95
            n_95 = (cumsum < 0.95).sum(dim=-1).float() + 1.0

            # Update aggregators with valid trajectories only
            self.score_sum += scores.sum().item()
            self.score_sq_sum += (scores**2).sum().item()
            self.n95_sum += n_95.sum().item()
            self.n95_sq_sum += (n_95**2).sum().item()
            self.count += valid_count

    def finalize(self):
        if self.count == 0:
            return {
                "mean_score": float("nan"),
                "std_score": float("nan"),
                "mean_n95": float("nan"),
                "std_n95": float("nan"),
                "count": 0,
                "degenerate_count": self.degenerate_count,
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
            "degenerate_count": self.degenerate_count,
        }


QWEN3_DEFAULT_HOOK_POINTS = [
    "input_ln",
    "attn",
    "post_attn_ln",
    "mlp",
    "block_out",
]
GEMMA2_DEFAULT_HOOK_POINTS = [
    "input_ln",
    "attn",
    "post_attn_proj_ln",
    "post_attn_ln",
    "mlp",
    "post_mlp_ln",
    "block_out",
]
RANDOM_DIRECTION_CONCEPT = "steering_random_direction"
RANDOM_DIRECTION_DATASET_CONCEPT = "steering_safety"


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


def _is_gemma2_style_layer(layer_module) -> bool:
    return all(
        hasattr(layer_module, name)
        for name in (
            "input_layernorm",
            "self_attn",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "mlp",
            "post_feedforward_layernorm",
        )
    )


def _parse_hook_points(hook_points_arg: str) -> list[str]:
    alias_map = {
        "ln1": "input_ln",
        "ln2": "post_attn_ln",
        "pre_mlp_ln": "post_attn_ln",
        "pre_ffn_ln": "post_attn_ln",
        "block": "block_out",
        "resid": "block_out",
        "post_attention_ln": "post_attn_proj_ln",
        "post_attn_norm": "post_attn_proj_ln",
        "post_ffn_ln": "post_mlp_ln",
        "post_feedforward_ln": "post_mlp_ln",
    }
    valid_points = {
        "input_ln",
        "attn",
        "post_attn_proj_ln",
        "post_attn_ln",
        "mlp",
        "post_mlp_ln",
        "block_out",
    }

    parsed_points: list[str] = []
    for raw_point in hook_points_arg.split(","):
        point = raw_point.strip().lower().replace("-", "_").replace(" ", "_")
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
        all_concepts = list(CONCEPT_CATEGORIES.items())
        all_concepts.append((RANDOM_DIRECTION_CONCEPT, {}))
        return all_concepts

    selected_concepts: list[str] = []
    for raw_concept in concepts_arg.split(","):
        concept_name = raw_concept.strip()
        if not concept_name:
            continue
        if concept_name == RANDOM_DIRECTION_CONCEPT:
            if concept_name not in selected_concepts:
                selected_concepts.append(concept_name)
            continue
        if concept_name not in CONCEPT_CATEGORIES:
            raise ValueError(
                f"Unknown concept '{concept_name}'. "
                f"Available: {sorted([*CONCEPT_CATEGORIES.keys(), RANDOM_DIRECTION_CONCEPT])}"
            )
        if concept_name not in selected_concepts:
            selected_concepts.append(concept_name)

    if not selected_concepts:
        raise ValueError("No valid concepts were provided.")

    concept_items: list[tuple[str, dict]] = []
    for name in selected_concepts:
        if name == RANDOM_DIRECTION_CONCEPT:
            concept_items.append((name, {}))
        else:
            concept_items.append((name, CONCEPT_CATEGORIES[name]))
    return concept_items


def _resolve_hook_module(layer_module, hook_point: str):
    if _is_gemma2_style_layer(layer_module):
        mapping = {
            "input_ln": layer_module.input_layernorm,
            "attn": layer_module.self_attn,
            "post_attn_proj_ln": layer_module.post_attention_layernorm,
            # Keep `post_attn_ln` aligned with Qwen-style semantics: capture
            # the normalization immediately before MLP.
            "post_attn_ln": layer_module.pre_feedforward_layernorm,
            "mlp": layer_module.mlp,
            "post_mlp_ln": layer_module.post_feedforward_layernorm,
            "block_out": layer_module,
        }
        return mapping[hook_point]

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


def _resolve_default_hook_points(layers_container) -> list[str]:
    if len(layers_container) > 0 and _is_gemma2_style_layer(layers_container[0]):
        return GEMMA2_DEFAULT_HOOK_POINTS.copy()
    return QWEN3_DEFAULT_HOOK_POINTS.copy()


def _auto_select_layers(
    layer_count: int, max_layers: int, min_layer_exclusive: int = -1
) -> list[int]:
    """Select approximately uniform layer indices from available layers.

    Layers are selected from ``(min_layer_exclusive, max_layers - 1)`` and
    capped at ``max_layers - 2`` to keep compatibility with existing behavior.
    """
    max_valid_layer = max_layers - 2
    first_valid_layer = min_layer_exclusive + 1
    if max_valid_layer < first_valid_layer:
        return []

    if layer_count <= 0:
        raise ValueError("auto layer count must be positive")

    available_layers = list(range(first_valid_layer, max_valid_layer + 1))
    total_available = len(available_layers)
    if layer_count >= total_available:
        return available_layers

    if layer_count == 1:
        return [available_layers[0]]

    sample_positions = [
        int(round(i * (total_available - 1) / (layer_count - 1)))
        for i in range(layer_count)
    ]
    selected_layers: list[int] = []
    for position in sample_positions:
        layer_idx = available_layers[position]
        if layer_idx not in selected_layers:
            selected_layers.append(layer_idx)

    # Fill missing slots if integer rounding produced duplicates.
    if len(selected_layers) < layer_count:
        for layer_idx in available_layers:
            if layer_idx not in selected_layers:
                selected_layers.append(layer_idx)
            if len(selected_layers) == layer_count:
                break

    return selected_layers


def _resolve_layers_to_run(
    layers_arg: str, max_layers: int, steer_layer_idx: int | None = None
) -> list[int]:
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
        min_layer_exclusive = -1 if steer_layer_idx is None else steer_layer_idx
        return _auto_select_layers(
            auto_count,
            max_layers,
            min_layer_exclusive=min_layer_exclusive,
        )

    return parse_layers_to_run(layers_arg, max_layers)


def _capture_occurs_after_steering(
    target_layer_idx: int, hook_point: str, steer_layer_idx: int
) -> bool:
    # Only block_out captures should remove alpha * v.
    if hook_point != "block_out":
        return False
    return target_layer_idx >= steer_layer_idx


def _build_run_tag(
    steer_layer_idx: int,
    layers_to_run: list[int],
    hook_points: list[str],
    args,
) -> str:
    run_config = {
        "steer_layer_idx": steer_layer_idx,
        "layers_to_run": layers_to_run,
        "hook_points": hook_points,
        "alpha_min": args.alpha_min,
        "alpha_max": args.alpha_max,
        "alpha_points": args.alpha_points,
        "test_size": args.test_size,
        "max_tokens": args.max_tokens,
        "remove_concept_vector": args.remove_concept_vector,
    }
    config_json = json.dumps(run_config, sort_keys=True)
    config_hash = hashlib.md5(config_json.encode()).hexdigest()[:10]
    return f"cfg_{config_hash}"


def _prepare_input_ids(
    tokenizer,
    positive_dataset,
    dataset_key: str,
    test_size: int,
    max_tokens: int,
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected_prompts = []
    total_tokens = 0

    for i in range(min(test_size, len(positive_dataset))):
        prompt = positive_dataset[i][dataset_key]
        tokens = tokenizer(prompt, return_tensors="pt", truncation=False)
        prompt_length = tokens.input_ids.shape[1]
        if total_tokens + prompt_length > max_tokens and len(selected_prompts) > 0:
            break
        selected_prompts.append(prompt)
        total_tokens += prompt_length
        if total_tokens >= max_tokens:
            break

    tokenized = tokenizer(
        selected_prompts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_tokens,
    ).to(device)
    return tokenized.input_ids, tokenized.attention_mask


def _hidden_to_flat_with_attention_mask(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    target_dtype=torch.float32,
) -> torch.Tensor:
    flat_hidden = hidden_to_flat(hidden, target_dtype=target_dtype)
    flat_mask = attention_mask.reshape(-1).to(dtype=torch.bool)
    return flat_hidden[flat_mask]


def _run_with_steering_and_capture(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
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
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
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

    steer_layer_idx = args.steer_layer
    layers_to_run = _resolve_layers_to_run(
        args.layers,
        max_layers,
        steer_layer_idx=steer_layer_idx,
    )
    layers_container = _get_layers_container(model)
    if args.hook_points.strip().lower() == "auto":
        hook_points = _resolve_default_hook_points(layers_container)
        logger.info(f"Auto hook points selected: {hook_points}")
    else:
        hook_points = _parse_hook_points(args.hook_points)

    logger.info(f"Layers to run: {layers_to_run}")

    if steer_layer_idx < 0 or steer_layer_idx >= max_layers:
        raise ValueError(
            f"Invalid steer layer index {steer_layer_idx}. "
            f"Expected in [0, {max_layers - 1}]"
        )

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

    run_tag = _build_run_tag(
        steer_layer_idx=steer_layer_idx,
        layers_to_run=layers_to_run,
        hook_points=hook_points,
        args=args,
    )
    logger.info(f"Run tag: {run_tag}")

    concept_items = _parse_concepts_to_run(args.concepts)

    def _run_single_vector_analysis(
        model_obj,
        concept_name: str,
        vector_type: str,
        steering_vector: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> None:
        results = {}
        alphas = torch.logspace(
            float(torch.log10(torch.tensor(args.alpha_min))),
            float(torch.log10(torch.tensor(args.alpha_max))),
            steps=args.alpha_points,
        ).tolist()

        iterator = tqdm(
            valid_targets,
            desc=f"Linearity ({concept_name}, {vector_type})",
        )

        for target in iterator:
            layer_idx, hook_point = target
            try:
                aggregator = PCAStatisticsAggregator(
                    token_chunk_size=args.pca_token_chunk
                )

                # Process in batches of prompts to save memory
                batch_size = 1  # Process one prompt (or small batch) at a time

                # Split input_ids into chunks
                total_prompts = input_ids.shape[0]

                for i in range(0, total_prompts, batch_size):
                    input_ids_batch = input_ids[i : i + batch_size]
                    attention_mask_batch = attention_mask[i : i + batch_size]

                    batch_collected_deltas = []

                    h_ref_batch = _run_with_steering_and_capture(
                        model=model_obj,
                        input_ids=input_ids_batch,
                        attention_mask=attention_mask_batch,
                        steering_vector=steering_vector,
                        alpha_value=0.0,
                        steer_layer_idx=steer_layer_idx,
                        target_layer_idx=layer_idx,
                        hook_point=hook_point,
                        device=device,
                    )
                    h_ref_flat_batch = _hidden_to_flat_with_attention_mask(
                        h_ref_batch,
                        target_dtype=torch.float32,
                        attention_mask=attention_mask_batch,
                    )

                    if 0.0 not in alphas:
                        batch_collected_deltas.append(
                            torch.zeros_like(h_ref_flat_batch)
                        )

                    for alpha in alphas:
                        h_batch = _run_with_steering_and_capture(
                            model=model_obj,
                            input_ids=input_ids_batch,
                            attention_mask=attention_mask_batch,
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

                        h_flat = _hidden_to_flat_with_attention_mask(
                            h_batch,
                            attention_mask=attention_mask_batch,
                            target_dtype=torch.float32,
                        )
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
            except Exception:
                logger.exception(
                    f"Failed target for {concept_name} ({vector_type}) at "
                    f"layer {layer_idx}, hook {hook_point}"
                )
                raise

            if layer_idx not in results:
                results[layer_idx] = {}

            results[layer_idx][hook_point] = {
                "explained_variance_ratio_pc1_mean": stats["mean_score"],
                "explained_variance_ratio_pc1_std": stats["std_score"],
                "n_components_95_mean": stats["mean_n95"],
                "n_components_95_std": stats["std_n95"],
                "count": stats["count"],
                "degenerate_count": stats["degenerate_count"],
                "mean_score": stats["mean_score"],
                "std_score": stats["std_score"],
                "alphas": alphas,
            }
            logger.info(
                f"PCA for {concept_name} ({vector_type}) "
                f"layer {layer_idx}, hook {hook_point}: "
                f"pc1={stats['mean_score']:.4f}, "
                f"n95={stats['mean_n95']:.2f}, "
                f"count={stats['count']}, "
                f"degenerate={stats['degenerate_count']}"
            )

        suffix = "_remove" if args.remove_concept_vector else ""
        save_path = (
            f"assets/linear/{model_name}/"
            f"pca_hooks_{concept_name}_{vector_type}_{run_tag}{suffix}.pt"
        )
        torch.save(
            {
                "model": model_full_name,
                "concept_category": concept_name,
                "vector_type": vector_type,
                "run_tag": run_tag,
                "steer_layer_idx": steer_layer_idx,
                "hook_points": hook_points,
                "layers": layers_to_run,
                "analysis_targets": valid_targets,
                "results": results,
            },
            save_path,
        )

    for concept_category_name, concept_category_config in concept_items:
        logger.info(f"Processing {concept_category_name}")

        if concept_category_name == RANDOM_DIRECTION_CONCEPT:
            safety_config = CONCEPT_CATEGORIES[RANDOM_DIRECTION_DATASET_CONCEPT]
            positive_dataset, _, dataset_key = load_concept_datasets(
                RANDOM_DIRECTION_DATASET_CONCEPT, safety_config
            )
            input_ids, attention_mask = _prepare_input_ids(
                tokenizer=tokenizer,
                positive_dataset=positive_dataset,
                dataset_key=dataset_key,
                test_size=args.test_size,
                max_tokens=args.max_tokens,
                device=device,
            )

            random_vector_dir = f"assets/linear/{model_name}/random_vectors"
            os.makedirs(random_vector_dir, exist_ok=True)
            random_vector_path = f"{random_vector_dir}/{RANDOM_DIRECTION_CONCEPT}.pt"

            vector_dim = getattr(model.config, "hidden_size")
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

            _run_single_vector_analysis(
                model_obj=model,
                concept_name=RANDOM_DIRECTION_CONCEPT,
                vector_type="random",
                steering_vector=random_vector,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            continue

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

        input_ids, attention_mask = _prepare_input_ids(
            tokenizer=tokenizer,
            positive_dataset=positive_dataset,
            dataset_key=dataset_key,
            test_size=args.test_size,
            max_tokens=args.max_tokens,
            device=device,
        )

        steering_vector = concept_vectors[steer_layer_idx, :]
        _run_single_vector_analysis(
            model_obj=model,
            concept_name=concept_category_name,
            vector_type="concept",
            steering_vector=steering_vector,
            input_ids=input_ids,
            attention_mask=attention_mask,
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
            "Layer selection. Supports: 'auto:<count>' (e.g. auto:8, sampled from "
            "layers after --steer_layer), "
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
        default="auto",
        help=(
            "Comma-separated hook points for PCA capture. "
            "Use 'auto' for model-specific defaults (Qwen3: input_ln,attn,"
            "post_attn_ln,mlp,block_out; Gemma2: input_ln,attn,post_attn_proj_ln,"
            "post_attn_ln,mlp,post_mlp_ln,block_out). Supported: input_ln,attn,"
            "post_attn_proj_ln,post_attn_ln,mlp,post_mlp_ln,block_out"
        ),
    )
    parser.add_argument(
        "--remove_concept_vector",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove concept vector from hidden states (default: true)",
    )
    parser.add_argument(
        "--pca_token_chunk",
        type=int,
        default=32,
        help=(
            "Number of token trajectories processed per SVD call. "
            "Lower this if the process exits unexpectedly (possible OOM)."
        ),
    )
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    logger.add("logs/linear.log")
    faulthandler.enable()
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
