import argparse
import json
import os
from datetime import datetime
from datetime import timezone
from typing import Any, Dict, List, Optional

import torch
import transformers
from dotenv import load_dotenv
from loguru import logger

from probe_utils import load_vector
from probe_utils import select_steer_layers
from utils import CONCEPT_CATEGORIES
from utils import MODEL_LAYERS
from utils import _get_layers_container
from utils import get_model_name_for_path
from utils import load_concept_datasets
from utils import make_steering_hook
from utils import set_seed


def build_inputs(tokenizer, prompt: str) -> Dict[str, torch.Tensor]:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = prompt
    encoded = tokenizer(
        text,
        return_tensors="pt",
        padding=False,
        truncation=True,
    )
    return {
        "input_ids": encoded.input_ids,
        "attention_mask": encoded.attention_mask,
    }


def _extract_hidden(output: torch.Tensor | tuple) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def _get_final_norm_module(model):
    direct_attrs = ["norm", "final_layernorm", "final_layer_norm", "ln_f"]
    for attr in direct_attrs:
        module = getattr(model, attr, None)
        if module is not None:
            return module

    containers = ["model", "transformer", "gpt_neox"]
    for container in containers:
        root = getattr(model, container, None)
        if root is None:
            continue
        for attr in direct_attrs:
            module = getattr(root, attr, None)
            if module is not None:
                return module

    raise AttributeError("Unable to locate final normalization module on model")


def make_pre_norm_subtract_hook(
    steering_vector: torch.Tensor, alpha_value: float, device=None
):
    def _hook(_module, inputs):
        if not inputs:
            return None
        hidden = inputs[0]
        target_device = device if device is not None else hidden.device
        vec = steering_vector.to(device=target_device, dtype=hidden.dtype)
        hidden = hidden + (alpha_value * vec)
        return (hidden,) + inputs[1:]

    return _hook


def capture_layer_hidden(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    layers_container = _get_layers_container(model)
    target_layer_module = layers_container[layer_idx]

    captured: dict[str, torch.Tensor] = {}

    def _layer_forward_hook(_module, _inputs, output):
        captured["h"] = _extract_hidden(output).detach()
        return output

    handle = target_layer_module.register_forward_hook(_layer_forward_hook)
    try:
        _ = model(input_ids, attention_mask=attention_mask)
    finally:
        handle.remove()

    hidden = captured.get("h", None)
    if hidden is None:
        raise RuntimeError("Failed to capture hidden states")
    return hidden


def resolve_concept_name(steer_vector: str) -> str:
    if steer_vector.endswith(".pt"):
        return os.path.basename(steer_vector).replace(".pt", "")
    return steer_vector


def compute_projection_for_prompt(
    model,
    tokenizer,
    prompt: str,
    unit_vector: torch.Tensor,
    steer_layer: int,
    device: str,
) -> float:
    encoded = build_inputs(tokenizer, prompt)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    with torch.no_grad():
        hidden = capture_layer_hidden(model, input_ids, attention_mask, steer_layer)
    token_idx = int(attention_mask[0].sum().item()) - 1
    vec = unit_vector.to(device=hidden.device, dtype=hidden.dtype)
    projection = torch.dot(hidden[0, token_idx], vec)
    return float(projection.item())


def compute_mean_projection(
    model,
    tokenizer,
    prompts: List[str],
    unit_vector: torch.Tensor,
    steer_layer: int,
    device: str,
) -> float:
    if not prompts:
        raise ValueError("No prompts provided for mean projection")
    total = 0.0
    for prompt in prompts:
        total += compute_projection_for_prompt(
            model,
            tokenizer,
            prompt,
            unit_vector,
            steer_layer,
            device,
        )
    return total / len(prompts)


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
    steering_vector: Optional[torch.Tensor] = None,
    steer_layer: Optional[int] = None,
    alpha: float = 0.0,
    remove_at_final_norm: bool = False,
    final_norm_module=None,
    final_norm_vector: Optional[torch.Tensor] = None,
    final_norm_alpha: Optional[float] = None,
) -> str:
    encoded = build_inputs(tokenizer, prompt)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    handles = []
    if steering_vector is not None and steer_layer is not None:
        layers_container = _get_layers_container(model)
        handles.append(
            layers_container[steer_layer].register_forward_hook(
                make_steering_hook(steering_vector, alpha, device=device)
            )
        )
        if remove_at_final_norm:
            if final_norm_module is None:
                final_norm_module = _get_final_norm_module(model)
            if final_norm_vector is None:
                final_norm_vector = steering_vector
            if final_norm_alpha is None:
                final_norm_alpha = alpha
            handles.append(
                final_norm_module.register_forward_pre_hook(
                    make_pre_norm_subtract_hook(
                        final_norm_vector, final_norm_alpha, device=device
                    )
                )
            )

    try:
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        for handle in handles:
            handle.remove()

    generated = output_ids[0][input_ids.shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def resolve_steer_layer(model_name: str, steer_layer_arg: Optional[int]) -> int:
    max_layers = MODEL_LAYERS[model_name]
    if steer_layer_arg is not None:
        if steer_layer_arg < 0 or steer_layer_arg >= max_layers - 1:
            raise ValueError(f"Invalid steer_layer {steer_layer_arg}")
        return steer_layer_arg
    layers = select_steer_layers(max_layers, total_layers=6)
    if not layers:
        raise ValueError("No steer layers selected")
    return layers[len(layers) // 2]


def resolve_final_norm_layer(
    model_name: str, final_norm_layer_arg: Optional[int]
) -> Optional[int]:
    if final_norm_layer_arg is None:
        return None
    max_layers = MODEL_LAYERS[model_name]
    if final_norm_layer_arg < 0 or final_norm_layer_arg >= max_layers:
        raise ValueError(f"Invalid final_norm_layer {final_norm_layer_arg}")
    return final_norm_layer_arg


def resolve_final_norm_alpha_layer(
    model_name: str, final_norm_alpha_layer_arg: Optional[int]
) -> Optional[int]:
    if final_norm_alpha_layer_arg is None:
        return None
    max_layers = MODEL_LAYERS[model_name]
    if final_norm_alpha_layer_arg < 0 or final_norm_alpha_layer_arg >= max_layers - 1:
        raise ValueError(f"Invalid final_norm_alpha_layer {final_norm_alpha_layer_arg}")
    return final_norm_alpha_layer_arg


def load_dataset_split(dataset_name: str, dataset_config: str, split: str):
    import datasets

    try:
        return datasets.load_dataset(dataset_name, dataset_config, split=split)
    except Exception as err:
        logger.warning(
            "Failed to load split '{}' from '{}': {}",
            split,
            dataset_name,
            err,
        )
        dataset = datasets.load_dataset(dataset_name, dataset_config)
        if split in dataset:
            return dataset[split]
        if "train" in dataset:
            logger.warning("Falling back to train split")
            return dataset["train"]
        first_split = list(dataset.keys())[0]
        logger.warning("Falling back to first split: {}", first_split)
        return dataset[first_split]


def sample_prompts(dataset, sample_size: int, seed: int) -> List[Dict[str, Any]]:
    if len(dataset) <= sample_size:
        return [dataset[i] for i in range(len(dataset))]
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:sample_size].tolist()
    return [dataset[i] for i in indices]


def extract_prompt(item: Dict[str, Any]) -> str:
    for key in ["prompt", "instruction", "behavior", "text"]:
        if key in item and item[key]:
            return str(item[key])
    raise ValueError("No prompt field found in dataset item")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Steer Qwen on HarmBench prompts")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--dataset", type=str, default="walledai/HarmBench")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--dataset_config", type=str, default="standard")
    parser.add_argument("--sample_size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument(
        "--steer_vector",
        type=str,
        default="steering_safety",
    )
    parser.add_argument("--steer_layer", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument(
        "--alpha_mode",
        type=str,
        default="auto",
        choices=["auto", "manual"],
    )
    parser.add_argument("--alpha_sample_size", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default="logs/harmbench_eval")
    parser.add_argument(
        "--run_mode",
        type=str,
        default="unsteering",
        choices=["steering", "unsteering"],
        help="Run only steering or unsteering generation",
    )
    parser.add_argument(
        "--remove_at_final_norm",
        action="store_true",
        help="Subtract concept vector before final normalization",
    )
    parser.add_argument(
        "--final_norm_layer",
        type=int,
        default=None,
        help="Layer index to use for final norm subtraction (defaults to last layer)",
    )
    parser.add_argument(
        "--final_norm_alpha_layer",
        type=int,
        default=17,
        help="Layer index used to compute alpha for final norm subtraction",
    )
    args = parser.parse_args()

    if args.model not in MODEL_LAYERS:
        raise ValueError(f"Unknown model: {args.model}")

    set_seed(args.seed)
    print(args)
    os.makedirs(args.output_dir, exist_ok=True)

    model_name = get_model_name_for_path(args.model)
    steer_layer = resolve_steer_layer(args.model, args.steer_layer)
    final_norm_layer = None
    final_norm_alpha_layer = None
    if args.remove_at_final_norm:
        final_norm_layer = resolve_final_norm_layer(args.model, args.final_norm_layer)
        if final_norm_layer is None:
            final_norm_layer = MODEL_LAYERS[args.model] - 1
        final_norm_alpha_layer = resolve_final_norm_alpha_layer(
            args.model, args.final_norm_alpha_layer
        )

    vector_path = args.steer_vector
    if not vector_path.endswith(".pt"):
        vector_path = os.path.join(
            "assets",
            "concept_vectors",
            model_name,
            f"{args.steer_vector}.pt",
        )

    steering_tensor = load_vector(vector_path)
    if steering_tensor.ndim == 2:
        steering_vector = steering_tensor[steer_layer]
    else:
        steering_vector = steering_tensor

    final_norm_vector = None
    final_norm_alpha_vector = None
    if args.remove_at_final_norm:
        if steering_tensor.ndim == 2:
            final_norm_vector = steering_tensor[final_norm_layer]
            if final_norm_alpha_layer is not None:
                final_norm_alpha_vector = steering_tensor[final_norm_alpha_layer]
        else:
            final_norm_vector = steering_tensor
            final_norm_alpha_vector = steering_tensor

    vector_norm = steering_vector.norm().item()
    if vector_norm == 0:
        raise ValueError("Steering vector has zero norm")
    steering_unit = steering_vector / vector_norm

    final_norm_unit = None
    if final_norm_vector is not None:
        final_norm_norm = final_norm_vector.norm().item()
        if final_norm_norm == 0:
            raise ValueError("Final norm steering vector has zero norm")
        final_norm_unit = final_norm_vector / final_norm_norm

    final_norm_alpha_unit = None
    if final_norm_alpha_vector is not None:
        final_norm_alpha_norm = final_norm_alpha_vector.norm().item()
        if final_norm_alpha_norm == 0:
            raise ValueError("Final norm alpha vector has zero norm")
        final_norm_alpha_unit = final_norm_alpha_vector / final_norm_alpha_norm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    logger.info("Device: {}", device)
    logger.info("Steer layer: {}", steer_layer)
    logger.info("Steer vector: {}", vector_path)
    logger.info("Alpha mode: {}", args.alpha_mode)
    logger.info("Remove at final norm: {}", args.remove_at_final_norm)
    if args.remove_at_final_norm:
        logger.info("Final norm layer: {}", final_norm_layer)
        logger.info("Final norm alpha layer: {}", final_norm_alpha_layer)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, device_map=device, dtype=dtype, trust_remote_code=True
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    final_norm_module = None
    if args.remove_at_final_norm:
        final_norm_module = _get_final_norm_module(model)

    alpha_zbar = None
    alpha_zbar_final_norm = None
    steering_vector_for_generation = steering_vector
    final_norm_vector_for_generation = final_norm_vector
    if args.alpha_mode == "auto":
        concept_name = resolve_concept_name(args.steer_vector)
        concept_config = CONCEPT_CATEGORIES.get(concept_name)
        if concept_config is None:
            raise ValueError(
                f"Auto alpha requires a concept category; got '{concept_name}'"
            )
        positive_dataset, _, dataset_key = load_concept_datasets(
            concept_name, concept_config
        )
        positive_items = sample_prompts(
            positive_dataset, args.alpha_sample_size, args.seed
        )
        positive_prompts = [str(item[dataset_key]) for item in positive_items]
        alpha_zbar = compute_mean_projection(
            model,
            tokenizer,
            positive_prompts,
            steering_unit,
            steer_layer,
            device,
        )
        logger.info("Alpha zbar: {:.6f}", alpha_zbar)
        steering_vector_for_generation = steering_unit
        if (
            args.remove_at_final_norm
            and final_norm_layer is not None
            and final_norm_alpha_layer is not None
        ):
            if final_norm_alpha_layer == steer_layer:
                alpha_zbar_final_norm = alpha_zbar
            else:
                alpha_zbar_final_norm = compute_mean_projection(
                    model,
                    tokenizer,
                    positive_prompts,
                    final_norm_alpha_unit,
                    final_norm_alpha_layer,
                    device,
                )
            logger.info("Final norm alpha zbar: {:.6f}", alpha_zbar_final_norm)
            final_norm_vector_for_generation = final_norm_unit
    else:
        logger.info("Alpha: {}", args.alpha)

    dataset = load_dataset_split(args.dataset, args.dataset_config, args.split)
    items = sample_prompts(dataset, args.sample_size, args.seed)

    steer_name = resolve_concept_name(args.steer_vector)
    run_tag = "unsteering_safety" if args.run_mode == "unsteering" else "steering"
    if args.remove_at_final_norm:
        run_tag = f"{run_tag}_finalnorm_remove"
    run_dir = os.path.join(
        args.output_dir,
        model_name,
        str(steer_layer),
        steer_name,
        args.alpha_mode,
        run_tag,
    )
    samples_dir = os.path.join(run_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    with open(os.path.join(run_dir, "steer_layer.txt"), "w") as f:
        f.write(f"{steer_layer}\n")

    records = []
    alpha_values = []
    final_norm_alpha_values = []
    for idx, item in enumerate(items):
        prompt = extract_prompt(item)
        alpha_value = args.alpha
        if args.alpha_mode == "auto":
            alpha_value = (alpha_zbar or 0.0) - compute_projection_for_prompt(
                model,
                tokenizer,
                prompt,
                steering_unit,
                steer_layer,
                device,
            )
        final_norm_alpha_value = None
        if (
            args.remove_at_final_norm
            and final_norm_layer is not None
            and final_norm_alpha_layer is not None
        ):
            if args.alpha_mode == "auto":
                if final_norm_alpha_layer == steer_layer:
                    final_norm_alpha_value = alpha_value
                else:
                    final_norm_alpha_value = (
                        alpha_zbar_final_norm or 0.0
                    ) - compute_projection_for_prompt(
                        model,
                        tokenizer,
                        prompt,
                        final_norm_alpha_unit,
                        final_norm_alpha_layer,
                        device,
                    )
            else:
                final_norm_alpha_value = args.alpha
        alpha_values.append(alpha_value)
        if final_norm_alpha_value is not None:
            final_norm_alpha_values.append(final_norm_alpha_value)
        logger.info("Running sample {} (alpha={:.6f})", idx, alpha_value)

        record = {
            "idx": idx,
            "prompt": prompt,
            "alpha": alpha_value,
        }
        if final_norm_alpha_value is not None:
            record["final_norm_alpha"] = final_norm_alpha_value

        if args.run_mode == "unsteering":
            unsteered = generate_text(
                model,
                tokenizer,
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                device=device,
                steering_vector=None,
                steer_layer=None,
                alpha=0.0,
            )
            record.update(
                {
                    "unsteered_text": unsteered,
                    "unsteered_length": len(unsteered.split()),
                }
            )
        else:
            steered = generate_text(
                model,
                tokenizer,
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                device=device,
                steering_vector=steering_vector_for_generation,
                steer_layer=steer_layer,
                alpha=alpha_value,
                remove_at_final_norm=args.remove_at_final_norm,
                final_norm_module=final_norm_module,
                final_norm_vector=final_norm_vector_for_generation,
                final_norm_alpha=final_norm_alpha_value,
            )
            record.update(
                {
                    "steered_text": steered,
                    "steered_length": len(steered.split()),
                }
            )

        records.append(record)
        sample_path = os.path.join(samples_dir, f"sample_{idx:04d}.json")
        with open(sample_path, "w") as f:
            json.dump(record, f, indent=2)

    unsteered_lengths = [
        r["unsteered_length"] for r in records if "unsteered_length" in r
    ]
    steered_lengths = [r["steered_length"] for r in records if "steered_length" in r]
    unsteered_avg_len = (
        sum(unsteered_lengths) / len(unsteered_lengths) if unsteered_lengths else None
    )
    steered_avg_len = (
        sum(steered_lengths) / len(steered_lengths) if steered_lengths else None
    )

    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "sample_size": len(records),
        "seed": args.seed,
        "steer_vector": vector_path,
        "steer_layer": steer_layer,
        "alpha_mode": args.alpha_mode,
        "alpha": args.alpha,
        "alpha_zbar": alpha_zbar,
        "alpha_mean": (sum(alpha_values) / len(alpha_values)) if alpha_values else None,
        "final_norm_layer": final_norm_layer,
        "final_norm_alpha_layer": final_norm_alpha_layer,
        "final_norm_alpha_zbar": alpha_zbar_final_norm,
        "final_norm_alpha_mean": (
            (sum(final_norm_alpha_values) / len(final_norm_alpha_values))
            if final_norm_alpha_values
            else None
        ),
        "remove_at_final_norm": args.remove_at_final_norm,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        "run_mode": args.run_mode,
        "run_tag": run_tag,
        "unsteered_avg_length": unsteered_avg_len,
        "steered_avg_length": steered_avg_len,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    is_last_layer = steer_layer == (MODEL_LAYERS[args.model] - 1)
    if args.remove_at_final_norm:
        generation_dir = os.path.join(run_dir, "finalnorm")
    else:
        generation_dir = run_dir
    os.makedirs(generation_dir, exist_ok=True)
    generation_path = os.path.join(generation_dir, "generation.json")
    with open(generation_path, "w") as f:
        json.dump(summary, f, indent=2)
    if args.run_mode == "steering" and is_last_layer:
        prefixed_path = os.path.join(generation_dir, "last_layer_generation.json")
        with open(prefixed_path, "w") as f:
            json.dump(summary, f, indent=2)

    logger.info("Saved generation results to {}", run_dir)


if __name__ == "__main__":
    main()
