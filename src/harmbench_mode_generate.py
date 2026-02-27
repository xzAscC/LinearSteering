import argparse
import json
import os
from glob import glob

import datasets
import torch
import transformers

from probe_utils import load_vector
from utils import _get_layers_container
from utils import get_model_name_for_path
from utils import make_steering_hook
from utils import set_seed


def build_prompt(tokenizer, prompt: str) -> dict[str, torch.Tensor]:
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = prompt
    return tokenizer(text, return_tensors="pt", truncation=True, padding=False)


def extract_prompt(item: dict) -> str:
    for key in ["prompt", "instruction", "behavior", "text"]:
        if key in item and item[key]:
            return str(item[key])
    raise ValueError("No prompt field found")


def resolve_vector_path(args, model_name: str) -> str | None:
    if args.mode == "no_steering":
        return None

    if args.mode == "dim_100_tokens":
        if args.dim_100_vector is not None:
            return args.dim_100_vector
        pattern = os.path.join(
            "assets",
            "concept_vectors",
            model_name,
            "pku_pos_neg_center_token_layer*.pt",
        )
        candidates = [path for path in glob(pattern) if os.path.isfile(path)]
        if not candidates:
            raise FileNotFoundError(f"No 100-token vector found: {pattern}")
        candidates.sort(key=os.path.getmtime, reverse=True)
        return candidates[0]

    if args.dim_all_vector is not None:
        return args.dim_all_vector
    return os.path.join(
        "assets",
        "concept_vectors",
        model_name,
        "pku_all_token_dim_pku_saferlhf_pos_neg_100x2_minimal_prompt.pt",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HarmBench outputs by mode")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--dataset", type=str, default="walledai/HarmBench")
    parser.add_argument("--dataset_config", type=str, default="standard")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--sample_size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument(
        "--mode",
        type=str,
        default="no_steering",
        choices=["no_steering", "dim_100_tokens", "dim_all_tokens"],
    )
    parser.add_argument("--steer_layer", type=int, default=17)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--dim_100_vector", type=str, default=None)
    parser.add_argument("--dim_all_vector", type=str, default=None)
    parser.add_argument(
        "--output_dir", type=str, default="logs/harmbench_mode_generate"
    )
    args = parser.parse_args()

    set_seed(args.seed)
    model_name = get_model_name_for_path(args.model)
    vector_path = resolve_vector_path(args, model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=device,
        dtype=dtype,
        trust_remote_code=True,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    steering_vector = None
    if vector_path is not None:
        vector = load_vector(vector_path)
        steering_vector = vector[args.steer_layer] if vector.ndim == 2 else vector

    dataset = datasets.load_dataset(args.dataset, args.dataset_config, split=args.split)
    if args.sample_size < len(dataset):
        g = torch.Generator().manual_seed(args.seed)
        idx = torch.randperm(len(dataset), generator=g)[: args.sample_size].tolist()
        items = [dataset[i] for i in idx]
    else:
        items = [dataset[i] for i in range(len(dataset))]

    layer_tag = (
        f"layer_{args.steer_layer}" if steering_vector is not None else "no_layer"
    )
    out_dir = os.path.join(args.output_dir, model_name, args.mode, layer_tag)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "responses.json")

    outputs = []
    for item in items:
        prompt = extract_prompt(item)
        model_inputs = build_prompt(tokenizer, prompt)
        input_ids = model_inputs.input_ids.to(device)
        attention_mask = model_inputs.attention_mask.to(device)

        handles = []
        if steering_vector is not None:
            layers = _get_layers_container(model)
            handles.append(
                layers[args.steer_layer].register_forward_hook(
                    make_steering_hook(steering_vector, args.alpha, device=device)
                )
            )

        try:
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.temperature > 0.0,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.eos_token_id,
                )
        finally:
            for handle in handles:
                handle.remove()

        answer = tokenizer.decode(
            output_ids[0][input_ids.shape[1] :], skip_special_tokens=True
        )
        outputs.append({"problem": prompt, "answer": answer})

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "mode": args.mode,
                "steer_layer": args.steer_layer
                if steering_vector is not None
                else None,
                "vector_path": vector_path,
                "alpha": args.alpha if steering_vector is not None else 0.0,
                "responses": outputs,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(out_path)


if __name__ == "__main__":
    main()
