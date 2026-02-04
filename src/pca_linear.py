import argparse
import torch
import transformers
import os
import torch.nn.functional as F
from utils import (
    MODEL_LAYERS,
    CONCEPT_CATEGORIES,
    set_seed,
    run_model_with_steering,
    hidden_to_flat,
    get_model_name_for_path,
    parse_layers_to_run,
)
import torch.multiprocessing as mp
import math
from extract_concepts import load_concept_datasets
from loguru import logger
from tqdm import tqdm

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
        
        eigenvalues = S ** 2
        total_variance = eigenvalues.sum(dim=-1)
        
        # Avoid division by zero
        epsilon = 1e-12
        valid_mask = total_variance > epsilon
        
        # 1. Linearity Score (PC1 / Total)
        scores = torch.ones_like(total_variance) # Default 1.0
        if valid_mask.any():
            scores[valid_mask] = eigenvalues[valid_mask][:, 0] / total_variance[valid_mask]
            
        # 2. n_95
        n_95 = torch.ones_like(total_variance) # Default 1.0
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
        self.score_sq_sum += (scores ** 2).sum().item()
        self.n95_sum += n_95.sum().item()
        self.n95_sq_sum += (n_95 ** 2).sum().item()
        self.count += scores.shape[0]

    def finalize(self):
        if self.count == 0:
            return {
                "mean_score": 1.0, "std_score": 0.0,
                "mean_n95": 1.0, "std_n95": 0.0
            }
            
        mean_score = self.score_sum / self.count
        var_score = max(0.0, self.score_sq_sum / self.count - mean_score**2)
        
        mean_n95 = self.n95_sum / self.count
        var_n95 = max(0.0, self.n95_sq_sum / self.count - mean_n95**2)
        
        return {
            "mean_score": mean_score,
            "std_score": var_score**0.5,
            "mean_n95": mean_n95,
            "std_n95": var_n95**0.5
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Model name to process. If not specified, process all models. Available: {list(MODEL_LAYERS.keys())}",
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
    parser.add_argument("--alpha_max", type=float, default=1e6)
    parser.add_argument("--alpha_points", type=int, default=200) # Fewer points than smoothness
    
    parser.add_argument(
        "--layers",
        type=str,
        default="50",
        help="Comma-separated percentages or layer indices.",
    )
    parser.add_argument(
        "--remove_concept_vector",
        action="store_true",
        help="Remove concept vector from hidden states",
    )
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    logger.add("logs/linear.log")
    logger.info(f"args: {args}")
    set_seed(args.seed)
    
    models_to_process = (
        [(args.model, MODEL_LAYERS[args.model])]
        if args.model is not None
        else list(MODEL_LAYERS.items())
    )
    
def worker_process(rank, world_size, model_full_name, task_queue, args, max_layers):
    """
    Worker process to handle concepts from a shared queue on a specific GPU.
    """
    # Setup device
    if torch.cuda.is_available():
        device = f"cuda:{rank}"
    else:
        device = "cpu"
    
    logger.info(f"Rank {rank}/{world_size}: Started. Device: {device}")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    # Load model
    logger.info(f"Rank {rank}: Loading model {model_full_name}")
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
        logger.error(f"Rank {rank}: Failed to load model {model_full_name}: {e}")
        return

    model_name = get_model_name_for_path(model_full_name)
    os.makedirs(f"assets/linear/{model_name}", exist_ok=True)
    
    layers_to_run = parse_layers_to_run(args.layers, max_layers)
    
    while True:
        try:
            task = task_queue.get(timeout=1.0)
        except Exception:
            # Queue likely empty or timeout
            break
            
        if task is None:
            # Sentinel received
            break
            
        concept_category_name, concept_category_config = task
        
        logger.info(f"Rank {rank}: Processing {concept_category_name}")
        
        concept_vectors_path = f"assets/concept_vectors/{model_name}/{concept_category_name}.pt"
        if not os.path.exists(concept_vectors_path):
            logger.warning(f"Rank {rank}: Concept vectors not found for {concept_category_name} in {model_name}. Skipping.")
            continue
            
        concept_vectors = torch.load(concept_vectors_path) # Load to CPU initially
        
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
            if total_tokens + prompt_length > args.max_tokens and len(selected_prompts) > 0:
                break
            selected_prompts.append(prompt)
            total_tokens += prompt_length
            if total_tokens >= args.max_tokens:
                break
        
        input_ids = tokenizer(
            selected_prompts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=args.max_tokens,
        ).to(device).input_ids

        # Generate random vector for comparison
        random_vector_dir = f"assets/linear/{model_name}/random_vectors"
        # Race condition on makedirs if multiple GPUs running? 
        # But makedirs(exist_ok=True) is atomic enough usually.
        os.makedirs(random_vector_dir, exist_ok=True)
        random_vector_path = f"{random_vector_dir}/{concept_category_name}.pt"
        
        vector_dim = concept_vectors.shape[1]
        
        # Lock file checking or rely on atomic nature or pre-generation?
        # Ideally random vectors should be pre-generated.
        # Here we just try to load, if not exist, generate.
        # Potential race condition if two ranks process same concept?
        # But we split concepts, so each concept is processed by ONLY one rank.
        # So no race condition on this file.
        
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
        for vector_type, vector_source in [("concept", concept_vectors), ("random", random_vector)]:
            results = {}
            
            # Using tqdm on specific position or just log? 
            # Tqdm across processes messes up stdout. Disable or simple log.
            iterator = layers_to_run
            if rank == 0:
                 iterator = tqdm(layers_to_run, desc=f"Linearity ({concept_category_name}, {vector_type})")
            
            for layer_idx in iterator:
                if vector_type == "concept":
                    steering_vector = concept_vectors[layer_idx, :]
                else:
                    steering_vector = random_vector
                
                # Create alphas range
                alphas = torch.logspace(
                        float(torch.log10(torch.tensor(args.alpha_min))),
                        float(torch.log10(torch.tensor(args.alpha_max))),
                        steps=args.alpha_points
                ).tolist()

                aggregator = PCAStatisticsAggregator()
                
                # Process in batches of prompts to save memory
                batch_size = 1 # Process one prompt (or small batch) at a time
                
                # Split input_ids into chunks
                total_prompts = input_ids.shape[0]
                
                for i in range(0, total_prompts, batch_size):
                    input_ids_batch = input_ids[i:i+batch_size]
                    
                    batch_collected_deltas = []
                    
                    # Handle baseline (alpha=0)
                    h_ref_batch = run_model_with_steering(
                        model=model,
                        input_ids=input_ids_batch,
                        steering_vector=steering_vector,
                        layer_idx=layer_idx,
                        alpha_value=0.0,
                        device=device
                    )
                    h_ref_flat_batch = hidden_to_flat(h_ref_batch, target_dtype=torch.float32)
                    
                    if 0.0 not in alphas:
                        batch_collected_deltas.append(torch.zeros_like(h_ref_flat_batch))
                    
                    for alpha in alphas:
                        h_batch = run_model_with_steering(
                            model=model,
                            input_ids=input_ids_batch,
                            steering_vector=steering_vector,
                            layer_idx=layer_idx,
                            alpha_value=alpha,
                            device=device
                        )
                        h_flat_batch = hidden_to_flat(h_batch, target_dtype=torch.float32)
                        
                        # Calculate delta
                        if args.remove_concept_vector:
                                steering_vec_device = steering_vector.to(device=h_batch.device, dtype=h_batch.dtype)
                                # Remove linear component from h first
                                h_flat_corrected = hidden_to_flat(h_batch - alpha * steering_vec_device, target_dtype=torch.float32)
                                # Then subtract baseline
                                delta = h_flat_corrected - h_ref_flat_batch
                        else:
                                delta = h_flat_batch - h_ref_flat_batch
                        
                        batch_collected_deltas.append(delta)
                    
                    # Stack deltas for this batch [N_steps, N_tokens_in_batch, Hidden]
                    batch_trajectory = torch.stack(batch_collected_deltas)
                    
                    # Update aggregator
                    aggregator.update(batch_trajectory)
                    
                    del batch_trajectory, batch_collected_deltas, h_ref_batch, h_ref_flat_batch

                stats = aggregator.finalize()
                
                results[layer_idx] = {
                    "mean_score": stats["mean_score"],
                    "std_score": stats["std_score"],
                    "n_components_95_mean": stats["mean_n95"],
                    "n_components_95_std": stats["std_n95"],
                    "alphas": alphas
                }
                logger.info(f"Rank {rank}: Linearity for {concept_category_name} in {model_name} at layer {layer_idx}: {stats['mean_score']:.4f} (n95: {stats['mean_n95']:.2f})")
            
            # Save results
            suffix = "_remove" if args.remove_concept_vector else ""
            save_path = f"assets/linear/{model_name}/linearity_{concept_category_name}_{vector_type}{suffix}.pt"
            torch.save(
                {
                    "model": model_full_name,
                    "concept_category": concept_category_name,
                    "vector_type": vector_type,
                    "results": results,
                },
                save_path
            )
    
    del model
    del tokenizer
    torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Model name to process. If not specified, process all models. Available: {list(MODEL_LAYERS.keys())}",
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
    parser.add_argument("--alpha_points", type=int, default=200) # Fewer points than smoothness
    
    parser.add_argument(
        "--layers",
        type=str,
        default="50",
        help="Comma-separated percentages or layer indices.",
    )
    parser.add_argument(
        "--remove_concept_vector",
        action="store_true",
        help="Remove concept vector from hidden states",
    )
    args = parser.parse_args()

    # Need to set start method for spawn
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    os.makedirs("logs", exist_ok=True)
    logger.add("logs/linear.log")
    logger.info(f"args: {args}")
    set_seed(args.seed)
    
    models_to_process = (
        [(args.model, MODEL_LAYERS[args.model])]
        if args.model is not None
        else list(MODEL_LAYERS.items())
    )
    
    num_gpus = torch.cuda.device_count()
    logger.info(f"Detected {num_gpus} GPUs.")
    
    all_concepts = list(CONCEPT_CATEGORIES.items())
    
    for model_full_name, max_layers in models_to_process:
        logger.info(f"Processing model: {model_full_name}")
        
        if num_gpus > 1:
            # Create a queue and populate it with tasks
            task_queue = mp.Queue()
            for concept in all_concepts:
                task_queue.put(concept)
            
            # Add sentinels
            for _ in range(num_gpus):
                task_queue.put(None)
                
            logger.info(f"Spawning {num_gpus} workers for {len(all_concepts)} concepts...")
            mp.spawn(
                worker_process,
                args=(num_gpus, model_full_name, task_queue, args, max_layers),
                nprocs=num_gpus,
                join=True
            )
        else:
            # Single process mode
            # Mock a queue for consistency or just call logic?
            # Easiest to just mock Queue
            task_queue = mp.Queue()
            for concept in all_concepts:
                task_queue.put(concept)
            task_queue.put(None)
            
            worker_process(0, 1, model_full_name, task_queue, args, max_layers)

if __name__ == "__main__":
    main()
