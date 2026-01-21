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
from extract_concepts import load_concept_datasets
from loguru import logger
from tqdm import tqdm

def compute_linearity_score(trajectory_data):
    """
    Compute the linearity score based on PCA variance explained.
    
    Args:
        trajectory_data (torch.Tensor): [num_steps, num_samples, hidden_dim]
            The collected hidden states for each step and sample.
            
    Returns:
        tuple: (mean_score, std_score)
            - mean_score: Average linearity score (PC1 var / Total var) across samples
            - std_score: Standard deviation of linearity score across samples
    """
    # Permute to [num_samples, num_steps, hidden_dim] to treat each sample's trajectory independently
    # We want to measure if EACH trajectory is a line.
    X = trajectory_data.permute(1, 0, 2).float() # [N, T, D]
    
    # Center each trajectory independently
    X_mean = X.mean(dim=1, keepdim=True)
    X_centered = X - X_mean
    
    # Compute SVD for each sample
    # torch.linalg.svdvals is efficient and batched
    # S has shape [N, min(T, D)]
    S = torch.linalg.svdvals(X_centered)
    
    # Variance is proportional to squared singular values
    eigenvalues = S ** 2
    
    total_variance = eigenvalues.sum(dim=-1)
    pc1_variance = eigenvalues[:, 0]
    
    # Avoid division by zero
    epsilon = 1e-12
    valid_mask = total_variance > epsilon
    
    scores = torch.ones_like(total_variance) # Default to 1.0 if no variance
    scores[valid_mask] = pc1_variance[valid_mask] / total_variance[valid_mask]
    
    return scores.mean().item(), scores.std().item()

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
    parser.add_argument("--alpha_min", type=float, default=1e-3)
    parser.add_argument("--alpha_max", type=float, default=1e7)
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
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    for model_full_name, max_layers in models_to_process:
        logger.info(f"Processing model: {model_full_name}")
        model_name = get_model_name_for_path(model_full_name)
        os.makedirs(f"assets/linear/{model_name}", exist_ok=True)
        
        logger.info(f"Loading model: {model_full_name}")
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
            continue

        layers_to_run = parse_layers_to_run(args.layers, max_layers)
        
        for concept_category_name, concept_category_config in CONCEPT_CATEGORIES.items():
            concept_vectors_path = f"assets/concept_vectors/{model_name}/{concept_category_name}.pt"
            if not os.path.exists(concept_vectors_path):
                logger.warning(f"Concept vectors not found for {concept_category_name} in {model_name}. Skipping.")
                continue
                
            concept_vectors = torch.load(concept_vectors_path)
            
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
            os.makedirs(random_vector_dir, exist_ok=True)
            random_vector_path = f"{random_vector_dir}/{concept_category_name}.pt"
            
            vector_dim = concept_vectors.shape[1]
            if os.path.exists(random_vector_path):
                random_vector_data = torch.load(random_vector_path)
                # handle both formats if old exists
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
                
                for layer_idx in tqdm(layers_to_run, desc=f"Linearity ({concept_category_name}, {vector_type})"):
                    if vector_type == "concept":
                        steering_vector = concept_vectors[layer_idx, :]
                    else:
                        steering_vector = random_vector
                    
                    # Run alpha=0 first to get baseline
                    h_ref = run_model_with_steering(
                        model=model,
                        input_ids=input_ids,
                        steering_vector=steering_vector,
                        layer_idx=layer_idx,
                        alpha_value=0.0,
                        device=device
                    )
                    # Flatten [Batch, Seq, D] -> [Batch*Seq, D]
                    h_ref_flat = hidden_to_flat(h_ref, target_dtype=torch.float32)
                    
                    # Custom GPU Incremental PCA
                    class IncrementalPCA_GPU:
                        def __init__(self, device, hidden_dim):
                            self.device = device
                            self.n_samples = 0
                            self.mean = torch.zeros(hidden_dim, device=device, dtype=torch.float32)
                            # We maintain the scatter matrix (sum of squares), not covariance, to be numerically stable with updates
                            self.scatter_matrix = torch.zeros((hidden_dim, hidden_dim), device=device, dtype=torch.float32)

                        def partial_fit(self, batch):
                            # batch: [m, d]
                            m = batch.shape[0]
                            if m == 0:
                                return
                            
                            batch = batch.to(self.device, dtype=torch.float32)
                            batch_mean = batch.mean(dim=0)
                            
                            # Update scatter matrix based on batch
                            # S_batch = \sum (x - mu_batch)(x - mu_batch)^T
                            batch_centered = batch - batch_mean
                            batch_scatter = batch_centered.T @ batch_centered
                            
                            # Combine with existing stats
                            # S_new = S_old + S_batch + \frac{n * m}{n + m} (mu_old - mu_batch)(mu_old - mu_batch)^T
                            if self.n_samples > 0:
                                n = self.n_samples
                                delta_mean = self.mean - batch_mean
                                mean_correction = (n * m / (n + m)) * torch.outer(delta_mean, delta_mean)
                                
                                self.scatter_matrix += batch_scatter + mean_correction
                                self.mean = (n * self.mean + m * batch_mean) / (n + m)
                            else:
                                self.scatter_matrix = batch_scatter
                                self.mean = batch_mean
                                
                            self.n_samples += m

                        @property
                        def explained_variance_ratio(self):
                            if self.n_samples < 2:
                                return torch.tensor([1.0], device=self.device) # Only 1 point, effectively linear? Or undefined. Return 1.0.
                                
                            # Covariance = Scatter / (n - 1)
                            cov = self.scatter_matrix / (self.n_samples - 1)
                            
                            # Check for NaNs or Infs
                            if torch.isnan(cov).any() or torch.isinf(cov).any():
                                return torch.tensor([0.0], device=self.device)

                            # Eigen decomposition
                            # Use eigh for symmetric matrices (faster and more stable)
                            try:
                                eigenvalues = torch.linalg.eigvalsh(cov)
                                # Sort descending (eigvalsh returns ascending)
                                eigenvalues = eigenvalues.flip(dims=(0,))
                                
                                total_var = eigenvalues.sum()
                                if total_var == 0:
                                    return torch.tensor([1.0], device=self.device)
                                    
                                return eigenvalues / total_var
                            except RuntimeError:
                                return torch.tensor([0.0], device=self.device)

                    # Initialize GPU PCA
                    ipca = IncrementalPCA_GPU(device, concept_vectors.shape[1])
                    
                    # Create alphas range
                    # We can skip 0.0 in the loop since we have it as baseline (delta=0)
                    alphas = torch.logspace(
                         float(torch.log10(torch.tensor(args.alpha_min))),
                         float(torch.log10(torch.tensor(args.alpha_max))),
                         steps=args.alpha_points
                    ).tolist()
                    
                    # Ensure we fit at least the baseline (origin of deltas)
                    # For PCA of deltas, the origin (0 vector) IS a data point (t=0).
                    # h(0) - h_ref = 0.
                    # We should include this point.
                    # partial_fit expects [batch, d].
                    # We can craft a zero batch.
                    # Actually, let's just add 0.0 to alphas if we want to seamlessly process it, 
                    # but calculating h(0) again is redundant. 
                    # We know delta at alpha=0 is 0.
                    # So let's just feed a zero vector of size [N_prompts, d] to PCA.
                    
                    # Feed zeros for alpha=0 case (baseline)
                    # Shape is [batch_size (dataset size), hidden_dim]
                    # We processed inputs in one go for dataset selection?
                    # Wait, input_ids is [test_size, seq_len].
                    # hidden_to_flat makes it [test_size * seq_len, d]
                    # So we need to feed zeros of that shape.
                    if 0.0 not in alphas:
                        # Construct appropriate zero batch
                        # We need to know shape. run_model... at alpha=0 gives us that.
                        # h_ref_flat is [Batch*Seq, D].
                        # Just feed zeros of this shape.
                        ipca.partial_fit(torch.zeros_like(h_ref_flat))
                    
                    for alpha in alphas:
                        h = run_model_with_steering(
                            model=model,
                            input_ids=input_ids,
                            steering_vector=steering_vector,
                            layer_idx=layer_idx,
                            alpha_value=alpha,
                            device=device
                        )
                        
                        h_flat = hidden_to_flat(h, target_dtype=torch.float32)
                        
                        # Calculate delta
                        if args.remove_concept_vector:
                             steering_vec_device = steering_vector.to(device=h.device, dtype=h.dtype)
                             # Remove linear component from h first
                             h_flat_corrected = hidden_to_flat(h - alpha * steering_vec_device, target_dtype=torch.float32)
                             # Then subtract baseline (which is h(0) - 0*v = h(0))
                             delta = h_flat_corrected - h_ref_flat
                        else:
                             delta = h_flat - h_ref_flat
                        
                        # Partial fit
                        ipca.partial_fit(delta)
                    
                    # Score
                    # explained_variance_ratio is tensor
                    ratios = ipca.explained_variance_ratio
                    mean_score = ratios[0].item()
                    
                    # Calculate number of components for 95% variance
                    cumsum = torch.cumsum(ratios, dim=0)
                    # Find first index where sum >= 0.95 (indices are 0-based, so +1)
                    # We use 0.95 - 1e-6 to handle float precision issues where it might be slightly under
                    n_95_indices = (cumsum >= 0.95).nonzero(as_tuple=True)[0]
                    if len(n_95_indices) > 0:
                        n_components_95 = n_95_indices[0].item() + 1
                    else:
                        n_components_95 = len(ratios)

                    std_score = 0.0 # Undefined for global PCA
                    
                    results[layer_idx] = {
                        "mean_score": mean_score,
                        "std_score": std_score,
                        "n_components_95": n_components_95,
                        "alphas": alphas
                    }
                    logger.info(f"Linearity for {concept_category_name} in {model_name} at layer {layer_idx}: {mean_score:.4f}")
                
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
        
        # Cleanup
        del model
        del tokenizer
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
