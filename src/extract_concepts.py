import argparse
import gc
import json
import os
from typing import Dict, List, Optional, Tuple, Any

import datasets
import torch
import transformer_lens
from loguru import logger
from tqdm import tqdm

from utils import (
    CONCEPT_CATEGORIES,
    MODEL_LAYERS,
    get_model_name_for_path,
    load_concept_datasets,
)


class DifferenceInMeans:
    def __init__(
        self,
        model: transformer_lens.HookedTransformer,
        positive_dataset: datasets.Dataset,
        negative_dataset: datasets.Dataset,
        layer: int,
        device: str,
        dataset_key: str,
        max_dataset_size: int = 300,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Used to calculate the concept vector using difference-in-means.

        Args:
            model (transformer_lens.HookedTransformer): The model to calculate the concept vector for.
            positive_dataset (datasets.Dataset): The positive dataset.
            negative_dataset (datasets.Dataset): The negative dataset.
            layer (int): The max layer index to calculate up to.
            device (str): The device to run on (e.g., 'cuda', 'cpu').
            dataset_key (str): The key in the dataset accessing the text (e.g., 'text' or 'prompt').
            max_dataset_size (int, optional): The maximum size of the dataset to use. Defaults to 300.
            dtype (torch.dtype, optional): The data type for tensors. Defaults to torch.bfloat16.
        """
        self.model = model
        self.positive_dataset = positive_dataset
        self.negative_dataset = negative_dataset
        self.layers = list(range(layer))
        self.device = device
        self.dataset_key = dataset_key
        self.max_dataset_size = max_dataset_size
        self.dtype = dtype

    def get_concept_vectors(
        self, save_path: str, is_save: bool = False
    ) -> torch.Tensor:
        """Calculate the concept vectors using difference-in-means.

        Args:
            save_path (str): The path to save the concept vectors.
            is_save (bool, optional): Whether to save the concept vectors. Defaults to False.

        Returns:
            torch.Tensor: The concept vectors of shape [num_layers, d_model].
        """
        model_dimension = self.model.cfg.d_model
        layer_length = len(self.layers)
        logger.info(
            f"Computing vectors for {layer_length} layers, dim {model_dimension}"
        )

        positive_concept_vector = torch.zeros(
            layer_length, model_dimension, device=self.device, dtype=self.dtype
        )
        negative_concept_vector = torch.zeros(
            layer_length, model_dimension, device=self.device, dtype=self.dtype
        )

        positive_token_length = 0
        negative_token_length = 0

        # Process Positive Dataset
        positive_dataset_size = min(len(self.positive_dataset), self.max_dataset_size)
        logger.info(
            f"Processing positive dataset ({positive_dataset_size} examples)..."
        )

        for i, example in tqdm(
            enumerate(self.positive_dataset), total=positive_dataset_size
        ):
            if i >= self.max_dataset_size:
                break

            torch.cuda.empty_cache()
            gc.collect()

            context = example[self.dataset_key]
            _, positive_cache = self.model.run_with_cache(context)

            for layer_idx in self.layers:
                # Shape: [seq_len, d_model]
                positive_hidden_state = positive_cache[
                    f"blocks.{layer_idx}.hook_resid_post"
                ].reshape(-1, model_dimension)

                # Sum over sequence length
                positive_concept_vector[layer_idx] += positive_hidden_state.sum(dim=0)

                if layer_idx == 0:
                    current_token_length = positive_hidden_state.shape[0]
                    positive_token_length += current_token_length

        # Process Negative Dataset
        negative_dataset_size = min(len(self.negative_dataset), self.max_dataset_size)
        logger.info(
            f"Processing negative dataset ({negative_dataset_size} examples)..."
        )

        for i, example in tqdm(
            enumerate(self.negative_dataset), total=negative_dataset_size
        ):
            if i >= self.max_dataset_size:
                break

            torch.cuda.empty_cache()
            gc.collect()

            context = example[self.dataset_key]
            # We need to run up to the max layer we are interested in.
            # stop_at_layer is 1-indexed for 'blocks.{i}'?
            # run_with_cache(stop_at_layer=L) returns cache for layers 0..L-1
            # But the loop iterates 'self.layers', which is range(layer).
            # The max index in self.layers is layer-1.
            # So stop_at_layer=layer should be sufficient?
            # Original code used: stop_at_layer=layer + 1 where `layer` was actually loop variable?
            # Ah, wait, original code: `for layer in self.layers` ...
            # Inside the second loop (negative), it had a `layer` variable collision?
            # "blocks.{layer}.hook_resid_post".
            # The `stop_at_layer` in original code used `layer + 1`.
            # If `layer` comes from `range(layer)` (the argument), then `layer` is an int.
            # But the original code had `for (concept_category_name...)` -> `concept_vector(...)` -> `get_concept_vectors(...)`
            # Inside `DifferenceInMeans`: `self.layers = list(range(layer))`
            # Loop for negative: `_, negative_cache = self.model.run_with_cache(context, stop_at_layer=layer + 1)`
            # WAIT. In the original code `for layer in self.layers` was AFTER the run_with_cache call.
            # So `layer` in `stop_at_layer=layer + 1` was referring to the class attribute or init arg `layer`?
            # It was inside `__init__`, `layer` is an int? NO.
            # In `get_concept_vectors`, `layer` argument does not exist. `self.layers` exists.
            # Where did `layer` come from in the original negative loop?
            # "_, negative_cache = self.model.run_with_cache(context, stop_at_layer=layer + 1)"
            # It seems `layer` was NOT defined in `get_concept_vectors` scope before that usage, unless it leaked from the previous loop `for layer in self.layers:`?
            # Python loop variables leak. So `layer` would be the last value of `self.layers`.
            # That looks like a BUG in the original code! providing stop_at_layer for the last layer only?
            # Actually, if we want cache for all layers 0..N, we need stop_at_layer=N+1?
            # If `self.layers` is `range(layer_arg)`, then max val is layer_arg-1.
            # We probably want to run up to `len(self.layers)` or `self.layers[-1] + 1`.

            # Let's fix this properly.
            max_layer_idx = self.layers[-1]

            _, negative_cache = self.model.run_with_cache(
                context,
                stop_at_layer=max_layer_idx + 2,  # Safety, ensures we get the block.
            )

            for layer_idx in self.layers:
                negative_hidden_state = negative_cache[
                    f"blocks.{layer_idx}.hook_resid_post"
                ].reshape(-1, model_dimension)

                negative_concept_vector[layer_idx] += negative_hidden_state.sum(dim=0)

                if layer_idx == 0:
                    current_token_length = negative_hidden_state.shape[0]
                    negative_token_length += current_token_length

        # Averaging
        if positive_token_length > 0:
            positive_concept_vector /= positive_token_length
        if negative_token_length > 0:
            negative_concept_vector /= negative_token_length

        concept_diff = positive_concept_vector - negative_concept_vector

        # Normalize the difference vector
        concept_diff = torch.nn.functional.normalize(concept_diff, dim=1)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(concept_diff, save_path)
        norm = concept_diff.norm(dim=1)
        # Save norm to json
        if is_save:
            norm_save_path = save_path.replace(".pt", "_norm.json")
            with open(norm_save_path, "w") as f:
                json.dump(norm.tolist(), f)
        logger.info(f"Concept diff norm: {norm}")
        logger.info(f"Concept vector shape: {concept_diff.shape}")

        return concept_diff


def get_concept_vectors(
    model: transformer_lens.HookedTransformer,
    positive_dataset: datasets.Dataset,
    negative_dataset: datasets.Dataset,
    layer: int,
    device: str,
    dataset_key: str,
    methods: str,
    save_path: str,
    max_dataset_size: int = 300,
) -> torch.Tensor:
    """Wrapper to get concept vectors using the specified method.

    Args:
        model: The model.
        positive_dataset: Positive dataset.
        negative_dataset: Negative dataset.
        layer: Max layer index.
        device: Device.
        dataset_key: Key for text in dataset.
        methods: Method name (only 'difference-in-means' supported).
        save_path: where to save.
        max_dataset_size: max examples.

    Returns:
        torch.Tensor: Concept vectors.
    """
    if methods == "difference-in-means":
        difference_in_means = DifferenceInMeans(
            model,
            positive_dataset,
            negative_dataset,
            layer=layer,
            device=device,
            dataset_key=dataset_key,
            max_dataset_size=max_dataset_size,
        )
        concept_vector = difference_in_means.get_concept_vectors(
            save_path=save_path,
            is_save=True,
        )
    else:
        raise ValueError(f"Invalid method: {methods}")
    return concept_vector


def obtain_concept_vector(
    model: transformer_lens.HookedTransformer,
    max_layers: int,
    concept_category_name: str,
    concept_category_config: Dict[str, Any],
    max_dataset_size: int = 300,
    model_name: str = "Qwen/Qwen3-1.7B",
    methods: str = "difference-in-means",
) -> Tuple[torch.Tensor, datasets.Dataset, str]:
    """Obtain concept vector for a given concept category.

    Args:
        model: The model to obtain the concept vector from
        max_layers: Maximum number of layers to process
        concept_category_name: Name of the concept category
        concept_category_config: Configuration dictionary for the concept category
        max_dataset_size: Maximum size of the dataset to use
        model_name: Name of the model (full name, e.g., "google/gemma-2-2b")
        methods: Method to use for obtaining concept vectors

    Returns:
        Tuple of (concept_vector, positive_dataset, dataset_key)
    """
    # Load datasets using the unified loading system
    positive_dataset, negative_dataset, dataset_key = load_concept_datasets(
        concept_category_name, concept_category_config
    )

    # Set up save path with safe model name
    safe_model_name = get_model_name_for_path(model_name)
    save_dir = f"assets/concept_vectors/{safe_model_name}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{concept_category_name}.pt"

    # Get concept vectors
    concept_vector = get_concept_vectors(
        model=model,
        positive_dataset=positive_dataset,
        negative_dataset=negative_dataset,
        layer=max_layers,
        device="cuda",
        dataset_key=dataset_key,
        methods=methods,
        save_path=save_path,
        max_dataset_size=max_dataset_size,
    )

    # Generate and save 3 random directions
    logger.info("Generating and saving 3 random directions...")
    for i in range(15):
        random_direction = torch.randn_like(concept_vector)
        random_direction = torch.nn.functional.normalize(random_direction, dim=1)
        # Use a generic name within the model folder
        random_save_path = os.path.join(save_dir, f"random_direction_{i + 1}.pt")
        torch.save(random_direction, random_save_path)

    torch.cuda.empty_cache()
    gc.collect()

    return concept_vector, positive_dataset, dataset_key


def concept_vector(
    model_name: Optional[str] = None,
    concept_category: Optional[str] = None,
    method: str = "difference-in-means",
    max_dataset_size: int = 300,
) -> None:
    """Main function to obtain concept vectors.

    Args:
        model_name: The model name (if None, process all models).
        concept_category: The concept category (if None, process all).
        method: Method to use.
        max_dataset_size: Max dataset size.
    """
    logger.info("Obtaining concept vectors...")
    os.makedirs("assets/concept_vectors", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    logger.add("logs/concept_vectors.log")

    device = "cuda"
    dtype = torch.bfloat16

    # Determine which models and concepts to process
    models_to_process = (
        [model_name]
        if model_name and model_name in MODEL_LAYERS
        else list(MODEL_LAYERS.keys())
    )

    concepts_to_process = (
        [(concept_category, CONCEPT_CATEGORIES[concept_category])]
        if concept_category and concept_category in CONCEPT_CATEGORIES
        else list(CONCEPT_CATEGORIES.items())
    )

    for model_name_iter in models_to_process:
        logger.info(f"Processing model: {model_name_iter}")

        try:
            model = transformer_lens.HookedTransformer.from_pretrained(
                model_name_iter, device=device, dtype=dtype, trust_remote_code=True
            )
        except Exception as e:
            logger.error(f"Failed to load model {model_name_iter}: {e}")
            continue

        max_layers = model.cfg.n_layers

        for concept_category_name, concept_category_config in concepts_to_process:
            logger.info(f"Processing concept: {concept_category_name}")
            try:
                obtain_concept_vector(
                    model,
                    max_layers,
                    concept_category_name,
                    concept_category_config,
                    max_dataset_size,
                    model_name_iter,
                    method,
                )
            except Exception as e:
                logger.error(
                    f"Error processing concept {concept_category_name} for model {model_name_iter}: {e}"
                )

        del model
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Obtain concept vectors")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Model name to process. Choices: {list(MODEL_LAYERS.keys())}",
    )
    parser.add_argument(
        "--concept_category",
        type=str,
        default=None,
        help=f"Concept category to process. Choices: {list(CONCEPT_CATEGORIES.keys())}",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="difference-in-means",
        choices=["difference-in-means"],
        help="Method to use for obtaining concept vectors",
    )
    parser.add_argument(
        "--max_dataset_size",
        type=int,
        default=30,
        help="Maximum size of the dataset to use",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.model is not None and args.model not in MODEL_LAYERS:
        parser.error(
            f"Invalid model: {args.model}. Must be one of {list(MODEL_LAYERS.keys())}"
        )
    if (
        args.concept_category is not None
        and args.concept_category not in CONCEPT_CATEGORIES
    ):
        parser.error(
            f"Invalid concept_category: {args.concept_category}. Must be one of {list(CONCEPT_CATEGORIES.keys())}"
        )

    concept_vector(
        model_name=args.model,
        concept_category=args.concept_category,
        method=args.method,
        max_dataset_size=args.max_dataset_size,
    )
