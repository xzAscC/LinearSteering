#!/usr/bin/env bash
set -euo pipefail

# Optional overrides via environment variables:
#   EXAMPLE_SIZES="10,20,50,100,200,400"
#   REPEATS="5"
#   REFERENCE_SIZE="all"
#   LAYERS="all"
#   SEED="42"
#   SAVE_VECTORS="1"  # set to 1 to pass --save_vectors

EXAMPLE_SIZES="${EXAMPLE_SIZES:-10,20,50,100,200,400}"
REPEATS="${REPEATS:-5}"
REFERENCE_SIZE="${REFERENCE_SIZE:-all}"
LAYERS="${LAYERS:-all}"
SEED="${SEED:-42}"
SAVE_VECTORS="${SAVE_VECTORS:-0}"

mapfile -t MODELS < <(
    uv run python -c "from src.utils import MODEL_LAYERS; [print(m) for m in MODEL_LAYERS.keys()]"
)

for model in "${MODELS[@]}"; do
    echo "Running concept vector stability for ${model}..."

    CMD=(
        uv run python src/concept_vector_stability.py
        --model "$model"
        --concept_category all
        --example_sizes "$EXAMPLE_SIZES"
        --repeats "$REPEATS"
        --reference_size "$REFERENCE_SIZE"
        --layers "$LAYERS"
        --seed "$SEED"
    )

    if [[ "$SAVE_VECTORS" == "1" ]]; then
        CMD+=(--save_vectors)
    fi

    "${CMD[@]}"

    echo "Plotting concept vector stability for ${model}..."
    uv run python src/plot_concept_vector_stability.py --model "$model" --concept_category all
done

echo "All models finished."
