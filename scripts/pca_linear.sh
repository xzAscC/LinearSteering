#!/usr/bin/env bash
set -euo pipefail

# Define models
models=("Qwen/Qwen3-14B" "meta-llama/Llama-3.1-8B-Instruct" "allenai/Olmo-3-1025-7B")

models=("Qwen/Qwen3-1.7B" "google/gemma-2-2b")

# Layers: 10 evenly spaced layers from 0-100
LAYERS="0, 10,20,30,40,50,60,70,80,90,100"

# Run analysis
for model in "${models[@]}"; do
    echo "Running linearity analysis for $model..."
    
    # 1. Without removing concept vector
    echo "  - Mode: Standard (Keep Concept Vector)"
    uv run src/pca_linear.py --model "$model" --layers "$LAYERS" --alpha_points 100
    
    # 2. With removing concept vector
    echo "  - Mode: Remove Concept Vector"
    uv run src/pca_linear.py --model "$model" --layers "$LAYERS" --alpha_points 100 --remove_concept_vector
done

# Generate plots
echo "Generating plots..."
uv run src/plot/plot_linear.py

echo "Done!"
