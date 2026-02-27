#!/usr/bin/env bash
set -euo pipefail

MODEL="Qwen/Qwen3-1.7B"
DATASET="walledai/HarmBench"
DATASET_CONFIG="standard"
SPLIT="train"
SAMPLE_SIZE="20"

for MODE in no_steering dim_100_tokens dim_all_tokens; do
  for LAYER in 13 14 15 16 17; do
    uv run python src/harmbench_mode_generate.py \
      --model "$MODEL" \
      --dataset "$DATASET" \
      --dataset_config "$DATASET_CONFIG" \
      --split "$SPLIT" \
      --sample_size "$SAMPLE_SIZE" \
      --mode "$MODE" \
      --steer_layer "$LAYER" \
      --output_dir "logs/harmbench_mode_generate/layer_${LAYER}"
  done
done
