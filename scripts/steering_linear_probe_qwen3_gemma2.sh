#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/steering_linear_probe_qwen3_gemma2.sh
# Optional env overrides:
#   CONCEPT=steering_detectable_format_json_format ALPHA_MODE=avg_norm
#   ALPHA_SCALES=0.01,0.1,1,10 MAX_PROMPTS=2048 BATCH_SIZE=4

MODELS=(
  "Qwen/Qwen3-1.7B"
  "google/gemma-2-2b"
)

CONCEPT="${CONCEPT:-steering_detectable_format_json_format}"
RANDOM_DIRECTION="${RANDOM_DIRECTION:-random_direction_1}"
ALPHA_MODE="${ALPHA_MODE:-avg_norm}"
ALPHA_VALUES="${ALPHA_VALUES:-1,10,100,1000,10000}"
ALPHA_SCALES="${ALPHA_SCALES:-0.01,0.1,1}"
MAX_PROMPTS="${MAX_PROMPTS:-8196}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-200}"
LR="${LR:-1e-3}"
TEST_RATIO="${TEST_RATIO:-0.2}"
STEER_LAYERS="${STEER_LAYERS:-auto:6}"
HOOK_POINTS="${HOOK_POINTS:-auto}"
SEED="${SEED:-42}"

echo "Models: ${MODELS[*]}"
echo "Concept: ${CONCEPT}"
echo "Alpha mode: ${ALPHA_MODE}"

for model in "${MODELS[@]}"; do
  if [[ "$model" == "Qwen/Qwen3-1.7B" ]]; then
    probe_layers="auto:8"
  elif [[ "$model" == "google/gemma-2-2b" ]]; then
    probe_layers="auto:6"
  else
    echo "Unknown model: $model"
    exit 1
  fi

  echo ""
  echo "Running model=$model steer_layers=$STEER_LAYERS probe_layers=$probe_layers"

  uv run python src/steering_linear_probe.py \
    --model "$model" \
    --concepts "$CONCEPT" \
    --random_direction "$RANDOM_DIRECTION" \
    --alpha_mode "$ALPHA_MODE" \
    --alpha_values "$ALPHA_VALUES" \
    --alpha_scales "$ALPHA_SCALES" \
    --max_prompts "$MAX_PROMPTS" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --test_ratio "$TEST_RATIO" \
    --steer_layers "$STEER_LAYERS" \
    --probe_layers "$probe_layers" \
    --hook_points "$HOOK_POINTS" \
    --seed "$SEED"
done

echo ""
echo "All runs finished. Results are saved in assets/linear_probe/<model_name>/"
