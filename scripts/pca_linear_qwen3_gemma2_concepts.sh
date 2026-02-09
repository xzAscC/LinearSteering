#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/pca_linear_qwen3_gemma2_concepts.sh
#   CONCEPTS="steering_safety,steering_detectable_format_json_format" bash scripts/pca_linear_qwen3_gemma2_concepts.sh
#   TEST_SIZE=32 MAX_TOKENS=800 ALPHA_POINTS=120 bash scripts/pca_linear_qwen3_gemma2_concepts.sh

MODELS=(
  "Qwen/Qwen3-1.7B"
  "google/gemma-2-2b"
)

IFS=',' read -r -a CONCEPT_LIST <<< "${CONCEPTS:-steering_safety,steering_detectable_format_json_format}"
CONCEPTS_ARG=""
for concept in "${CONCEPT_LIST[@]}"; do
  concept_trimmed="$(printf '%s' "$concept" | xargs)"
  if [[ -n "$concept_trimmed" ]]; then
    if [[ -z "$CONCEPTS_ARG" ]]; then
      CONCEPTS_ARG="$concept_trimmed"
    else
      CONCEPTS_ARG+=",${concept_trimmed}"
    fi
  fi
done

if [[ -z "$CONCEPTS_ARG" ]]; then
  echo "No valid concepts in CONCEPTS."
  exit 1
fi

TEST_SIZE="${TEST_SIZE:-16}"
MAX_TOKENS="${MAX_TOKENS:-500}"
LAYERS="${LAYERS:-auto:8}"
STEER_LAYER="${STEER_LAYER:-0}"
ALPHA_MIN="${ALPHA_MIN:-1}"
ALPHA_MAX="${ALPHA_MAX:-100000}"
ALPHA_POINTS="${ALPHA_POINTS:-200}"
REMOVE_CONCEPT_VECTOR="${REMOVE_CONCEPT_VECTOR:-true}"

echo "Models: ${MODELS[*]}"
echo "Concepts: ${CONCEPT_LIST[*]}"
echo "Hook points: auto (Qwen3/Gemma2 model-specific defaults)"

for model in "${MODELS[@]}"; do
  echo ""
  echo "Running model=$model concepts=$CONCEPTS_ARG"

  cmd=(
    uv run python src/pca_linear.py
    --model "$model"
    --concepts "$CONCEPTS_ARG"
    --test_size "$TEST_SIZE"
    --max_tokens "$MAX_TOKENS"
    --layers "$LAYERS"
    --steer_layer "$STEER_LAYER"
    --hook_points auto
    --alpha_min "$ALPHA_MIN"
    --alpha_max "$ALPHA_MAX"
    --alpha_points "$ALPHA_POINTS"
  )

  if [[ "$REMOVE_CONCEPT_VECTOR" == "true" ]]; then
    cmd+=(--remove_concept_vector)
  else
    cmd+=(--no-remove_concept_vector)
  fi

  "${cmd[@]}"
done

echo ""
echo "All runs finished. Results are saved in assets/linear/<model_name>/"
