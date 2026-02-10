#!/usr/bin/env bash
set -Eeuo pipefail

# Usage:
#   bash scripts/steering_linear_probe_qwen3_gemma2.sh
# Optional env overrides:
#   CONCEPTS=steering_detectable_format_json_format,steering_random_direction,steering_safety ALPHA_MODE=avg_norm
#   ALPHA_SCALES=0.01,0.1,1,10 MAX_PROMPTS=2048 BATCH_SIZE=4

MODELS=(
  "Qwen/Qwen3-1.7B"
  "google/gemma-2-2b"
)

CURRENT_MODEL=""
CURRENT_CONCEPT=""

on_error() {
  local exit_code="$1"
  local line_no="$2"
  local cmd="$3"
  echo ""
  echo "ERROR: script failed with exit code ${exit_code} at line ${line_no}" >&2
  echo "Command: ${cmd}" >&2
  if [[ -n "${CURRENT_MODEL}" ]]; then
    echo "Context: model=${CURRENT_MODEL}" >&2
  fi
  if [[ -n "${CURRENT_CONCEPT}" ]]; then
    echo "Context: concept=${CURRENT_CONCEPT}" >&2
  fi
}

trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

CONCEPTS="${CONCEPTS:-steering_detectable_format_json_format,steering_random_direction,steering_safety}"
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
echo "Concepts: ${CONCEPTS}"
echo "Alpha mode: ${ALPHA_MODE}"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 'uv' is not installed or not in PATH." >&2
  exit 1
fi

IFS=',' read -r -a concepts_to_run <<< "${CONCEPTS:-}" || true

valid_concepts=()
for raw_concept in "${concepts_to_run[@]}"; do
  concept="${raw_concept//[[:space:]]/}"
  if [[ -n "$concept" ]]; then
    valid_concepts+=("$concept")
  fi
done

if [[ "${#valid_concepts[@]}" -eq 0 ]]; then
  echo "ERROR: no valid concepts found in CONCEPTS='${CONCEPTS}'." >&2
  exit 1
fi

for model in "${MODELS[@]}"; do
  if [[ "$model" == "Qwen/Qwen3-1.7B" ]]; then
    probe_layers="auto:8"
  elif [[ "$model" == "google/gemma-2-2b" ]]; then
    probe_layers="auto:6"
  else
    echo "Unknown model: $model"
    exit 1
  fi

  for concept in "${valid_concepts[@]}"; do
    CURRENT_MODEL="$model"
    CURRENT_CONCEPT="$concept"

    echo ""
    echo "Running model=$model concept=$concept steer_layers=$STEER_LAYERS probe_layers=$probe_layers"

    log_dir="logs/steering_linear_probe"
    mkdir -p "$log_dir"
    log_file="${log_dir}/$(date +%Y%m%d_%H%M%S)_${model//\//_}_${concept}.log"
    echo "Log file: $log_file"

    cmd=(
      uv run python src/steering_linear_probe.py
      --model "$model"
      --concepts "$concept"
      --random_direction "$RANDOM_DIRECTION"
      --alpha_mode "$ALPHA_MODE"
      --alpha_values "$ALPHA_VALUES"
      --alpha_scales "$ALPHA_SCALES"
      --max_prompts "$MAX_PROMPTS"
      --batch_size "$BATCH_SIZE"
      --epochs "$EPOCHS"
      --lr "$LR"
      --test_ratio "$TEST_RATIO"
      --steer_layers "$STEER_LAYERS"
      --probe_layers "$probe_layers"
      --hook_points "$HOOK_POINTS"
      --seed "$SEED"
    )

    if ! "${cmd[@]}" 2>&1 | tee "$log_file"; then
      cmd_exit_code="${PIPESTATUS[0]}"
      echo "ERROR: run failed with exit code ${cmd_exit_code} (model=${model}, concept=${concept})." >&2
      echo "Inspect log: $log_file" >&2
      exit "$cmd_exit_code"
    fi
  done
done

echo ""
echo "All runs finished. Results are saved in assets/linear_probe/<model_name>/"
