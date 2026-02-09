#!/usr/bin/env bash
set -euo pipefail

# Quick smoke check for src/pca_linear.py output quality.
# Usage:
#   bash scripts/quick_verify_pca_linear.sh
# Optional env overrides:
#   MODEL=Qwen/Qwen3-1.7B CONCEPT=steering_safety STEER_LAYER=0
#   MODEL=Qwen/Qwen3-1.7B CONCEPT=steering_random_direction STEER_LAYER=0

MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
CONCEPT="${CONCEPT:-steering_safety}"
STEER_LAYER="${STEER_LAYER:-0}"
HOOK_POINTS="${HOOK_POINTS:-block_out}"
LAYERS="${LAYERS:-auto:2}"
ALPHA_POINTS="${ALPHA_POINTS:-16}"
MAX_TOKENS="${MAX_TOKENS:-128}"
TEST_SIZE="${TEST_SIZE:-4}"

echo "[1/3] Run with concept-vector removal"
uv run python src/pca_linear.py \
  --model "$MODEL" \
  --concepts "$CONCEPT" \
  --steer_layer "$STEER_LAYER" \
  --hook_points "$HOOK_POINTS" \
  --layers "$LAYERS" \
  --alpha_points "$ALPHA_POINTS" \
  --max_tokens "$MAX_TOKENS" \
  --test_size "$TEST_SIZE" \
  --remove_concept_vector

echo "[2/3] Run without concept-vector removal"
uv run python src/pca_linear.py \
  --model "$MODEL" \
  --concepts "$CONCEPT" \
  --steer_layer "$STEER_LAYER" \
  --hook_points "$HOOK_POINTS" \
  --layers "$LAYERS" \
  --alpha_points "$ALPHA_POINTS" \
  --max_tokens "$MAX_TOKENS" \
  --test_size "$TEST_SIZE" \
  --no-remove_concept_vector

echo "[3/3] Print compact comparison"
VERIFY_MODEL="$MODEL" VERIFY_CONCEPT="$CONCEPT" uv run python - <<'PY'
import glob
import os
import torch

from utils import get_model_name_for_path

model = os.environ["VERIFY_MODEL"]
concept = os.environ["VERIFY_CONCEPT"]
model_name = get_model_name_for_path(model)
vector_type = "random" if concept == "steering_random_direction" else "concept"

base = f"assets/linear/{model_name}"

def latest_result_path(is_remove: bool) -> str:
    suffix = "_remove" if is_remove else ""
    tagged_pattern = f"{base}/pca_hooks_{concept}_{vector_type}_cfg_*{suffix}.pt"
    tagged_paths = sorted(glob.glob(tagged_pattern), key=os.path.getmtime, reverse=True)
    if tagged_paths:
        return tagged_paths[0]

    legacy_path = f"{base}/pca_hooks_{concept}_{vector_type}{suffix}.pt"
    return legacy_path

remove_path = latest_result_path(is_remove=True)
keep_path = latest_result_path(is_remove=False)

def summarize(path: str):
    data = torch.load(path, map_location="cpu")
    print(f"\n{path}")
    for layer_idx, hooks in sorted(data["results"].items()):
        for hook_name, stats in hooks.items():
            print(
                f"  layer={layer_idx:<3} hook={hook_name:<10} "
                f"pc1_mean={stats['mean_score']:.4f} "
                f"n95_mean={stats['n_components_95_mean']:.2f} "
                f"count={stats['count']} "
                f"degenerate={stats.get('degenerate_count', 'NA')}"
            )

summarize(remove_path)
summarize(keep_path)
PY

echo "Done. Focus on degenerate count and whether pc1_mean still collapses to 1.0"
