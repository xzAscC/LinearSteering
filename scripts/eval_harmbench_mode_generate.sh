#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

INPUT_PATH=""
INPUT_DIR="logs/harmbench_mode_generate"
JUDGE_MODEL="${MOONSHOT_MODEL:-kimi-k2.5}"
JUDGE_BASE_URL="${MOONSHOT_BASE_URL:-https://api.moonshot.ai/v1}"
JUDGE_API_KEY="${MOONSHOT_API_KEY:-${KIMI_API_KEY:-}}"
JUDGE_TEMPERATURE="1.0"
JUDGE_TIMEOUT="30.0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input_path)
      INPUT_PATH="$2"
      shift 2
      ;;
    --input_dir)
      INPUT_DIR="$2"
      shift 2
      ;;
    --judge_model)
      JUDGE_MODEL="$2"
      shift 2
      ;;
    --judge_base_url)
      JUDGE_BASE_URL="$2"
      shift 2
      ;;
    --judge_api_key)
      JUDGE_API_KEY="$2"
      shift 2
      ;;
    --judge_temperature)
      JUDGE_TEMPERATURE="$2"
      shift 2
      ;;
    --judge_timeout)
      JUDGE_TIMEOUT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$JUDGE_API_KEY" && -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$REPO_ROOT/.env"
  set +a
  JUDGE_API_KEY="${MOONSHOT_API_KEY:-${KIMI_API_KEY:-}}"
fi

if [[ -n "$INPUT_PATH" && "$INPUT_PATH" != /* ]]; then
  INPUT_PATH="$REPO_ROOT/$INPUT_PATH"
fi

if [[ "$INPUT_DIR" != /* ]]; then
  INPUT_DIR="$REPO_ROOT/$INPUT_DIR"
fi

if [[ -n "$INPUT_PATH" && ! -f "$INPUT_PATH" ]]; then
  echo "input_path does not exist: $INPUT_PATH" >&2
  exit 1
fi

if [[ -z "$INPUT_PATH" && ! -d "$INPUT_DIR" ]]; then
  echo "input_dir does not exist: $INPUT_DIR" >&2
  exit 1
fi

CMD=(
  uv run python src/harmbench_mode_eval.py
  --judge_model "$JUDGE_MODEL"
  --judge_base_url "$JUDGE_BASE_URL"
  --judge_temperature "$JUDGE_TEMPERATURE"
  --judge_timeout "$JUDGE_TIMEOUT"
)

if [[ -n "$JUDGE_API_KEY" ]]; then
  CMD+=(--judge_api_key "$JUDGE_API_KEY")
fi

if [[ -n "$INPUT_PATH" ]]; then
  CMD+=(--input_path "$INPUT_PATH")
else
  CMD+=(--input_dir "$INPUT_DIR")
fi

(cd "$REPO_ROOT" && "${CMD[@]}")
