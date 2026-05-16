#!/usr/bin/env bash
# VeriPlay quickstart: one-GPU smoke validation.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STORAGE_PATH="${STORAGE_PATH:-${ROOT_DIR}/veriplay_storage}"
export BASE_MODEL="${MODEL_PATH:-${BASE_MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}}"
export CURRICULUM_MODEL_PATH="${CURRICULUM_MODEL_PATH:-${BASE_MODEL}}"
export EXECUTOR_MODEL_PATH="${EXECUTOR_MODEL_PATH:-${BASE_MODEL}}"
export CRITIC_MODEL_PATH="${CRITIC_MODEL_PATH:-${BASE_MODEL}}"

ITERS=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --iters)
      ITERS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

export ITER_COUNT="${ITERS}"
export PHASE4_RUN_LABEL="${PHASE4_RUN_LABEL:-quickstart}"
export CURRICULUM_STEPS="${CURRICULUM_STEPS:-6}"
export EXECUTOR_STEPS="${EXECUTOR_STEPS:-20}"
export CRITIC_STEPS="${CRITIC_STEPS:-3}"
export N_QUESTIONS="${N_QUESTIONS:-32}"
export N_TRAJ_PER_Q="${N_TRAJ_PER_Q:-4}"
export EXCLUDE_SATURATED="${EXCLUDE_SATURATED:-1}"
export SATURATION_THRESHOLD="${SATURATION_THRESHOLD:-0.01}"
export CURRICULUM_ROLLOUT_GPU_UTIL="${CURRICULUM_ROLLOUT_GPU_UTIL:-0.55}"
export EXECUTOR_ROLLOUT_GPU_UTIL="${EXECUTOR_ROLLOUT_GPU_UTIL:-0.55}"
export EXECUTOR_SAVE_CONTENTS="${EXECUTOR_SAVE_CONTENTS:-[model]}"
export CRITIC_SMOKE_LIGHT_CKPT="${CRITIC_SMOKE_LIGHT_CKPT:-1}"
export VERIPLAY_LIGHT_CKPT="${VERIPLAY_LIGHT_CKPT:-1}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
export STAGE_TIMEOUT_SECONDS="${STAGE_TIMEOUT_SECONDS:-3600}"

mkdir -p "${STORAGE_PATH}"

echo "VeriPlay quickstart"
echo "  iters=${ITER_COUNT}"
echo "  model=${BASE_MODEL}"
echo "  storage=${STORAGE_PATH}"

cd "${ROOT_DIR}/Agent0"
if [ "${ITER_COUNT}" = "1" ]; then
  bash scripts/three_way_iteration.sh ITER=1
else
  bash scripts/three_way_loop.sh
fi

echo "Done. Results are under ${STORAGE_PATH}."

