#!/usr/bin/env bash
set -euo pipefail

# Phase 4 Step F.1 主控脚本�?# 这里只做 dry-run：验�?curriculum -> question_gen/eval -> executor -> critic
# 四个 stage 能在单卡上顺序运行，并且每个 stage 之间 GPU 都能清理干净�?
ITER=1
for arg in "$@"; do
  case "$arg" in
    ITER=*) ITER="${arg#ITER=}" ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STORAGE_PATH="${STORAGE_PATH:-./veriplay_storage}"
PHASE4_RUN_LABEL="${PHASE4_RUN_LABEL:-phase4_f1}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
CURRICULUM_MODEL_PATH="${CURRICULUM_MODEL_PATH:-${MODEL_NAME}}"
EXECUTOR_MODEL_PATH="${EXECUTOR_MODEL_PATH:-${MODEL_NAME}}"
CRITIC_MODEL_PATH="${CRITIC_MODEL_PATH:-${MODEL_NAME}}"

CURRICULUM_STEPS="${CURRICULUM_STEPS:-1}"
EXECUTOR_STEPS="${EXECUTOR_STEPS:-1}"
CRITIC_STEPS="${CRITIC_STEPS:-1}"
N_QUESTIONS="${N_QUESTIONS:-4}"
N_TRAJ_PER_Q="${N_TRAJ_PER_Q:-2}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
EXCLUDE_SATURATED="${EXCLUDE_SATURATED:-0}"
SATURATION_THRESHOLD="${SATURATION_THRESHOLD:-0.05}"
STAGE_TIMEOUT_SECONDS="${STAGE_TIMEOUT_SECONDS:-3600}"
CURRICULUM_SAVE_FREQ="${CURRICULUM_SAVE_FREQ:--1}"
CURRICULUM_ROLLOUT_GPU_UTIL="${CURRICULUM_ROLLOUT_GPU_UTIL:-0.35}"
EXECUTOR_SAVE_FREQ="${EXECUTOR_SAVE_FREQ:--1}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${PHASE4_RUN_LABEL}_iter${ITER}_${RUN_ID}"
RUN_DIR="${STORAGE_PATH}/${PHASE4_RUN_LABEL}/iter_${ITER}/${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
SUMMARY_FILE="${RUN_DIR}/summary.tsv"

mkdir -p "${LOG_DIR}"
echo -e "stage\tstatus\tseconds\tgpu_mib_after\tlog" > "${SUMMARY_FILE}"

cleanup_gpu() {
  bash "${ROOT_DIR}/scripts/cleanup_gpu.sh"
}

gpu_used() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' '
}

run_stage() {
  local stage="$1"
  shift
  local log_file="${LOG_DIR}/${stage}.log"
  local start_ts end_ts elapsed used

  echo "[three_way] ===== ${stage} START ====="
  cleanup_gpu | tee -a "${log_file}"
  start_ts="$(date +%s)"

  set +e
  timeout --foreground "${STAGE_TIMEOUT_SECONDS}" "$@" 2>&1 | tee -a "${log_file}"
  local status="${PIPESTATUS[0]}"
  set -e

  end_ts="$(date +%s)"
  elapsed="$((end_ts - start_ts))"
  cleanup_gpu | tee -a "${log_file}"
  used="$(gpu_used)"

  if [ "${status}" -ne 0 ]; then
    echo -e "${stage}\tFAIL\t${elapsed}\t${used}\t${log_file}" >> "${SUMMARY_FILE}"
    echo "[three_way] ${stage} failed, see ${log_file}"
    exit "${status}"
  fi

  echo -e "${stage}\tPASS\t${elapsed}\t${used}\t${log_file}" >> "${SUMMARY_FILE}"
  echo "[three_way] ===== ${stage} PASS (${elapsed}s, gpu=${used} MiB) ====="
}

QUESTION_EXP="${RUN_NAME}_question"
CRITIC_TRAIN_DATA="${ROOT_DIR}/critic_train/data/iter_${ITER}/train.parquet"

echo "[three_way] run_dir=${RUN_DIR}"
echo "[three_way] iter=${ITER}"
echo "[three_way] curriculum_steps=${CURRICULUM_STEPS}, executor_steps=${EXECUTOR_STEPS}, critic_steps=${CRITIC_STEPS}"
echo "[three_way] n_questions=${N_QUESTIONS}, n_traj_per_q=${N_TRAJ_PER_Q}"
echo "[three_way] exclude_saturated=${EXCLUDE_SATURATED}, saturation_threshold=${SATURATION_THRESHOLD}, stage_timeout_seconds=${STAGE_TIMEOUT_SECONDS}"

run_stage "01_curriculum" bash -c "
  cd '${ROOT_DIR}/curriculum_train' && \
  MODEL_NAME='${CURRICULUM_MODEL_PATH}' \
  SAVE_NAME='${RUN_NAME}_curriculum' \
  EXPERIMENT_NAME='${RUN_NAME}_curriculum' \
  STORAGE_PATH='${STORAGE_PATH}' \
  CURRICULUM_STEPS='${CURRICULUM_STEPS}' \
  CURRICULUM_SAVE_FREQ='${CURRICULUM_SAVE_FREQ}' \
  CURRICULUM_ROLLOUT_GPU_UTIL='${CURRICULUM_ROLLOUT_GPU_UTIL}' \
  VERIPLAY_LIGHT_CKPT='${VERIPLAY_LIGHT_CKPT:-0}' \
  bash scripts/curriculum_train_1gpu_smoke.sh
"

run_stage "02_question_gen_eval" bash -c "
  cd '${ROOT_DIR}/curriculum_train' && \
  MODEL_NAME='${CRITIC_MODEL_PATH}' \
  EXPERIMENT_NAME='${QUESTION_EXP}' \
  STORAGE_PATH='${STORAGE_PATH}' \
  ENABLE_VERIPLAY_REWARD=1 \
  N_QUESTIONS='${N_QUESTIONS}' \
  N_TRAJ_PER_Q='${N_TRAJ_PER_Q}' \
  MAX_NUM_BATCHED_TOKENS='${MAX_NUM_BATCHED_TOKENS}' \
  bash scripts/curriculum_phase4_multiq_smoke.sh
"

echo "[three_way] building critic train data from ${QUESTION_EXP}"
mkdir -p "$(dirname "${CRITIC_TRAIN_DATA}")"
build_args=(
  --record_dir "${STORAGE_PATH}/records/curriculum_train/${QUESTION_EXP}"
  --out_parquet "${CRITIC_TRAIN_DATA}"
  --iter_id "${ITER}"
  --saturation_threshold "${SATURATION_THRESHOLD}"
)
if [ "${EXCLUDE_SATURATED}" != "1" ]; then
  build_args+=(--no-exclude_saturated)
fi

(
  cd "${ROOT_DIR}/critic_train"
  python3 scripts/build_critic_train_data.py "${build_args[@]}" | tee "${LOG_DIR}/02b_build_critic_data.log"
)

run_stage "03_executor" bash -c "
  cd '${ROOT_DIR}/executor_train' && \
  MODEL_NAME='${EXECUTOR_MODEL_PATH}' \
  RUN_NAME='${RUN_NAME}_executor' \
  EXECUTOR_STEPS='${EXECUTOR_STEPS}' \
  EXECUTOR_SAVE_FREQ='${EXECUTOR_SAVE_FREQ}' \
  EXECUTOR_CKPT_DIR='${STORAGE_PATH}/models/${RUN_NAME}_executor' \
  bash examples/train/math_tir/train_1gpu_smoke.sh
"

run_stage "04_critic" bash -c "
  cd '${ROOT_DIR}/critic_train' && \
  MODEL_NAME='${CRITIC_MODEL_PATH}' \
  SAVE_NAME='${RUN_NAME}_critic' \
  STORAGE_PATH='${STORAGE_PATH}' \
  TRAIN_DATA='${CRITIC_TRAIN_DATA}' \
  MAX_STEPS='${CRITIC_STEPS}' \
  TRAIN_BATCH_SIZE=2 \
  ROLLOUT_N=2 \
  KL_COEF=0.05 \
  CRITIC_SMOKE_LIGHT_CKPT=1 \
  bash scripts/critic_train_1gpu_smoke.sh
"

total_seconds="$(awk -F '\t' 'NR > 1 {sum += $3} END {print sum + 0}' "${SUMMARY_FILE}")"
echo "[three_way] dry-run finished"
echo "[three_way] summary=${SUMMARY_FILE}"
echo "[three_way] total_stage_seconds=${total_seconds}"
cat "${SUMMARY_FILE}"
