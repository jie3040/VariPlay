#!/usr/bin/env bash
set -euo pipefail

# Phase 4 Step F.3: run multiple three-way iterations sequentially.
# 每个 iter 结束后，�?FSDP shard 合并�?HuggingFace 目录，作为下一�?iter 的初始化模型�?
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STORAGE_PATH="${STORAGE_PATH:-./veriplay_storage}"
PHASE4_RUN_LABEL="${PHASE4_RUN_LABEL:-phase4_f3}"
ITER_COUNT="${ITER_COUNT:-3}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"

CURRICULUM_STEPS="${CURRICULUM_STEPS:-6}"
EXECUTOR_STEPS="${EXECUTOR_STEPS:-20}"
CRITIC_STEPS="${CRITIC_STEPS:-3}"

curriculum_model="${CURRICULUM_MODEL_PATH:-${BASE_MODEL}}"
executor_model="${EXECUTOR_MODEL_PATH:-${BASE_MODEL}}"
critic_model="${CRITIC_MODEL_PATH:-${BASE_MODEL}}"

mkdir -p "${STORAGE_PATH}/${PHASE4_RUN_LABEL}/reports"
loop_summary="${STORAGE_PATH}/${PHASE4_RUN_LABEL}/loop_summary.tsv"
echo -e "iter\trun_name\trun_dir\tcurriculum_model\texecutor_model\tcritic_model" > "${loop_summary}"

latest_run_dir() {
  local iter="$1"
  find "${STORAGE_PATH}/${PHASE4_RUN_LABEL}/iter_${iter}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1
}

merge_actor_to_hf() {
  local actor_dir="$1"
  local hf_dir="${actor_dir}/huggingface"
  if [ -f "${hf_dir}/model.safetensors" ] || ls "${hf_dir}"/model-*.safetensors >/dev/null 2>&1; then
    echo "[three_way_loop] merged HF already exists: ${hf_dir}"
    return
  fi
  echo "[three_way_loop] merging actor checkpoint: ${actor_dir}"
  (
    cd "${ROOT_DIR}/critic_train"
    python3 scripts/model_merger.py --local_dir "${actor_dir}"
  )
}

latest_actor_dir() {
  local model_dir="$1"
  local actor_dir
  actor_dir="$(
    find "${model_dir}" -mindepth 2 -maxdepth 2 -type d -path "*/global_step_*/actor" 2>/dev/null \
      | sort -V \
      | tail -n 1
  )"
  if [ -z "${actor_dir}" ]; then
    echo "[three_way_loop] cannot find actor checkpoint under ${model_dir}" >&2
    exit 1
  fi
  echo "${actor_dir}"
}

for iter in $(seq 1 "${ITER_COUNT}"); do
  echo "[three_way_loop] ===== ITER ${iter}/${ITER_COUNT} START ====="
  CURRICULUM_MODEL_PATH="${curriculum_model}" \
  EXECUTOR_MODEL_PATH="${executor_model}" \
  CRITIC_MODEL_PATH="${critic_model}" \
  PHASE4_RUN_LABEL="${PHASE4_RUN_LABEL}" \
  STORAGE_PATH="${STORAGE_PATH}" \
  CURRICULUM_STEPS="${CURRICULUM_STEPS}" \
  EXECUTOR_STEPS="${EXECUTOR_STEPS}" \
  CRITIC_STEPS="${CRITIC_STEPS}" \
  bash "${ROOT_DIR}/scripts/three_way_iteration.sh" "ITER=${iter}"

  run_dir="$(latest_run_dir "${iter}")"
  if [ -z "${run_dir}" ]; then
    echo "[three_way_loop] cannot find run dir for iter ${iter}" >&2
    exit 1
  fi
  run_id="$(basename "${run_dir}")"
  run_name="${PHASE4_RUN_LABEL}_iter${iter}_${run_id}"
  echo -e "${iter}\t${run_name}\t${run_dir}\t${curriculum_model}\t${executor_model}\t${critic_model}" >> "${loop_summary}"

  if [ "${iter}" -lt "${ITER_COUNT}" ]; then
    curriculum_actor="$(latest_actor_dir "${STORAGE_PATH}/models/${run_name}_curriculum")"
    executor_actor="$(latest_actor_dir "${STORAGE_PATH}/models/${run_name}_executor")"
    critic_actor="$(latest_actor_dir "${STORAGE_PATH}/models/${run_name}_critic")"

    merge_actor_to_hf "${curriculum_actor}"
    merge_actor_to_hf "${executor_actor}"
    merge_actor_to_hf "${critic_actor}"

    curriculum_model="${curriculum_actor}/huggingface"
    executor_model="${executor_actor}/huggingface"
    critic_model="${critic_actor}/huggingface"
  fi

  echo "[three_way_loop] ===== ITER ${iter}/${ITER_COUNT} PASS ====="
done

echo "[three_way_loop] finished"
echo "[three_way_loop] summary=${loop_summary}"
cat "${loop_summary}"
