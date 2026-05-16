#!/usr/bin/env bash
set -euo pipefail
set -x

# Phase 2 critic smoke:
# - 启动 executor service，用 force tool-call smoke 保证 trajectory �?tool_calls
# - 启动 critic service，生�?verifier code
# - 调用 compute_score 写出�?veriplay 字段�?JSONL
# - �?Criterion A/B/C 验收

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
STORAGE_PATH="${STORAGE_PATH:-./veriplay_storage}"
export STORAGE_PATH
export RECORD_DIR="${RECORD_DIR:-${STORAGE_PATH}/records}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-agent0_critic_phase2_smoke}"
export RECORD_STAGE="curriculum_train"
export ENABLE_CRITIC=1
export CRITIC_N_VERIFIERS="${CRITIC_N_VERIFIERS:-3}"
export CRITIC_PORT="${CRITIC_PORT:-6000}"
export CRITIC_TIMEOUT="${CRITIC_TIMEOUT:-60}"
export DISABLE_FALLBACK="${DISABLE_FALLBACK:-0}"
export USE_LOCAL_SANDBOX=1
export CURRICULUM_NUM_EXECUTOR_SERVERS=1
export MODEL_NAME
export CURRICULUM_MODEL_PATH="${CURRICULUM_MODEL_PATH:-${MODEL_NAME}}"
export EXECUTOR_MODEL_PATH="${EXECUTOR_MODEL_PATH:-${MODEL_NAME}}"
export CURRICULUM_ROLLOUT_BATCH_SIZE=1
export CURRICULUM_ROLLOUT_N=1
export EXECUTOR_NUM_CANDIDATES=2
export EXECUTOR_MAX_TURNS=3
export EXECUTOR_MAX_TOKENS=512
export GIT_REV="${GIT_REV:-$(git rev-parse --short HEAD 2>/dev/null || echo unknown)}"

mkdir -p "${STORAGE_PATH}/temp_results" "${RECORD_DIR}/curriculum_train/${EXPERIMENT_NAME}" logs
rm -f "${RECORD_DIR}/curriculum_train/${EXPERIMENT_NAME}/step_000001.jsonl"

cleanup() {
  pkill -f "[s]tart_vllm_server_tool.py --port 5000" || true
  pkill -f "[s]tart_critic_server.py --port ${CRITIC_PORT}" || true
}
trap cleanup EXIT
cleanup

RUN_ID=$(date +%s%N)
CRITIC_FALLBACK_ARGS=()
DISCRIMINATIVE_ARGS=()
if [ "${DISABLE_FALLBACK}" = "1" ]; then
  export CRITIC_MIN_VALID_RATE="${CRITIC_MIN_VALID_RATE:-0.0}"
  DISCRIMINATIVE_ARGS=(--allow_zero --source_filter model)
else
  CRITIC_FALLBACK_ARGS=(--fallback_on_invalid)
fi

CUDA_VISIBLE_DEVICES=0 python3 vllm_service_init/start_vllm_server_tool.py \
  --port 5000 \
  --model_path "${MODEL_NAME}" \
  --gpu_mem_util 0.18 \
  --num_candidates "${EXECUTOR_NUM_CANDIDATES}" \
  --max_turns "${EXECUTOR_MAX_TURNS}" \
  --max_tokens "${EXECUTOR_MAX_TOKENS}" \
  --max_model_len 1024 \
  --enforce_eager \
  --disable_idle_worker \
  --force_tool_call_smoke \
  --skip_model_load_for_smoke \
  > "logs/phase2_executor_${RUN_ID}.log" 2>&1 &

for i in $(seq 1 180); do
  if python3 - <<'PY'
import requests
try:
    r = requests.get("http://127.0.0.1:5000/health", timeout=1)
    raise SystemExit(0 if r.ok else 1)
except Exception:
    raise SystemExit(1)
PY
  then
    echo "service port 5000 is reachable"
    break
  fi
  sleep 1
  if [ "$i" -eq 180 ]; then
    echo "service port 5000 did not become reachable"
    exit 1
  fi
done

CUDA_VISIBLE_DEVICES=0 python3 critic_service_init/start_critic_server.py \
  --port "${CRITIC_PORT}" \
  --model_path "${MODEL_NAME}" \
  --gpu_mem_util "${CRITIC_GPU_MEM_UTIL:-0.10}" \
  --max_model_len 1024 \
  --max_tokens 512 \
  --enforce_eager \
  "${CRITIC_FALLBACK_ARGS[@]}" \
  > "logs/phase2_critic_${RUN_ID}.log" 2>&1 &

for port in "${CRITIC_PORT}"; do
  for i in $(seq 1 180); do
    if python3 - <<PY
import requests
try:
    r = requests.get("http://127.0.0.1:${port}/health", timeout=1)
    raise SystemExit(0 if r.ok else 1)
except Exception:
    raise SystemExit(1)
PY
    then
      echo "service port ${port} is reachable"
      break
    fi
    sleep 1
    if [ "$i" -eq 180 ]; then
      echo "service port ${port} did not become reachable"
      exit 1
    fi
  done
done

python3 scripts/run_critic_smoke.py --experiment_name "${EXPERIMENT_NAME}"

RECORD_FILE="${RECORD_DIR}/curriculum_train/${EXPERIMENT_NAME}/step_000001.jsonl"
python3 scripts/verify_phase2.py --jsonl "${RECORD_FILE}" --min_valid_rate "${CRITIC_MIN_VALID_RATE:-0.5}"
python3 scripts/discriminative_test.py --jsonl "${RECORD_FILE}" "${DISCRIMINATIVE_ARGS[@]}"

echo "phase2 critic smoke finished: ${RECORD_FILE}"
