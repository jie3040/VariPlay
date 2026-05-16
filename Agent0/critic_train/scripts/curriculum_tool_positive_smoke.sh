#!/usr/bin/env bash
set -euo pipefail
set -x

# Phase 1.5 tool-call-positive smoke�?# - 不跑完整 RL 训练
# - 启动 executor 服务
# - 用硬编码 curriculum output 调用 compute_score
# - 验证 JSONL 中至少存在一个非�?tool_calls

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
SAVE_NAME="${SAVE_NAME:-agent0_curriculum_tool_positive_smoke}"
STORAGE_PATH="${STORAGE_PATH:-./veriplay_storage}"
export STORAGE_PATH
export RECORD_DIR="${RECORD_DIR:-${STORAGE_PATH}/records}"
export RECORD_STAGE="curriculum_train"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-${SAVE_NAME}}"
export KEEP_TEMP_RESULTS="${KEEP_TEMP_RESULTS:-0}"
export CURRICULUM_NUM_EXECUTOR_SERVERS=1
export USE_LOCAL_SANDBOX=1
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

RUN_ID=$(date +%s%N)
export RUN_ID

cleanup() {
  pkill -f "start_vllm_server_tool.py --port 5000" || true
}
trap cleanup EXIT

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
  > "logs/curriculum_tool_positive_executor_${RUN_ID}.log" 2>&1 &

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
    echo "executor service port 5000 is reachable"
    break
  fi
  sleep 1
  if [ "$i" -eq 180 ]; then
    echo "executor service did not become reachable"
    exit 1
  fi
done

python3 - <<'PY'
import importlib.util
import os

reward_path = os.path.abspath("examples/reward_function/curriculum_reward.py")
spec = importlib.util.spec_from_file_location("curriculum_reward_tool_smoke", reward_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
compute_score = module.compute_score

question = "Compute the value of 1234567 × 7654321 step by step using Python."
answer = str(1234567 * 7654321)
predict = f"<question>\n{question}\n</question>\n\n\\boxed{{{answer}}}"
prompt_messages = [
    {
        "role": "system",
        "content": "Tool-positive smoke: validate executor trajectory recording with a forced Python tool call.",
    },
    {
        "role": "user",
        "content": question,
    },
]

scores = compute_score(
    [predict],
    ["tool-positive-smoke"],
    global_step=1,
    prompt_texts=["system: Tool-positive smoke\nuser: " + question],
    prompt_messages=[prompt_messages],
    experiment_name=os.environ["EXPERIMENT_NAME"],
)
print("scores:", scores)
PY

RECORD_FILE="${RECORD_DIR}/curriculum_train/${EXPERIMENT_NAME}/step_000001.jsonl"
python3 - <<PY
import json

record_file = "${RECORD_FILE}"
rows = [json.loads(line) for line in open(record_file, encoding="utf-8") if line.strip()]
total_tool_calls = sum(len(ec.get("tool_calls", [])) for row in rows for ec in row["executor_results"])
assert total_tool_calls > 0, f"tool_calls still empty: {total_tool_calls}"
for row in rows:
    for ec in row["executor_results"]:
        for tc in ec.get("tool_calls", []):
            assert {"turn", "code", "stdout", "stderr", "status"}.issubset(tc), tc
print(f"tool_calls validated, total={total_tool_calls}, file={record_file}")
PY

echo "tool-call-positive curriculum smoke finished"
