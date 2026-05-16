#!/usr/bin/env bash
set -euo pipefail
set -x

# 1GPU curriculum train smoke run�?# - 使用 0.5B 本地模型同时作为 curriculum agent �?executor agent
# - 只启�?1 �?executor vLLM 服务
# - 只跑 1 �?GRPO step，目的是验证第一阶段训练链路，不追求效果

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
SAVE_NAME="${SAVE_NAME:-agent0_curriculum_1gpu_smoke}"
STORAGE_PATH="${STORAGE_PATH:-./veriplay_storage}"
export STORAGE_PATH
export CURRICULUM_NUM_EXECUTOR_SERVERS=1
export RECORD_DIR="${RECORD_DIR:-${STORAGE_PATH}/records}"
export RECORD_STAGE="curriculum_train"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-${SAVE_NAME}}"
export KEEP_TEMP_RESULTS="${KEEP_TEMP_RESULTS:-0}"
export MODEL_NAME
export CURRICULUM_MODEL_PATH="${CURRICULUM_MODEL_PATH:-${MODEL_NAME}}"
export EXECUTOR_MODEL_PATH="${EXECUTOR_MODEL_PATH:-${MODEL_NAME}}"
export CURRICULUM_ROLLOUT_BATCH_SIZE=2
export CURRICULUM_ROLLOUT_N=2
export EXECUTOR_NUM_CANDIDATES=2
export EXECUTOR_MAX_TURNS=1
export EXECUTOR_MAX_TOKENS=128
export GIT_REV="${GIT_REV:-$(git rev-parse --short HEAD 2>/dev/null || echo unknown)}"

mkdir -p "${STORAGE_PATH}/models" "${STORAGE_PATH}/temp_results" "${RECORD_DIR}/curriculum_train/${EXPERIMENT_NAME}" logs data/smoke_curriculum

python3 scripts/make_smoke_curriculum_data.py --out_dir data/smoke_curriculum

RUN_ID=$(date +%s%N)
export RUN_ID

cleanup() {
  pkill -f "start_vllm_server_tool.py --port 5000" || true
  rm -f /tmp/curriculum_smoke_vllm_${RUN_ID}.log
}
trap cleanup EXIT

# executor 服务只给 reward 函数使用，smoke 时显存占用压低、候选数压低�?CUDA_VISIBLE_DEVICES=0 python3 vllm_service_init/start_vllm_server_tool.py \
  --port 5000 \
  --model_path "${MODEL_NAME}" \
  --gpu_mem_util 0.18 \
  --num_candidates 2 \
  --max_turns 1 \
  --max_tokens 128 \
  --max_model_len 512 \
  --enforce_eager \
  --disable_idle_worker \
  > "logs/curriculum_smoke_executor_${RUN_ID}.log" 2>&1 &
EXECUTOR_PID=$!

# �?Flask/vLLM 服务起来。最多等 180 秒�?for i in $(seq 1 180); do
  if python3 - <<'PY'
import requests
try:
    r = requests.get("http://127.0.0.1:5000/health", timeout=1)
    raise SystemExit(0 if r.ok else 1)
except Exception:
    pass
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

CUDA_VISIBLE_DEVICES=0 python3 -m verl.trainer.main \
  config=examples/config.yaml \
  data.train_files="$(pwd)/data/smoke_curriculum/train.parquet" \
  data.val_files="$(pwd)/data/smoke_curriculum/val.parquet" \
  data.prompt_key=problem \
  data.answer_key=answer \
  data.max_prompt_length=256 \
  data.max_response_length=256 \
  data.rollout_batch_size=2 \
  data.val_batch_size=1 \
  data.filter_overlong_prompts=False \
  data.format_prompt=./examples/format_prompt/questioner.jinja \
  worker.actor.model.model_path="${MODEL_NAME}" \
  worker.actor.model.trust_remote_code=True \
  worker.actor.global_batch_size=2 \
  worker.actor.micro_batch_size_per_device_for_update=1 \
  worker.actor.micro_batch_size_per_device_for_experience=1 \
  worker.actor.use_torch_compile=False \
  worker.actor.padding_free=True \
  worker.actor.fsdp.enable_full_shard=False \
  worker.actor.fsdp.enable_cpu_offload=False \
  worker.actor.offload.offload_params=True \
  worker.actor.offload.offload_optimizer=True \
  worker.rollout.n=2 \
  worker.rollout.tensor_parallel_size=1 \
  worker.rollout.gpu_memory_utilization=0.35 \
  worker.rollout.enforce_eager=True \
  worker.rollout.max_num_batched_tokens=512 \
  worker.rollout.max_model_len=512 \
  worker.ref.fsdp.enable_full_shard=False \
  worker.ref.fsdp.enable_cpu_offload=True \
  worker.ref.offload.offload_params=True \
  worker.reward.reward_type=batch \
  worker.reward.reward_function=./examples/reward_function/curriculum_reward.py:compute_score \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_loss=True \
  algorithm.disable_kl=False \
  algorithm.kl_coef=1e-2 \
  trainer.logger=['console'] \
  trainer.project_name=agent0 \
  trainer.experiment_name="${SAVE_NAME}" \
  trainer.save_checkpoint_path="${STORAGE_PATH}/models/${SAVE_NAME}" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.val_before_train=False \
  trainer.val_freq=-1 \
  trainer.save_freq=-1 \
  trainer.total_epochs=1 \
  trainer.max_steps=1

echo "single-GPU curriculum smoke training finished"
