#!/usr/bin/env bash
set -euo pipefail
set -x

# Phase 4 Step E: Critic-Coder 1GPU plumbing smoke.
# 目标只验�?extras/group_by/ckpt/loss 有限，不评价模型能力�?
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
STORAGE_PATH="${STORAGE_PATH:-./veriplay_storage}"
SAVE_NAME="${SAVE_NAME:-critic_smoke_phase4}"
TRAIN_DATA="${TRAIN_DATA:-$(pwd)/data/iter_1/train.parquet}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
ROLLOUT_N="${ROLLOUT_N:-2}"
MAX_STEPS="${MAX_STEPS:-2}"
KL_COEF="${KL_COEF:-0.05}"

export STORAGE_PATH
export CRITIC_REWARD_DEBUG="${CRITIC_REWARD_DEBUG:-1}"
export GROUP_BY_SANITY_DEBUG="${GROUP_BY_SANITY_DEBUG:-1}"
export CRITIC_SMOKE_LIGHT_CKPT="${CRITIC_SMOKE_LIGHT_CKPT:-1}"
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0

mkdir -p "${STORAGE_PATH}/models" logs

cleanup() {
  ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT
cleanup

CUDA_VISIBLE_DEVICES=0 python3 -m verl.trainer.main \
  config=examples/config.yaml \
  data.train_files="${TRAIN_DATA}" \
  data.val_files="${TRAIN_DATA}" \
  data.prompt_key=question \
  data.answer_key=verifier_code \
  data.max_prompt_length=1024 \
  data.max_response_length=512 \
  data.rollout_batch_size="${TRAIN_BATCH_SIZE}" \
  data.val_batch_size="${TRAIN_BATCH_SIZE}" \
  data.filter_overlong_prompts=False \
  data.format_prompt=./examples/format_prompt/critic.jinja \
  data.shuffle=True \
  worker.actor.model.model_path="${MODEL_NAME}" \
  worker.actor.model.trust_remote_code=True \
  worker.actor.global_batch_size="${TRAIN_BATCH_SIZE}" \
  worker.actor.micro_batch_size_per_device_for_update=1 \
  worker.actor.micro_batch_size_per_device_for_experience=1 \
  worker.actor.use_torch_compile=False \
  worker.actor.padding_free=True \
  worker.actor.fsdp.enable_full_shard=False \
  worker.actor.fsdp.enable_cpu_offload=False \
  worker.actor.offload.offload_params=True \
  worker.actor.offload.offload_optimizer=True \
  worker.rollout.n="${ROLLOUT_N}" \
  worker.rollout.tensor_parallel_size=1 \
  worker.rollout.gpu_memory_utilization=0.45 \
  worker.rollout.enforce_eager=True \
  worker.rollout.max_num_batched_tokens=2048 \
  worker.rollout.max_model_len=2048 \
  worker.ref.fsdp.enable_full_shard=False \
  worker.ref.fsdp.enable_cpu_offload=True \
  worker.ref.offload.offload_params=True \
  worker.reward.reward_type=batch \
  worker.reward.reward_function=./examples/reward_function/critic_reward.py:compute_score \
  algorithm.adv_estimator=grpo \
  algorithm.group_by=group_id \
  algorithm.use_kl_loss=True \
  algorithm.disable_kl=False \
  algorithm.kl_coef="${KL_COEF}" \
  trainer.logger=['console'] \
  trainer.project_name=agent0 \
  trainer.experiment_name="${SAVE_NAME}" \
  trainer.save_checkpoint_path="${STORAGE_PATH}/models/${SAVE_NAME}" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.val_before_train=False \
  trainer.val_freq=-1 \
  trainer.save_freq=1 \
  trainer.save_limit=2 \
  trainer.total_epochs=1 \
  trainer.max_steps="${MAX_STEPS}" \
  2>&1 | tee "logs/${SAVE_NAME}.log"

echo "critic 1GPU smoke finished"
echo "checkpoint dir: ${STORAGE_PATH}/models/${SAVE_NAME}"
