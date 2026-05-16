#!/usr/bin/env bash
set -euo pipefail

model_path="${1:-Qwen/Qwen2.5-0.5B-Instruct}"
export VLLM_DISABLE_COMPILE_CACHE=1

CUDA_VISIBLE_DEVICES="${CRITIC_GPU:-0}" python3 critic_service_init/start_critic_server.py \
  --port "${CRITIC_PORT:-6000}" \
  --model_path "${model_path}" \
  --gpu_mem_util "${CRITIC_GPU_MEM_UTIL:-0.3}" \
  --max_model_len "${CRITIC_MAX_MODEL_LEN:-1024}" \
  --max_tokens "${CRITIC_MAX_TOKENS:-512}" \
  --enforce_eager
