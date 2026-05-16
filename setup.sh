#!/usr/bin/env bash
# VeriPlay setup script.
# Tested during development on Ubuntu-like Linux with CUDA 12.x.

set -euo pipefail

CURRICULUM_ENV="${CURRICULUM_ENV:-veriplay-curriculum}"
EXECUTOR_ENV="${EXECUTOR_ENV:-veriplay-executor}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
SKIP_FLASH_ATTN="${SKIP_FLASH_ATTN:-0}"

echo "[1/4] Creating conda envs..."
conda create -n "${CURRICULUM_ENV}" "python=${PYTHON_VERSION}" -y
conda create -n "${EXECUTOR_ENV}" "python=${PYTHON_VERSION}" -y

echo "[2/4] Installing curriculum/critic dependencies..."
conda run -n "${CURRICULUM_ENV}" pip install -r Agent0/curriculum_train/requirements.txt
conda run -n "${CURRICULUM_ENV}" pip install -r Agent0/requirements.txt
conda run -n "${CURRICULUM_ENV}" pip install -e Agent0/executor_train/verl

if [ "${SKIP_FLASH_ATTN}" != "1" ]; then
  conda run -n "${CURRICULUM_ENV}" pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
fi

echo "[3/4] Installing executor dependencies..."
conda run -n "${EXECUTOR_ENV}" pip install -r Agent0/requirements.txt
conda run -n "${EXECUTOR_ENV}" pip install -e Agent0/executor_train/verl

if [ "${SKIP_FLASH_ATTN}" != "1" ]; then
  conda run -n "${EXECUTOR_ENV}" pip install flash-attn==2.8.3 --no-build-isolation --no-cache-dir
fi

echo "[4/4] Setup done."
echo "For one-command smoke runs, activate the unified smoke env first:"
echo "  conda activate ${CURRICULUM_ENV}"
echo "  MODEL_PATH=Qwen/Qwen2.5-Coder-1.5B-Instruct bash quickstart.sh"
