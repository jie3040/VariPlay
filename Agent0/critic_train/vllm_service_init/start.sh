# Start four executor model services used by curriculum training reward.
# Each process owns one GPU and exposes /hello on ports 5000-5003.
#
# Usage:
#   bash vllm_service_init/start.sh <executor_model_path> <run_id>
model_path=$1
run_id=$2

# vLLM compile cache can become expensive and sometimes stale during repeated
# experiments, so the original recipe disables it.
export VLLM_DISABLE_COMPILE_CACHE=1

# These ports are hard-coded in curriculum_reward.py via fetch(index, path).
CUDA_VISIBLE_DEVICES=4 python vllm_service_init/start_vllm_server_tool.py --port 5000 --model_path $model_path &
CUDA_VISIBLE_DEVICES=5 python vllm_service_init/start_vllm_server_tool.py --port 5001 --model_path $model_path &
CUDA_VISIBLE_DEVICES=6 python vllm_service_init/start_vllm_server_tool.py --port 5002 --model_path $model_path &
CUDA_VISIBLE_DEVICES=7 python vllm_service_init/start_vllm_server_tool.py --port 5003 --model_path $model_path &
