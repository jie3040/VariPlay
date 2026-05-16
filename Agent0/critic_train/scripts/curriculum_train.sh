#!/bin/bash

# Train one curriculum-agent iteration.
# Usage:
#   bash scripts/curriculum_train.sh <executor_agent_path> <curriculum_agent_path> <save_name>
#
# The executor model is launched as four vLLM tool-solving services on GPU 4-7.
# The curriculum model is trained with GRPO on GPU 0-3, using those services as
# the reward signal.
project_name=agent0

# Model used to solve generated questions during reward computation.
executor_agent_path=$1
# Model to update as the curriculum/question-generating agent.
curriculum_agent_path=$2
# Experiment/checkpoint directory name under $STORAGE_PATH/models.
save_path=$3
echo "save_path: $save_path"

# Unique id for this run; useful when several service groups coexist.
RUN_ID=$(date +%s%N)
export RUN_ID

echo "RUN_ID=$RUN_ID"

# Start executor-side vLLM Flask services used by curriculum_reward.py.
bash vllm_service_init/start.sh $executor_agent_path $RUN_ID
echo "vLLM services started with RUN_ID=$RUN_ID"

echo "Start training curriculum: $curriculum_agent_path -> $save_path"

# Main GRPO training entrypoint. The reward function parses each generated
# question, sends it to executor services, scores uncertainty/diversity/tool use,
# then returns per-sample rewards to the trainer.
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=4096 \
    worker.actor.model.model_path=$curriculum_agent_path \
    trainer.experiment_name=$save_path \
    trainer.save_checkpoint_path=${STORAGE_PATH}/models/$save_path \
    trainer.total_epochs=1000 \
    worker.reward.reward_function=./examples/reward_function/curriculum_reward.py:compute_score \
    trainer.val_freq=-1 \
    trainer.n_gpus_per_node=4 \
    data.format_prompt=./examples/format_prompt/questioner.jinja \
    worker.rollout.n=4 \
    worker.actor.global_batch_size=128 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.max_steps=6 \
    trainer.save_freq=1

sleep 5

# Convert the sharded actor checkpoint into a Hugging Face model directory.
echo "merging model"
python scripts/model_merger.py --local_dir ${STORAGE_PATH}/models/$save_path/global_step_5/actor

sleep 10

# Clean up vLLM service and training helper processes started by this script.
pkill python

echo "curriculum agent training finished"
