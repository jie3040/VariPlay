#!/bin/bash

# Evaluate generated curriculum questions with the current executor model.
# Usage:
#   bash question_evaluate/evaluate.sh <executor_model_path> <save_name>
#
# Each GPU processes one json shard from question_generate.bash and writes:
#   $STORAGE_PATH/generated_question/<save_name>_<gpu_id>_results.json
model_name=$1
save_name=$2

pids=()

# Launch eight independent evaluators, one per GPU/shard.
for i in {0..7}; do
  CUDA_VISIBLE_DEVICES=$i python question_evaluate/evaluate.py --model $model_name --suffix $i --save_name $save_name &
  pids[$i]=$!
done

# The first task is used as a progress anchor before timeout monitoring starts.
wait ${pids[0]}
echo "Task 0 finished."

timeout_duration=3600

# After one hour, kill stragglers so the data-curation stage can finish.
(
  sleep $timeout_duration
  echo "Timeout reached. Killing remaining tasks..."
  for i in {1..7}; do
    if kill -0 ${pids[$i]} 2>/dev/null; then
      kill -9 ${pids[$i]} 2>/dev/null
      echo "Killed task $i"
    fi
  done
) &

for i in {1..7}; do
  wait ${pids[$i]} 2>/dev/null
done
