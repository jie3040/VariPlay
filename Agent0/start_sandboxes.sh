#!/bin/bash
# Start 4 SandboxFusion servers on different ports
#
# SandboxFusion executes Python code produced by the model. During curriculum
# training, start_vllm_server_tool.py calls these endpoints to run code blocks
# and feed the stdout back into the executor model.

SANDBOX_DIR="$HOME/Agent0/Agent0/SandboxFusion"
PORTS=(8080 8081 8082 8083)
N_SANDBOXES=3 # n - 1 desired sandboxes

echo "Starting SandboxFusion servers..."

cd $SANDBOX_DIR

for i in $(seq 0 $N_SANDBOXES); do
    PORT=${PORTS[$i]}
    LOG_FILE="${SANDBOX_DIR}/sandbox_${PORT}.log"

    echo "Starting SandboxFusion server on port $PORT..."

    # Start one independent sandbox process for this port.
    make run-online PORT=$PORT > $LOG_FILE 2>&1 &
    PID=$!
    echo "Started server on port $PORT (PID: $PID)"

    # Give it time to start
    sleep 10
done

cd -

echo "All SandboxFusion servers started!"
echo "Logs in: $SANDBOX_DIR/sandbox_*.log"
