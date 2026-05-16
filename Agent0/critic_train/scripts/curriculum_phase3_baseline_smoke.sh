#!/usr/bin/env bash
set -euo pipefail

# Same fixed seed batch as Phase 3, but keep original Agent0 reward formula.
export ENABLE_VERIPLAY_REWARD=0
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-agent0_critic_phase3_baseline_smoke}"

bash "$(dirname "$0")/curriculum_phase3_smoke.sh"
