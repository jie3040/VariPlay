# VeriPlay Architecture

VeriPlay adds an executable verifier player to Agent0's original two-player self-evolution loop.

## Three Players

| Player | Role | Main training signal |
|---|---|---|
| Task-Setter / Curriculum | Generates tasks/questions | Uncertainty, tool signal, verifier writability, repetition penalty |
| Solver / Executor | Solves tasks with tool use | Task correctness and tool-integrated reward |
| Critic-Coder | Writes Python trajectory verifiers | `r_critic` from gold pass rate, adversarial reject rate and redundancy |

## Data Flow

```text
question
  -> executor trajectories
  -> Critic-Coder verifier generation
  -> verifier subprocess execution
  -> perturbation engine
  -> critic_scores and reward_breakdown
  -> curriculum JSONL record
  -> critic GRPO train data
  -> next iteration checkpoints
```

## Core Modules

- `Agent0/curriculum_train/examples/reward_function/record_writer.py`
- `Agent0/curriculum_train/critic_service_init/parser.py`
- `Agent0/curriculum_train/critic_service_init/verifier_exec.py`
- `Agent0/curriculum_train/critic_service_init/perturb_engine.py`
- `Agent0/curriculum_train/critic_service_init/critic_scorer.py`
- `Agent0/critic_train/scripts/build_critic_train_data.py`
- `Agent0/critic_train/examples/reward_function/critic_reward.py`
- `Agent0/scripts/three_way_iteration.sh`
- `Agent0/scripts/three_way_loop.sh`

## Reward Intuition

The key signal is not merely whether the Critic-Coder can produce syntactically valid code. A useful verifier must also distinguish original trajectories from perturbed ones. For this reason the final Phase 3 implementation records:

```text
verifier_writable = model_valid_rate * model_mean_adv_reject_rate
```

The individual components are also saved for ablation.

