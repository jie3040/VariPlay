# Critic Train

Phase 4 introduces this directory as the RL entrypoint for the Critic-Coder.

Current scope:

- Step B: clone from `curriculum_train` and switch prompt / reward / config to Critic-Coder.
- Step C: build off-policy Critic training parquet from curriculum JSONL records.
- Step D: use precomputed `r_critic` as the reward signal and group GRPO by `group_id`.

Step E single-GPU training smoke is intentionally not implemented here yet; it should start only after Step B-D review passes.

