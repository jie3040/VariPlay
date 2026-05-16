# Phase 4: Three-Way Co-Evolution

Phase 4 introduced Critic GRPO training and a single-GPU schedule that runs the three agents serially:

```text
curriculum -> question/eval -> executor -> critic -> checkpoint merge -> next iteration
```

## Step Summary

- Step A: D3 multi-question smoke passed.
- Step B-D: `critic_train/` scaffold, data builder, reward function and group-by plumbing passed.
- Step E: Critic 1GPU GRPO smoke passed; lightweight checkpointing was accepted for Phase 4.
- Step F.1: Schedule dry run passed in 6m33s.
- Step F.2: One-iteration real-scale smoke passed in 34m23s; saturation threshold adjusted to 0.01.
- Step F.3: Three-iteration run completed; D2/D3/D4 passed, D1 failed due to ceiling behavior.

## F.3 Results

| Criterion | Value | Status |
|---|---:|---|
| D1 adv_reject_rate delta | -0.014375 | FAIL |
| D2 model-source top-25 percent | 1.0 | PASS |
| D3 variance ratio | 2.03x | PASS |
| D4 mean verifier code length | 684.66 | PASS |

The D1 failure is preserved as a caveat rather than tuned away.

