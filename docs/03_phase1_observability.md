# Phase 1 and 1.5: Observability

Phase 1 added structured JSONL records to the curriculum training loop. The record captures task prompts, generated questions, raw model output, reward breakdown, trajectories, tool calls and version metadata.

Phase 1.5 fixed three critical gaps before Critic-Coder work:

- `curriculum.prompt.system` and `curriculum.prompt.user` are non-empty.
- `curriculum.raw_output` is preserved.
- Tool-call-positive smoke data validates non-empty trajectory tool calls.

The schema is implemented in:

```text
Agent0/curriculum_train/examples/reward_function/record_schema.json
Agent0/curriculum_train/examples/reward_function/record_writer.py
```

The later `veriplay` object was intentionally reserved in the schema so Phase 2 and Phase 3 could fill `critic_output`, `verifier_executions`, `perturbations` and `critic_scores` without changing the record contract.

