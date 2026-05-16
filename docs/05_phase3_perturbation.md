# Phase 3: Perturbation and VeriPlay Reward

Phase 3 added five rule-based trajectory perturbations:

1. `arg_mutation`
2. `step_drop`
3. `step_swap`
4. `early_terminate`
5. `tool_substitute`

Each perturbation is applied to executor trajectories and then checked by model-source verifiers. This produces adversarial reject signals.

The main scoring components are:

```text
gold_pass_rate
adv_reject_rate
redundancy
r_critic
verifier_writable
```

The accepted implementation uses:

```text
verifier_writable = model_valid_rate * model_mean_adv_reject_rate
```

This deviates from the original spec's `valid_rate` only formula, but was accepted because valid-rate alone became constant in smoke runs and did not measure discriminative value.

