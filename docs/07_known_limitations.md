# Known Limitations and Future Work

1. Full checkpoint save can OOM under Ray CPU marshalling. Phase 4 used lightweight checkpoints that skip optimizer state.
2. D1 showed a ceiling effect: `adv_reject_rate` began at 0.9272 and did not improve.
3. Smoke scale is small: 32 questions, 4 trajectories, 3 iterations.
4. D3 was computed post-hoc from the same records rather than by running a separate baseline iteration.
5. Qwen3-4B executor experiments were postponed for compute reasons.
6. The group-by patch works for smoke but should be cleaned up before a polished release.
7. The saturation threshold is empirical (`0.01`) and should be revalidated at larger scale.
8. Some original Agent0 scripts still assume specific sandbox services; production users should configure their own sandbox layer.
9. Phase 4 critic training is off-policy and short-horizon. Longer on-policy critic training remains future work.
10. The curriculum answer parser issue was intentionally deferred because verifier-based reward became the main signal.

