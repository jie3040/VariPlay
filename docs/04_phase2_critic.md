# Phase 2 and 2.5: Critic-Coder

Phase 2 introduced a Critic-Coder inference service that writes Python verifier functions.

Verifier contract:

```python
def check(trajectory):
    return passed, fail_step
```

Constraints:

- Python standard library only.
- Subprocess timeout for verifier execution.
- Return a `(bool, int)` tuple.
- Inspect trajectory fields such as messages, tool calls and extracted answer.

Phase 2.5 added `source: "model" | "fallback"` attribution. This was necessary because fallback verifiers can prove plumbing but cannot prove model capability.

Empirical model choice:

| Critic model | Fallback | Valid model verifiers | Model-discriminative |
|---|---|---:|---:|
| Qwen2.5-0.5B-Instruct | disabled | 2/5 | 0 |
| Qwen2.5-Coder-1.5B-Instruct | disabled | 5/5 | 1 |
| Qwen2.5-0.5B-Instruct | enabled | 2 model + 1 fallback | fallback only |

Conclusion: use Qwen2.5-Coder-1.5B-Instruct as the default Critic-Coder base model for Phase 3 and Phase 4.

