# Checkpoint Policy

This GitHub package does not include model checkpoints.

Reasons:

- Checkpoints are large.
- They may carry separate model license obligations.
- Smoke-run checkpoints contain environment-specific paths and are not needed to understand or reproduce the code.

The original Phase 4 run produced curriculum, executor and critic checkpoints for each iteration under `$STORAGE_PATH/models/`. To reproduce them, run:

```bash
MODEL_PATH=Qwen/Qwen2.5-Coder-1.5B-Instruct bash quickstart.sh --iters 3
```

The code will write fresh checkpoints under:

```text
$STORAGE_PATH/models/
```

