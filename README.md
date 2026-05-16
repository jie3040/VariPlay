# VeriPlay: Self-Play with Executable Verifiers as Grounded Adaptive Discriminators

VeriPlay extends Agent0 with a third player: a Critic-Coder that writes executable Python verifiers for solver trajectories. The verifiers are run against original and perturbed trajectories, then turned into grounded reward signals for curriculum generation and critic training. This repository contains the smoke-scale implementation used to validate the pipeline on a single 48GB GPU.

Paper: TBD  
Architecture: [docs/01_architecture.md](docs/01_architecture.md)  
Limitations: [docs/07_known_limitations.md](docs/07_known_limitations.md)

## What Is VeriPlay?

Agent0 trains a curriculum agent and an executor agent through tool-integrated self-play. VeriPlay adds a Critic-Coder that produces executable verifiers, uses trajectory perturbations to test whether those verifiers are discriminative, and feeds the resulting verifier signal back into curriculum and critic rewards. The project was validated at smoke scale with Qwen2.5-Coder-1.5B-Instruct on one 48GB GPU. The strongest evidence collected so far is a 2.03x question-level reward variance ratio over the Agent0-style baseline reward.

## Architecture

```text
Task-Setter / Curriculum Agent
        |
        | generates questions
        v
Solver / Executor Agent + tools
        |
        | produces trajectories, answers, tool calls
        v
Critic-Coder
        |
        | writes Python check(trajectory) verifiers
        v
Verifier Sandbox + Perturbation Engine
        |
        | computes gold pass rate, adversarial reject rate, redundancy
        v
VeriPlay Rewards
        |
        +--> curriculum reward
        +--> critic GRPO reward
        +--> offline analysis records
```

## Key Results

| Metric | Value | Status |
|---|---:|---|
| Question-level reward variance ratio, D3 | 2.03x | PASS |
| Model-source verifier ratio in top-25 percent, D2 | 1.0 | PASS |
| Verifier code mean length, D4 | 684.66 chars | PASS |
| adv_reject_rate improvement, D1 | -0.014 | FAIL, ceiling caveat |

Honest caveat: VeriPlay is validated at smoke scale: 32 questions x 4 trajectories x 3 iterations with Qwen2.5-Coder-1.5B. Full-scale experiments with larger executors, more questions, and longer co-evolution remain future work because of compute limits. See [docs/07_known_limitations.md](docs/07_known_limitations.md).

## Requirements

- Linux environment with CUDA 12.x
- One NVIDIA GPU; smoke runs were tested on a 48GB vGPU
- Python 3.12 for curriculum/critic and executor environments
- Conda or Mamba
- Approximately 200GB disk for full smoke artifacts
- Approximately 64GB CPU RAM recommended
- A local or remote code-execution sandbox. Phase smoke scripts can use the local subprocess sandbox path for verifier execution; full Agent0 executor runs may require SandboxFusion-style tool services.

## Quickstart

### 1. Setup

```bash
git clone https://github.com/jie3040/VariPlay.git
cd VariPlay
bash setup.sh
```

This creates two conda environments. The curriculum env is intentionally installed with the extra smoke-run dependencies so `quickstart.sh` can run as a single process:

- `veriplay-curriculum`: curriculum, critic service, perturbation, verifier scoring, and one-command smoke execution
- `veriplay-executor`: executor/tool-integrated training

### 2. Run a one-iteration smoke

```bash
conda activate veriplay-curriculum
MODEL_PATH=Qwen/Qwen2.5-Coder-1.5B-Instruct bash quickstart.sh
```

Expected shape:

```text
01_curriculum       PASS
02_question_gen_eval PASS
03_executor         PASS
04_critic           PASS
```

The original Phase 4 F.2 run took 2063 seconds, about 34m23s, on a single 48GB GPU.

### 3. Run the 3-iteration validation

```bash
MODEL_PATH=Qwen/Qwen2.5-Coder-1.5B-Instruct bash quickstart.sh --iters 3
```

The original F.3 run completed 12 stages and produced the D2/D3/D4 evidence in `results/d3_evidence/`.

## How VeriPlay Differs From Agent0

| Area | Agent0 | VeriPlay |
|---|---|---|
| Players | Curriculum + Executor | Curriculum + Executor + Critic-Coder |
| Discriminator | Self-consistency and tool-aware reward | Executable verifier written from trajectory context |
| Negative evidence | Limited | Rule-based perturbation engine with five perturbation types |
| Records | Training logs | Structured JSONL with prompts, raw outputs, trajectories, verifier executions, perturbations and critic scores |
| Critic training | Not present | Off-policy GRPO smoke over verifier rewards |
| Multi-stage loop | Two-player | Three-way serial co-evolution loop for one GPU |

## Repository Layout

```text
Agent0/                 Modified Agent0 implementation with VeriPlay modules
docs/                   Architecture, phase summaries, limitations
results/d3_evidence/    D3 and D1-D4 evidence from Phase 4 F.3
results/sample_jsonl/   Redacted JSONL samples
tests/                  Focused VeriPlay unit tests
setup.sh                Environment setup
quickstart.sh           One-GPU smoke launcher
```

## Unit Tests

The packaged VeriPlay-specific test suite currently contains 27 focused tests, exceeding the 23-test minimum from the project spec:

```bash
pytest tests/
```

Test groups cover:

- Critic parser
- Verifier subprocess execution
- Perturbation engine
- Critic scorer
- Critic reward function
- Critic train data builder

## Results Data

The key evidence is included in:

```text
results/d3_evidence/f3_d3.md
results/d3_evidence/f3_d1_d2_d4.json
```

The D3 post-hoc result:

```text
n_questions = 32
baseline variance = 0.002315
veriplay variance = 0.004694
ratio = 2.03x
```

## Citation

```bibtex
@misc{veriplay2026,
  title  = {VeriPlay: Self-Play with Executable Verifiers as Grounded Adaptive Discriminators},
  author = {TBD},
  year   = {2026},
  url    = {https://arxiv.org/abs/XXXX.XXXXX}
}
```

## License

Apache 2.0. This project is derived from [Agent0](https://github.com/aiming-lab/Agent0), which is also released under Apache 2.0.

## Acknowledgements

VeriPlay builds on Agent0, VeRL, SandboxFusion-style tool execution, and Qwen open models. The project also draws inspiration from the self-play and agent-training literature reviewed in the accompanying research notes.
