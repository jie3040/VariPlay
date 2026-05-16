#!/usr/bin/env python3
"""Run one Phase 2 critic-enabled tool-positive smoke sample."""

import argparse
import importlib.util
import os
from pathlib import Path


def load_compute_score():
    reward_path = Path("examples/reward_function/curriculum_reward.py").resolve()
    spec = importlib.util.spec_from_file_location("curriculum_reward_phase2_smoke", reward_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", required=True)
    args = parser.parse_args()

    os.environ["EXPERIMENT_NAME"] = args.experiment_name
    os.environ["ENABLE_CRITIC"] = "1"

    question = "Compute the value of 1234567 × 7654321 step by step using Python."
    answer = str(1234567 * 7654321)
    predict = f"<question>\n{question}\n</question>\n\n\\boxed{{{answer}}}"
    prompt_messages = [
        {
            "role": "system",
            "content": "Phase 2 critic smoke: validate verifier generation and execution on a tool trajectory.",
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    compute_score = load_compute_score()
    scores = compute_score(
        [predict],
        ["phase2-critic-smoke"],
        global_step=1,
        prompt_texts=["system: Phase 2 critic smoke\nuser: " + question],
        prompt_messages=[prompt_messages],
        experiment_name=args.experiment_name,
    )
    print("scores:", scores)


if __name__ == "__main__":
    main()
