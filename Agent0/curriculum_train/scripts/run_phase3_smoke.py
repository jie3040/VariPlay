#!/usr/bin/env python3
"""Run a fixed multi-row Phase 3 smoke batch for reward variance comparison."""

import argparse
import importlib.util
import os
from pathlib import Path


SEED_TASKS = [
    ("Compute the value of 1234567 × 7654321 step by step using Python.", 1234567 * 7654321),
    ("Use Python to compute 246813 × 13579, then give the final answer.", 246813 * 13579),
    ("Calculate 314159 × 2718 with a Python tool and report the result.", 314159 * 2718),
    ("Find 98765 × 4321. Show the Python calculation before the answer.", 98765 * 4321),
]


def load_compute_score():
    reward_path = Path("examples/reward_function/curriculum_reward.py").resolve()
    spec = importlib.util.spec_from_file_location("curriculum_reward_phase3_smoke", reward_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", required=True)
    args = parser.parse_args()

    os.environ["EXPERIMENT_NAME"] = args.experiment_name
    os.environ["ENABLE_CRITIC"] = "1"

    predicts = []
    prompt_texts = []
    prompt_messages = []
    for idx, (question, answer) in enumerate(SEED_TASKS):
        predicts.append(f"<question>\n{question}\n</question>\n\n\\boxed{{{answer}}}")
        prompt_texts.append(f"system: Phase 3 smoke fixed seed\nuser: {question}")
        prompt_messages.append([
            {
                "role": "system",
                "content": "Phase 3 smoke: fixed seed batch for reward variance comparison.",
            },
            {"role": "user", "content": question},
        ])

    compute_score = load_compute_score()
    scores = compute_score(
        predicts,
        ["phase3-smoke"] * len(predicts),
        global_step=1,
        prompt_texts=prompt_texts,
        prompt_messages=prompt_messages,
        experiment_name=args.experiment_name,
    )
    print("scores:", scores)


if __name__ == "__main__":
    main()
