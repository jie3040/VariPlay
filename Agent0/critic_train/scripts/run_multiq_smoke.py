#!/usr/bin/env python3
"""Run Phase 4 Step A multi-question smoke through the curriculum reward."""

import argparse
import importlib.util
import os
from pathlib import Path

from multiq_seed_questions import MULTIQ_SEEDS


def load_compute_score():
    """动态加载当�?reward function，避免把 examples 目录安装成包�?""
    reward_path = Path("examples/reward_function/curriculum_reward.py").resolve()
    spec = importlib.util.spec_from_file_location("curriculum_reward_phase4_multiq", reward_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_score


def build_batch(n_questions: int, n_traj_per_question: int):
    """�?8 �?seed 展开�?question × trajectory 的固�?smoke batch�?""
    seeds = MULTIQ_SEEDS[:n_questions]
    predicts = []
    ground_truths = []
    prompt_texts = []
    prompt_messages = []
    uids = []
    for q_idx, seed in enumerate(seeds):
        question = seed["question"]
        answer = seed["expected_answer"]
        for traj_idx in range(n_traj_per_question):
            predicts.append(f"<question>\n{question}\n</question>\n\n\\boxed{{{answer}}}")
            ground_truths.append(answer)
            prompt_texts.append(
                f"system: Phase 4 multi-question smoke\n"
                f"user: {question}\n"
                f"difficulty: {seed['difficulty']}"
            )
            prompt_messages.append([
                {
                    "role": "system",
                    "content": "Phase 4 multi-question smoke: fixed seeds for reward variance.",
                },
                {"role": "user", "content": question},
            ])
            uids.append(f"multiq-q{q_idx:02d}-traj{traj_idx:02d}")
    return predicts, ground_truths, prompt_texts, prompt_messages, uids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record_dir", default=None)
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--n_questions", type=int, default=8)
    parser.add_argument("--n_traj_per_question", type=int, default=4)
    args = parser.parse_args()

    if args.n_questions < 1 or args.n_questions > len(MULTIQ_SEEDS):
        raise ValueError(f"--n_questions must be between 1 and {len(MULTIQ_SEEDS)}")
    if args.n_traj_per_question < 1:
        raise ValueError("--n_traj_per_question must be >= 1")

    os.environ["EXPERIMENT_NAME"] = args.experiment_name
    os.environ["ENABLE_CRITIC"] = "1"
    if args.record_dir:
        os.environ["RECORD_DIR"] = args.record_dir

    predicts, ground_truths, prompt_texts, prompt_messages, uids = build_batch(
        n_questions=args.n_questions,
        n_traj_per_question=args.n_traj_per_question,
    )
    compute_score = load_compute_score()
    scores = compute_score(
        predicts,
        ground_truths,
        global_step=1,
        prompt_texts=prompt_texts,
        prompt_messages=prompt_messages,
        uids=uids,
        experiment_name=args.experiment_name,
    )
    print(
        "phase4 multiq scores:",
        {
            "n_questions": args.n_questions,
            "n_traj_per_question": args.n_traj_per_question,
            "n_rows": len(scores),
            "scores": scores,
        },
    )


if __name__ == "__main__":
    main()

