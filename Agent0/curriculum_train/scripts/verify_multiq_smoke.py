#!/usr/bin/env python3
"""Check Phase 4 multi-question smoke JSONL completeness."""

import argparse
import json
from collections import Counter, defaultdict


EXPECTED_TYPES = {
    "arg_mutation",
    "step_drop",
    "step_swap",
    "early_terminate",
    "tool_substitute",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--n_questions", type=int, default=8)
    parser.add_argument("--n_traj_per_question", type=int, default=4)
    parser.add_argument("--require_veriplay", choices=["0", "1"], default="1")
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.jsonl, encoding="utf-8") if line.strip()]
    expected_rows = args.n_questions * args.n_traj_per_question
    assert len(rows) >= expected_rows, f"too few rows: {len(rows)} < {expected_rows}"

    by_question = defaultdict(list)
    for row in rows:
        question = row["curriculum"]["parsed"]["question"]
        by_question[question].append(row)

    assert len(by_question) >= args.n_questions, (
        f"too few distinct questions: {len(by_question)} < {args.n_questions}"
    )
    counts = Counter({question: len(items) for question, items in by_question.items()})
    for question, count in counts.items():
        assert count >= args.n_traj_per_question, (
            f"question has too few trajectories: {count} < {args.n_traj_per_question}: {question}"
        )

    print(f"rows={len(rows)} distinct_questions={len(by_question)}")
    for idx, (question, items) in enumerate(by_question.items()):
        print(f"question[{idx}] trajectories={len(items)} text={question[:80]}")
        if args.require_veriplay == "1":
            for row_idx, row in enumerate(items):
                perturbations = row["veriplay"].get("perturbations")
                assert perturbations is not None, f"{question}: row {row_idx} perturbations is null"
                types = {item["perturbation_type"] for item in perturbations}
                assert types == EXPECTED_TYPES, f"{question}: row {row_idx} perturbation types={types}"
                assert row.get("critic_scores"), f"{question}: row {row_idx} critic_scores empty"
    print("multi-question smoke completeness OK")


if __name__ == "__main__":
    main()

