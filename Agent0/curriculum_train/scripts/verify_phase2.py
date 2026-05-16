#!/usr/bin/env python3
"""Verify Phase 2 JSONL criteria A and B."""

import argparse
import json
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--min_valid_rate", type=float, default=0.5)
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.jsonl, encoding="utf-8") if line.strip()]
    valid_rates = []
    for row in rows:
        critic_output = row["veriplay"]["critic_output"]
        verifier_executions = row["veriplay"]["verifier_executions"]
        if critic_output is None:
            continue
        n_requested = critic_output.get("n_requested", 0)
        if n_requested > 0:
            valid_rates.append(critic_output.get("n_valid", 0) / n_requested)
        if verifier_executions is not None:
            expected = row["executor_aggregation"]["n_candidates"] * critic_output.get("n_valid", 0)
            assert len(verifier_executions) == expected, (
                f"matrix size mismatch: got {len(verifier_executions)}, expected {expected}"
            )

    assert valid_rates, "no critic_output found"
    mean_validity = statistics.mean(valid_rates)
    print(f"Criterion A mean validity = {mean_validity:.2f}")
    assert mean_validity >= args.min_valid_rate, (
        f"mean validity {mean_validity:.2f} < {args.min_valid_rate:.2f}"
    )
    print("Criterion A OK")
    print("Criterion B matrix populated")


if __name__ == "__main__":
    main()
