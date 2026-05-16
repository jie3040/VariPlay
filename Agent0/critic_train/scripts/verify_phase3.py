#!/usr/bin/env python3
"""Verify Phase 3 perturbation and critic-score criteria A/B."""

import argparse
import json
import statistics


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
    parser.add_argument("--min_successful", type=int, default=3)
    parser.add_argument("--min_model_adv_reject", type=float, default=0.4)
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.jsonl, encoding="utf-8") if line.strip()]
    assert rows, "empty jsonl"

    model_adv_rates = []
    for row_idx, row in enumerate(rows):
        perturbations = row["veriplay"].get("perturbations")
        assert perturbations is not None, f"row {row_idx}: perturbations is null"
        types = {item["perturbation_type"] for item in perturbations}
        successful = sum(1 for item in perturbations if item.get("applied_successfully"))
        print(f"row={row_idx} types={sorted(types)} successful={successful}")
        assert types == EXPECTED_TYPES, f"row {row_idx}: missing perturbation types {types}"
        assert successful >= args.min_successful, (
            f"row {row_idx}: successful perturbations {successful} < {args.min_successful}"
        )

        for score in row.get("critic_scores") or []:
            if score.get("source") == "model":
                model_adv_rates.append(float(score.get("adv_reject_rate", 0.0)))

    assert model_adv_rates, "no model critic_scores found"
    mean_adv = statistics.mean(model_adv_rates)
    print(f"model adv_reject_rate mean = {mean_adv:.3f}")
    assert mean_adv >= args.min_model_adv_reject, (
        f"model adv_reject_rate {mean_adv:.3f} < {args.min_model_adv_reject:.3f}"
    )
    print("Criterion A OK")
    print("Criterion B OK")


if __name__ == "__main__":
    main()
