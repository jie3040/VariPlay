#!/usr/bin/env python3
"""Compare reward variance between baseline and VeriPlay JSONL files."""

import argparse
import json
import statistics
from collections import defaultdict


def _rows(path):
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    return rows


def _question_key(row):
    return row.get("curriculum", {}).get("parsed", {}).get("question") or row.get("meta", {}).get("uid", "")


def _overall_values(rows):
    return [float(row["reward_breakdown"]["overall"]) for row in rows]


def _question_mean_values(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[_question_key(row)].append(float(row["reward_breakdown"]["overall"]))
    values = [statistics.mean(grouped[key]) for key in sorted(grouped)]
    return values, grouped


def _variance(values):
    if len(values) < 2:
        return 0.0
    return statistics.pvariance(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_jsonl", required=True)
    parser.add_argument("--veriplay_jsonl", required=True)
    parser.add_argument("--level", choices=["trajectory", "question"], default="trajectory")
    parser.add_argument("--min_ratio", type=float, default=1.5)
    parser.add_argument("--allow_equal_zero", action="store_true")
    args = parser.parse_args()

    baseline_rows = _rows(args.baseline_jsonl)
    veriplay_rows = _rows(args.veriplay_jsonl)
    if args.level == "question":
        baseline_values, baseline_grouped = _question_mean_values(baseline_rows)
        veriplay_values, veriplay_grouped = _question_mean_values(veriplay_rows)
        common_questions = sorted(set(baseline_grouped) & set(veriplay_grouped))
        print(f"level: question")
        print(f"baseline rows/questions: {len(baseline_rows)}/{len(baseline_grouped)}")
        print(f"veriplay rows/questions: {len(veriplay_rows)}/{len(veriplay_grouped)}")
        print("per-question mean rewards:")
        for question in common_questions:
            base_mean = statistics.mean(baseline_grouped[question])
            veri_mean = statistics.mean(veriplay_grouped[question])
            print(f"- {question[:90]} | baseline={base_mean:.6f} veriplay={veri_mean:.6f}")
    else:
        baseline_values = _overall_values(baseline_rows)
        veriplay_values = _overall_values(veriplay_rows)
        print(f"level: trajectory")
        print(f"baseline rows: {len(baseline_rows)}")
        print(f"veriplay rows: {len(veriplay_rows)}")

    baseline_var = _variance(baseline_values)
    veriplay_var = _variance(veriplay_values)
    ratio = float("inf") if baseline_var == 0 and veriplay_var > 0 else (
        1.0 if baseline_var == 0 and veriplay_var == 0 else veriplay_var / baseline_var
    )

    print(f"baseline rewards: {baseline_values}")
    print(f"veriplay rewards: {veriplay_values}")
    print(f"baseline variance: {baseline_var:.6f}")
    print(f"veriplay variance: {veriplay_var:.6f}")
    print(f"ratio: {ratio:.3f}x")
    if args.allow_equal_zero and baseline_var == 0 and veriplay_var == 0:
        print("Criterion C MEASURED: both variances are zero")
        return
    assert ratio >= args.min_ratio, f"variance ratio {ratio:.3f} < {args.min_ratio:.3f}"
    print("Criterion D3 OK" if args.level == "question" else "Criterion C OK")


if __name__ == "__main__":
    main()
