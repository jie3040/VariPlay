"""离线对比 baseline vs VeriPlay �?question-level reward variance�?""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

W_UNC = 0.40
W_VERIFIABLE = 0.40
W_TOOL = 0.10
W_REP = 1.00


def _question_key(record: dict) -> str:
    parsed = record.get("curriculum", {}).get("parsed", {})
    return parsed.get("question") or record.get("curriculum", {}).get("prompt", {}).get("user", "")


def _rewards(record: dict) -> tuple[float, float]:
    br = record.get("reward_breakdown", {})
    rep = float(br.get("repetition_penalty", 0.0) or 0.0)
    if not br.get("format_valid", False):
        return -1.0 - rep, -1.0 - rep

    unc = float(br.get("uncertainty", 0.0) or 0.0)
    tool = float(br.get("tool_reward", 0.0) or 0.0)
    verifiable = float(br.get("verifier_writable", 0.0) or 0.0)
    baseline = unc + tool - rep
    veriplay = W_UNC * unc + W_VERIFIABLE * verifiable + W_TOOL * tool - W_REP * rep
    return baseline, veriplay


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record_jsonl", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    records_by_question: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with open(args.record_jsonl, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            question = _question_key(record)
            if question:
                records_by_question[question].append(_rewards(record))

    q_means_baseline = []
    q_means_veriplay = []
    for rows in records_by_question.values():
        q_means_baseline.append(sum(b for b, _ in rows) / len(rows))
        q_means_veriplay.append(sum(v for _, v in rows) / len(rows))

    var_baseline = statistics.variance(q_means_baseline) if len(q_means_baseline) > 1 else 0.0
    var_veriplay = statistics.variance(q_means_veriplay) if len(q_means_veriplay) > 1 else 0.0
    ratio = var_veriplay / var_baseline if var_baseline > 0 else math.inf
    status = "PASS" if ratio >= 1.5 else "FAIL"

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write("# D3 Question-Level Variance (Post-hoc Analysis)\n\n")
        f.write(f"- n_questions: {len(records_by_question)}\n")
        f.write(f"- baseline variance: {var_baseline:.6f}\n")
        f.write(f"- veriplay variance: {var_veriplay:.6f}\n")
        f.write(f"- ratio: {ratio:.2f}x\n")
        f.write(f"- status: {status}\n\n")
        f.write("## Per-question Means\n\n")
        f.write("| idx | baseline | veriplay |\n")
        f.write("|---:|---:|---:|\n")
        for idx, (base, veri) in enumerate(zip(q_means_baseline, q_means_veriplay), start=1):
            f.write(f"| {idx} | {base:.6f} | {veri:.6f} |\n")

    print(
        f"D3 {status}: ratio={ratio:.2f}x "
        f"(baseline_var={var_baseline:.6f}, veriplay_var={var_veriplay:.6f}, "
        f"n_questions={len(records_by_question)})"
    )


if __name__ == "__main__":
    main()
