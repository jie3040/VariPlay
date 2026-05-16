"""æ±‡æ€?Phase 4 Step F.3 çš?D1/D2/D4 æŒ‡æ ‡ã€?""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_records(record_dir: str) -> list[dict]:
    path = Path(record_dir)
    if path.is_dir():
        path = path / "step_000001.jsonl"
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def iter_adv_reject(records: list[dict]) -> float:
    return mean([
        float(r.get("reward_breakdown", {}).get("model_mean_adv_reject_rate", 0.0) or 0.0)
        for r in records
    ])


def d2_model_source_ratio(records: list[dict]) -> float:
    scores = []
    for record in records:
        for score in record.get("critic_scores", []) or []:
            scores.append({
                "r_critic": float(score.get("r_critic", 0.0) or 0.0),
                "source": score.get("source", ""),
            })
    if not scores:
        return 0.0
    scores.sort(key=lambda item: item["r_critic"], reverse=True)
    n_top = max(1, int(len(scores) * 0.25))
    top = scores[:n_top]
    return sum(1 for item in top if item["source"] == "model") / len(top)


def d4_mean_code_len(records: list[dict]) -> float:
    lengths = []
    for record in records:
        critic_output = record.get("veriplay", {}).get("critic_output", {}) or {}
        for verifier in critic_output.get("parsed_verifiers", []) or []:
            if verifier.get("source") == "model" and verifier.get("valid") and verifier.get("code"):
                lengths.append(len(verifier["code"]))
    return mean(lengths)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter_record_dirs", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    per_iter = []
    for idx, record_dir in enumerate(args.iter_record_dirs, start=1):
        records = load_records(record_dir)
        per_iter.append({
            "iter": idx,
            "record_dir": record_dir,
            "n_records": len(records),
            "adv_reject_rate": iter_adv_reject(records),
            "d2_model_source_ratio": d2_model_source_ratio(records),
            "d4_mean_code_len": d4_mean_code_len(records),
        })

    d1_delta = per_iter[-1]["adv_reject_rate"] - per_iter[0]["adv_reject_rate"]
    result = {
        "per_iter": per_iter,
        "d1_delta": d1_delta,
        "d1_pass": d1_delta >= 0.05,
        "d2_pass": per_iter[-1]["d2_model_source_ratio"] >= 0.80,
        "d4_pass": per_iter[-1]["d4_mean_code_len"] >= 80.0,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
