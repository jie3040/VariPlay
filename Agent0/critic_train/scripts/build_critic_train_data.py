#!/usr/bin/env python3
"""�?curriculum record JSONL 抽取 Critic-Coder 训练数据�?""

import argparse
import glob
import hashlib
import json
import os
from typing import Dict, Iterable, List

import pandas as pd


REQUIRED_COLUMNS = [
    "question",
    "verifier_code",
    "parsed_code",
    "source",
    "valid",
    "r_critic",
    "gold_pass_rate",
    "adv_reject_rate",
    "redundancy",
    "group_id",
    "iter_id",
]


def iter_jsonl_files(record_dir: str) -> Iterable[str]:
    if os.path.isfile(record_dir):
        yield record_dir
        return
    yield from sorted(glob.glob(os.path.join(record_dir, "step_*.jsonl")))


def _score_by_idx(critic_scores: List[Dict]) -> Dict[int, Dict]:
    scores = {}
    for score in critic_scores or []:
        if "verifier_idx" in score:
            scores[int(score["verifier_idx"])] = score
    return scores


def build_rows(args) -> List[Dict]:
    rows = []
    for jsonl_file in iter_jsonl_files(args.record_dir):
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                question = record.get("curriculum", {}).get("parsed", {}).get("question", "")
                if not question:
                    continue

                critic_output = record.get("veriplay", {}).get("critic_output") or {}
                parsed_verifiers = critic_output.get("parsed_verifiers") or []
                raw_outputs = critic_output.get("raw_outputs") or []
                score_map = _score_by_idx(record.get("critic_scores") or [])
                if not parsed_verifiers or not score_map:
                    continue

                q_hash = hashlib.md5(question.encode("utf-8")).hexdigest()[:8]
                group_id = f"i{args.iter_id}_{q_hash}"

                for verifier in parsed_verifiers:
                    if verifier.get("source") != args.filter_source:
                        continue
                    if not verifier.get("valid"):
                        continue
                    verifier_idx = int(verifier.get("verifier_idx", -1))
                    score = score_map.get(verifier_idx)
                    if score is None:
                        continue
                    raw_output = raw_outputs[verifier_idx] if 0 <= verifier_idx < len(raw_outputs) else ""
                    rows.append({
                        "question": question,
                        "verifier_code": raw_output,
                        "parsed_code": verifier.get("code") or "",
                        "source": verifier.get("source", ""),
                        "valid": bool(verifier.get("valid")),
                        "r_critic": float(score.get("r_critic", 0.0)),
                        "gold_pass_rate": float(score.get("gold_pass_rate", 0.0)),
                        "adv_reject_rate": float(score.get("adv_reject_rate", 0.0)),
                        "redundancy": float(score.get("redundancy", 0.0)),
                        "group_id": group_id,
                        "iter_id": int(args.iter_id),
                    })
    return rows


def filter_groups(df: pd.DataFrame, min_group_size: int, exclude_saturated: bool, threshold: float) -> pd.DataFrame:
    group_stats = df.groupby("group_id")["r_critic"].agg(["std", "count"]).reset_index()
    valid = group_stats[group_stats["count"] >= min_group_size]
    if exclude_saturated:
        valid = valid[valid["std"].fillna(0.0) >= threshold]
    valid_groups = set(valid["group_id"].tolist())
    return df[df["group_id"].isin(valid_groups)].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record_dir", required=True, help="JSONL 文件�?records/curriculum_train/{exp}/ 目录")
    ap.add_argument("--out_parquet", required=True)
    ap.add_argument("--iter_id", type=int, required=True)
    ap.add_argument("--filter_source", default="model", help="默认只保�?source=model �?verifier")
    ap.add_argument("--min_group_size", type=int, default=2)
    ap.add_argument(
        "--exclude_saturated",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="丢弃 r_critic std 过低�?group，避�?GRPO advantage �?0",
    )
    ap.add_argument("--saturation_threshold", type=float, default=0.05)
    args = ap.parse_args()

    rows = build_rows(args)
    if not rows:
        raise SystemExit("ERROR: no critic train rows extracted")

    df = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    before_rows = len(df)
    before_groups = df["group_id"].nunique()
    df = filter_groups(
        df,
        min_group_size=args.min_group_size,
        exclude_saturated=args.exclude_saturated,
        threshold=args.saturation_threshold,
    )
    if df.empty:
        raise SystemExit(
            "ERROR: all groups filtered out. Try --no-exclude_saturated only for debugging, "
            "or collect more non-saturated records."
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out_parquet)), exist_ok=True)
    df.to_parquet(args.out_parquet, index=False)

    print(f"Input rows/groups: {before_rows}/{before_groups}")
    print(f"After filtering: {len(df)} rows, {df['group_id'].nunique()} groups")
    print(f"Wrote {len(df)} rows to {args.out_parquet}")
    print(
        "r_critic stats: "
        f"min={df.r_critic.min():.3f} mean={df.r_critic.mean():.3f} max={df.r_critic.max():.3f}"
    )
    print(f"columns: {list(df.columns)}")


if __name__ == "__main__":
    main()

