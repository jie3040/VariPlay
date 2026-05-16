import os
import sys
import importlib.util
from argparse import Namespace

import pandas as pd


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

spec = importlib.util.spec_from_file_location(
    "build_critic_train_data",
    os.path.join(ROOT, "scripts", "build_critic_train_data.py"),
)
build_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_module)
build_rows = build_module.build_rows
filter_groups = build_module.filter_groups


def _record(question, raw_outputs, parsed_verifiers, critic_scores):
    return {
        "curriculum": {"parsed": {"question": question}},
        "veriplay": {
            "critic_output": {
                "raw_outputs": raw_outputs,
                "parsed_verifiers": parsed_verifiers,
            }
        },
        "critic_scores": critic_scores,
    }


def _verifier(idx, source="model", valid=True):
    return {
        "verifier_idx": idx,
        "source": source,
        "valid": valid,
        "code": "def check(trajectory):\n    return True, -1",
    }


def _score(idx, value):
    return {
        "verifier_idx": idx,
        "source": "model",
        "r_critic": value,
        "gold_pass_rate": 1.0,
        "adv_reject_rate": value,
        "redundancy": 0.0,
    }


def test_build_rows_filters_non_model_and_invalid(tmp_path):
    jsonl = tmp_path / "step_000001.jsonl"
    records = [
        _record(
            "question with signal",
            ["raw0", "raw1", "raw2", "raw3"],
            [_verifier(0), _verifier(1), _verifier(2, source="fallback"), _verifier(3, valid=False)],
            [_score(0, 0.2), _score(1, 0.8), _score(2, 1.0), _score(3, 0.0)],
        )
    ]
    jsonl.write_text("\n".join(__import__("json").dumps(row) for row in records), encoding="utf-8")

    args = Namespace(record_dir=str(tmp_path), iter_id=1, filter_source="model")
    rows = build_rows(args)

    assert len(rows) == 2
    assert {row["source"] for row in rows} == {"model"}
    assert all(row["valid"] for row in rows)
    assert {row["r_critic"] for row in rows} == {0.2, 0.8}
    assert all(row["group_id"].startswith("i1_") for row in rows)


def test_exclude_saturated_drops_zero_std_groups():
    df = pd.DataFrame([
        {"group_id": "keep", "r_critic": 0.2},
        {"group_id": "keep", "r_critic": 0.8},
        {"group_id": "drop_saturated", "r_critic": 1.0},
        {"group_id": "drop_saturated", "r_critic": 1.0},
        {"group_id": "drop_single", "r_critic": 0.5},
    ])

    filtered = filter_groups(df, min_group_size=2, exclude_saturated=True, threshold=0.05)

    assert set(filtered["group_id"]) == {"keep"}
