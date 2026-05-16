#!/usr/bin/env python3
"""Phase 2 smoking-gun test: original trajectory passes, perturbed one fails."""

import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from critic_service_init.verifier_exec import run_verifier_in_sandbox


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--allow_zero", action="store_true",
                        help="Measurement mode: print count without failing when no verifier discriminates.")
    parser.add_argument("--source_filter", choices=["model", "fallback"],
                        help="Only evaluate verifiers from this source.")
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.jsonl, encoding="utf-8") if line.strip()]
    n_discriminative = 0
    for row_idx, row in enumerate(rows):
        critic_output = row["veriplay"].get("critic_output")
        if not critic_output:
            continue
        candidates = [candidate for candidate in row["executor_results"] if candidate.get("tool_calls")]
        if not candidates:
            continue
        candidate = candidates[0]
        perturbed = copy.deepcopy(candidate)
        perturbed["tool_calls"] = []
        perturbed["messages"] = [
            message for message in perturbed.get("messages", [])
            if "```python" not in str(message.get("content", ""))
            and "Code execution result:" not in str(message.get("content", ""))
        ]

        for verifier in critic_output.get("parsed_verifiers", []):
            if not verifier.get("valid"):
                continue
            if args.source_filter and verifier.get("source") != args.source_filter:
                continue
            original = run_verifier_in_sandbox(verifier["code"], candidate)
            mutated = run_verifier_in_sandbox(verifier["code"], perturbed)
            print(
                f"row={row_idx} verifier={verifier['verifier_idx']} "
                f"source={verifier.get('source', 'model')} "
                f"original={original['passed']} perturbed={mutated['passed']}"
            )
            if original["passed"] and not mutated["passed"]:
                n_discriminative += 1

    if n_discriminative < 1 and not args.allow_zero:
        raise AssertionError("NO verifier showed discrimination")
    status = "OK" if n_discriminative >= 1 else "MEASURED"
    print(f"Criterion C {status}: {n_discriminative} verifier(s) discriminative")


if __name__ == "__main__":
    main()
