"""Phase 2 subprocess-based verifier sandbox.

LLM 生成�?verifier code 不能在训练主进程里直�?exec。本模块�?`python3 -c` 子进程运�?verifier，并通过 stdin 传入 trajectory JSON�?"""

import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List


VERIFIER_RUNNER_TEMPLATE = r'''
import json
import sys

{VERIFIER_CODE}

trajectory = json.loads(sys.stdin.read())

try:
    result = check(trajectory)
    if not isinstance(result, tuple) or len(result) != 2:
        print(json.dumps({{
            "passed": False,
            "fail_step": -1,
            "error": "verifier did not return (bool, int) tuple"
        }}))
        sys.exit(0)
    passed, fail_step = result
    print(json.dumps({{
        "passed": bool(passed),
        "fail_step": int(fail_step),
        "error": None
    }}))
except Exception as e:
    print(json.dumps({{
        "passed": False,
        "fail_step": -1,
        "error": f"{{type(e).__name__}}: {{e}}"
    }}))
'''


def run_verifier_in_sandbox(
    verifier_code: str,
    trajectory: dict,
    timeout_sec: float = 5.0,
) -> Dict:
    """Run one verifier in a subprocess and return a normalized result dict."""
    runner_code = VERIFIER_RUNNER_TEMPLATE.format(VERIFIER_CODE=verifier_code)
    trajectory_json = json.dumps(trajectory, ensure_ascii=False)
    start = time.time()

    try:
        proc = subprocess.run(
            ["python3", "-c", runner_code],
            input=trajectory_json,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
    except subprocess.TimeoutExpired:
        elapsed_ms = int((time.time() - start) * 1000)
        return {
            "passed": False,
            "fail_step": -1,
            "exec_time_ms": elapsed_ms,
            "exec_error": "TimeoutExpired",
            "exec_stderr": "",
        }
    except Exception as exc:
        elapsed_ms = int((time.time() - start) * 1000)
        return {
            "passed": False,
            "fail_step": -1,
            "exec_time_ms": elapsed_ms,
            "exec_error": f"subprocess_error: {type(exc).__name__}: {exc}",
            "exec_stderr": "",
        }

    elapsed_ms = int((time.time() - start) * 1000)
    if proc.returncode != 0:
        return {
            "passed": False,
            "fail_step": -1,
            "exec_time_ms": elapsed_ms,
            "exec_error": f"subprocess_returncode={proc.returncode}",
            "exec_stderr": proc.stderr[:500],
        }

    try:
        result = json.loads(proc.stdout.strip().split("\n")[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        return {
            "passed": False,
            "fail_step": -1,
            "exec_time_ms": elapsed_ms,
            "exec_error": f"output_parse_error: {exc}",
            "exec_stderr": proc.stderr[:500],
        }

    return {
        "passed": bool(result["passed"]),
        "fail_step": int(result["fail_step"]),
        "exec_time_ms": elapsed_ms,
        "exec_error": result.get("error"),
        "exec_stderr": proc.stderr[:500],
    }


def execute_verifier_matrix(
    parsed_verifiers: List[Dict],
    executor_results: List[Dict],
    max_workers: int = 8,
    is_perturbed: bool = False,
) -> List[Dict]:
    """并发执行 valid verifier × executor candidate 的矩阵�?""
    valid_verifiers = [verifier for verifier in parsed_verifiers if verifier.get("valid")]
    tasks = [(verifier, candidate) for verifier in valid_verifiers for candidate in executor_results]

    def _run_one(task):
        verifier, candidate = task
        trajectory = {
            "messages": candidate.get("messages", []),
            "tool_calls": candidate.get("tool_calls", []),
            "extracted_answer": candidate.get("extracted_answer"),
        }
        result = run_verifier_in_sandbox(
            verifier_code=verifier["code"],
            trajectory=trajectory,
        )
        return {
            "candidate_idx": int(candidate.get("candidate_idx", 0)),
            "verifier_idx": int(verifier.get("verifier_idx", 0)),
            "source": verifier.get("source", "model"),
            "fallback_reason": verifier.get("fallback_reason"),
            "is_perturbed": bool(is_perturbed),
            **result,
        }

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(_run_one, tasks))
