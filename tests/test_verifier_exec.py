from critic_service_init.verifier_exec import execute_verifier_matrix, run_verifier_in_sandbox


DUMMY_TRAJECTORY = {
    "messages": [{"role": "assistant", "content": "Therefore \\boxed{4}"}],
    "tool_calls": [],
    "extracted_answer": "4",
}


def test_verifier_true():
    code = """
def check(trajectory):
    return True, -1
"""
    result = run_verifier_in_sandbox(code, DUMMY_TRAJECTORY)
    assert result["passed"] is True
    assert result["fail_step"] == -1
    assert result["exec_error"] is None
    assert result["exec_time_ms"] < 1000


def test_verifier_false():
    code = """
def check(trajectory):
    return False, 2
"""
    result = run_verifier_in_sandbox(code, DUMMY_TRAJECTORY)
    assert result["passed"] is False
    assert result["fail_step"] == 2
    assert result["exec_error"] is None


def test_verifier_exception():
    code = """
def check(trajectory):
    raise RuntimeError("boom")
"""
    result = run_verifier_in_sandbox(code, DUMMY_TRAJECTORY)
    assert result["passed"] is False
    assert result["exec_error"].startswith("RuntimeError")


def test_verifier_timeout():
    code = """
def check(trajectory):
    while True:
        pass
"""
    result = run_verifier_in_sandbox(code, DUMMY_TRAJECTORY, timeout_sec=0.2)
    assert result["passed"] is False
    assert result["exec_error"] == "TimeoutExpired"


def test_execute_verifier_matrix():
    verifiers = [
        {"verifier_idx": 0, "valid": True, "code": "def check(trajectory):\n    return True, -1"},
        {"verifier_idx": 1, "valid": False, "code": None},
    ]
    candidates = [{"candidate_idx": 3, **DUMMY_TRAJECTORY}]
    results = execute_verifier_matrix(verifiers, candidates)
    assert len(results) == 1
    assert results[0]["candidate_idx"] == 3
    assert results[0]["verifier_idx"] == 0
    assert results[0]["passed"] is True
