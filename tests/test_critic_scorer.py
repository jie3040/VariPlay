from critic_service_init.critic_scorer import (
    adv_reject_rate,
    compute_critic_rewards,
    gold_pass_rate,
    redundancy,
)


def parsed_verifiers():
    return [
        {
            "verifier_idx": 0,
            "source": "model",
            "valid": True,
            "code": "def check(t):\n    return bool(t.get('tool_calls')), -1",
        },
        {
            "verifier_idx": 1,
            "source": "model",
            "valid": True,
            "code": "def check(t):\n    return True, -1",
        },
        {
            "verifier_idx": 2,
            "source": "model",
            "valid": False,
            "code": None,
        },
    ]


def verifier_executions():
    return [
        {"candidate_idx": 0, "verifier_idx": 0, "passed": True},
        {"candidate_idx": 1, "verifier_idx": 0, "passed": False},
        {"candidate_idx": 0, "verifier_idx": 1, "passed": True},
        {"candidate_idx": 1, "verifier_idx": 1, "passed": True},
    ]


def perturbations():
    return [
        {
            "perturbation_type": "step_drop",
            "applied_successfully": True,
            "verifier_results": [
                {"verifier_idx": 0, "passed": False},
                {"verifier_idx": 1, "passed": True},
            ],
        },
        {
            "perturbation_type": "early_terminate",
            "applied_successfully": True,
            "verifier_results": [
                {"verifier_idx": 0, "passed": False},
                {"verifier_idx": 1, "passed": False},
            ],
        },
        {
            "perturbation_type": "arg_mutation",
            "applied_successfully": False,
            "verifier_results": [
                {"verifier_idx": 0, "passed": True},
            ],
        },
    ]


def test_gold_pass_rate():
    assert gold_pass_rate(0, verifier_executions()) == 0.5
    assert gold_pass_rate(1, verifier_executions()) == 1.0
    assert gold_pass_rate(9, verifier_executions()) == 0.0


def test_adv_reject_rate_ignores_skipped_perturbations():
    assert adv_reject_rate(0, perturbations()) == 1.0
    assert adv_reject_rate(1, perturbations()) == 0.5
    assert adv_reject_rate(9, perturbations()) == 0.0


def test_redundancy_for_valid_and_invalid():
    assert redundancy(0, parsed_verifiers()) > 0.0
    assert redundancy(2, parsed_verifiers()) == 0.0
    assert redundancy(9, parsed_verifiers()) == 0.0


def test_compute_critic_rewards_values():
    scores = compute_critic_rewards(verifier_executions(), perturbations(), parsed_verifiers())
    by_idx = {item["verifier_idx"]: item for item in scores}
    assert len(scores) == 3
    assert by_idx[0]["gold_pass_rate"] == 0.5
    assert by_idx[0]["adv_reject_rate"] == 1.0
    assert by_idx[0]["r_critic"] > 0.0
    assert by_idx[2]["r_critic"] == 0.0
