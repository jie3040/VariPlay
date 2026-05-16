import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from examples.reward_function.critic_reward import compute_score


VALID_TEMPLATE = """I will write a verifier.
```python
def check(trajectory):
    return True, -1
```"""


def test_valid_verifiers_receive_precomputed_r_critic():
    predicts = [VALID_TEMPLATE for _ in range(3)]
    extras = [
        {"r_critic": 0.2, "gold_pass_rate": 1.0, "adv_reject_rate": 0.2, "redundancy": 0.0},
        {"r_critic": 0.7, "gold_pass_rate": 1.0, "adv_reject_rate": 0.7, "redundancy": 0.0},
        {"r_critic": 1.0, "gold_pass_rate": 1.0, "adv_reject_rate": 1.0, "redundancy": 0.0},
    ]

    scores = compute_score(predicts, [""] * len(predicts), extras=extras)

    assert [score["valid"] for score in scores] == [1, 1, 1]
    assert [score["overall"] for score in scores] == [0.2, 0.7, 1.0]
    assert scores[1]["adv_reject_rate"] == 0.7


def test_invalid_verifiers_are_penalized():
    predicts = [
        "no code block here",
        """```python
def not_check(trajectory):
    return True
```""",
    ]

    scores = compute_score(predicts, ["", ""], extras=[{"r_critic": 1.0}, {"r_critic": 1.0}])

    assert [score["valid"] for score in scores] == [0, 0]
    assert [score["overall"] for score in scores] == [-0.5, -0.5]
    assert all(score["syntax_error"] for score in scores)

