"""Critic reward scoring utilities for Phase 3 VeriPlay."""

from __future__ import annotations

import difflib
import os
from typing import Dict, List


DEFAULT_ALPHA = 0.4
DEFAULT_BETA = 0.5
DEFAULT_GAMMA = 0.1


def gold_pass_rate(verifier_idx: int, verifier_executions: List[Dict]) -> float:
    """�?verifier 在所有原�?executor trajectories 上的通过率�?""
    relevant = [
        item for item in verifier_executions or []
        if int(item.get("verifier_idx", -1)) == int(verifier_idx)
    ]
    if not relevant:
        return 0.0
    return sum(1 for item in relevant if item.get("passed")) / len(relevant)


def adv_reject_rate(verifier_idx: int, perturbations: List[Dict]) -> float:
    """�?verifier 在所�?successful perturbations 上的拒绝率�?""
    passed_values = []
    for perturbation in perturbations or []:
        if not perturbation.get("applied_successfully"):
            continue
        for result in perturbation.get("verifier_results", []):
            if int(result.get("verifier_idx", -1)) == int(verifier_idx):
                passed_values.append(bool(result.get("passed")))
    if not passed_values:
        return 0.0
    return 1.0 - (sum(1 for passed in passed_values if passed) / len(passed_values))


def redundancy(target_idx: int, parsed_verifiers: List[Dict]) -> float:
    """与其�?valid verifier 的最大代码相似度�?""
    target = next(
        (
            verifier for verifier in parsed_verifiers or []
            if int(verifier.get("verifier_idx", -1)) == int(target_idx)
        ),
        None,
    )
    if not target or not target.get("valid") or not target.get("code"):
        return 0.0

    others = [
        verifier for verifier in parsed_verifiers or []
        if int(verifier.get("verifier_idx", -1)) != int(target_idx)
        and verifier.get("valid")
        and verifier.get("code")
    ]
    if not others:
        return 0.0

    return max(
        difflib.SequenceMatcher(None, str(target["code"]), str(verifier["code"])).ratio()
        for verifier in others
    )


def compute_critic_rewards(
    verifier_executions: List[Dict],
    perturbations: List[Dict],
    parsed_verifiers: List[Dict],
) -> List[Dict]:
    """为每�?verifier 计算 Phase 3 critic reward�?
    invalid verifier 也会返回一项，分数�?0，便于后续训练数据对齐�?    """
    alpha = float(os.getenv("CRITIC_SCORE_ALPHA", str(DEFAULT_ALPHA)))
    beta = float(os.getenv("CRITIC_SCORE_BETA", str(DEFAULT_BETA)))
    gamma = float(os.getenv("CRITIC_SCORE_GAMMA", str(DEFAULT_GAMMA)))

    scores = []
    for verifier in parsed_verifiers or []:
        verifier_idx = int(verifier.get("verifier_idx", 0))
        source = verifier.get("source", "model")
        if not verifier.get("valid"):
            scores.append({
                "verifier_idx": verifier_idx,
                "source": source,
                "gold_pass_rate": 0.0,
                "adv_reject_rate": 0.0,
                "redundancy": 0.0,
                "r_critic": 0.0,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
            })
            continue

        gold = gold_pass_rate(verifier_idx, verifier_executions)
        adv = adv_reject_rate(verifier_idx, perturbations)
        red = redundancy(verifier_idx, parsed_verifiers)
        r_critic = alpha * gold + beta * adv - gamma * red
        scores.append({
            "verifier_idx": verifier_idx,
            "source": source,
            "gold_pass_rate": gold,
            "adv_reject_rate": adv,
            "redundancy": red,
            "r_critic": r_critic,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
        })
    return scores
