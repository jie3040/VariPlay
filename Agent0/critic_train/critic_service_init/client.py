"""HTTP client for the Phase 2 Critic-Coder service."""

from typing import List

import requests


def call_critic_service_batch(
    questions: List[str],
    n_candidates: int = 3,
    port: int = 6000,
    timeout: int = 60,
) -> List[dict]:
    """批量调用 critic service；失败时返回结构�?error，不中断训练�?""
    url = f"http://0.0.0.0:{port}/generate_verifiers"
    responses = []
    for question in questions:
        if not question:
            responses.append({
                "raw_outputs": [],
                "parsed_verifiers": [],
                "n_valid": 0,
                "n_requested": n_candidates,
                "generation_time_ms": 0,
                "error": "empty question",
            })
            continue
        try:
            response = requests.post(
                url,
                json={"question": question, "n_candidates": n_candidates},
                timeout=timeout,
            )
            response.raise_for_status()
            responses.append(response.json())
        except Exception as exc:
            responses.append({
                "raw_outputs": [],
                "parsed_verifiers": [],
                "n_valid": 0,
                "n_requested": n_candidates,
                "generation_time_ms": 0,
                "error": f"{type(exc).__name__}: {exc}",
            })
    return responses
