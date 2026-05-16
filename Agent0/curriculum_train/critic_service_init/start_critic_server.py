#!/usr/bin/env python3
"""Phase 2 Critic-Coder vLLM Flask service."""

import argparse
import math
import os
import time

from flask import Flask, jsonify, request
from transformers import AutoTokenizer
import vllm

from critic_service_init.critic_prompts import CRITIC_SYSTEM_PROMPT, CRITIC_USER_TEMPLATE
from critic_service_init.parser import parse_verifier_code


parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=6000)
parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
parser.add_argument("--gpu_mem_util", type=float, default=0.3)
parser.add_argument("--max_model_len", type=int, default=1024)
parser.add_argument("--max_tokens", type=int, default=512)
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--top_p", type=float, default=0.9)
parser.add_argument("--enforce_eager", action="store_true")
parser.add_argument("--fallback_on_invalid", action="store_true",
                    help="Smoke-only: if no valid verifier is generated, inject a deterministic verifier.")
args = parser.parse_args()
fallback_disabled = os.getenv("DISABLE_FALLBACK", "0") == "1"

print("[critic] loading tokenizer/model...")
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = vllm.LLM(
    model=args.model_path,
    tokenizer=args.model_path,
    gpu_memory_utilization=args.gpu_mem_util,
    max_model_len=args.max_model_len,
    enforce_eager=args.enforce_eager,
)

app = Flask(__name__)


def _parse_with_source(raw_output: str, verifier_idx: int, source: str = "model", fallback_reason=None):
    parsed = parse_verifier_code(raw_output, verifier_idx)
    parsed["source"] = source
    parsed["fallback_reason"] = fallback_reason
    return parsed


def _build_prompt(question: str) -> str:
    system_prompt = CRITIC_SYSTEM_PROMPT.replace("{TASK_DESCRIPTION}", question)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": CRITIC_USER_TEMPLATE},
    ]
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"system: {messages[0]['content']}\nuser: {messages[1]['content']}\nassistant:"


def generate_verifiers(question: str, n_candidates: int = 3) -> dict:
    """Generate verifier candidates and parse them into executable code."""
    start = time.time()
    prompt = _build_prompt(question)
    sampling_params = vllm.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        n=n_candidates,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None,
    )
    response = model.generate([prompt], sampling_params, use_tqdm=False)
    raw_outputs = [output.text for output in response[0].outputs]
    parsed_verifiers = [
        _parse_with_source(raw_output, verifier_idx)
        for verifier_idx, raw_output in enumerate(raw_outputs)
    ]
    fallback_used = False
    min_valid_for_smoke = math.ceil(max(n_candidates, 1) * 0.5)
    valid_count = sum(1 for verifier in parsed_verifiers if verifier["valid"])
    if args.fallback_on_invalid and not fallback_disabled:
        fallback_code = _build_fallback_verifier(question)
        fallback_raw = f"```python\n{fallback_code}\n```"
        if not parsed_verifiers:
            raw_outputs = [fallback_raw for _ in range(max(n_candidates, 1))]
            parsed_verifiers = [
                _parse_with_source(
                    fallback_raw,
                    verifier_idx,
                    source="fallback",
                    fallback_reason="empty_model_outputs",
                )
                for verifier_idx in range(len(raw_outputs))
            ]
            valid_count = sum(1 for verifier in parsed_verifiers if verifier["valid"])
        else:
            # smoke-only：固定保留一�?deterministic、tool-aware verifier�?            # 避免 0.5B critic 生成“语法有效但�?False”的 verifier 卡住 Criterion C�?            if not parsed_verifiers[0]["valid"]:
                valid_count += 1
            raw_outputs[0] = fallback_raw
            parsed_verifiers[0] = _parse_with_source(
                fallback_raw,
                0,
                source="fallback",
                fallback_reason="smoke_tool_aware_reference",
            )
        fallback_used = True
    if args.fallback_on_invalid and not fallback_disabled and valid_count < min_valid_for_smoke:
        for verifier_idx, verifier in enumerate(parsed_verifiers):
            if valid_count >= min_valid_for_smoke:
                break
            if verifier["valid"]:
                continue
            raw_outputs[verifier_idx] = fallback_raw
            parsed_verifiers[verifier_idx] = _parse_with_source(
                fallback_raw,
                verifier_idx,
                source="fallback",
                fallback_reason="validity_below_threshold",
            )
            valid_count += 1
    elapsed_ms = int((time.time() - start) * 1000)
    return {
        "raw_outputs": raw_outputs,
        "parsed_verifiers": parsed_verifiers,
        "n_valid": sum(1 for verifier in parsed_verifiers if verifier["valid"]),
        "n_requested": n_candidates,
        "generation_time_ms": elapsed_ms,
        "fallback_used": fallback_used,
        "fallback_disabled": fallback_disabled,
    }


def _build_fallback_verifier(question: str) -> str:
    """Build a simple smoke-only verifier when the tiny critic model fails.

    这个 fallback 只在 `--fallback_on_invalid` 下启用，用来验证 Phase 2 �?    record/verifier execution/discriminative plumbing，不作为正式研究信号�?    """
    import re

    numbers = re.findall(r"-?\d+", question or "")
    expected = None
    if len(numbers) >= 2 and ("*" in question or "×" in question or "multiply" in question.lower()):
        expected = str(int(numbers[0]) * int(numbers[1]))
    expected_literal = repr(expected)
    return f'''def check(trajectory):
    expected = {expected_literal}
    tool_calls = trajectory.get("tool_calls") or []
    extracted = str(trajectory.get("extracted_answer") or "")
    if expected is None:
        return bool(extracted and extracted.lower() != "none"), -1 if extracted else 0
    for call in tool_calls:
        if call.get("status") == "Finished" and expected in str(call.get("stdout", "")):
            return True, -1
    return False, 0'''


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": args.model_path})


@app.route("/generate_verifiers", methods=["POST"])
def generate_verifiers_route():
    payload = request.get_json(force=True) or {}
    question = payload.get("question", "")
    n_candidates = int(payload.get("n_candidates", 3))
    return jsonify(generate_verifiers(question, n_candidates=n_candidates))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=args.port, threaded=True)
