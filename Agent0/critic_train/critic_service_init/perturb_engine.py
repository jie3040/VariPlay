"""Rule-based trajectory perturbations for Phase 3 VeriPlay.

扰动只用于评�?verifier 的鲁棒性，不会改写原始 executor 结果�?"""

from __future__ import annotations

import copy
import re
from typing import Dict, List, Tuple


PerturbResult = Tuple[dict, dict]


class PerturbEngine:
    """生成 5 种规则化扰动，每种返�?(perturbed_trajectory, meta)�?""

    PERTURBATION_TYPES = [
        "arg_mutation",
        "step_drop",
        "step_swap",
        "early_terminate",
        "tool_substitute",
    ]

    def apply_all(self, traj: dict) -> List[PerturbResult]:
        """按固定顺序返�?5 个扰动，便于复现和验收�?""
        return [
            self.arg_mutation(traj),
            self.step_drop(traj),
            self.step_swap(traj),
            self.early_terminate(traj),
            self.tool_substitute(traj),
        ]

    def arg_mutation(self, traj: dict) -> PerturbResult:
        """把第一�?tool_call �?stdout 数字 +1，并同步 messages 中的执行结果�?""
        perturbed = _clone_trajectory(traj)
        meta = _meta("arg_mutation")
        tool_calls = perturbed.get("tool_calls") or []
        if not tool_calls:
            return _skip(perturbed, meta, "no_tool_call_to_perturb")

        target = tool_calls[0]
        stdout = str(target.get("stdout", ""))
        match = re.search(r"-?\d+", stdout)
        if not match:
            return _skip(perturbed, meta, "no_numeric_stdout")

        original_value = int(match.group(0))
        mutated_value = original_value + 1
        mutated_stdout = stdout[:match.start()] + str(mutated_value) + stdout[match.end():]
        target["stdout"] = mutated_stdout
        meta["perturbation_params"] = {
            "original_value": original_value,
            "mutated_value": mutated_value,
        }
        meta["perturbed_summary"] = f"changed first tool stdout {original_value} -> {mutated_value}"

        _replace_first_message_fragment(
            perturbed,
            str(original_value),
            str(mutated_value),
            must_contain="Code execution result:",
        )
        return perturbed, meta

    def step_drop(self, traj: dict) -> PerturbResult:
        """删除第一�?assistant code message + 后续 tool result user message�?""
        perturbed = _clone_trajectory(traj)
        meta = _meta("step_drop")
        messages = perturbed.get("messages") or []
        pair = _first_tool_message_pair(messages)
        if pair is None:
            return self.early_terminate(traj, perturbation_type="step_drop", skip_reason=None)

        assistant_idx, user_idx = pair
        perturbed["messages"] = [
            message for idx, message in enumerate(messages)
            if idx not in {assistant_idx, user_idx}
        ]
        if perturbed.get("tool_calls"):
            perturbed["tool_calls"] = perturbed["tool_calls"][1:]
        meta["perturbation_params"] = {"dropped_turn_idx": assistant_idx}
        meta["perturbed_summary"] = f"removed assistant/tool-result message pair at {assistant_idx}/{user_idx}"
        return _ensure_nonempty(perturbed, meta)

    def step_swap(self, traj: dict) -> PerturbResult:
        """交换前两�?assistant/user turn；不足两�?turn 时退化为 step_drop�?""
        perturbed = _clone_trajectory(traj)
        meta = _meta("step_swap")
        messages = perturbed.get("messages") or []
        pairs = _tool_message_pairs(messages)
        if len(pairs) < 2:
            dropped, dropped_meta = self.step_drop(traj)
            dropped_meta["perturbation_type"] = "step_swap"
            dropped_meta["perturbation_params"] = {"fallback": "step_drop"}
            return dropped, dropped_meta

        first = pairs[0]
        second = pairs[1]
        block1 = messages[first[0]: first[1] + 1]
        block2 = messages[second[0]: second[1] + 1]
        swapped = (
            messages[:first[0]]
            + block2
            + messages[first[1] + 1: second[0]]
            + block1
            + messages[second[1] + 1:]
        )
        perturbed["messages"] = swapped
        if len(perturbed.get("tool_calls") or []) >= 2:
            tool_calls = perturbed["tool_calls"]
            tool_calls[0], tool_calls[1] = tool_calls[1], tool_calls[0]
        meta["perturbation_params"] = {"swapped_turn_pairs": [(first[0], second[0])]}
        meta["perturbed_summary"] = f"swapped tool turns starting at {first[0]} and {second[0]}"
        return _ensure_nonempty(perturbed, meta)

    def early_terminate(
        self,
        traj: dict,
        perturbation_type: str = "early_terminate",
        skip_reason: str | None = None,
    ) -> PerturbResult:
        """在第一�?assistant 消息处提前给出错误答案�?""
        perturbed = _clone_trajectory(traj)
        meta = _meta(perturbation_type)
        messages = perturbed.get("messages") or []
        wrong_answer = f"{traj.get('extracted_answer') or 'answer'}_perturbed"
        assistant_idx = next(
            (idx for idx, message in enumerate(messages) if message.get("role") == "assistant"),
            None,
        )
        wrong_message = {
            "role": "assistant",
            "content": f"I will stop early. Therefore \\boxed{{{wrong_answer}}}.",
        }
        if assistant_idx is None:
            perturbed["messages"] = _prefix_messages(messages) + [wrong_message]
        else:
            perturbed["messages"] = messages[:assistant_idx] + [wrong_message]
        perturbed["tool_calls"] = []
        perturbed["extracted_answer"] = wrong_answer
        meta["perturbation_params"] = {"injected_wrong_answer": wrong_answer}
        meta["perturbed_summary"] = f"terminated early with wrong answer {wrong_answer}"
        if skip_reason:
            meta["skip_reason"] = skip_reason
        return _ensure_nonempty(perturbed, meta)

    def tool_substitute(self, traj: dict) -> PerturbResult:
        """�?python code fence 改成 bash，并�?tool_call.code �?bash 前缀�?""
        perturbed = _clone_trajectory(traj)
        meta = _meta("tool_substitute")
        tool_calls = perturbed.get("tool_calls") or []
        if not tool_calls:
            return _skip(perturbed, meta, "no_tool_call_to_perturb")

        affected = 0
        for call in tool_calls:
            code = str(call.get("code", ""))
            if not code.startswith("bash "):
                call["code"] = "bash " + code
                affected += 1
        for message in perturbed.get("messages") or []:
            content = str(message.get("content", ""))
            if "```python" in content:
                message["content"] = content.replace("```python", "```bash")
        meta["perturbation_params"] = {"affected_turn_count": affected}
        meta["perturbed_summary"] = f"changed {affected} tool call(s) from python to bash"
        return _ensure_nonempty(perturbed, meta)


def _clone_trajectory(traj: dict) -> dict:
    cloned = copy.deepcopy(traj or {})
    cloned.setdefault("messages", [])
    cloned.setdefault("tool_calls", [])
    cloned.setdefault("extracted_answer", "")
    return cloned


def _meta(perturbation_type: str) -> dict:
    return {
        "perturbation_type": perturbation_type,
        "perturbation_params": {},
        "applied_successfully": True,
        "skip_reason": None,
        "perturbed_summary": "",
    }


def _skip(perturbed: dict, meta: dict, reason: str) -> PerturbResult:
    meta["applied_successfully"] = False
    meta["skip_reason"] = reason
    meta["perturbed_summary"] = f"skipped: {reason}"
    return _ensure_nonempty(perturbed, meta)


def _ensure_nonempty(perturbed: dict, meta: dict) -> PerturbResult:
    if not perturbed.get("messages"):
        perturbed["messages"] = [{"role": "user", "content": ""}]
        meta["applied_successfully"] = False
        meta["skip_reason"] = meta.get("skip_reason") or "trajectory_too_short"
    return perturbed, meta


def _prefix_messages(messages: list) -> list:
    prefix = [
        message for message in messages
        if message.get("role") in {"system", "user"}
    ]
    return prefix[:2] if prefix else [{"role": "user", "content": ""}]


def _tool_message_pairs(messages: list) -> List[Tuple[int, int]]:
    pairs = []
    for idx, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        if "```python" not in str(message.get("content", "")):
            continue
        next_user = next(
            (
                user_idx for user_idx in range(idx + 1, len(messages))
                if messages[user_idx].get("role") == "user"
                and "Code execution result:" in str(messages[user_idx].get("content", ""))
            ),
            None,
        )
        if next_user is not None:
            pairs.append((idx, next_user))
    return pairs


def _first_tool_message_pair(messages: list):
    pairs = _tool_message_pairs(messages)
    return pairs[0] if pairs else None


def _replace_first_message_fragment(perturbed: dict, old: str, new: str, must_contain: str):
    for message in perturbed.get("messages") or []:
        content = str(message.get("content", ""))
        if must_contain in content and old in content:
            message["content"] = content.replace(old, new, 1)
            return
