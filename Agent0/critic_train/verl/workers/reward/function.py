# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import inspect
import os
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, TypedDict

import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from .config import RewardConfig


class RewardScore(TypedDict):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]


SequentialRewardFunction = Callable[[str, str], RewardScore]

BatchRewardFunction = Callable[[List[str], List[str]], List[RewardScore]]


class FunctionRewardManager(ABC):
    """Reward manager for rule-based reward."""

    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        if config.reward_function is None:
            raise ValueError("Reward function is not provided.")

        if not os.path.exists(config.reward_function):
            raise FileNotFoundError(f"Reward function file {config.reward_function} not found.")

        spec = importlib.util.spec_from_file_location("custom_reward_fn", config.reward_function)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_reward_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load reward function: {e}")

        if not hasattr(module, config.reward_function_name):
            raise AttributeError(f"Module {module} does not have function {config.reward_function_name}.")

        reward_fn = getattr(module, config.reward_function_name)
        print(f"Using reward function `{config.reward_function_name}` from `{config.reward_function}`.")
        self.reward_fn = partial(reward_fn, **config.reward_function_kwargs)
        self.config = config
        self.tokenizer = tokenizer

    def _call_reward_fn(self, *args, **kwargs):
        """只传�?reward 函数签名支持的额外参数，兼容�?reward�?""
        signature = inspect.signature(self.reward_fn)
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
            return self.reward_fn(*args, **kwargs)
        supported_kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
        return self.reward_fn(*args, **supported_kwargs)

    @abstractmethod
    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        """Compute reward for a batch of data."""
        ...


class SequentialFunctionRewardManager(FunctionRewardManager):
    reward_fn: SequentialRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        response_ids = data.batch["responses"]
        response_length = data.batch["response_mask"].sum(dim=-1)
        for i in range(len(data)):
            valid_response_ids = response_ids[i][: response_length[i]]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            ground_truth = data.non_tensor_batch["ground_truth"][i]

            score = self._call_reward_fn(
                response_str,
                ground_truth,
                global_step=int(data.meta_info.get("global_step", 0)),
            )
            reward_tensor[i, response_length[i] - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics


class BatchFunctionRewardManager(FunctionRewardManager):
    reward_fn: BatchRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        response_str, ground_truth, prompt_texts, prompt_messages, uids, extras = [], [], [], [], [], []
        response_ids = data.batch["responses"]
        response_length = data.batch["response_mask"].sum(dim=-1)
        reserved_extra_keys = {
            "raw_prompt_ids",
            "prompt_text",
            "prompt_messages",
            "ground_truth",
            "uid",
            "multi_modal_data",
        }
        for i in range(len(data)):
            valid_response_ids = response_ids[i][: response_length[i]]
            response_str.append(
                self.tokenizer.decode(valid_response_ids, skip_special_tokens=self.config.skip_special_tokens)
            )
            ground_truth.append(data.non_tensor_batch["ground_truth"][i])
            if "raw_prompt_ids" in data.non_tensor_batch:
                prompt_texts.append(
                    self.tokenizer.decode(
                        data.non_tensor_batch["raw_prompt_ids"][i],
                        skip_special_tokens=self.config.skip_special_tokens,
                    )
                )
            else:
                prompt_texts.append("")
            if "prompt_text" in data.non_tensor_batch:
                prompt_texts[-1] = str(data.non_tensor_batch["prompt_text"][i])
            if "prompt_messages" in data.non_tensor_batch:
                prompt_messages.append(data.non_tensor_batch["prompt_messages"][i])
            else:
                prompt_messages.append([])
            if "uid" in data.non_tensor_batch:
                uids.append(str(data.non_tensor_batch["uid"][i]))
            else:
                uids.append("")
            extra = {}
            for key, value in data.non_tensor_batch.items():
                if key in reserved_extra_keys:
                    continue
                item = value[i]
                if hasattr(item, "item"):
                    try:
                        item = item.item()
                    except ValueError:
                        pass
                extra[key] = item
            extras.append(extra)

        scores = self._call_reward_fn(
            response_str,
            ground_truth,
            global_step=int(data.meta_info.get("global_step", 0)),
            prompt_texts=prompt_texts,
            prompt_messages=prompt_messages,
            uids=uids,
            extras=extras,
        )
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        for i, score in enumerate(scores):
            reward_tensor[i, response_length[i] - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics
