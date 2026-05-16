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

import regex as re
from typing import Dict, List
import json
from mathruler.grader import extract_boxed_content, grade_answer
import os
import sys
import time
import random
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.cluster import AgglomerativeClustering
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from record_writer import RecordWriter
from critic_service_init.client import call_critic_service_batch
from critic_service_init.verifier_exec import execute_verifier_matrix
from critic_service_init.perturb_engine import PerturbEngine
from critic_service_init.critic_scorer import compute_critic_rewards

STORAGE_PATH = os.getenv("STORAGE_PATH","")

def _bleu_distance_matrix(sentences):
    """Build a pairwise BLEU distance matrix for generated questions.

    Similar questions have low distance. The curriculum reward later uses this
    to penalize repetitive tasks inside the same training batch. 惩罚重复的题�?
    """
    n = len(sentences)
    dist = np.zeros((n, n))
    smoother = SmoothingFunction().method1
    for i in tqdm(range(n), desc="  - Calculating BLEU distance matrix", leave=False):
        for j in range(i, n):
            if i == j:
                score = 1.0
            else:
                ref = [sentences[j].split()]
                hyp = sentences[i].split()
                score = sentence_bleu(ref, hyp, smoothing_function=smoother)
            dist[i, j] = dist[j, i] = 1 - score
    return dist

def cluster_share_per_problem(
        problems,
        distance_threshold: float = 0.5,
        linkage: str = "average"):
    """Return each problem's cluster share as a repetition penalty.

    If many generated problems are similar, their cluster is large and each item
    receives a larger penalty. This pushes the curriculum agent toward diversity.
    """
    if not problems:
        return []
    if len(problems) <= 1:
        # AgglomerativeClustering 至少需�?2 个样本；smoke/validation 可能只有 1 条�?        # 单样本没有“重复问题”可聚类，Phase 1 记录链路里按 0 惩罚处理�?        return [0.0] * len(problems)
    print('start clustering')
    start_time = time.time()
    dist_mat = _bleu_distance_matrix(problems)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage=linkage
    )
    labels = clustering.fit_predict(dist_mat)
    print(f'end clustering, time: {time.time() - start_time}')
    total = len(problems)
    cluster_size = Counter(labels)
    cluster_ratio = {lab: sz / total for lab, sz in cluster_size.items()}

    proportions = [cluster_ratio[lab] for lab in labels]
    return proportions

def generate_temp_filename(prefix="temp", suffix=".json"):
    """Create a unique temporary json path under $STORAGE_PATH/temp_results."""
    timestamp = int(time.time() * 1000)
    rand_part = random.randint(0, 99999)
    return f"{STORAGE_PATH}/temp_results/{prefix}_{timestamp}_{rand_part}{suffix}"
def split_list(lst, n=4):
    """Split a batch into n near-equal chunks for executor services."""
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

os.environ["NO_PROXY"] = "0.0.0.0,127.0.0.1"

def fetch(index,i):
    """Ask one local executor service to process a temporary question file."""
    response = requests.get(f"http://127.0.0.1:{5000+index}/hello?name={i}")
    return True

def generate_results(data):
    """Evaluate generated questions with executor vLLM services.

    Input items contain the curriculum agent's proposed question and answer.
    Output items contain executor self-consistency scores computed by
    start_vllm_server_tool.py.
    """
    # smoke run 只有 1 �?GPU 时，可以通过环境变量只启�?1 �?executor 服务�?    num_services = int(os.getenv("CURRICULUM_NUM_EXECUTOR_SERVERS", "4"))
    datas = split_list(data, num_services)
    random_names = [generate_temp_filename(prefix=f"temp_{i}", suffix=".json") for i in range(num_services)]
    for i in range(num_services):
        with open(random_names[i],'w') as f:
            json.dump(datas[i],f,indent=4)

    final_results = []
    with ThreadPoolExecutor(max_workers=num_services) as executor:
        futures = [executor.submit(fetch, i,random_names[i]) for i in range(num_services)]

        for future in tqdm(as_completed(futures), total=len(futures), desc="  - Servers processing"):
            future.result() # Simplified to just get the result

    for i in tqdm(range(num_services), desc="  - Reading result files", leave=False):
        with open(random_names[i].replace('.json','_results.json'),'r') as f:
            final_results.extend(json.load(f))
    if os.getenv("KEEP_TEMP_RESULTS", "0") != "1":
        for i in range(num_services):
            os.remove(random_names[i].replace('.json','_results.json'))
    return final_results

def format_reward(predict: str) -> float:
    """Optional format checker for <think>...</think> plus boxed answer."""
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, predict)
    return 1.0 if format_match else 0.0


def accuracy_reward(predict: str, ground_truth: str) -> float:
    """Optional direct answer reward against a known ground truth."""
    answer = extract_boxed_content(predict)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0

def calculate_tool_reward(predict: str, weight: float = 0.05, cap: int = 4) -> float:
    """Reward tool-aware generations by counting ```output markers."""
    if not predict:
        return 0.0

    tool_call_count = len(re.findall(r"```output", predict))

    capped_calls = min(tool_call_count, cap)

    return capped_calls * weight


def compute_curriculum_reward(
    *,
    final_result: Dict,
    parsed_question: str,
    parsed_answer: str,
    critic_output,
    critic_scores,
    cluster_penalty: float,
    n_tool_markers: int,
) -> tuple:
    """Phase 3 VeriPlay curriculum reward, gated by ENABLE_VERIPLAY_REWARD."""
    format_valid = bool(parsed_question and parsed_answer)
    if not format_valid:
        overall = -1.0 - cluster_penalty
        return overall, {
            "overall": overall,
            "format_valid": False,
            "uncertainty": 0.0,
            "verifier_writable": 0.0,
            "model_valid_rate": 0.0,
            "model_mean_adv_reject_rate": 0.0,
            "tool_reward": 0.0,
            "repetition_penalty": cluster_penalty,
            "weights": _reward_weights(),
        }

    p = float(final_result.get("self_consistency", final_result.get("score", 0.0)) or 0.0)
    r_unc = min(p, 1.0 - p)
    r_tool = 0.05 * min(int(n_tool_markers or 0), 4)
    parsed_verifiers = (critic_output or {}).get("parsed_verifiers", [])
    n_requested = max(int((critic_output or {}).get("n_requested", 1) or 1), 1)
    n_valid_model = sum(
        1 for verifier in parsed_verifiers
        if verifier.get("valid") and verifier.get("source", "model") == "model"
    )
    valid_rate_model = n_valid_model / n_requested
    model_adv_rates = [
        float(score.get("adv_reject_rate", 0.0))
        for score in (critic_scores or [])
        if score.get("source") == "model"
    ]
    if model_adv_rates:
        # Phase 3 使用“写得出 verifier 且能拒绝扰动”的组合信号�?        # fallback verifier 不计入，避免 curriculum 学到工程托底偏好�?        model_mean_adv_reject_rate = sum(model_adv_rates) / len(model_adv_rates)
        r_verifiable = valid_rate_model * model_mean_adv_reject_rate
    else:
        model_mean_adv_reject_rate = 0.0
        r_verifiable = valid_rate_model
    weights = _reward_weights()
    overall = (
        weights["W_UNC"] * r_unc
        + weights["W_VERIFIABLE"] * r_verifiable
        + weights["W_TOOL"] * r_tool
        - weights["W_REP"] * cluster_penalty
    )
    overall = max(overall, -1.0)
    return overall, {
        "overall": overall,
        "format_valid": True,
        "uncertainty": r_unc,
        "verifier_writable": r_verifiable,
        "model_valid_rate": valid_rate_model,
        "model_mean_adv_reject_rate": model_mean_adv_reject_rate,
        "tool_reward": r_tool,
        "repetition_penalty": cluster_penalty,
        "weights": weights,
    }


def _reward_weights() -> Dict[str, float]:
    return {
        "W_UNC": float(os.getenv("W_UNC", "0.40")),
        "W_VERIFIABLE": float(os.getenv("W_VERIFIABLE", "0.40")),
        "W_TOOL": float(os.getenv("W_TOOL", "0.10")),
        "W_REP": float(os.getenv("W_REP", "1.00")),
    }


def build_perturbations_with_results(
    executor_candidates: List[Dict],
    parsed_verifiers: List[Dict],
) -> List[Dict]:
    """Apply Phase 3 perturbations and execute valid verifiers on each perturbed trajectory."""
    engine = PerturbEngine()
    perturbations = []
    max_workers = int(os.getenv("VERIFIER_EXEC_MAX_WORKERS", "8"))
    for candidate in executor_candidates or []:
        candidate_idx = int(candidate.get("candidate_idx", 0))
        for perturbed, meta in engine.apply_all(candidate):
            row = {
                "candidate_idx": candidate_idx,
                **meta,
                "perturbed_trajectory": perturbed,
                "verifier_results": [],
            }
            if meta.get("applied_successfully"):
                row["verifier_results"] = execute_verifier_matrix(
                    parsed_verifiers=parsed_verifiers,
                    executor_results=[perturbed],
                    max_workers=max_workers,
                    is_perturbed=True,
                )
            perturbations.append(row)
    return perturbations


def _as_text(value) -> str:
    """�?numpy/object 类型安全转成字符串，避免 JSON 序列化失败�?""
    if value is None:
        return ""
    try:
        return str(value)
    except Exception:
        return repr(value)


def _answer_counts(extracted_results):
    counts = Counter([_as_text(item) for item in extracted_results if item])
    return dict(counts)


def _normalize_prompt_messages(prompt_messages):
    """DataLoader 会把 list[dict] 包成 numpy object array，这里统一还原�?list�?""
    if prompt_messages is None:
        return []
    if isinstance(prompt_messages, np.ndarray):
        prompt_messages = prompt_messages.tolist()
    if isinstance(prompt_messages, dict):
        return [prompt_messages]
    if isinstance(prompt_messages, tuple):
        prompt_messages = list(prompt_messages)
    if isinstance(prompt_messages, list):
        return prompt_messages
    return []


def _build_record(
    *,
    predict: str,
    ground_truth: str,
    prompt_text: str,
    prompt_messages,
    parsed_question: str,
    parsed_answer: str,
    final_result: Dict,
    score: Dict[str, float],
    penalty: float,
    tool_reward: float,
    uncertainty: float,
    critic_output,
    verifier_executions,
    perturbations,
    critic_scores,
    verifier_writable,
    model_valid_rate,
    model_mean_adv_reject_rate,
    reward_weights,
    global_step: int,
    rollout_id: int,
    uid: str,
):
    """把一�?curriculum rollout 整理�?Section 7.1 约定�?JSONL 行�?""
    extracted_results = final_result.get("extracted_results", [])
    answer_counts = final_result.get("answer_counts") or _answer_counts(extracted_results)
    executor_candidates = final_result.get("executor_candidates", [])
    if not executor_candidates:
        messages = final_result.get("messages_per_candidate", [])
        tools = final_result.get("tool_calls_per_candidate", [])
        outputs = final_result.get("all_outputs", [])
        for idx, output in enumerate(outputs):
            executor_candidates.append({
                "candidate_idx": idx,
                "messages": messages[idx] if idx < len(messages) else [],
                "tool_calls": tools[idx] if idx < len(tools) else [],
                "extracted_answer": extracted_results[idx] if idx < len(extracted_results) else "",
                "n_turns": len(messages[idx]) if idx < len(messages) else 0,
                "completed": bool(extracted_results[idx]) if idx < len(extracted_results) else False,
                "final_output": output,
            })

    match_curriculum_answer = bool(final_result.get("score", 0) > 0)
    prompt_system, prompt_user = "", prompt_text or ""
    prompt_messages = _normalize_prompt_messages(prompt_messages)
    if len(prompt_messages) > 0:
        try:
            prompt_system = "\n".join(
                _as_text(message.get("content", ""))
                for message in prompt_messages
                if isinstance(message, dict) and message.get("role") == "system"
            )
            prompt_user = "\n".join(
                _as_text(message.get("content", ""))
                for message in prompt_messages
                if isinstance(message, dict) and message.get("role") == "user"
            ) or prompt_user
        except Exception:
            prompt_system, prompt_user = "", prompt_text or ""
    return {
        "meta": {
            "stage": "curriculum_train",
            "experiment_name": os.getenv("EXPERIMENT_NAME", os.getenv("SAVE_NAME", "unknown_experiment")),
            "global_step": int(global_step),
            "rollout_id": int(rollout_id),
            "candidate_idx": int(rollout_id),
            "timestamp": RecordWriter.utc_now(),
            "uid": uid or None,
            "model_paths": {
                "curriculum": os.getenv("CURRICULUM_MODEL_PATH", os.getenv("MODEL_NAME", None)),
                "executor": os.getenv("EXECUTOR_MODEL_PATH", os.getenv("MODEL_NAME", None)),
                "critic": os.getenv("CRITIC_MODEL_PATH", os.getenv("MODEL_NAME", None)),
            },
            "config_snapshot": {
                "rollout_batch_size": os.getenv("CURRICULUM_ROLLOUT_BATCH_SIZE"),
                "rollout_n": os.getenv("CURRICULUM_ROLLOUT_N"),
                "executor_num_candidates": os.getenv("EXECUTOR_NUM_CANDIDATES"),
                "executor_max_turns": os.getenv("EXECUTOR_MAX_TURNS"),
                "executor_max_tokens": os.getenv("EXECUTOR_MAX_TOKENS"),
                "rev": os.getenv("GIT_REV"),
                "enable_critic": os.getenv("ENABLE_CRITIC"),
                "enable_veriplay_reward": os.getenv("ENABLE_VERIPLAY_REWARD"),
            },
        },
        "curriculum": {
            "prompt": {
                "system": prompt_system,
                "user": prompt_user,
                "text": prompt_text,
            },
            "raw_output": predict,
            "ground_truth": ground_truth,
            "parsed": {
                "question": parsed_question,
                "answer": parsed_answer,
                "format_valid": bool(parsed_question and parsed_answer),
                "n_tool_markers": len(re.findall(r"```output", predict or "")),
            },
        },
        "executor_results": executor_candidates,
        "executor_aggregation": {
            "n_candidates": len(extracted_results),
            "all_extracted_answers": extracted_results,
            "answer_counts": answer_counts,
            "majority_answer": final_result.get("answer", ""),
            "self_consistency": final_result.get("self_consistency", final_result.get("score", 0.0)),
            "match_curriculum_answer": match_curriculum_answer,
            "score_returned_to_curriculum": final_result.get("score", 0.0),
        },
        "reward_breakdown": {
            "format_valid": bool(parsed_question and parsed_answer),
            "uncertainty": uncertainty,
            "tool_reward": tool_reward,
            "repetition_penalty": penalty,
            "verifier_writable": verifier_writable,
            "model_valid_rate": model_valid_rate,
            "model_mean_adv_reject_rate": model_mean_adv_reject_rate,
            "weights": reward_weights,
            "overall": score["overall"],
        },
        "veriplay": {
            "critic_output": critic_output,
            "verifier_executions": verifier_executions,
            "perturbations": perturbations,
        },
        "critic_scores": critic_scores,
    }


def compute_score(
    predicts: List[str],
    ground_truths: List[str],
    format_weight: float = 0.1,
    file_path: str = "",
    global_step: int = 0,
    prompt_texts: List[str] = None,
    prompt_messages: List = None,
    uids: List[str] = None,
    experiment_name: str = None,
) -> List[Dict[str, float]]:
    """Main reward function called by verl.trainer.main during curriculum RL.

    For each curriculum output:
    1. Parse <question>...</question> and the final boxed answer.
    2. Let executor services solve the question multiple times with tools.
    3. Reward uncertain-but-valid questions via min(score, 1-score).
    4. Penalize repetitive questions using BLEU clustering.
    5. Add a small reward for tool-use traces.
    """
    results = []
    prompt_texts = prompt_texts or [""] * len(predicts)
    prompt_messages = prompt_messages or [[] for _ in predicts]
    uids = uids or [""] * len(predicts)
    with open('test.json','w') as f:
        json.dump(predicts,f,indent=4)
    for i in tqdm(range(len(predicts)), desc=" - Parsing predictions"):
        questions = re.findall(r"<question>(.*?)</question>", predicts[i], re.DOTALL)
        answers = extract_boxed_content(predicts[i])
        if questions and answers:
            try:
                question = questions[-1].strip()
                answer = answers[-1].strip()
                results.append({"question": question, "answer": answer})
            except:
                results.append({"question": "", "answer": ""})
        else:
            results.append({"question": "", "answer": ""})

    final_results = generate_results(results)
    critic_enabled = os.getenv("ENABLE_CRITIC", "0") == "1"
    veriplay_reward_enabled = os.getenv("ENABLE_VERIPLAY_REWARD", "0") == "1"
    critic_responses = [None] * len(final_results)
    verifier_execution_rows = [None] * len(final_results)
    perturbation_rows = [None] * len(final_results)
    critic_score_rows = [None] * len(final_results)
    if critic_enabled:
        critic_responses = call_critic_service_batch(
            questions=[result.get("question", "") for result in final_results],
            n_candidates=int(os.getenv("CRITIC_N_VERIFIERS", "3")),
            port=int(os.getenv("CRITIC_PORT", "6000")),
            timeout=int(os.getenv("CRITIC_TIMEOUT", "60")),
        )
        for idx, critic_response in enumerate(critic_responses):
            if not critic_response or critic_response.get("error"):
                verifier_execution_rows[idx] = []
                continue
            verifier_execution_rows[idx] = execute_verifier_matrix(
                parsed_verifiers=critic_response.get("parsed_verifiers", []),
                executor_results=final_results[idx].get("executor_candidates", []),
                max_workers=int(os.getenv("VERIFIER_EXEC_MAX_WORKERS", "8")),
                is_perturbed=False,
            )
            if veriplay_reward_enabled:
                perturbation_rows[idx] = build_perturbations_with_results(
                    executor_candidates=final_results[idx].get("executor_candidates", []),
                    parsed_verifiers=critic_response.get("parsed_verifiers", []),
                )
                critic_score_rows[idx] = compute_critic_rewards(
                    verifier_executions=verifier_execution_rows[idx],
                    perturbations=perturbation_rows[idx],
                    parsed_verifiers=critic_response.get("parsed_verifiers", []),
                )
    penalty = cluster_share_per_problem([result['question'] for result in final_results], distance_threshold=0.5)
    assert len(penalty) == len(final_results)
    scores = []
    writer = RecordWriter.get(experiment_name=experiment_name)
    for i in tqdm(range(len(final_results)), desc=" - Calculating final scores"):
        tool_reward = calculate_tool_reward(predicts[i])
        uncertainty = min(final_results[i]["score"], 1 - final_results[i]["score"]) if final_results[i]['question'] else -1
        if veriplay_reward_enabled:
            final_score, reward_breakdown = compute_curriculum_reward(
                final_result=final_results[i],
                parsed_question=results[i]["question"] if i < len(results) else "",
                parsed_answer=results[i]["answer"] if i < len(results) else "",
                critic_output=critic_responses[i],
                critic_scores=critic_score_rows[i],
                cluster_penalty=penalty[i],
                n_tool_markers=len(re.findall(r"```output", predicts[i] or "")),
            )
            score = {
                "overall": final_score,
                "format": 1 if final_results[i]['question'] else 0,
                "accuracy": penalty[i],
                "tool_reward": reward_breakdown["tool_reward"],
                "uncertainty": reward_breakdown["uncertainty"],
                "repetition_penalty": reward_breakdown["repetition_penalty"],
                "verifier_writable": reward_breakdown["verifier_writable"],
                "model_valid_rate": reward_breakdown["model_valid_rate"],
                "model_mean_adv_reject_rate": reward_breakdown["model_mean_adv_reject_rate"],
                "weights": reward_breakdown["weights"],
            }
            tool_reward = reward_breakdown["tool_reward"]
            uncertainty = reward_breakdown["uncertainty"]
        else:
            final_score = uncertainty - penalty[i] + tool_reward
            score = {
                "overall": final_score,
                "format": 1 if final_results[i]['question'] else 0,
                "accuracy": penalty[i],
                "tool_reward": tool_reward,
                "uncertainty": uncertainty,
                "repetition_penalty": penalty[i],
                "verifier_writable": None,
                "model_valid_rate": None,
                "model_mean_adv_reject_rate": None,
                "weights": None,
            }
        scores.append(score)
        writer.add_row(_build_record(
            predict=predicts[i],
            ground_truth=ground_truths[i] if i < len(ground_truths) else "",
            prompt_text=prompt_texts[i] if i < len(prompt_texts) else "",
            prompt_messages=prompt_messages[i] if i < len(prompt_messages) else [],
            parsed_question=results[i]["question"] if i < len(results) else "",
            parsed_answer=results[i]["answer"] if i < len(results) else "",
            final_result=final_results[i],
            score=score,
            penalty=penalty[i],
            tool_reward=tool_reward,
            uncertainty=uncertainty,
            critic_output=critic_responses[i],
            verifier_executions=verifier_execution_rows[i],
            perturbations=perturbation_rows[i],
            critic_scores=critic_score_rows[i],
            verifier_writable=score.get("verifier_writable"),
            model_valid_rate=score.get("model_valid_rate"),
            model_mean_adv_reject_rate=score.get("model_mean_adv_reject_rate"),
            reward_weights=score.get("weights"),
            global_step=global_step,
            rollout_id=i,
            uid=uids[i] if i < len(uids) else "",
        ))
    writer.flush(global_step)
    return scores
