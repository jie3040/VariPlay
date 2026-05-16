#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This script enhances the LLM's problem-solving capabilities by integrating a code execution tool. 
It processes each question through a multi-turn conversational approach, allowing the model to generate, execute, and reason based on code output.
The generation process for each of the 10 candidates is now a stateful, iterative loop.

Setup Instructions:
    # 1. Install required libraries
    pip install stopit flask vllm transformers torch requests

    # 2. Ensure the code execution sandbox API is running and accessible.

    # 3. Run the server
    python your_server_file_name.py --port 5000 --model_path Qwen/Qwen3-4B-Base
'''

from flask import Flask, request, jsonify
import vllm
import argparse
import json
import os
import subprocess
import threading
import time
import torch
from transformers import AutoTokenizer
from mathruler.grader import extract_boxed_content, grade_answer
import stopit
import requests
import re
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# ---------------------------- Code Execution Tool --------------------------- #

SANDBOX_API_URLS = [
    'IP1:PORT1/run_code',
    'IP2:PORT2/run_code',
    'IP3:PORT3/run_code',
    'IP4:PORT4/run_code'
]

# Round-robin state for distributing code snippets across sandbox instances.
api_counter_lock = threading.Lock()
api_counter = 0

def execute_code_in_sandbox(code: str) -> str:
    """
    Calls an external sandbox API to execute Python code, with load balancing.
    """
    if os.getenv("USE_LOCAL_SANDBOX", "0") == "1":
        # 仅用�?Phase 1.5 tool-call-positive smoke：远程复现环境没有配�?        # SandboxFusion URL 时，用本�?python3 验证 tool_calls 记录链路�?        try:
            completed = subprocess.run(
                ["python3", "-c", code],
                capture_output=True,
                text=True,
                timeout=20,
            )
            if completed.returncode == 0:
                return completed.stdout.strip() or "[No output]"
            return f"Execution failed with status: {completed.returncode}\nStderr: {completed.stderr.strip()}"
        except Exception as e:
            return f"Execution Error: {e}"

    global api_counter
    with api_counter_lock:
        # Pick the next sandbox endpoint in a thread-safe round-robin way.
        target_url = SANDBOX_API_URLS[api_counter % len(SANDBOX_API_URLS)]
        api_counter += 1

    try:
        # SandboxFusion expects a small JSON payload containing code and language.
        payload = {"code": code, "language": "python"}
        headers = {'Content-Type': 'application/json'}
        response = requests.post(target_url, headers=headers, data=json.dumps(payload), timeout=20)
        response.raise_for_status()
        result = response.json()

        if result.get("status") == "Success" and result.get("run_result"):
            run_info = result["run_result"]
            if run_info.get("status") == "Finished":
                stdout = run_info.get("stdout", "")
                return stdout if stdout else "[No output]"
            else:
                stderr = run_info.get('stderr', '')
                return f"Execution failed with status: {run_info.get('status')}\nStderr: {stderr}"
        else:
            return f"API Error: {result}"
    except Exception as e:
        return f"Execution Error: {e}"


# ---------------------------- Initial Setup --------------------------------- #

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=str, default='5000')
parser.add_argument('--model_path', type=str, default='Qwen/Qwen3-4B-Base')
parser.add_argument('--gpu_mem_util', type=float, default=0.8,
                    help='The maximum GPU memory utilization fraction for vLLM.')
parser.add_argument('--num_candidates', type=int, default=10,
                    help='每个题目采样多少�?executor 解答；smoke run 可以调小�?)
parser.add_argument('--max_turns', type=int, default=4,
                    help='executor 最多工具交互轮数；smoke run 可以调小�?)
parser.add_argument('--max_tokens', type=int, default=2048,
                    help='executor 单轮最大生�?token 数；smoke run 可以调小�?)
parser.add_argument('--max_model_len', type=int, default=None,
                    help='vLLM 最大上下文长度；smoke run 可以调到 512 节省 KV cache�?)
parser.add_argument('--enforce_eager', action='store_true',
                    help='禁用 cudagraph 捕获�?GPU smoke run 更省显存也更稳�?)
parser.add_argument('--disable_idle_worker', action='store_true',
                    help='关闭保活 GPU �?idle worker，方�?1GPU smoke run 节省显存�?)
parser.add_argument('--force_tool_call_smoke', action='store_true',
                    help='仅用�?Phase 1.5：强制构造一�?Python tool call 来验证记录链路�?)
parser.add_argument('--skip_model_load_for_smoke', action='store_true',
                    help='仅用�?force_tool_call_smoke：跳�?executor vLLM 加载，把单卡显存留给 critic�?)
args = parser.parse_args()


tokenizer = None
model = None
sampling_params_single_turn = None

if args.skip_model_load_for_smoke:
    if not args.force_tool_call_smoke:
        raise ValueError("--skip_model_load_for_smoke can only be used together with --force_tool_call_smoke")
    print('[init] Skipping executor model load for forced tool-call smoke.')
else:
    print('[init] Loading model...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # Each Flask service hosts one vLLM model instance on the GPU selected by
    # CUDA_VISIBLE_DEVICES in vllm_service_init/start.sh.
    model = vllm.LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
    )

    sampling_params_single_turn = vllm.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.7,
        top_p=0.9,
        n=1,
        stop_token_ids=[tokenizer.eos_token_id]
    )

SYSTEM_PROMPT = (
    "Solve the following problem step by step. You now have the ability to selectively write executable Python code to enhance your reasoning process.\n"
    "First, provide your reasoning and write a self-contained Python code block wrapped in ```python ... ``` to help you calculate the answer. You must use the `print()` function to output the results.\n"
    "After you write the code block, STOP. I will execute it for you.\n"
    "I will then provide the output under 'Code execution result:'. You must use this result (even if it's an error) to continue your reasoning and provide the final answer.\n"
    "The final answer must be enclosed in \\boxed{...}."
    "Code Format:\n"
    "Each code snippet is wrapped between ```. You need to use print() to output intermediate results.\n"
    "Answer Format:\n"
    "The last part of your response should be: \\boxed{...}"
)

# ---------------------------- GPU Idle Worker ------------------- #
stop_event = threading.Event()
pause_event = threading.Event()

def gpu_idle_worker():
    """Keep the GPU context warm while the Flask server waits for requests."""
    print('[idle_worker] GPU idle worker started.')
    running = True
    while not stop_event.is_set():
        if pause_event.is_set():
            if running:
                running = False
            time.sleep(0.1)
            continue
        else:
            if not running:
                running = True
        try:
            a = torch.rand((2000, 2000), dtype=torch.float32, device='cuda')
            b = torch.rand((2000, 2000), dtype=torch.float32, device='cuda')
            torch.matmul(a, b)
            torch.cuda.synchronize()
        except RuntimeError:
            time.sleep(1)
    print('[idle_worker] GPU idle worker stopped.')

idle_thread = None
if not args.disable_idle_worker:
    idle_thread = threading.Thread(target=gpu_idle_worker, daemon=True)
    idle_thread.start()

# ---------------------------- Core Logic (Refactored) ----------------------- #
@stopit.threading_timeoutable(default='TIMED_OUT')
def grade_answer_with_timeout(res1, res2):
    """Run mathruler grading with a timeout to avoid stuck symbolic checks."""
    return grade_answer(res1, res2)

# Code execution can happen for many candidates, so use a thread pool.
sandbox_executor = ThreadPoolExecutor(max_workers=64)

def generate_with_tool_use(question: str, num_candidates: int = 10, max_turns: int = 4):
    """
    Generates answers using a multi-turn conversation loop (up to max_turns).
    Handles code execution and history updates dynamically.
    """
    # Initialize conversation history for all candidates
    conversations = [[{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': question}] for _ in range(num_candidates)]
    tool_calls_per_candidate = [[] for _ in range(num_candidates)]
    final_assistant_messages = [""] * num_candidates
    active_indices = list(range(num_candidates))

    if args.force_tool_call_smoke:
        # smoke-only 分支：不依赖模型是否自发写代码，直接验证 tool_calls 的结构化记录�?        numbers = re.findall(r"-?\d+", question)
        if len(numbers) >= 2:
            code_to_run = f"print({numbers[0]} * {numbers[1]})"
        else:
            code_to_run = "print(1234567 * 7654321)"
        for idx in range(num_candidates):
            assistant_code = f"I will compute it with Python.\n```python\n{code_to_run}\n```"
            conversations[idx].append({"role": "assistant", "content": assistant_code})
            exec_result = execute_code_in_sandbox(code_to_run)
            conversations[idx].append({"role": "user", "content": f"Code execution result: {exec_result}"})
            final_assistant_messages[idx] = f"The Python result is {exec_result}. Therefore \\boxed{{{exec_result}}}."
            conversations[idx].append({"role": "assistant", "content": final_assistant_messages[idx]})
            tool_calls_per_candidate[idx].append({
                "turn": 0,
                "code": code_to_run,
                "stdout": exec_result,
                "stderr": "",
                "status": "Finished" if "Execution" not in exec_result and "failed" not in exec_result.lower() else "Error",
            })
        return final_assistant_messages, conversations, tool_calls_per_candidate

    for turn in range(max_turns):
        if not active_indices:
            break

        # Prepare prompts only for active candidates
        prompts = [tokenizer.apply_chat_template(conversations[i], tokenize=False, add_generation_prompt=True) for i in active_indices]
        
        # Batch generate
        responses = model.generate(prompts, sampling_params_single_turn, use_tqdm=False)

        tasks_to_run = []
        indices_with_code = set()

        # Step 1: Process model outputs
        for i, response in enumerate(responses):
            original_index = active_indices[i]
            model_output = response.outputs[0].text.strip()
            
            # Clean up potential incomplete code blocks
            code_block_start_tag = "```python"
            code_block_end_tag = "```"
            start_index = model_output.find(code_block_start_tag)
            if start_index != -1:
                end_index = model_output.find(code_block_end_tag, start_index + len(code_block_start_tag))
                if end_index != -1:
                    model_output = model_output[:end_index + len(code_block_end_tag)]
            
            # Update history
            conversations[original_index].append({'role': 'assistant', 'content': model_output})

            # Check for Code
            code_match = re.search(r"```python\n(.*?)\n```", model_output, re.DOTALL)
            
            # Check for Boxed Answer
            has_boxed = r'\boxed' in model_output

            if code_match and not has_boxed:
                # Found code, no final answer yet -> Queue for execution
                code_to_run = (code_match.group(1) or "").strip()
                if code_to_run:
                    future = sandbox_executor.submit(execute_code_in_sandbox, code_to_run)
                    # 记录 turn/code，等 sandbox 返回后补 stdout/status�?                    tasks_to_run.append((future, original_index, turn, code_to_run))
                    indices_with_code.add(original_index)
                else:
                    # Empty code block, treat as text step
                    pass
            elif has_boxed:
                # Found answer -> Mark as finished
                final_assistant_messages[original_index] = model_output
            else:
                # Pure text reasoning -> Will continue to next turn if logic requires, 
                # or strictly speaking, we keep it active to allow further reasoning.
                pass

        # Step 2: Collect Sandbox Results
        results_map = {}
        for future, idx, tool_turn, code_to_run in tasks_to_run:
            try:
                results_map[idx] = future.result()
            except Exception as e:
                results_map[idx] = f"Sandbox Error: {e}"
            status = "Finished"
            if "Error:" in results_map[idx] or "failed" in results_map[idx].lower() or results_map[idx] == "TIMED_OUT":
                status = "Error"
            tool_calls_per_candidate[idx].append({
                "turn": tool_turn,
                "code": code_to_run,
                "stdout": results_map[idx],
                "stderr": "",
                "status": status,
            })

        # Step 3: Prepare next turn indices
        next_active_indices = []
        for i, response in enumerate(responses):
            original_index = active_indices[i]
            
            # If we already found a boxed answer, this candidate is done.
            if final_assistant_messages[original_index]:
                continue
            
            # If it had code, append result and keep active
            if original_index in indices_with_code:
                exec_result = results_map.get(original_index, "Result not found.")
                tool_feedback = f"Code execution result: {exec_result}"
                conversations[original_index].append({'role': 'user', 'content': tool_feedback})
                next_active_indices.append(original_index)
            
            # If it was just text (and no boxed), we keep it active for the next turn
            # (assuming it needs more steps), unless it was the last turn.
            else:
                next_active_indices.append(original_index)
        
        active_indices = next_active_indices

    # Fill in any candidates that didn't finish with \boxed with their last output
    for i in range(num_candidates):
        if not final_assistant_messages[i]:
            # Use the last assistant message as the best effort result
            # Traverse backwards to find the last assistant message
            for msg in reversed(conversations[i]):
                if msg['role'] == 'assistant':
                    final_assistant_messages[i] = msg['content']
                    break
    
    return final_assistant_messages, conversations, tool_calls_per_candidate


def consolidate_and_grade(question, golden_answer, assistant_messages, messages_per_candidate=None, tool_calls_per_candidate=None):
    '''Majority-vote executor answers and compare them with the curriculum answer.'''
    results = [extract_boxed_content(msg) for msg in assistant_messages]
    
    answer_counts = {}
    for res in results:
        if not res: continue
        matched = False
        
        for exist_ans in list(answer_counts.keys()):
            if res == exist_ans or ('no ' in res.lower() and 'no ' in exist_ans.lower()):
                answer_counts[exist_ans] += 1
                matched = True
                break
            
            try:
                is_match = False
                match_result_1 = grade_answer_with_timeout(res, exist_ans, timeout=20)
                if match_result_1 and match_result_1 != 'TIMED_OUT':
                    is_match = True

                if not is_match:
                    match_result_2 = grade_answer_with_timeout(exist_ans, res, timeout=20)
                    if match_result_2 and match_result_2 != 'TIMED_OUT':
                        is_match = True
                
                if is_match:
                    answer_counts[exist_ans] += 1
                    matched = True
                    break

            except Exception:
                continue
        
        if not matched:
            answer_counts[res] = 1

    if not answer_counts:
        majority_ans, max_count = '', 0
    else:
        majority_ans = max(answer_counts, key=answer_counts.get)
        max_count = answer_counts[majority_ans]

    score = max_count / len(assistant_messages) if assistant_messages else 0.0

    executor_candidates = []
    messages_per_candidate = messages_per_candidate or [[] for _ in assistant_messages]
    tool_calls_per_candidate = tool_calls_per_candidate or [[] for _ in assistant_messages]
    for idx, output in enumerate(assistant_messages):
        executor_candidates.append({
            "candidate_idx": idx,
            "messages": messages_per_candidate[idx] if idx < len(messages_per_candidate) else [],
            "tool_calls": tool_calls_per_candidate[idx] if idx < len(tool_calls_per_candidate) else [],
            "extracted_answer": results[idx] if idx < len(results) else "",
            "n_turns": len(messages_per_candidate[idx]) if idx < len(messages_per_candidate) else 0,
            "completed": bool(results[idx]) if idx < len(results) else False,
            "final_output": output,
        })

    return {
        'question': question,
        'answer':   majority_ans,
        'score':    score if grade_answer(majority_ans, golden_answer) and score > 0.1 else 0,
        'self_consistency': score,
        'answer_counts': answer_counts,
        'all_outputs':  assistant_messages,
        'extracted_results': results,
        'messages_per_candidate': messages_per_candidate,
        'tool_calls_per_candidate': tool_calls_per_candidate,
        'executor_candidates': executor_candidates,
    }

# ---------------------------- Flask Application --------------------------- #
app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    """轻量健康检查端点，用于 smoke 脚本等待 vLLM/Flask 服务启动�?""
    return jsonify({'status': 'ok'})

@app.route('/hello', methods=['GET'])
def hello():
    """Process one temporary json file sent by curriculum_reward.generate_results.

    The input file contains a shard of generated questions. This endpoint solves
    each question with multi-turn tool use, writes a sibling *_results.json file,
    and returns a small status response.
    """
    # smoke-only 路径没有加载 executor vLLM，不需要触�?CUDA�?    # 这样可以避免和同卡上�?critic vLLM 服务互相影响�?    should_pause_gpu = not args.skip_model_load_for_smoke
    if should_pause_gpu:
        pause_event.set()
        torch.cuda.synchronize()

    name = request.args.get('name', 'None')
    
    with open(name, 'r') as f:
        data = json.load(f)
    if os.getenv("KEEP_TEMP_RESULTS", "0") != "1":
        os.remove(name)

    questions = [item.get('question', '') for item in data]
    answers   = [item.get('answer',   '') for item in data]

    results_all = []
    
    # Using TQDM for clean progress visualization
    progress_bar = tqdm(zip(questions, answers), total=len(questions), desc=f"Processing {os.path.basename(name)}")
    
    for q, a in progress_bar:
        try:
            if q and a:
                # Multi-turn generation
                final_assistant_messages, messages_per_candidate, tool_calls_per_candidate = generate_with_tool_use(
                    q,
                    num_candidates=args.num_candidates,
                    max_turns=args.max_turns,
                )
                
                # Consolidate and Grade
                item = consolidate_and_grade(
                    q,
                    a,
                    final_assistant_messages,
                    messages_per_candidate=messages_per_candidate,
                    tool_calls_per_candidate=tool_calls_per_candidate,
                )
                results_all.append(item)
            else:
                results_all.append({
                    'question': q,
                    'answer': a,
                    'score': -1,
                    'self_consistency': 0.0,
                    'answer_counts': {},
                    'all_outputs': [],
                    'extracted_results': [],
                    'messages_per_candidate': [],
                    'tool_calls_per_candidate': [],
                    'executor_candidates': [],
                })
        except Exception as e:
            # Only printing critical errors to not mess up TQDM too much
            print(f'\n[server] Error processing question: {str(e)}')
            results_all.append({
                'question': q,
                'answer': a,
                'score': -1,
                'self_consistency': 0.0,
                'answer_counts': {},
                'all_outputs': [],
                'extracted_results': [],
                'messages_per_candidate': [],
                'tool_calls_per_candidate': [],
                'executor_candidates': [],
                'error': f'unhandled exception: {str(e)}',
            })
    
    out_path = name.replace('.json', '_results.json')
    with open(out_path, 'w') as f:
        json.dump(results_all, f, indent=4)

    if should_pause_gpu:
        pause_event.clear()
    return jsonify({'message': f'Processed {name}, results saved to {out_path}.'})

# ------------------------- Main Application Entrypoint --------------------------- #
if __name__ == '__main__':
    try:
        # start.sh launches four copies of this app on ports 5000-5003.
        app.run(host='127.0.0.1', port=int(args.port), threaded=True)
    finally:
        stop_event.set()
        if idle_thread is not None and idle_thread.is_alive():
            idle_thread.join()
        print('[main] Application shutdown complete.')
