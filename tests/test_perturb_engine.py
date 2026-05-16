from critic_service_init.perturb_engine import PerturbEngine


def sample_traj():
    return {
        "candidate_idx": 0,
        "messages": [
            {"role": "system", "content": "solve with tools"},
            {"role": "user", "content": "compute 6*7"},
            {"role": "assistant", "content": "```python\nprint(6 * 7)\n```"},
            {"role": "user", "content": "Code execution result: 42"},
            {"role": "assistant", "content": "Therefore \\boxed{42}."},
        ],
        "tool_calls": [
            {"turn": 0, "code": "print(6 * 7)", "stdout": "42", "stderr": "", "status": "Finished"}
        ],
        "extracted_answer": "42",
    }


def two_tool_traj():
    traj = sample_traj()
    traj["messages"] = traj["messages"][:-1] + [
        {"role": "assistant", "content": "```python\nprint(40 + 2)\n```"},
        {"role": "user", "content": "Code execution result: 42"},
        {"role": "assistant", "content": "Therefore \\boxed{42}."},
    ]
    traj["tool_calls"].append(
        {"turn": 1, "code": "print(40 + 2)", "stdout": "42", "stderr": "", "status": "Finished"}
    )
    return traj


def test_apply_all_returns_five_types():
    results = PerturbEngine().apply_all(sample_traj())
    assert [meta["perturbation_type"] for _, meta in results] == [
        "arg_mutation",
        "step_drop",
        "step_swap",
        "early_terminate",
        "tool_substitute",
    ]


def test_arg_mutation_changes_stdout_and_message():
    perturbed, meta = PerturbEngine().arg_mutation(sample_traj())
    assert meta["applied_successfully"] is True
    assert perturbed["tool_calls"][0]["stdout"] == "43"
    assert "Code execution result: 43" in perturbed["messages"][3]["content"]


def test_step_drop_removes_tool_pair():
    perturbed, meta = PerturbEngine().step_drop(sample_traj())
    assert meta["applied_successfully"] is True
    assert len(perturbed["tool_calls"]) == 0
    assert all("Code execution result:" not in m["content"] for m in perturbed["messages"])


def test_step_swap_swaps_two_tool_turns():
    perturbed, meta = PerturbEngine().step_swap(two_tool_traj())
    assert meta["applied_successfully"] is True
    assert perturbed["tool_calls"][0]["code"] == "print(40 + 2)"
    assert meta["perturbation_params"]["swapped_turn_pairs"]


def test_early_terminate_injects_wrong_answer():
    perturbed, meta = PerturbEngine().early_terminate(sample_traj())
    assert meta["applied_successfully"] is True
    assert perturbed["tool_calls"] == []
    assert perturbed["extracted_answer"] == "42_perturbed"
    assert "\\boxed{42_perturbed}" in perturbed["messages"][-1]["content"]


def test_tool_substitute_changes_code_fence_and_code():
    perturbed, meta = PerturbEngine().tool_substitute(sample_traj())
    assert meta["applied_successfully"] is True
    assert "```bash" in perturbed["messages"][2]["content"]
    assert perturbed["tool_calls"][0]["code"].startswith("bash ")


def test_no_tool_call_edges_are_recorded():
    traj = {"messages": [{"role": "user", "content": "hi"}], "tool_calls": [], "extracted_answer": ""}
    engine = PerturbEngine()
    _, arg_meta = engine.arg_mutation(traj)
    _, tool_meta = engine.tool_substitute(traj)
    early, early_meta = engine.early_terminate(traj)
    assert arg_meta["applied_successfully"] is False
    assert tool_meta["skip_reason"] == "no_tool_call_to_perturb"
    assert early_meta["applied_successfully"] is True
    assert early["messages"]


def test_special_characters_survive_perturbations():
    traj = sample_traj()
    traj["messages"][1]["content"] = "计算 6×7；保留符号 ✓"
    perturbed, meta = PerturbEngine().arg_mutation(traj)
    assert meta["applied_successfully"] is True
    assert "计算 6×7" in perturbed["messages"][1]["content"]
