"""Parse Critic-Coder outputs into executable verifier code."""

import re
from typing import Dict, Optional


def parse_verifier_code(
    raw_output: str,
    verifier_idx: int,
    source: str = "model",
    fallback_reason: Optional[str] = None,
) -> Dict:
    """�?LLM 输出中提取最后一�?Python code block，并做基础语法检查�?""
    base = {
        "verifier_idx": verifier_idx,
        "source": source,
        "fallback_reason": fallback_reason,
    }
    matches = re.findall(r"```python\s*\n(.*?)\n```", raw_output or "", re.DOTALL)
    if not matches:
        return {
            **base,
            "valid": False,
            "code": None,
            "syntax_error": "No python code block found",
        }

    code = matches[-1].strip()
    if "def check" not in code:
        return {
            **base,
            "valid": False,
            "code": code,
            "syntax_error": "check() function not defined",
        }

    try:
        compile(code, f"<verifier_{verifier_idx}>", "exec")
    except SyntaxError as exc:
        return {
            **base,
            "valid": False,
            "code": code,
            "syntax_error": f"SyntaxError: {exc}",
        }

    return {
        **base,
        "valid": True,
        "code": code,
        "syntax_error": None,
    }
