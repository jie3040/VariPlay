from critic_service_init.parser import parse_verifier_code


def test_parse_valid_code_block():
    raw = """Some text.
```python
def check(trajectory):
    return True, -1
```
"""
    parsed = parse_verifier_code(raw, 0)
    assert parsed["valid"] is True
    assert parsed["syntax_error"] is None
    assert "def check" in parsed["code"]
    assert parsed["source"] == "model"
    assert parsed["fallback_reason"] is None


def test_parse_last_code_block():
    raw = """```python
def helper():
    pass
```
Final:
```python
def check(trajectory):
    return False, 2
```
"""
    parsed = parse_verifier_code(raw, 1)
    assert parsed["valid"] is True
    assert "return False, 2" in parsed["code"]


def test_no_code_block_invalid():
    parsed = parse_verifier_code("def check(trajectory): return True, -1", 2)
    assert parsed["valid"] is False
    assert parsed["syntax_error"] == "No python code block found"


def test_missing_check_invalid():
    raw = """```python
def verify(trajectory):
    return True, -1
```"""
    parsed = parse_verifier_code(raw, 3)
    assert parsed["valid"] is False
    assert parsed["syntax_error"] == "check() function not defined"


def test_syntax_error_invalid():
    raw = """```python
def check(trajectory):
    if True
        return True, -1
```"""
    parsed = parse_verifier_code(raw, 4)
    assert parsed["valid"] is False
    assert "SyntaxError" in parsed["syntax_error"]


def test_parse_fallback_source_metadata():
    raw = """```python
def check(trajectory):
    return True, -1
```"""
    parsed = parse_verifier_code(
        raw,
        5,
        source="fallback",
        fallback_reason="validity_below_threshold",
    )
    assert parsed["valid"] is True
    assert parsed["source"] == "fallback"
    assert parsed["fallback_reason"] == "validity_below_threshold"
