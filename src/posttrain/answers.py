"""Answer parsing, verification and GRPO reward functions.

Everything that decides whether a completion is "correct" lives here, so the
evaluation, the preference-pair builder and the RL reward all share exactly one
definition of correctness. Keeping it in one place is what makes the numbers
comparable across stages.
"""
from __future__ import annotations

import ast
import operator
import re

FINAL_MARKER = "####"
_NUM = r"-?\$?\d[\d,]*(?:\.\d+)?"
_FINAL_RE = re.compile(r"####\s*(" + _NUM + ")")
_ANY_NUM_RE = re.compile(_NUM)
_CALC_RE = re.compile(r"<<[^>]*>>")
_ANSWER_IS_RE = re.compile(r"answer is[:\s]*(" + _NUM + ")", re.IGNORECASE)
# "expression = c" for the failure analysis. The whole left-hand chain is captured
# ("5 + 3 + 2 = 10", "(2 + 3) * 4 = 20"), not just the last two operands, so
# correct multi-step lines are not misread as arithmetic errors.
_TERM = r"\(*\s*-?\$?\d[\d,]*(?:\.\d+)?\s*\)*"
_ARITH_RE = re.compile(
    r"(" + _TERM + r"(?:\s*[+\-*/x×÷]\s*" + _TERM + r")+)\s*=\s*\$?(-?\d[\d,]*(?:\.\d+)?)"
)
_MID_CHAIN = re.compile(r"(?:[\d+\-*/×÷(]|\bx)$")
_NOT_PLAIN_RHS = re.compile(r"\s*/|\s+\d+/\d|[A-Za-z]|\s*[+\-*/x×÷]\s*\(?\d")
_AST_OPS ={ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def normalize_number(text: str | None) -> float | None:
    """'$1,234.50' -> 1234.5. Returns None when the text is not a number."""
    if text is None:
        return None
    t = text.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        return float(t)
    except ValueError:
        return None


def gold_answer(gsm8k_answer: str) -> float:
    """Gold numeric answer from a GSM8K 'answer' field (text after '####')."""
    value = normalize_number(gsm8k_answer.split(FINAL_MARKER)[-1])
    if value is None:
        raise ValueError(f"Unparseable gold answer: {gsm8k_answer[-40:]!r}")
    return value


def clean_rationale(gsm8k_answer: str) -> str:
    """Strip GSM8K calculator annotations (<<48/2=24>>) and normalise the final line.

    The annotations are a tool-use artefact of the original dataset; a model
    trained on them learns to emit text it never uses.
    """
    body, _, final = gsm8k_answer.rpartition(FINAL_MARKER)
    body = _CALC_RE.sub("", body).strip()
    return f"{body}\n{FINAL_MARKER} {final.strip()}"


def extract_strict(completion: str) -> float | None:
    """Number after the first '####'. None if the model never committed to an answer."""
    m = _FINAL_RE.search(completion)
    return normalize_number(m.group(1)) if m else None


def extract_lenient(completion: str) -> float | None:
    """Best-effort answer: '####' if present, else the first 'The answer is X', else the last number.

    Base models state "The answer is 5." and then keep generating unrelated text,
    so the first explicit statement is more faithful than the last number.
    """
    strict = extract_strict(completion)
    if strict is not None:
        return strict
    m = _ANSWER_IS_RE.search(completion)
    if m:
        return normalize_number(m.group(1))
    nums = _ANY_NUM_RE.findall(completion)
    return normalize_number(nums[-1]) if nums else None


def is_correct(pred: float | None, gold: float, tol: float = 1e-4) -> bool:
    return pred is not None and abs(pred - gold) <= tol * max(1.0, abs(gold))


def truncate_completion(text: str) -> str:
    """Cut a base-model continuation where it starts inventing the next question."""
    for stop in ("\nQuestion:", "\n\nQuestion", "<|endoftext|>", "<|im_end|>"):
        idx = text.find(stop)
        if idx != -1:
            text = text[:idx]
    return text.strip()


def _safe_eval(expr: str) -> float | None:
    """Evaluate + - * / arithmetic with precedence and parentheses. None if it is not that."""
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in _AST_OPS:
            return _AST_OPS[type(node.op)](ev(node.left), ev(node.right))
        raise ValueError
    text = expr.replace(",", "").replace("$", "").replace("×", "*").replace("x", "*").replace("÷", "/")
    try:
        return ev(ast.parse(text.strip(), mode="eval").body)
    except (SyntaxError, ValueError, ZeroDivisionError, RecursionError):
        return None


def arithmetic_errors(completion: str) -> list[str]:
    """Return every 'expression = c' step whose stated result is wrong."""
    errors = []
    for m in _ARITH_RE.finditer(completion):
        before, after = completion[:m.start()].rstrip(), completion[m.end():]
        # Skip what cannot be checked reliably: a match that starts mid-expression
        # ("60 students x 2/3 = 40", "8 x 1 1/2 = 12"), a right-hand side that is not a
        # plain number ("= 1 1/2", "= 1/4", "= 1.2Y"), or a chain continuing after "=".
        if _MID_CHAIN.search(before) or _NOT_PLAIN_RHS.match(after):
            continue
        expr, c = m.group(1).strip(), m.group(2)
        true, z = _safe_eval(expr), normalize_number(c)
        if true is None or z is None:  # e.g. unbalanced brackets: not checkable, so not counted
            continue
        if abs(true - z) > 1e-2 * max(1.0, abs(true)):
            errors.append(f"{expr} = {c} (should be {true:g})")
    return errors


# --------------------------------------------------------------------------- #
# GRPO reward functions. TRL calls them with the completions plus every extra
# dataset column as a keyword argument; here that is `gold` (a float).
# --------------------------------------------------------------------------- #
def _text(c) -> str:
    # conversational completions arrive as [{"role": ..., "content": ...}]
    return c[0]["content"] if isinstance(c, list) else c


def correctness_reward(completions, gold, **kwargs) -> list[float]:
    """Verifiable reward: 1.0 if the committed '####' answer matches the gold number."""
    return [1.0 if is_correct(extract_strict(_text(c)), g) else 0.0
            for c, g in zip(completions, gold)]


def format_reward(completions, **kwargs) -> list[float]:
    """Small shaping reward: exactly one final answer line, and nothing after it."""
    out = []
    for c in completions:
        t = _text(c).strip()
        ok = t.count(FINAL_MARKER) == 1 and re.search(r"####\s*" + _NUM + r"\s*$", t) is not None
        out.append(1.0 if ok else 0.0)
    return out
