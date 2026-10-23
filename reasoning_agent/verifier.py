import math
import re

from llm import call_json
from prompts import RESOLVER_SYSTEM, REVIEWER_SYSTEM
from tools import parse_slots


def values_equal(a, b):
    if isinstance(a, list) or isinstance(b, list):
        return isinstance(a, list) and isinstance(b, list) and sorted(map(str, a)) == sorted(map(str, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        return math.isclose(a, b, abs_tol=1e-6)
    return str(a).strip().lower() == str(b).strip().lower()


def constraint_problems(question, sol):
    """Rule-based sanity checks that do not depend on the LLM."""
    v, t, ql = sol["answer_value"], sol["answer_type"], question.lower()
    problems = []
    if t in ("number", "duration_minutes"):
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return ["answer_value is not a number"]
        if v < 0:
            problems.append(f"result is negative ({v})")
        if "how many" in ql and abs(v - round(v)) > 1e-9:
            problems.append(f"a count must be a whole number, got {v}")
    elif t == "slot_list":
        given = parse_slots(question)
        m = re.search(r"needs?\s+(?:a\s+)?(\d+)[\s-]*minutes?", ql)
        need = int(m.group(1)) if m else 0
        for s in v or []:
            if s not in given:
                problems.append(f"{s} is not one of the given slots")
            else:
                a, b = s.split("-")
                length = (int(b[:2]) * 60 + int(b[3:])) - (int(a[:2]) * 60 + int(a[3:]))
                if length < need:
                    problems.append(f"{s} is only {length} min, shorter than {need}")
    elif not str(v).strip():
        problems.append("empty answer")
    return problems


def verify(question, sol, llm):
    """Returns dict(passed, checks, retry_useful, feedback)."""
    checks = []

    def add(name, passed, details):
        checks.append({"check_name": name, "passed": bool(passed), "details": details})

    resolve = call_json(llm, RESOLVER_SYSTEM, {"question": question})
    if not sol["solvable"]:
        agree = not resolve.get("solvable", False)
        add("solvable", False, sol["reason"])
        add("independent_resolve", agree, "Independent re-solve also found it unsolvable." if agree
            else f"Independent re-solve found an answer: {resolve.get('answer')}")
    else:
        add("solvable", True, "A solution was produced.")
        add("tool_execution", not sol["tool_errors"], "; ".join(sol["tool_errors"]) or "All tool calls ran.")
        idx = sol["answer_tool"]
        if isinstance(idx, int) and 0 <= idx < len(sol["tool_results"]):
            tv = sol["tool_results"][idx]
            add("tool_consistency", values_equal(tv, sol["answer_value"]),
                f"tool result {tv!r} vs proposed answer {sol['answer_value']!r}")
        else:
            add("tool_consistency", True, "No tool output to compare against.")
        problems = constraint_problems(question, sol)
        add("constraints", not problems, "; ".join(problems) or "Constraints satisfied.")
        agree = resolve.get("solvable", False) and values_equal(resolve.get("answer_value"), sol["answer_value"])
        add("independent_resolve", agree, f"re-solve gave {resolve.get('answer_value')!r}, proposed {sol['answer_value']!r}")
        review = call_json(llm, REVIEWER_SYSTEM, {"question": question, "solution": {
            "answer": sol["answer"], "summary": sol["summary"], "working": sol["working"]}})
        add("explanation_review", review.get("approved", False), "; ".join(review.get("issues", [])) or "Explanation is consistent.")

    failed = [c for c in checks if not c["passed"]]
    names = {c["check_name"] for c in failed}
    # If an independent re-solve agrees with the executor and only the problem itself is at fault
    # (unsolvable data or an impossible count), retrying the same problem will not help.
    retry_useful = bool(failed) and not (agree and names <= {"solvable", "constraints"})
    feedback = " | ".join(f"{c['check_name']}: {c['details']}" for c in failed)
    return {"passed": not failed, "checks": checks, "retry_useful": retry_useful, "feedback": feedback}
