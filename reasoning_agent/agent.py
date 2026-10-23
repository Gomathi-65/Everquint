"""Multi-step reasoning agent: planner -> executor -> verifier, with limited retries.

Usage:
    python agent.py                      # interactive CLI (mock LLM, no API key needed)
    python agent.py -q "What is 12 * 7 + 5?"
    LLM_PROVIDER=anthropic python agent.py   # real LLM (needs ANTHROPIC_API_KEY)
"""
import argparse
import json
import time
from pathlib import Path

from executor import execute
from llm import get_llm
from planner import make_plan
from verifier import verify

MAX_RETRIES = 2
TRACE_FILE = Path(__file__).parent / "logs" / "trace.jsonl"


def _write_trace(entry):
    TRACE_FILE.parent.mkdir(exist_ok=True)
    with TRACE_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def solve(question, llm=None, max_retries=MAX_RETRIES, trace=True):
    llm = llm or get_llm()
    feedback, attempts, plan, sol, ver = "", 0, {"steps": []}, None, None
    for _ in range(max_retries + 1):
        attempts += 1
        plan = make_plan(question, llm, feedback)
        sol = execute(question, plan, llm, feedback)
        ver = verify(question, sol, llm)
        if trace:  # full working (chain of thought) goes to the debug log only, never to the user
            _write_trace({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "question": question, "attempt": attempts,
                          "plan": plan, "solution": sol, "verification": ver})
        if ver["passed"] or not ver["retry_useful"]:
            break
        feedback = ver["feedback"]

    plan_text = " ".join(f"{i}. {s}" for i, s in enumerate(plan["steps"], 1))
    if ver["passed"]:
        answer, status, visible = sol["answer"], "success", sol["summary"]
    else:
        answer, status = "I could not produce a verified answer.", "failed"
        visible = sol.get("reason") if not sol["solvable"] else "Verification failed: " + ver["feedback"]
    return {"answer": answer, "status": status, "reasoning_visible_to_user": visible,
            "metadata": {"plan": plan_text, "checks": ver["checks"], "retries": attempts - 1}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-q", "--question")
    ap.add_argument("--provider", help="mock | anthropic | openai (default: $LLM_PROVIDER or mock)")
    args = ap.parse_args()
    llm = get_llm(args.provider)
    if args.question:
        print(json.dumps(solve(args.question, llm), indent=2, ensure_ascii=False))
        return
    print("Type a question (empty line to quit).")
    while True:
        q = input("> ").strip()
        if not q:
            break
        print(json.dumps(solve(q, llm), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
