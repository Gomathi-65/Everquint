from llm import call_json
from prompts import EXECUTOR_SYSTEM
from tools import run_tool


def execute(question, plan, llm, feedback=""):
    """Follow the plan: ask the LLM for tool calls + a proposed answer, then run the tools in code."""
    data = call_json(llm, EXECUTOR_SYSTEM, {"question": question, "plan": plan["steps"], "feedback": feedback})
    if not data.get("solvable", False):
        return {"solvable": False, "reason": str(data.get("reason", "unsolvable")), "tool_results": [], "tool_errors": []}
    results, errors = [], []
    for call in data.get("tool_calls", []):
        try:
            results.append(run_tool(call["tool"], call.get("args", {})))
        except Exception as e:
            results.append(None)
            errors.append(f"{call.get('tool')}: {e}")
    return {"solvable": True, "answer_type": data.get("answer_type", "text"), "answer_value": data.get("answer_value"),
            "answer": str(data.get("answer", "")), "summary": str(data.get("summary", "")),
            "working": str(data.get("working", "")), "answer_tool": data.get("answer_tool"),
            "tool_calls": data.get("tool_calls", []), "tool_results": results, "tool_errors": errors}
