from llm import call_json
from prompts import PLANNER_SYSTEM


def make_plan(question, llm, feedback=""):
    data = call_json(llm, PLANNER_SYSTEM, {"question": question, "feedback": feedback})
    return {"problem_type": str(data.get("problem_type", "other")),
            "steps": [str(s) for s in data.get("steps", [])]}
