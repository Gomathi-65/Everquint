"""A rule-based stand-in for an LLM so the agent runs without an API key.

It plays four roles (planner, executor, resolver, reviewer) behind the same
`complete(system, user)` interface as the real providers. With flaky=True the
executor makes realistic first-attempt mistakes (midnight, boundary, forgotten
base amount) so the verifier and retry loop can be demonstrated. Retries
(feedback present) are answered correctly.
"""
import json
import re

from llm import LLM
from tools import calc, fit_slots, fmt_duration, parse_slots, time_diff, to_minutes

TIME = r"\b(\d{1,2}:\d{2})\b"
MULT = {"twice": 2, "double": 2, "thrice": 3, "half": 0.5, "two times": 2, "three times": 3,
        "four times": 4, "five times": 5, "ten times": 10}


def _num(s):
    return float(s) if "." in s else int(s)


def build_spec(q, naive=False):
    """Solve a question the way an LLM+tools would. Returns None when unsolvable."""
    ql = q.lower()

    slots = parse_slots(q)
    need_m = re.search(r"needs?\s+(?:a\s+)?(\d+)[\s-]*minutes?", ql)
    if slots and need_m:
        need = int(need_m.group(1))
        fit = fit_slots(slots, need)
        if naive:  # boundary bug: a slot of exactly `need` minutes is wrongly rejected
            fit = [s for s in slots if to_minutes(s.split("-")[1]) - to_minutes(s.split("-")[0]) > need]
        return dict(kind="scheduling", answer_type="slot_list", answer_value=fit,
                    tool_calls=[{"tool": "fit_slots", "args": {"slots": slots, "need": need}}], answer_tool=0,
                    answer=", ".join(fit) if fit else "None of the slots can fit the meeting",
                    summary=f"A slot fits if it is at least {need} minutes long; {len(fit)} of {len(slots)} slots qualify.",
                    working=f"need={need}; lengths checked for {slots}; fitting: {fit}")

    times = re.findall(TIME, q)
    if len(times) == 2 and re.search(r"how long|duration|how much time|journey|how many (hours|minutes)", ql):
        s, e = times
        nd = bool(re.search(r"next day|following day|overnight|next morning|day after", ql)) or to_minutes(e) < to_minutes(s)
        mins = time_diff(s, e, nd)
        calls = [{"tool": "time_diff", "args": {"start": s, "end": e, "next_day": nd}}]
        if naive and nd:  # midnight bug: plain absolute difference, no next-day handling
            mins, calls = abs(to_minutes(e) - to_minutes(s)), []
        return dict(kind="time_difference", answer_type="duration_minutes", answer_value=mins,
                    tool_calls=calls, answer_tool=0 if calls else None, answer=fmt_duration(mins),
                    summary=f"Departure {s}, arrival {e}{' (next day)' if nd else ''}: {mins} minutes.",
                    working=f"{e} - {s}{' + 24h' if nd else ''} = {mins} min")

    m = re.search(r"(\d+) boxes (?:of|with) (\d+) \w+(?: each)?.*?(?:sells?|gives? away|uses?) (\d+).*?(\d+) more boxes", ql)
    if m:
        a, b, c, d = m.groups()
        return _calc_spec("counting", f"{a}*{b} - {c} + {d}*{b}", "pens" if "pens" in ql else "items",
                          f"{a} boxes of {b}, minus {c}, plus {d} boxes of {b}.")

    m = re.search(r"\$?(\d+(?:\.\d+)?)\D+?discount\w*\D+?(\d+(?:\.\d+)?)%\D+?(\d+(?:\.\d+)?)%\s*(?:tax|vat|gst)", ql)
    if m:
        p, disc, tax = m.groups()
        return _calc_spec("percentage", f"{p}*(1-{disc}/100)*(1+{tax}/100)", "$",
                          f"Price {p}, {disc}% discount, then {tax}% tax.")

    m = re.search(r"(\d+(?:\.\d+)?)\s*km/h.*?(\d+(?:\.\d+)?)\s*hours?", ql)
    if m and "how far" in ql:
        return _calc_spec("rate", f"{m[1]}*{m[2]}", "km", f"Distance = speed {m[1]} km/h x time {m[2]} h.")
    m = re.search(r"(\d+(?:\.\d+)?)\s*km\s+in\s+(\d+(?:\.\d+)?)\s*hours?", ql)
    if m and "speed" in ql:
        return _calc_spec("rate", f"{m[1]}/{m[2]}", "km/h", f"Speed = {m[1]} km / {m[2]} h.")

    m = re.search(r"has (\d+) (\w+) (\w+) and (twice|double|thrice|half|\w+ times) as many (\w+)", ql)
    if m:
        base, _, noun, word, _ = m.groups()
        mult = MULT.get(word) or _num(word.split()[0])
        expr = f"{base} + {base}*{mult}"
        if naive:  # forgets to add the base amount
            expr = f"{base}*{mult}"
        r = re.search(r"(?:gives? away|eats?|loses?|sells?|sold)\s+(\d+)", ql)
        if r:
            expr += f" - {r[1]}"
        return _calc_spec("counting", expr, noun, f"Base {base}, multiplier {mult}" + (f", minus {r[1]}." if r else "."))

    m = re.search(r"what is (\d+(?:\.\d+)?)% of (\d+(?:\.\d+)?)", ql)
    if m:
        return _calc_spec("percentage", f"{m[2]}*{m[1]}/100", "", f"{m[1]}% of {m[2]}.")

    m = re.search(r"what is ([\d\s+\-*/().x\u00d7]+)\?", ql)
    if m and re.search(r"[+\-*/x\u00d7]", m[1]):
        return _calc_spec("other", m[1].replace("x", "*").replace("\u00d7", "*").strip(), "", "Direct arithmetic.")
    return None


def _calc_spec(kind, expr, unit, summary):
    v = calc(expr)
    if unit == "$":
        text = f"${v:,.2f}"
    else:
        text = f"{v} {unit}".strip()
    return dict(kind=kind, answer_type="number", answer_value=v, tool_calls=[{"tool": "calc", "args": {"expr": expr}}],
                answer_tool=0, answer=text, summary=f"{summary} Result: {text}.", working=f"{expr} = {v}")


PLANS = {
    "time_difference": ["Extract departure and arrival times", "Convert both to minutes since midnight",
                        "If arrival is earlier than departure, treat it as the next day",
                        "Subtract and format as hours and minutes", "Check the duration is positive"],
    "scheduling": ["Extract the needed length and the free slots", "Compute each slot's length",
                   "Keep slots whose length is at least the needed length (equal fits)", "List the fitting slots"],
    "counting": ["Extract the starting quantities", "Apply multipliers and additions or removals",
                 "Compute the total with the calculator", "Check the result is a whole, non-negative count"],
    "percentage": ["Extract the base value and percentages", "Apply each percentage in order",
                   "Compute with the calculator", "Format the result"],
    "rate": ["Extract speed, distance and time", "Pick the right formula", "Compute with the calculator", "Format with units"],
    "other": ["Parse the expression", "Evaluate with the calculator", "Report the result"],
}


class MockLLM(LLM):
    def __init__(self, flaky=False):
        self.flaky = flaky

    def complete(self, system, user):
        role = system.split("\n", 1)[0].strip()
        payload, _ = json.JSONDecoder().raw_decode(user)
        handler = {"ROLE: PLANNER": self._plan, "ROLE: EXECUTOR": self._execute,
                   "ROLE: RESOLVER": self._resolve, "ROLE: REVIEWER": self._review}[role]
        return json.dumps(handler(payload))

    def _plan(self, p):
        spec = build_spec(p["question"])
        if spec is None:
            return {"problem_type": "other", "steps": ["Identify what is asked", "List the quantities given",
                                                       "Report if data is missing"]}
        return {"problem_type": spec["kind"], "steps": PLANS[spec["kind"]]}

    def _execute(self, p):
        naive = self.flaky and not p.get("feedback")
        spec = build_spec(p["question"], naive=naive)
        if spec is None:
            return {"solvable": False, "reason": "The question does not give enough information to compute an answer."}
        return {"solvable": True, "quantities": {}, "tool_calls": spec["tool_calls"], "answer_tool": spec["answer_tool"],
                "answer_type": spec["answer_type"], "answer_value": spec["answer_value"], "answer": spec["answer"],
                "summary": spec["summary"], "working": spec["working"]}

    def _resolve(self, p):
        spec = build_spec(p["question"])
        if spec is None:
            return {"solvable": False, "reason": "Not enough information."}
        return {"solvable": True, "answer_value": spec["answer_value"], "answer": spec["answer"]}

    def _review(self, p):
        s = p["solution"]
        issues = []
        if not s.get("working"):
            issues.append("No working was provided.")
        if not s.get("answer"):
            issues.append("No answer was provided.")
        return {"approved": not issues, "issues": issues}
