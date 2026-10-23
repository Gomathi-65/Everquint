"""Deterministic tools the executor can call. The LLM plans; code does the arithmetic."""
import ast
import operator
import re

_TIME_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*$")
_SLOT_RE = re.compile(r"(\d{1,2}:\d{2})\s*[\u2013\u2014-]\s*(\d{1,2}:\d{2})")


def to_minutes(t):
    m = _TIME_RE.match(str(t))
    if not m:
        raise ValueError(f"bad time: {t!r}")
    h, mi = int(m[1]), int(m[2])
    if not (0 <= h <= 23 and 0 <= mi <= 59):
        raise ValueError(f"time out of range: {t!r}")
    return h * 60 + mi


def fmt_time(minutes):
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def fmt_duration(minutes):
    h, m = divmod(int(minutes), 60)
    parts = []
    if h:
        parts.append(f"{h} hour{'s' if h != 1 else ''}")
    if m or not parts:
        parts.append(f"{m} minute{'s' if m != 1 else ''}")
    return " ".join(parts)


def parse_slots(text):
    """Find 'HH:MM-HH:MM' ranges in text and return them normalised as '09:00-09:30'."""
    return [f"{fmt_time(to_minutes(a))}-{fmt_time(to_minutes(b))}" for a, b in _SLOT_RE.findall(text)]


def time_diff(start, end, next_day=False):
    """Minutes from start to end. Set next_day=True when the end is on the following day."""
    d = to_minutes(end) - to_minutes(start)
    if next_day:
        return d + 1440
    if d < 0:
        raise ValueError("end is before start; set next_day=true if the trip crosses midnight")
    return d


def fit_slots(slots, need):
    """Return the slots that are at least `need` minutes long (a slot of exactly `need` fits)."""
    out = []
    for s in slots:
        a, b = s.split("-")
        if to_minutes(b) - to_minutes(a) >= need:
            out.append(s)
    return out


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.USub: operator.neg, ast.UAdd: operator.pos}


def calc(expr):
    """Safely evaluate + - * / and parentheses."""
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("unsupported expression")
    v = ev(ast.parse(str(expr), mode="eval"))
    if isinstance(v, float):
        v = round(v, 6)
        if v.is_integer():
            v = int(v)
    return v


TOOLS = {"time_diff": time_diff, "fit_slots": fit_slots, "calc": calc}


def run_tool(name, args):
    if name not in TOOLS:
        raise ValueError(f"unknown tool {name!r}")
    return TOOLS[name](**args)
