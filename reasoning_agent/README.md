# Multi-Step Reasoning Agent with Self-Checking

An agent that solves word problems (time, counting, percentages, rates, scheduling) in three internal phases, checks its own work, retries a limited number of times, and returns only a short answer to the user.

```
question -> Planner -> Executor -> Verifier --pass--> JSON answer
               ^                      |
               +---- feedback --------+  (up to 2 retries, else status = "failed")
```

## How to run

Python 3.9+. The default mock LLM needs no API key and no packages.

```
python agent.py                                   # interactive CLI
python agent.py -q "What is 12 * 7 + 5?"          # single question
python tests/run_tests.py                         # test suite + writes logs/example_runs.md
```

Real LLM (same code, one setting):

```
pip install anthropic            # or: pip install openai
export ANTHROPIC_API_KEY=...     # Windows PowerShell: $env:ANTHROPIC_API_KEY="..."
export LLM_PROVIDER=anthropic    # or openai (OPENAI_API_KEY)
python agent.py
```

Model names can be overridden with `ANTHROPIC_MODEL` / `OPENAI_MODEL`. You can also call it from code: `from agent import solve; solve("...")` returns the dict below.

## Output

```json
{
  "answer": "3 hours 35 minutes",
  "status": "success",
  "reasoning_visible_to_user": "Departure 14:30, arrival 18:05: 215 minutes.",
  "metadata": {"plan": "1. ... 2. ...", "checks": [{"check_name": "...", "passed": true, "details": "..."}], "retries": 0}
}
```

Users only need `answer` and `reasoning_visible_to_user`. `metadata` is for debugging. The full working (chain of thought) is written only to `logs/trace.jsonl`, never returned.

## Project structure

| File | Role |
|---|---|
| `agent.py` | Agent loop, retry logic, CLI, output schema |
| `prompts.py` | **All prompts live here** (planner, executor, resolver, reviewer), each with 3 examples and a fixed JSON format |
| `planner.py` / `executor.py` / `verifier.py` | The three phases, each in its own module |
| `tools.py` | Deterministic tools: `calc`, `time_diff`, `fit_slots` |
| `llm.py` | `LLM` interface, Anthropic and OpenAI clients, JSON parsing helper |
| `mock_llm.py` | Rule-based stand-in so everything runs without a key |
| `tests/run_tests.py` | 8 easy + 8 tricky cases, tool unit tests |
| `logs/example_runs.md` | Example run logs (question, final JSON, verifier passed, retried) |

## The three phases

1. **Planner** reads the question and returns a short numbered plan.
2. **Executor** follows the plan. The LLM proposes tool calls and an answer; the code then runs the tools, so arithmetic is never trusted to the model.
3. **Verifier** runs six checks:
   - `solvable`: did the executor find an answer?
   - `tool_execution`: did every tool call run without error?
   - `tool_consistency`: does the proposed answer equal the tool output?
   - `constraints`: rule-based sanity checks (no negative results, counts are whole numbers, returned slots are real and long enough).
   - `independent_resolve`: a second LLM call solves the question from scratch, without seeing the plan or solution, and the answers are compared.
   - `explanation_review`: an LLM reviewer looks for contradictions between the answer and the explanation.

If any check fails, the failure details are sent back to the planner and executor as feedback and the loop retries, up to `MAX_RETRIES = 2`. If the independent re-solve agrees with the executor but the problem itself is impossible (missing data, a fractional count), retrying is pointless, so the agent stops and returns `status: "failed"` with the reason.

## Assumptions

- A train arriving at an earlier clock time than it left (23:30 -> 02:15) arrived the next day.
- A slot that is exactly as long as the meeting fits.
- "How many ..." questions must have whole-number answers; otherwise the agent reports failure instead of guessing.
- The mock LLM only understands the question patterns in the tests. It exists to demonstrate the agent loop and to run offline. With `flaky=True` it makes realistic first-attempt mistakes (midnight, boundary, forgotten base amount) so retries can be shown.

## Prompt documentation

**Why the prompts are designed this way**
- Four separate prompts with one job each. The planner never solves, the executor never plans, and the resolver never sees the plan, so its answer is an independent check.
- Every prompt fixes a JSON output format and gives 3 examples, one of which shows a failure case (feedback for the planner, `solvable: false` for the executor, a rejected solution for the reviewer).
- The executor must call tools for all arithmetic. LLMs are unreliable at arithmetic and time maths, and a tool result can be compared mechanically with the proposed answer.
- The executor returns `summary` (short, user-facing) separately from `working` (detailed, debug only), which is how the chain of thought stays out of the user's view.

**Design lessons**
- Relying on the LLM reviewer alone is weak: a reviewer reading a consistent but wrong explanation will approve it. That is why the independent re-solve and the code-based checks exist. In the flaky tests the reviewer approves every wrong answer, and the re-solve is what catches them.
- Retrying with the same prompt gives the same mistake, so the verifier's failure details are passed back as feedback.
- Retrying an impossible question wastes calls, so the loop stops when the re-solve agrees with the executor and only the problem itself is at fault.

**Notes from real-LLM runs**
Not filled in yet. I built and tested everything against the mock LLM; the Anthropic and OpenAI clients have not been run against a live API. After running with a real key, add what actually failed here (formatting of JSON, ambiguous questions, retries that did not help).

**What I would improve with more time**
- Run the suite against real models and tune the prompts from the failures.
- Add more problem types and a larger evaluation set with accuracy numbers.
- Use structured-output or tool-calling APIs instead of parsing JSON from text.
- Let the verifier use a different model from the executor to reduce correlated mistakes.
- Log token counts and latency per phase.
