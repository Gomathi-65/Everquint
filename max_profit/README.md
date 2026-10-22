# Max Profit

Given `n` units of time, find the mix of Theatres (T), Pubs (P) and Commercial Parks (C) that earns the most, and list every mix that ties for the best.

## How to run

Requires Python 3.8+. No external packages.

```
python max_profit.py 13      # run for n = 13
python max_profit.py         # prompts for n
python max_profit.py test    # run the built-in checks
```

Example output:

```
Time Unit: 7
Earnings: $3000
Solutions
1. T: 1 P: 0 C: 0
2. T: 0 P: 1 C: 0
```

## Problem rules used

| Building | Build time | Earnings per unit of time |
|---|---|---|
| Theatre (T) | 5 | $1500 |
| Pub (P) | 4 | $1000 |
| Commercial Park (C) | 10 | $2000 |

## Assumptions

- Only one building can be under construction at a time, so buildings are built one after another.
- A building earns money only after it is finished, for each remaining unit up to `n`.
- Land is unlimited, so the parcel sizes (2x1, 1x1, 3x1) do not change the answer.
- A building that finishes at or after time `n` earns nothing, so it is never built.
- Waiting never helps (building earlier always earns at least as much), so idle time is not modelled.

## Approach

Dynamic programming over time. `solve(t)` returns the best earnings from time `t` onwards and the set of `(T, P, C)` mixes that reach it. At each step it tries each building type: starting it at `t` finishes at `t + build_time`, earns `rate * (n - finish)`, and adds the best result from `finish` onwards. Results are cached with `lru_cache`.

All mixes that tie for the maximum are kept, which is why test case 1 (`n = 7`) returns two solutions.

Complexity: O(n) states with 3 transitions each. The solution sets are small, so this runs instantly for large `n`.

## Verified against the examples

| n | Earnings | Solutions |
|---|---|---|
| 7 | $3000 | T:1 P:0 C:0 and T:0 P:1 C:0 |
| 8 | $4500 | T:1 P:0 C:0 |
| 13 | $16500 | T:2 P:0 C:0 |

## Files

- `max_profit.py` - solution, command line entry point and tests.