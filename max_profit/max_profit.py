"""Max Profit problem: choose the building mix that maximises earnings after n time units.

Buildings are built one at a time (no parallel construction). A building earns
money only once finished, for every remaining unit of time. Land is unlimited,
so parcel sizes do not affect the answer.
"""
import sys
from functools import lru_cache

# name: (build_time, earnings_per_unit)
BUILDINGS = {"T": (5, 1500), "P": (4, 1000), "C": (10, 2000)}


def max_profit(n):
    """Return (best_earnings, sorted list of (T, P, C) mixes achieving it)."""

    @lru_cache(maxsize=None)
    def solve(t):
        # Best earnings from time t onward, and the set of optimal (T,P,C) mixes.
        best, mixes = 0, {(0, 0, 0)}
        for idx, (build, rate) in enumerate(BUILDINGS.values()):
            finish = t + build
            if finish >= n:  # would earn nothing
                continue
            sub_best, sub_mixes = solve(finish)
            total = rate * (n - finish) + sub_best
            if total > best:
                best, mixes = total, set()
            if total == best:
                for m in sub_mixes:
                    m = list(m)
                    m[idx] += 1
                    mixes.add(tuple(m))
        return best, frozenset(mixes)

    best, mixes = solve(0)
    return best, sorted(mixes, reverse=True)


def format_result(n):
    best, mixes = max_profit(n)
    lines = [f"Time Unit: {n}", f"Earnings: ${best}", "Solutions"]
    for i, (t, p, c) in enumerate(mixes, 1):
        lines.append(f"{i}. T: {t} P: {p} C: {c}")
    return "\n".join(lines)


def _tests():
    assert max_profit(7) == (3000, [(1, 0, 0), (0, 1, 0)])
    assert max_profit(8) == (4500, [(1, 0, 0)])
    assert max_profit(13) == (16500, [(2, 0, 0)])
    assert max_profit(1) == (0, [(0, 0, 0)])
    print("all tests passed")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        _tests()
    else:
        n = int(sys.argv[1]) if len(sys.argv) > 1 else int(input("Time unit: "))
        print(format_result(n))