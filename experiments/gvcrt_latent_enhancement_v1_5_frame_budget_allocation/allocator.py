"""Exact finite-candidate GOP allocation in serialized integer bytes."""
import itertools
from decimal import Decimal


def _score(value):
    return Decimal.from_float(float(value))


def exact_frame_dp(frames, budget, overhead_bytes, order):
    """Return the exact best streamed assignment, or the zero/no-stream fallback."""
    zero = next(c for c in frames[0] if c["name"] == "Z")
    fallback = (sum((_score(next(c for c in row if c["name"] == "Z")["d"]) for row in frames), Decimal()), 0,
                tuple(order.index("Z") for _ in frames), ["Z"] * len(frames))
    if budget < overhead_bytes:
        return result(fallback, budget, False)

    # State values are (objective, candidate-index tuple, candidate-name list).
    states = {overhead_bytes: (Decimal(), (), [])}
    for row in frames:
        expanded = {}
        for used, (objective, indices, names) in states.items():
            for candidate in row:
                total = used + candidate["bytes"]
                if total > budget:
                    continue
                value = (objective + _score(candidate["d"]), indices + (order.index(candidate["name"]),),
                         names + [candidate["name"]])
                old = expanded.get(total)
                if old is None or (value[0], value[1]) < (old[0], old[1]):
                    expanded[total] = value
        # Keep all byte totals. Equal-objective prefixes cannot be Pareto-pruned:
        # a costlier prefix can later tie on total bytes with a lexicographically
        # preferable candidate tuple.
        states = expanded
        if not states:
            return result(fallback, budget, False)

    choices = [(v[0], used, v[1], v[2]) for used, v in states.items()]
    best = min(choices + [fallback], key=lambda x: (x[0], x[1], x[2]))
    return result(best, budget, best[1] != 0)


def exact_sequence(candidates, budget, order):
    feasible = [c for c in candidates if c["bytes"] <= budget]
    chosen = min(feasible, key=lambda c: (c["d"], c["bytes"], order.index(c["name"])))
    return {"objective": float(chosen["d"]), "actual_bytes": chosen["bytes"], "budget_bytes": budget,
            "stream_sent": chosen["bytes"] > 0, "choices": list(chosen["choices"])}


def result(value, budget, sent):
    return {"objective": float(value[0]), "actual_bytes": value[1], "budget_bytes": budget,
            "stream_sent": sent, "choices": list(value[3])}


def brute_force(frames, budget, overhead_bytes, order):
    zero_d = sum((_score(next(c for c in row if c["name"] == "Z")["d"]) for row in frames), Decimal())
    possibilities = [(zero_d, 0, tuple(order.index("Z") for _ in frames), ["Z"] * len(frames))]
    for combination in itertools.product(*frames):
        used = overhead_bytes + sum(c["bytes"] for c in combination)
        if used <= budget:
            possibilities.append((sum((_score(c["d"]) for c in combination), Decimal()), used,
                                  tuple(order.index(c["name"]) for c in combination),
                                  [c["name"] for c in combination]))
    return result(min(possibilities, key=lambda x: (x[0], x[1], x[2])), budget,
                  min(possibilities, key=lambda x: (x[0], x[1], x[2]))[1] != 0)


def self_test():
    order = ["Z", "E", "O1", "O2"]
    rows = [
        [{"name": "Z", "bytes": 1, "d": .9}, {"name": "E", "bytes": 4, "d": .6},
         {"name": "O1", "bytes": 3, "d": .7}, {"name": "O2", "bytes": 2, "d": .8}],
        [{"name": "Z", "bytes": 1, "d": .8}, {"name": "E", "bytes": 4, "d": .5},
         {"name": "O1", "bytes": 3, "d": .6}, {"name": "O2", "bytes": 2, "d": .7}],
        [{"name": "Z", "bytes": 1, "d": .7}, {"name": "E", "bytes": 4, "d": .4},
         {"name": "O1", "bytes": 3, "d": .5}, {"name": "O2", "bytes": 2, "d": .6}],
        [{"name": "Z", "bytes": 1, "d": .6}, {"name": "E", "bytes": 4, "d": .3},
         {"name": "O1", "bytes": 3, "d": .4}, {"name": "O2", "bytes": 2, "d": .5}],
    ]
    checks = 0
    for n in (2, 3, 4):
        for budget in range(0, 24):
            assert exact_frame_dp(rows[:n], budget, 5, order) == brute_force(rows[:n], budget, 5, order)
            checks += 1
    return {"short_gop_bruteforce_cases": checks, "passed": True}
