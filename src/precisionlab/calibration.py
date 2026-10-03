"""Deterministic, token-budgeted weighted feature coverage with source caps.

This is a transparent greedy heuristic, not a new quantizer or a claim of
optimal submodular maximization. All costs must come from the pinned processor.
"""

from collections import Counter
import hashlib
import math
import random


def select(samples, costs, budget, *, strategy="coverage", seed=42, source_cap=2):
    if budget <= 0 or source_cap < 1:
        raise ValueError("Budget and source cap must be positive")
    rows = [r for r in samples if r["split"] == "calibration"]
    if not rows:
        raise ValueError("No calibration samples; dev/test data cannot be selected")
    if strategy not in {"coverage", "random", "stratified"}:
        raise ValueError("Unknown selection strategy")
    for r in rows:
        if r["id"] not in costs or type(costs[r["id"]]) is not int or costs[r["id"]] < 1:
            raise ValueError("Every calibration sample requires a positive measured integer token cost")
    universe = Counter(f for r in rows for f in set(r["features"] + ["task:" + r["task"]]))
    weights = {f: 1 / math.sqrt(count) for f, count in universe.items()}
    tie = lambda r: hashlib.sha256(f"{seed}:{r['id']}".encode()).hexdigest()
    remaining = sorted(rows, key=tie)
    random.Random(seed).shuffle(remaining)
    chosen, covered, sources, tasks, spent = [], Counter(), Counter(), Counter(), 0
    while True:
        feasible = [r for r in remaining if spent + costs[r["id"]] <= budget and sources[r["group_id"]] < source_cap]
        if not feasible:
            break
        if strategy == "random":
            row = feasible[0]
        elif strategy == "stratified":
            row = min(feasible, key=lambda r: (tasks[r["task"]], tie(r)))
        else:
            def score(r):
                # Diminishing marginal coverage: prevent one rare feature monopolizing all slots.
                gain = sum(weights[f] / (1 + covered[f]) for f in set(r["features"] + ["task:" + r["task"]]))
                return gain / costs[r["id"]]
            row = min(feasible, key=lambda r: (-score(r), tie(r)))
        chosen.append(row)
        spent += costs[row["id"]]
        sources[row["group_id"]] += 1
        tasks[row["task"]] += 1
        covered.update(set(row["features"] + ["task:" + row["task"]]))
        remaining.remove(row)
    return chosen, {"strategy": strategy, "seed": seed, "budget_tokens": budget,
                    "spent_tokens": spent, "unused_tokens": budget - spent,
                    "samples": len(chosen), "source_groups": len(sources),
                    "source_cap": source_cap, "task_counts": dict(tasks),
                    "feature_counts": dict(covered), "uncovered_features": sorted(universe.keys() - covered.keys())}
