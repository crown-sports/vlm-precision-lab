"""Budgeted high-precision rescue from FULL measured candidate predictions.

Single-layer sensitivity can nominate candidates but cannot certify a combined
recipe. This function refuses unevaluated combinations and excessive measured
file sizes. It ranks observed recipes; it does not claim a global optimum.
"""

import math


def choose_recipe(candidates, *, max_weight_bytes, min_task_em, max_p95_ms=None):
    if not min_task_em:
        raise ValueError("Supply explicit per-task minimum exact-match scores")
    if any(not 0 <= target <= 1 for target in min_task_em.values()):
        raise ValueError("Task targets must be between zero and one")
    audit, feasible = [], []
    for candidate in candidates:
        reasons = []
        if candidate.get("export_reload_verified") is not True:
            reasons.append("export/reload not verified")
        size = candidate.get("weight_bytes")
        if type(size) is not int or size < 1 or size > max_weight_bytes:
            reasons.append("measured weight-file budget not met")
        for task, target in min_task_em.items():
            result = candidate.get("task_em", {}).get(task)
            if not isinstance(result, (int, float)) or not math.isfinite(result) or not 0 <= result <= 1 or result < target:
                reasons.append(f"task constraint failed: {task}")
        if max_p95_ms is not None:
            p95 = candidate.get("p95_ms")
            if not isinstance(p95, (int, float)) or not math.isfinite(p95) or p95 <= 0 or p95 > max_p95_ms:
                reasons.append("measured p95 budget not met")
        audit.append({"recipe": candidate["name"], "accepted": not reasons, "reasons": reasons})
        if not reasons:
            feasible.append(candidate)
    winner = min(feasible, key=lambda c: (c["weight_bytes"], c.get("p95_ms", float("inf")), c["name"])) if feasible else None
    return {"winner": winner, "audit": audit,
            "claim": "Smallest measured feasible recipe among evaluated candidates; not a global optimum"}
