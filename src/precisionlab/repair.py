"""Budgeted high-precision rescue from FULL measured candidate predictions.

Single-layer sensitivity can nominate candidates but cannot certify a combined
recipe. This function refuses unevaluated combinations and excessive measured
file sizes. It ranks observed recipes; it does not claim a global optimum.
"""

import math


def choose_recipe(candidates, *, max_weight_bytes, min_task_em, max_p95_ms=None,
                  min_success_rate=None, min_requests_per_second=None,
                  max_ttft_ms=None, max_peak_device_memory_bytes=None,
                  objective="smallest"):
    if not min_task_em:
        raise ValueError("Supply explicit per-task minimum exact-match scores")
    if type(max_weight_bytes) is not int or max_weight_bytes < 1:
        raise ValueError("Weight budget must be a positive integer")
    if any(type(target) not in (int, float) or not math.isfinite(target) or not 0 <= target <= 1 for target in min_task_em.values()):
        raise ValueError("Task targets must be between zero and one")
    if objective not in {"smallest", "throughput"}:
        raise ValueError("Objective must be smallest or throughput")
    for target in (max_p95_ms, max_ttft_ms, max_peak_device_memory_bytes):
        if target is not None and (type(target) not in (int, float) or not math.isfinite(target) or target <= 0):
            raise ValueError("Maximum resource targets must be finite positive numbers")
    for target in (min_success_rate, min_requests_per_second):
        if target is not None and (type(target) not in (int, float) or not math.isfinite(target) or target < 0):
            raise ValueError("Minimum service targets must be finite nonnegative numbers")
    if min_success_rate is not None and min_success_rate > 1:
        raise ValueError("Success-rate target must be between zero and one")
    service_required = objective == "throughput" or any(x is not None for x in
        (min_success_rate, min_requests_per_second, max_ttft_ms, max_peak_device_memory_bytes))
    if service_required:
        workloads = {c.get("service_profile", {}).get("workload_sha256") for c in candidates}
        if len(workloads - {None}) > 1:
            raise ValueError("Service candidates used different workloads or concurrency")
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
            if type(result) not in (int, float) or not math.isfinite(result) or not 0 <= result <= 1 or result < target:
                reasons.append(f"task constraint failed: {task}")
        if max_p95_ms is not None:
            p95 = candidate.get("p95_ms")
            if type(p95) not in (int, float) or not math.isfinite(p95) or p95 <= 0 or p95 > max_p95_ms:
                reasons.append("measured p95 budget not met")
        profile = candidate.get("service_profile", {})
        if service_required:
            if not profile.get("workload_sha256") or candidate.get("service_verified") is not True:
                reasons.append("service measurement not verified")
            checks = [("success_rate", min_success_rate, "minimum"),
                ("successful_requests_per_second", min_requests_per_second, "minimum"),
                ("ttft_p95_ms", max_ttft_ms, "maximum"),
                ("peak_device_memory_bytes", max_peak_device_memory_bytes, "maximum")]
            if objective == "throughput" and min_requests_per_second is None:
                checks.append(("successful_requests_per_second", 0, "minimum"))
            for key, target, direction in checks:
                if target is None:
                    continue
                value = profile.get(key)
                valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
                if not valid or (value < target if direction == "minimum" else value > target):
                    reasons.append("service constraint failed: " + key)
        audit.append({"recipe": candidate["name"], "accepted": not reasons, "reasons": reasons})
        if not reasons:
            feasible.append(candidate)
    rank = (lambda c: (-c["service_profile"]["successful_requests_per_second"], c["weight_bytes"], c["name"])) if objective == "throughput" else (
        lambda c: (c["weight_bytes"], c.get("p95_ms", float("inf")), c["name"]))
    winner = min(feasible, key=rank) if feasible else None
    return {"winner": winner, "audit": audit,
            "objective": objective,
            "claim": "Best measured feasible recipe for the chosen objective among evaluated candidates; not a global optimum"}
