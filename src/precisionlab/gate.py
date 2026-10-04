"""Check observed service results against explicit quality and latency budgets."""

import json
import math
from pathlib import Path

from .data import fingerprint, index_predictions, load_samples, read_jsonl, sha256
from .metrics import exact
from .serving import percentile, request_body


def finite_number(value, *, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def gate_service(dataset, run, constraints):
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text())
    profile = json.loads((run / "profile.json").read_text())
    if manifest.get("mode") != "service" or manifest.get("profile") != profile:
        raise ValueError("Expected a consistent service manifest and profile")
    if sha256(run / "predictions.jsonl") != manifest["predictions_sha256"]:
        raise ValueError("Service predictions changed after measurement")
    predictions = read_jsonl(run / "predictions.jsonl")
    ids = {p["id"] for p in predictions}
    rows = [r for r in load_samples(dataset) if r["split"] == manifest["controls"]["split"] and r["id"] in ids]
    if not rows or fingerprint(rows, dataset) != manifest["controls"]["dataset_sha256"]:
        raise ValueError("Dataset no longer matches the measured requests")
    index_predictions(predictions, rows)
    if (manifest["controls"].get("input_fingerprint_scope") != "request" or
            any(p.get("input_fingerprint_scope") != "request" for p in predictions)):
        raise ValueError("Service checks require request fingerprints")
    indexed = {p["id"]: p for p in predictions}
    for row in rows:
        _, expected = request_body(row, dataset, manifest["served_model_alias"],
            manifest["controls"]["decode"]["max_tokens"], manifest["controls"]["seed"])
        if indexed[row["id"]]["input_sha256"] != expected:
            raise ValueError("Recorded request no longer matches its image, prompt or decoding settings")
    if not all(type(p.get("success")) is bool and finite_number(p.get("latency_seconds"), positive=True) for p in predictions):
        raise ValueError("Invalid request success flags or latency records")
    if not finite_number(profile.get("wall_seconds"), positive=True):
        raise ValueError("Invalid measurement wall time")
    successful = [p for p in predictions if p["success"]]
    tasks = sorted({r["task"] for r in rows})
    em = {t: sum(indexed[r["id"]]["success"] and exact(r["answer"], indexed[r["id"]]["prediction"])
        for r in rows if r["task"] == t) / sum(r["task"] == t for r in rows) for t in tasks}
    for p in successful:
        if (not finite_number(p.get("ttft_seconds"), positive=True) or
                p["ttft_seconds"] > p["latency_seconds"] or p.get("finish_reason") not in {"stop", "length"}):
            raise ValueError("Invalid completed stream timing or finish reason")
    measured = {"requests": len(rows), "successful_requests": len(successful),
        "success_rate": len(successful) / len(rows), "task_em": em,
        "successful_requests_per_second": len(successful) / profile["wall_seconds"],
        "latency_p95_ms": percentile([p["latency_seconds"] * 1000 for p in predictions], .95),
        "ttft_p95_ms": percentile([p["ttft_seconds"] * 1000 for p in successful], .95),
        "truncated_requests": sum(p.get("finish_reason") == "length" for p in successful)}
    if any(profile.get(k) != v for k, v in measured.items()):
        raise ValueError("Profile differs from its raw prediction records")
    allowed = {"min_task_em", "min_success_rate", "min_requests_per_second", "max_latency_p95_ms",
        "max_ttft_p95_ms", "max_truncated_requests", "min_source_groups"}
    if set(constraints) - allowed:
        raise ValueError("Unknown service constraints: " + ", ".join(sorted(set(constraints) - allowed)))
    targets = constraints.get("min_task_em")
    if not isinstance(targets, dict) or not targets:
        raise ValueError("Specify minimum quality for the required tasks")
    if any(not finite_number(t) or t > 1 for t in targets.values()):
        raise ValueError("Task thresholds must be finite scores from zero to one")
    checks = [{"metric": "task_em." + t, "observed": em.get(t), "target": target,
        "passed": t in em and em[t] >= target} for t, target in targets.items()]
    source_groups = len({r["group_id"] for r in rows})
    measured["source_groups"] = source_groups
    definitions = {"min_success_rate": ("success_rate", "min"),
        "min_requests_per_second": ("successful_requests_per_second", "min"),
        "max_latency_p95_ms": ("latency_p95_ms", "max"), "max_ttft_p95_ms": ("ttft_p95_ms", "max"),
        "max_truncated_requests": ("truncated_requests", "max"), "min_source_groups": ("source_groups", "min")}
    for key, (metric, direction) in definitions.items():
        if key not in constraints:
            continue
        target = constraints[key]
        if not finite_number(target) or (key == "min_success_rate" and target > 1):
            raise ValueError("Invalid service threshold: " + key)
        value = measured[metric]
        passed = value is not None and (value >= target if direction == "min" else value <= target)
        checks.append({"metric": metric, "observed": value, "target": target, "passed": passed})
    return {"accepted": all(c["passed"] for c in checks), "checks": checks,
        "requests": len(rows), "source_groups": source_groups, "unconstrained_tasks": sorted(set(tasks) - targets.keys()),
        "evidence_scope": "Observed quality and service budgets on this run; not a statistical guarantee or verified weight identity",
        "predictions_sha256": manifest["predictions_sha256"], "dataset_sha256": manifest["controls"]["dataset_sha256"]}
