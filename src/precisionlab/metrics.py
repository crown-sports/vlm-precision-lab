"""Paired task regressions with source-cluster bootstrap confidence intervals."""

from collections import defaultdict
import random
import re
import unicodedata

from .data import index_predictions


def normalize(text):
    # Preserve signs, decimal points and case; 12.50 must not match 1250 or 12.5.
    return unicodedata.normalize("NFKC", text).strip()


def exact(answer, prediction):
    return normalize(answer) == normalize(prediction)


def receipt_currency_spacing(text):
    """Diagnostic Rp-prefix spacing only; retain every numeric separator/sign.

    CORD annotations can insert a space absent from the printed currency prefix.
    This optional scoring view does not replace literal EM or normalize values.
    """
    return re.sub(r"^([+-]?Rp)\s+(?=[0-9])", r"\1", normalize(text))


def endpoints(text):
    match = re.fullmatch(r"([A-Za-z0-9_-]+)\s*,\s*([A-Za-z0-9_-]+)", normalize(text))
    return set(match.groups()) if match and match[1] != match[2] else None


def semantic_correct(sample, prediction):
    if sample["task"] == "graph_edge":
        expected, actual = endpoints(sample["answer"]), endpoints(prediction)
        return expected is not None and actual == expected
    return exact(sample["answer"], prediction)


def interval(values):
    ordered = sorted(values)
    return [ordered[int((len(ordered) - 1) * 0.025)], ordered[int((len(ordered) - 1) * 0.975)]]


def task_summary(rows, repetitions, seed):
    n = len(rows)
    before = sum(r["before"] for r in rows) / n
    after = sum(r["after"] for r in rows) / n
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(int(row["after"]) - int(row["before"]))
    clusters = list(groups.values())
    rng = random.Random(seed)
    draws = []
    for _ in range(repetitions):
        sampled = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        draws.append(sum(sum(v) for v in sampled) / sum(len(v) for v in sampled))
    regressions = sum(r["before"] and not r["after"] for r in rows)
    recovered = sum(not r["before"] and r["after"] for r in rows)
    return {"samples": n, "independent_groups": len(clusters), "reference_em": before,
            "candidate_em": after, "delta_pp": (after - before) * 100,
            "delta_ci95_pp": [x * 100 for x in interval(draws)],
            "regressions": regressions, "recoveries": recovered,
            "regression_rate_on_reference_correct": regressions / sum(r["before"] for r in rows) if any(r["before"] for r in rows) else None,
            "insufficient_groups": len(clusters) < 30}


def compare(samples, reference, candidate, *, repetitions=2000, seed=42, allow_input_change=False):
    if repetitions < 100:
        raise ValueError("Use at least 100 bootstrap repetitions")
    ref, cand = index_predictions(reference, samples), index_predictions(candidate, samples)
    scopes = {p.get("input_fingerprint_scope", "tensor") for p in [*ref.values(), *cand.values()]}
    if len(scopes) != 1:
        raise ValueError("Cannot compare request fingerprints with actual input-tensor fingerprints")
    scope = scopes.pop()
    changed = [s["id"] for s in samples if ref[s["id"]]["input_sha256"] != cand[s["id"]]["input_sha256"]]
    if changed and not allow_input_change:
        kind = "tensors" if scope == "tensor" else "requests"
        raise ValueError(f"Input {kind} changed for {len(changed)} samples; attribution requires a separate processor control")
    cases = [{"id": s["id"], "group_id": s["group_id"], "task": s["task"],
              "answer": s["answer"], "image": s["image"], "prompt": s["prompt"], "reference": ref[s["id"]]["prediction"],
              "candidate": cand[s["id"]]["prediction"],
              "before": exact(s["answer"], ref[s["id"]]["prediction"]),
              "after": exact(s["answer"], cand[s["id"]]["prediction"]),
              "reference_content_correct": semantic_correct(s, ref[s["id"]]["prediction"]),
              "candidate_content_correct": semantic_correct(s, cand[s["id"]]["prediction"])} for s in samples]
    semantic_cases = [dict(c, before=semantic_correct(s, c["reference"]), after=semantic_correct(s, c["candidate"])) for s, c in zip(samples, cases)]
    semantic_tasks = {t: task_summary([r for r in semantic_cases if r["task"] == t], repetitions, seed) for t in sorted({r["task"] for r in cases})}
    graph_cases = [c for c in cases if c["task"] == "graph_edge"]
    format_rates = {}
    if graph_cases:
        for stage in ("reference", "candidate"):
            valid = sum(endpoints(c[stage]) is not None and normalize(c[stage]) == ",".join(sorted(endpoints(c[stage]))) for c in graph_cases)
            format_rates[stage] = valid / len(graph_cases)
    return {"overall": task_summary(cases, repetitions, seed),
            "tasks": {t: task_summary([r for r in cases if r["task"] == t], repetitions, seed)
                      for t in sorted({r["task"] for r in cases})},
            "bootstrap": {"unit": "source group", "paired": True, "repetitions": repetitions, "seed": seed},
            "semantic_tasks": semantic_tasks,
            "semantic_metric": "Unordered exact endpoints for graph_edge; literal exact match for reading tasks",
            "graph_format_rates": format_rates,
            "semantic_failures": [r for r in semantic_cases if r["before"] and not r["after"]],
            "input_changes": changed,
            "input_fingerprint_scope": scope,
            "interpretation": "Input-change comparison is descriptive, not isolated quantization evidence" if changed else (
                "Same client requests; server-side input tensors and model identity are not verified" if scope == "request"
                else "Same input tensors; also check model, decoding and backend manifests"),
            "failures": [r for r in cases if r["before"] and not r["after"]]}
