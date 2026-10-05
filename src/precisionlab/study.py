"""Verify independent receipt splits and summarize repeated service workloads."""

import json
from pathlib import Path
from statistics import median

from .data import fingerprint, load_samples, read_jsonl
from .gate import gate_service


def check_splits(dev, test):
    """Check actual image content as well as source ids across separate files."""
    rows = {"dev": load_samples(dev), "test": load_samples(test)}
    for split, samples in rows.items():
        if any(r["split"] != split for r in samples):
            raise ValueError(f"Unexpected rows in {split} dataset")
    groups = [{r["group_id"] for r in rows[s]} for s in ("dev", "test")]
    # CORD imports carry checked image hashes. Compute them for other datasets.
    from .data import sha256
    images = [{sha256(Path(path).parent / r["image"]) for r in rows[split]}
              for split, path in (("dev", dev), ("test", test))]
    if groups[0] & groups[1] or images[0] & images[1]:
        raise ValueError("Receipt source or image content leaks between dev and test")
    return {split: {"questions": len(rows[split]), "source_groups": len(groups[i]),
                   "dataset_sha256": fingerprint(rows[split], path)}
            for i, (split, path) in enumerate((("dev", dev), ("test", test)))}


def summarize_repetitions(dataset, runs):
    """Recompute each profile; never pool repeated images as independent data."""
    if not runs:
        raise ValueError("No repetitions")
    profiles, signatures, predictions = [], [], []
    tasks = {r["task"]: 0 for r in load_samples(dataset)}
    for run in map(Path, runs):
        gate_service(dataset, run, {"min_task_em": tasks})
        manifest = json.loads((run / "manifest.json").read_text())
        profiles.append(manifest["profile"])
        signatures.append((manifest["controls"], manifest.get("base_revision"),
                           manifest["served_model_alias"], manifest["profile"]["workload_sha256"]))
        predictions.append({p["id"]: p["prediction"] for p in read_jsonl(run / "predictions.jsonl")})
    if any(s != signatures[0] for s in signatures[1:]):
        raise ValueError("Repetitions have different workloads or model declarations")
    metrics = ("latency_p95_ms", "ttft_p95_ms", "successful_requests_per_second", "success_rate")
    summary = {}
    for metric in metrics:
        values = [p[metric] for p in profiles]
        summary[metric] = {"values": values, "median": median(values), "min": min(values), "max": max(values)} if all(v is not None for v in values) else {"values": values, "median": None}
    return {"repetitions": len(runs), "metrics": summary,
            "quality_requests": profiles[0]["requests"],
            "answer_changes_from_first_run": [sum(p[k] != predictions[0][k] for k in p) for p in predictions],
            "quality_sample_size_scope": "Unique receipts in the first run; repeats are timing observations, not extra independent quality data"}
