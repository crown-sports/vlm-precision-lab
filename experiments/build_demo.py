"""Build a static replay from the published experiment, without model inference."""

import argparse
import json
import os
from pathlib import Path
import shutil

from precisionlab.data import fingerprint, index_predictions, load_samples, read_jsonl, sha256
from precisionlab.metrics import compare, exact, semantic_correct
from precisionlab.provenance import validate_manifests


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_COMMIT = "59bf5cc10e31379543c6e925a1d7982d9db22ed9"


def load_evidence(root=ROOT):
    dataset = root / "examples/diagnostic-v2/samples.jsonl"
    results = root / "experiments/results/2026-10-03"
    reference = results / "p1-bf16/predictions.jsonl"
    candidate = results / "p1-awq-reload/predictions.jsonl"
    samples = [s for s in load_samples(dataset) if s["split"] == "dev"]
    ref_rows, cand_rows = read_jsonl(reference), read_jsonl(candidate)
    ref, cand = index_predictions(ref_rows, samples), index_predictions(cand_rows, samples)
    dataset_hash = fingerprint(samples, dataset)
    controls = validate_manifests(reference, candidate, dataset_sha256=dataset_hash)
    if not controls["manifests_verified"]:
        raise ValueError("Replay requires both run manifests")
    manifests = [json.loads(p.with_name("manifest.json").read_text()) for p in (reference, candidate)]
    if manifests[0]["script_sha256"] != manifests[1]["script_sha256"]:
        raise ValueError("Replay expects the same inference script")
    recorded = json.loads((results / "p1-awq-report.json").read_text())
    measured = compare(samples, ref_rows, cand_rows)
    for key in ("overall", "tasks", "semantic_tasks", "graph_format_rates", "input_changes"):
        if measured[key] != recorded[key]:
            raise ValueError(f"Recorded report differs from recomputed evidence: {key}")
    cases = []
    for sample in samples:
        a, b = ref[sample["id"]], cand[sample["id"]]
        cases.append({
            **{key: sample[key] for key in ("id", "group_id", "task", "image", "image_sha256", "prompt", "answer")},
            "reference": a["prediction"], "candidate": b["prediction"],
            "reference_strict": exact(sample["answer"], a["prediction"]),
            "candidate_strict": exact(sample["answer"], b["prediction"]),
            "reference_content": semantic_correct(sample, a["prediction"]),
            "candidate_content": semantic_correct(sample, b["prediction"]),
            "input_sha256": a["input_sha256"],
        })
    return {
        "model": "Qwen3-VL-8B-Instruct",
        "base_revision": controls["base_revision"],
        "recorded_date": "2026-10-03",
        "evidence_commit": EVIDENCE_COMMIT,
        "dataset_sha256": dataset_hash,
        "cases": cases,
        "runs": [{key: m[key] for key in (
            "weight_file_bytes", "peak_torch_allocated_bytes", "inference_seconds",
            "generation_p95_ms", "predictions_sha256", "script_sha256",
        )} for m in manifests],
    }


def build(output, *, copy_images=False, root=ROOT):
    payload = load_evidence(root)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    for case in payload["cases"]:
        image = root / "examples/diagnostic-v2" / case["image"]
        if copy_images:
            destination = output.parent / "images" / image.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(image, destination)
            case["image"] = f"images/{image.name}"
        else:
            case["image"] = Path(os.path.relpath(image, output.parent)).as_posix()
    # JSON stays data: a prediction containing </script> must not become markup.
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    template = (root / "demo/template.html").read_text()
    if template.count("__RECORDED_EVIDENCE__") != 1:
        raise ValueError("Template must contain one evidence placeholder")
    output.write_text(template.replace("__RECORDED_EVIDENCE__", encoded))
    return len(payload["cases"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "demo/index.html")
    parser.add_argument("--copy-images", action="store_true")
    args = parser.parse_args()
    count = build(args.output, copy_images=args.copy_images)
    print(f"Built {args.output}: {count} verified recorded cases; no inference")
