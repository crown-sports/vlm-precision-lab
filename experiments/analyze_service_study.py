"""Recompute receipt quality, paired intervals and timing from raw study files."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from precisionlab.data import fingerprint, load_samples, read_jsonl, sha256
from precisionlab.gate import gate_service
from precisionlab.metrics import compare, exact, normalize, receipt_currency_spacing
from precisionlab.provenance import validate_manifests
from precisionlab.study import check_splits, summarize_repetitions


def error_audit(samples, predictions):
    """Lexical clues for manual review; never substitute for primary exact match."""
    by_source = defaultdict(list)
    for row in samples:
        by_source[row["group_id"]].append(row)
    indexed = {p["id"]: p for p in predictions}
    counts, cases = Counter(), []
    for row in samples:
        pred = indexed[row["id"]]
        if pred["success"] and exact(row["answer"], pred["prediction"]):
            continue
        text = normalize(pred["prediction"])
        same_other_fields = [r["task"] for r in by_source[row["group_id"]]
                             if r["id"] != row["id"] and exact(r["answer"], text)]
        expected_digits = "".join(re.findall(r"[0-9+-]", normalize(row["answer"])))
        actual_digits = "".join(re.findall(r"[0-9+-]", text))
        if not pred["success"]:
            kind = "request_failed"
        elif expected_digits and expected_digits == actual_digits:
            kind = "same_signed_digit_string"
        elif same_other_fields:
            kind = "matches_another_labelled_field"
        else:
            kind = "other_mismatch"
        counts[kind] += 1
        cases.append({"id": row["id"], "group_id": row["group_id"], "task": row["task"],
                      "answer": row["answer"], "prediction": pred["prediction"], "clue": kind,
                      "other_matching_fields": same_other_fields, "image": row["image"]})
    return {"counts": dict(counts), "cases": cases,
            "scope": "String-based review clues only. Equal digits do not establish numeric equivalence (12.50 vs 1,250), and another-field matches do not prove the model's reasoning mechanism."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("cord-service-protocol.json"))
    args = parser.parse_args()
    root = args.study
    recorded = json.loads((root / "study.json").read_text())
    if recorded["status"] != "completed":
        parser.error("Study has not completed")
    protocol = json.loads((root / "protocol.json").read_text())
    # Early runner revisions pretty-printed the archive but hashed the source.
    # Verify both the exact frozen source bytes and semantic archive equality.
    if sha256(args.protocol) != recorded["protocol_sha256"] or json.loads(args.protocol.read_text()) != protocol:
        raise ValueError("Frozen protocol or its measurement archive changed")
    splits = check_splits(args.dev, args.test)
    if splits != json.loads((root / "split-evidence.json").read_text()):
        raise ValueError("Study datasets changed")
    quality, audits, performance, memory, spacing = {}, {}, {}, {}, {}
    for split, dataset in (("dev", args.dev), ("test", args.test)):
        samples = load_samples(dataset)
        files = [root / mode / f"{split}-c1-r1" / "predictions.jsonl" for mode in ("bf16", "awq")]
        provenance = validate_manifests(*files, dataset_sha256=fingerprint(samples, dataset))
        result = compare(samples, *(read_jsonl(p) for p in files),
                         repetitions=protocol["bootstrap"]["repetitions"], seed=protocol["bootstrap"]["seed"])
        result["provenance"] = provenance
        result["margin_check"] = {t: {"lower_bound_pp": m["delta_ci95_pp"][0],
                                     "target_min_delta_pp": -protocol["quality_margin_pp"],
                                     "supported_at_observed_interval": m["delta_ci95_pp"][0] >= -protocol["quality_margin_pp"]}
                                  for t, m in result["tasks"].items()}
        quality[split] = result
        audits[split] = {mode: error_audit(samples, read_jsonl(file)) for mode, file in zip(("bf16", "awq"), files)}
        diagnostic = compare([dict(s, answer=receipt_currency_spacing(s["answer"])) for s in samples],
            *([dict(p, prediction=receipt_currency_spacing(p["prediction"])) for p in read_jsonl(file)] for file in files),
            repetitions=protocol["bootstrap"]["repetitions"], seed=protocol["bootstrap"]["seed"])
        spacing[split] = {"tasks": diagnostic["tasks"], "overall": diagnostic["overall"],
                          "scope": "Post hoc diagnostic after dev image review: ignore only whitespace following an exact Rp prefix; preserve case, currency, signs and all decimal/thousands punctuation. Primary quality and frozen gates still use literal annotation EM."}
    gates = {}
    for mode in ("bf16", "awq"):
        performance[mode] = {str(c): summarize_repetitions(args.dev,
            [root / mode / f"dev-c{c}-r{r + 1}" for r in range(protocol["dev_repetitions"])])
            for c in protocol["dev_concurrency"]}
        gates[mode] = gate_service(args.test, root / mode / "test-c1-r1", protocol["test_gate"])
        samples = read_jsonl(root / mode / "gpu-samples.jsonl")
        measured = {stage: max(int(s["used_mib"]) for s in samples if s["stage"] == stage)
                    for stage in sorted({s["stage"] for s in samples})}
        saved = json.loads((root / mode / "memory-profile.json").read_text())
        if saved["peak_framebuffer_mib_by_stage"] != measured or saved["samples"] != len(samples) or saved["errors"]:
            raise ValueError("GPU memory profile is incomplete or differs from raw samples")
        memory[mode] = saved
    kernels = re.findall(r"Using (\w+) for CompressedTensorsWNA16", (root / "awq/server.log").read_text())
    result = {"quality": quality, "error_audit": audits, "currency_spacing_diagnostic": spacing, "performance": performance, "memory": memory,
              "test_gates": gates, "awq_kernel_log_evidence": sorted(set(kernels)), "splits": splits,
              "protocol_sha256": recorded["protocol_sha256"],
              "interpretation": "Deployment comparison on pinned receipts and requests; one launch per variant and request-level fingerprints limit causal claims. Error audit clues are not new accuracy metrics."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"test_quality": quality["test"]["tasks"], "kernels": result["awq_kernel_log_evidence"],
                      "test_accepted": {k: v["accepted"] for k, v in gates.items()}}))


if __name__ == "__main__":
    main()
