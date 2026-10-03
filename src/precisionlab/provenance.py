"""Separate controlled comparisons from changed input/decoding experiments."""

import json
from pathlib import Path


def validate_manifests(reference_file, candidate_file, *, allow_input_change=False):
    files = [Path(p).parent / "manifest.json" for p in (reference_file, candidate_file)]
    if not all(p.exists() for p in files):
        return {"manifests_verified": False, "note": "Prediction files alone do not verify model, processor or decoding controls"}
    ref, cand = [json.loads(p.read_text()) for p in files]
    if ref.get("base_revision") != cand.get("base_revision"):
        raise ValueError("Base model revisions differ")
    for key in ("dataset_sha256", "split", "decode", "seed"):
        if ref["controls"][key] != cand["controls"][key]:
            raise ValueError(f"Comparison controls differ: {key}")
    settings = [key for key in ("max_pixels", "min_pixels", "use_fast") if ref["controls"][key] != cand["controls"][key]]
    if settings and not allow_input_change:
        raise ValueError(f"Processor controls differ: {settings}")
    return {"manifests_verified": True, "base_revision": ref["base_revision"],
            "processor_file_hashes_match": ref["controls"]["processor_files"] == cand["controls"]["processor_files"],
            "processor_settings_changed": settings, "reference_mode": ref["mode"], "candidate_mode": cand["mode"]}
