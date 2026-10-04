"""Separate controlled comparisons from changed input/decoding experiments."""

import json
from pathlib import Path

from .data import sha256


def validate_manifests(reference_file, candidate_file, *, allow_input_change=False, dataset_sha256=None):
    files = [Path(p).parent / "manifest.json" for p in (reference_file, candidate_file)]
    if not all(p.exists() for p in files):
        return {"manifests_verified": False, "note": "Prediction files alone do not verify model, processor or decoding controls"}
    ref, cand = [json.loads(p.read_text()) for p in files]
    hashes_verified = []
    for record, predictions in zip((ref, cand), (reference_file, candidate_file)):
        if dataset_sha256 is not None and record["controls"]["dataset_sha256"] != dataset_sha256:
            raise ValueError("Comparison dataset differs from the recorded images, labels or prompts")
        expected = record.get("predictions_sha256")
        if expected and sha256(predictions) != expected:
            raise ValueError("Comparison predictions changed after measurement")
        hashes_verified.append(bool(expected))
    if ref.get("base_revision") and cand.get("base_revision") and ref["base_revision"] != cand["base_revision"]:
        raise ValueError("Base model revisions differ")
    for key in ("dataset_sha256", "split", "decode", "seed"):
        if ref["controls"][key] != cand["controls"][key]:
            raise ValueError(f"Comparison controls differ: {key}")
    if ref["controls"].get("input_fingerprint_scope", "tensor") != cand["controls"].get("input_fingerprint_scope", "tensor"):
        raise ValueError("Input fingerprint scopes differ")
    settings = [key for key in ("max_pixels", "min_pixels", "use_fast", "server_config_declared")
                if ref["controls"].get(key) != cand["controls"].get(key)]
    if settings and not allow_input_change:
        raise ValueError(f"Processor controls differ: {settings}")
    return {"manifests_verified": bool(dataset_sha256 and all(hashes_verified) and ref.get("base_revision") and cand.get("base_revision")),
            "prediction_hashes_verified": all(hashes_verified),
            "base_revision": ref.get("base_revision"),
            "input_fingerprint_scope": ref["controls"].get("input_fingerprint_scope", "tensor"),
            "server_identity_verified": ref.get("server_identity_verified", False) and cand.get("server_identity_verified", False),
            "processor_file_hashes_match": (ref["controls"]["processor_files"] == cand["controls"]["processor_files"]
                if ref["controls"].get("processor_files") and cand["controls"].get("processor_files") else None),
            "processor_settings_changed": settings, "reference_mode": ref["mode"], "candidate_mode": cand["mode"]}
