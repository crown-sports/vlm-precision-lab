"""Strict sample/prediction joins and reproducible local dataset fingerprints."""

import hashlib
import json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))


def load_samples(path):
    rows = read_jsonl(path)
    ids, groups, image_splits = set(), {}, {}
    for row in rows:
        required = {"id", "group_id", "split", "task", "image", "prompt", "answer", "features"}
        if required - row.keys():
            raise ValueError(f"Sample missing keys: {required - row.keys()}")
        if not all(isinstance(row[k], str) and row[k] for k in required - {"features"}):
            raise ValueError("Identifiers, paths, prompts and answers must be nonempty strings")
        if row["id"] in ids:
            raise ValueError(f"Duplicate sample id: {row['id']}")
        ids.add(row["id"])
        if row["split"] not in {"calibration", "dev", "test"}:
            raise ValueError("Split must be calibration, dev or test")
        old = groups.setdefault(row["group_id"], row["split"])
        if old != row["split"]:
            raise ValueError(f"Source group leaks across splits: {row['group_id']}")
        image = Path(row["image"])
        root = Path(path).resolve().parent
        full = (root / image).resolve()
        if image.is_absolute() or not full.is_relative_to(root):
            raise ValueError("Images must be relative paths inside the dataset directory")
        digest = sha256(full)
        if row.get("image_sha256") and digest != row["image_sha256"]:
            raise ValueError(f"Image hash mismatch: {row['id']}")
        old = image_splits.setdefault(digest, row["split"])
        if old != row["split"]:
            raise ValueError("Identical image content leaks across splits")
        if not isinstance(row["features"], list) or not all(isinstance(x, str) for x in row["features"]):
            raise ValueError("Features must be strings")
    if not rows:
        raise ValueError("Empty dataset")
    return rows


def fingerprint(rows, path):
    """Hash labels/prompts AND image bytes, independent of the directory name."""
    expanded = []
    for row in rows:
        record = dict(row)
        record["image_sha256"] = sha256(Path(path).parent / row["image"])
        expanded.append(record)
    raw = json.dumps(expanded, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def index_predictions(rows, samples):
    result = {}
    for row in rows:
        if row["id"] in result:
            raise ValueError(f"Duplicate prediction id: {row['id']}")
        if not isinstance(row.get("prediction"), str) or not row.get("input_sha256"):
            raise ValueError("Predictions require text and an input fingerprint")
        if row.get("input_fingerprint_scope", "tensor") not in {"tensor", "request"}:
            raise ValueError("Input fingerprint scope must be tensor or request")
        result[row["id"]] = row
    expected = {r["id"] for r in samples}
    if result.keys() != expected:
        raise ValueError(f"Prediction ids do not match dataset: missing={len(expected-result.keys())}, extra={len(result.keys()-expected)}")
    return result
