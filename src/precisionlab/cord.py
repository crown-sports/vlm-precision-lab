"""Import pinned CORD-v2 receipt fields without mixing document splits."""

from collections import Counter
from io import BytesIO
import json
from pathlib import Path

from PIL import Image

from .data import fingerprint, load_samples, sha256, write_jsonl


FIELDS = {
    "total.total_price": "total amount",
    "total.cashprice": "cash amount paid",
    "total.changeprice": "change returned",
    "sub_total.tax_price": "tax amount",
}


def field_samples(ground_truth, *, split, image, image_sha256, source_id):
    truth = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    parsed = truth["gt_parse"]
    samples, skipped = [], Counter()
    for field, description in FIELDS.items():
        group, key = field.split(".")
        container = parsed.get(group, {})
        value = container.get(key) if isinstance(container, dict) else None
        if not isinstance(value, str) or not value.strip():
            skipped[field + ":missing-or-ambiguous"] += 1
            continue
        samples.append({"id": source_id + ":" + field, "group_id": source_id,
            "split": split, "task": "receipt_" + key, "image": image,
            "image_sha256": image_sha256,
            "prompt": f"Read this receipt. What is the {description}? Return only the amount exactly as printed, preserving currency text and punctuation. Do not explain.",
            "answer": value.strip(), "features": ["real-receipt", field],
            "source": {"dataset": "naver-clova-ix/cord-v2", "field": field}})
    return samples, skipped


def import_cord(parquet_dir, output, source_manifest, *, splits=("train", "validation", "test")):
    # Optional dependency stays outside the small runtime used for compare/demo.
    import pyarrow.parquet as pq

    metadata = json.loads(Path(source_manifest).read_text())
    if metadata["id"] != "naver-clova-ix/cord-v2":
        raise ValueError("Expected the CORD-v2 source manifest")
    parquet_dir, output = Path(parquet_dir), Path(output)
    mapping = {"train": "calibration", "validation": "dev", "test": "test"}
    if not splits or set(splits) - mapping.keys():
        raise ValueError("CORD splits must be train, validation or test")
    selected = [x for x in metadata["siblings"] if any(
        x["rfilename"].startswith("data/" + split + "-") for split in splits)]
    if not selected:
        raise ValueError("No source parquet files selected")
    # Verify all sources before creating output. An arbitrary parquet file cannot
    # be presented as the pinned public release.
    for item in selected:
        path = parquet_dir / item["rfilename"]
        if not path.is_file() or path.stat().st_size != item["size"] or sha256(path) != item["lfs"]["sha256"]:
            raise ValueError("Pinned CORD source hash mismatch: " + item["rfilename"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    rows, skipped, documents = [], Counter(), Counter()
    for item in sorted(selected, key=lambda x: x["rfilename"]):
        name = Path(item["rfilename"]).name
        source_split = name.split("-")[0]
        parquet = pq.ParquetFile(parquet_dir / item["rfilename"])
        offset = 0
        for batch in parquet.iter_batches(batch_size=1, columns=["image", "ground_truth"]):
            for record in batch.to_pylist():
                raw = record["image"].get("bytes")
                if not raw:
                    raise ValueError("CORD image must be embedded in the pinned parquet")
                source_id = f"cord:{name}:{offset}"
                with Image.open(BytesIO(raw)) as im:
                    im.verify()
                    suffix = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}.get(im.format)
                if suffix is None:
                    raise ValueError("Unsupported CORD image encoding")
                relative = f"images/{name}-{offset}{suffix}"
                path = output / relative
                path.write_bytes(raw)
                samples, omitted = field_samples(record["ground_truth"], split=mapping[source_split],
                    image=relative, image_sha256=sha256(path), source_id=source_id)
                rows.extend(samples)
                skipped.update(omitted)
                documents[source_split] += 1
                offset += 1
    write_jsonl(output / "samples.jsonl", rows)
    load_samples(output / "samples.jsonl")  # Check image identity and split leakage.
    manifest = {"repository": metadata["id"], "revision": metadata["sha"],
        "license": "CC-BY-4.0", "source_files": selected, "documents": dict(documents),
        "samples": len(rows), "skipped_fields": dict(skipped), "fields": FIELDS,
        "dataset_sha256": fingerprint(rows, output / "samples.jsonl"),
        "scoring": "literal agreement with CORD parsed-field annotations; currency spacing can differ from print, so this is not guaranteed verbatim OCR or numeric-content loss"}
    (output / "source.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
