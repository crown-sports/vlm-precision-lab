import json
from io import BytesIO

from PIL import Image
import pytest

from precisionlab.cord import field_samples, import_cord
from precisionlab.data import load_samples, sha256


def test_receipt_fields_preserve_annotation_and_skip_ambiguous_values():
    truth = {"gt_parse": {"total": {"total_price": "Rp 61.799", "cashprice": ["one", "two"]},
        "sub_total": {"tax_price": "-1,000"}}}
    rows, skipped = field_samples(truth, split="dev", image="r.png", image_sha256="fixture",
        source_id="fixture-document")
    assert {r["answer"] for r in rows} == {"Rp 61.799", "-1,000"}
    assert len({r["group_id"] for r in rows}) == 1
    assert skipped["total.cashprice:missing-or-ambiguous"] == 1
    assert all(r["answer"] not in r["prompt"] for r in rows)


def test_parquet_import_verifies_source_and_detects_shared_images(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    root = tmp_path / "sources"
    (root / "data").mkdir(parents=True)
    items = []
    for split, color in [("train", "white"), ("validation", "black")]:
        buffer = BytesIO()
        Image.new("RGB", (2, 2), color).save(buffer, format="PNG")
        table = pa.Table.from_pylist([{"image": {"bytes": buffer.getvalue(), "path": None},
            "ground_truth": json.dumps({"gt_parse": {"total": {"total_price": "12,000"}}})}])
        path = root / "data" / (split + "-fixture.parquet")
        pq.write_table(table, path)
        items.append({"rfilename": "data/" + path.name, "size": path.stat().st_size, "lfs": {"sha256": sha256(path)}})
    source = tmp_path / "fixture-source.json"
    source.write_text(json.dumps({"id": "naver-clova-ix/cord-v2", "sha": "fixture-only-not-a-public-benchmark", "siblings": items}))
    manifest = import_cord(root, tmp_path / "import", source, splits=("train", "validation"))
    rows = load_samples(tmp_path / "import/samples.jsonl")
    assert manifest["documents"] == {"train": 1, "validation": 1}
    assert {r["split"] for r in rows} == {"calibration", "dev"}
    with (root / items[0]["rfilename"]).open("ab") as f:
        f.write(b"tamper")
    with pytest.raises(ValueError, match="source hash mismatch"):
        import_cord(root, tmp_path / "tampered", source, splits=("train",))
    assert not (tmp_path / "tampered").exists()
