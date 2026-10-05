"""Separate files must not hide image leakage behind different source ids."""

from PIL import Image
import pytest

from precisionlab.data import write_jsonl
from precisionlab.study import check_splits


def datasets(tmp_path):
    paths = []
    for split, color in (("dev", "white"), ("test", "black")):
        root = tmp_path / split
        root.mkdir()
        Image.new("RGB", (3, 3), color).save(root / "image.png")
        row = {"id": split, "group_id": split, "image": "image.png", "split": split,
               "task": "receipt_total_price", "prompt": "Read total", "answer": "12.50", "features": []}
        path = root / "samples.jsonl"
        write_jsonl(path, [row])
        paths.append((path, row))
    return paths


def test_checks_independent_files_and_reports_unique_receipts(tmp_path):
    (dev, _), (test, _) = datasets(tmp_path)
    result = check_splits(dev, test)
    assert result["dev"]["questions"] == result["test"]["source_groups"] == 1
    assert result["dev"]["dataset_sha256"] != result["test"]["dataset_sha256"]


def test_detects_same_content_under_different_names_and_groups(tmp_path):
    (dev, _), (test, _) = datasets(tmp_path)
    (test.parent / "image.png").write_bytes((dev.parent / "image.png").read_bytes())
    with pytest.raises(ValueError, match="image content leaks"):
        check_splits(dev, test)


def test_detects_same_source_with_changed_image_and_wrong_split(tmp_path):
    (dev, _), (test, row) = datasets(tmp_path)
    write_jsonl(test, [{**row, "group_id": "dev"}])
    with pytest.raises(ValueError, match="source or image"):
        check_splits(dev, test)
    write_jsonl(test, [{**row, "split": "dev"}])
    with pytest.raises(ValueError, match="Unexpected rows in test"):
        check_splits(dev, test)
