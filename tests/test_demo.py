"""Keep the public replay tied to the published run, including negative cases."""

import importlib.util
import json
from pathlib import Path
import re
import shutil

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_demo", ROOT / "experiments/build_demo.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def test_replay_contains_full_paired_run_and_original_images(tmp_path):
    output = tmp_path / "index.html"
    assert demo.build(output, copy_images=True) == 300
    html = output.read_text()
    raw = re.search(r'<script type="application/json" id="evidence-data">(.*?)</script>', html, re.S)[1]
    cases = json.loads(raw)["cases"]
    assert len({c["group_id"] for c in cases}) == 150
    assert sum(c["reference_strict"] and not c["candidate_strict"] for c in cases) == 11
    assert sum(c["reference_content"] and not c["candidate_content"] for c in cases) == 0
    assert sum(not c["reference_content"] and not c["candidate_content"] for c in cases) == 47
    for case in cases:
        assert demo.sha256(output.parent / case["image"]) == case["image_sha256"]


def test_replay_rejects_a_changed_summary(tmp_path):
    shutil.copytree(ROOT / "examples", tmp_path / "examples")
    results = Path("experiments/results/2026-10-03")
    for name in ("p1-bf16", "p1-awq-reload"):
        shutil.copytree(ROOT / results / name, tmp_path / results / name)
    report = tmp_path / results / "p1-awq-report.json"
    value = json.loads((ROOT / results / report.name).read_text())
    value["semantic_tasks"]["graph_edge"]["candidate_em"] = 1.0
    report.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Recorded report differs"):
        demo.load_evidence(tmp_path)
