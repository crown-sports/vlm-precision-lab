import json

import pytest

from precisionlab.calibration import select
from precisionlab.data import fingerprint, load_samples, write_jsonl
from precisionlab.metrics import compare, exact, task_summary
from precisionlab.repair import choose_recipe
from precisionlab.provenance import validate_manifests
from precisionlab.report import render


def sample(i, *, group=None, split="calibration", task="numbers", features=None):
    return {"id": str(i), "group_id": group or str(i), "split": split, "task": task,
            "features": features or ["common"], "answer": "-12.50", "image": "image.png", "prompt": "Read amount"}


def prediction(i, text="-12.50", fingerprint="same"):
    return {"id": str(i), "prediction": text, "input_sha256": fingerprint}


def test_actual_images_and_related_sources_cannot_leak(tmp_path):
    from PIL import Image
    Image.new("RGB", (2, 2), "white").save(tmp_path / "image.png")
    rows = [sample(0), sample(1, group="0", split="test")]
    file = tmp_path / "samples.jsonl"
    write_jsonl(file, rows)
    with pytest.raises(ValueError, match="Source group leaks"):
        load_samples(file)
    rows[1]["group_id"] = "new"
    write_jsonl(file, rows)
    with pytest.raises(ValueError, match="Identical image content leaks"):
        load_samples(file)
    rows = [sample(0)]
    original = fingerprint(rows, file)
    Image.new("RGB", (2, 2), "black").save(tmp_path / "image.png")
    assert fingerprint(rows, file) != original


def test_selection_is_budgeted_reproducible_and_never_reads_dev_labels():
    rows = [sample(i, group=str(i//2), task="ocr" if i%2 else "graph", features=["rare"] if i==5 else ["common"]) for i in range(12)]
    rows.append(sample(99, split="dev", features=["super-rare"]))
    costs = {r["id"]: 3 + int(r["id"]) % 4 for r in rows}
    chosen, stats = select(rows, costs, 23, source_cap=1)
    assert sum(costs[r["id"]] for r in chosen) <= 23
    assert len({r["group_id"] for r in chosen}) == len(chosen)
    assert all(r["split"] == "calibration" for r in chosen)
    rows[-1]["answer"] = "corrupted held-out answer"
    assert select(rows, costs, 23, source_cap=1) == (chosen, stats)
    assert select(list(reversed(rows)), costs, 23, source_cap=1) == (chosen, stats)
    with pytest.raises(ValueError, match="No calibration"):
        select([rows[-1]], costs, 23)


def test_paired_losses_cannot_be_hidden_by_recoveries():
    rows = [sample(i, split="dev", task="ocr" if i<2 else "graph") for i in range(4)]
    reference = [prediction(0), prediction(1), prediction(2,"wrong"), prediction(3,"wrong")]
    candidate = [prediction(0,"wrong"), prediction(1,"wrong"), prediction(2), prediction(3)]
    result = compare(rows, reference, candidate, repetitions=200)
    assert result["overall"]["delta_pp"] == 0
    assert result["tasks"]["ocr"]["delta_pp"] == -100
    assert result["tasks"]["graph"]["delta_pp"] == 100
    assert result["overall"]["regressions"] == result["overall"]["recoveries"] == 2
    assert len(result["failures"]) == 2
    assert not exact("-12.50", "12.50")
    assert not exact("12.50", "12.5")


def test_correlated_variants_bootstrap_as_sources_not_independent_images():
    rows = [{"group_id": "a", "before": True, "after": False} for _ in range(50)]
    rows += [{"group_id": "b", "before": True, "after": True} for _ in range(50)]
    stats = task_summary(rows, 1000, 42)
    assert stats["independent_groups"] == 2
    assert stats["delta_ci95_pp"] == [-100, 0]
    assert stats["insufficient_groups"]


def test_mismatched_inputs_missing_rows_and_duplicate_predictions_fail():
    rows = [sample(0, split="dev")]
    with pytest.raises(ValueError, match="Input tensors changed"):
        compare(rows, [prediction(0)], [prediction(0,fingerprint="changed")])
    with pytest.raises(ValueError, match="missing=1"):
        compare(rows, [prediction(0)], [])
    with pytest.raises(ValueError, match="Duplicate prediction"):
        compare(rows, [prediction(0)], [prediction(0),prediction(0)])


def test_measured_recipe_constraints_reject_proxy_only_success():
    candidates = [dict(name="proxy",weight_bytes=100,task_em={"ocr":1},export_reload_verified=False),
                  dict(name="large",weight_bytes=300,task_em={"ocr":1},export_reload_verified=True),
                  dict(name="bad-ocr",weight_bytes=130,task_em={"ocr":0.7},export_reload_verified=True),
                  dict(name="verified",weight_bytes=180,task_em={"ocr":0.98},export_reload_verified=True)]
    result = choose_recipe(candidates, max_weight_bytes=200, min_task_em={"ocr":0.95})
    assert result["winner"]["name"] == "verified"
    assert sum(a["accepted"] for a in result["audit"]) == 1
    assert choose_recipe(candidates,max_weight_bytes=120,min_task_em={"ocr":0.95})["winner"] is None


def test_report_escapes_untrusted_model_outputs(tmp_path):
    result = compare([sample(0,split="dev")], [prediction(0)], [prediction(0,"<script>alert(1)</script>")])
    output = tmp_path / "report.html"
    render(result,output)
    assert "<script>" not in output.read_text()
    assert "&lt;script&gt;" in output.read_text()
    assert json.loads(output.with_suffix(".json").read_text())["overall"]["regressions"] == 1


def test_changed_decoding_cannot_be_claimed_as_quantization(tmp_path):
    ref, candidate = tmp_path / "ref", tmp_path / "candidate"
    ref.mkdir(); candidate.mkdir()
    controls = dict(dataset_sha256="data",split="dev",decode={"do_sample":False,"max_new_tokens":24},seed=42,
                    max_pixels=393216,min_pixels=3136,use_fast=False,processor_files={"tokenizer":"hash"})
    record = dict(base_revision="model",mode="bf16",controls=controls)
    (ref/"manifest.json").write_text(json.dumps(record))
    controls["decode"]["max_new_tokens"] = 2
    (candidate/"manifest.json").write_text(json.dumps(record))
    with pytest.raises(ValueError,match="decode"):
        validate_manifests(ref/"predictions.jsonl",candidate/"predictions.jsonl")


def test_endpoint_order_regression_does_not_fake_relationship_regression():
    row = dict(sample(0,split="dev",task="graph_edge"),answer="A,C")
    result = compare([row],[prediction(0,"A,C")],[prediction(0,"C,A")])
    assert result["tasks"]["graph_edge"]["delta_pp"] == -100
    assert result["semantic_tasks"]["graph_edge"]["delta_pp"] == 0
    assert result["graph_format_rates"] == {"reference":1,"candidate":0}
