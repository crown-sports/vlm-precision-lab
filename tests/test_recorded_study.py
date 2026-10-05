"""Keep the published service study tied to raw evidence and its negative result.

No GPU inference or unbundled receipt-image verification is performed here.
"""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import unicodedata

import pytest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "experiments/results/2026-10-05/cord-service"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def matches(answer, prediction):
    return unicodedata.normalize("NFKC", answer).strip() == unicodedata.normalize("NFKC", prediction).strip()


def test_recorded_evidence_index_and_measured_scripts_are_complete():
    index = read(EVIDENCE / "evidence-index.json")
    files = {str(p.relative_to(EVIDENCE)): p for p in EVIDENCE.rglob("*")
             if p.is_file() and p.name != "evidence-index.json"}
    assert files.keys() == index["sha256"].keys()
    for name, path in files.items():
        assert not path.is_symlink()
        assert digest(path) == index["sha256"][name], name
    study = read(EVIDENCE / "study.json")
    assert study["status"] == "completed"
    assert digest(EVIDENCE / "measured-runner-initial.py") == study["script_sha256"]
    assert digest(EVIDENCE / "measured-runner-resume.py") == study["resume"]["script_sha256"]
    assert digest(EVIDENCE / "protocol.json") == study["protocol_sha256"]
    assert digest(ROOT / "experiments/cord-service-protocol.json") == study["protocol_sha256"]
    assert read(EVIDENCE / "study-before-resume.json")["status"] == "failed"
    assert read(EVIDENCE / "cord-service-04-resume.exit.json")["exit_code"] == 0
    assert study["resume"]["retained_variant"] == "bf16"
    adapter = read(EVIDENCE / "checkpoint-evidence.json")["config_adapter"]
    assert adapter["weights_changed"] is False
    assert digest(EVIDENCE / "awq-controlled-config.json") == adapter["controlled_sha256"]
    assert read(EVIDENCE / "analysis.json")["awq_kernel_log_evidence"] == ["MarlinLinearKernel"]


def test_recorded_raw_predictions_preserve_counts_scores_and_content_losses():
    analysis = read(EVIDENCE / "analysis.json")
    ledgers = {s: rows(EVIDENCE / f"{s}-samples.jsonl") for s in ("dev", "test")}
    assert len(ledgers["dev"]) == 273
    assert len(ledgers["test"]) == 258
    assert len({s["group_id"] for s in ledgers["test"]}) == 99
    completed, requests, warmups = 0, 0, 0
    shared = read(EVIDENCE / "shared-controls.json")
    for mode in ("bf16", "awq"):
        for path in sorted((EVIDENCE / mode).glob("*/predictions.jsonl")):
            split = "test" if path.parent.name.startswith("test-") else "dev"
            samples = ledgers[split]
            predictions = rows(path)
            indexed = {r["id"]: r for r in predictions}
            assert len(indexed) == len(predictions) == len(samples)
            assert indexed.keys() == {s["id"] for s in samples}
            assert all(r["success"] and r["finish_reason"] == "stop" for r in predictions)
            manifest = read(path.with_name("manifest.json"))
            profile = read(path.with_name("profile.json"))
            assert manifest["predictions_sha256"] == digest(path)
            assert manifest["profile"] == profile
            assert manifest["controls"]["server_config_declared"] == shared
            assert manifest["controls"]["dataset_sha256"] == read(EVIDENCE / f"{split}-source.json")["dataset_sha256"]
            assert profile["requests"] == profile["successful_requests"] == len(predictions)
            assert profile["success_rate"] == 1 and profile["truncated_requests"] == 0
            assert profile["successful_requests_per_second"] == pytest.approx(len(predictions) / profile["wall_seconds"])
            by_task = defaultdict(list)
            for sample in samples:
                by_task[sample["task"]].append(matches(sample["answer"], indexed[sample["id"]]["prediction"]))
            expected = {t: sum(scores) / len(scores) for t, scores in by_task.items()}
            assert profile["task_em"] == expected
            if path.parent.name == f"{split}-c1-r1":
                key = "reference_em" if mode == "bf16" else "candidate_em"
                assert {t: a[key] for t, a in analysis["quality"][split]["tasks"].items()} == expected
            completed += 1
            requests += len(predictions)
            warmups += profile["warmup_requests"]
    assert (completed, requests, warmups) == (14, 3792, 56)
    for mode in ("bf16", "awq"):
        gate = read(EVIDENCE / mode / "test-gate.json")
        assert gate == analysis["test_gates"][mode]
        assert gate["accepted"] is False
        failed = [c["metric"] for c in gate["checks"] if not c["passed"]]
        assert set(failed) == {f"task_em.{t}" for t in analysis["quality"]["test"]["tasks"]}
    reference = {p["id"]: p for p in rows(EVIDENCE / "bf16/test-c1-r1/predictions.jsonl")}
    candidate = {p["id"]: p for p in rows(EVIDENCE / "awq/test-c1-r1/predictions.jsonl")}
    losses = {s["id"]: (s["answer"], candidate[s["id"]]["prediction"]) for s in ledgers["test"]
              if matches(s["answer"], reference[s["id"]]["prediction"])
              and not matches(s["answer"], candidate[s["id"]]["prediction"])}
    prefix = "cord:test-00000-of-00001-9c204eb3f4e11791.parquet"
    assert losses == {f"{prefix}:20:total.total_price": ("377,859", "377,059"),
                      f"{prefix}:42:total.total_price": ("16,500", "50,000")}
    for sample in ledgers["test"]:
        assert reference[sample["id"]]["input_sha256"] == candidate[sample["id"]]["input_sha256"]
    assert analysis["quality"]["test"]["tasks"]["receipt_total_price"]["delta_ci95_pp"][0] < -2
