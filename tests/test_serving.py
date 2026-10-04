"""Protocol fixtures verify the client; they are never model benchmark results."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

from PIL import Image
import pytest

from precisionlab.data import fingerprint, read_jsonl, write_jsonl
from precisionlab.metrics import compare
from precisionlab.provenance import validate_manifests
from precisionlab.serving import evaluate_service, request_body
from precisionlab.repair import choose_recipe
from precisionlab.gate import gate_service


@pytest.fixture
def labelled_images(tmp_path):
    Image.new("RGB", (3, 3), "white").save(tmp_path / "image.png")
    rows = [{"id": str(i), "group_id": str(i), "image": "image.png", "split": "dev",
        "task": "amount", "prompt": f"Read amount {i}", "answer": "-12.50", "features": []} for i in range(3)]
    file = tmp_path / "samples.jsonl"
    write_jsonl(file, rows)
    return file, rows


@pytest.fixture
def service():
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            prompt = body["messages"][0]["content"][1]["text"]
            if prompt.endswith("1"):
                self.send_response(429)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            events = [{"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]},
                {"choices": [{"index": 0, "delta": {"content": "-12"}}]},
                {"choices": [{"index": 0, "delta": {"content": ".50"}, "finish_reason": "stop"}]},
                {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 3}}]
            raw = ("".join("data: " + json.dumps(e) + "\n\n" for e in events) + "data: [DONE]\n\n").encode()
            if prompt.endswith("2"):
                raw = raw.replace(b"data: [DONE]\n\n", b"")
            if prompt.endswith("3"):
                raw = b'data: {"choices":[{"delta":{"content":"-12.50"},"finish_reason":[]}]}\n\ndata: [DONE]\n\n'
            for i in range(0, len(raw), 7):
                self.wfile.write(raw[i:i+7])
                self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1", seen
    server.shutdown()
    server.server_close()
    thread.join()


def test_service_measures_answers_and_keeps_failures_in_denominator(labelled_images, service, tmp_path, monkeypatch):
    dataset, rows = labelled_images
    endpoint, seen = service
    monkeypatch.setenv("PRECISIONLAB_API_KEY", "test-only-private-key")
    output = tmp_path / "run"
    profile = evaluate_service(dataset, output, endpoint=endpoint, model="fixture-only",
        concurrency=2, warmup=0, timeout=2)
    assert profile["requests"] == 3
    assert profile["successful_requests"] == 1
    assert profile["success_rate"] == profile["task_em"]["amount"] == 1/3
    assert profile["successful_requests_per_second"] == 1/profile["wall_seconds"]
    assert 0 < profile["ttft_p95_ms"] <= profile["latency_p95_ms"]
    predictions = read_jsonl(output / "predictions.jsonl")
    assert predictions[0]["prediction"] == "-12.50"
    assert predictions[1]["http_status"] == 429
    assert predictions[2]["error_type"] == "ValueError"
    assert all("answer" not in str(request) for request in seen)
    assert "test-only-private-key" not in "".join(p.read_text() for p in output.iterdir())
    report = compare(rows, predictions, predictions)
    assert "server-side input tensors" in report["interpretation"]
    assert validate_manifests(output / "predictions.jsonl", output / "predictions.jsonl")["manifests_verified"] is False
    gate = gate_service(dataset, output, {"min_task_em": {"amount": .3}, "min_success_rate": 1})
    assert not gate["accepted"]
    assert [c["metric"] for c in gate["checks"] if not c["passed"]] == ["success_rate"]
    profile["task_em"]["amount"] = 1
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["profile"] = profile
    (output / "manifest.json").write_text(json.dumps(manifest))
    (output / "profile.json").write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="raw prediction records"):
        gate_service(dataset, output, {"min_task_em": {"amount": .9}})


def test_service_gate_exit_code_and_threshold_validation(labelled_images, service, tmp_path, monkeypatch):
    from precisionlab.cli import main
    dataset, _ = labelled_images
    endpoint, _ = service
    output = tmp_path / "single-run"
    evaluate_service(dataset, output, endpoint=endpoint, model="fixture-only", warmup=0, limit=1)
    result = gate_service(dataset, output, {"min_task_em": {"amount": 1}, "min_source_groups": 1})
    assert result["accepted"] and result["source_groups"] == 1
    with pytest.raises(ValueError, match="finite scores"):
        gate_service(dataset, output, {"min_task_em": {"amount": float("nan")}})
    targets = tmp_path / "targets.json"
    targets.write_text(json.dumps({"min_task_em": {"amount": 1}, "min_source_groups": 30}))
    decision = tmp_path / "decision.json"
    monkeypatch.setattr("sys.argv", ["precisionlab", "gate", "--dataset", str(dataset), "--run", str(output),
        "--constraints", str(targets), "--output", str(decision)])
    with pytest.raises(SystemExit) as raised:
        main()
    assert raised.value.code == 2
    assert not json.loads(decision.read_text())["accepted"]


def test_equal_requests_ignore_alias_but_never_masquerade_as_tensors(labelled_images):
    dataset, rows = labelled_images
    a, ha = request_body(rows[0], dataset, "bf16", 24, 42)
    b, hb = request_body(rows[0], dataset, "awq", 24, 42)
    assert ha == hb and a["model"] != b["model"]
    assert ha != request_body(rows[0], dataset, "awq", 32, 42)[1]
    ref = [{"id": r["id"], "prediction": "-12.50", "input_sha256": "same"} for r in rows]
    cand = [{**p, "input_fingerprint_scope": "request"} for p in ref]
    with pytest.raises(ValueError, match="request fingerprints"):
        compare(rows, ref, cand, allow_input_change=True)


def test_malformed_stream_is_a_failed_row_not_a_benchmark_crash(labelled_images, service, tmp_path):
    dataset, rows = labelled_images
    endpoint, _ = service
    write_jsonl(dataset, [{**rows[0], "prompt": "Read amount 3"}])
    output = tmp_path / "malformed-run"
    profile = evaluate_service(dataset, output, endpoint=endpoint, model="fixture-only", warmup=0)
    assert profile["success_rate"] == profile["task_em"]["amount"] == 0
    assert profile["ttft_p95_ms"] is None
    assert read_jsonl(output / "predictions.jsonl")[0]["error_type"] == "ValueError"
    assert not gate_service(dataset, output, {"min_task_em": {"amount": 1}})["accepted"]


def test_comparison_refuses_changed_labels_or_predictions(labelled_images, service, tmp_path):
    dataset, rows = labelled_images
    endpoint, _ = service
    write_jsonl(dataset, rows[:1])
    output = tmp_path / "comparison-run"
    evaluate_service(dataset, output, endpoint=endpoint, model="fixture-only",
        base_revision="fixture-revision", warmup=0)
    predictions = output / "predictions.jsonl"
    digest = fingerprint(rows[:1], dataset)
    assert validate_manifests(predictions, predictions, dataset_sha256=digest)["manifests_verified"]
    changed = [{**rows[0], "answer": "different label"}]
    with pytest.raises(ValueError, match="dataset differs"):
        validate_manifests(predictions, predictions, dataset_sha256=fingerprint(changed, dataset))
    records = read_jsonl(predictions)
    records[0]["prediction"] = "edited after measurement"
    write_jsonl(predictions, records)
    with pytest.raises(ValueError, match="predictions changed"):
        validate_manifests(predictions, predictions, dataset_sha256=digest)


def test_endpoint_credentials_and_invalid_warmup_are_not_recorded(labelled_images, service, tmp_path):
    dataset, _ = labelled_images
    with pytest.raises(ValueError, match="without credentials"):
        evaluate_service(dataset, tmp_path / "invalid", endpoint="http://secret:pw@localhost/v1", model="fixture")
    assert not (tmp_path / "invalid").exists()
    endpoint, _ = service
    with pytest.raises(ValueError, match="warmup failed"):
        evaluate_service(dataset, tmp_path / "warmup", endpoint=endpoint, model="fixture", warmup=2)
    assert not (tmp_path / "warmup/profile.json").exists()


def test_recipe_runtime_gates_require_equal_workloads_and_real_measurements():
    def candidate(name, size, throughput, success=1):
        return dict(name=name, weight_bytes=size, task_em={"amount": .99},
            export_reload_verified=True, service_verified=True,
            service_profile=dict(workload_sha256="fixture-workload", success_rate=success,
                successful_requests_per_second=throughput, ttft_p95_ms=50))
    options = [candidate("tiny-but-unreliable", 100, 30, .5), candidate("fast", 200, 20), candidate("small", 150, 10)]
    constraints = dict(max_weight_bytes=250, min_task_em={"amount": .95}, min_success_rate=.99,
        min_requests_per_second=5, max_ttft_ms=100, objective="throughput")
    result = choose_recipe(options, **constraints)
    assert result["winner"]["name"] == "fast"
    assert not result["audit"][0]["accepted"]
    assert choose_recipe(options, **constraints, max_peak_device_memory_bytes=1000)["winner"] is None
    options[1]["service_profile"]["workload_sha256"] = "different-concurrency"
    with pytest.raises(ValueError, match="different workloads"):
        choose_recipe(options, **constraints)
