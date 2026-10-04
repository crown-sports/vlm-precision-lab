"""Measure a vision chat service against the user's labelled images.

The fingerprint covers the request, not the server's hidden input tensors.
Throughput is closed-loop successful requests / full measurement wall time.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import base64
import hashlib
from http.client import HTTPException
import json
import math
import mimetypes
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .data import fingerprint, load_samples, sha256, write_jsonl
from .metrics import exact


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def request_body(sample, dataset, model, max_tokens, seed):
    path = Path(dataset).parent / sample["image"]
    mime = mimetypes.guess_type(path.name)[0]
    if mime not in {"image/png", "image/jpeg", "image/webp"}:
        raise ValueError("Service images must be PNG, JPEG or WebP")
    image = "data:" + mime + ";base64," + base64.b64encode(path.read_bytes()).decode()
    body = {"model": model, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": image}},
        {"type": "text", "text": sample["prompt"]},
    ]}], "temperature": 0, "seed": seed, "max_tokens": max_tokens,
        "stream": True, "stream_options": {"include_usage": True}}
    # Different served model aliases must not make otherwise equal inputs differ.
    canonical = {k: v for k, v in body.items() if k != "model"}
    digest = hashlib.sha256(json.dumps(canonical, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    return body, digest


def stream_prediction(url, body, *, api_key, timeout):
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    tick = time.monotonic()
    encoded = json.dumps(body).encode()
    request = Request(url, data=encoded, headers=headers, method="POST")
    first_content, finish, usage, pieces, done = None, None, {}, [], False
    with urlopen(request, timeout=timeout) as response:
        if "text/event-stream" not in response.headers.get("Content-Type", ""):
            raise ValueError("Expected a streaming SSE response")
        data_lines = []
        for raw in response:
            if time.monotonic() - tick > timeout:
                raise TimeoutError("Stream exceeded the request deadline")
            line = raw.decode("utf-8").rstrip("\r\n")
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip(" "))
                continue
            if line or not data_lines:
                continue
            event = "\n".join(data_lines)
            data_lines = []
            if event == "[DONE]":
                done = True
                break
            item = json.loads(event)
            if not isinstance(item, dict):
                raise ValueError("Expected a JSON object in each stream event")
            if item.get("error"):
                raise ValueError("Server returned an error event")
            if item.get("usage") is not None:
                if not isinstance(item["usage"], dict):
                    raise ValueError("Invalid token usage record")
                usage = item["usage"]
            choices = item.get("choices", [])
            if not isinstance(choices, list):
                raise ValueError("Invalid choices record")
            if not choices:
                continue
            if len(choices) != 1 or not isinstance(choices[0], dict) or choices[0].get("index", 0) != 0:
                raise ValueError("Expected exactly one completion")
            choice = choices[0]
            delta = choice.get("delta", {})
            if not isinstance(delta, dict):
                raise ValueError("Invalid content delta")
            content = delta.get("content")
            if content is not None:
                if not isinstance(content, str):
                    raise ValueError("Non-text content delta")
                if content and first_content is None:
                    first_content = time.monotonic() - tick
                pieces.append(content)
            if choice.get("finish_reason") is not None:
                if not isinstance(choice["finish_reason"], str):
                    raise ValueError("Invalid finish reason")
                finish = choice["finish_reason"]
    if not done or finish not in {"stop", "length"} or first_content is None:
        raise ValueError("Incomplete stream or no visible answer")
    return {"prediction": "".join(pieces), "latency_seconds": time.monotonic() - tick,
        "ttft_seconds": first_content, "finish_reason": finish,
        "input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens")}


def evaluate_service(dataset, output, *, endpoint, model, base_revision=None,
                     split="dev", concurrency=1, max_tokens=32, seed=42,
                     warmup=2, timeout=120, limit=0, server_config=None):
    parsed = urlsplit(endpoint)
    if (parsed.scheme not in {"http", "https"} or not parsed.netloc or
            parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("Supply an HTTP(S) API base URL without credentials, query or fragment")
    if concurrency < 1 or max_tokens < 1 or timeout <= 0 or warmup < 0 or limit < 0:
        raise ValueError("Invalid concurrency, token, timeout, warmup or limit")
    rows = [r for r in load_samples(dataset) if r["split"] == split]
    if limit:
        rows = rows[:limit]
    if not rows:
        raise ValueError("No samples in the requested split")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    url = endpoint.rstrip("/") + "/chat/completions"
    api_key = os.environ.get("PRECISIONLAB_API_KEY", "")
    controls = {"dataset_sha256": fingerprint(rows, dataset), "split": split,
        "seed": seed, "decode": {"temperature": 0, "max_tokens": max_tokens},
        "input_fingerprint_scope": "request", "server_config_declared": server_config or {}}

    def predict(row):
        body, digest = request_body(row, dataset, model, max_tokens, seed)
        tick = time.monotonic()
        try:
            prediction = stream_prediction(url, body, api_key=api_key, timeout=timeout)
            prediction["success"] = True
        except (HTTPError, URLError, HTTPException, TimeoutError, OSError, ValueError) as error:
            # Do not put remote error bodies, URLs or auth material in results.
            prediction = {"prediction": "", "success": False, "error_type": type(error).__name__,
                "latency_seconds": time.monotonic() - tick}
            if isinstance(error, HTTPError):
                prediction["http_status"] = error.code
        return {"id": row["id"], "input_sha256": digest,
            "input_fingerprint_scope": "request", **prediction}

    # The whole benchmark is invalid if its service cannot complete a warmup.
    for i in range(warmup):
        if not predict(rows[i % len(rows)])["success"]:
            raise ValueError("Service warmup failed; no usable benchmark was recorded")
    tick = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        predictions = list(pool.map(predict, rows))
    elapsed = time.monotonic() - tick
    write_jsonl(output / "predictions.jsonl", predictions)
    successful = [p for p in predictions if p["success"]]
    counts, correct = {}, {}
    for row, result in zip(rows, predictions):
        task = row["task"]
        counts[task] = counts.get(task, 0) + 1
        correct[task] = correct.get(task, 0) + int(result["success"] and exact(row["answer"], result["prediction"]))
    profile = {"requests": len(rows), "successful_requests": len(successful),
        "success_rate": len(successful) / len(rows), "wall_seconds": elapsed,
        "successful_requests_per_second": len(successful) / elapsed,
        "latency_p50_ms": percentile([p["latency_seconds"] * 1000 for p in predictions], 0.5),
        "latency_p95_ms": percentile([p["latency_seconds"] * 1000 for p in predictions], 0.95),
        "ttft_p95_ms": percentile([p["ttft_seconds"] * 1000 for p in successful], 0.95),
        "truncated_requests": sum(p.get("finish_reason") == "length" for p in successful),
        "task_em": {t: correct[t] / counts[t] for t in sorted(counts)},
        "concurrency": concurrency, "warmup_requests": warmup,
        "latency_scope": "request serialization through completion or failure; includes all requests, excludes client image encoding",
        "ttft_scope": "first nonempty visible content delta; excludes hidden reasoning tokens",
        "throughput_scope": "closed-loop; includes client preparation and failed requests in wall time",
        "cache_scope": "warmup performed; no server cache reset claimed"}
    workload = {"controls": controls, "concurrency": concurrency, "warmup": warmup}
    profile["workload_sha256"] = hashlib.sha256(json.dumps(workload, sort_keys=True).encode()).hexdigest()
    manifest = {"schema_version": 2, "mode": "service", "base_revision": base_revision,
        "controls": controls, "profile": profile, "samples": len(rows), "served_model_alias": model,
        "predictions_sha256": sha256(output / "predictions.jsonl"), "script_sha256": sha256(__file__),
        "server_identity_verified": False,
        "evidence_scope": "same client requests; model revision and server settings are declarations, not server attestation",
        "recorded_at": datetime.now(timezone.utc).isoformat()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    return profile
