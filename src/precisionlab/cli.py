import argparse
import json
from pathlib import Path

from .calibration import select
from .data import fingerprint, load_samples, read_jsonl, write_jsonl
from .metrics import compare
from .provenance import validate_manifests
from .repair import choose_recipe
from .report import render
from .synthetic import generate


def main():
    p = argparse.ArgumentParser(description="Paired multimodal precision checks")
    sub = p.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate", help="Create original synthetic diagnostic fixtures")
    g.add_argument("--output", type=Path, required=True)
    g.add_argument("--font", type=Path, required=True)
    g.add_argument("--calibration", type=int, default=192)
    g.add_argument("--dev", type=int, default=300)
    g.add_argument("--test", type=int, default=0)
    g.add_argument("--seed", type=int, default=741)
    s = sub.add_parser("select", help="Select calibration samples under measured token costs")
    s.add_argument("--dataset", type=Path, required=True)
    s.add_argument("--costs", type=Path, required=True)
    s.add_argument("--budget", type=int, required=True)
    s.add_argument("--strategy", choices=["random", "stratified", "coverage"], default="coverage")
    s.add_argument("--output", type=Path, required=True)
    c = sub.add_parser("compare", help="Report paired task regressions")
    c.add_argument("--dataset", type=Path, required=True)
    c.add_argument("--split", choices=["calibration", "dev", "test"], default="dev")
    c.add_argument("--reference", type=Path, required=True)
    c.add_argument("--candidate", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--allow-input-change", action="store_true", help="Descriptive processor control only")
    r = sub.add_parser("choose", help="Select only measured, exported and reloaded recipes")
    r.add_argument("--candidates", type=Path, required=True)
    r.add_argument("--constraints", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    e = sub.add_parser("evaluate", help="Measure quality and latency of a vision chat service")
    e.add_argument("--dataset", type=Path, required=True)
    e.add_argument("--endpoint", required=True, help="API base URL, e.g. http://localhost:8000/v1")
    e.add_argument("--model", required=True, help="Served model alias")
    e.add_argument("--base-revision", help="Declared original model revision")
    e.add_argument("--split", choices=["calibration", "dev", "test"], default="dev")
    e.add_argument("--output", type=Path, required=True)
    e.add_argument("--concurrency", type=int, default=1)
    e.add_argument("--max-tokens", type=int, default=32)
    e.add_argument("--warmup", type=int, default=2)
    e.add_argument("--timeout", type=float, default=120)
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--server-config", type=Path, help="JSON of declared engine/processor settings")
    d = sub.add_parser("import-cord", help="Import verified CORD-v2 parquet files as grouped receipt tasks")
    d.add_argument("--parquet-dir", type=Path, required=True)
    d.add_argument("--source-manifest", type=Path, required=True)
    d.add_argument("--output", type=Path, required=True)
    d.add_argument("--splits", nargs="+", choices=["train", "validation", "test"], default=["validation"])
    b = sub.add_parser("gate", help="Accept or reject a measured service against explicit budgets")
    b.add_argument("--dataset", type=Path, required=True)
    b.add_argument("--run", type=Path, required=True)
    b.add_argument("--constraints", type=Path, required=True)
    b.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.command == "generate":
        rows = generate(a.output, a.font, calibration=a.calibration, dev=a.dev, test=a.test, seed=a.seed)
        print(json.dumps({"samples": len(rows), "path": str(a.output / "samples.jsonl")}))
    elif a.command == "select":
        rows = load_samples(a.dataset)
        costs = json.loads(a.costs.read_text())
        chosen, stats = select(rows, costs, a.budget, strategy=a.strategy)
        write_jsonl(a.output, chosen)
        stats["dataset_sha256"] = fingerprint(rows, a.dataset)
        a.output.with_suffix(".selection.json").write_text(json.dumps(stats, indent=2) + "\n")
        print(json.dumps(stats))
    elif a.command == "compare":
        rows = [r for r in load_samples(a.dataset) if r["split"] == a.split]
        if not rows:
            p.error("No samples in requested split")
        dataset_digest = fingerprint(rows, a.dataset)
        provenance = validate_manifests(a.reference, a.candidate,
            allow_input_change=a.allow_input_change, dataset_sha256=dataset_digest)
        result = compare(rows, read_jsonl(a.reference), read_jsonl(a.candidate), allow_input_change=a.allow_input_change)
        result["provenance"] = provenance
        result["dataset_sha256"] = dataset_digest
        a.output.parent.mkdir(parents=True, exist_ok=True)
        render(result, a.output, dataset_root=a.dataset.parent)
        print(json.dumps(result["tasks"]))
    elif a.command == "choose":
        result = choose_recipe(json.loads(a.candidates.read_text()), **json.loads(a.constraints.read_text()))
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
    elif a.command == "import-cord":
        from .cord import import_cord
        result = import_cord(a.parquet_dir, a.output, a.source_manifest, splits=a.splits)
        print(json.dumps({k: result[k] for k in ("documents", "samples", "skipped_fields", "dataset_sha256")}))
    elif a.command == "evaluate":
        from .serving import evaluate_service
        result = evaluate_service(a.dataset, a.output, endpoint=a.endpoint, model=a.model,
            base_revision=a.base_revision, split=a.split, concurrency=a.concurrency,
            max_tokens=a.max_tokens, warmup=a.warmup, timeout=a.timeout, limit=a.limit,
            server_config=json.loads(a.server_config.read_text()) if a.server_config else None)
        print(json.dumps({k: result[k] for k in ("requests", "success_rate", "task_em", "latency_p95_ms", "ttft_p95_ms", "successful_requests_per_second")}))
    elif a.command == "gate":
        from .gate import gate_service
        result = gate_service(a.dataset, a.run, json.loads(a.constraints.read_text()))
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"accepted": result["accepted"], "failed": [c["metric"] for c in result["checks"] if not c["passed"]],
            "requests": result["requests"], "source_groups": result["source_groups"]}))
        raise SystemExit(0 if result["accepted"] else 2)


if __name__ == "__main__":
    main()
