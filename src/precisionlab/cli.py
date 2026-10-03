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
        provenance = validate_manifests(a.reference, a.candidate, allow_input_change=a.allow_input_change)
        result = compare(rows, read_jsonl(a.reference), read_jsonl(a.candidate), allow_input_change=a.allow_input_change)
        result["provenance"] = provenance
        result["dataset_sha256"] = fingerprint(rows, a.dataset)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        render(result, a.output, dataset_root=a.dataset.parent)
        print(json.dumps(result["tasks"]))
    elif a.command == "choose":
        result = choose_recipe(json.loads(a.candidates.read_text()), **json.loads(a.constraints.read_text()))
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))


if __name__ == "__main__":
    main()
