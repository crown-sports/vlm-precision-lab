"""Render scientific figures from paired report JSON; no invented model data."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--candidate-label", default="W4 RTN fake quant")
    p.add_argument("--title", default="Output-order loss is not relationship-perception loss")
    p.add_argument("--footer", default="Qwen3-VL-8B · 300 synthetic development cases / 150 sources · same input hashes\nDense fake-quant execution; no INT4 kernel or real-world generalization claim.")
    a = p.parse_args()
    result = json.loads(a.report.read_text())
    items = [("Numeric fields\nexact match", result["tasks"]["numeric_ocr"]),
             ("Interface IDs\nexact match", result["tasks"]["identifier_ocr"]),
             ("Graph endpoints\nstrict exact match", result["tasks"]["graph_edge"]),
             ("Graph endpoints\nunordered content", result["semantic_tasks"]["graph_edge"])]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(11.4, 5.3), constrained_layout=True)
    for i, (label, stats) in enumerate(items):
        for offset, key, color, title in [(-0.18, "reference_em", "#2454ac", "BF16"), (0.18, "candidate_em", "#dd8850", a.candidate_label)]:
            value = stats[key] * 100
            ax.bar(i+offset, value, width=0.32, color=color, label=title if i==0 else None)
            ax.text(i+offset, value+1.8, f"{value:.0f}%", ha="center", fontsize=11)
    ax.set_xticks(range(len(items)), [label for label, _ in items])
    ax.set_ylim(0, 119); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_ylabel("Correct (%)")
    ax.set_axisbelow(True); ax.grid(axis="y", alpha=0.18)
    ax.legend(loc="upper right", frameon=False)
    ax.set_title(a.title, loc="left", pad=20, fontsize=17, weight="bold")
    fig.text(0.01, -0.04, a.footer, color="#5c6573", fontsize=10)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=160, bbox_inches="tight")
    fig.savefig(a.output.with_suffix(".svg"), bbox_inches="tight")
    svg = a.output.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")


if __name__ == "__main__":
    main()
