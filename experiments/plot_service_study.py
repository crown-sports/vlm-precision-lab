"""Plot measured receipt quality and repeated service costs from the analysis."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = json.loads(args.analysis.read_text())
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    colors = {"bf16": "#315da8", "awq": "#d58632"}
    names = {"bf16": "BF16", "awq": "AWQ W4A16"}
    tasks = sorted(value["quality"]["test"]["tasks"])
    x = np.arange(len(tasks))
    for mode, offset, metric in (("bf16", -.18, "reference_em"), ("awq", .18, "candidate_em")):
        scores = [value["quality"]["test"]["tasks"][t][metric] * 100 for t in tasks]
        bars = axes[0, 0].bar(x + offset, scores, .34, color=colors[mode], label=names[mode])
        axes[0, 0].bar_label(bars, fmt="%.1f", fontsize=9)
    axes[0, 0].set_xticks(x, [t.removeprefix("receipt_") + "\nN=" + str(value["quality"]["test"]["tasks"][t]["samples"]) for t in tasks])
    axes[0, 0].set(title="Held-out receipt field exact match", ylabel="Exact match (%)", ylim=(0, 108))
    axes[0, 0].legend(loc="lower right")
    concurrency = sorted(value["performance"]["bf16"], key=int)
    for axis, metric, title, ylabel in ((axes[0, 1], "successful_requests_per_second", "Warm dev workload throughput", "Successful requests / second"),
                                       (axes[1, 0], "latency_p95_ms", "Warm dev request p95 latency", "Latency (ms)")):
        for mode, offset in (("bf16", -.14), ("awq", .14)):
            stats = [value["performance"][mode][c]["metrics"][metric] for c in concurrency]
            centers = np.arange(len(concurrency)) + offset
            axis.plot(centers, [s["median"] for s in stats], "o-", color=colors[mode], label=names[mode])
            for center, stat in zip(centers, stats):
                axis.scatter([center] * len(stat["values"]), stat["values"], color=colors[mode], marker="_", s=80)
        axis.set_xticks(np.arange(len(concurrency)), ["Concurrency " + c for c in concurrency])
        axis.set(title=title, ylabel=ylabel, ylim=(0, None))
        axis.grid(axis="y", alpha=.15)
        axis.legend()
    for mode, offset in (("bf16", -.18), ("awq", .18)):
        peaks = value["memory"][mode]["peak_framebuffer_mib_by_stage"]
        scores = [peaks["startup"] / 1024, max(v for k, v in peaks.items() if k != "startup") / 1024]
        bars = axes[1, 1].bar(np.arange(2) + offset, scores, .34, color=colors[mode], label=names[mode])
        axes[1, 1].bar_label(bars, fmt="%.2f", fontsize=9)
    axes[1, 1].set_xticks(np.arange(2), ["Startup sampled peak", "Workload sampled peak"])
    axes[1, 1].set(title="Same GPU and 2 GiB KV cache", ylabel="Whole-device framebuffer (GiB)", ylim=(0, 32))
    axes[1, 1].legend(loc="lower right")
    fig.suptitle("Qwen3-VL-8B / CORD-v2 / RTX 5090 / vLLM 0.11.0", fontsize=16)
    fig.supxlabel("Quality: 258 questions from 99 labelled test receipts. Timing: 3 repeats of 273 dev questions; marks show individual repeats.\nSingle sequential deployment per variant; prefix and processor caches disabled. Memory includes driver reservation and workspaces.", fontsize=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
