"""Render the README example from the same evidence used by the replay."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from build_demo import ROOT, load_evidence


def main():
    evidence = load_evidence()
    case = next(c for c in evidence["cases"] if c["id"] == "f65ff21481a45d46-0")
    figure = plt.figure(figsize=(12, 6.5), facecolor="#ffffff")
    left = figure.add_axes([0.02, 0.14, 0.62, 0.73])
    left.imshow(Image.open(ROOT / "examples/diagnostic-v2" / case["image"]))
    left.axis("off")
    right = figure.add_axes([0.68, 0.10, 0.30, 0.76])
    right.axis("off")
    figure.text(0.035, 0.93, "One recorded case: which nodes does L6 connect?", fontsize=18, weight="normal", color="#1b2936")
    right.text(0, 0.90, "Expected (alphabetical order)", fontsize=12, color="#546578")
    right.text(0, 0.79, case["answer"], fontsize=30, family="monospace", color="#1b2936")
    for y, label, key in ((0.62, "BF16", "reference"), (0.28, "AWQ W4A16 / HF reload", "candidate")):
        right.text(0, y, label, fontsize=12, color="#546578")
        right.text(0, y - 0.09, case[key], fontsize=25, family="monospace", color="#1b2936")
        strict = "pass" if case[f"{key}_strict"] else "fail (order)"
        content = "pass" if case[f"{key}_content"] else "fail"
        right.text(0, y - 0.16, f"Strict match: {strict}", fontsize=12, color="#216546" if case[f"{key}_strict"] else "#80541b")
        right.text(0, y - 0.22, f"Endpoint content: {content}", fontsize=12, color="#216546")
    figure.text(0.035, 0.085, "Same input tensor. The AWQ answer preserves the endpoints but violates the requested order.", fontsize=12, color="#546578")
    figure.text(0.035, 0.035, f"Synthetic development case {case['id']} | Qwen3-VL-8B-Instruct | 2026-10-03", fontsize=10, color="#546578")
    target = ROOT / "docs/figures/recorded-case.png"
    figure.savefig(target, dpi=180)
    plt.close(figure)
    print(target)


if __name__ == "__main__":
    main()
