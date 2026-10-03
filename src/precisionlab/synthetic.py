"""Original, deterministic diagnostic fixtures; not a real-world benchmark."""

import hashlib
import json
from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFont

from .data import sha256, write_jsonl


def generate(output, font, *, calibration=192, dev=300, test=0, seed=741):
    output, font = Path(output), Path(font)
    if output.exists():
        raise ValueError("Use a new directory; existing datasets are immutable")
    if any(n < 0 or n % 2 for n in (calibration, dev, test)):
        raise ValueError("Use nonnegative even counts (two variants per source)")
    output.mkdir(parents=True)
    (output / "images").mkdir()
    rows = []
    tasks = ["numeric_ocr", "identifier_ocr", "graph_edge"]
    for split, count in [("calibration", calibration), ("dev", dev), ("test", test)]:
        for group in range(count // 2):
            task = tasks[group % len(tasks)]
            group_id = hashlib.sha256(f"{seed}:{split}:{group}".encode()).hexdigest()[:16]
            rng = random.Random(int(group_id, 16))
            size = rng.choice([12, 16, 22, 28])
            face = ImageFont.truetype(str(font), size)
            title = ImageFont.truetype(str(font), 28)
            identifier = f"SW-{rng.randrange(100, 1000)}-P{rng.randrange(1, 49):02d}"
            amount = rng.randrange(1000, 99000)
            sign = "-" if rng.random() < 0.25 else ""
            targets = rng.sample(["A", "B", "C", "D", "E", "F"], 4)
            labels = rng.sample(["L1", "L2", "L3", "L4", "L5", "L6"], 2)
            foils = [f"{rng.randrange(50, 990)}.{rng.randrange(100):02d}" for _ in range(4)]
            vlan = str(rng.choice([10, 20, 200, 230]))
            # The second variant changes the answer, not just its surrounding prompt.
            # Both variants remain in the same source group and split.
            for variant in range(2):
                im = Image.new("RGB", (768, 512), "#f7f8fb")
                draw = ImageDraw.Draw(im)
                draw.rounded_rectangle((24, 24, 744, 488), 12, fill="white", outline="#b8c2d1", width=2)
                draw.text((50, 46), "FIELD INSPECTION RECORD", font=title, fill="#142339")
                draw.line((50, 95, 718, 95), fill="#d6dce5", width=2)
                if task == "numeric_ocr":
                    value = amount + variant
                    answer = f"{sign}{value//100}.{value%100:02d}"
                    names = ["Subtotal", "Tax", "Amount due", "Previous balance"]
                    values = list(foils)
                    values[2] = answer
                    for i, (name, value) in enumerate(zip(names, values)):
                        y = 134 + i * 64
                        draw.text((62, y), name, font=face, fill="#1d2a3e")
                        draw.text((475, y), value, font=face, fill="#1d2a3e")
                    prompt = "Read the Amount due field. Return only its exact value, preserving the sign and both decimal digits."
                    features = ["digits", "decimal", "negative" if sign else "positive", f"font:{size}"]
                elif task == "identifier_ocr":
                    answer = identifier if not variant else identifier[:-2] + f"{(int(identifier[-2:]) % 48)+1:02d}"
                    draw.text((62, 142), "Target interface:", font=face, fill="#1d2a3e")
                    draw.text((352, 142), answer, font=face, fill="#1d2a3e")
                    draw.text((62, 224), "VLAN: " + vlan, font=face, fill="#1d2a3e")
                    draw.text((62, 310), "Status: operational", font=face, fill="#1d2a3e")
                    prompt = "Read the Target interface field. Return only the complete identifier, preserving hyphens and zero padding."
                    features = ["digits", "identifier", "hyphen", "zero-padding", f"font:{size}"]
                else:
                    pairs = [(targets[0], targets[1]), (targets[2], targets[3])]
                    if variant:
                        pairs = [(targets[0], targets[3]), (targets[2], targets[1])]
                    positions = {n: (120 + (i % 2) * 460, 175 + (i // 2) * 218) for i, n in enumerate(targets)}
                    for label, (a, b) in zip(labels, pairs):
                        x1, y1 = positions[a]; x2, y2 = positions[b]
                        draw.line((x1, y1, x2, y2), fill="#485e80", width=3)
                        # Labels stay away from the shared crossing point.
                        cx, cy = int(x1 * 0.65 + x2 * 0.35), int(y1 * 0.65 + y2 * 0.35)
                        draw.rectangle((cx - 27, cy - 21, cx + 27, cy + 18), fill="white")
                        draw.text((cx - 19, cy - 16), label, font=face, fill="#172b49")
                    for node, (x, y) in positions.items():
                        draw.ellipse((x - 29, y - 29, x + 29, y + 29), fill="#e4edfb", outline="#355c93", width=2)
                        draw.text((x - 10, y - 17), node, font=title, fill="#142b4d")
                    answer = ",".join(sorted(pairs[0]))
                    prompt = f"Which two nodes are directly joined by the link labelled {labels[0]}? Return only the node letters in alphabetical order, separated by one comma (for example A,C)."
                    features = ["graph", "edge-label", "crossing" if variant else "parallel", f"font:{size}"]
                sample_id = f"{group_id}-{variant}"
                image = f"images/{sample_id}.png"
                im.save(output / image)
                rows.append({"id": sample_id, "group_id": group_id, "split": split, "task": task,
                    "image": image, "image_sha256": sha256(output / image), "prompt": prompt, "answer": answer,
                    "features": features, "variant": variant, "source": "original synthetic diagnostic fixture"})
    write_jsonl(output / "samples.jsonl", rows)
    card = {"seed": seed, "font_sha256": sha256(font), "counts": {"calibration": calibration, "dev": dev, "test": test},
            "groups": "two answer-changing variants per independent generated record",
            "scope": "shared renderer families; NOT held-out template families or real-world generalization",
            "license": "Apache-2.0 generated images/annotations; font not redistributed"}
    (output / "dataset-card.json").write_text(json.dumps(card, indent=2) + "\n")
    return rows
