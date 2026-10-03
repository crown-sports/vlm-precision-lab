"""Fixed-input Qwen3-VL baseline, RTN fake quant, or compressed-tensors reload.

Fake quant is diagnostic only. Dense/dequantized HF execution is not proof of
INT4-kernel latency or device-memory savings. Use a separate engine control.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import time

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from precisionlab.data import fingerprint, load_samples, sha256
from precisionlab.metrics import exact


def prepare(processor, row, dataset):
    # The answer and labels are not passed to the model or processor.
    im = Image.open(Path(dataset).parent / row["image"]).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": row["prompt"]}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[text], images=[im], return_tensors="pt", padding=False)


def tensor_fingerprint(inputs):
    h = hashlib.sha256()
    for key, value in sorted(inputs.items()):
        tensor = value.detach().cpu().contiguous()
        h.update(key.encode()); h.update(str(tensor.dtype).encode()); h.update(str(list(tensor.shape)).encode())
        h.update(tensor.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def processor_files(model_path):
    names = ["chat_template.json", "preprocessor_config.json", "video_preprocessor_config.json", "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt"]
    return {name: sha256(Path(model_path) / name) for name in names if (Path(model_path) / name).exists()}


def fake_rtn(model, group_size, retained):
    """Symmetric W4, per-group RTN; restore no layers via incompatible AWQ scales."""
    changed = []
    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.Linear) or "visual" in name or name == "lm_head":
                continue
            # Retaining a whole decoder layer keeps Q/K/V at the same precision.
            if any(f".layers.{layer}." in name for layer in retained):
                continue
            w = module.weight
            if w.shape[-1] % group_size:
                raise ValueError(f"Group size does not divide {name}")
            # Work in bounded row chunks rather than doubling the largest tensor.
            for offset in range(0, w.shape[0], 128):
                block = w[offset:offset+128].float().reshape(-1, group_size)
                scale = block.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 7
                block = (block / scale).round().clamp(-7, 7) * scale
                w[offset:offset+128].copy_(block.reshape_as(w[offset:offset+128]).to(w.dtype))
            changed.append({"module": name, "parameters": w.numel()})
    return changed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--split", choices=["calibration", "dev", "test"], default="dev")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=["bf16", "fake-rtn", "reload-hf", "costs"], default="bf16")
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--retain-layers", type=int, nargs="*", default=[])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-pixels", type=int, default=393216)
    p.add_argument("--max-new-tokens", type=int, default=24)
    a = p.parse_args()
    if a.output.exists():
        p.error("Use a new output directory")
    a.output.mkdir(parents=True)
    random.seed(42); torch.manual_seed(42)
    all_rows = load_samples(a.dataset)
    rows = [r for r in all_rows if r["split"] == a.split]
    if a.limit:
        rows = rows[:a.limit]
    if not rows:
        p.error("Empty selected split")
    if a.mode == "costs" and a.split != "calibration":
        p.error("Calibration costs must be measured on the calibration split")
    processor = AutoProcessor.from_pretrained(a.model, local_files_only=True, min_pixels=3136, max_pixels=a.max_pixels, use_fast=False)
    prepared = [(row, prepare(processor, row, a.dataset)) for row in rows]
    costs = {r["id"]: int(x["input_ids"].numel()) for r, x in prepared}
    (a.output / "costs.json").write_text(json.dumps(costs, indent=2) + "\n")
    controls = {"dataset_sha256": fingerprint(rows, a.dataset), "split": a.split,
                "processor_files": processor_files(a.model), "min_pixels": 3136, "max_pixels": a.max_pixels,
                "use_fast": False, "seed": 42,
                "decode": {"do_sample": False, "max_new_tokens": a.max_new_tokens, "use_cache": True}}
    if a.mode == "costs":
        (a.output / "manifest.json").write_text(json.dumps(controls, indent=2) + "\n")
        return
    torch.set_num_threads(4)
    started = time.monotonic()
    model = Qwen3VLForConditionalGeneration.from_pretrained(a.model, dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, attn_implementation="sdpa", local_files_only=True).eval()
    torch.cuda.synchronize()
    load_seconds = time.monotonic() - started
    changes = fake_rtn(model, a.group_size, a.retain_layers) if a.mode == "fake-rtn" else []
    torch.cuda.reset_peak_memory_stats()
    predictions = []
    # Warmup is identical across modes and excluded from measured generation time.
    first = prepared[0][1].to("cuda")
    with torch.inference_mode():
        model.generate(**first, do_sample=False, max_new_tokens=2, use_cache=True)
    torch.cuda.synchronize()
    total_started = time.monotonic()
    with (a.output / "predictions.jsonl").open("w") as stream, torch.inference_mode():
        for row, inputs in prepared:
            input_sha = tensor_fingerprint(inputs)
            prefix = inputs["input_ids"].shape[-1]
            input_tokens = inputs["input_ids"].numel()
            inputs = inputs.to("cuda")
            torch.cuda.synchronize(); tick = time.monotonic()
            result = model.generate(**inputs, **controls["decode"])
            torch.cuda.synchronize(); elapsed = time.monotonic() - tick
            suffix = result[:, prefix:]
            text = processor.batch_decode(suffix, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            record = {"id": row["id"], "prediction": text, "input_sha256": input_sha,
                      "input_tokens": input_tokens, "output_tokens": suffix.numel(), "generation_seconds": elapsed}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n"); stream.flush()
            predictions.append(record)
            if len(predictions) % 25 == 0:
                print(json.dumps({"completed": len(predictions), "total": len(rows), "seconds": time.monotonic()-total_started}), flush=True)
    times = sorted(r["generation_seconds"] for r in predictions)
    correct = Counter(r["task"] for r, pred in zip(rows, predictions) if exact(r["answer"], pred["prediction"]))
    counts = Counter(r["task"] for r in rows)
    packages = {}
    for name in ["torch", "transformers", "llmcompressor", "compressed-tensors", "Pillow", "accelerate"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    manifest = {"schema_version": 1, "mode": a.mode, "controls": controls, "samples": len(rows),
        "model_repository": "Qwen/Qwen3-VL-8B-Instruct", "base_revision": json.loads(Path(__file__).with_name("model-source.json").read_text())["sha"],
        "checkpoint_config_sha256": sha256(a.model / "config.json"), "retained_layers": a.retain_layers,
        "fake_rtn_modules": changes, "group_size": a.group_size, "packages": packages,
        "python": platform.python_version(), "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
        "load_seconds": load_seconds, "inference_seconds": time.monotonic()-total_started,
        "generation_p50_ms": times[len(times)//2]*1000, "generation_p95_ms": times[min(len(times)-1, int(len(times)*0.95))]*1000,
        "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(),
        "weight_file_bytes": sum(f.stat().st_size for f in a.model.glob("*.safetensors")),
        "task_em": {t: correct[t]/counts[t] for t in counts}, "predictions_sha256": sha256(a.output / "predictions.jsonl"),
        "script_sha256": sha256(__file__), "recorded_at": datetime.now(timezone.utc).isoformat(),
        "performance_scope": "batch=1 HF generation only; excludes image preparation, fingerprinting and service queuing; dense execution is not an INT4 kernel benchmark"}
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"mode": a.mode, "task_em": manifest["task_em"], "inference_seconds": manifest["inference_seconds"], "peak_bytes": manifest["peak_torch_allocated_bytes"]}), flush=True)


if __name__ == "__main__":
    main()
