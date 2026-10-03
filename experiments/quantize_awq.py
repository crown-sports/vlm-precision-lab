"""Use upstream AWQ W4A16; selected calibration ids cannot be dev/test ids.

Adapted from the official LLM Compressor multimodal Qwen3-VL example.
AWQ scales, packing and export are upstream functionality, not our invention.
"""

import argparse
from datetime import datetime, timezone
import faulthandler
import importlib.metadata
import json
from pathlib import Path
import resource
import time
import types

from datasets import Array2D, Dataset, Features, Sequence, Value
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.awq import AWQModifier
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
import llmcompressor.pipelines.sequential.pipeline as sequential_pipeline

from precisionlab.data import fingerprint, load_samples, read_jsonl, sha256
from run_hf import prepare, tensor_fingerprint
from qwen_offload_compat import fast_pos_embed_interpolate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--selected", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--retain-layers", type=int, nargs="*", default=[])
    p.add_argument("--max-pixels", type=int, default=393216)
    p.add_argument("--model-placement", choices=["resident-gpu", "cpu-offload"], default="resident-gpu",
        help="Keep the model on GPU when CPU cgroup memory cannot hold BF16 weights")
    p.add_argument("--activation-cache-device", default="cpu",
        help="Intermediate cache device; cuda:1 uses a second visible GPU")
    a = p.parse_args()
    faulthandler.enable()
    faulthandler.dump_traceback_later(180, repeat=True)
    if a.output.exists():
        p.error("Use a new model output directory")
    torch.set_num_threads(4); torch.manual_seed(42)
    all_rows = load_samples(a.dataset)
    calibration = {r["id"]: r for r in all_rows if r["split"] == "calibration"}
    selected = read_jsonl(a.selected)
    if not selected or len({r["id"] for r in selected}) != len(selected):
        p.error("Empty or duplicate calibration selection")
    for row in selected:
        if row["id"] not in calibration or row != calibration[row["id"]]:
            p.error("Selection must contain unchanged calibration rows from the full dataset")
    processor = AutoProcessor.from_pretrained(a.model, local_files_only=True, min_pixels=3136, max_pixels=a.max_pixels, use_fast=False)
    prepared = [prepare(processor, row, a.dataset) for row in selected]
    max_length = max(int(x["input_ids"].numel()) for x in prepared)
    # Same upstream W4A16 default group size; ignore visual and lm_head consistently.
    ignore = ["re:.*lm_head", "re:.*visual.*"] + [f"re:.*layers\\.{layer}\\..*" for layer in a.retain_layers]
    recipe = [AWQModifier(duo_scaling=False), QuantizationModifier(scheme="W4A16", ignore=ignore)]
    # Explicit float32 arrays avoid millions of Python float objects per image.
    features = Features({"input_ids": Sequence(Value("int64")), "attention_mask": Sequence(Value("int64")),
        "pixel_values": Array2D(shape=(None, prepared[0]["pixel_values"].shape[1]), dtype="float32"),
        "image_grid_thw": Array2D(shape=(None, 3), dtype="int64")})
    arrow_rows = [{k: v.squeeze(0).numpy() if k in {"input_ids", "attention_mask"} else v.numpy()
                   for k, v in inputs.items()} for inputs in prepared]
    print(json.dumps({"stage": "building_float32_arrow", "samples": len(selected), "cpu_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}), flush=True)
    dataset = Dataset.from_list(arrow_rows, features=features).with_format("numpy")
    print(json.dumps({"stage": "loading_model", "cpu_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}), flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(a.model, dtype=torch.bfloat16,
        device_map="cuda:0" if a.model_placement == "resident-gpu" else "cpu",
        attn_implementation="sdpa", local_files_only=True)
    # Known accelerate/meta issue in this pinned stack; use the credited upstream fix.
    model.model.visual.fast_pos_embed_interpolate = types.MethodType(fast_pos_embed_interpolate, model.model.visual)

    def collator(batch):
        if len(batch) != 1:
            raise ValueError("Only calibration batch size one is supported")
        # input_ids/masks/grids are integer; pixel_values must stay float32.
        result = {k: torch.tensor(v, dtype=torch.float32 if k == "pixel_values" else torch.int64) for k, v in batch[0].items()}
        for key in ("input_ids", "attention_mask"):
            result[key] = result[key].unsqueeze(0)
        return result

    # Verify float32 serialization and batch layout before calibrating weights.
    for i, original in enumerate(prepared):
        restored = collator([dataset[i]])
        if tensor_fingerprint(original) != tensor_fingerprint(restored):
            raise ValueError(f"Calibration tensors changed during serialization: {selected[i]['id']}")
    calibration_input_tokens = sum(int(x["input_ids"].numel()) for x in prepared)
    del original, restored, prepared, arrow_rows

    started = time.monotonic()
    print(json.dumps({"stage": "calibrating", "cpu_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}), flush=True)
    # Version 0.9's dispatcher unconditionally moves weights to CPU. In a
    # dedicated resident-GPU process, bypass only that dispatch step; tracing,
    # calibration and AWQ arithmetic remain upstream. Restore the function even
    # on failure. This is a pinned-stack memory workaround, not a new quantizer.
    original_dispatch = sequential_pipeline.dispatch_for_sequential
    if a.model_placement == "resident-gpu":
        if any(t.device.type != "cuda" for t in model.parameters()):
            raise ValueError("Resident-GPU calibration requires all parameters on GPU")
        sequential_pipeline.dispatch_for_sequential = lambda module: module
    try:
        oneshot(model=model, tokenizer=processor.tokenizer, dataset=dataset, recipe=recipe,
            max_seq_length=max_length, num_calibration_samples=len(selected), data_collator=collator,
            sequential_targets=["Qwen3VLTextDecoderLayer"],
            sequential_offload_device=a.activation_cache_device)
    finally:
        sequential_pipeline.dispatch_for_sequential = original_dispatch
    a.output.mkdir(parents=True)
    model.save_pretrained(a.output, save_compressed=True)
    processor.save_pretrained(a.output)
    manifest = {"schema_version": 1, "method": "upstream AWQ W4A16", "llmcompressor": importlib.metadata.version("llmcompressor"),
        "upstream_example": "https://github.com/vllm-project/llm-compressor/blob/main/examples/multimodal_vision/qwen3_vl_example.py",
        "base_revision": json.loads(Path(__file__).with_name("model-source.json").read_text())["sha"],
        "calibration_dataset_sha256": fingerprint(selected, a.dataset), "selected_file_sha256": sha256(a.selected),
        "samples": len(selected), "calibration_input_tokens": calibration_input_tokens,
        "max_sequence_length": max_length, "max_pixels": a.max_pixels, "ignore": ignore,
        "retained_layers": a.retain_layers, "seconds": time.monotonic()-started,
        "weight_file_bytes": sum(f.stat().st_size for f in a.output.glob("*.safetensors")),
        "cpu_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "calibration_tensors_verified": True,
        "model_placement": a.model_placement, "activation_cache_device": a.activation_cache_device,
        "offload_workaround": "https://github.com/vllm-project/llm-compressor/pull/1958",
        "offload_workaround_sha256": sha256(Path(__file__).with_name("qwen_offload_compat.py")),
        "script_sha256": sha256(__file__), "recorded_at": datetime.now(timezone.utc).isoformat(),
        "export_verified": True, "reload_verified": False,
        "warning": "Export success does not verify reload quality, optimized kernels, device memory savings or latency"}
    (a.output / "compression-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
