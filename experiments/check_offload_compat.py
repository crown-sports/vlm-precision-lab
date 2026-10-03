"""Record normal-device arithmetic parity for the credited position workaround."""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

from qwen_offload_compat import fast_pos_embed_interpolate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing result")
    torch.manual_seed(42)
    checks = []
    for dtype in [torch.float32, torch.bfloat16]:
        dummy = SimpleNamespace(num_grid_per_side=4,
            config=SimpleNamespace(spatial_merge_size=2),
            pos_embed=torch.nn.Embedding(16, 8, dtype=dtype))
        for grid in [[[1, 4, 4]], [[1, 6, 4], [2, 4, 6]]]:
            tensor = torch.tensor(grid, dtype=torch.int64)
            with torch.no_grad():
                original = Qwen3VLVisionModel.fast_pos_embed_interpolate(dummy, tensor)
                patched = fast_pos_embed_interpolate(dummy, tensor)
            equal = torch.equal(original, patched)
            checks.append(dict(dtype=str(dtype), grid=grid, output_shape=list(original.shape),
                identical=equal, max_abs_error=float((original.float()-patched.float()).abs().max())))
            if not equal:
                raise AssertionError(checks[-1])
    result = dict(scope="normal-device interpolation arithmetic; not a standalone offload test",
        upstream_fix="https://github.com/vllm-project/llm-compressor/pull/1958",
        transformers=importlib.metadata.version("transformers"), torch=importlib.metadata.version("torch"),
        original_function_sha256=hashlib.sha256(inspect.getsource(Qwen3VLVisionModel.fast_pos_embed_interpolate).encode()).hexdigest(),
        checks=checks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
