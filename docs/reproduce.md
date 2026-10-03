# Reproduce the recorded experiment

Use the published `examples/diagnostic-v2/samples.jsonl` for the recorded image
bytes. A regenerated dataset with another font is a different experiment.

## CPU report and demo

```bash
python3 -m venv .venv
.venv/bin/pip install '.[test]'
.venv/bin/pytest -q
.venv/bin/precisionlab compare --dataset examples/diagnostic-v2/samples.jsonl \
  --reference experiments/results/2026-10-03/p1-bf16/predictions.jsonl \
  --candidate experiments/results/2026-10-03/p1-awq-reload/predictions.jsonl \
  --output runs/awq-report.html
.venv/bin/python experiments/build_demo.py
```

Open `demo/index.html` in a browser. The builder validates the images, input
fingerprints, manifests and recorded summary before creating the page. It does
not run inference. For a self-contained hosting directory, use
`--output runs/site/index.html --copy-images`; the images are copied beside it.

## New diagnostic data

```bash
.venv/bin/precisionlab generate --output runs/diagnostic \
  --font /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf \
  --calibration 192 --dev 300
```

Supply a locally available font. Its hash and the seed are recorded. Every
source has two answer-changing variants that stay in the same split. The
renderer family is shared across splits, so this does not test new templates.

## GPU environment

Use a separate Linux Python 3.12 environment and a CUDA-compatible driver.

```bash
uv venv --python 3.12 .venv-gpu
uv pip install --python .venv-gpu/bin/python torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-gpu/bin/python -r experiments/requirements-gpu.txt
uv pip install --python .venv-gpu/bin/python --no-deps .
.venv-gpu/bin/python experiments/download_model.py --output models/Qwen3-VL-8B-Instruct
```

Weights retain the author's license. The downloader pins the Hugging Face
revision and verifies LFS SHA256 or Git blob hashes, including when ModelScope
is the transport.

## BF16, calibration, AWQ and independent reload

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 .venv-gpu/bin/python experiments/run_hf.py \
  --model models/Qwen3-VL-8B-Instruct --dataset examples/diagnostic-v2/samples.jsonl \
  --mode bf16 --output runs/bf16
HF_HUB_OFFLINE=1 .venv-gpu/bin/python experiments/run_hf.py \
  --model models/Qwen3-VL-8B-Instruct --dataset examples/diagnostic-v2/samples.jsonl \
  --split calibration --mode costs --output runs/costs
.venv-gpu/bin/precisionlab select --dataset examples/diagnostic-v2/samples.jsonl \
  --costs runs/costs/costs.json --budget 74000 --strategy stratified \
  --output runs/calibration.jsonl
CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 .venv-gpu/bin/python experiments/quantize_awq.py \
  --model models/Qwen3-VL-8B-Instruct --dataset examples/diagnostic-v2/samples.jsonl \
  --selected runs/calibration.jsonl --output models/awq-stratified \
  --model-placement resident-gpu --activation-cache-device cuda:1
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 .venv-gpu/bin/python experiments/run_hf.py \
  --model models/awq-stratified --dataset examples/diagnostic-v2/samples.jsonl \
  --mode reload-hf --output runs/awq-reload
.venv-gpu/bin/precisionlab compare --dataset examples/diagnostic-v2/samples.jsonl \
  --reference runs/bf16/predictions.jsonl --candidate runs/awq-reload/predictions.jsonl \
  --output runs/comparison.html
```

Run on idle GPUs. Output directories are immutable; choose new directories for
new runs. These scripts check file export and quality. HF may execute
unpacked dense weights, so this path does not measure optimized INT4 service
performance.

The recorded container had a 16 GiB CPU limit. With the pinned 0.9 stack, CPU
offload and postprocessing relocation exceeded that limit. The resident-GPU
option bypasses three old dispatch/relocation calls inside a dedicated process
and restores them on exit. It keeps upstream AWQ arithmetic; GPU memory must
hold the weights and calibration activations. The position-embedding fix is
credited in `NOTICE`. See the [failure records](experiments.zh-CN.md) before
choosing this workaround for another environment.
