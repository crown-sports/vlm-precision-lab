# VLM Precision Lab

**Find which exact-reading tasks regress after multimodal quantization, then verify a deployable precision-retention recipe.**

[中文详细设计](docs/design.zh-CN.md) · [实测结果](docs/experiments.zh-CN.md) · [Experiment protocol](docs/protocol.md) · [GPU scripts](experiments/)

Reading `-723.15` as `723.15`, truncating an interface id, or connecting the wrong
nodes is a task failure even when the average benchmark score looks unchanged.
This project compares the same image inputs, exposes losses and recoveries by
task, and keeps calibration selection separate from method development and test
data.

Status: **P0 research prototype**. The original diagnostic dataset has 192
calibration cases and 300 development cases. It is a synthetic pipeline check,
not evidence of real-world generalization. The task-driven layer search and
optimized engine deployment remain research work; no algorithm superiority,
memory reduction, speedup or hiring outcome is claimed by the initial code.

The first 300-case BF16 / RTN check found **no content regression**: numeric and
identifier reading were both 100%; unordered graph endpoints were 50% → 52%.
Strict graph EM fell 41% → 32%, largely reflecting output order. This is not
evidence for a perception-rescue algorithm. Full records and the calibrated AWQ
follow-up are documented in the [experiment log](docs/experiments.zh-CN.md).

![Measured content and format scores on the synthetic development set](docs/figures/p0-metrics.png)

## Implemented

- Strict local dataset checks: source-group isolation, identical-image leakage,
  safe image paths and image-content fingerprints.
- Measured token-budget calibration selection: random, task-stratified and
  weighted feature coverage, with per-source caps.
- Paired exact-match regressions by task; gains and losses separately; source
  cluster bootstrap intervals; graph content/format separation; escaped offline HTML.
- Input tensor and decoding control checks. Changed inputs require an explicit
  descriptive processor-control comparison.
- Selection among fully measured recipes under file, task and optional p95
  constraints; no feasible recipe returns no winner.
- Qwen3-VL-8B BF16 / symmetric RTN fake-quant diagnostics, upstream AWQ W4A16
  export and a separate HF reload check.

AWQ quantization, packing and model execution are upstream functionality.
The feature selector is a transparent heuristic, and the recipe selector only
ranks measured candidates. Neither is advertised as a new optimal quantizer.

## Install and create a diagnostic dataset

```bash
python3 -m venv .venv
.venv/bin/pip install '.[test]'
.venv/bin/pytest -q
.venv/bin/precisionlab generate --output runs/diagnostic \
  --font /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf \
  --calibration 192 --dev 300
```

Supply a font available on your system. The dataset records its font hash and
seed. Different fonts generate different image bytes; use the published input
bundle for exact reproduction of a recorded experiment. Fonts are not bundled.

The exact [recorded input bundle](examples/diagnostic-v2/) and [raw predictions](experiments/results/2026-10-03/)
are included. To reproduce the existing report without a GPU:

```bash
precisionlab compare --dataset examples/diagnostic-v2/samples.jsonl \
  --reference experiments/results/2026-10-03/p0-bf16/predictions.jsonl \
  --candidate experiments/results/2026-10-03/p0-fake-rtn/predictions.jsonl \
  --output runs/reproduced-report.html
```

Every source has two answer-changing variants. Both stay in one split. The
images contain no answer in their filenames. The first generated families use
the same renderer across splits and therefore do not test unseen templates.

## GPU workflow

Use a separate Linux Python 3.12 environment. Install CUDA 12.8 PyTorch before
the remaining pinned requirements; the driver must support the GPU.

```bash
uv venv --python 3.12 .venv-gpu
uv pip install --python .venv-gpu/bin/python torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-gpu/bin/python -r experiments/requirements-gpu.txt
uv pip install --python .venv-gpu/bin/python --no-deps .
python3 experiments/download_model.py --output models/Qwen3-VL-8B-Instruct
```

Weights use the model author's license and are downloaded separately. The
manifest pins the Hugging Face revision and checks the original LFS SHA256 or
Git blob hashes, including when ModelScope is used as transport.

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 .venv-gpu/bin/python experiments/run_hf.py \
  --model models/Qwen3-VL-8B-Instruct --dataset runs/diagnostic/samples.jsonl \
  --mode bf16 --output runs/bf16
HF_HUB_OFFLINE=1 .venv-gpu/bin/python experiments/run_hf.py \
  --model models/Qwen3-VL-8B-Instruct --dataset runs/diagnostic/samples.jsonl \
  --split calibration --mode costs --output runs/costs
.venv-gpu/bin/precisionlab select --dataset runs/diagnostic/samples.jsonl \
  --costs runs/costs/costs.json --budget 74000 --strategy stratified \
  --output runs/calibration.jsonl
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 .venv-gpu/bin/python experiments/quantize_awq.py \
  --model models/Qwen3-VL-8B-Instruct --dataset runs/diagnostic/samples.jsonl \
  --selected runs/calibration.jsonl --output models/awq-stratified \
  --model-placement resident-gpu
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 .venv-gpu/bin/python experiments/run_hf.py \
  --model models/awq-stratified --dataset runs/diagnostic/samples.jsonl \
  --mode reload-hf --output runs/awq-reload
.venv-gpu/bin/precisionlab compare --dataset runs/diagnostic/samples.jsonl \
  --reference runs/bf16/predictions.jsonl --candidate runs/awq-reload/predictions.jsonl \
  --output runs/comparison.html
```

Use idle GPUs. Output directories are immutable: choose a new directory for a
new run. HF reload can execute dequantized dense weights; this is a file-format
and quality check, not evidence of INT4 inference acceleration. Memory, latency,
power and service throughput require separate controlled backend measurements.

The pinned 0.9 stack normally offloads BF16 weights to CPU. A container CPU
limit of 16 GiB killed the recorded offload attempt, despite idle GPUs. The
script's resident-GPU option skips only that old dispatch step in its dedicated
process. For a second activation-cache GPU, use `CUDA_VISIBLE_DEVICES=0,1`
and `--activation-cache-device cuda:1`. This changes calibration placement, not
AWQ arithmetic; sufficient GPU memory is required. The Qwen position-embedding
compatibility fix is credited to upstream in `NOTICE`.

## Research contribution to test

Can task-specific counterfactuals and measured regressions select precision
retention more effectively than reconstruction error or random retention, under
the **same actual resource budget and supported backend constraints**?

The intended sequence is:

1. Verify a stable critical-task regression against BF16.
2. Compare calibration strategies at matched costs.
3. Nominate retained layers using measured interventions on development data.
4. Rebuild AWQ candidates from BF16, export and reload the complete model.
5. Compare feasible candidates, then freeze and evaluate independent test data.

Single-layer scores are not assumed additive. Restoring unscaled BF16 weights
into AWQ-transformed layers is invalid; retained-layer AWQ recipes are rebuilt
from the original model. If the hypothesis fails, keep the simpler baseline and
publish the negative result.

Related work already includes [AutoRound AutoScheme](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md)
and [MBQ](https://github.com/thu-nics/MBQ). The project does not claim to invent
mixed precision or modality sensitivity. See the Chinese design for contribution
boundaries, strong baselines, release gates and the planned second-model study.

## Limitations

Initial exact-match tasks use Latin letters and numerical fields in generated
static images. Chinese documents, public real data, multiple images and video
are not yet evaluated. CPU tests verify evidence handling, not model quality.
Three hundred development cases cannot establish a 1–2 pp noninferiority claim.
There is no custom CUDA kernel, pruning, distillation or NPU deployment result
in this initial version.

Apache-2.0. Model weights and source datasets retain their own licenses.
