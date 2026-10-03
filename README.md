# VLM Precision Lab

Compare a multimodal model before and after quantization, down to the image,
answer and input tensor that produced each result.

[中文](README.zh-CN.md) · [Live demo](https://chrischen-coder.github.io/vlm-precision-lab/) · [Design](docs/design.zh-CN.md) · [Experiment records](docs/experiments.zh-CN.md)

## Start with an actual case

The image asks which nodes link **L6** connects. The expected answer is `B,D`.
BF16 returned `B,D`; the exported AWQ model returned `D,B`.

![Recorded image and model answers](docs/figures/recorded-case.png)

The AWQ answer fails the requested alphabetical order but identifies the correct
endpoints. Counting it as a recognition failure would send us looking for the
wrong fix. The [demo](https://chrischen-coder.github.io/vlm-precision-lab/)
lets you switch scoring rules and inspect **all 300 recorded cases**, including
cases both models got wrong. It replays saved results; it does not run a model.

## Why this tool exists

A quantized model can lose some correct answers and recover others while its
average score barely changes. For amounts, identifiers and graph relationships,
those individual errors matter. Changes to image processing or decoding can also
look like quantization damage.

Precision Lab joins paired predictions to their original images, checks the
comparison controls, and reports losses and recoveries separately for each task.
It distinguishes exact output compliance from unordered graph endpoint accuracy.
Numbers retain their signs and decimal digits in both metrics.

## Current result

Qwen3-VL-8B-Instruct, decoder AWQ W4A16; vision modules and `lm_head` stay BF16.
The 300 synthetic development cases come from 150 generated sources.

| Measurement | BF16 | AWQ export + HF reload |
| --- | ---: | ---: |
| Weight files, decimal GB | 17.53 | 7.22 |
| Numeric fields, exact match | 100/100 | 100/100 |
| Identifiers, exact match | 100/100 | 100/100 |
| Graph endpoints, unordered | 50/100 | 53/100 |
| Graph answers, strict match | 41/100 | 33/100 |
| Peak Torch allocated memory, GiB | 16.47 | 20.26 |

Weight files shrank **58.8%**. This sample contains **zero content regressions**;
all 11 strict regressions are graph answers that preserve the endpoints.
It therefore gives us no reason to build a precision-rescue algorithm yet.

The HF quality run used more memory and took longer after compression. It is not
an optimized INT4 service benchmark. These generated images also do not establish
quality on real documents. [Raw predictions, manifests and logs](experiments/results/2026-10-03/)
and the [measurement details](docs/experiments.zh-CN.md) are included.

## What works today

- Dataset checks for source groups, duplicate-image leakage and image hashes.
- Calibration selection under measured input-token costs: random, task-stratified
  and weighted feature coverage, with a cap per source.
- Paired task reports with original images, content/format scores and
  source-group bootstrap intervals.
- Input tensor, model revision, processor-setting and decoding checks.
- AWQ export and an independent HF reload on RTX 5090.
- Selection among already measured recipes using file size, task scores and an
  optional p95 limit. No feasible recipe returns no winner.

Layer intervention, task-driven precision search, real-document evaluation and
optimized-engine benchmarks are the next work. AWQ itself is supplied by
[LLM Compressor](https://github.com/vllm-project/llm-compressor).

## Run locally

```bash
python3 -m venv .venv
.venv/bin/pip install '.[test]'
.venv/bin/pytest -q
.venv/bin/precisionlab compare --dataset examples/diagnostic-v2/samples.jsonl \
  --reference experiments/results/2026-10-03/p1-bf16/predictions.jsonl \
  --candidate experiments/results/2026-10-03/p1-awq-reload/predictions.jsonl \
  --output runs/awq-report.html
```

The report requires no GPU. To view the demo from a clone, open `demo/index.html`
in a browser; its images use the adjacent recorded dataset. Rebuild it with
`.venv/bin/python experiments/build_demo.py`.

[GPU setup, dataset generation and reproduction commands](docs/reproduce.md)
are documented separately. Model weights and fonts are downloaded separately.

## The research question

If representative data exposes stable content regressions, can actual task
errors select retained high-precision layers better than reconstruction error or
budget-matched random retention? Each proposed recipe must be rebuilt from BF16,
exported, reloaded and measured as a complete model.

[AutoRound](https://github.com/intel/auto-round) and
[MBQ](https://github.com/thu-nics/MBQ) already address precision allocation and
multimodal quantization. Our [design](docs/design.zh-CN.md) sets out the comparisons
needed to establish an improvement over existing methods.

Apache-2.0. Model weights and source datasets retain their own licenses.
