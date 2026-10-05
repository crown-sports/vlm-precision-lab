# Recorded CORD-v2 deployment study

AWQ W4A16 reduced sampled whole-device workload memory **21.67 → 12.11 GiB (−44.1%)** on one RTX 5090. Median dev throughput across three repeats increased **21.7% at concurrency 1** and **12.5% at concurrency 4**. Both BF16 and AWQ **failed** the predeclared 95% per-field quality targets. AWQ introduced two total-amount errors despite improving aggregate exact match.

![Measured quality and service costs](summary.png)

[Detailed case study in Chinese](../../../../docs/cord-study.zh-CN.md) · [Frozen protocol](../../../cord-service-protocol.json) · [Runner](../../../run_service_study.py)

## Test quality

Literal agreement with CORD parsed-field annotations, not the official CORD parsing score or guaranteed verbatim OCR. Currency spacing in the printed text can differ from the annotation. Paired receipt-cluster bootstrap: 10,000 draws, seed 42. Primary runs are the first concurrency-1 run per variant.

| Field | N | BF16 EM | AWQ EM | Delta pp [95% CI] | Losses / recoveries |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cash | 65 | 78.46% | 84.62% | +6.15 [+1.54, +12.31] | 0 / 4 |
| Change | 56 | 78.57% | 80.36% | +1.79 [0.00, +5.36] | 0 / 1 |
| Tax | 42 | 73.81% | 80.95% | +7.14 [0.00, +16.67] | 0 / 3 |
| Total | 95 | 82.11% | 84.21% | +2.11 [−3.16, +7.37] | 2 / 4 |

Total's lower interval bound is below the predeclared −2 pp margin. Zero observed losses on other small task samples do not rule out rare population errors. All 3792 measured requests succeeded without truncation; service latency targets passed. The quality gates remain **FAIL**.

The two total regressions were manually checked against unmodified source images:

- [Test row 20](review-images/test-20.png): expected/BF16 `377,859`; AWQ `377,059`.
- [Test row 42](review-images/test-42.png): expected/BF16 Total `16,500`; AWQ returned the printed Cash value `50,000`.

An additional **post hoc** diagnostic ignores only whitespace immediately after an exact `Rp` prefix. It preserves signs, currency case and all numeric separators. It does not replace primary scoring or the frozen gates; total remains below 95%, and both content regressions remain. Signed digit equality is only a review clue, never proof of monetary equivalence.

## Controls and evidence

Model revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`; dataset revision `7f0115a4b758a71d6473b8d085751692da2fef98`. Existing synthetic-calibrated AWQ export, 177 calibration questions; no CORD calibration or prompt tuning. vLLM 0.11.0, Torch 2.8.0/CUDA 12.8, Transformers 4.57.3, compressed-tensors 0.11.0. Same base processor/tokenizer files, 2 GiB KV cache, eager execution, prefix and processor caches disabled. The AWQ server log records `MarlinLinearKernel`.

Dev: 100 receipts / 273 questions, concurrency 1 and 4, three repeats each. Test: 100 source receipts / 99 with selected fields / 258 questions, concurrency 1, once each. **3792 measured requests across 14 workloads**, plus 56 unscored warmup requests. Repeated timing requests are not additional independent quality observations.

One successfully measured deployment per variant, sequential BF16 then AWQ. An AWQ startup failed before inference because the 0.11 config reader rejected `scale_dtype: null` and `zp_dtype: null` from the 0.13 exporter. The resumed run removes only those null metadata entries in a derivative config; non-null values are rejected, original weights and export files are unchanged. BF16 measurements were retained and verified. Failed and resumed launches are both archived; cold startup timing is not compared.

| File | Purpose |
| --- | --- |
| `protocol.json`, `study.json`, `study-before-resume.json` | Frozen controls, completion, and the failed-startup record |
| `bf16/`, `awq/` | All 14 raw prediction files, manifests, profiles, launches, server logs and GPU samples |
| `checkpoint-evidence.json`, `awq-controlled-config.json` | Verified original files, processor identity and null-only adapter hashes |
| `measured-runner-initial.py`, `measured-runner-resume.py` | Exact scripts whose hashes were recorded during execution |
| `dev-source.json`, `test-source.json` | Source parquet hashes, import counts and recorded dataset fingerprints |
| `dev-samples.jsonl`, `test-samples.jsonl` | Unchanged labels/prompts/image-hash ledgers; image paths refer to the corresponding imported dataset directory |
| `analysis.json`, `analysis-source.py`, `plot-source.py` | Derived results and the scripts used to generate the published analysis/figure |
| `manual-review.json`, `review-images/` | Five inspected original receipts, observations, hashes and source attribution |
| `serving-environment.freeze.txt` | Observed service dependencies, omitting the local editable project path |
| `evidence-index.json` | SHA256 of all bundled files except the index itself |

No weight files, full image set or source parquet files are bundled. The five unmodified review images and annotation ledgers retain [NAVER CLOVA / CORD attribution and CC BY 4.0](review-images/NOTICE.md). Import manifests are historical observations; their original scoring description predates the currency-spacing diagnosis and is preserved.

## Recheck

`pytest -q` checks the archive's hashes, raw prediction/manifest consistency, request counts, scores and preserved content regressions without a GPU. It cannot verify unbundled image bytes. To recompute the full analysis, download/import the pinned validation and test data as described in the case study, then run:

```bash
python experiments/analyze_service_study.py \
  --study experiments/results/2026-10-05/cord-service \
  --dev runs/cord-dev/samples.jsonl --test runs/cord-test/samples.jsonl \
  --output runs/rechecked-analysis.json
```

This verifies actual imported image hashes and split overlap, request fingerprints, retained profiles, raw memory samples and gates. It does not run inference again. Single-device sequential execution, timing variation, unknown training contamination and request-level rather than GPU-tensor fingerprints limit generalization and causal claims. AWQ and Marlin are upstream algorithms; this repository provides the controlled evaluation and failure analysis.
