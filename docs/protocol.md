# P0 protocol

P0 asks whether a reproducible critical-task regression exists. It does not
demonstrate a new algorithm or publish a held-out test result.

## Data

- Original generated static 768 × 512 images. Fixed seed 741 and recorded font
  hash. 192 calibration examples / 96 sources; 300 development examples / 150
  sources; 100 development examples per task.
- Numerical amount, complete interface identifier, and labelled link endpoints.
- Two answer-changing variants per source, isolated by source across splits.
  Renderer families are shared; no cross-template generalization claim.
- Exact match after NFKC and outer whitespace normalization. Signs, decimal
  precision, hyphens, case and comma order remain significant.
- Report unordered endpoint correctness and alphabetical output format
  separately for graph tasks. Sorting known undirected endpoints is a required
  simple control; order-only losses are not graph-perception regressions.
- No test set inference or tuning in P0. Later test data need independent real
  sources and held-out renderer families, with licenses and original provenance.

## Fixed controls

Model: Qwen/Qwen3-VL-8B-Instruct,
revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`.
Image processor: slow, min_pixels 3136, max_pixels 393216. Same tokenized tensors
must hash identically across candidates. Greedy generation, seed 42, 24 new
tokens, batch 1, SDPA, BF16 activation/model execution. Labels are not part of
the prompt. Preserve full raw outputs instead of forgiving postprocessing.

## Stage labels

- `bf16`: original weights.
- `fake-rtn`: symmetric group-128 W4 RTN on language-model Linear weights;
  visual/lm_head untouched; BF16 storage and dense execution. Diagnostic only.
- `awq-*`: upstream AWQ W4A16 with ignored visual/lm_head and optional complete
  retained decoder layers; original BF16 model reloaded before each calibration.
- `reload-hf`: packed checkpoint loaded by Transformers/compressed-tensors;
  verify same input hashes and outputs. Dense/dequantized execution is not an
  optimized INT4 engine or a guaranteed VRAM improvement.

An AWQ export/reload comparison does not isolate AWQ from RTN: they are different
quantization algorithms. Fake vs packed numerical parity requires the same
algorithm and exact scales, which P0 does not yet record before AWQ export.

## Calibration

Use measured calibration input token costs and a 74,000-token cap. Stratified
selection is the initial strong baseline; coverage and random are comparison
variables. Report actual spent tokens, sample count, task/source distribution,
pixels and calibration time. Match actual costs before algorithm claims.

## Report

Show reference and candidate absolute scores per task, net change, paired losses
and recoveries, source counts, and paired source-cluster bootstrap intervals
(2,000 repetitions, seed 42). Sources under 30 are explicitly flagged. The
bootstrap is descriptive in P0 and cannot alone certify equivalence, especially
for zero observed rare failures. Later noninferiority requires a predeclared
test, sufficient independent sources and treatment of multiple task groups.

Record the checkpoint and script hashes, predictions and image hashes, processor
configuration, decoding, environment versions, GPU and allocation measurements.
HF batch-1 generation times exclude image preparation, hashing and service
queueing; report that scope. Weight-file bytes exclude tokenizer/config files.
Peak PyTorch allocated bytes exclude context/other processes. Do not report
service throughput, power or optimal precision allocation from these numbers.

## Decision

A stable paired critical-task loss supports proceeding to task-aware layer
interventions. A useful diagnostic tool may still exist if no loss occurs, but
algorithmic rescue claims require a real loss and matched-budget improvement.
Later deployment release requires the complete candidate to export, reload and
run in the target engine, with task and resource gates met.
