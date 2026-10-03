# P0 evidence bundle

Synthetic development data only; no held-out test was evaluated. The first BF16 / fake-RTN comparison does not support a content-rescue algorithm. See the Chinese experiment log and protocol for scope.

- `p0-bf16`, `p0-fake-rtn`: paired predictions, input tensor hashes and environment manifests.
- `p0-run_hf.py`: exact runner snapshot used for those two results; its SHA256 matches both manifests. Later runners release GPU inputs per batch, so their memory peaks are not directly comparable.
- `calibration-stratified.*`, `p0-costs`: measured-cost calibration selection.
- `awq-attempt3.exit.json`, `awq-attempt4.exit.json`, `awq-attempt5.exit.json`: failed attempts, not completed exports.
- `awq-stratified-attempt*.log.gz`: raw logs compressed without changing bytes.
- `awq-attempt4-source.py.gz`, `awq-attempt5-source.py.gz`: exact source bytes from failed attempts, before subsequent memory workarounds.
- `p0-rtn-report.*`: escaped report, including images and content/format distinctions.
- `p1-bf16`, `p1-awq-reload`, `p1-awq-report.*`: revised-runner BF16 control and independently loaded AWQ comparison. These directory names identify a run revision, not completion of the real-data research milestone.
- `awq-attempt6.exit.json`, `compression-manifest.json`: successful upstream AWQ export; the export manifest's `reload_verified=false` describes the state at export time. Independent reload evidence is in the subsequent run and comparison; the original export record is not rewritten.
- `awq-export-file-hashes.json`, `offload-compat-parity.json`: checkpoint SHA256s and the normal-device interpolation parity check.

The compressed logs may contain progress-bar control characters and diagnostic stack dumps. A timed faulthandler stack dump is not itself an exception. Completion is determined by the supervised exit record and complete manifests, not by a progress bar.

Weights are downloaded separately under the model author's license. No private API endpoint, credential or user document is included.
