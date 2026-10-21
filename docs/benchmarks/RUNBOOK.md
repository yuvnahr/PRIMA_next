# Benchmark Campaign Runbook

## Scope

The campaign command runs GoEmotions, HotpotQA, and LoCoMo through one shared
provider session. It does not imply that every task invokes every PRIMA component.
GoEmotions uses the bounded affect route; ingestion never generates; `model_only`
never retrieves.

Use the repository Python 3.14 environment directly:

```powershell
.venv\Scripts\python.exe --version
.venv\Scripts\python.exe -m benchmarks.campaign.cli run --config benchmarks/campaign/smoke.yaml
.venv\Scripts\python.exe -m benchmarks.campaign.cli run --config benchmarks/campaign/smoke.yaml --resume
```

The smoke configuration uses fixture data and a fake provider. Remove or choose a
new output directory before starting a fresh smoke; use `--resume` only for the same
configuration and dataset hashes.

## Production preflight

The canonical Kaggle notebook is the matched production entry point. After human
review, use integrated `dev` and an explicit agreed commit. Full execution defaults
off, and both real hardware/full evaluation and actual Kaggle Python validation
remain separate gates. `full_gpu.example.yaml` now describes one GoEmotions session;
it is not a standalone resource/time admission or permission to run two full
benchmarks in one allocation.

Production checks clean tracked source, exact uploaded notebook code (literal
CONFIG edits only), data/model/prompt/schema/policy identities, typed runtime
settings, offline auxiliary artifacts, live quota and pilot freshness, full IDs,
resources and export reserve. Dependencies use the base list or minimal optional
CPU retrieval list without whole-environment upgrades. Missing pinned model
snapshots fail; do not download or fall back inside production requests. The
preflight artifact records snapshot hashes; native manifests report actual loaded
embedding identity/device. BERTScore is supplementary and runs separately after
Qwen is released.

Keep one shared model and one worker. Optional repeated client trials must match
native outputs, show a reproducible wall win and pass budget/resource gates.
Requested workers, outstanding requests, observed backend slots and observable
model execution overlap are distinct fields. Unknown/one slot retains one;
explicit LoCoMo CPU overlap may keep inference serialized. Failed restoration
blocks full execution. Sampling reports actual median/max gaps, missed deadlines
and coverage, with NVML boundary snapshots separate from periodic nvidia-smi.

## Interruption and resume

SIGINT, SIGTERM, and Windows console-break cancellation return exit code 130 after
the active mode is marked partial or failed. Re-run the identical command with
`--resume`. Resume retains the campaign ID, verifies the manifest fingerprint, and
suppresses case IDs already present in the append-only checkpoint.

Before accepting results, confirm:

- root and child manifests are complete or explicitly partial;
- checkpoint and prediction case IDs are unique;
- processed/completed/failed/category totals reconcile under each benchmark's schema;
- paired comparisons are `valid`, never `refused`;
- telemetry reports `max_active_requests` at or below the configured bound;
- the artifact tree contains no credentials or developer-machine paths.

Partial campaigns may support diagnostics but are not headline result sets.

## Canonical result bundles and recovery

`benchmarks.common.bundle.create_bundle()` leaves live campaign files unchanged.
ZIP `bundle_manifest.json` names canonical relative paths/hashes and verified
checkpoint-derived prediction/native views. Full prompt prefixes/templates are
stored once with reconstructible inputs and exact prompt hashes. Raw responses,
all checkpoints, pilot repetitions/outcomes/configs and resource/timing traces
remain available. The shared artifact, comparison, aggregation and notebook
readers use these references. Legacy ZIPs remain readable. Plain-path tools should
materialize a verified copy first; never mutate the immutable exported bundle.

```python
from pathlib import Path
from benchmarks.common.bundle import validate_bundle, materialize_bundle
validate_bundle(Path("result.zip"))
materialize_bundle(Path("result.zip"), Path("new-empty-campaign-directory"))
```

Resume still requires the exact original config/output namespace, source, model,
data and selected IDs; materialization does not waive identity checks. It restores
literal files for older readers and preserves export provenance. Packaging checks
CRC, unique safe paths, hashes, JSON/JSONL, schemas/counts, required artifacts and
secret absence. The completed ZIP timing/hash sidecar is separate to avoid a
self-referential archive mutation. Archive validity never proves quality,
resource compliance or an admitted full run.

Reports distinguish label correction from runtime reflection, unavailable
label-only confidence, active versus calendar time, logical versus actual tokens,
QA versus conversation replay, all-hop candidate/accumulated/supplied/cited evidence
and official versus legacy scores. Missing old fields stay unavailable.
