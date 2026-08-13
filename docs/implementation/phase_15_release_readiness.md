# Phase 15 — Fault Injection and Release Readiness

## Outcome

Phase 15 closes the release-validation scope without launching a full-dataset GPU
run. Fault injection covers provider and structured-output failures, storage and
maintenance failures, crash-safe artifacts, incompatible/duplicate resume, operating
system cancellation, and the campaign inference bound. A diagnostic integration
test proves the full workflow call trace and the exclusions of shorter routes.

The fake-provider fixture campaign completes GoEmotions, HotpotQA, and LoCoMo under
one provider session. SIGINT and SIGTERM subprocess tests interrupt after a durable
item checkpoint, then resume to exactly one record per selected case.

## Contract changes

- The fake campaign provider accepts an explicit test-only response delay and returns
  schema-valid factual-QA output when a response schema is requested.
- Campaign cancellation records the active mode as partial/failed and returns exit
  code 130 for supported process signals.
- Root manifests store relative child-manifest references; preflight summaries do not
  serialize local dataset/output paths.
- Aggregate comparisons, telemetry, and rendered errors pass through the established
  artifact redaction boundary.

## Evidence

The command transcript, architecture truth matrix, benchmark readiness, risks, and
go/no-go decision are maintained in `RELEASE_READINESS.md`. Operational and claim
guidance is in the benchmark runbook, result-claim policy, and troubleshooting guide.

## Deferrals

No real provider, heavyweight model, full benchmark split, or GPU campaign was run.
Those deployment-specific checks remain intentionally gated by the unresolved full
configuration and must be performed in the target environment.
