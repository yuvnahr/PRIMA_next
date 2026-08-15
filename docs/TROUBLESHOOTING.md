# Troubleshooting

## Campaign will not start

- **Unresolved placeholder:** supply every environment-specific field referenced by
  the full GPU configuration. This is an intentional preflight stop.
- **Dataset hash mismatch:** use the exact validated split or create a new campaign;
  do not resume against changed data.
- **Endpoint/capability failure:** verify the configured provider endpoint, model
  revision, context window, and structured-output support. There is no hidden model
  or backend fallback.
- **Concurrency rejected:** keep `max_gpu_requests: 1`, or explicitly declare and
  validate the available GPUs/API concurrency.

## Campaign was interrupted

Run the identical config with `--resume`. An incompatible resume must be rejected.
If the manifest says partial, inspect the child checkpoint before retrying; duplicate
case IDs indicate artifact corruption and must block result use.

## Optional metrics are unavailable

LoCoMo core metrics still run. Install the semantic-metric extras only when ROUGE-L
or BERTScore is intentionally enabled; configure BERTScore device and batch size.
Capability errors should remain visible rather than changing the metric silently.

## Runtime item failures

Provider timeout/rate limit, malformed or truncated structured output, retrieval,
memory-store, graph-capability, and maintenance-worker failures must appear as typed
outcomes or failure records. Check runtime diagnostics for requested versus active
backends and `fallback_used`; do not reinterpret an error string as a prediction.

## Telemetry is missing

Host/GPU telemetry is best effort and cannot crash a campaign. Missing `psutil` or a
GPU probe is reported as unavailable. Inference request counts, latency, tokens,
timeouts, retries, and semaphore observations remain campaign-owned.
