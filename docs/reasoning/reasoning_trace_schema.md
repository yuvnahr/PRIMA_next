# Reasoning trace schema

`AnswerResult.trace_summary` contains compact `ReasoningTraceEvent` records. Normal mode returns only the final stop event; `diagnostic` returns the request-local lifecycle.

Required event names: `ReasoningStarted`, `TaskRouted`, `RetrievalStarted`, `EvidenceRetrieved`, `EvidenceIntegrated`, `SufficiencyEvaluated`, `InformationNeedCreated`, `ReasoningContinued`, `ReasoningStopped`, `AnswerSynthesized`, and `ReasoningFailed`.

Allowed fields are mode, route, hop, query, source IDs, counts, retrieval confidence, verifier status/reason code, safe need ID, stop reason, budgets, and error category. There is no `chain_of_thought`, `thought`, prompt body, document body, key, or secret field. Trace events are not persisted by the controller.
