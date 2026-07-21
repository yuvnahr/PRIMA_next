# Adaptive bounded evidence acquisition

`ReasoningController` is the sole production loop. It starts with the original question, uses the existing retrieval controller on every hop, deduplicates evidence by source ID and normalized passage, verifies structured sufficiency, generates one explicit follow-up entity need, and evaluates one centralized stop policy before continuing.

Modes are `bypass`, `single_pass`, `adaptive`, `deliberative`, and `diagnostic`. `adaptive` is the environment default. Set `PRIMA_REASONING_MODE=single_pass` for immediate rollback to the legacy one-retrieval behavior. `single_pass` executes exactly one retrieval before synthesis or abstention.

The initial verifier and need generator are deterministic, generic safeguards: the verifier checks question-term support and explicit contradictory `X is Y` / `X is not Y` statements; the generator selects one new capitalized entity from accumulated evidence. They do not know benchmark answers, titles, categories, datasets, or supporting facts. This gives controlled behavior without a new LLM prompt or a retry loop. A future structured LLM verifier can replace these components behind the existing controller boundary after it has a validated schema contract.

Stopping is centralized for sufficient evidence, unanswerable/ambiguous/contradictory evidence, hop and retrieval limits, time, context, duplicate queries, and no-new-evidence. Intermediate queries, evidence, hypotheses, and traces remain in-memory only.
