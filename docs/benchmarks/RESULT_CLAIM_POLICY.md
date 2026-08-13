# Benchmark Result-Claim Policy

Only validated, complete benchmark modes may support headline claims. A partial mode
must be labeled partial with its selected IDs and failure counts. Never compare modes
when the campaign refuses the pair: model/revision, selected IDs and order, seed,
generation settings, context budget, and retry policy must match.

Claims must state the benchmark, split, dataset hash, system/profile, context source,
sample count, failure count, and whether the provider is fake or real. Fixture and
fake-provider runs establish wiring, resilience, and accounting only; they do not
measure model quality.

Task-specific boundaries:

- GoEmotions measures affect/classification behavior, not the complete QA wrapper.
  Report parse recovery separately from label corrections, regressions, and trained
  classifier baselines.
- HotpotQA must identify supplied distractor, official-retrieved, or oracle context.
  Oracle-context results are never headline eligible.
- LoCoMo preview results are not full-dataset results. Core exact match and F1 do not
  depend on ROUGE-L or BERTScore; optional semantic metrics must retain capability
  and configuration metadata.

Gold labels, expected answers, and evidence annotations may exist in evaluator-owned
case/checkpoint artifacts for scoring. They must never enter model prompts, retrieval
queries, runtime request metadata, or provider telemetry. Do not attribute a paired
delta to a wrapper component unless the compared systems isolate that component.
