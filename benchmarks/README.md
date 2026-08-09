# Benchmarks

This package contains dataset-specific benchmark integrations for PRIMA and other agent runtimes.
It is intentionally separate from the existing `evaluation/` package.

`benchmarks/` owns public benchmark adapters, dataset loading, benchmark runner contracts, and benchmark-local outputs.
`evaluation/` remains the generic evaluation framework for reports, plots, metrics, and scientific validation.

## Architecture

Each benchmark should follow the same flow:

```text
Raw benchmark data
-> Loader
-> Adapter
-> Conversation
-> Agent runtime
-> Benchmark outputs
-> Evaluator
```

The benchmark layer never reaches into PRIMA's memory internals. It does not create memory notes,
call retrievers, touch embedding indexes, or depend on PRIMA-specific storage. Instead, every benchmark
is adapted into common conversation objects and passed to an agent through a generic interface.

## Plugin Philosophy

Benchmarks plug into the system by implementing shared interfaces from `benchmarks.common`.
Future datasets such as LongMemEval, HotpotQA, GoEmotions, EmotionLines, and DailyDialog should add
their own loaders and adapters while reusing the same conversation abstraction and runner contract.

## Why Adapters Exist

Public benchmarks use different schemas, naming conventions, task types, and metadata. Adapters isolate
those details from the runtime by converting raw dataset records into benchmark-neutral conversation
objects. This lets agent implementations evolve independently from dataset formats.
