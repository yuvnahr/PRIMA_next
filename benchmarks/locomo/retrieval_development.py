"""Frozen, CPU retrieval-only comparisons using canonical replay and retrieval producers."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from benchmarks.common import atomic_write_json
from benchmarks.common.artifacts import append_jsonl
from benchmarks.locomo.experiment import IngestionPolicy, _run_conversation, fingerprint, representation_identity
from benchmarks.locomo.loader import LoCoMoDataset
from config.runtime_config import RuntimeConfig
from llm.generation_config import GenerationConfig
from memory.embedding_backend import EmbeddingBackendConfig, model_snapshot
from memory.embedding_pipeline import use_embedding_pipeline
from memory.maintenance.background_supervisor import MaintenanceMode
from memory.retrieval.retrieval_request import RetrievalRequest
from reasoning.evidence_audit import retrieval_hop
from runtime.contracts import ExecutionOptions, ExecutionProfile, PrimaRequest, TaskKind
from runtime.prima_runtime import PrimaRuntime


def development_manifest(dataset_path: Path, *, seed: int = 20261020, questions_per_category: int = 4) -> dict[str, Any]:
    conversations = LoCoMoDataset(dataset_path).conversations()
    if len(conversations) < 2 or questions_per_category < 1:
        raise ValueError("Development requires at least two conversations and a positive question bound")
    ids = sorted(conversation.id for conversation in conversations)
    random.Random(seed).shuffle(ids)  # noqa: S311  # nosec B311
    development = ids[:max(1, len(ids) * 3 // 5)]
    selected = []
    for conversation in conversations:
        if conversation.id not in development:
            continue
        categories: dict[str, list[str]] = defaultdict(list)
        for question in conversation.questions:
            categories[str(question.category)].append(f"{conversation.id}:{question.question_id}")
        for category in sorted(categories):
            cases = sorted(categories[category])
            random.Random(f"{seed}:{conversation.id}:{category}").shuffle(cases)  # noqa: S311  # nosec B311
            selected.extend(cases[:questions_per_category])
    return {"version": "locomo-retrieval-development-v1", "dataset_sha256": fingerprint(dataset_path), "seed": seed,
            "development_conversation_ids": sorted(development), "validation_conversation_ids": sorted(set(ids) - set(development)),
            "selected_question_ids": sorted(selected), "questions_per_category_per_conversation": questions_per_category,
            "complete_histories": True, "prior_exposure": "All ten LoCoMo conversations were previously exposed; validation is held out only from this tuning.",
            "input_representation": representation_identity(conversations, False)}


class _NoGeneration:
    provider_name = "test"

    def chat(self, **kwargs: Any) -> Any:
        raise AssertionError("Retrieval-only development must never call a generation provider")


async def _replay(runtime: PrimaRuntime, conversation: Any, policy: IngestionPolicy, captions: bool, target: Path) -> dict[str, Any]:
    generation = GenerationConfig(model="unused-retrieval-only", provider="test")
    logs: list[dict[str, Any]] = []
    await _run_conversation(replace(conversation, questions=()), runtime, ExecutionProfile.PRIMA_FULL,
        policy, generation, 5, 20, MaintenanceMode.DISABLED, set(), lambda record: None,
        include_captions=captions, admission_checkpoint=logs.append)
    audit = logs[0]
    if not audit["complete"]:
        raise RuntimeError(f"Replay failed for {conversation.id}")
    atomic_write_json(target, audit)
    return audit


def run_development(dataset_path: Path, manifest: dict[str, Any], candidates: dict[str, Any], output: Path, *,
                    ingestion_policy: IngestionPolicy = IngestionPolicy.CONTROLLED_DOCUMENT,
                    include_captions: bool = False, gate_diagnostic: bool = False,
                    _replay_cache: dict[tuple[Any, ...], Any] | None = None) -> dict[str, Any]:
    expected = development_manifest(dataset_path, seed=manifest["seed"], questions_per_category=manifest["questions_per_category_per_conversation"])
    if manifest != expected:
        raise ValueError("Development manifest identity/split/selected IDs do not match the frozen dataset")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "traces.jsonl").exists():
        raise ValueError("Use a fresh development output namespace")
    conversations = [conversation for conversation in LoCoMoDataset(dataset_path).conversations()
                     if conversation.id in manifest["development_conversation_ids"]]
    selected = set(manifest["selected_question_ids"])
    replay_cache = _replay_cache if _replay_cache is not None else {}
    ledger: dict[str, Any] = {}
    for name, candidate in candidates.items():
        config = RuntimeConfig.model_validate(candidate)
        if config.embedding_device != "cpu" or config.reranker_device != "cpu" or config.retrieval_candidate_pool_size != 30:
            raise ValueError("This development ladder requires CPU auxiliaries and the unchanged 30-candidate pool")
        try:
            if config.embedding_backend != "stable":
                model_snapshot(config.embedding_model or "sentence-transformers/all-MiniLM-L6-v2", config.embedding_revision, config.embedding_local_snapshot)
            if config.reranker_backend == "cross_encoder":
                model_snapshot(config.reranker_model, config.reranker_revision, config.reranker_local_snapshot)
        except OSError as exc:
            ledger[name] = {"status": "externally_pending", "reason": str(exc), "runtime_config": candidate}
            continue
        started = time.perf_counter()
        rows = []
        replay_seconds = 0.0
        effective_replay_seconds = 0.0
        for conversation in conversations:
            backend = EmbeddingBackendConfig.from_runtime_config(config)
            admission_settings = tuple((key, value) for key, value in config.model_dump().items()
                                       if key.startswith(("affect_", "admission_")))
            cache_key = (conversation.id, backend, config.representation_mode, config.identity_normalization,
                         ingestion_policy.value, include_captions, admission_settings)
            cached = replay_cache.get(cache_key)
            cache_hit = cached is not None
            runtime = PrimaRuntime(mode="benchmark", memory_backend="in_memory", memory_repository=cached[0] if cached else None,
                                   runtime_config=config, llm_client=_NoGeneration(), maintenance_enabled=False, log_path=output / "runtime.log")
            if cached is None:
                replay_started = time.perf_counter()
                admission_path = output / "admission" / (hashlib.sha256(repr(cache_key).encode()).hexdigest()[:20] + ".json")
                admission = asyncio.run(_replay(runtime, conversation, ingestion_policy, include_captions, admission_path))
                duration = time.perf_counter() - replay_started
                admitted = {turn["turn_id"] for turn in admission["turns"] if (turn["admission"] or {}).get("stored")}
                cached = (runtime.memory_repository, duration, len(conversation.turns), deepcopy(runtime.embedding_pipeline),
                          len(runtime.memory_repository.list()), str(admission_path), admitted)
                replay_cache[cache_key] = cached
                replay_seconds += duration
            effective_replay_seconds += cached[1]
            runtime.embedding_pipeline = deepcopy(cached[3])
            with use_embedding_pipeline(runtime.embedding_pipeline, runtime.embedding_backend_config):
                for question in conversation.questions:
                    case_id = f"{conversation.id}:{question.question_id}"
                    if case_id not in selected:
                        continue
                    before = time.perf_counter()
                    reflection_audit = None
                    if gate_diagnostic:
                        result = asyncio.run(runtime.execute(PrimaRequest(task_kind=TaskKind.FACTUAL_QA, profile=ExecutionProfile.PRIMA_FULL,
                            input_text=question.question, session_id=conversation.id, options=ExecutionOptions(max_llm_calls=0),
                            metadata={"historical_context": True})))
                        hops = result.output_data["evidence_audit"]["hops"]
                        audit = dict(hops[-1]) if hops else {"stage_source_ids": dict.fromkeys(("dense", "sparse", "fused", "reranked", "returned")), "timings_ms": {}}
                        audit["elapsed_ms"] = (time.perf_counter() - before) * 1000
                        reflection_audit = result.output_data["reflection_audit"]
                        if result.diagnostics.model_call_count:
                            raise RuntimeError("No-model gate diagnostics invoked a provider")
                    else:
                        response = runtime.retrieval_controller.retrieve(RetrievalRequest(question.question, top_k=5, profile="simple_rag",
                                                                                          session_id=conversation.id, historical=True))
                        audit = retrieval_hop(response, conversation_id=conversation.id, query=question.question, hop=0,
                                              elapsed_ms=(time.perf_counter() - before) * 1000)
                    gold = {f"{conversation.id}:{turn}" for turn in question.evidence}
                    stage_recalls = {stage: len(gold & set(ids or ())) / len(gold) if gold and ids is not None else None
                                     for stage, ids in audit["stage_source_ids"].items()}
                    row = {"candidate": name, "case_id": case_id, "category": str(question.category),
                           "stage_recall": stage_recalls, "audit": audit, "replay_cache_hit": cache_hit,
                           "replayed_turns": cached[2]}
                    row.update(admission_recall=len(set(question.evidence) & cached[6]) / len(question.evidence) if question.evidence else None,
                               admission_log=cached[5], memory_count=cached[4], reflection_audit=reflection_audit)
                    rows.append(row)
                    append_jsonl(output / "traces.jsonl", row)
            if len(runtime.memory_repository.list()) != cached[4]:
                raise RuntimeError("Retrieval-only QA changed the replayed memory population")
        if len(rows) != len(selected) or {row["case_id"] for row in rows} != selected:
            raise RuntimeError("Development selected and observed question IDs do not reconcile")
        answerable = [row for row in rows if row["category"] != "5" and row["stage_recall"]["returned"] is not None]
        stage_means = {stage: sum(row["stage_recall"][stage] for row in answerable) / len(answerable) if answerable else None
                       for stage in ("dense", "sparse", "fused", "reranked", "returned")}
        ledger[name] = {"status": "measured_retrieval_only_unpromoted", "runtime_config": candidate,
                        "embedding_config": asdict(EmbeddingBackendConfig.from_runtime_config(config)),
                        "question_count": len(rows), "annotated_answerable_count": len(answerable),
                        "stage_recall": stage_means, "actual_replay_seconds": replay_seconds,
                        "effective_replay_seconds": effective_replay_seconds, "wall_seconds": time.perf_counter() - started,
                        "mean_query_ms": sum(row["audit"]["elapsed_ms"] for row in rows) / len(rows),
                        "mean_stage_ms": {stage: sum(row["audit"]["timings_ms"].get(stage, 0) for row in rows) / len(rows)
                                          for stage in ("embedding", "dense", "sparse", "fusion", "reranking")},
                        "actual_runtime_manifest": runtime.runtime_manifest()}
        ledger[name]["memory_count_by_conversation"] = {conversation.id: replay_cache[(conversation.id,
            EmbeddingBackendConfig.from_runtime_config(config), config.representation_mode, config.identity_normalization,
            ingestion_policy.value, include_captions, admission_settings)][4] for conversation in conversations}
        atomic_write_json(output / "development_report.json", {"manifest": manifest, "ledger": ledger, "production_promotion": False})
    report = {"manifest": manifest, "ledger": ledger, "production_promotion": False,
              "ingestion_policy": ingestion_policy.value, "normal_admission_headline_measured": False,
              "include_captions": include_captions, "gate_diagnostic_without_model": gate_diagnostic,
              "input_representation": representation_identity(list(LoCoMoDataset(dataset_path).conversations()), include_captions),
              "trace_sha256": fingerprint(output / "traces.jsonl") if (output / "traces.jsonl").exists() else None}
    atomic_write_json(output / "development_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.candidates is None:
        if args.manifest.exists():
            raise ValueError("Refusing to overwrite a frozen development manifest")
        atomic_write_json(args.manifest, development_manifest(args.dataset))
    else:
        if args.output is None:
            raise ValueError("--output is required for measured comparisons")
        report = run_development(args.dataset, json.loads(args.manifest.read_text()), json.loads(args.candidates.read_text()), args.output)
        print(json.dumps({name: {key: value for key, value in row.items() if key not in {"runtime_config", "embedding_config"}}
                          for name, row in report["ledger"].items()}, indent=2))


if __name__ == "__main__":
    main()
