"""Canonical, paired, question-resumable LoCoMo execution."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import platform
import random
import shutil
import subprocess  # nosec B404
import threading
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from benchmarks.common import (
    BenchmarkArtifactStore,
    BenchmarkManifest,
    BenchmarkMode,
    BenchmarkSpec,
    BenchmarkSummary,
    CheckpointRecord,
    FailureCategory,
    FailureRecord,
    ItemTiming,
    LifecycleStage,
    PredictionRecord,
    RunStatus,
    TokenUsage,
    atomic_write_json,
    resume_manifest,
)
from benchmarks.locomo.config import (
    MAX_CONVERSATIONS,
    MAX_QUESTIONS,
    MODEL,
    OUTPUT_PATH,
    PARALLEL_WORKERS,
    PROVIDER,
    SEED,
    TOP_K,
)
from benchmarks.locomo.evaluate import (
    LoCoMoEvaluator,
    answer_token_coverage,
    evidence_recall,
    normalized_exact_match,
)
from benchmarks.locomo.loader import LoCoMoDataset
from config.runtime_mode import RuntimeMode
from llm.generation_config import GenerationConfig
from memory.maintenance.background_supervisor import MaintenanceBarrier, MaintenanceMode
from runtime.contracts import (
    DiagnosticMode,
    ExecutionOptions,
    ExecutionProfile,
    ExecutionStatus,
    PrimaRequest,
    PrimaResponse,
    TaskKind,
)
from runtime.prima_runtime import PrimaRuntime
from runtime.route_profiles import select_route

LOCOMO_PROFILES = (
    ExecutionProfile.MODEL_ONLY, ExecutionProfile.SIMPLE_RAG, ExecutionProfile.PRIMA_FULL,
)


class IngestionPolicy(str, Enum):
    """Explicit treatment of conversation turns before questions."""

    CONTROLLED_DOCUMENT = "controlled_document_ingestion"
    SIMPLE_VECTOR = "simple_vector_memory_baseline"
    NORMAL_PRIMA = "normal_prima_admission"


DEFAULT_POLICIES = {
    ExecutionProfile.MODEL_ONLY: IngestionPolicy.CONTROLLED_DOCUMENT,
    ExecutionProfile.SIMPLE_RAG: IngestionPolicy.SIMPLE_VECTOR,
    ExecutionProfile.PRIMA_FULL: IngestionPolicy.NORMAL_PRIMA,
}
FAILURE_STAGES = {
    "INGESTION_FAILURE": "ingestion",
    "RUNTIME_FAILURE": "runtime",
    "ANSWER_PARSE_FAILURE": "answer",
    "RETRIEVAL_MISS": "retrieval",
    "EVIDENCE_SELECTION_FAILURE": "retrieval",
    "ANSWER_SCORING_FAILURE": "scoring",
}


class SharedRuntimeFactory:
    """Create isolated runtimes while reusing one provider client."""

    def __init__(self, factory: Callable[..., Any]) -> None:
        self.factory, self.client, self.lock = factory, None, threading.Lock()

    def create(self, **kwargs: Any) -> Any:
        with self.lock:
            if self.client is not None:
                kwargs["llm_client"] = self.client
            runtime = self.factory(**kwargs)
            if self.client is None:
                self.client = getattr(runtime, "llm_client", None)
            return runtime


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    try:
        return subprocess.run(  # noqa: S603  # nosec B603
            [git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def question_case_id(conversation_id: str, question_id: str | None) -> str:
    """Return the stable per-conversation LoCoMo question identity."""
    return f"{conversation_id}:{question_id}"


def select_conversations(
    conversations: list[Any], *, seed: int, max_conversations: int,
    max_questions: int, full_dataset: bool,
) -> list[Any]:
    """Select a preview or an explicitly requested full dataset."""
    if full_dataset and (max_conversations > 0 or max_questions > 0):
        raise ValueError("full_dataset cannot be combined with question/conversation limits")
    if not full_dataset and max_conversations == 0:
        raise ValueError("Unlimited LoCoMo execution requires full_dataset=True/--full-dataset")
    selected = list(conversations)
    random.Random(seed).shuffle(selected)  # noqa: S311  # nosec B311
    if not full_dataset:
        selected = selected[:max_conversations]
    if max_questions > 0:
        from dataclasses import replace

        selected = [replace(item, questions=item.questions[:max_questions]) for item in selected]
    return selected


def _bounded_question(conversation: Any, question: Any, turns: int) -> str:
    context = conversation.turns[-turns:] if turns else ()
    transcript = "\n".join(_turn_text(turn) for turn in context)
    return f"Bounded conversation context:\n{transcript}\n\nQuestion: {question.question}"


def _turn_text(turn: Any) -> str:
    return f"[{turn.timestamp}] {turn.speaker}: {turn.text}"


def _turn_request(
    conversation_id: str, turn: Any, policy: IngestionPolicy, generation: GenerationConfig,
) -> PrimaRequest:
    text = _turn_text(turn)
    if policy is IngestionPolicy.NORMAL_PRIMA:
        return PrimaRequest(
            task_kind=TaskKind.CONVERSATION, profile=ExecutionProfile.PRIMA_FULL,
            input_text=text, session_id=conversation_id, generation_config=generation,
            metadata={
                "turn_id": turn.turn_id, "source_session_id": turn.session_id,
                "timestamp": turn.timestamp, "ingestion_policy": policy.value,
            },
        )
    return PrimaRequest(
        task_kind=TaskKind.DOCUMENT_INGESTION, profile=ExecutionProfile.INGESTION_ONLY,
        input_text=text, session_id=conversation_id,
        metadata={"document_metadata": {**dict(turn.metadata), "ingestion_policy": policy.value}},
    )


async def _run_conversation(
    conversation: Any, runtime: Any, profile: ExecutionProfile, policy: IngestionPolicy,
    generation: GenerationConfig, top_k: int, bounded_context_turns: int,
    maintenance: MaintenanceMode, pending_ids: set[str],
    checkpoint: Callable[[dict[str, Any]], None], context_budget: int = 1600,
) -> dict[str, Any]:
    admitted_turn_ids: set[str] = set()
    created_memory_ids: set[str] = set()
    ingestion_failure: PrimaResponse | None = None
    if hasattr(runtime, "start_maintenance"):
        await runtime.start_maintenance()
    try:
        for turn in conversation.turns:
            response = await runtime.execute(_turn_request(conversation.id, turn, policy, generation))
            if response.status is not ExecutionStatus.COMPLETED:
                ingestion_failure = response
                break
            memory_id = response.output_data.get("memory_id")
            created_memory_ids.update(str(item) for item in response.output_data.get("memory_ids_created", ()) if item)
            if memory_id:
                created_memory_ids.add(str(memory_id))
            admission = response.output_data.get("memory_admission", {})
            if memory_id or (isinstance(admission, dict) and admission.get("stored")):
                admitted_turn_ids.add(str(turn.turn_id))
        await _barrier(runtime, maintenance, MaintenanceBarrier.AFTER_CONVERSATION)
        for question in conversation.questions:
            case_id = question_case_id(conversation.id, question.question_id)
            if case_id not in pending_ids:
                continue
            if ingestion_failure is not None:
                record = _question_record(
                    conversation, question, profile, policy, None, admitted_turn_ids,
                    created_memory_ids, ingestion_failure=ingestion_failure,
                )
            else:
                await _barrier(runtime, maintenance, MaintenanceBarrier.BEFORE_QUESTION)
                input_text = (
                    _bounded_question(conversation, question, bounded_context_turns)
                    if profile is ExecutionProfile.MODEL_ONLY else question.question
                )
                started = time.perf_counter()
                response = await runtime.execute(PrimaRequest(
                    task_kind=TaskKind.FACTUAL_QA, profile=profile,
                    input_text=input_text, session_id=conversation.id,
                    generation_config=generation, diagnostic_mode=DiagnosticMode.DIAGNOSTIC,
                    options=ExecutionOptions(
                        top_k=top_k, reasoning_mode="adaptive", max_hops=3,
                        max_retrieval_calls=3, max_context_tokens=context_budget,
                    ),
                ))
                record = _question_record(
                    conversation, question, profile, policy, response, admitted_turn_ids,
                    created_memory_ids, elapsed_ms=(time.perf_counter() - started) * 1000,
                )
            checkpoint(record)
        await _barrier(runtime, maintenance, MaintenanceBarrier.BEFORE_FINALIZATION)
        maintenance_state = (
            runtime.maintenance_supervisor.diagnostics()
            if hasattr(runtime, "maintenance_supervisor") else {}
        )
        return {
            "schema_version": "1.0", "conversation_id": conversation.id,
            "ingestion_policy": policy.value, "memory_growth": len(created_memory_ids),
            "admitted_turn_count": len(admitted_turn_ids), "maintenance": maintenance_state,
            "maintenance_complete": _maintenance_complete(maintenance_state),
        }
    finally:
        if hasattr(runtime, "stop_maintenance"):
            await runtime.stop_maintenance(graceful=True)


async def _barrier(runtime: Any, mode: MaintenanceMode, barrier: MaintenanceBarrier) -> bool:
    method = getattr(runtime, "apply_maintenance_barrier", None)
    return bool(await method(mode, barrier)) if method else False


def _maintenance_complete(state: dict[str, Any]) -> bool:
    return not state.get("enabled", False) or (
        not state.get("queue_size", 0) and not state.get("failure_count", 0)
    )


def _question_record(
    conversation: Any, question: Any, profile: ExecutionProfile,
    policy: IngestionPolicy, response: PrimaResponse | None,
    admitted_turn_ids: set[str], created_memory_ids: set[str],
    *, elapsed_ms: float = 0.0, ingestion_failure: PrimaResponse | None = None,
) -> dict[str, Any]:
    expected_evidence = tuple(str(item) for item in question.evidence)
    response_dict = response.to_dict() if response is not None else {}
    retrieval = response.output_data.get("retrieval", {}) if response is not None else {}
    stages = retrieval.get("stage_source_ids", {}) if isinstance(retrieval, dict) else {}
    candidate_ids = set(stages.get("fused_top30", ()))
    if not candidate_ids:
        candidate_ids = set(stages.get("dense_top30", ())) | set(stages.get("sparse_top30", ()))
    final_ids, final_texts = set(), []
    selected_source_ids = set()
    if response is not None:
        generation = response.output_data.get("generation", {})
        if isinstance(generation, dict):
            selected_source_ids = {str(item) for item in generation.get("selected_source_ids", ())}
        for evidence in response.evidence:
            if evidence.source_id not in selected_source_ids:
                continue
            source_turn_id = evidence.metadata.get("provenance", {}).get("source_turn_id")
            if source_turn_id:
                final_ids.add(str(source_turn_id))
            final_texts.append(evidence.text)
    ingestion_error = None
    if ingestion_failure is not None:
        ingestion_error = "; ".join(ingestion_failure.errors) or ingestion_failure.status.value
    runtime_error = None
    if ingestion_failure is None and (response is None or response.status is not ExecutionStatus.COMPLETED):
        runtime_error = "; ".join(response.errors if response else ()) or "runtime returned no response"
    prediction = response.output_text if response and response.output_text else ""
    record = {
        "schema_version": "1.0",
        "case_id": question_case_id(conversation.id, question.question_id),
        "conversation_id": conversation.id, "question_id": question.question_id,
        "runtime_profile": profile.value, "ingestion_policy": policy.value,
        "question": question.question, "expected_answer": question.answer,
        "category": question.category, "expected_evidence": list(expected_evidence),
        "prediction": prediction, "outcome": response.outcome.value if response and response.outcome else None,
        "candidate_evidence_ids": sorted(candidate_ids), "final_evidence_ids": sorted(final_ids),
        "admitted_evidence_ids": sorted(admitted_turn_ids),
        "candidate_evidence_recall": evidence_recall(expected_evidence, candidate_ids),
        "final_evidence_recall": evidence_recall(expected_evidence, final_ids),
        "memory_admission_recall": evidence_recall(expected_evidence, admitted_turn_ids),
        "answer_token_coverage": answer_token_coverage(question.answer or "", final_texts),
        "memory_growth": len(created_memory_ids),
        "maintenance_complete": _maintenance_complete(
            response.diagnostics.maintenance if response is not None else {}
        ),
        "latency_ms": round(elapsed_ms or (response.diagnostics.latency_ms if response else 0.0), 3),
        "retrieval_count": response.diagnostics.retrieval_count if response else 0,
        "reflection_interventions": response.diagnostics.reflection_count if response else 0,
        "model_call_count": response.diagnostics.model_call_count if response else 0,
        "model_usage": dict(response.diagnostics.model_usage) if response else {},
        "raw_runtime_response": response_dict,
        "ingestion_error": ingestion_error, "runtime_error": runtime_error,
        "execution_failed": bool(ingestion_error or runtime_error), "scored": True,
    }
    record["failure_category"] = _classify_failure(record, profile)
    record["failure_stage"] = FAILURE_STAGES.get(str(record["failure_category"]))
    return record


def _classify_failure(record: dict[str, Any], profile: ExecutionProfile) -> str | None:
    if record["ingestion_error"]:
        return "INGESTION_FAILURE"
    if record["runtime_error"]:
        return "RUNTIME_FAILURE"
    if str(record.get("category")) == "5" and record.get("outcome") == "abstained":
        return None
    if not str(record["prediction"]).strip():
        record["execution_failed"] = True
        return "ANSWER_PARSE_FAILURE"
    if profile is not ExecutionProfile.MODEL_ONLY and record["expected_evidence"]:
        if record["candidate_evidence_recall"] == 0:
            return "RETRIEVAL_MISS"
        if record["final_evidence_recall"] < record["candidate_evidence_recall"]:
            return "EVIDENCE_SELECTION_FAILURE"
    if not normalized_exact_match(str(record["prediction"]), str(record["expected_answer"])):
        return "ANSWER_SCORING_FAILURE"
    return None


def _failure_record(record: dict[str, Any]) -> FailureRecord | None:
    subcode = record.get("failure_category")
    if not subcode:
        return None
    failed = bool(record["execution_failed"])
    return FailureRecord(
        case_id=record["case_id"],
        category=FailureCategory.RUNTIME if failed else FailureCategory.EVALUATION,
        subcode=str(subcode),
        stage=LifecycleStage.EXECUTE_CASE if failed else LifecycleStage.EVALUATE,
        message=record.get("ingestion_error") or record.get("runtime_error") or str(subcode),
        details={"locomo_record": record},
    )


def _prediction_record(record: dict[str, Any]) -> PredictionRecord:
    usage = record.get("model_usage", {})
    now = datetime.now(timezone.utc)
    return PredictionRecord(
        case_id=record["case_id"], mode=BenchmarkMode.CONVERSATION_QA,
        prediction=record["prediction"],
        timing=ItemTiming(started_at=now, finished_at=now, total_ms=record["latency_ms"]),
        tokens=TokenUsage(
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
        ),
        diagnostics={"locomo_record": record},
    )


def _checkpoint(store: BenchmarkArtifactStore, record: dict[str, Any]) -> None:
    failure = _failure_record(record)
    if record["execution_failed"]:
        if failure is None:
            raise ValueError("failed LoCoMo question requires failure details")
        store.append_checkpoint(CheckpointRecord(
            case_id=record["case_id"], status=RunStatus.FAILED, failure=failure,
        ))
        return
    prediction = _prediction_record(record)
    store.append_checkpoint(CheckpointRecord(
        case_id=record["case_id"], status=RunStatus.COMPLETE,
        prediction=prediction, failure=failure,
    ))


def _records(store: BenchmarkArtifactStore) -> list[dict[str, Any]]:
    records = []
    for checkpoint in store.checkpoints():
        if checkpoint.prediction:
            record = checkpoint.prediction.diagnostics.get("locomo_record")
        elif checkpoint.failure:
            record = checkpoint.failure.details.get("locomo_record")
        else:
            record = None
        if isinstance(record, dict):
            records.append(dict(record))
    return records


def _dependencies() -> dict[str, str]:
    versions = {}
    for package in ("pydantic", "nltk", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


def _manifest(
    dataset: Path, profile: ExecutionProfile, policy: IngestionPolicy,
    generation: GenerationConfig, selected_ids: list[str], seed: int,
    config: dict[str, Any],
) -> BenchmarkManifest:
    commit, root = git_commit(), Path(__file__).parents[2]
    route = select_route(TaskKind.FACTUAL_QA, profile)
    return BenchmarkManifest(
        campaign_id=f"locomo-{uuid4()}",
        benchmark=BenchmarkSpec(
            name="LoCoMo", version="1.0", mode=BenchmarkMode.CONVERSATION_QA,
            dataset_name=dataset.name,
            description="Long-conversation QA over canonical PRIMA profiles.",
        ),
        source_fingerprint=None if commit else fingerprint(Path(__file__)), git_commit=commit,
        dataset_hash=fingerprint(dataset), selected_ids=tuple(selected_ids),
        provider=generation.provider, model=generation.model,
        generation_config=generation.to_dict(), benchmark_config=config,
        runtime_profile=profile.value,
        active_capabilities={
            "planned_components": [component.value for component in route.components],
            "ingestion_policy": policy.value,
        },
        repository_mode="benchmark:in_memory",
        prompt_hashes={
            "answer_generation": fingerprint(root / "workflow" / "answer_generation.py"),
            "prompt_builder": fingerprint(root / "llm" / "prompt_builder.py"),
        },
        seed=seed, dependencies=_dependencies(),
        hardware={
            "platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(),
        },
    )


def _write_artifacts(
    root: Path, records: list[dict[str, Any]], metrics: dict[str, Any],
    conversation_diagnostics: list[dict[str, Any]], scope: str,
) -> dict[str, str]:
    raw, metric_dir = root / "raw", root / "metrics"
    atomic_write_json(raw / "questions.json", records)
    atomic_write_json(raw / "conversation_diagnostics.json", conversation_diagnostics)
    atomic_write_json(metric_dir / "metrics.json", metrics)
    atomic_write_json(metric_dir / "locomo_summary.json", {
        "schema_version": "1.0", "dataset_scope": scope, **metrics,
    })
    return {
        "manifest": str(root / "manifest.json"),
        "summary": str(root / "summary.json"),
        "checkpoint": str(root / "checkpoints" / "records.jsonl"),
        "predictions": str(root / "predictions" / "records.jsonl"),
        "failures": str(root / "failures" / "records.jsonl"),
        "metrics": str(metric_dir / "metrics.json"),
        "questions": str(raw / "questions.json"),
    }


def run_locomo_experiment(
    max_conversations: int | None = None, max_questions: int | None = None,
    seed: int = SEED, top_k: int = TOP_K, provider: str = PROVIDER,
    model: str = MODEL, parallel_workers: int = PARALLEL_WORKERS,
    dataset_path: str | None = None, output_path: str | None = None,
    include_rouge_l: bool = False, include_bertscore: bool = False,
    bertscore_device: str = "cpu", bertscore_batch_size: int = 16,
    maintenance_mode: MaintenanceMode | str = MaintenanceMode.DISABLED,
    runtime_profile: ExecutionProfile | str = ExecutionProfile.PRIMA_FULL,
    ingestion_policy: IngestionPolicy | str | None = None,
    bounded_context_turns: int = 20, full_dataset: bool = False,
    context_budget: int = 1600,
    resume: bool = False, runtime_factory: Callable[..., Any] = PrimaRuntime,
    generation_config: GenerationConfig | None = None,
    _runtime_pool: SharedRuntimeFactory | None = None,
) -> dict[str, Any]:
    """Run one canonical LoCoMo profile with exact per-question resume."""
    if parallel_workers < 1 or bounded_context_turns < 1:
        raise ValueError("parallel_workers and bounded_context_turns must be positive")
    profile, maintenance = ExecutionProfile(runtime_profile), MaintenanceMode(maintenance_mode)
    if profile not in LOCOMO_PROFILES:
        raise ValueError("LoCoMo requires model_only, simple_rag, or prima_full")
    policy = IngestionPolicy(ingestion_policy) if ingestion_policy else DEFAULT_POLICIES[profile]
    max_conversations = MAX_CONVERSATIONS if max_conversations is None else max_conversations
    max_questions = MAX_QUESTIONS if max_questions is None else max_questions
    if max_conversations < 0 or max_questions < 0:
        raise ValueError("question/conversation limits cannot be negative")
    dataset = LoCoMoDataset(Path(dataset_path)) if dataset_path else LoCoMoDataset()
    conversations = select_conversations(
        list(dataset.conversations()), seed=seed, max_conversations=max_conversations,
        max_questions=max_questions, full_dataset=full_dataset,
    )
    selected_ids = [
        question_case_id(conversation.id, question.question_id)
        for conversation in conversations for question in conversation.questions
    ]
    output_root = Path(output_path) if output_path else OUTPUT_PATH
    root, store = output_root / profile.value, BenchmarkArtifactStore(output_root / profile.value)
    if not resume and store.layout.manifest.exists():
        raise ValueError(f"Output already contains a campaign; use resume or a new path: {root}")
    generation = generation_config or GenerationConfig(model=model, provider=provider, seed=seed)
    scope = "full_dataset" if full_dataset else "preview"
    config = {
        "dataset_scope": scope, "full_dataset": full_dataset,
        "max_conversations": max_conversations, "max_questions": max_questions,
        "top_k": top_k, "bounded_context_turns": bounded_context_turns,
        "ingestion_policy": policy.value, "maintenance_mode": maintenance.value,
        "rouge_l": include_rouge_l, "bertscore": include_bertscore,
        "bertscore_device": bertscore_device,
        "bertscore_batch_size": bertscore_batch_size,
        "context_budget": context_budget,
    }
    manifest = _manifest(dataset.dataset_path, profile, policy, generation, selected_ids, seed, config)
    if resume:
        manifest = resume_manifest(store.read_manifest(), manifest)
    store.initialize(manifest)
    existing = _records(store) if resume else []
    completed_ids = {record["case_id"] for record in existing}
    pending_ids = set(selected_ids) - completed_ids
    pool = _runtime_pool or SharedRuntimeFactory(runtime_factory)
    conversation_diagnostics = []

    def execute(conversation: Any) -> dict[str, Any]:
        if not any(question_case_id(conversation.id, q.question_id) in pending_ids for q in conversation.questions):
            return {"schema_version": "1.0", "conversation_id": conversation.id, "resumed": True}
        runtime = pool.create(
            mode=RuntimeMode.BENCHMARK, memory_backend="in_memory",
            maintenance_enabled=maintenance is not MaintenanceMode.DISABLED,
            generation_config=generation,
        )
        return asyncio.run(_run_conversation(
            conversation, runtime, profile, policy, generation, top_k,
            bounded_context_turns, maintenance, pending_ids,
            lambda record: _checkpoint(store, record), context_budget,
        ))

    if parallel_workers > 1 and len(conversations) > 1:
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            conversation_diagnostics = list(executor.map(execute, conversations))
    else:
        conversation_diagnostics = [execute(conversation) for conversation in conversations]
    records = _records(store)
    evaluator = LoCoMoEvaluator(
        include_rouge_l=include_rouge_l, include_bertscore=include_bertscore,
        bertscore_device=bertscore_device, bertscore_batch_size=bertscore_batch_size,
    )
    metrics = evaluator.evaluate(records)
    completed = sum(not record["execution_failed"] for record in records)
    failed = len(records) - completed
    failures = Counter(str(record["failure_category"]) for record in records if record.get("failure_category"))
    if len(records) != completed + failed or sum(failures.values()) != metrics["failure_taxonomy"]["total_failures"]:
        raise RuntimeError("LoCoMo category/failure totals do not reconcile")
    artifacts = _write_artifacts(root, records, metrics, conversation_diagnostics, scope)
    store.finalize(BenchmarkSummary(
        campaign_id=manifest.campaign_id, status=RunStatus.COMPLETE,
        total_cases=len(selected_ids), completed_cases=completed,
        failed_cases=failed, cancelled_cases=0, metrics=metrics,
        started_at=manifest.created_at,
    ))
    return {
        "dataset_scope": scope, "full_dataset": full_dataset,
        "runtime_profile": profile.value, "ingestion_policy": policy.value,
        "conversations": len(conversations), "questions": len(records),
        "completed": completed, "failed": failed,
        "metrics": metrics, "artifacts": artifacts,
    }


def run_paired_locomo_experiment(**kwargs: Any) -> dict[str, dict[str, Any]]:
    """Run all canonical LoCoMo profiles over one selected question set."""
    kwargs.pop("runtime_profile", None)
    kwargs.pop("ingestion_policy", None)
    pool = SharedRuntimeFactory(kwargs.get("runtime_factory", PrimaRuntime))
    results = {
        profile.value: run_locomo_experiment(
            **kwargs, runtime_profile=profile,
            ingestion_policy=DEFAULT_POLICIES[profile], _runtime_pool=pool,
        )
        for profile in LOCOMO_PROFILES
    }
    selected = {
        BenchmarkArtifactStore(Path(result["artifacts"]["manifest"]).parent).read_manifest().selected_ids
        for result in results.values()
    }
    if len(selected) != 1:
        raise RuntimeError("Paired LoCoMo profiles selected different questions")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LoCoMo through canonical PRIMA profiles")
    parser.add_argument("--profile", choices=tuple(profile.value for profile in LOCOMO_PROFILES), default="prima_full")
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--ingestion-policy", choices=tuple(policy.value for policy in IngestionPolicy))
    parser.add_argument("--max-conversations", type=int, default=MAX_CONVERSATIONS)
    parser.add_argument("--max-questions", type=int, default=MAX_QUESTIONS)
    parser.add_argument("--full-dataset", action="store_true")
    parser.add_argument("--bounded-context-turns", type=int, default=20)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--provider", default=PROVIDER)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--parallel-workers", type=int, default=PARALLEL_WORKERS)
    parser.add_argument("--dataset-path")
    parser.add_argument("--output-path", default=str(OUTPUT_PATH))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rouge-l", action="store_true")
    parser.add_argument("--bertscore", action="store_true")
    parser.add_argument("--bertscore-device", default="cpu")
    parser.add_argument("--bertscore-batch-size", type=int, default=16)
    parser.add_argument("--maintenance-mode", choices=tuple(mode.value for mode in MaintenanceMode), default="disabled")
    args = parser.parse_args()
    values = vars(args)
    paired, profile = values.pop("paired"), values.pop("profile")
    values["runtime_profile"] = profile
    result = run_paired_locomo_experiment(**values) if paired else run_locomo_experiment(**values)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
