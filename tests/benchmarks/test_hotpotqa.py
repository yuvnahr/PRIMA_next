from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from benchmarks.common import BenchmarkArtifactStore
from benchmarks.hotpotqa.adapter import HotpotQAAdapter
from benchmarks.hotpotqa.evaluate import (
    HotpotQAEvaluator,
    answer_scores,
    project_supporting_facts,
    supporting_fact_scores,
    validate_predictions,
)
from benchmarks.hotpotqa.experiment import (
    HOTPOT_PROFILES,
    run_hotpotqa_experiment,
    run_paired_hotpotqa_experiment,
)
from benchmarks.hotpotqa.loader import HotpotQADataset
from runtime.contracts import (
    EvidenceReference,
    ExecutionOutcome,
    ExecutionProfile,
    ExecutionStatus,
    PrimaRequest,
    PrimaResponse,
    RuntimeDiagnostics,
)

FIXTURE = [
    {"_id": "ok", "question": "Who won?", "answer": "The Alpha", "type": "bridge", "level": "easy", "supporting_facts": [["Doc A", 1]], "context": [["Doc A", ["Intro.", "Alpha won."]], ["Doc B", ["Distractor."]]]},
    {"_id": "fail", "question": "Fail?", "answer": "no", "supporting_facts": [["Fail Doc", 0]], "context": [["Fail Doc", ["FAIL"]]]},
]


def write_fixture(tmp_path: Path, rows=FIXTURE) -> Path:
    path = tmp_path / "hotpot.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def response(request: PrimaRequest, *, text: str = "", failed: bool = False, evidence=()) -> PrimaResponse:
    return PrimaResponse(
        request_id=request.request_id,
        task_kind=request.task_kind,
        profile=request.profile,
        status=ExecutionStatus.FAILED if failed else ExecutionStatus.COMPLETED,
        outcome=ExecutionOutcome.FAILED if failed else (
            ExecutionOutcome.INGESTED if request.profile is ExecutionProfile.INGESTION_ONLY
            else ExecutionOutcome.ANSWERED
        ),
        output_text=text or None,
        output_data={
            "hop_count": 1 if evidence else 0,
            "stop_reason": "sufficient",
            "generation": {"selected_source_ids": [item.source_id for item in evidence]},
        },
        evidence=tuple(evidence),
        diagnostics=RuntimeDiagnostics(
            route_name=f"{request.task_kind.value}:{request.profile.value}",
            latency_ms=4.0,
            retrieval_count=len(evidence),
            reflection_count=1 if request.profile is ExecutionProfile.PRIMA_FULL else 0,
            model_call_count=0 if request.profile is ExecutionProfile.INGESTION_ONLY else 1,
            model_usage={"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        ),
        errors=("synthetic ingestion failure",) if failed else (),
    )


class FakeRuntime:
    instances: list[FakeRuntime] = []
    client_count = 0
    requests: list[PrimaRequest] = []

    def __init__(self, llm_client=None, **_config):
        if llm_client is None:
            llm_client = object()
            type(self).client_count += 1
        self.llm_client = llm_client
        self.docs = []
        type(self).instances.append(self)

    async def execute(self, request: PrimaRequest) -> PrimaResponse:
        type(self).requests.append(request)
        if request.profile is ExecutionProfile.INGESTION_ONLY:
            self.docs.append((request.input_text, request.metadata["document_metadata"]))
            return response(request, failed=request.input_text == "FAIL")
        evidence = ()
        if request.profile is not ExecutionProfile.MODEL_ONLY:
            metadata = self.docs[-1][1]
            evidence = (EvidenceReference(
                source_id=metadata["document_id"], text=self.docs[-1][0], score=1.0,
                metadata={"hop": 0, "query": request.input_text, "provenance": metadata},
            ),)
        return response(request, text="alpha", evidence=evidence)


class ExplodingRuntime(FakeRuntime):
    async def execute(self, request: PrimaRequest) -> PrimaResponse:
        if request.input_text == "FAIL":
            raise RuntimeError("interrupted")
        return await super().execute(request)


def reset_fake() -> None:
    FakeRuntime.instances, FakeRuntime.requests, FakeRuntime.client_count = [], [], 0


def test_loader_validates_schema_and_duplicate_ids(tmp_path: Path) -> None:
    assert len(HotpotQADataset(write_fixture(tmp_path)).load()) == 2
    test_path = write_fixture(tmp_path, [{"_id": "test", "question": "Q?", "context": [["T", ["S."]]]}])
    assert HotpotQADataset(test_path).conversations()[0].questions[0].answer is None
    with pytest.raises(ValueError, match="duplicate sample ID"):
        HotpotQADataset(write_fixture(tmp_path, [FIXTURE[0], FIXTURE[0]])).load()


def test_context_modes_preserve_provenance_and_label_oracle_diagnostic_only() -> None:
    distractor = HotpotQAAdapter("distractor").adapt([FIXTURE[0]])[0]
    official = HotpotQAAdapter("official_retrieved").adapt([FIXTURE[0]])[0]
    legacy_alias = HotpotQAAdapter("fullwiki").adapt([FIXTURE[0]])[0]
    oracle = HotpotQAAdapter("oracle").adapt([FIXTURE[0]])[0]
    assert len(distractor.turns) == len(official.turns) == len(legacy_alias.turns) == 3
    assert [(t.metadata["source_title"], t.metadata["sentence_id"]) for t in oracle.turns] == [("Doc A", 1)]
    assert official.metadata["context_source"] == "official_retrieved_context"
    assert oracle.metadata["diagnostic"] is True
    assert all("gold" not in json.dumps(turn.metadata).lower() for turn in distractor.turns)


def test_official_metrics_and_canonical_support_projection() -> None:
    assert answer_scores("The Alpha!", "alpha") == (1.0, 1.0, 1.0, 1.0)
    assert answer_scores("yes", "no") == (0.0, 0.0, 0.0, 0.0)
    assert supporting_fact_scores([["A", 0], ["B", 1]], [["A", 0], ["C", 2]]) == (0.0, 0.5, 0.5, 0.5)
    metadata = {
        "output_data": {"generation": {"selected_source_ids": ["m1"]}},
        "evidence": [{"source_id": "m1", "metadata": {"hop": 0, "query": "q", "provenance": {"source_title": "Exact Title", "sentence_id": 2}}}],
    }
    facts, provenance = project_supporting_facts(metadata)
    assert facts == [["Exact Title", 2]] and provenance[0]["source_id"] == "m1"
    selected_metadata = {
        "output_data": {"generation": {"selected_source_ids": ["m1"]}},
        "evidence": [
            *metadata["evidence"],
            {"source_id": "m2", "metadata": {"provenance": {"source_title": "Uncited", "sentence_id": 9}}},
        ],
    }
    assert project_supporting_facts(selected_metadata)[0] == [["Exact Title", 2]]
    validate_predictions({"answer": {"x": "yes"}, "sp": {"x": facts}})
    rows = [{"expected_answer": "alpha", "prediction": "alpha", "supporting_facts": [["A", 0]], "gold_supporting_facts": [["A", 0]]}]
    assert all(HotpotQAEvaluator().evaluate(rows)[key] == 1.0 for key in ("em", "f1", "sp_em", "sp_f1", "joint_em", "joint_f1"))


@pytest.mark.parametrize("profile", HOTPOT_PROFILES)
def test_all_profiles_use_typed_execute_for_ingestion_and_question(tmp_path: Path, profile: ExecutionProfile) -> None:
    reset_fake()
    result = run_hotpotqa_experiment(
        mode="distractor", runtime_profile=profile, dataset_path=write_fixture(tmp_path),
        output_path=tmp_path / "out", max_samples=1, sampling="sequential",
        runtime_factory=FakeRuntime, quiet=True,
    )
    assert result["runtime_profile"] == profile.value
    assert [request.profile for request in FakeRuntime.requests] == [
        ExecutionProfile.INGESTION_ONLY,
        ExecutionProfile.INGESTION_ONLY,
        ExecutionProfile.INGESTION_ONLY,
        profile,
    ]
    assert all(isinstance(request, PrimaRequest) for request in FakeRuntime.requests)


def test_resume_is_per_item_and_rejects_configuration_drift(tmp_path: Path) -> None:
    dataset, output = write_fixture(tmp_path), tmp_path / "out"
    with pytest.raises(RuntimeError, match="interrupted"):
        run_hotpotqa_experiment(
            mode="distractor", dataset_path=dataset, output_path=output,
            max_samples=2, sampling="sequential", runtime_factory=ExplodingRuntime, quiet=True,
        )
    store = BenchmarkArtifactStore(output / "distractor" / "prima_full")
    assert [row.case_id for row in store.checkpoints()] == ["ok"]
    resumed = run_hotpotqa_experiment(
        mode="distractor", dataset_path=dataset, output_path=output, max_samples=2,
        sampling="sequential", resume=True, runtime_factory=FakeRuntime, quiet=True,
    )
    assert resumed["processed"] == 2
    assert len(store.checkpoints()) == 2
    with pytest.raises(ValueError, match="benchmark_config|model"):
        run_hotpotqa_experiment(
            mode="distractor", dataset_path=dataset, output_path=output,
            max_samples=2, sampling="sequential", resume=True,
            model="changed", runtime_factory=FakeRuntime, quiet=True,
        )


def test_failure_reconciliation_wrong_answer_is_not_runtime_failure_and_client_is_shared(tmp_path: Path) -> None:
    reset_fake()
    result = run_hotpotqa_experiment(
        mode="distractor", dataset_path=write_fixture(tmp_path), output_path=tmp_path / "out",
        max_samples=2, sampling="sequential", parallel_workers=2,
        runtime_factory=FakeRuntime, quiet=True,
    )
    assert result["processed"] == result["completed"] + result["failed"] == 2
    assert result["failure_total"] == sum(result["failure_categories"].values())
    assert result["failure_categories"]["ANSWER_SCORING_FAILURE"] == 1
    assert result["failure_categories"]["INGESTION_FAILURE"] == 1
    assert "RUNTIME_FAILURE" not in result["failure_categories"]
    assert FakeRuntime.client_count == 1


def test_paired_modes_have_same_cases_model_and_only_declared_profile_routes(tmp_path: Path) -> None:
    reset_fake()
    results = run_paired_hotpotqa_experiment(
        mode="distractor", dataset_path=write_fixture(tmp_path), output_path=tmp_path / "paired",
        max_samples=1, sampling="sequential", runtime_factory=FakeRuntime, quiet=True,
    )
    manifests = [BenchmarkArtifactStore(Path(item["artifacts"]["manifest"]).parent).read_manifest() for item in results.values()]
    assert {manifest.selected_ids for manifest in manifests} == {("ok",)}
    assert len({(manifest.provider, manifest.model, manifest.generation_config["seed"]) for manifest in manifests}) == 1
    assert {manifest.runtime_profile for manifest in manifests} == {profile.value for profile in HOTPOT_PROFILES}
    normalized = [manifest.model_dump(
        exclude={"campaign_id", "manifest_fingerprint", "runtime_profile", "active_capabilities", "created_at", "updated_at"}
    ) for manifest in manifests]
    assert normalized[1:] == normalized[:-1]
    assert FakeRuntime.client_count == 1


def test_oracle_manifest_forbids_headline_claims(tmp_path: Path) -> None:
    result = run_hotpotqa_experiment(
        mode="oracle", dataset_path=write_fixture(tmp_path), output_path=tmp_path / "oracle",
        max_samples=1, sampling="sequential", runtime_factory=FakeRuntime, quiet=True,
    )
    manifest = BenchmarkArtifactStore(Path(result["artifacts"]["manifest"]).parent).read_manifest()
    assert result["headline_eligible"] is False
    assert manifest.benchmark_config["oracle_diagnostic_only"] is True
    assert manifest.active_capabilities["headline_eligible"] is False


def test_hotpot_has_no_compatibility_adapter_or_gold_in_runtime_requests() -> None:
    root = Path(__file__).parents[2]
    experiment = (root / "benchmarks" / "hotpotqa" / "experiment.py").read_text(encoding="utf-8")
    assert "PrimaRuntimeAdapter" not in experiment and ".answer_question(" not in experiment
    assert "checkpoint_every" not in experiment
    production = "\n".join(path.read_text(encoding="utf-8") for folder in ("runtime", "reasoning", "memory") for path in (root / folder).glob("*.py"))
    assert "benchmarks.hotpotqa" not in production


def test_opt_in_ollama_smoke_is_configured() -> None:
    assert os.getenv("PRIMA_LLM_MODEL", "qwen3.5:4b")
