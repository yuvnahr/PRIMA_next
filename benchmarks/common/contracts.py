"""Versioned, benchmark-neutral campaign and artifact contracts."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from urllib.parse import parse_qsl, urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION: Literal["1.0"] = "1.0"


def utc_now() -> datetime:
    """Return an aware UTC timestamp."""

    return datetime.now(timezone.utc)


class BenchmarkMode(Enum):
    """Supported benchmark data shapes."""

    CLASSIFICATION = "classification"
    CONVERSATION_QA = "conversation_qa"


class RunStatus(Enum):
    """Persisted campaign and item terminal/progress states."""

    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"
    COMPLETE = "complete"


class LifecycleStage(Enum):
    """Ordered shared benchmark lifecycle."""

    PREFLIGHT = "preflight"
    LOAD = "load"
    SELECT_CASES = "select_cases"
    INITIALIZE_RUNTIME = "initialize_runtime"
    EXECUTE_CASE = "execute_case"
    CHECKPOINT = "checkpoint"
    EVALUATE = "evaluate"
    FINALIZE = "finalize"


class FailureCategory(Enum):
    """Shared failure taxonomy; benchmark-specific detail belongs in ``subcode``."""

    PREFLIGHT = "preflight"
    DATA = "data"
    CONFIGURATION = "configuration"
    RUNTIME = "runtime"
    PROVIDER = "provider"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    ARTIFACT = "artifact"
    EVALUATION = "evaluation"
    UNKNOWN = "unknown"


class ContractModel(BaseModel):
    """Immutable, strict base for JSON benchmark artifacts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    def to_dict(self) -> dict[str, Any]:
        """Serialize using JSON-compatible values."""

        return self.model_dump(mode="json")


class BenchmarkSpec(ContractModel):
    """Identity and data-shape declaration for a benchmark implementation."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    mode: BenchmarkMode
    dataset_name: str = Field(min_length=1)
    description: str = ""


class BenchmarkCase(ContractModel):
    """Shared case identity without imposing task-specific gold fields."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    case_id: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClassificationCase(BenchmarkCase):
    """Independent multilabel or single-label classification case."""

    text: str
    gold_labels: tuple[str, ...]


class ConversationQACase(BenchmarkCase):
    """Conversation history followed by one QA request."""

    conversation_id: str = Field(min_length=1)
    turns: tuple[dict[str, Any], ...]
    question: str
    gold_answer: str | None = None
    gold_evidence: tuple[str, ...] = ()


class ItemTiming(ContractModel):
    """Measured wall-clock fields for one case."""

    started_at: datetime
    finished_at: datetime
    total_ms: float = Field(ge=0)
    provider_ms: float | None = Field(default=None, ge=0)


class TokenUsage(ContractModel):
    """Provider-reported per-item token accounting when available."""

    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class PredictionRecord(ContractModel):
    """One task-appropriate prediction with timing and token diagnostics."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    case_id: str = Field(min_length=1)
    mode: BenchmarkMode
    prediction: str | tuple[str, ...]
    timing: ItemTiming
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    diagnostics: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_prediction_shape(self) -> PredictionRecord:
        """Keep classification and QA outputs independent."""

        if self.mode is BenchmarkMode.CLASSIFICATION and not isinstance(self.prediction, tuple):
            raise ValueError("classification predictions must be a tuple of labels")
        if self.mode is BenchmarkMode.CONVERSATION_QA and not isinstance(self.prediction, str):
            raise ValueError("conversation QA predictions must be text")
        return self


class FailureRecord(ContractModel):
    """Shared failure record with an optional benchmark-owned subcode."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    case_id: str | None = None
    category: FailureCategory
    subcode: str | None = None
    stage: LifecycleStage
    message: str
    retryable: bool = False
    occurred_at: datetime = Field(default_factory=utc_now)
    details: dict[str, Any] = Field(default_factory=dict)


class CheckpointRecord(ContractModel):
    """Append-only item outcome used for exact resume."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    case_id: str = Field(min_length=1)
    status: RunStatus
    prediction: PredictionRecord | None = None
    failure: FailureRecord | None = None
    recorded_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_outcome(self) -> CheckpointRecord:
        if self.prediction is not None and self.prediction.case_id != self.case_id:
            raise ValueError("checkpoint and prediction case IDs differ")
        if self.failure is not None and self.failure.case_id not in {None, self.case_id}:
            raise ValueError("checkpoint and failure case IDs differ")
        if self.status is RunStatus.COMPLETE and self.prediction is None:
            raise ValueError("complete checkpoint requires a prediction")
        if self.status is RunStatus.FAILED and self.failure is None:
            raise ValueError("failed checkpoint requires a failure record")
        return self


class CampaignReference(ContractModel):
    """Immutable lineage link to a prior campaign manifest."""

    campaign_id: str
    manifest_fingerprint: str
    status: RunStatus
    resumed_at: datetime = Field(default_factory=utc_now)


class BenchmarkManifest(ContractModel):
    """Complete reproducibility and resume-compatibility manifest."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    campaign_id: str = Field(min_length=1)
    benchmark: BenchmarkSpec
    source_fingerprint: str | None = None
    git_commit: str | None = None
    dataset_hash: str = Field(min_length=1)
    selected_ids: tuple[str, ...]
    provider: str
    model: str
    model_revision: str | None = None
    generation_config: dict[str, Any]
    benchmark_config: dict[str, Any] = Field(default_factory=dict)
    runtime_profile: str
    active_capabilities: dict[str, Any]
    repository_mode: str
    prompt_hashes: dict[str, str]
    seed: int | None
    dependencies: dict[str, str]
    python_version: str = Field(default_factory=lambda: sys.version)
    hardware: dict[str, Any]
    status: RunStatus = RunStatus.PARTIAL
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    resume_lineage: tuple[CampaignReference, ...] = ()
    manifest_fingerprint: str = ""

    @field_validator("selected_ids")
    @classmethod
    def selected_ids_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("selected case IDs must be unique")
        return value

    @model_validator(mode="after")
    def validate_source_and_fingerprint(self) -> BenchmarkManifest:
        if not self.source_fingerprint and not self.git_commit:
            raise ValueError("manifest requires source_fingerprint or git_commit")
        for value in (
            self.generation_config,
            self.benchmark_config,
            self.active_capabilities,
            self.dependencies,
            self.hardware,
        ):
            _reject_credentials(value)
        expected = manifest_fingerprint(self)
        if self.manifest_fingerprint and self.manifest_fingerprint != expected:
            raise ValueError("manifest fingerprint does not match its compatibility fields")
        if not self.manifest_fingerprint:
            object.__setattr__(self, "manifest_fingerprint", expected)
        return self


class BenchmarkSummary(ContractModel):
    """Final or partial campaign counts and benchmark metrics."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    campaign_id: str
    status: RunStatus
    total_cases: int = Field(ge=0)
    completed_cases: int = Field(ge=0)
    failed_cases: int = Field(ge=0)
    cancelled_cases: int = Field(ge=0)
    metrics: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime
    finished_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def complete_means_all_cases_terminal(self) -> BenchmarkSummary:
        terminal = self.completed_cases + self.failed_cases + self.cancelled_cases
        if terminal > self.total_cases:
            raise ValueError("terminal case counts exceed total cases")
        if self.status is RunStatus.COMPLETE and terminal != self.total_cases:
            raise ValueError("complete summary requires every selected case to be terminal")
        return self


class ProgressEvent(ContractModel):
    """Shared progress notification emitted at lifecycle boundaries."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    stage: LifecycleStage
    status: RunStatus = RunStatus.PARTIAL
    current: int = Field(default=0, ge=0)
    total: int = Field(default=0, ge=0)
    case_id: str | None = None
    message: str = ""
    timestamp: datetime = Field(default_factory=utc_now)


def stable_case_id(benchmark_name: str, identity: Any) -> str:
    """Return a stable opaque case ID from benchmark identity data."""

    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(f"{benchmark_name}\0{payload}".encode()).hexdigest()
    return f"{benchmark_name.lower()}_{digest[:24]}"


def manifest_fingerprint(manifest: BenchmarkManifest) -> str:
    """Hash fields that must remain identical across resume attempts."""

    payload = manifest.model_dump(
        mode="json",
        exclude={
            "campaign_id",
            "manifest_fingerprint",
            "status",
            "created_at",
            "updated_at",
            "resume_lineage",
        },
    )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _reject_credentials(value: Any) -> None:
    markers = ("api_key", "apikey", "authorization", "password", "secret", "credentials", "access_token")
    if isinstance(value, dict):
        for key, item in value.items():
            if any(marker in str(key).lower() for marker in markers):
                raise ValueError(f"credentials are forbidden in benchmark artifacts: {key}")
            _reject_credentials(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_credentials(item)
    elif isinstance(value, str) and value.startswith(("http://", "https://")):
        forbidden = ("token", "key", "auth", "signature", "password", "secret")
        if any(
            marker in key.lower()
            for key, _ in parse_qsl(urlsplit(value).query)
            for marker in forbidden
        ):
            raise ValueError("credentials are forbidden in benchmark artifact URLs")
