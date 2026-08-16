"""Typed request, response, and diagnostics models for the canonical runtime API."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from action.execution_policy import ExecutionPolicy
from config.runtime_mode import RuntimeMode
from llm.generation_config import GenerationConfig

SCHEMA_VERSION: Literal["1.0"] = "1.0"


class TaskKind(Enum):
    """Supported classes of work accepted by the public runtime boundary."""

    CONVERSATION = "conversation"
    HISTORICAL_REPLAY = "historical_replay"
    FACTUAL_QA = "factual_qa"
    DOCUMENT_INGESTION = "document_ingestion"
    EMOTION_CLASSIFICATION = "emotion_classification"
    TOOL_REQUEST = "tool_request"


class ExecutionProfile(Enum):
    """Named component-selection policies independent of benchmarks."""

    MODEL_ONLY = "model_only"
    SIMPLE_RAG = "simple_rag"
    PRIMA_FULL = "prima_full"
    AFFECT_ONLY = "affect_only"
    INGESTION_ONLY = "ingestion_only"


class ExecutionStatus(Enum):
    """Terminal status returned by a canonical runtime request."""

    COMPLETED = "completed"
    NOT_IMPLEMENTED = "not_implemented"
    REJECTED = "rejected"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExecutionOutcome(Enum):
    """Typed request result independent of workflow lifecycle status."""

    ANSWERED = "answered"
    ABSTAINED = "abstained"
    FAILED = "failed"
    INGESTED = "ingested"
    CLASSIFIED = "classified"
    ACTIONED = "actioned"
    CANCELLED = "cancelled"


class DiagnosticMode(Enum):
    """Amount of trace detail retained in the public response."""

    STANDARD = "standard"
    DIAGNOSTIC = "diagnostic"


class RuntimeComponent(Enum):
    """Route-addressable responsibilities derived from the target diagram."""

    INPUT_PARSER = "input_parser"
    TASK_SCHEDULER = "task_scheduler"
    STATE_MANAGER = "state_manager"
    POLICY_ROUTER = "policy_router"
    EXECUTION_DISPATCHER = "execution_dispatcher"
    PERSISTENT_STATE = "persistent_state"
    AFFECT_ENGINE = "affect_engine"
    QUERY_REWRITER = "query_rewriter"
    DENSE_RETRIEVAL = "dense_retrieval"
    SPARSE_RETRIEVAL = "sparse_retrieval"
    TEMPORAL_RETRIEVAL = "temporal_retrieval"
    GRAPH_TRAVERSAL = "graph_traversal"
    GRAPH_REASONING = "graph_reasoning"
    FUSION = "fusion"
    RERANKER = "reranker"
    CONFIDENCE_ESTIMATOR = "confidence_estimator"
    CONTEXT_COMPRESSOR = "context_compressor"
    PLANNER = "planner"
    WORLD_MODEL = "world_model"
    UNCERTAINTY_ESTIMATOR = "uncertainty_estimator"
    REFLECTION = "reflection"
    MODEL_EXECUTOR = "model_executor"
    TOOL_EXECUTOR = "tool_executor"
    OUTPUT_VALIDATOR = "output_validator"
    OUTPUT_SHAPER = "output_shaper"
    STATE_COMMIT = "state_commit"
    MEMORY_INDEX = "memory_index"
    MEMORY_COMMIT = "memory_commit"
    DOCUMENT_ENCODER = "document_encoder"
    EMOTION_CLASSIFIER = "emotion_classifier"
    MAINTENANCE_EVENTS = "maintenance_events"


class ContractModel(BaseModel):
    """Immutable, extra-forbidding base for public runtime contract models."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ComponentCapability(ContractModel):
    """Whether one component is callable through the canonical boundary."""

    component: RuntimeComponent
    available: bool
    reason: str | None = None


class StateDelta(ContractModel):
    """Serializable state changes committed by a request."""

    changes: dict[str, Any] = Field(default_factory=dict)


class EvidenceReference(ContractModel):
    """Evidence exposed in a response without benchmark gold information."""

    source_id: str
    text: str = ""
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RuntimeDiagnostics(ContractModel):
    """Planned, executed, and skipped components for one runtime request."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    route_name: str = "unplanned"
    planned_components: tuple[RuntimeComponent, ...] = ()
    enabled_components: tuple[RuntimeComponent, ...] = ()
    executed_components: tuple[RuntimeComponent, ...] = ()
    not_executed_components: tuple[RuntimeComponent, ...] = ()
    skipped_components: tuple[RuntimeComponent, ...] = ()
    capabilities: tuple[ComponentCapability, ...] = ()
    notes: tuple[str, ...] = ()
    runtime_mode: RuntimeMode = RuntimeMode.TEST
    memory_repository: str = "unconfigured"
    memory_persistent: bool = False
    state_version: int = 0
    execution_decision: str | None = None
    decision_rule: str | None = None
    decision_thresholds: dict[str, float | int] = Field(default_factory=dict)
    decision_history: tuple[dict[str, Any], ...] = ()
    component_details: dict[str, dict[str, Any]] = Field(default_factory=dict)
    correction_budget: dict[str, float | int] = Field(default_factory=dict)
    correction_attempts: tuple[dict[str, Any], ...] = ()
    maintenance: dict[str, Any] = Field(default_factory=dict)
    diagnostic_mode: DiagnosticMode = DiagnosticMode.STANDARD
    latency_ms: float = 0.0
    retrieval_count: int = 0
    retrieval_call_count: int = 0
    retrieval_result_count: int = 0
    reflection_count: int = 0
    workflow_reflection_count: int = 0
    reasoning_reflection_count: int = 0
    accepted_correction_count: int = 0
    model_call_count: int = 0
    model_usage: dict[str, int] = Field(default_factory=dict)
    provider: dict[str, Any] = Field(default_factory=dict)
    trace_event_count: int = 0
    trace_events: tuple[dict[str, Any], ...] = ()


class ExecutionOptions(ContractModel):
    """Trusted request-scoped execution controls kept outside caller metadata."""

    top_k: int = Field(default=5, gt=0)
    reasoning_mode: Literal["bypass", "single_pass", "adaptive", "deliberative", "diagnostic"] = "adaptive"
    max_hops: int = Field(default=3, gt=0)
    max_retrieval_calls: int = Field(default=3, gt=0)
    max_context_tokens: int = Field(default=1600, gt=0)
    context_compression_enabled: bool = True
    required_reranker_backend: Literal["disabled", "lexical_fallback", "cross_encoder"] | None = None
    execution_policy: ExecutionPolicy = Field(default_factory=ExecutionPolicy)
    tool_results: tuple[str, ...] = ()


_INTERNAL_CONTROL_KEYS = {
    "route",
    "task_kind",
    "profile",
    "generation_config",
    "execution_policy",
    "top_k",
    "reasoning_mode",
    "max_hops",
    "max_retrieval_calls",
    "max_context_tokens",
    "context_compression_enabled",
    "required_reranker_backend",
    "retrieval_query_override",
    "correction_budget",
    "tool_results",
}


class PrimaRequest(ContractModel):
    """Canonical typed input accepted by :meth:`PrimaRuntime.execute`."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    request_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    task_kind: TaskKind
    profile: ExecutionProfile
    input_text: str
    session_id: str | None = None
    generation_config: GenerationConfig | None = None
    options: ExecutionOptions = Field(default_factory=ExecutionOptions)
    diagnostic_mode: DiagnosticMode = DiagnosticMode.STANDARD
    redact_prompts: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("input_text")
    @classmethod
    def validate_input_text(cls, value: str) -> str:
        """Reject empty or whitespace-only request content."""

        if not value.strip():
            raise ValueError("input_text must not be empty")
        return value

    @field_validator("metadata")
    @classmethod
    def reject_internal_controls(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Keep untrusted metadata from changing workflow execution controls."""

        forbidden = sorted(_INTERNAL_CONTROL_KEYS & {str(key) for key in value})
        if forbidden:
            raise ValueError(f"metadata contains reserved execution keys: {', '.join(forbidden)}")
        return value

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible values."""

        return self.model_dump(mode="json")

    def to_json(self) -> str:
        """Serialize to versioned JSON."""

        return self.model_dump_json()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PrimaRequest:
        """Validate a request from a dictionary."""

        return cls.model_validate(value)

    @classmethod
    def from_json(cls, value: str | bytes) -> PrimaRequest:
        """Validate a request from JSON."""

        return cls.model_validate_json(value)


class PrimaResponse(ContractModel):
    """Canonical typed result returned by :meth:`PrimaRuntime.execute`."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    request_id: str
    task_kind: TaskKind
    profile: ExecutionProfile
    status: ExecutionStatus
    outcome: ExecutionOutcome | None = None
    output_text: str | None = None
    output_data: dict[str, Any] = Field(default_factory=dict)
    state_delta: StateDelta = Field(default_factory=StateDelta)
    evidence: tuple[EvidenceReference, ...] = ()
    diagnostics: RuntimeDiagnostics = Field(default_factory=RuntimeDiagnostics)
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible values."""

        return self.model_dump(mode="json")

    def to_json(self) -> str:
        """Serialize to versioned JSON."""

        return self.model_dump_json()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PrimaResponse:
        """Validate a response from a dictionary."""

        return cls.model_validate(value)

    @classmethod
    def from_json(cls, value: str | bytes) -> PrimaResponse:
        """Validate a response from JSON."""

        return cls.model_validate_json(value)
