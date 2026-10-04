"""Validated YAML/JSON campaign configuration."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from config.runtime_config import RuntimeConfig

PLACEHOLDER = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProviderConfig(ConfigModel):
    kind: Literal["fake", "ollama", "openai", "anthropic", "lmstudio", "local"]
    model: str = Field(min_length=1)
    revision: str | None = None
    endpoint: str | None = None
    health_path: str = ""
    context_window: int = Field(gt=0)
    structured_output: bool = True
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=128, gt=0)
    safety_overhead_tokens: int = Field(default=256, ge=0)
    timeout_seconds: float = Field(default=60.0, gt=0)
    retries: int = Field(default=0, ge=0)
    fake_delay_seconds: float = Field(default=0.0, ge=0.0)


class RepositoryConfig(ConfigModel):
    url: str | None = None
    requested_ref: str | None = None
    commit_sha: str | None = None


class SchedulerConfig(ConfigModel):
    mode: Literal["sequential", "interleaved"] = "sequential"
    max_gpu_requests: int = Field(default=1, gt=0)
    cpu_workers: int = Field(default=3, gt=0)
    gpu_devices: tuple[str, ...] = ()
    allow_api_concurrency: bool = False
    ollama_parallel_slots: int = Field(default=1, ge=1, le=2)
    min_free_disk_gb: float = Field(default=0.1, ge=0.0)


class FailurePolicyConfig(ConfigModel):
    continue_benchmark_failures: bool = True
    continue_item_failures: bool = True


class TelemetryConfig(ConfigModel):
    enabled: bool = True
    gpu: bool = False


class BenchmarkModeConfig(ConfigModel):
    id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    benchmark: Literal["goemotions", "hotpotqa", "locomo"]
    dataset_path: Path
    profile: Literal["model_only", "simple_rag", "prima_full", "affect_only"]
    variant: str
    seed: int = 13
    context_budget: int = Field(gt=0)
    max_items: int = Field(default=0, ge=0)
    repository_mode: Literal["in_memory"] = "in_memory"
    optional_metrics: tuple[str, ...] = ()
    options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_route(self) -> BenchmarkModeConfig:
        profiles = {
            "goemotions": {"affect_only"},
            "hotpotqa": {"model_only", "simple_rag", "prima_full"},
            "locomo": {"model_only", "simple_rag", "prima_full"},
        }
        variants = {
            "goemotions": {
                "model_only_zero_shot",
                "schema_constrained_model_only",
                "affect_telemetry_preserve_labels",
                "bounded_prima_affect_decision",
                "trained_encoder",
            },
            "hotpotqa": {"distractor", "official_retrieved", "oracle"},
            "locomo": {
                "controlled_document_ingestion",
                "simple_vector_memory_baseline",
                "normal_prima_admission",
            },
        }
        if self.profile not in profiles[self.benchmark]:
            raise ValueError(f"{self.benchmark} does not support profile {self.profile}")
        if self.variant not in variants[self.benchmark]:
            raise ValueError(f"{self.benchmark} does not support variant {self.variant}")
        return self


class ComparisonConfig(ConfigModel):
    id: str = Field(min_length=1)
    left: str
    right: str
    metric: str = Field(min_length=1)
    bootstrap_samples: int = Field(default=1000, gt=0)


class CampaignConfig(ConfigModel):
    schema_version: Literal["1.0"] = "1.0"
    run_kind: Literal["benchmark", "smoke"] = "benchmark"
    output_root: Path
    provider: ProviderConfig
    repository: RepositoryConfig = Field(default_factory=RepositoryConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    failure_policy: FailurePolicyConfig = Field(default_factory=FailurePolicyConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig.from_environment)
    benchmarks: tuple[BenchmarkModeConfig, ...]
    comparisons: tuple[ComparisonConfig, ...] = ()

    @model_validator(mode="after")
    def validate_campaign(self) -> CampaignConfig:
        if not self.benchmarks:
            raise ValueError("campaign requires at least one benchmark mode")
        if self.run_kind == "smoke":
            if len(self.benchmarks) != 1 or self.comparisons:
                raise ValueError("Smoke requires one benchmark mode without paired comparisons")
            mode = self.benchmarks[0]
            count = mode.max_items
            if mode.benchmark == "locomo":
                conversations = mode.options.get("max_conversations", 0)
                if type(conversations) is not int or conversations <= 0 or mode.options.get("full_dataset"):
                    raise ValueError("Smoke LoCoMo requires bounded conversations and questions")
                count *= conversations
            if not 0 < count <= 3:
                raise ValueError("Smoke must select at most three examples, with a positive limit")
        modes = {mode.id: mode for mode in self.benchmarks}
        if len(modes) != len(self.benchmarks):
            raise ValueError("benchmark mode IDs must be unique")
        if len({comparison.id for comparison in self.comparisons}) != len(self.comparisons):
            raise ValueError("comparison IDs must be unique")
        for comparison in self.comparisons:
            if comparison.left not in modes or comparison.right not in modes:
                raise ValueError(f"comparison {comparison.id!r} references an unknown mode")
            if modes[comparison.left].benchmark != modes[comparison.right].benchmark:
                raise ValueError(f"comparison {comparison.id!r} must pair modes from one benchmark")
            left, right = modes[comparison.left], modes[comparison.right]
            if left.seed != right.seed or left.context_budget != right.context_budget:
                raise ValueError(f"comparison {comparison.id!r} requires the same seed and context budget")
            if left.max_items != right.max_items:
                raise ValueError(f"comparison {comparison.id!r} requires the same selected-item limit")
            if _uses_schema(left) != _uses_schema(right):
                raise ValueError(f"comparison {comparison.id!r} requires the same generation configuration")
        if self.scheduler.max_gpu_requests > 1:
            validated = self.provider.kind in {"openai", "anthropic"} and self.scheduler.allow_api_concurrency
            pilot = (
                self.provider.kind in {"ollama", "fake"}
                and self.scheduler.max_gpu_requests == self.scheduler.ollama_parallel_slots == 2
                and self.scheduler.mode == "sequential"
                and self.scheduler.cpu_workers == 2
                and all(
                    mode.options.get("parallel_workers") == 2
                    and not (mode.benchmark == "goemotions" and mode.variant == "trained_encoder")
                    for mode in self.benchmarks
                )
            )
            if not (validated or pilot):
                raise ValueError(
                    "max_gpu_requests > 1 requires a bounded two-worker native pilot or validated API concurrency"
                )
        return self


def load_campaign_config(path: str | Path, environment: dict[str, str] | None = None) -> CampaignConfig:
    """Load, resolve exact environment placeholders, validate, and anchor paths."""

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Campaign config not found: {source}")
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
    elif source.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError("YAML campaign configs require PyYAML from requirements.txt") from exc
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    else:
        raise ValueError("Campaign config must be JSON, YAML, or YML")
    if not isinstance(payload, dict):
        raise ValueError("Campaign config root must be an object")
    resolved = _resolve_placeholders(payload, environment or dict(os.environ))
    config = CampaignConfig.model_validate(resolved)
    modes = tuple(mode.model_copy(update={"dataset_path": _anchor(source.parent, mode.dataset_path)}) for mode in config.benchmarks)
    return config.model_copy(update={"output_root": _anchor(source.parent, config.output_root), "benchmarks": modes})


def _resolve_placeholders(value: Any, environment: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_placeholders(item, environment) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_placeholders(item, environment) for item in value]
    if isinstance(value, str):
        match = PLACEHOLDER.fullmatch(value)
        if match:
            name = match.group(1)
            if not environment.get(name):
                raise ValueError(f"Unresolved campaign environment placeholder: ${{{name}}}")
            return environment[name]
    return value


def _anchor(parent: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (parent / path).resolve()


def _uses_schema(mode: BenchmarkModeConfig) -> bool:
    return mode.benchmark == "goemotions" and mode.variant != "model_only_zero_shot"
