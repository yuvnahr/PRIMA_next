"""Thin adapters from campaign modes to the existing benchmark runners."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, cast

from benchmarks.campaign.config import BenchmarkModeConfig, CampaignConfig
from benchmarks.campaign.provider_session import SharedProviderSession
from llm.generation_config import GenerationConfig, StructuredOutputMode


def execute_mode(
    campaign: CampaignConfig,
    mode: BenchmarkModeConfig,
    session: SharedProviderSession,
    output_root: Path,
    *,
    resume: bool,
) -> dict[str, Any]:
    generation = _generation(campaign, mode)
    options = dict(mode.options)
    workers = min(int(options.pop("parallel_workers", campaign.scheduler.cpu_workers)), campaign.scheduler.cpu_workers)
    common: dict[str, Any] = {
        "provider": campaign.provider.kind,
        "model": campaign.provider.model,
        "seed": mode.seed,
        "max_samples": mode.max_items,
        "output_path": output_root,
        "parallel_workers": workers,
        "resume": resume,
        "generation_config": generation,
    }
    if mode.benchmark == "goemotions":
        run_goemotions_experiment = import_module("benchmarks.goemotions.experiment").run_goemotions_experiment
        return cast(dict[str, Any], run_goemotions_experiment(
            dataset_path=mode.dataset_path,
            system=mode.variant,
            split=str(options.pop("split", "test")),
            batch_size=int(options.pop("batch_size", 1)),
            device=str(options.pop("device", "auto")),
            bootstrap_samples=int(options.pop("bootstrap_samples", 1000)),
            progress=False,
            llm_client=session.client,
            **common,
            **options,
        ))
    common.pop("max_samples")
    if mode.benchmark == "hotpotqa":
        run_hotpotqa_experiment = import_module("benchmarks.hotpotqa.experiment").run_hotpotqa_experiment
        return cast(dict[str, Any], run_hotpotqa_experiment(
            dataset_path=mode.dataset_path,
            mode=mode.variant,
            runtime_profile=mode.profile,
            max_samples=mode.max_items,
            sampling=str(options.pop("sampling", "sequential")),
            progress=False,
            quiet=True,
            runtime_factory=session.runtime_factory,
            **common,
            **options,
        ))
    run_locomo_experiment = import_module("benchmarks.locomo.experiment").run_locomo_experiment
    return cast(dict[str, Any], run_locomo_experiment(
        dataset_path=str(mode.dataset_path),
        runtime_profile=mode.profile,
        ingestion_policy=mode.variant,
        max_questions=mode.max_items or None,
        max_conversations=options.pop("max_conversations", None),
        full_dataset=bool(options.pop("full_dataset", False)),
        include_rouge_l="rouge_l" in mode.optional_metrics,
        include_bertscore="bertscore" in mode.optional_metrics,
        runtime_factory=session.runtime_factory,
        **common,
        **options,
    ))


def child_manifest_path(mode: BenchmarkModeConfig, output_root: Path) -> Path:
    if mode.benchmark == "goemotions":
        return output_root / mode.variant / "manifest.json"
    if mode.benchmark == "hotpotqa":
        return output_root / mode.variant / mode.profile / "manifest.json"
    return output_root / mode.profile / "manifest.json"


def _generation(campaign: CampaignConfig, mode: BenchmarkModeConfig) -> GenerationConfig:
    structured = StructuredOutputMode.NONE
    if mode.benchmark == "goemotions" and mode.variant != "model_only_zero_shot":
        structured = StructuredOutputMode.JSON_SCHEMA
    return GenerationConfig(
        model=campaign.provider.model,
        provider=campaign.provider.kind,
        temperature=campaign.provider.temperature,
        seed=mode.seed,
        max_output_tokens=campaign.provider.max_output_tokens,
        timeout_seconds=campaign.provider.timeout_seconds,
        retries=campaign.provider.retries,
        structured_output=structured,
    )
