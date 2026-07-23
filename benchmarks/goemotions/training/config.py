"""Resource-aware immutable DeBERTa training configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    model_id: str = "microsoft/deberta-v3-small"
    seed: int = 13
    max_length: int = 128
    train_batch_size: int = 2
    eval_batch_size: int = 4
    gradient_accumulation: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    epochs: int = 3
    loss: str = "bce"
    device: str = "auto"
    num_workers: int = 0
    mixed_precision: bool = True
    patience: int = 2
    gradient_checkpointing: bool = True
