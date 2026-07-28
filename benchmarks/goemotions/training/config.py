"""Resource-aware immutable DeBERTa training configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    model_id: str = "microsoft/deberta-v3-small"
    revision: str = "a59be8aa63396e73dbb45a1487e4cde4be98bfa4"
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

    def __post_init__(self) -> None:
        if len(self.revision) < 7 or any(character not in "0123456789abcdefABCDEF" for character in self.revision):
            raise ValueError("revision must be an immutable Hugging Face commit hash")
