"""Independent multilabel loss ablations; choose exactly one per run."""

from __future__ import annotations


def build_loss(name: str, positive_weight=None):
    """Return a torch loss lazily, avoiding a Torch dependency at import time."""
    import torch
    import torch.nn.functional as functional

    if name == "bce":
        return torch.nn.BCEWithLogitsLoss()
    if name == "weighted_bce":
        return torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    if name == "focal":
        return lambda logits, targets: _focal(functional, logits, targets)
    if name == "asymmetric":
        return lambda logits, targets: _asymmetric(torch, logits, targets)
    raise ValueError(f"Unsupported multilabel loss: {name}")


def _focal(functional, logits, targets, gamma: float = 2.0):
    losses = functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probabilities = logits.sigmoid()
    pt = targets * probabilities + (1 - targets) * (1 - probabilities)
    return ((1 - pt).pow(gamma) * losses).mean()


def _asymmetric(torch, logits, targets, gamma_negative: float = 4.0, gamma_positive: float = 1.0):
    probabilities = logits.sigmoid().clamp(1e-8, 1 - 1e-8)
    positive = targets * torch.log(probabilities) * (1 - probabilities).pow(gamma_positive)
    negative = (1 - targets) * torch.log(1 - probabilities) * probabilities.pow(gamma_negative)
    return -(positive + negative).mean()
