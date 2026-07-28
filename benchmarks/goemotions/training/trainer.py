"""Small resource-aware DeBERTa training and smoke-training runner."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from benchmarks.goemotions.metrics import evaluate
from benchmarks.goemotions.training.config import TrainingConfig
from benchmarks.goemotions.training.data import load_split, multi_hot, validate_splits
from benchmarks.goemotions.training.losses import build_loss


def smoke_train(data_dir: Path, output_dir: Path, config: TrainingConfig | None = None, limit: int = 4) -> dict[str, Any]:
    """Run one real train and dev step, then save/reload a temporary model state."""
    config = config or TrainingConfig()
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if limit < 1:
        raise ValueError("limit must be positive")
    validation = validate_splits(data_dir)
    labels = list(validation["labels"])
    train, dev = load_split(data_dir, "train")[:limit], load_split(data_dir, "dev")[:limit]
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = _device(torch, config.device)
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.revision, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(config.model_id, revision=config.revision, num_labels=len(labels), problem_type="multi_label_classification", trust_remote_code=False).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = build_loss(config.loss)
    train_batch = tokenizer([row.text for row in train], padding=True, truncation=True, max_length=config.max_length, return_tensors="pt").to(device)
    targets = torch.tensor([multi_hot(row, labels) for row in train], dtype=torch.float32, device=device)
    loss = loss_fn(model(**train_batch).logits, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    with torch.inference_mode():
        dev_batch = tokenizer([row.text for row in dev], padding=True, truncation=True, max_length=config.max_length, return_tensors="pt").to(device)
        dev_logits = model(**dev_batch).logits
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "smoke_state.pt"
    torch.save(model.state_dict(), checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    checkpoint.unlink()
    return {"status": "passed", "device": str(device), "train_examples": len(train), "dev_examples": len(dev), "loss": round(float(loss.detach().cpu()), 6), "dev_shape": list(dev_logits.shape), "split_hashes": validation["hashes"], "peak_cuda_memory": int(torch.cuda.max_memory_allocated(device)) if str(device).startswith("cuda") else None}


def _device(torch, requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for training but unavailable.")
    return torch.device(requested)


def train(data_dir: Path, output_dir: Path, config: TrainingConfig | None = None) -> dict[str, Any]:
    """Train on train only, select the best checkpoint by development macro F1."""
    config = config or TrainingConfig()
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    validation = validate_splits(data_dir)
    labels = list(validation["labels"])
    train_rows, dev_rows = load_split(data_dir, "train"), load_split(data_dir, "dev")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = _device(torch, config.device)
    amp_enabled = config.mixed_precision and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.revision, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_id,
        revision=config.revision,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id={label: index for index, label in enumerate(labels)},
        problem_type="multi_label_classification",
        trust_remote_code=False,
    ).to(device)
    if config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    positive_weight = _positive_weight(torch, train_rows, labels, device) if config.loss == "weighted_bce" else None
    loss_fn, optimizer = build_loss(config.loss, positive_weight), torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    output_dir.mkdir(parents=True, exist_ok=True)
    best, stale, history = -1.0, 0, []
    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        batches = 0
        for index in range(0, len(train_rows), config.train_batch_size):
            batch = train_rows[index : index + config.train_batch_size]
            with torch.autocast(device_type="cuda", enabled=amp_enabled):
                loss = loss_fn(model(**_tokens(tokenizer, batch, config.max_length, device)).logits, _targets(torch, batch, labels, device)) / config.gradient_accumulation
            scaler.scale(loss).backward()
            batches += 1
            if batches % config.gradient_accumulation == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        if batches % config.gradient_accumulation:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        metrics, dev_logits, dev_probabilities = _evaluate(model, tokenizer, dev_rows, labels, config, torch, device)
        history.append({"epoch": epoch + 1, **metrics})
        if metrics["macro_f1"] > best:
            best, stale = metrics["macro_f1"], 0
            model.save_pretrained(output_dir / "best")
            tokenizer.save_pretrained(output_dir / "best")
            (output_dir / "dev_logits.json").write_text(json.dumps(dev_logits), encoding="utf-8")
            (output_dir / "dev_probabilities.json").write_text(json.dumps(dev_probabilities), encoding="utf-8")
        else:
            stale += 1
            if stale >= config.patience:
                break
    metadata = {"model_id": config.model_id, "config": config.__dict__ if hasattr(config, "__dict__") else {name: getattr(config, name) for name in config.__dataclass_fields__}, "dev_selection_metric": "macro_f1", "best_dev_macro_f1": best, "history": history, "split_hashes": validation["hashes"], "model_fingerprint": _fingerprint(config.model_id, validation["hashes"]["train"])}
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def _tokens(tokenizer: Any, rows: list[Any], max_length: int, device: Any) -> Any:
    return tokenizer([row.text for row in rows], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)


def _targets(torch: Any, rows: list[Any], labels: list[str], device: Any) -> Any:
    return torch.tensor([multi_hot(row, labels) for row in rows], dtype=torch.float32, device=device)


def _positive_weight(torch: Any, rows: list[Any], labels: list[str], device: Any) -> Any:
    positives = _targets(torch, rows, labels, device).sum(dim=0)
    return ((len(rows) - positives) / positives.clamp_min(1)).clamp_max(float(len(rows)))


def _evaluate(model: Any, tokenizer: Any, rows: list[Any], labels: list[str], config: TrainingConfig, torch: Any, device: Any) -> tuple[dict[str, Any], list[list[float]], list[dict[str, float]]]:
    model.eval()
    predicted: list[frozenset[str]] = []
    logits: list[list[float]] = []
    probabilities_by_label: list[dict[str, float]] = []
    with torch.inference_mode():
        for index in range(0, len(rows), config.eval_batch_size):
            batch_logits = model(**_tokens(tokenizer, rows[index : index + config.eval_batch_size], config.max_length, device)).logits
            logits.extend(batch_logits.cpu().tolist())
            probabilities = torch.sigmoid(batch_logits).cpu().tolist()
            for row in probabilities:
                selected = frozenset(label for label, value in zip(labels, row, strict=True) if label != "neutral" and value >= 0.5) or frozenset({"neutral"})
                predicted.append(selected)
                probabilities_by_label.append(dict(zip(labels, row, strict=True)))
    return evaluate([row.labels for row in rows], predicted, labels), logits, probabilities_by_label


def _fingerprint(model_id: str, train_hash: str) -> str:
    return hashlib.sha256(f"{model_id}:{train_hash}".encode()).hexdigest()
