"""CLI for the optional GoEmotions DeBERTa smoke-training path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.goemotions.training.config import TrainingConfig
from benchmarks.goemotions.training.trainer import smoke_train, train
from benchmarks.goemotions.training.thresholds import fit_thresholds_from_dev
from benchmarks.goemotions.training.calibration import fit_calibration_from_dev


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GoEmotions DeBERTa training or a tiny smoke step.")
    parser.add_argument("--data-dir", type=Path, default=Path("benchmarks/goemotions/external/goemotions/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/goemotions/models/smoke"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--full", action="store_true", help="Run configured train/dev training; never uses test labels.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--loss", choices=("bce", "weighted_bce", "focal", "asymmetric"), default="bce")
    parser.add_argument("--fit-thresholds", type=Path, metavar="DEV_PROBABILITIES_JSON")
    parser.add_argument("--fit-calibration", type=Path, metavar="DEV_LOGITS_JSON")
    parser.add_argument("--model-id", default="microsoft/deberta-v3-small")
    args = parser.parse_args()
    if args.fit_thresholds and args.fit_calibration:
        parser.error("Choose one of --fit-thresholds or --fit-calibration.")
    if args.fit_thresholds:
        result = fit_thresholds_from_dev(args.data_dir, args.fit_thresholds, args.output_dir / "thresholds.json", model_id=args.model_id)
    elif args.fit_calibration:
        result = fit_calibration_from_dev(args.data_dir, args.fit_calibration, args.output_dir / "calibration.json", model_id=args.model_id)
    else:
        config = TrainingConfig(model_id=args.model_id, device=args.device, epochs=args.epochs, loss=args.loss)
        result = train(args.data_dir, args.output_dir, config) if args.full else smoke_train(args.data_dir, args.output_dir, config, args.limit)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
