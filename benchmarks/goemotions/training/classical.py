"""Controlled TF-IDF/logistic GoEmotions trained baseline experiments."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import time
from pathlib import Path
from typing import Any

from affect.affect_engine import DynamicAffectEngine
from affect.affect_perception import GoEmotionsDecisionController
from affect.classifiers.goemotions_adapter import GoEmotionsProfileAdapter
from affect.emotion_prediction import EmotionPrediction
from affect.taxonomies.goemotions import LABELS
from benchmarks.goemotions.development import freeze_json, read_frozen_json
from benchmarks.goemotions.metrics import evaluate, paired_bootstrap_sample_f1
from benchmarks.goemotions.training.data import load_split, validate_splits
from benchmarks.goemotions.training.thresholds import select_thresholds, write_thresholds
from llm.generation_config import GenerationConfig
from llm.llm_client import LLMClient
from runtime.contracts import ExecutionProfile, PrimaRequest, TaskKind
from runtime.prima_runtime import PrimaRuntime


def run_tfidf_logreg(
    data_dir: Path, output_dir: Path, *, features: str = "word", balanced: bool = False, seed: int = 13
) -> dict[str, Any]:
    """Train only on train and select thresholds only on dev."""
    import joblib

    started = time.perf_counter()
    identity = validate_splits(data_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Training output must be empty")
    payload = _fit_model(data_dir, features, balanced, seed)
    vectorizer, classifier = payload["vectorizer"], payload["classifier"]
    dev = load_split(data_dir, "dev")
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "artifact_manifest.json").exists():
        raise ValueError("Frozen output already exists; choose a new directory")
    joblib.dump(payload, output_dir / "model.joblib")
    probabilities = _probability_rows(classifier.predict_proba(vectorizer.transform([row.text for row in dev])))
    global_predictions = _runtime_predictions(probabilities, {label: 0.5 for label in LABELS})
    global_metrics = evaluate([row.labels for row in dev], global_predictions, list(LABELS))
    threshold_artifact = select_thresholds(probabilities, [row.labels for row in dev], list(LABELS), min_support=5)
    final_predictions, updates = _runtime_predictions(
        probabilities, threshold_artifact["thresholds"], return_updates=True
    )
    final_metrics = evaluate([row.labels for row in dev], final_predictions, list(LABELS))
    write_thresholds(
        output_dir / "thresholds.json",
        threshold_artifact,
        model_id=f"tfidf-{features}-logreg",
        dev_hash=identity["hashes"]["dev"],
    )
    thresholds_path = output_dir / "thresholds.json"
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    thresholds.update({"model_sha256": _file_hash(output_dir / "model.joblib"),
                       "configuration": {"features": features, "balanced": balanced, "seed": seed}})
    thresholds_path.write_text(json.dumps(thresholds, indent=2, sort_keys=True), encoding="utf-8")
    artifact = _write_artifact_manifest(output_dir, identity, payload)
    (output_dir / "dev_predictions.json").write_text(json.dumps([
        {"id": row.example_id, "gold_labels": sorted(row.labels), "baseline_labels": sorted(before),
         "predicted_labels": sorted(after), "parse_error": None}
        for row, before, after in zip(dev, global_predictions, final_predictions, strict=True)
    ], indent=2), encoding="utf-8")
    changes = _change_analysis([row.labels for row in dev], global_predictions, final_predictions)
    result = {
        "system": "trained_tfidf_logistic",
        "system_family": "trained_baseline",
        "wrapper_attribution": False,
        "features": features,
        "balanced": balanced,
        "seed": seed,
        "model_artifact": "model.joblib",
        "global_threshold_metrics": global_metrics,
        "dev_thresholded_metrics": final_metrics,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "threshold_change_analysis": changes,
        "runtime_routes_verified": len(updates),
        "artifact_fingerprint": artifact["fingerprint"],
        "paired_dev_sample_f1": paired_bootstrap_sample_f1(
            [row.labels for row in dev], global_predictions, final_predictions, samples=1000, seed=seed),
        "selection_scope": "threshold_fit_dev_not_independent_validation",
        "production_promoted": False,
        "test_evaluated": False,
    }
    result["runtime_seconds"] = round(time.perf_counter() - started, 3)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _fit_model(data_dir: Path, features: str, balanced: bool, seed: int) -> dict[str, Any]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import FeatureUnion
    from sklearn.preprocessing import MultiLabelBinarizer

    if features not in {"word", "char", "combined"}:
        raise ValueError("features must be word, char, or combined")
    train = load_split(data_dir, "train")
    vectorizers: list[tuple[str, Any]] = []
    if features in {"word", "combined"}:
        vectorizers.append(
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=80_000))
        )
    if features in {"char", "combined"}:
        vectorizers.append(
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2, max_features=100_000
                ),
            )
        )
    vectorizer = vectorizers[0][1] if len(vectorizers) == 1 else FeatureUnion(vectorizers)
    binarizer = MultiLabelBinarizer(classes=list(LABELS))
    x_train, y_train = (
        vectorizer.fit_transform([row.text for row in train]),
        binarizer.fit_transform([row.labels for row in train]),
    )
    classifier = OneVsRestClassifier(
        LogisticRegression(
            solver="liblinear", max_iter=200, class_weight="balanced" if balanced else None, random_state=seed
        )
    )
    classifier.fit(x_train, y_train)
    return {"vectorizer": vectorizer, "classifier": classifier, "labels": list(LABELS),
            "features": features, "balanced": balanced, "seed": seed}


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dependencies() -> dict[str, str]:
    return {"python": platform.python_version(), **{name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "scikit-learn", "joblib")}}


def _feature_settings(payload: dict[str, Any]) -> dict[str, Any]:
    vectorizer = payload["vectorizer"]
    vectorizers = vectorizer.transformer_list if payload["features"] == "combined" else [(payload["features"], vectorizer)]
    keys = ("analyzer", "ngram_range", "min_df", "max_df", "max_features", "sublinear_tf", "norm",
            "use_idf", "smooth_idf", "lowercase", "token_pattern", "strip_accents")
    return json.loads(json.dumps({name: {key: component.get_params()[key] for key in keys}
                                 for name, component in vectorizers}))


def _write_artifact_manifest(output_dir: Path, identity: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    return freeze_json(output_dir / "artifact_manifest.json", {
        "version": 1, "system_family": "trained_baseline", "dataset_identity": identity,
        "configuration": {key: payload[key] for key in ("features", "balanced", "seed", "labels")},
        "dependencies": _dependencies(),
        "feature_settings": _feature_settings(payload),
        "logistic_settings": payload["classifier"].estimator.get_params(),
        "artifacts": {name: _file_hash(output_dir / name) for name in ("model.joblib", "thresholds.json")},
        "threshold_split": "dev", "training_split": "train", "calibrated": False,
    })


def _validate_thresholds(path: Path, identity: dict[str, Any], model_hash: str,
                         configuration: dict[str, Any]) -> dict[str, float]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    values = artifact.get("thresholds", {})
    if (artifact.get("dev_dataset_sha256") != identity["hashes"]["dev"]
            or artifact.get("model_sha256") != model_hash
            or artifact.get("configuration") != {key: configuration[key] for key in ("features", "balanced", "seed")}):
        raise ValueError("Threshold model/config/dev identity mismatch")
    if set(values) != set(LABELS) or any(type(value) not in (int, float) or not math.isfinite(value)
                                       or not 0 <= value <= 1 for value in values.values()):
        raise ValueError("Thresholds must be finite [0, 1] values in official label order")
    return values


def load_frozen_model(path: Path, *, data_dir: Path | None = None, thresholds_path: Path | None = None,
                      expected_fingerprint: str | None = None) -> dict[str, Any]:
    """Validate trusted local artifact hashes before unpickling; pin fingerprint at handoff."""
    import joblib

    artifact = read_frozen_json(path.with_name("artifact_manifest.json"), expected_fingerprint)
    thresholds_path = thresholds_path or path.with_name("thresholds.json")
    if artifact["dependencies"] != _dependencies():
        raise ValueError("Frozen classifier dependency identity changed")
    for name, actual_path in (("model.joblib", path), ("thresholds.json", thresholds_path)):
        if artifact["artifacts"][name] != _file_hash(actual_path):
            raise ValueError(f"Frozen classifier artifact hash mismatch: {name}")
    identity = artifact["dataset_identity"]
    if data_dir is not None and validate_splits(data_dir) != identity:
        raise ValueError("Frozen classifier dataset identity changed")
    _validate_thresholds(thresholds_path, identity, artifact["artifacts"]["model.joblib"], artifact["configuration"])
    payload = joblib.load(path)
    if (payload.get("labels") != list(LABELS)
            or any(payload.get(key) != value for key, value in artifact["configuration"].items())):
        raise ValueError("Frozen classifier configuration or official label order changed")
    if _feature_settings(payload) != artifact["feature_settings"]:
        raise ValueError("Frozen vectorizer feature settings changed")
    return payload


def freeze_tfidf_candidate(
    data_dir: Path, source_thresholds: Path, output_dir: Path, *, features: str = "combined",
    balanced: bool = True, seed: int = 13, source_model: Path | None = None,
) -> None:
    """Freeze the existing train-only fit with its matching dev-selected thresholds."""
    identity = validate_splits(data_dir)
    source_model = source_model or source_thresholds.with_name("model.joblib")
    payload = load_frozen_model(source_model, data_dir=data_dir, thresholds_path=source_thresholds)
    if any(payload[key] != value for key, value in {"features": features, "balanced": balanced, "seed": seed}.items()):
        raise ValueError("Requested freeze settings differ from fitted features/configuration")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Freeze destination must be empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Copy the exact fitted model; refitting would invalidate model-bound thresholds.
    shutil.copy2(source_model, output_dir / "model.joblib")
    shutil.copy2(source_thresholds, output_dir / "thresholds.json")
    _write_artifact_manifest(output_dir, identity, payload)


def run_frozen_tfidf(
    model_path: Path,
    thresholds_path: Path,
    data_dir: Path,
    output_dir: Path,
    *,
    split: str = "test",
    max_samples: int = 0,
    sample_manifest: Path | None = None,
    progress: bool = True,
    expected_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Score a frozen trained baseline through the public classification route."""
    split_validation = validate_splits(data_dir)
    payload, rows = (load_frozen_model(model_path, data_dir=data_dir, thresholds_path=thresholds_path,
                                        expected_fingerprint=expected_fingerprint),
                     load_split(data_dir, split))
    if sample_manifest and max_samples:
        raise ValueError("Use either sample_manifest or max_samples, not both.")
    if sample_manifest:
        from benchmarks.goemotions.experiment import _examples_from_manifest

        rows = _examples_from_manifest(rows, data_dir / f"{split}.tsv", sample_manifest)
    if max_samples:
        rows = rows[:max_samples]
    probabilities: list[dict[str, float]] = []
    started = time.perf_counter()
    for index in range(0, len(rows), 256):
        probabilities.extend(
            _probability_rows(
                payload["classifier"].predict_proba(
                    payload["vectorizer"].transform([row.text for row in rows[index : index + 256]])
                )
            )
        )
        if progress:
            completed = min(index + 256, len(rows))
            rate = completed / max(time.perf_counter() - started, 1e-6)
            print(
                f"GoEmotions {split}: {completed}/{len(rows)} [{completed * 100 // len(rows)}%] | {rate:.1f} samples/s",
                flush=True,
            )
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))["thresholds"]
    predictions, updates = _runtime_predictions(probabilities, thresholds, return_updates=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "system": "trained_tfidf_logistic",
        "system_family": "trained_baseline",
        "wrapper_attribution": False,
        "split": split,
        "split_validation": split_validation,
        "sample_count": len(rows),
        "metrics": evaluate([row.labels for row in rows], predictions, list(LABELS)),
        "runtime_routes_verified": len(updates),
        "model": str(model_path),
        "thresholds": str(thresholds_path),
        "sample_manifest": str(sample_manifest) if sample_manifest else None,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def evaluate_frozen_tfidf(
    data_dir: Path, threshold_path: Path, *, features: str, balanced: bool, seed: int = 13
) -> dict[str, Any]:
    """Compatibility API: score an already frozen model without retraining on evaluation."""
    import tempfile

    payload = load_frozen_model(threshold_path.with_name("model.joblib"), data_dir=data_dir,
                                thresholds_path=threshold_path)
    if any(payload[key] != value for key, value in {"features": features, "balanced": balanced, "seed": seed}.items()):
        raise ValueError("Evaluation settings differ from the frozen fit")
    with tempfile.TemporaryDirectory() as directory:
        return run_frozen_tfidf(threshold_path.with_name("model.joblib"), threshold_path, data_dir,
                                Path(directory), split="test", progress=False)


def _probability_rows(matrix: Any) -> list[dict[str, float]]:
    return [{label: float(value) for label, value in zip(LABELS, row, strict=True)} for row in matrix]


def _runtime_predictions(
    probabilities: list[dict[str, float]], thresholds: dict[str, float], *, return_updates: bool = False
):
    async def execute_batch():
        predictions, updates = [], []
        generation = GenerationConfig(model="tfidf-logreg", provider="ollama")
        client = LLMClient(provider_name=generation.provider)
        for row in probabilities:
            prediction = GoEmotionsDecisionController(thresholds).decide(row, model_id="tfidf-logreg")

            # Independent Reddit examples must not share emotional persistence.
            class Current:
                value = prediction

                def predict(self, text: str) -> EmotionPrediction:
                    return self.value

            runtime = PrimaRuntime(
                affect_engine=DynamicAffectEngine(classifier=GoEmotionsProfileAdapter(Current())),
                llm_client=client,
                generation_config=generation,
                maintenance_enabled=False,
            )
            update = await runtime.execute(
                    PrimaRequest(
                        task_kind=TaskKind.EMOTION_CLASSIFICATION,
                        profile=ExecutionProfile.AFFECT_ONLY,
                        input_text="classifier input",
                    )
            )
            if update.diagnostics.route_name != "emotion_classification:affect_only":
                raise RuntimeError("TF-IDF baseline did not use the canonical classification route.")
            predictions.append(frozenset(prediction.selected_labels))
            updates.append(update)
        return (predictions, updates) if return_updates else predictions

    return asyncio.run(execute_batch())


def _change_analysis(
    gold: list[frozenset[str]], initial: list[frozenset[str]], final: list[frozenset[str]]
) -> dict[str, int]:
    changed = beneficial = harmful = 0
    for truth, before, after in zip(gold, initial, final, strict=True):
        if before != after:
            changed += 1
            beneficial += int(before != truth and after == truth)
            harmful += int(before == truth and after != truth)
    return {"changed": changed, "beneficial": beneficial, "harmful": harmful, "unchanged": len(gold) - changed}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train or score a TF-IDF/logistic GoEmotions baseline.")
    parser.add_argument("--train", action="store_true", help="Write model.joblib and thresholds.json to --output-dir.")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--expected-fingerprint", help="Pin artifact_manifest.json identity before loading.")
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("benchmarks/goemotions/external/goemotions/data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), default="test")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--sample-manifest", type=Path)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--features", choices=("word", "char", "combined"), default="combined")
    parser.add_argument("--balanced", action="store_true", default=True)
    parser.add_argument("--unbalanced", dest="balanced", action="store_false")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    if args.train:
        result = run_tfidf_logreg(
            args.data_dir, args.output_dir, features=args.features, balanced=args.balanced, seed=args.seed
        )
    else:
        if not args.model or not args.thresholds:
            parser.error("--model and --thresholds are required unless --train is used.")
        result = run_frozen_tfidf(
            args.model,
            args.thresholds,
            args.data_dir,
            args.output_dir,
            split=args.split,
            max_samples=args.max_samples,
            sample_manifest=args.sample_manifest,
            progress=not args.no_progress,
            expected_fingerprint=args.expected_fingerprint,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
