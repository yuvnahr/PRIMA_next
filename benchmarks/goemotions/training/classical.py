"""Controlled TF-IDF multilabel experiments that feed PRIMA's controller."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from affect.affect_engine import DynamicAffectEngine
from affect.affect_perception import GoEmotionsDecisionController
from affect.classifiers.goemotions_adapter import GoEmotionsProfileAdapter
from affect.emotion_prediction import EmotionPrediction
from affect.taxonomies.goemotions import LABELS
from benchmarks.goemotions.metrics import evaluate
from benchmarks.goemotions.training.data import load_split, validate_splits
from benchmarks.goemotions.training.thresholds import select_thresholds, write_thresholds


def run_tfidf_logreg(data_dir: Path, output_dir: Path, *, features: str = "word", balanced: bool = False, seed: int = 13) -> dict[str, Any]:
    """Train only on train, select thresholds on dev, and run PRIMA per example."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import FeatureUnion
    from sklearn.preprocessing import MultiLabelBinarizer

    if features not in {"word", "char", "combined"}:
        raise ValueError("features must be word, char, or combined")
    started = time.perf_counter()
    train, dev = load_split(data_dir, "train"), load_split(data_dir, "dev")
    vectorizers: list[tuple[str, Any]] = []
    if features in {"word", "combined"}:
        vectorizers.append(("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=80_000)))
    if features in {"char", "combined"}:
        vectorizers.append(("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2, max_features=100_000)))
    vectorizer = vectorizers[0][1] if len(vectorizers) == 1 else FeatureUnion(vectorizers)
    binarizer = MultiLabelBinarizer(classes=list(LABELS))
    x_train, y_train = vectorizer.fit_transform([row.text for row in train]), binarizer.fit_transform([row.labels for row in train])
    classifier = OneVsRestClassifier(LogisticRegression(solver="liblinear", max_iter=200, class_weight="balanced" if balanced else None, random_state=seed))
    classifier.fit(x_train, y_train)
    import joblib
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "classifier": classifier, "labels": list(LABELS), "features": features, "balanced": balanced, "seed": seed}, output_dir / "model.joblib")
    probabilities = _probability_rows(classifier.predict_proba(vectorizer.transform([row.text for row in dev])))
    global_predictions = _prima_predictions(probabilities, {label: 0.5 for label in LABELS})
    global_metrics = evaluate([row.labels for row in dev], global_predictions, list(LABELS))
    threshold_artifact = select_thresholds(probabilities, [row.labels for row in dev], list(LABELS), min_support=5)
    final_predictions, updates = _prima_predictions(probabilities, threshold_artifact["thresholds"], return_updates=True)
    final_metrics = evaluate([row.labels for row in dev], final_predictions, list(LABELS))
    write_thresholds(output_dir / "thresholds.json", threshold_artifact, model_id=f"tfidf-{features}-logreg", dev_hash=validate_splits(data_dir)["hashes"]["dev"])
    changes = _change_analysis([row.labels for row in dev], global_predictions, final_predictions)
    result = {"system": "prima_tfidf_logreg", "features": features, "balanced": balanced, "seed": seed, "model_artifact": "model.joblib", "global_threshold_metrics": global_metrics, "dev_thresholded_metrics": final_metrics, "runtime_seconds": round(time.perf_counter() - started, 3), "prima_change_analysis": changes, "updates_verified": len(updates)}
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def load_frozen_model(path: Path) -> dict[str, Any]:
    import joblib

    payload = joblib.load(path)
    if payload.get("labels") != list(LABELS):
        raise ValueError("Frozen classifier label order is not GoEmotions official order.")
    return payload


def freeze_tfidf_candidate(data_dir: Path, source_thresholds: Path, output_dir: Path, *, features: str = "combined", balanced: bool = True, seed: int = 13) -> None:
    """Fit train-only selected model and package it with an already-dev-selected threshold file."""
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import FeatureUnion
    from sklearn.preprocessing import MultiLabelBinarizer

    train = load_split(data_dir, "train")
    vectorizer = FeatureUnion([("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=80_000)), ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2, max_features=100_000))])
    matrix = vectorizer.fit_transform([row.text for row in train])
    labels = MultiLabelBinarizer(classes=list(LABELS)).fit_transform([row.labels for row in train])
    classifier = OneVsRestClassifier(LogisticRegression(solver="liblinear", max_iter=200, class_weight="balanced" if balanced else None, random_state=seed)).fit(matrix, labels)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "classifier": classifier, "labels": list(LABELS), "features": features, "balanced": balanced, "seed": seed}, output_dir / "model.joblib")
    shutil.copy2(source_thresholds, output_dir / "thresholds.json")
    (output_dir / "configuration.json").write_text(json.dumps({"features": features, "balanced": balanced, "seed": seed, "train_split": "train.tsv", "threshold_split": "dev.tsv"}, indent=2, sort_keys=True), encoding="utf-8")


def run_frozen_tfidf(model_path: Path, thresholds_path: Path, data_dir: Path, output_dir: Path, *, split: str = "test", max_samples: int = 0, sample_manifest: Path | None = None, progress: bool = True) -> dict[str, Any]:
    """Score a frozen candidate through fresh PRIMA affect state per sample."""
    payload, rows = load_frozen_model(model_path), load_split(data_dir, split)
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
        probabilities.extend(_probability_rows(payload["classifier"].predict_proba(payload["vectorizer"].transform([row.text for row in rows[index : index + 256]]))))
        if progress:
            completed = min(index + 256, len(rows))
            rate = completed / max(time.perf_counter() - started, 1e-6)
            print(f"GoEmotions {split}: {completed}/{len(rows)} [{completed * 100 // len(rows)}%] | {rate:.1f} samples/s", flush=True)
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))["thresholds"]
    predictions, updates = _prima_predictions(probabilities, thresholds, return_updates=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {"split": split, "sample_count": len(rows), "metrics": evaluate([row.labels for row in rows], predictions, list(LABELS)), "updates_verified": len(updates), "model": str(model_path), "thresholds": str(thresholds_path), "sample_manifest": str(sample_manifest) if sample_manifest else None}
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def evaluate_frozen_tfidf(data_dir: Path, threshold_path: Path, *, features: str, balanced: bool, seed: int = 13) -> dict[str, Any]:
    """One untouched-test evaluation after development configuration is frozen."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import FeatureUnion
    from sklearn.preprocessing import MultiLabelBinarizer

    train, test = load_split(data_dir, "train"), load_split(data_dir, "test")
    vectorizers: list[tuple[str, Any]] = []
    if features in {"word", "combined"}:
        vectorizers.append(("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=80_000)))
    if features in {"char", "combined"}:
        vectorizers.append(("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2, max_features=100_000)))
    vectorizer = vectorizers[0][1] if len(vectorizers) == 1 else FeatureUnion(vectorizers)
    y_train = MultiLabelBinarizer(classes=list(LABELS)).fit_transform([row.labels for row in train])
    classifier = OneVsRestClassifier(LogisticRegression(solver="liblinear", max_iter=200, class_weight="balanced" if balanced else None, random_state=seed)).fit(vectorizer.fit_transform([row.text for row in train]), y_train)
    thresholds = json.loads(threshold_path.read_text(encoding="utf-8"))["thresholds"]
    predictions, updates = _prima_predictions(_probability_rows(classifier.predict_proba(vectorizer.transform([row.text for row in test]))), thresholds, return_updates=True)
    return {"metrics": evaluate([row.labels for row in test], predictions, list(LABELS)), "updates_verified": len(updates), "features": features, "balanced": balanced, "threshold_artifact": str(threshold_path)}


def _probability_rows(matrix: Any) -> list[dict[str, float]]:
    return [{label: float(value) for label, value in zip(LABELS, row, strict=True)} for row in matrix]


def _prima_predictions(probabilities: list[dict[str, float]], thresholds: dict[str, float], *, return_updates: bool = False):
    predictions, updates = [], []
    for row in probabilities:
        prediction = GoEmotionsDecisionController(thresholds).decide(row, model_id="tfidf-logreg")
        # Independent Reddit examples must not share emotional persistence.
        class Current:
            value = prediction

            def predict(self, text: str) -> EmotionPrediction:
                return self.value
        update = DynamicAffectEngine(classifier=GoEmotionsProfileAdapter(Current())).process("classifier input")
        predictions.append(frozenset(prediction.selected_labels))
        updates.append(update)
    return (predictions, updates) if return_updates else predictions


def _change_analysis(gold: list[frozenset[str]], initial: list[frozenset[str]], final: list[frozenset[str]]) -> dict[str, int]:
    changed = beneficial = harmful = 0
    for truth, before, after in zip(gold, initial, final, strict=True):
        if before != after:
            changed += 1
            beneficial += int(before != truth and after == truth)
            harmful += int(before == truth and after != truth)
    return {"changed": changed, "beneficial": beneficial, "harmful": harmful, "unchanged": len(gold) - changed}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Train or score a PRIMA TF-IDF GoEmotions candidate.")
    parser.add_argument("--train", action="store_true", help="Write model.joblib and thresholds.json to --output-dir.")
    parser.add_argument("--model", type=Path)
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
        result = run_tfidf_logreg(args.data_dir, args.output_dir, features=args.features, balanced=args.balanced, seed=args.seed)
    else:
        if not args.model or not args.thresholds:
            parser.error("--model and --thresholds are required unless --train is used.")
        result = run_frozen_tfidf(args.model, args.thresholds, args.data_dir, args.output_dir, split=args.split, max_samples=args.max_samples, sample_manifest=args.sample_manifest, progress=not args.no_progress)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
