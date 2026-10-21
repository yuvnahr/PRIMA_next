"""Frozen pre-B2 scalar statistic and sampling from e0dd105, independent of new helpers."""

import random


def _sample_f1(gold, predicted):
    scores = []
    for truth, guess in zip(gold, predicted, strict=True):
        tp, fp, fn = len(truth & guess), len(guess - truth), len(truth - guess)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def old_bootstrap(gold, baseline, final, *, seed=13, samples=1000):
    if not len(gold) == len(baseline) == len(final):
        raise ValueError("Paired bootstrap inputs must have equal lengths.")
    if not gold or samples < 1:
        return {"samples": samples, "delta": 0.0, "lower": 0.0, "upper": 0.0}
    generator = random.Random(seed)  # noqa: S311 - frozen deterministic reference
    deltas = []
    for _ in range(samples):
        indices = [generator.randrange(len(gold)) for _ in gold]
        before = _sample_f1([gold[i] for i in indices], [baseline[i] for i in indices])
        after = _sample_f1([gold[i] for i in indices], [final[i] for i in indices])
        deltas.append(after - before)
    deltas.sort()
    return {
        "samples": samples,
        "delta": _sample_f1(gold, final) - _sample_f1(gold, baseline),
        "lower": deltas[int(0.025 * (samples - 1))], "upper": deltas[int(0.975 * (samples - 1))],
    }
