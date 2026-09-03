from __future__ import annotations

import math
from collections import defaultdict


def binary_metrics(labels: list[int], predictions: list[int]) -> dict[str, float | int]:
    if len(labels) != len(predictions) or not labels:
        raise ValueError("labels/predictions must be aligned and non-empty")
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions, strict=True))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions, strict=True))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions, strict=True))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions, strict=True))
    total = len(labels)
    def div(a: int | float, b: int | float) -> float:
        return float(a / b) if b else 0.0

    precision = div(tp, tp + fp)
    recall = div(tp, tp + fn)
    specificity = div(tn, tn + fp)
    return {
        "images": total,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": div(tp + tn, total),
        "precision": precision,
        "recall": recall,
        "f1": div(2 * precision * recall, precision + recall),
        "fpr": div(fp, fp + tn),
        "fnr": div(fn, fn + tp),
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2.0,
    }


def per_defect_type(rows: list[dict]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if int(row["label"]) == 1:
            grouped[str(row["defect_type"])].append(row)
    output = {}
    for name, values in sorted(grouped.items()):
        detected = sum(bool(v["prediction"]) for v in values)
        output[name] = {"count": len(values), "detected": detected, "recall": detected / len(values) if values else 0.0}
    return output


def _score_summary(values: list[float]) -> dict[str, float]:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("finite non-empty scores are required")
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        index = min(
            len(ordered) - 1,
            max(0, round((len(ordered) - 1) * fraction)),
        )
        return float(ordered[index])

    return {
        "min": float(ordered[0]),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": float(ordered[-1]),
        "mean": float(sum(ordered) / len(ordered)),
    }


def operating_point_summary(rows: list[dict]) -> dict[str, object]:
    """Summarize branch scores, thresholds, decisions, and fusion parameters."""

    if not rows:
        raise ValueError("at least one prediction row is required")
    labels = [int(row["label"]) for row in rows]
    branches: dict[str, object] = {}
    for branch in ("general", "local", "known"):
        thresholds = {float(row[f"{branch}_threshold"]) for row in rows}
        if len(thresholds) != 1:
            raise ValueError(f"{branch} threshold changed within one category")
        scores = [float(row[f"{branch}_score"]) for row in rows]
        branches[branch] = {
            "threshold": thresholds.pop(),
            "score_summary": {
                "normal": _score_summary(
                    [score for score, label in zip(scores, labels, strict=True) if label == 0]
                ),
                "defect": _score_summary(
                    [score for score, label in zip(scores, labels, strict=True) if label == 1]
                ),
            },
            "decision_metrics": binary_metrics(
                labels, [int(bool(row[f"{branch}_decision"])) for row in rows]
            ),
        }
    modes = {str(row["fusion_mode"]) for row in rows}
    thresholds = {float(row["fusion_threshold"]) for row in rows}
    weight_tuples = {
        tuple(float(row["fusion_weights"][name]) for name in ("general", "local", "known"))
        for row in rows
    }
    if len(modes) != 1 or len(thresholds) != 1 or len(weight_tuples) != 1:
        raise ValueError("fusion parameters changed within one category")
    weights = weight_tuples.pop()
    return {
        "branches": branches,
        "fusion": {
            "mode": modes.pop(),
            "threshold": thresholds.pop(),
            "weights": dict(zip(("general", "local", "known"), weights, strict=True)),
            "score_summary": {
                "normal": _score_summary(
                    [float(row["fusion_score"]) for row in rows if int(row["label"]) == 0]
                ),
                "defect": _score_summary(
                    [float(row["fusion_score"]) for row in rows if int(row["label"]) == 1]
                ),
            },
        },
    }
