"""Small dependency-light binary classification and segmentation metrics."""

from __future__ import annotations

import numpy as np


def _arrays(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.int8).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(y) != len(s) or not len(y):
        raise ValueError("labels and scores must be non-empty and equally sized")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(s).all():
        raise ValueError("labels must be binary and scores finite")
    return y, s


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    y, s = _arrays(labels, scores)
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC requires both classes")
    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    start = 0
    while start < len(s):
        end = start + 1
        while end < len(s) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    positive_rank_sum = float(ranks[y == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def binary_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    y, s = _arrays(labels, scores)
    positives = int(y.sum())
    if positives == 0:
        raise ValueError("average precision requires positive samples")
    order = np.argsort(-s, kind="mergesort")
    sorted_labels = y[order]
    cumulative = np.cumsum(sorted_labels)
    precision = cumulative / np.arange(1, len(y) + 1)
    return float(precision[sorted_labels == 1].sum() / positives)


def binary_metrics_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    y, s = _arrays(labels, scores)
    predicted = s >= threshold
    tp = int(np.sum(predicted & (y == 1)))
    fp = int(np.sum(predicted & (y == 0)))
    fn = int(np.sum(~predicted & (y == 1)))
    tn = int(np.sum(~predicted & (y == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray, *, max_candidates: int = 512) -> tuple[float, dict[str, float | int]]:
    y, s = _arrays(labels, scores)
    unique = np.unique(s)
    if len(unique) > max_candidates:
        quantiles = np.linspace(0.0, 1.0, max_candidates)
        candidates = np.unique(np.quantile(unique, quantiles))
    else:
        candidates = unique
    best_threshold = float(candidates[0])
    best_metrics = binary_metrics_at_threshold(y, s, best_threshold)
    for candidate in candidates[1:]:
        metrics = binary_metrics_at_threshold(y, s, float(candidate))
        if (metrics["f1"], metrics["recall"], -metrics["fp"], candidate) > (
            best_metrics["f1"],
            best_metrics["recall"],
            -best_metrics["fp"],
            best_threshold,
        ):
            best_threshold = float(candidate)
            best_metrics = metrics
    return best_threshold, best_metrics
