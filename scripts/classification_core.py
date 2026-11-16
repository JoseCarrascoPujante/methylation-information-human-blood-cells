"""Training-only variance ranking, nearest-centroid rules and empirical metrics."""
from __future__ import annotations

import numpy as np


class UnresolvedPrediction(ValueError):
    """A prediction edge case for which the scientific protocol has no rule."""

    def __init__(self, reason: str, details: dict):
        super().__init__(reason)
        self.reason = reason
        self.details = details


def rank_and_centroids(beta, training_indices, population_labels, classes, probe_ids):
    """Fit R and class means using exactly the explicitly indexed training data."""
    training_indices = np.asarray(training_indices, dtype=np.int64)
    labels = np.asarray(population_labels)
    if len(np.unique(training_indices)) != len(training_indices):
        raise ValueError("Training indices must be unique.")
    if set(labels[training_indices]) != set(classes):
        raise ValueError("Training labels do not contain the specified classes.")
    means, variances = [], []
    for cell_type in classes:
        columns = training_indices[labels[training_indices] == cell_type]
        if len(columns) < 2:
            raise ValueError("Sample variance requires at least two training donors per class.")
        values = np.asarray(beta[:, columns], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite training beta value.")
        means.append(values.mean(axis=1))
        variances.append(values.var(axis=1, ddof=1))
    centroids = np.column_stack(means)
    within = np.column_stack(variances).mean(axis=1)
    between = centroids.var(axis=1, ddof=1)
    denominator = within + between
    scores = np.divide(between, denominator, out=np.zeros_like(between), where=denominator > 0)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite training ranking score.")
    ids = np.asarray(probe_ids, dtype=str)
    if len(ids) != beta.shape[0]:
        raise ValueError("Probe IDs do not align with the training matrix.")
    order = np.lexsort((ids, -scores))
    return order, scores, centroids


def distances(sample_beta, centroids, method: str):
    """Return sample-by-class distances; inputs have signature features in rows."""
    x = np.asarray(sample_beta, dtype=np.float64)
    mu = np.asarray(centroids, dtype=np.float64)
    if x.ndim != 2 or mu.ndim != 2 or x.shape[0] != mu.shape[0] or x.shape[0] < 1:
        raise ValueError("Samples and centroids must share a nonempty feature axis.")
    if not np.isfinite(x).all() or not np.isfinite(mu).all():
        raise ValueError("Nonfinite classifier input.")
    if method == "euclidean":
        # Squared Euclidean distance has exactly the same nearest-centroid rule.
        return np.square(x[:, :, None] - mu[:, None, :]).sum(axis=0)
    if method == "correlation":
        constant_x = np.all(x == x[:1, :], axis=0)
        constant_mu = np.all(mu == mu[:1, :], axis=0)
        if np.any(constant_x) or np.any(constant_mu):
            raise UnresolvedPrediction("undefined_correlation", {
                "constant_sample_positions": np.flatnonzero(constant_x).tolist(),
                "constant_centroid_positions": np.flatnonzero(constant_mu).tolist(),
            })
        xc = x - x.mean(axis=0, keepdims=True)
        mc = mu - mu.mean(axis=0, keepdims=True)
        xn = np.sqrt(np.square(xc).sum(axis=0))
        mn = np.sqrt(np.square(mc).sum(axis=0))
        if np.any(xn == 0) or np.any(mn == 0):
            raise UnresolvedPrediction("undefined_correlation", {
                "constant_sample_positions": np.flatnonzero(xn == 0).tolist(),
                "constant_centroid_positions": np.flatnonzero(mn == 0).tolist(),
            })
        correlation = (xc.T @ mc) / np.outer(xn, mn)
        return 1.0 - np.clip(correlation, -1.0, 1.0)
    raise ValueError(f"Unknown distance rule: {method}")


def classify(sample_beta, centroids, classes, method: str):
    cost = distances(sample_beta, centroids, method)
    minimum = cost.min(axis=1)
    tied = (cost == minimum[:, None]).sum(axis=1) > 1
    if np.any(tied):
        locations = np.flatnonzero(tied)
        raise UnresolvedPrediction("exact_class_distance_tie", {
            "sample_positions": locations.tolist(),
            "class_order": list(classes), "distance_rows": cost[locations].tolist(),
        })
    selected = cost.argmin(axis=1)
    ordered = np.sort(cost, axis=1)
    margin = ordered[:, 1] - ordered[:, 0]
    return np.asarray(classes)[selected], cost, margin


def choose_size(correct_counts: dict[int, int]) -> int:
    """The balanced inner folds reduce selection to an exact integer comparison."""
    best = max(correct_counts.values())
    return min(size for size, count in correct_counts.items() if count == best)


def empirical_metrics(true, predicted, classes):
    classes = list(classes)
    lookup = {name: index for index, name in enumerate(classes)}
    if len(true) != len(predicted) or not len(true):
        raise ValueError("True and predicted labels must have equal nonzero lengths.")
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for observed, prediction in zip(true, predicted):
        if observed not in lookup or prediction not in lookup:
            raise ValueError("A prediction or true label is outside the evaluation classes.")
        confusion[lookup[observed], lookup[prediction]] += 1
    row_counts = confusion.sum(axis=1)
    if np.any(row_counts == 0):
        raise ValueError("Every specified true class must be represented.")
    count = int(confusion.sum())
    correct = int(np.trace(confusion))
    joint = confusion.astype(np.float64) / count
    p_true, p_pred = joint.sum(axis=1), joint.sum(axis=0)
    denominator = np.outer(p_true, p_pred)
    nonzero = joint > 0
    mi = float(np.sum(joint[nonzero] * np.log2(joint[nonzero] / denominator[nonzero])))
    entropy = float(-np.sum(p_true[p_true > 0] * np.log2(p_true[p_true > 0])))
    return {
        "n_samples": count, "n_cell_types": len(classes), "correct": correct,
        "accuracy": correct / count,
        "balanced_accuracy": float(np.mean(np.diag(confusion) / row_counts)),
        "empirical_mutual_information_bits": mi, "true_label_entropy_bits": entropy,
    }, confusion
