from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .scoring import (
    BENIGN_CLUSTER_SCORE,
    PREDICT_DISTANCE_EXCEEDED,
    PREDICT_NOT_CONFIDENT,
    PREDICT_UNSEEN_EVENT,
    TRAIN_NO_ATTACK_LABEL,
)

DEFAULT_MANUAL_SAMPLES_PER_CLUSTER = 10


@dataclass(frozen=True)
class PredictionPolicy:
    name: str
    reject_scores: frozenset[int]
    benign_scores: frozenset[int]
    incident_scores: frozenset[int] | None = None
    multiclass_attack_ids: bool = True
    true_benign_label: int = TRAIN_NO_ATTACK_LABEL

    def normalize(self, score: int) -> int | None:
        if score in self.reject_scores:
            return None
        if score in self.benign_scores:
            if self.incident_scores is not None:
                return 0
            return self.true_benign_label
        if self.incident_scores is not None:
            if score in self.incident_scores:
                return 1
            return None
        if score >= 0:
            return score
        return None

    def is_predicted_incident(self, normalized_score: int) -> bool:
        if self.incident_scores is not None:
            return normalized_score == 1
        return normalized_score != self.true_benign_label


BASELINE_PREDICTION_POLICY = PredictionPolicy(
    name="attack_type_cluster_score",
    reject_scores=frozenset(
        {
            PREDICT_NOT_CONFIDENT,
            PREDICT_UNSEEN_EVENT,
            PREDICT_DISTANCE_EXCEEDED,
        }
    ),
    benign_scores=frozenset({BENIGN_CLUSTER_SCORE}),
    incident_scores=None,
    multiclass_attack_ids=True,
)


def rebalanced_prediction_policy(reject_score: int) -> PredictionPolicy:
    return PredictionPolicy(
        name="binary_incident_triage",
        reject_scores=frozenset(
            {
                reject_score,
                PREDICT_NOT_CONFIDENT,
                PREDICT_UNSEEN_EVENT,
                PREDICT_DISTANCE_EXCEEDED,
            }
        ),
        benign_scores=frozenset({0, BENIGN_CLUSTER_SCORE}),
        incident_scores=frozenset({1}),
        multiclass_attack_ids=False,
    )


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _distribution(values: Iterable[int]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(Counter(values).items())}


def build_workload_metrics(
    *,
    clusters: np.ndarray,
    manual_samples_per_cluster: int = DEFAULT_MANUAL_SAMPLES_PER_CLUSTER,
) -> dict[str, int | float]:
    clusters = np.asarray(clusters, dtype=int).reshape(-1)
    total = int(clusters.shape[0])
    clustered_mask = clusters >= 0
    clustered = int(clustered_mask.sum())
    unclustered = total - clustered
    cluster_ids, cluster_counts = np.unique(clusters[clustered_mask], return_counts=True)
    n_clusters = int(cluster_ids.shape[0])

    representative_alerts = int(n_clusters * manual_samples_per_cluster)
    capped_representative_alerts = int(
        sum(min(int(size), manual_samples_per_cluster) for size in cluster_counts)
    )

    return {
        "total_sequences": total,
        "clustered_sequences": clustered,
        "unclustered_sequences": unclustered,
        "n_clusters": n_clusters,
        "manual_samples_per_cluster": int(manual_samples_per_cluster),
        "representative_alerts": representative_alerts,
        "representative_alerts_capped": capped_representative_alerts,
        "coverage": _safe_divide(clustered, total),
        "reduction_on_clustered": 1.0 - _safe_divide(representative_alerts, clustered),
        "overall_reduction": 1.0
        - _safe_divide(representative_alerts + unclustered, total),
        "overall_reduction_capped": 1.0
        - _safe_divide(capped_representative_alerts + unclustered, total),
        "events_per_cluster_all": _safe_divide(total, n_clusters),
        "events_per_cluster_clustered": _safe_divide(clustered, n_clusters),
    }


def build_cluster_quality_metrics(
    *,
    clusters: np.ndarray,
    labels: np.ndarray | None,
) -> dict[str, int | float]:
    if labels is None:
        return {}

    clusters = np.asarray(clusters, dtype=int).reshape(-1)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if clusters.shape != labels.shape:
        return {}

    clustered_mask = clusters >= 0
    clustered = int(clustered_mask.sum())
    if clustered == 0:
        return {
            "cluster_purity": 0.0,
            "mixed_cluster_count": 0,
        }

    majority_total = 0
    mixed_clusters = 0
    for cluster in np.unique(clusters[clustered_mask]):
        cluster_labels = labels[clusters == cluster]
        counts = Counter(int(value) for value in cluster_labels.tolist())
        majority_total += max(counts.values())
        if len(counts) > 1:
            mixed_clusters += 1

    return {
        "cluster_purity": _safe_divide(majority_total, clustered),
        "mixed_cluster_count": int(mixed_clusters),
    }


def build_prediction_metrics(
    *,
    predictions: np.ndarray,
    labels: np.ndarray | None,
    policy: PredictionPolicy = BASELINE_PREDICTION_POLICY,
) -> dict[str, object]:
    predictions = np.asarray(predictions, dtype=int).reshape(-1)
    total = int(predictions.shape[0])
    result: dict[str, object] = {
        "policy": policy.name,
        "total_sequences": total,
        "score_distribution": _distribution(int(value) for value in predictions.tolist()),
    }

    normalized: list[int | None] = [policy.normalize(int(score)) for score in predictions.tolist()]
    reject_mask = np.asarray([value is None for value in normalized], dtype=bool)
    auto_count = int((~reject_mask).sum())
    reject_count = int(reject_mask.sum())
    result.update(
        {
            "auto_decided_count": auto_count,
            "reject_count": reject_count,
            "auto_decided_rate": _safe_divide(auto_count, total),
            "reject_rate": _safe_divide(reject_count, total),
            "reject_score_distribution": _distribution(
                int(score) for score, rejected in zip(predictions.tolist(), reject_mask.tolist()) if rejected
            ),
        }
    )

    if labels is None:
        return result

    labels = np.asarray(labels, dtype=int).reshape(-1)
    if labels.shape != predictions.shape:
        result["label_shape_mismatch"] = True
        return result

    if policy.incident_scores is not None:
        comparable_labels = np.where(labels == policy.true_benign_label, 0, 1)
    else:
        comparable_labels = labels

    auto_indices = np.flatnonzero(~reject_mask)
    rejected_indices = np.flatnonzero(reject_mask)
    normalized_auto = [int(normalized[index]) for index in auto_indices]
    true_auto = comparable_labels[auto_indices]

    strict_correct = sum(
        int(normalized_score == int(label))
        for normalized_score, label in zip(normalized, comparable_labels.tolist())
        if normalized_score is not None
    )
    result["strict_accuracy_rejects_wrong"] = _safe_divide(strict_correct, total)

    if auto_count > 0:
        auto_correct = sum(
            int(pred == int(label)) for pred, label in zip(normalized_auto, true_auto.tolist())
        )
        result["auto_decided_accuracy"] = _safe_divide(auto_correct, auto_count)

    true_incident = labels != policy.true_benign_label
    auto_true_incident = true_incident[auto_indices]
    auto_pred_incident = np.asarray(
        [policy.is_predicted_incident(score) for score in normalized_auto],
        dtype=bool,
    )
    tp = int(np.logical_and(auto_true_incident, auto_pred_incident).sum())
    fp = int(np.logical_and(~auto_true_incident, auto_pred_incident).sum())
    tn = int(np.logical_and(~auto_true_incident, ~auto_pred_incident).sum())
    fn = int(np.logical_and(auto_true_incident, ~auto_pred_incident).sum())
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    result["binary_incident_detection"] = {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": _safe_divide(tp + tn, auto_count),
    }

    total_incidents = int(true_incident.sum())
    rejected_incidents = int(true_incident[rejected_indices].sum())
    total_benign = int((~true_incident).sum())
    rejected_benign = reject_count - rejected_incidents
    result["reject_by_truth"] = {
        "rejected_incidents": rejected_incidents,
        "rejected_benign": rejected_benign,
        "incident_reject_rate": _safe_divide(rejected_incidents, total_incidents),
        "benign_reject_rate": _safe_divide(rejected_benign, total_benign),
    }

    if policy.multiclass_attack_ids and auto_count > 0:
        classes = sorted(set(labels.tolist()) | {score for score in normalized_auto})
        per_class: dict[str, dict[str, float | int]] = {}
        f1_sum = 0.0
        weighted_f1_sum = 0.0
        support_sum = 0
        for cls in classes:
            true_is_cls = true_auto == cls
            pred_is_cls = np.asarray([score == cls for score in normalized_auto], dtype=bool)
            class_tp = int(np.logical_and(true_is_cls, pred_is_cls).sum())
            class_fp = int(np.logical_and(~true_is_cls, pred_is_cls).sum())
            class_fn = int(np.logical_and(true_is_cls, ~pred_is_cls).sum())
            class_precision = _safe_divide(class_tp, class_tp + class_fp)
            class_recall = _safe_divide(class_tp, class_tp + class_fn)
            class_f1 = _safe_divide(
                2 * class_precision * class_recall,
                class_precision + class_recall,
            )
            support = int(true_is_cls.sum())
            per_class[str(int(cls))] = {
                "precision": class_precision,
                "recall": class_recall,
                "f1": class_f1,
                "support": support,
            }
            f1_sum += class_f1
            weighted_f1_sum += class_f1 * support
            support_sum += support

        result["multiclass_attack_type"] = {
            "macro_f1": _safe_divide(f1_sum, len(classes)),
            "weighted_f1": _safe_divide(weighted_f1_sum, support_sum),
            "per_class": per_class,
        }

    return result


def build_run_evaluation(
    *,
    clusters: np.ndarray | None = None,
    train_labels: np.ndarray | None = None,
    predictions: np.ndarray | None = None,
    test_labels: np.ndarray | None = None,
    prediction_policy: PredictionPolicy = BASELINE_PREDICTION_POLICY,
    manual_samples_per_cluster: int = DEFAULT_MANUAL_SAMPLES_PER_CLUSTER,
) -> dict[str, object]:
    result: dict[str, object] = {}
    if clusters is not None:
        result["workload_reduction"] = build_workload_metrics(
            clusters=clusters,
            manual_samples_per_cluster=manual_samples_per_cluster,
        )
        quality = build_cluster_quality_metrics(clusters=clusters, labels=train_labels)
        if quality:
            result["cluster_quality"] = quality
    if predictions is not None:
        result["prediction"] = build_prediction_metrics(
            predictions=predictions,
            labels=test_labels,
            policy=prediction_policy,
        )
    return result
