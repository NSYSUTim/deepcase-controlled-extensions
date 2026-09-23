"""Operational metrics with explicit DeepCASE Reject semantics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class OperationalMetrics:
    alerts: int
    incidents: int
    analyst_workload: float
    triage_incident_recall: float
    triage_precision: float
    relaxed_f1: float
    reject_rate: float
    automation_coverage: float
    accepted_error_rate: float
    classified_macro_f1: float
    incident_group_recall: float
    incident_groups: int
    auto_incident_rate: float
    reject_low_confidence_rate: float
    reject_unknown_event_rate: float
    reject_distance_rate: float


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _binary_f1(y_true: np.ndarray, y_pred: np.ndarray, positive: int) -> float:
    true_positive = np.sum((y_true == positive) & (y_pred == positive))
    false_positive = np.sum((y_true != positive) & (y_pred == positive))
    false_negative = np.sum((y_true == positive) & (y_pred != positive))
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    if not np.isfinite(precision) or not np.isfinite(recall) or precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def operational_metrics(
    y_true: np.ndarray,
    prediction: np.ndarray,
    incident_group: np.ndarray,
) -> OperationalMetrics:
    """Score final triage decisions, not the training-only incident head.

    DeepCASE score > 0 is Incident, score == 0 is Non-Incident, and every
    negative code is Reject. Incident and Reject alerts consume analyst review;
    Non-Incident is the only automatically dismissed decision.
    """

    y_true = np.asarray(y_true, dtype=np.int8)
    prediction = np.asarray(prediction, dtype=float)
    if y_true.shape != prediction.shape:
        raise ValueError("Truth and prediction shapes differ")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Ground truth must be binary")
    auto_incident = prediction > 0
    auto_nonincident = prediction == 0
    reject = prediction < 0
    review = auto_incident | reject
    accepted = ~reject

    tp = int(np.sum(review & (y_true == 1)))
    fp = int(np.sum(review & (y_true == 0)))
    fn = int(np.sum(auto_nonincident & (y_true == 1)))
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    relaxed_f1 = (
        0.0
        if not np.isfinite(precision) or not np.isfinite(recall) or precision + recall == 0
        else 2.0 * precision * recall / (precision + recall)
    )

    if accepted.any():
        accepted_truth = y_true[accepted]
        accepted_prediction = auto_incident[accepted].astype(np.int8)
        accepted_error = float(np.mean(accepted_truth != accepted_prediction))
        macro_f1 = 0.5 * (
            _binary_f1(accepted_truth, accepted_prediction, 0)
            + _binary_f1(accepted_truth, accepted_prediction, 1)
        )
    else:
        accepted_error = float("nan")
        macro_f1 = float("nan")

    incident_mask = y_true == 1
    valid_groups = np.asarray(incident_group, dtype=object)[incident_mask]
    group_review = review[incident_mask]
    groups = np.unique(valid_groups[pd_not_null(valid_groups)])
    detected = 0
    for group in groups:
        detected += int(np.any(group_review[valid_groups == group]))

    return OperationalMetrics(
        alerts=int(len(y_true)),
        incidents=int(incident_mask.sum()),
        analyst_workload=float(review.mean()),
        triage_incident_recall=recall,
        triage_precision=precision,
        relaxed_f1=float(relaxed_f1),
        reject_rate=float(reject.mean()),
        automation_coverage=float(accepted.mean()),
        accepted_error_rate=accepted_error,
        classified_macro_f1=float(macro_f1),
        incident_group_recall=_safe_divide(detected, len(groups)),
        incident_groups=int(len(groups)),
        auto_incident_rate=float(auto_incident.mean()),
        reject_low_confidence_rate=float(np.mean(prediction == -1)),
        reject_unknown_event_rate=float(np.mean(prediction == -2)),
        reject_distance_rate=float(np.mean(prediction == -3)),
    )


def pd_not_null(values: np.ndarray) -> np.ndarray:
    return np.asarray([value is not None and str(value) not in {"nan", "<NA>"} for value in values])


def metrics_dict(*args, **kwargs) -> dict:
    return asdict(operational_metrics(*args, **kwargs))

