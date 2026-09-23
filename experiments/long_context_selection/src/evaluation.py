"""Fail-closed metrics for three-way DeepCASE policy outputs."""

from __future__ import annotations

import math

import numpy as np


def _divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _class_f1(truth: np.ndarray, decision: np.ndarray, positive: int) -> float:
    predicted = decision == positive
    actual = truth == positive
    tp = int(np.sum(predicted & actual))
    fp = int(np.sum(predicted & ~actual))
    fn = int(np.sum(~predicted & actual))
    return _divide(2 * tp, 2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def strict_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    incident_groups: np.ndarray,
) -> dict:
    """Score Reject as unresolved, never as a successful incident detection."""

    truth = np.asarray(truth, dtype=np.int8)
    prediction = np.asarray(prediction, dtype=float)
    groups = np.asarray(incident_groups, dtype=object)
    if truth.shape != prediction.shape or truth.shape != groups.shape:
        raise ValueError("truth, prediction and group arrays must align")
    if not np.isin(truth, [0, 1]).all():
        raise ValueError("truth must be binary")

    auto_incident = prediction > 0
    auto_nonincident = prediction == 0
    reject = prediction < 0
    review = auto_incident | reject
    decision = np.full(len(truth), -1, dtype=np.int8)
    decision[auto_nonincident] = 0
    decision[auto_incident] = 1
    incident = truth == 1
    nonincident = ~incident

    auto_tp = int(np.sum(auto_incident & incident))
    auto_fp = int(np.sum(auto_incident & nonincident))
    auto_tn = int(np.sum(auto_nonincident & nonincident))
    auto_fn = int(np.sum(auto_nonincident & incident))
    rejected_incident = int(np.sum(reject & incident))
    rejected_nonincident = int(np.sum(reject & nonincident))

    accepted = ~reject
    accepted_error = (
        float(np.mean(decision[accepted] != truth[accepted]))
        if accepted.any()
        else float("nan")
    )
    valid_group = incident & np.asarray(
        [value is not None and str(value) not in {"nan", "<NA>"} for value in groups]
    )
    unique_groups = np.unique(groups[valid_group])
    detected_groups = sum(
        int(np.any(auto_incident[valid_group][groups[valid_group] == group]))
        for group in unique_groups
    )
    macro_f1 = 0.5 * (
        _class_f1(truth, decision, 0) + _class_f1(truth, decision, 1)
    )
    result = {
        "alerts": int(len(truth)),
        "incidents": int(incident.sum()),
        "nonincidents": int(nonincident.sum()),
        "automatic_incident_recall": _divide(auto_tp, int(incident.sum())),
        "automatic_incident_precision": _divide(auto_tp, auto_tp + auto_fp),
        "strict_macro_f1": float(macro_f1),
        "review_workload": float(review.mean()),
        "automatic_incident_rate": float(auto_incident.mean()),
        "automatic_nonincident_rate": float(auto_nonincident.mean()),
        "reject_rate": float(reject.mean()),
        "accepted_decision_error": accepted_error,
        "false_escalations_per_1000_nonincidents": 1000.0
        * _divide(auto_fp, int(nonincident.sum())),
        "incident_group_automatic_recall": _divide(detected_groups, len(unique_groups)),
        "incident_groups": int(len(unique_groups)),
        "auto_true_incident": auto_tp,
        "auto_false_incident": auto_fp,
        "auto_true_nonincident": auto_tn,
        "auto_false_nonincident": auto_fn,
        "rejected_incident": rejected_incident,
        "rejected_nonincident": rejected_nonincident,
        "reject_low_confidence": int(np.sum(prediction == -1)),
        "reject_unknown_event": int(np.sum(prediction == -2)),
        "reject_distance": int(np.sum(prediction == -3)),
        "reject_other": int(np.sum(prediction < -3)),
    }
    return {
        key: (None if isinstance(value, float) and not math.isfinite(value) else value)
        for key, value in result.items()
    }


def choose_workload_matched_point(points: list[dict], target_workload: float) -> int:
    """Select with workload only; outcomes cannot affect the operating point."""

    if not points:
        raise ValueError("at least one operating point is required")
    ranked = []
    for order, point in enumerate(points):
        workload = float(point["test"]["review_workload"])
        ranked.append((abs(workload - target_workload), workload, order))
    return min(ranked)[2]
