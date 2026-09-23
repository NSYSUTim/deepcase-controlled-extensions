from __future__ import annotations

import numpy as np

from context_selection_experiment.src.evaluation import (
    choose_workload_matched_point,
    strict_metrics,
)


def test_reject_never_counts_as_an_automatically_detected_incident():
    truth = np.asarray([1, 1, 0, 0], dtype=np.int8)
    prediction = np.asarray([1, -1, 0, -3], dtype=float)
    groups = np.asarray(["a", "b", None, None], dtype=object)
    result = strict_metrics(truth, prediction, groups)
    assert result["automatic_incident_recall"] == 0.5
    assert result["review_workload"] == 0.75
    assert result["rejected_incident"] == 1
    assert result["incident_group_automatic_recall"] == 0.5


def test_workload_match_ignores_outcomes_and_has_frozen_ties():
    points = [
        {"test": {"review_workload": 0.6, "automatic_incident_recall": 0.0}},
        {"test": {"review_workload": 0.4, "automatic_incident_recall": 1.0}},
    ]
    # Equal absolute differences: lower workload wins even though its outcome
    # happens to be higher. Reversing outcomes would not change the choice.
    assert choose_workload_matched_point(points, 0.5) == 1
