from __future__ import annotations

import numpy as np

from context_selection_experiment.src.association import (
    AssociationModel,
    recency_bands,
    select_by_association,
    select_recency_anchored_association,
)


def toy_model() -> AssociationModel:
    weights = np.zeros((4, 2, 4), dtype=np.float32)
    weights[0, 0, 1] = 3.0
    weights[0, 1, 2] = -2.0
    return AssociationModel(
        weights=weights,
        base_log_odds=np.zeros(4, dtype=np.float32),
        global_to_local=np.arange(4, dtype=np.int16),
        seen_global_events=np.arange(3, dtype=np.int16),
        unknown_id=3,
        bands=np.asarray([1, 1, 0, 0], dtype=np.int8),
        smoothing=1.0,
        maximum_absolute_log_odds=6.0,
    )


def test_recency_bands_cover_oldest_to_newest_columns():
    result = recency_bands(6, [[1, 2], [3, 6]])
    assert result.tolist() == [1, 1, 1, 1, 0, 0]


def test_abs_selector_has_no_label_argument_and_keeps_chronology():
    model = toy_model()
    candidate_rows = np.asarray([[0, 1, 2, 3]], dtype=np.int32)
    event_ids = np.asarray([0, 2, 1, 3], dtype=np.int16)
    target = np.asarray([0], dtype=np.int16)
    selected = select_by_association(
        model, candidate_rows, event_ids, target, budget=2
    )
    # old event 2 has |-2| and recent event 1 has |+3|.
    assert selected.tolist() == [[1, 2]]


def test_class_oracle_changes_direction_but_not_training_weights():
    model = toy_model()
    candidate_rows = np.asarray([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.int32)
    event_ids = np.asarray([0, 2, 1, 3], dtype=np.int16)
    target = np.asarray([0, 0], dtype=np.int16)
    labels = np.asarray([1, 0], dtype=np.int8)
    selected = select_by_association(
        model,
        candidate_rows,
        event_ids,
        target,
        budget=1,
        labels_for_oracle=labels,
    )
    assert selected[:, 0].tolist() == [2, 1]


def test_anchored_selector_keeps_newest_slots_and_has_no_label_argument():
    model = toy_model()
    candidate_rows = np.asarray([[0, 1, 2, 3]], dtype=np.int32)
    event_ids = np.asarray([0, 2, 1, 3], dtype=np.int16)
    target = np.asarray([0], dtype=np.int16)
    selected = select_recency_anchored_association(
        model,
        candidate_rows,
        event_ids,
        target,
        budget=3,
        recency_anchors=2,
    )
    assert selected.tolist() == [[1, 2, 3]]


def test_anchored_selector_preserves_left_padding_for_short_history():
    model = toy_model()
    candidate_rows = np.asarray([[-1, -1, -1, 0]], dtype=np.int32)
    event_ids = np.asarray([0], dtype=np.int16)
    target = np.asarray([0], dtype=np.int16)
    selected = select_recency_anchored_association(
        model,
        candidate_rows,
        event_ids,
        target,
        budget=3,
        recency_anchors=2,
    )
    assert selected.tolist() == [[-1, -1, 0]]
