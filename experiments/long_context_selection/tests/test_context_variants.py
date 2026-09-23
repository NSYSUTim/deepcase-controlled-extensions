from __future__ import annotations

import numpy as np

from context_selection_experiment.prepare_context_variants import (
    event_ids_from_candidate_rows,
    oracle_context,
)


def test_candidate_rows_map_to_event_ids_with_left_padding():
    candidates = np.asarray([[-1, 0, 2], [1, 2, 3]], dtype=np.int32)
    events = np.asarray([7, 8, 9, 10], dtype=np.int16)
    result = event_ids_from_candidate_rows(candidates, events, pad_id=11)
    assert result.tolist() == [[11, 7, 9], [8, 9, 10]]


def test_oracle_uses_predecessor_labels_and_preserves_chronology():
    # Candidate row 4 is the current target only conceptually; changing a
    # separate target label cannot affect this function because none is passed.
    candidates = np.asarray([[0, 1, 2, 3]], dtype=np.int32)
    events = np.asarray([10, 11, 12, 13, 14], dtype=np.int16)
    predecessor_labels = np.asarray([1, 0, 1, 0, 1], dtype=np.int8)
    result = oracle_context(
        candidates, events, predecessor_labels, pad_id=15, budget=3
    )
    # Both incident predecessors (rows 0 and 2), then the most-recent filler
    # (row 3); output is chronological rather than priority ordered.
    assert result.tolist() == [[10, 12, 13]]


def test_oracle_keeps_most_recent_incidents_when_budget_is_full():
    candidates = np.asarray([[0, 1, 2, 3]], dtype=np.int32)
    events = np.asarray([10, 11, 12, 13], dtype=np.int16)
    labels = np.ones(4, dtype=np.int8)
    result = oracle_context(candidates, events, labels, pad_id=14, budget=2)
    assert result.tolist() == [[12, 13]]
