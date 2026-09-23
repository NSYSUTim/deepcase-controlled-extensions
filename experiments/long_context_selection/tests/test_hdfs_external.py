from __future__ import annotations

import numpy as np
import pytest

from context_selection_experiment.src.hdfs_external import (
    line_bucket,
    parse_event_line,
    preceding_windows,
    safety_limit_per_1000,
    split_code,
)


def test_hash_split_is_duplicate_safe_and_newline_invariant():
    line = b"1 2 3"
    assert line_bucket(line) == line_bucket(line + b"\r\n")
    assert split_code("hdfs_test_normal", line) == split_code(
        "hdfs_test_abnormal", line
    )
    assert split_code("hdfs_train", line) == 0


def test_parser_rejects_noninteger_and_empty_sequences():
    assert parse_event_line(b"1  2 3\n").tolist() == [1, 2, 3]
    with pytest.raises(ValueError):
        parse_event_line(b"1 nope 3")
    with pytest.raises(ValueError):
        parse_event_line(b"\n")


def test_preceding_windows_are_strict_and_left_padded():
    result = preceding_windows(np.asarray([1, 2, 3]), width=2, pad_id=9)
    assert result.tolist() == [[9, 9], [9, 1], [1, 2]]


def test_hybrid_safety_limit_handles_zero_without_division():
    assert safety_limit_per_1000(0.0) == 1.0
    assert safety_limit_per_1000(0.5) == 1.5
    assert safety_limit_per_1000(2.0) == pytest.approx(2.1)
