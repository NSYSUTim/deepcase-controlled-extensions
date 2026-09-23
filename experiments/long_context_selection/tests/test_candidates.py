from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from context_selection_experiment.prepare_candidates import build_candidates


ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


def test_candidates_are_strictly_past_and_machine_local():
    frame = pd.DataFrame(
        {
            "timestamp": [1, 1, 2, 3, 2],
            "machine": ["a", "a", "a", "a", "b"],
        }
    )
    path = ARTIFACTS / "test_rows_machine.npy"
    build_candidates(frame, path, cap=3, timeout=10)
    rows = np.load(path)
    assert rows[0].tolist() == [-1, -1, -1]
    assert rows[1].tolist() == [-1, -1, -1]
    assert rows[2].tolist() == [-1, 0, 1]
    assert rows[3].tolist() == [0, 1, 2]
    assert rows[4].tolist() == [-1, -1, -1]


def test_timeout_excludes_old_rows():
    frame = pd.DataFrame({"timestamp": [1, 4], "machine": ["a", "a"]})
    path = ARTIFACTS / "test_rows_timeout.npy"
    build_candidates(frame, path, cap=2, timeout=2)
    rows = np.load(path)
    assert rows[1].tolist() == [-1, -1]
