from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from my_capstone.scoring import BENIGN_CLUSTER_SCORE, build_cluster_scores


class ScoringTests(unittest.TestCase):
    def test_build_cluster_scores_uses_benign_default_for_unlabeled_cluster(self) -> None:
        clusters = np.array([0, 0, 1, 1, -1])
        labels = np.array([-1, -1, 2, -1, -1])

        scores = build_cluster_scores(clusters=clusters, labels=labels)

        self.assertEqual(
            scores.tolist(),
            [BENIGN_CLUSTER_SCORE, BENIGN_CLUSTER_SCORE, 2, 2, -1],
        )


if __name__ == "__main__":
    unittest.main()
