from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from my_capstone.evaluation import (  # noqa: E402
    BASELINE_PREDICTION_POLICY,
    build_prediction_metrics,
    build_workload_metrics,
    rebalanced_prediction_policy,
)
from my_capstone_research.cross_host.context_builder import CrossHostContextBuilder  # noqa: E402


class EvaluationTests(unittest.TestCase):
    def test_research_builder_returns_probability_confidence(self) -> None:
        builder = CrossHostContextBuilder(
            input_size=6,
            output_size=6,
            hidden_size=8,
            max_length=4,
            local_context_length=2,
            companion_context_length=2,
        )
        X = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        y = torch.tensor([[1], [2]], dtype=torch.long)

        confidence, _attention = builder.forward(X)
        self.assertGreaterEqual(float(confidence.detach().min()), 0.0)
        self.assertLessEqual(float(confidence.detach().max()), 1.0)

        query_confidence, _query_attention, _inverse = builder.query(
            X,
            y,
            iterations=0,
            batch_size=2,
            verbose=False,
        )
        self.assertGreaterEqual(float(query_confidence.min()), 0.0)
        self.assertLessEqual(float(query_confidence.max()), 1.0)
        np.testing.assert_allclose(
            query_confidence.detach().numpy().sum(axis=1),
            np.ones(2),
            rtol=1e-5,
        )

    def test_workload_metrics_match_deepcase_manual_formula(self) -> None:
        clusters = np.asarray([0, 0, 0, 1, 1, -1])
        metrics = build_workload_metrics(clusters=clusters, manual_samples_per_cluster=2)
        self.assertEqual(metrics["n_clusters"], 2)
        self.assertEqual(metrics["clustered_sequences"], 5)
        self.assertEqual(metrics["representative_alerts"], 4)
        self.assertAlmostEqual(metrics["coverage"], 5 / 6)
        self.assertAlmostEqual(metrics["overall_reduction"], 1 - (4 + 1) / 6)

    def test_prediction_metrics_separate_reject_from_benign(self) -> None:
        labels = np.asarray([-1, -1, 0, 1, 2])
        predictions = np.asarray([-4, 0, 0, -1, -3])
        metrics = build_prediction_metrics(
            predictions=predictions,
            labels=labels,
            policy=BASELINE_PREDICTION_POLICY,
        )
        self.assertEqual(metrics["auto_decided_count"], 3)
        self.assertEqual(metrics["reject_count"], 2)
        self.assertAlmostEqual(metrics["auto_decided_accuracy"], 2 / 3)

    def test_rebalanced_metrics_compare_binary_truth(self) -> None:
        labels = np.asarray([-1, 3, 9, -1])
        predictions = np.asarray([0, 1, -5, 1])
        metrics = build_prediction_metrics(
            predictions=predictions,
            labels=labels,
            policy=rebalanced_prediction_policy(-5),
        )
        self.assertEqual(metrics["auto_decided_count"], 3)
        self.assertAlmostEqual(metrics["auto_decided_accuracy"], 2 / 3)
        self.assertEqual(metrics["binary_incident_detection"]["tp"], 1)
        self.assertEqual(metrics["binary_incident_detection"]["fp"], 1)


if __name__ == "__main__":
    unittest.main()
