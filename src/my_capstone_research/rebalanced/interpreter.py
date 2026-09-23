from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp
from deepcase.interpreter import Interpreter
from deepcase.interpreter.utils import group_by, sp_unique, unique_2d

from ..common.execution import REBALANCED_REJECT_SCORE
from .calibration import calibrated_incident_posterior, triage_decision
from .config import RebalancedConfig


class RebalancedInterpreter(Interpreter):
    """在原版 Interpreter 上加入 calibrated posterior 與 triage decision。"""

    def __init__(
        self,
        context_builder,
        features: int,
        *,
        eps: float = 0.1,
        min_samples: int = 5,
        threshold: float = 0.2,
        low_threshold: float = 0.25,
        high_threshold: float = 0.70,
        distance_scale: float | None = None,
    ) -> None:
        super().__init__(
            context_builder=context_builder,
            features=features,
            eps=eps,
            min_samples=min_samples,
            threshold=threshold,
        )
        self.low_threshold = float(low_threshold)
        self.high_threshold = float(high_threshold)
        self.distance_scale = float(distance_scale if distance_scale is not None else eps)
        self.cluster_priors: dict[int, dict[int, float]] = {}

    def assign_cluster_priors(self, priors: np.ndarray) -> None:
        priors = np.asarray(priors, dtype=float)
        if priors.shape != self.clusters.shape:
            raise ValueError("cluster priors 的 shape 必須和 clusters 一致。")

        self.cluster_priors = {}
        clustered_events = group_by(self.events[self.clusters != -1])
        for event, indices in clustered_events:
            vectors = self.vectors[indices]
            _vectors, inverse, _counts = sp_unique(vectors)
            if event not in self.tree:
                continue

            self.cluster_priors[event] = {}
            _data, index_tree, _indptr, _node_data = self.tree[event].get_arrays()
            _, index_vector = zip(*group_by(inverse))
            event_priors = priors[indices]

            for tree_index, mapping in zip(index_tree, index_vector):
                self.cluster_priors[event][tree_index] = float(np.max(event_priors[mapping]))

    def predict(self, X, y, iterations=100, batch_size=1024, verbose=False):
        X_unique, y_unique, inverse_result = unique_2d(X, y)
        confidence_matrix, attention, _inverse = self.context_builder.query(
            X=X_unique,
            y=y_unique,
            iterations=iterations,
            batch_size=batch_size,
            verbose=verbose,
        )
        y_flat = y_unique.squeeze(1)
        confidence = confidence_matrix[np.arange(y_flat.shape[0]), y_flat]
        mask = confidence >= self.threshold

        vectors = self.vectorize(
            X=X_unique[mask],
            attention=attention[mask],
            size=self.features,
        )
        vectors = np.round(vectors, decimals=4)

        result = np.full(X_unique.shape[0], -1, dtype=float)
        masked_indices = np.flatnonzero(mask.cpu().numpy())
        event_groups = group_by(y_unique[mask].squeeze(1).cpu().numpy())

        for event, local_indices in event_groups:
            unique_indices = masked_indices[local_indices]
            if event not in self.tree:
                result[unique_indices] = -2
                continue

            vectors_event = vectors[local_indices]
            vectors_unique, inverse_vector, _counts = sp_unique(vectors_event)
            distances, neighbours = self.tree[event].query(
                X=vectors_unique.toarray(),
                return_distance=True,
                dualtree=vectors_unique.shape[0] >= 1000,
            )
            neighbours = self.tree[event].get_arrays()[1][neighbours][:, 0]

            unique_target_confidences = np.zeros(vectors_unique.shape[0], dtype=float)
            for vector_id, mapping in group_by(inverse_vector):
                vector_indices = unique_indices[mapping]
                unique_target_confidences[int(vector_id)] = float(
                    np.max(confidence[vector_indices].detach().cpu().numpy())
                )

            unique_scores = np.full(vectors_unique.shape[0], REBALANCED_REJECT_SCORE, dtype=float)
            for row_index, neighbour in enumerate(neighbours):
                distance = float(distances[row_index, 0])
                if distance > self.eps:
                    unique_scores[row_index] = -3
                    continue

                prior = self.cluster_priors.get(event, {}).get(neighbour, 0.5)
                posterior = calibrated_incident_posterior(
                    cluster_prior=prior,
                    confidence=float(unique_target_confidences[row_index]),
                    distance=distance,
                    distance_scale=self.distance_scale,
                )
                decision = triage_decision(
                    posterior,
                    low_threshold=self.low_threshold,
                    high_threshold=self.high_threshold,
                )
                if decision == "incident":
                    unique_scores[row_index] = 1
                elif decision == "benign":
                    unique_scores[row_index] = 0
                else:
                    unique_scores[row_index] = REBALANCED_REJECT_SCORE

            result[unique_indices] = unique_scores[inverse_vector]

        return result[inverse_result.cpu().numpy()]

    def to_dict(self):
        payload = super().to_dict()
        payload.update(
            {
                "low_threshold": self.low_threshold,
                "high_threshold": self.high_threshold,
                "distance_scale": self.distance_scale,
                "cluster_priors": self.cluster_priors,
            }
        )
        return payload

    @classmethod
    def from_dict(cls, dictionary, context_builder=None):
        if context_builder is not None:
            dictionary["context_builder"] = context_builder

        result = cls(
            context_builder=dictionary.get("context_builder"),
            features=dictionary.get("features", 100),
            eps=dictionary.get("eps", 0.1),
            min_samples=dictionary.get("min_samples", 5),
            threshold=dictionary.get("threshold", 0.2),
            low_threshold=dictionary.get("low_threshold", 0.25),
            high_threshold=dictionary.get("high_threshold", 0.70),
            distance_scale=dictionary.get("distance_scale"),
        )
        result.clusters = dictionary.get("clusters", np.zeros(0))
        result.vectors = dictionary.get("vectors", sp.csc_matrix((0, result.features)))
        result.events = dictionary.get("events", np.zeros(0))
        result.tree = dictionary.get("tree", {})
        result.labels = dictionary.get("labels", {})
        result.cluster_priors = dictionary.get("cluster_priors", {})
        return result


def build_rebalanced_interpreter_plan(config: RebalancedConfig) -> dict[str, Any]:
    return {
        "baseline_policy": "原版 DeepCASE 只把 cluster score 當成硬標籤回傳。",
        "research_policy": "保留 L1 + DBSCAN，但 prediction 改成 cluster prior + confidence + distance 的 calibrated posterior，再做 reject-aware triage。",
        "preserve_l1_distance": config.preserve_l1_distance,
        "preserve_dbscan": config.preserve_dbscan,
        "future_variant": [
            "distance-aware calibration",
            "HDBSCAN 取代 DBSCAN",
        ],
    }
