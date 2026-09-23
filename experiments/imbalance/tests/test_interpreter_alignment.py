"""Semantic invariants independent of the upstream implementation's output."""

import numpy as np
import scipy.sparse as sp
import pytest

from src.model import AttendedCache, fit_cached_interpreter_allow_unlabeled, predict_cached_interpreter


def cache(vectors, confidence=None, events=None):
    vectors = np.asarray(vectors, dtype=float)
    size = len(vectors)
    return AttendedCache(
        confidence=np.full(size, 0.9) if confidence is None else np.asarray(confidence),
        vectors=sp.csc_matrix(vectors),
        events=np.zeros(size, dtype=int) if events is None else np.asarray(events),
        inverse=np.arange(size),
    )


def fit(data, scores, min_samples=3, eps=0.01):
    return fit_cached_interpreter_allow_unlabeled(
        None, data, np.asarray(scores), features=2, eps=eps,
        min_samples=min_samples, threshold=0.2,
    )


def test_noise_is_not_inserted_into_scored_tree():
    data = cache([[0.5, 0.5]] + [[1, 0]] * 3)
    model = fit(data, [-9, 1, -9, -9])
    assert model.clusters[0] == -1
    np.testing.assert_array_equal(
        predict_cached_interpreter(model, data), [-3, 1, 1, 1]
    )


def test_vector_sorting_cannot_exchange_cluster_labels():
    data = cache([[1, 0]] * 3 + [[0, 1]] * 3)
    model = fit(data, [1, -9, -9, 0, -9, -9])
    np.testing.assert_array_equal(
        predict_cached_interpreter(model, data), [1, 1, 1, 0, 0, 0]
    )


def test_low_confidence_unknown_and_distance_rejects_remain_distinct():
    train = cache([[1, 0]] * 3)
    model = fit(train, [1, -9, -9])
    test = cache([[1, 0], [1, 0], [0, 1], [1, 0]],
                 confidence=[0.1, 0.9, 0.9, 0.9], events=[0, 1, 0, 0])
    np.testing.assert_array_equal(predict_cached_interpreter(model, test), [-1, -2, -3, 1])


def test_kdtree_permutation_cannot_exchange_labels():
    rng = np.random.default_rng(37)
    positions = rng.permutation(np.linspace(0, 1, 101))
    vectors = np.column_stack([positions, 1 - positions])
    labels = (positions > 0.5).astype(float)
    data = cache(np.repeat(vectors, 3, axis=0))
    model = fit(data, np.repeat(labels, 3), eps=0.001)
    np.testing.assert_array_equal(predict_cached_interpreter(model, cache(vectors)), labels)


def test_all_unlabelled_clusters_are_rejected():
    data = cache([[1, 0]] * 3)
    model = fit(data, [-9] * 3)
    np.testing.assert_array_equal(predict_cached_interpreter(model, data), [-1] * 3)
