import numpy as np
import torch

from src.evaluation import operational_metrics
from src.model import (
    IncidentAlignedContextBuilder,
    binary_focal_loss,
    build_attended_cache,
    class_balanced_alpha,
    fit_cached_interpreter_allow_unlabeled,
    fit_interpreter_allow_unlabeled,
    predict_cached_interpreter,
    predict_interpreter,
    set_reproducible_seed,
)
from src.rccr import (
    matched_workload_predictions,
    sample_group_balanced_within_rule_pairs,
    score_attended_vectors,
    smoothed_rule_logits,
    train_linear_risk_model,
)


def test_reject_consumes_workload_and_recovers_incident():
    truth = np.array([1, 1, 0, 0])
    prediction = np.array([0, -1, 0, -3], dtype=float)
    groups = np.array(["a", "b", None, None], dtype=object)
    result = operational_metrics(truth, prediction, groups)
    assert result.analyst_workload == 0.5
    assert result.triage_incident_recall == 0.5
    assert result.triage_precision == 0.5
    assert result.reject_rate == 0.5
    assert result.automation_coverage == 0.5
    assert result.auto_incident_rate + result.reject_rate == result.analyst_workload
    assert (
        result.reject_low_confidence_rate
        + result.reject_unknown_event_rate
        + result.reject_distance_rate
        == result.reject_rate
    )
    assert result.incident_group_recall == 0.5


def test_focal_loss_downweights_easy_examples():
    easy = binary_focal_loss(torch.tensor([8.0]), torch.tensor([1.0]), 2.0, 0.5)
    hard = binary_focal_loss(torch.tensor([-2.0]), torch.tensor([1.0]), 2.0, 0.5)
    assert hard > easy


def test_effective_number_alpha_favours_rare_positive():
    alpha = class_balanced_alpha(10, 1000, 0.9999)
    assert 0.9 < alpha < 1.0


def test_cached_interpreter_is_exactly_equivalent_to_original_path():
    set_reproducible_seed(19)
    model = IncidentAlignedContextBuilder(input_size=5, hidden_size=4, max_length=3)
    context = np.array(
        [
            [4, 0, 1],
            [4, 0, 1],
            [4, 0, 1],
            [4, 2, 1],
            [4, 2, 1],
            [4, 2, 1],
        ],
        dtype=np.int64,
    )
    target = np.array([2, 2, 2, 3, 3, 3], dtype=np.int64)
    scores = np.array([1, -9, -9, 0, -9, -9], dtype=np.int8)
    original = fit_interpreter_allow_unlabeled(
        model,
        context,
        target,
        scores,
        features=5,
        eps=0.2,
        min_samples=2,
        threshold=0.0,
        query_iterations=0,
    )
    expected = predict_interpreter(
        original, context, target, np.arange(len(context)), query_iterations=0
    )
    fit_cache = build_attended_cache(
        model,
        context,
        target,
        None,
        features=5,
        query_iterations=0,
        preserve_rows=True,
    )
    cached = fit_cached_interpreter_allow_unlabeled(
        model,
        fit_cache,
        scores,
        features=5,
        eps=0.2,
        min_samples=2,
        threshold=0.0,
    )
    prediction_cache = build_attended_cache(
        model,
        context,
        target,
        np.arange(len(context)),
        features=5,
        query_iterations=0,
        preserve_rows=False,
    )
    actual = predict_cached_interpreter(cached, prediction_cache)
    np.testing.assert_array_equal(actual, expected)


def test_within_rule_sampler_never_crosses_rule_boundary():
    events = np.array([1, 1, 1, 2, 2, 2])
    labels = np.array([1, 0, 0, 1, 0, 0])
    groups = np.array(["g1", None, None, "g2", None, None], dtype=object)
    positive, negative = sample_group_balanced_within_rule_pairs(
        events, labels, groups, 100, np.random.default_rng(7)
    )
    assert np.all(labels[positive] == 1)
    assert np.all(labels[negative] == 0)
    np.testing.assert_array_equal(events[positive], events[negative])
    assert set(groups[positive]) == {"g1", "g2"}


def test_pairwise_ranker_learns_context_order_without_rule_shortcut():
    import scipy.sparse as sp

    vectors = sp.csr_matrix(
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
    )
    events = np.array([0, 0, 1, 1])
    labels = np.array([1, 0, 1, 0])
    groups = np.array(["a", None, "b", None], dtype=object)
    prior = smoothed_rule_logits(events, labels, input_size=2)
    model = train_linear_risk_model(
        vectors,
        events,
        labels,
        groups,
        prior,
        objective="within_rule_pairwise",
        seed=11,
        epochs=10,
        steps_per_epoch=10,
        batch_size=8,
        learning_rate=0.1,
        weight_decay=0.0,
    )
    score = score_attended_vectors(vectors, events, prior, model)
    assert score[0] > score[1]
    assert score[2] > score[3]


def test_matched_ranking_preserves_rejects_and_review_count():
    reference = np.array([-1.0, -2.0, -3.0, 1.0, 1.0, 0.0, 0.0])
    scores = np.array([99.0, 99.0, 99.0, 0.1, 0.2, 0.9, 0.8])
    prediction = matched_workload_predictions(scores, reference, seed=5)
    np.testing.assert_array_equal(prediction[:3], reference[:3])
    assert np.sum(prediction > 0) == np.sum(reference > 0)
    assert np.sum(prediction < 0) == np.sum(reference < 0)
    assert prediction[5] == 1.0
    assert prediction[6] == 1.0
