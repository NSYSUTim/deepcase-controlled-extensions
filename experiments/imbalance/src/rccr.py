"""Reject-preserving rule-conditioned contextual ranking.

The DeepCASE ContextBuilder and Interpreter rejection policy remain fixed.
Only accepted alerts are re-ranked, with the number sent to review matched
exactly to the paired DeepCASE operating point.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import scipy.sparse as sp
from scipy.special import expit


@dataclass(frozen=True)
class LinearRiskModel:
    weights: np.ndarray
    bias: float
    uses_rule_prior: bool
    objective: str
    training_log: tuple[dict, ...]
    eligible_positive_rows: int
    eligible_positive_groups: int
    eligible_rules: int


def smoothed_rule_logits(
    event_ids: np.ndarray,
    labels: np.ndarray,
    input_size: int,
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    """Estimate finite train-only rule logits with symmetric smoothing."""

    event_ids = np.asarray(event_ids, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int8)
    if event_ids.shape != labels.shape:
        raise ValueError("Event and label arrays differ in shape")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Rule priors require binary labels")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    positive = np.bincount(event_ids[labels == 1], minlength=input_size).astype(float)
    negative = np.bincount(event_ids[labels == 0], minlength=input_size).astype(float)
    probability = (positive + alpha) / (positive + negative + 2.0 * alpha)
    return np.log(probability / (1.0 - probability))


def _group_map(positions: np.ndarray, groups: np.ndarray) -> dict[str, np.ndarray]:
    mapping: dict[str, list[int]] = {}
    for position in positions.tolist():
        mapping.setdefault(str(groups[position]), []).append(int(position))
    return {
        group: np.asarray(rows, dtype=np.int64) for group, rows in mapping.items()
    }


def pairability_from_revealed(
    event_ids: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> dict:
    """Describe legal same-rule positive/negative pairs using revealed data only."""

    event_ids = np.asarray(event_ids, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int8)
    groups = np.asarray(groups, dtype=object)
    negative_events = set(event_ids[labels == 0].tolist())
    eligible = np.flatnonzero(
        (labels == 1) & np.isin(event_ids, np.asarray(sorted(negative_events)))
    )
    return {
        "eligible_positive_rows": int(len(eligible)),
        "eligible_positive_groups": int(len(np.unique(groups[eligible]))),
        "eligible_rules": int(len(np.unique(event_ids[eligible]))),
    }


def sample_group_balanced_within_rule_pairs(
    event_ids: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample incident groups uniformly and negatives from the same rule."""

    if size < 1:
        raise ValueError("Pair batch size must be positive")
    event_ids = np.asarray(event_ids, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int8)
    groups = np.asarray(groups, dtype=object)
    if event_ids.shape != labels.shape or labels.shape != groups.shape:
        raise ValueError("Pair-sampling arrays differ in shape")
    negative_by_event = {
        int(event): np.flatnonzero((labels == 0) & (event_ids == event))
        for event in np.unique(event_ids[labels == 0])
    }
    eligible_positive = np.asarray(
        [
            position
            for position in np.flatnonzero(labels == 1)
            if int(event_ids[position]) in negative_by_event
        ],
        dtype=np.int64,
    )
    if len(eligible_positive) == 0:
        raise ValueError("No revealed within-rule positive/negative pair exists")
    positive_by_group = _group_map(eligible_positive, groups)
    group_names = np.asarray(sorted(positive_by_group), dtype=object)
    selected_groups = rng.choice(group_names, size=size, replace=True)
    positive = np.empty(size, dtype=np.int64)
    negative = np.empty(size, dtype=np.int64)
    for offset, group in enumerate(selected_groups):
        positive[offset] = rng.choice(positive_by_group[str(group)])
        event = int(event_ids[positive[offset]])
        negative[offset] = rng.choice(negative_by_event[event])
    if not np.array_equal(event_ids[positive], event_ids[negative]):
        raise AssertionError("Within-rule sampler emitted a cross-rule pair")
    return positive, negative


def sample_group_balanced_binary_batch(
    labels: np.ndarray,
    groups: np.ndarray,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a 1:1 BCE batch with incident groups sampled uniformly."""

    if size < 2:
        raise ValueError("Binary batch size must be at least two")
    labels = np.asarray(labels, dtype=np.int8)
    groups = np.asarray(groups, dtype=object)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    if len(positive) == 0 or len(negative) == 0:
        raise ValueError("Binary batches require both classes")
    positive_by_group = _group_map(positive, groups)
    group_names = np.asarray(sorted(positive_by_group), dtype=object)
    positive_size = size // 2
    chosen_groups = rng.choice(group_names, size=positive_size, replace=True)
    chosen_positive = np.asarray(
        [rng.choice(positive_by_group[str(group)]) for group in chosen_groups],
        dtype=np.int64,
    )
    chosen_negative = rng.choice(
        negative, size=size - positive_size, replace=len(negative) < size - positive_size
    ).astype(np.int64)
    result = np.concatenate([chosen_positive, chosen_negative])
    rng.shuffle(result)
    return result


def train_linear_risk_model(
    vectors: sp.spmatrix,
    event_ids: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    rule_logits: np.ndarray,
    *,
    objective: str,
    seed: int,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> LinearRiskModel:
    """Fit a small linear contextual scorer with a fixed rule prior."""

    if objective not in {"context_bce", "rule_context_bce", "within_rule_pairwise"}:
        raise ValueError(f"Unknown RC-CR objective: {objective}")
    if vectors.shape[0] != len(event_ids) or len(event_ids) != len(labels):
        raise ValueError("Training feature arrays differ in length")
    rng = np.random.default_rng(seed + 41_009)
    uses_rule_prior = objective != "context_bce"
    weights = np.zeros(vectors.shape[1], dtype=np.float64)
    bias = 0.0
    first_moment = np.zeros_like(weights)
    second_moment = np.zeros_like(weights)
    bias_first_moment = 0.0
    bias_second_moment = 0.0
    adam_step = 0
    beta_one = 0.9
    beta_two = 0.999
    epsilon = 1e-8
    event_ids = np.asarray(event_ids, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int8)
    groups = np.asarray(groups, dtype=object)
    prior = rule_logits[event_ids].astype(np.float64, copy=False)
    pairability = pairability_from_revealed(event_ids, labels, groups)
    log: list[dict] = []

    for epoch in range(1, epochs + 1):
        total = 0.0
        for _ in range(steps_per_epoch):
            adam_step += 1
            if objective == "within_rule_pairwise":
                positive, negative = sample_group_balanced_within_rule_pairs(
                    event_ids, labels, groups, batch_size, rng
                )
                difference = vectors[positive] - vectors[negative]
                margin = np.asarray(difference @ weights).reshape(-1)
                loss = float(np.logaddexp(0.0, -margin).mean())
                coefficient = expit(margin) - 1.0
                gradient = np.asarray(difference.T @ coefficient).reshape(-1)
                gradient = gradient / batch_size + weight_decay * weights
                bias_gradient = 0.0
            else:
                rows = sample_group_balanced_binary_batch(labels, groups, batch_size, rng)
                logits = np.asarray(vectors[rows] @ weights).reshape(-1) + bias
                if uses_rule_prior:
                    logits = logits + prior[rows]
                target = labels[rows].astype(np.float64, copy=False)
                loss = float((np.logaddexp(0.0, logits) - target * logits).mean())
                error = expit(logits) - target
                gradient = np.asarray(vectors[rows].T @ error).reshape(-1)
                gradient = gradient / len(rows) + weight_decay * weights
                bias_gradient = float(error.mean())

            first_moment = beta_one * first_moment + (1.0 - beta_one) * gradient
            second_moment = beta_two * second_moment + (1.0 - beta_two) * gradient**2
            corrected_first = first_moment / (1.0 - beta_one**adam_step)
            corrected_second = second_moment / (1.0 - beta_two**adam_step)
            weights -= learning_rate * corrected_first / (
                np.sqrt(corrected_second) + epsilon
            )
            if objective != "within_rule_pairwise":
                bias_first_moment = (
                    beta_one * bias_first_moment + (1.0 - beta_one) * bias_gradient
                )
                bias_second_moment = (
                    beta_two * bias_second_moment
                    + (1.0 - beta_two) * bias_gradient**2
                )
                corrected_bias_first = bias_first_moment / (1.0 - beta_one**adam_step)
                corrected_bias_second = bias_second_moment / (1.0 - beta_two**adam_step)
                bias -= learning_rate * corrected_bias_first / (
                    np.sqrt(corrected_bias_second) + epsilon
                )
            total += loss
        log.append({"epoch": epoch, "loss": total / steps_per_epoch})

    return LinearRiskModel(
        weights=weights,
        bias=bias,
        uses_rule_prior=uses_rule_prior,
        objective=objective,
        training_log=tuple(log),
        **pairability,
    )


def score_attended_vectors(
    vectors: sp.spmatrix,
    event_ids: np.ndarray,
    rule_logits: np.ndarray,
    model: LinearRiskModel | None,
) -> np.ndarray:
    """Return finite rule/context scores for unique or expanded cache rows."""

    event_ids = np.asarray(event_ids, dtype=np.int64)
    if vectors.shape[0] != len(event_ids):
        raise ValueError("Scoring vectors and events differ in length")
    if model is None:
        result = rule_logits[event_ids].astype(np.float64, copy=True)
    else:
        result = np.asarray(vectors @ model.weights).reshape(-1) + model.bias
        if model.uses_rule_prior:
            result = result + rule_logits[event_ids]
    if not np.isfinite(result).all():
        raise ValueError("Non-finite risk score generated")
    return result


def matched_workload_predictions(
    scores: np.ndarray,
    reference_prediction: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Preserve Reject codes and exactly match DeepCASE's review count.

    Ties are broken with a seeded random key that is independent of row order
    and outcome labels.  The same seed must be used for paired methods.
    """

    scores = np.asarray(scores, dtype=float)
    reference_prediction = np.asarray(reference_prediction, dtype=float)
    if scores.shape != reference_prediction.shape:
        raise ValueError("Scores and reference predictions differ in shape")
    accepted = np.flatnonzero(reference_prediction >= 0)
    incident_slots = int(np.sum(reference_prediction > 0))
    if incident_slots > len(accepted):
        raise AssertionError("Reference incident count exceeds accepted rows")
    result = reference_prediction.copy()
    result[accepted] = 0.0
    if incident_slots:
        rng = np.random.default_rng(seed)
        tie_key = rng.random(len(accepted))
        order = np.lexsort((tie_key, -scores[accepted]))
        result[accepted[order[:incident_slots]]] = 1.0
    if int(np.sum(result < 0)) != int(np.sum(reference_prediction < 0)):
        raise AssertionError("Reject count changed during matched ranking")
    if int(np.sum(result > 0)) != incident_slots:
        raise AssertionError("Incident review slots were not matched")
    return result


def serialize_risk_model(model: LinearRiskModel) -> dict:
    result = asdict(model)
    result["weights"] = model.weights.tolist()
    result["training_log"] = list(model.training_log)
    return result
