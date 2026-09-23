"""Training-only additive evidence teacher and fixed-budget selectors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


@dataclass(frozen=True)
class AssociationModel:
    weights: np.ndarray
    base_log_odds: np.ndarray
    global_to_local: np.ndarray
    seen_global_events: np.ndarray
    unknown_id: int
    bands: np.ndarray
    smoothing: float
    maximum_absolute_log_odds: float


def recency_bands(width: int, intervals: list[list[int]]) -> np.ndarray:
    """Map oldest-to-newest columns to preregistered newest-first rank bands."""

    result = np.full(width, -1, dtype=np.int8)
    rank = width - np.arange(width)
    for band, (lower, upper) in enumerate(intervals):
        result[(rank >= lower) & (rank <= upper)] = band
    if np.any(result < 0):
        raise ValueError("recency intervals do not cover every candidate position")
    return result


def fold_event_mapping(
    target_global: np.ndarray,
    train_indices: np.ndarray,
    global_event_count: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    seen = np.unique(target_global[train_indices]).astype(np.int16)
    mapping = np.full(global_event_count, len(seen), dtype=np.int16)
    mapping[seen] = np.arange(len(seen), dtype=np.int16)
    return mapping, seen, len(seen)


def _candidate_local_events(
    candidate_rows: np.ndarray,
    event_ids_global: np.ndarray,
    global_to_local: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid = candidate_rows >= 0
    safe = np.where(valid, candidate_rows, 0)
    local = global_to_local[event_ids_global[safe]]
    return local, valid


def _deduplicated_codes(
    target: np.ndarray,
    candidate: np.ndarray,
    valid: np.ndarray,
    bands: np.ndarray,
    vocabulary_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    band_count = int(bands.max()) + 1
    feature_count = vocabulary_size * band_count * vocabulary_size
    code = (
        (target[:, None].astype(np.int64) * band_count + bands[None, :])
        * vocabulary_size
        + candidate
    )
    row = np.broadcast_to(np.arange(len(target), dtype=np.int64)[:, None], code.shape)
    keys = row[valid] * feature_count + code[valid]
    unique = np.unique(keys)
    return unique // feature_count, unique % feature_count


def fit_association_model(
    candidate_rows: np.ndarray,
    event_ids_global: np.ndarray,
    target_global: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    *,
    global_event_count: int,
    bands: np.ndarray,
    smoothing: float,
    maximum_absolute_log_odds: float,
    batch_rows: int,
) -> AssociationModel:
    """Fit target-conditioned Bernoulli log-odds using training rows only."""

    mapping, seen, unknown = fold_event_mapping(
        target_global, train_indices, global_event_count
    )
    vocabulary_size = unknown + 1
    band_count = int(bands.max()) + 1
    feature_count = vocabulary_size * band_count * vocabulary_size
    positive = np.zeros(feature_count, dtype=np.int64)
    negative = np.zeros(feature_count, dtype=np.int64)
    local_target_all = mapping[target_global]
    train_target = local_target_all[train_indices]
    train_labels = labels[train_indices]
    target_positive = np.bincount(
        train_target[train_labels == 1], minlength=vocabulary_size
    ).astype(np.int64)
    target_negative = np.bincount(
        train_target[train_labels == 0], minlength=vocabulary_size
    ).astype(np.int64)

    for start in range(0, len(train_indices), batch_rows):
        rows = train_indices[start : start + batch_rows]
        block = np.asarray(candidate_rows[rows])
        candidate, valid = _candidate_local_events(block, event_ids_global, mapping)
        local_rows, codes = _deduplicated_codes(
            local_target_all[rows], candidate, valid, bands, vocabulary_size
        )
        batch_labels = labels[rows][local_rows]
        positive += np.bincount(codes[batch_labels == 1], minlength=feature_count)
        negative += np.bincount(codes[batch_labels == 0], minlength=feature_count)

    target_for_feature = (
        np.arange(feature_count, dtype=np.int64) // (band_count * vocabulary_size)
    )
    alpha = float(smoothing)
    p_pos = (positive + alpha) / (
        target_positive[target_for_feature] + 2.0 * alpha
    )
    p_neg = (negative + alpha) / (
        target_negative[target_for_feature] + 2.0 * alpha
    )
    logit_pos = np.log(p_pos) - np.log1p(-p_pos)
    logit_neg = np.log(p_neg) - np.log1p(-p_neg)
    weights = (logit_pos - logit_neg).reshape(
        vocabulary_size, band_count, vocabulary_size
    )

    # A target event absent from either training class cannot support a
    # target-conditioned contrast. Fall back to a class-conditional global
    # candidate/band table rather than inventing an infinite association.
    positive_global = positive.reshape(
        vocabulary_size, band_count, vocabulary_size
    ).sum(axis=0)
    negative_global = negative.reshape(
        vocabulary_size, band_count, vocabulary_size
    ).sum(axis=0)
    total_positive = int(np.sum(train_labels == 1))
    total_negative = int(np.sum(train_labels == 0))
    gp = (positive_global + alpha) / (total_positive + 2.0 * alpha)
    gn = (negative_global + alpha) / (total_negative + 2.0 * alpha)
    global_weight = (np.log(gp) - np.log1p(-gp)) - (
        np.log(gn) - np.log1p(-gn)
    )
    unsupported = (target_positive == 0) | (target_negative == 0)
    weights[unsupported] = global_weight
    weights = np.clip(
        weights, -maximum_absolute_log_odds, maximum_absolute_log_odds
    )

    prior = np.log(target_positive + alpha) - np.log(target_negative + alpha)
    absent = np.sum(np.log1p(-p_pos) - np.log1p(-p_neg), axis=0)
    # ``p_*`` is flat by feature; aggregate absent evidence by target.
    absent = (
        np.log1p(-p_pos) - np.log1p(-p_neg)
    ).reshape(vocabulary_size, band_count, vocabulary_size).sum(axis=(1, 2))
    base = prior + absent
    global_prior = np.log(total_positive + alpha) - np.log(total_negative + alpha)
    global_absent = np.sum(np.log1p(-gp) - np.log1p(-gn))
    base[unsupported] = global_prior + global_absent
    return AssociationModel(
        weights=weights.astype(np.float32),
        base_log_odds=base.astype(np.float32),
        global_to_local=mapping,
        seen_global_events=seen,
        unknown_id=unknown,
        bands=bands.astype(np.int8),
        smoothing=alpha,
        maximum_absolute_log_odds=float(maximum_absolute_log_odds),
    )


def association_scores(
    model: AssociationModel,
    candidate_rows: np.ndarray,
    event_ids_global: np.ndarray,
    target_global: np.ndarray,
    indices: np.ndarray,
    *,
    batch_rows: int,
) -> np.ndarray:
    """Score unique candidate/band presences under the fitted additive model."""

    result = np.empty(len(indices), dtype=np.float32)
    local_target_all = model.global_to_local[target_global]
    vocabulary_size = model.weights.shape[0]
    for start in range(0, len(indices), batch_rows):
        rows = indices[start : start + batch_rows]
        block = np.asarray(candidate_rows[rows])
        candidate, valid = _candidate_local_events(
            block, event_ids_global, model.global_to_local
        )
        targets = local_target_all[rows]
        local_rows, codes = _deduplicated_codes(
            targets, candidate, valid, model.bands, vocabulary_size
        )
        score = model.base_log_odds[targets].astype(np.float64, copy=True)
        flat_weight = model.weights.reshape(-1)
        np.add.at(score, local_rows, flat_weight[codes])
        result[start : start + len(rows)] = score
    return result


def probability_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    probability = expit(np.asarray(scores, dtype=np.float64))
    labels = np.asarray(labels, dtype=np.int8)
    return {
        "rows": int(len(labels)),
        "prevalence": float(labels.mean()),
        "roc_auc": float(roc_auc_score(labels, probability)),
        "average_precision": float(average_precision_score(labels, probability)),
        "log_loss": float(log_loss(labels, probability, labels=[0, 1])),
        "brier_score": float(brier_score_loss(labels, probability)),
    }


def select_by_association(
    model: AssociationModel,
    candidate_rows: np.ndarray,
    event_ids_global: np.ndarray,
    target_global: np.ndarray,
    *,
    budget: int,
    labels_for_oracle: np.ndarray | None = None,
) -> np.ndarray:
    """Return chronological candidate row IDs selected by evidence magnitude.

    With ``labels_for_oracle=None`` this function cannot read outcomes and ranks
    absolute training-only evidence. Supplying labels creates the explicitly
    non-deployable class oracle and is kept as a separate call site.
    """

    candidate, valid = _candidate_local_events(
        candidate_rows, event_ids_global, model.global_to_local
    )
    target = model.global_to_local[target_global]
    position = np.arange(candidate_rows.shape[1], dtype=np.int16)[None, :]
    evidence = model.weights[target[:, None], model.bands[None, :], candidate]
    if labels_for_oracle is None:
        priority = np.abs(evidence)
    else:
        direction = np.where(np.asarray(labels_for_oracle)[:, None] == 1, 1.0, -1.0)
        priority = evidence * direction
    # Stable scientific tie rule: more recent candidate wins equal evidence.
    priority = priority.astype(np.float64) + position * 1e-10
    priority[~valid] = -np.inf
    width = candidate_rows.shape[1]
    chosen = np.argpartition(priority, width - budget, axis=1)[:, -budget:]
    chosen.sort(axis=1)
    return np.take_along_axis(candidate_rows, chosen, axis=1)


def select_recency_anchored_association(
    model: AssociationModel,
    candidate_rows: np.ndarray,
    event_ids_global: np.ndarray,
    target_global: np.ndarray,
    *,
    budget: int,
    recency_anchors: int,
) -> np.ndarray:
    """Keep newest slots and spend the remaining budget on older evidence.

    This deployable selector has no label argument. Anchors are source slots,
    so left padding remains left padding for short histories. Association
    candidates exclude the anchored suffix, preventing duplicate row use.
    Returned row IDs remain chronological.
    """

    width = candidate_rows.shape[1]
    if not 0 < recency_anchors < budget <= width:
        raise ValueError("require 0 < recency_anchors < budget <= pool width")
    evidence_budget = budget - recency_anchors
    cutoff = width - recency_anchors
    older = candidate_rows[:, :cutoff]
    candidate, valid = _candidate_local_events(
        older, event_ids_global, model.global_to_local
    )
    target = model.global_to_local[target_global]
    position = np.arange(cutoff, dtype=np.int16)[None, :]
    evidence = model.weights[
        target[:, None], model.bands[None, :cutoff], candidate
    ]
    priority = np.abs(evidence).astype(np.float64) + position * 1e-10
    priority[~valid] = -np.inf
    chosen_old = np.argpartition(
        priority, cutoff - evidence_budget, axis=1
    )[:, -evidence_budget:]
    anchor_positions = np.broadcast_to(
        np.arange(cutoff, width, dtype=np.int64)[None, :],
        (len(candidate_rows), recency_anchors),
    )
    chosen = np.concatenate([chosen_old, anchor_positions], axis=1)
    chosen.sort(axis=1)
    return np.take_along_axis(candidate_rows, chosen, axis=1)
