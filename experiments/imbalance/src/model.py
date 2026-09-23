"""Incident-aligned DeepCASE training and unchanged-Interpreter adapters."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import KDTree

from deepcase.context_builder import ContextBuilder
from deepcase.context_builder.loss import LabelSmoothing
from deepcase.interpreter import Interpreter
from deepcase.interpreter.utils import group_by, sp_unique, unique_2d

from .data import (
    RevealedLabels,
    sample_group_balanced_supervision,
    sample_uniform,
)


AUXILIARY_METHODS = {
    "aux_bce",
    "aux_weighted_bce",
    "aux_focal",
    "aux_group_balanced_bce",
    "proposed",
}


@dataclass
class EpochLog:
    epoch: int
    event_loss: float
    incident_loss: float | None
    total_loss: float


@dataclass(frozen=True)
class AttendedCache:
    """Threshold-independent Interpreter inputs.

    ``confidence`` and ``vectors`` are exactly those produced by the original
    ``attention_query``/``vectorize`` path at the configured query-iteration
    count.  Thresholding, DBSCAN and nearest-cluster decisions are deliberately
    not cached because they are the operating-point mechanisms under study.
    """

    confidence: np.ndarray
    vectors: sp.spmatrix
    events: np.ndarray
    inverse: np.ndarray


class IncidentAlignedContextBuilder(ContextBuilder):
    """DeepCASE ContextBuilder with a training-only incident projection.

    `z` is exactly the dense, unrounded version of the attention-weighted
    event-count vector that `Interpreter.vectorize` later clusters. Applying an
    incident objective here, rather than to an unrelated hidden state, ensures
    that supervision shapes the deployed representation.
    """

    def __init__(self, input_size: int, hidden_size: int, max_length: int):
        super().__init__(
            input_size=input_size,
            output_size=input_size,
            hidden_size=hidden_size,
            max_length=max_length,
        )
        self.incident_head = nn.Linear(input_size, 1)

    def attended_representation(
        self, X: torch.Tensor, attention: torch.Tensor
    ) -> torch.Tensor:
        if attention.ndim == 3:
            attention = attention[:, 0, :]
        encoded = self.embedding_one_hot(X)
        return torch.bmm(attention.unsqueeze(1), encoded).squeeze(1)

    def forward_incident(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, attention = self.forward(X, steps=1, teach_ratio=0.0)
        representation = self.attended_representation(X, attention)
        logits = self.incident_head(representation).squeeze(1)
        return logits, representation


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def class_balanced_alpha(positive_count: int, negative_count: int, beta: float) -> float:
    """Return positive alpha from effective-number class weights."""

    if positive_count <= 0 or negative_count <= 0 or not 0 <= beta < 1:
        raise ValueError("Invalid class counts or beta")
    effective_positive = (1.0 - beta**positive_count) / (1.0 - beta)
    effective_negative = (1.0 - beta**negative_count) / (1.0 - beta)
    weight_positive = 1.0 / effective_positive
    weight_negative = 1.0 / effective_negative
    return float(weight_positive / (weight_positive + weight_negative))


def binary_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float,
    alpha_positive: float,
) -> torch.Tensor:
    target = target.to(dtype=logits.dtype)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probability = torch.sigmoid(logits)
    p_t = probability * target + (1.0 - probability) * (1.0 - target)
    alpha_t = alpha_positive * target + (1.0 - alpha_positive) * (1.0 - target)
    return (alpha_t * (1.0 - p_t).pow(gamma) * ce).mean()


def _incident_loss(
    method: str,
    logits: torch.Tensor,
    target: torch.Tensor,
    positive_count: int,
    negative_count: int,
    gamma: float,
    beta: float,
) -> torch.Tensor:
    target_float = target.to(dtype=logits.dtype)
    if method in {"aux_bce", "aux_group_balanced_bce"}:
        return F.binary_cross_entropy_with_logits(logits, target_float)
    if method == "aux_weighted_bce":
        pos_weight = torch.as_tensor(
            negative_count / positive_count, dtype=logits.dtype, device=logits.device
        )
        return F.binary_cross_entropy_with_logits(
            logits, target_float, pos_weight=pos_weight
        )
    if method == "aux_focal":
        alpha = class_balanced_alpha(positive_count, negative_count, beta)
        return binary_focal_loss(logits, target_float, gamma, alpha)
    if method == "proposed":
        # Class balance is enforced by the sampler; alpha=0.5 avoids correcting twice.
        return binary_focal_loss(logits, target_float, gamma, 0.5)
    raise ValueError(f"No incident loss for method {method}")


def train_context_builder(
    model: IncidentAlignedContextBuilder,
    context: np.ndarray,
    target_event: np.ndarray,
    train_indices: np.ndarray,
    revealed: RevealedLabels,
    metadata,
    label_column: str,
    group_column: str,
    method: str,
    *,
    seed: int,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    label_smoothing: float,
    incident_lambda: float,
    focal_gamma: float,
    effective_beta: float,
    progress: Callable[[str], None] | None = None,
) -> list[EpochLog]:
    if method != "deepcase" and method not in AUXILIARY_METHODS:
        raise ValueError(f"Unknown neural method: {method}")
    set_reproducible_seed(seed)
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    event_criterion = LabelSmoothing(model.decoder_event.out.out_features, label_smoothing)
    event_rng = np.random.default_rng(seed + 10_003)
    supervised_rng = np.random.default_rng(seed + 20_003)
    positive_count = len(revealed.positive_indices)
    negative_count = len(revealed.negative_indices)
    logs: list[EpochLog] = []
    model.train()

    for epoch in range(1, epochs + 1):
        event_sum = 0.0
        incident_sum = 0.0
        total_sum = 0.0
        for _ in range(steps_per_epoch):
            event_indices = sample_uniform(train_indices, batch_size, event_rng)
            X_event = torch.as_tensor(context[event_indices], dtype=torch.long, device=device)
            y_event = torch.as_tensor(target_event[event_indices], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            confidence, _ = model.forward(
                X_event, y_event[:, None], steps=1, teach_ratio=0.0
            )
            event_loss = event_criterion(confidence[:, 0], y_event) / len(event_indices)
            total_loss = event_loss
            incident_loss = None

            if method in AUXILIARY_METHODS:
                if method in {"aux_group_balanced_bce", "proposed"}:
                    supervised_indices = sample_group_balanced_supervision(
                        revealed,
                        metadata,
                        group_column,
                        batch_size,
                        supervised_rng,
                    )
                else:
                    supervised_indices = sample_uniform(
                        revealed.indices, batch_size, supervised_rng
                    )
                X_supervised = torch.as_tensor(
                    context[supervised_indices], dtype=torch.long, device=device
                )
                # Pandas/Arrow may expose a read-only NumPy view.  PyTorch warns
                # that writing through such a tensor is undefined, so take an
                # owned copy even though this tensor is presently read-only.
                incident_targets = np.array(
                    metadata.loc[supervised_indices, label_column].to_numpy(),
                    dtype=np.float32,
                    copy=True,
                )
                y_incident = torch.as_tensor(
                    incident_targets,
                    dtype=torch.float32,
                    device=device,
                )
                logits, _ = model.forward_incident(X_supervised)
                incident_loss = _incident_loss(
                    method,
                    logits,
                    y_incident,
                    positive_count,
                    negative_count,
                    focal_gamma,
                    effective_beta,
                )
                total_loss = event_loss + incident_lambda * incident_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            event_sum += float(event_loss.detach())
            incident_sum += 0.0 if incident_loss is None else float(incident_loss.detach())
            total_sum += float(total_loss.detach())

        item = EpochLog(
            epoch=epoch,
            event_loss=event_sum / steps_per_epoch,
            incident_loss=None if method == "deepcase" else incident_sum / steps_per_epoch,
            total_loss=total_sum / steps_per_epoch,
        )
        logs.append(item)
        if progress is not None:
            progress(str(asdict(item)))
    return logs


def compact_interpreter_fit_set(
    context: np.ndarray,
    target_event: np.ndarray,
    train_indices: np.ndarray,
    revealed: RevealedLabels,
    labels_all: np.ndarray,
    max_multiplicity: int,
    no_label: int = -9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Compress exact sequence multiplicity without changing DBSCAN core status.

    DBSCAN only asks whether neighbourhood weight reaches `min_samples`.
    Capping exact duplicate multiplicity at that same value preserves this
    status while preventing millions of identical alerts from being expanded.
    """

    patterns = np.concatenate(
        [context[train_indices], target_event[train_indices, None]], axis=1
    )
    unique, inverse, counts = np.unique(
        patterns, axis=0, return_inverse=True, return_counts=True
    )
    unique_scores = np.full(len(unique), no_label, dtype=np.int8)
    positions = np.searchsorted(train_indices, revealed.indices)
    if not np.array_equal(train_indices[positions], revealed.indices):
        raise ValueError("Revealed labels are not a subset of training rows")
    np.maximum.at(unique_scores, inverse[positions], labels_all[revealed.indices])
    repeats = np.minimum(counts, max_multiplicity).astype(np.int64)
    expanded = np.repeat(np.arange(len(unique)), repeats)
    expanded_patterns = unique[expanded]
    expanded_scores = unique_scores[expanded]
    stats = {
        "raw_train_rows": int(len(train_indices)),
        "unique_context_target_patterns": int(len(unique)),
        "expanded_fit_rows": int(len(expanded)),
        "labeled_unique_patterns": int(np.sum(unique_scores != no_label)),
        "positive_unique_patterns": int(np.sum(unique_scores == 1)),
    }
    return (
        expanded_patterns[:, :-1],
        expanded_patterns[:, -1],
        expanded_scores,
        stats,
    )


def fit_interpreter_allow_unlabeled(
    model: IncidentAlignedContextBuilder,
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    scores: np.ndarray,
    *,
    features: int,
    eps: float,
    min_samples: int,
    threshold: float,
    query_iterations: int,
    batch_size: int = 4096,
    no_label: int = -9,
) -> Interpreter:
    """Fit the original Interpreter; all-unlabelled clusters remain Reject."""

    interpreter = Interpreter(
        context_builder=model,
        features=features,
        eps=eps,
        min_samples=min_samples,
        threshold=threshold,
    )
    X_tensor = torch.as_tensor(X_fit, dtype=torch.long)
    y_tensor = torch.as_tensor(y_fit[:, None], dtype=torch.long)
    clusters = interpreter.cluster(
        X_tensor,
        y_tensor,
        iterations=query_iterations,
        batch_size=batch_size,
        verbose=False,
    )
    cluster_scores = np.full(len(clusters), -1.0, dtype=float)
    for cluster in np.unique(clusters):
        if cluster == -1:
            continue
        mask = clusters == cluster
        observed = scores[mask]
        observed = observed[observed != no_label]
        cluster_scores[mask] = -1.0 if len(observed) == 0 else float(observed.max())
    interpreter.score(cluster_scores, verbose=False)
    return interpreter


def build_attended_cache(
    model: IncidentAlignedContextBuilder,
    context: np.ndarray,
    target_event: np.ndarray,
    indices: np.ndarray | None,
    *,
    features: int,
    query_iterations: int,
    preserve_rows: bool,
    batch_size: int = 4096,
) -> AttendedCache:
    """Compute attention once while preserving original Interpreter semantics."""

    model.eval()
    selected_context = context if indices is None else context[indices]
    selected_target = target_event if indices is None else target_event[indices]
    X = torch.as_tensor(selected_context, dtype=torch.long)
    y = torch.as_tensor(selected_target[:, None], dtype=torch.long)
    X_unique, y_unique, inverse = unique_2d(X, y)
    adapter = Interpreter(context_builder=model, features=features)
    confidence_all, attention = adapter.attention_query(
        X_unique,
        y_unique,
        iterations=query_iterations,
        batch_size=batch_size,
        verbose=False,
    )
    observed = confidence_all.detach().cpu().numpy()
    vectors = adapter.vectorize(X_unique, attention, size=features)
    vectors = np.round(vectors, decimals=4)
    unique_events = y_unique.squeeze(1).cpu().numpy()
    inverse_array = inverse.cpu().numpy()
    if preserve_rows:
        return AttendedCache(
            confidence=observed[inverse_array],
            vectors=vectors[inverse_array],
            events=selected_target.copy(),
            inverse=np.arange(len(selected_target), dtype=np.int64),
        )
    return AttendedCache(
        confidence=observed,
        vectors=vectors,
        events=unique_events,
        inverse=inverse_array,
    )


def fit_cached_interpreter_allow_unlabeled(
    model: IncidentAlignedContextBuilder,
    cache: AttendedCache,
    scores: np.ndarray,
    *,
    features: int,
    eps: float,
    min_samples: int,
    threshold: float,
    no_label: int = -9,
) -> Interpreter:
    """Fit the original DBSCAN/KDTree policy from cached attended vectors."""

    if len(scores) != len(cache.events):
        raise ValueError("Cached fit scores and events have different lengths")
    interpreter = Interpreter(
        context_builder=model,
        features=features,
        eps=eps,
        min_samples=min_samples,
        threshold=threshold,
    )
    confidence_mask = cache.confidence >= threshold
    vectors = cache.vectors[confidence_mask]
    clustered = np.full(int(confidence_mask.sum()), -1, dtype=int)
    for _, event_positions in group_by(cache.events[confidence_mask]):
        labels = interpreter.dbscan.dbscan(
            vectors[event_positions],
            eps=eps,
            min_samples=min_samples,
            verbose=False,
        )
        labels[labels != -1] += max(0, clustered.max() + 1)
        clustered[event_positions] = labels
    clusters = np.full(len(cache.events), -1, dtype=int)
    clusters[confidence_mask] = clustered
    interpreter.clusters = clusters
    interpreter.vectors = vectors
    interpreter.events = cache.events

    cluster_scores = np.full(len(clusters), -1.0, dtype=float)
    for cluster in np.unique(clusters):
        if cluster == -1:
            continue
        cluster_mask = clusters == cluster
        observed = scores[cluster_mask]
        observed = observed[observed != no_label]
        cluster_scores[cluster_mask] = (
            -1.0 if len(observed) == 0 else float(observed.max())
        )
    # Upstream Interpreter.score filters events/scores to clustered rows but
    # indexes vectors with positions in that shorter array. Noise rows can
    # therefore be inserted into a scored tree. Build all three aligned arrays
    # with the SAME mask, then key labels by the public KDTree data-row index.
    # This repairs bookkeeping; DBSCAN, max scoring and all Reject conditions
    # are unchanged and shared by every experimental method.
    valid = clusters != -1
    valid_vectors = cache.vectors[valid]
    valid_events = cache.events[valid]
    valid_scores = cluster_scores[valid]
    for event, positions in group_by(valid_events):
        event_vectors, inverse, _ = sp_unique(valid_vectors[positions])
        vector_scores = np.full(event_vectors.shape[0], -np.inf, dtype=float)
        np.maximum.at(vector_scores, inverse, valid_scores[positions])
        interpreter.tree[event] = KDTree(event_vectors.toarray(), metric="manhattan")
        interpreter.labels[event] = dict(enumerate(vector_scores.tolist()))
    interpreter.scoring_index_version = "aligned_v2"
    return interpreter


def predict_cached_interpreter(
    interpreter: Interpreter,
    cache: AttendedCache,
) -> np.ndarray:
    """Apply the original confidence, unknown-event and distance rejects."""

    confidence_mask = cache.confidence >= interpreter.threshold
    vectors = cache.vectors[confidence_mask]
    result = np.full(vectors.shape[0], -4.0, dtype=float)
    for event, event_positions in group_by(cache.events[confidence_mask]):
        if event not in interpreter.tree:
            result[event_positions] = -2.0
            continue
        event_vectors, inverse, _ = sp_unique(vectors[event_positions])
        distance, neighbours = interpreter.tree[event].query(
            event_vectors.toarray(),
            return_distance=True,
            dualtree=event_vectors.shape[0] >= 1e3,
        )
        # KDTree.query returns indices into its INPUT data array already, not
        # into the internal tree permutation. Applying that permutation again
        # exchanges scores between geometrically unrelated vectors.
        neighbours = neighbours[:, 0]
        scores = np.asarray(
            [interpreter.labels[event][neighbour] for neighbour in neighbours]
        )
        result[event_positions] = np.where(
            distance[:, 0] <= interpreter.eps,
            scores,
            -3.0,
        )[inverse]
    expanded = np.full(len(cache.events), -1.0, dtype=float)
    expanded[confidence_mask] = result
    return expanded[cache.inverse]


def predict_interpreter(
    interpreter: Interpreter,
    context: np.ndarray,
    target_event: np.ndarray,
    indices: np.ndarray,
    query_iterations: int,
    batch_size: int = 4096,
) -> np.ndarray:
    X = torch.as_tensor(context[indices], dtype=torch.long)
    y = torch.as_tensor(target_event[indices, None], dtype=torch.long)
    return interpreter.predict(
        X,
        y,
        iterations=query_iterations,
        batch_size=batch_size,
        verbose=False,
    )


def fit_rule_prior(
    target_event: np.ndarray,
    revealed: RevealedLabels,
    input_size: int,
) -> np.ndarray:
    """Laplace-smoothed per-rule incident posterior using only revealed labels."""

    positive = np.bincount(
        target_event[revealed.positive_indices], minlength=input_size
    ).astype(float)
    negative = np.bincount(
        target_event[revealed.negative_indices], minlength=input_size
    ).astype(float)
    return (positive + 1.0) / (positive + negative + 2.0)


def predict_rule_prior(
    posterior: np.ndarray,
    target_event: np.ndarray,
    indices: np.ndarray,
    threshold: float,
    unk_id: int,
) -> np.ndarray:
    events = target_event[indices]
    result = (posterior[events] >= threshold).astype(float)
    result[events == unk_id] = -2.0
    return result
