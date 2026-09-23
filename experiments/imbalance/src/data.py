"""Prepared-data loading, leakage-safe folds, and label-reveal protocols."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PreparedData:
    metadata: pd.DataFrame
    context: np.ndarray
    target: np.ndarray
    vocabulary: tuple[str, ...]
    pad_id: int


@dataclass(frozen=True)
class ScenarioFold:
    train_scenarios: tuple[str, ...]
    validation_scenario: str
    test_scenario: str


@dataclass(frozen=True)
class EncodedFold:
    context: np.ndarray
    target: np.ndarray
    global_to_local: np.ndarray
    input_size: int
    unk_id: int
    pad_id: int
    seen_global_events: np.ndarray


@dataclass(frozen=True)
class RevealedLabels:
    indices: np.ndarray
    labels: np.ndarray
    positive_indices: np.ndarray
    negative_indices: np.ndarray
    requested_ratio: int
    realized_ratio: float
    label_budget: int
    positive_groups_covered: int
    positive_groups_available: int


@dataclass(frozen=True)
class AmbiguousRuleDefinition:
    """A training-truth-only diagnostic definition of noisy rules.

    This object must never be passed to model fitting or operating-point
    selection.  It exists solely to report whether context helps on rules
    whose full *training-scenario* labels are genuinely mixed.
    """

    rules: tuple[str, ...]
    lower_incident_rate: float
    upper_incident_rate: float
    minimum_per_class: int
    training_rows: int
    training_positive: int
    training_negative: int


def load_prepared(parquet_path: Path, contexts_path: Path) -> PreparedData:
    metadata = pd.read_parquet(parquet_path)
    arrays = np.load(contexts_path, allow_pickle=True)
    context = arrays["context"]
    target = arrays["target"]
    vocabulary = tuple(str(item) for item in arrays["vocabulary"].tolist())
    pad_id = int(arrays["pad_id"])
    if len(metadata) != len(context) or len(metadata) != len(target):
        raise ValueError("Metadata, context, and target row counts differ")
    if not np.array_equal(metadata["row_id"].to_numpy(), np.arange(len(metadata))):
        raise ValueError("row_id is not a dense index; positional alignment is unsafe")
    if context.ndim != 2 or target.ndim != 1:
        raise ValueError("Unexpected context or target dimensionality")
    if context.max() > pad_id or target.max() >= pad_id:
        raise ValueError("Prepared event IDs exceed the declared vocabulary")
    return PreparedData(metadata, context, target, vocabulary, pad_id)


def scenario_fold(scenarios: Iterable[str], test_scenario: str) -> ScenarioFold:
    """Create a deterministic nested holdout without consulting outcomes.

    The next scenario in the pre-registered order is validation; all remaining
    scenarios are training. The test scenario never influences model or
    hyperparameter selection.
    """

    ordered = tuple(scenarios)
    if test_scenario not in ordered:
        raise ValueError(f"Unknown test scenario: {test_scenario}")
    if len(ordered) < 3:
        raise ValueError("At least three scenarios are required")
    test_position = ordered.index(test_scenario)
    validation = ordered[(test_position + 1) % len(ordered)]
    train = tuple(item for item in ordered if item not in {test_scenario, validation})
    return ScenarioFold(train, validation, test_scenario)


def fold_indices(metadata: pd.DataFrame, fold: ScenarioFold) -> dict[str, np.ndarray]:
    scenario = metadata["scenario"].astype(str).to_numpy()
    result = {
        "train": np.flatnonzero(np.isin(scenario, fold.train_scenarios)),
        "validation": np.flatnonzero(scenario == fold.validation_scenario),
        "test": np.flatnonzero(scenario == fold.test_scenario),
    }
    sets = [set(values.tolist()) for values in result.values()]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise AssertionError("Fold row overlap detected")
    return result


def encode_for_fold(data: PreparedData, train_indices: np.ndarray) -> EncodedFold:
    """Fit an event vocabulary on training scenarios and map test-only events to UNK."""

    seen = np.unique(data.target[train_indices]).astype(np.int64)
    global_to_local = np.full(data.pad_id + 1, -1, dtype=np.int16)
    global_to_local[seen] = np.arange(len(seen), dtype=np.int16)
    unk_id = len(seen)
    local_pad_id = len(seen) + 1
    global_to_local[: data.pad_id][global_to_local[: data.pad_id] < 0] = unk_id
    global_to_local[data.pad_id] = local_pad_id
    encoded_context = global_to_local[data.context]
    encoded_target = global_to_local[data.target]
    return EncodedFold(
        context=encoded_context,
        target=encoded_target,
        global_to_local=global_to_local,
        input_size=local_pad_id + 1,
        unk_id=unk_id,
        pad_id=local_pad_id,
        seen_global_events=seen,
    )


def define_ambiguous_rules(
    metadata: pd.DataFrame,
    train_indices: np.ndarray,
    label_column: str,
    *,
    rule_column: str = "event_key",
    lower_incident_rate: float = 0.05,
    upper_incident_rate: float = 0.95,
    minimum_per_class: int = 50,
) -> AmbiguousRuleDefinition:
    """Define a secondary context-value stratum without consulting test labels.

    A rule is included only when the complete training scenarios contain at
    least ``minimum_per_class`` examples from each class and its incident rate
    lies within the pre-specified open interval.  Full training truth is used
    only to define this oracle diagnostic stratum; label-reveal scarcity remains
    unchanged and the resulting rule list must not affect fitting or tuning.
    """

    if not 0.0 <= lower_incident_rate < upper_incident_rate <= 1.0:
        raise ValueError("Invalid ambiguous-rule incident-rate interval")
    if minimum_per_class < 1:
        raise ValueError("minimum_per_class must be positive")
    frame = metadata.loc[train_indices, [rule_column, label_column]].copy()
    if not frame[label_column].isin([0, 1]).all():
        raise ValueError("Ambiguous-rule definition requires binary labels")
    grouped = frame.groupby(rule_column, observed=True)[label_column].agg(
        positive="sum", total="count"
    )
    grouped["negative"] = grouped["total"] - grouped["positive"]
    grouped["incident_rate"] = grouped["positive"] / grouped["total"]
    selected = grouped[
        (grouped["positive"] >= minimum_per_class)
        & (grouped["negative"] >= minimum_per_class)
        & (grouped["incident_rate"] > lower_incident_rate)
        & (grouped["incident_rate"] < upper_incident_rate)
    ]
    return AmbiguousRuleDefinition(
        rules=tuple(sorted(str(item) for item in selected.index)),
        lower_incident_rate=float(lower_incident_rate),
        upper_incident_rate=float(upper_incident_rate),
        minimum_per_class=int(minimum_per_class),
        training_rows=int(selected["total"].sum()),
        training_positive=int(selected["positive"].sum()),
        training_negative=int(selected["negative"].sum()),
    )


def _balanced_positive_sample(
    candidates: np.ndarray,
    groups: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample without replacement, cycling uniformly across incident groups."""

    if count > len(candidates):
        raise ValueError("Requested more positive labels than are available")
    group_to_rows: dict[str, list[int]] = {}
    for index, group in zip(candidates.tolist(), groups.tolist(), strict=True):
        group_to_rows.setdefault(str(group), []).append(int(index))
    for rows in group_to_rows.values():
        rng.shuffle(rows)
    group_names = np.asarray(sorted(group_to_rows), dtype=object)
    selected: list[int] = []
    while len(selected) < count:
        rng.shuffle(group_names)
        progress = False
        for group in group_names:
            rows = group_to_rows[str(group)]
            if rows:
                selected.append(rows.pop())
                progress = True
                if len(selected) == count:
                    break
        if not progress:
            raise AssertionError("Positive group sampler exhausted unexpectedly")
    return np.asarray(selected, dtype=np.int64)


def reveal_training_labels(
    metadata: pd.DataFrame,
    train_indices: np.ndarray,
    label_column: str,
    group_column: str,
    label_budget: int,
    negative_to_positive_ratio: int,
    seed: int,
) -> RevealedLabels:
    """Reveal an exact-ratio, fixed-budget training subset without deleting events."""

    if label_budget <= 1 or negative_to_positive_ratio < 1:
        raise ValueError("Invalid label budget or imbalance ratio")
    labels_all = metadata[label_column].to_numpy(dtype=np.int8)
    train_labels = labels_all[train_indices]
    positive_candidates = train_indices[train_labels == 1]
    negative_candidates = train_indices[train_labels == 0]
    positive_count = label_budget // (negative_to_positive_ratio + 1)
    positive_count = max(1, positive_count)
    negative_count = min(label_budget - positive_count, positive_count * negative_to_positive_ratio)
    if negative_count > len(negative_candidates):
        negative_count = len(negative_candidates)
    if positive_count > len(positive_candidates):
        positive_count = len(positive_candidates)
    if positive_count == 0 or negative_count == 0:
        raise ValueError("Both classes must have at least one revealed label")

    rng = np.random.default_rng(seed)
    positive_groups = metadata.loc[positive_candidates, group_column].astype(str).to_numpy()
    selected_positive = _balanced_positive_sample(
        positive_candidates, positive_groups, positive_count, rng
    )
    selected_negative = rng.choice(
        negative_candidates, size=negative_count, replace=False
    ).astype(np.int64)
    selected = np.concatenate([selected_positive, selected_negative])
    rng.shuffle(selected)
    selected_labels = labels_all[selected]
    available_groups = np.unique(positive_groups)
    covered_groups = metadata.loc[selected_positive, group_column].nunique(dropna=True)
    return RevealedLabels(
        indices=selected,
        labels=selected_labels,
        positive_indices=selected_positive,
        negative_indices=selected_negative,
        requested_ratio=negative_to_positive_ratio,
        realized_ratio=float(negative_count / positive_count),
        label_budget=int(len(selected)),
        positive_groups_covered=int(covered_groups),
        positive_groups_available=int(len(available_groups)),
    )


def sample_uniform(
    candidates: np.ndarray, size: int, rng: np.random.Generator
) -> np.ndarray:
    return rng.choice(candidates, size=size, replace=len(candidates) < size).astype(np.int64)


def sample_group_balanced_supervision(
    revealed: RevealedLabels,
    metadata: pd.DataFrame,
    group_column: str,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a class-balanced batch with incident groups sampled uniformly."""

    positive_size = size // 2
    negative_size = size - positive_size
    groups = metadata.loc[revealed.positive_indices, group_column].astype(str).to_numpy()
    group_names = np.unique(groups)
    selected_positive = np.empty(positive_size, dtype=np.int64)
    for offset in range(positive_size):
        group = rng.choice(group_names)
        rows = revealed.positive_indices[groups == group]
        selected_positive[offset] = rng.choice(rows)
    selected_negative = sample_uniform(
        revealed.negative_indices, negative_size, rng
    )
    selected = np.concatenate([selected_positive, selected_negative])
    rng.shuffle(selected)
    return selected
