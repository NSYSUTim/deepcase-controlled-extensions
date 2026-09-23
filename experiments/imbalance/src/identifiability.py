"""Training-only diagnostics for rule-conditional context identifiability.

The full-truth diagnostics in this module are descriptive only.  They must not
be supplied to a fitted model or an operating-point selector.  Model-fitting
eligibility is computed separately from the actually revealed labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import RevealedLabels


@dataclass(frozen=True)
class IdentifiabilitySummary:
    rules_total: int
    rules_mixed_full_truth: int
    rules_pairable_revealed: int
    full_truth_rows_in_mixed_rules: int
    full_truth_positive_in_mixed_rules: int
    revealed_positive_pairable: int
    revealed_negative_pairable: int
    revealed_positive_groups_pairable: int
    fraction_train_rows_in_mixed_rules: float
    fraction_train_positives_in_mixed_rules: float
    fraction_revealed_positives_pairable: float
    pairability_pass: bool


def _binary_entropy(rate: pd.Series) -> pd.Series:
    values = rate.to_numpy(dtype=float)
    result = np.zeros_like(values)
    interior = (values > 0.0) & (values < 1.0)
    p = values[interior]
    result[interior] = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
    return pd.Series(result, index=rate.index, dtype=float)


def rule_identifiability_table(
    metadata: pd.DataFrame,
    train_indices: np.ndarray,
    revealed: RevealedLabels,
    label_column: str,
    group_column: str,
    *,
    rule_column: str = "event_key",
    minimum_full_per_class: int = 50,
    minimum_revealed_per_class: int = 1,
    lower_full_incident_rate: float = 0.05,
    upper_full_incident_rate: float = 0.95,
) -> pd.DataFrame:
    """Return one row per training rule without consulting validation/test truth."""

    if minimum_full_per_class < 1 or minimum_revealed_per_class < 1:
        raise ValueError("Minimum class counts must be positive")
    if not 0.0 <= lower_full_incident_rate < upper_full_incident_rate <= 1.0:
        raise ValueError("Invalid full-truth incident-rate interval")
    required = {rule_column, label_column, group_column}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")

    train = metadata.loc[
        train_indices, [rule_column, label_column, group_column]
    ].copy()
    if not train[label_column].isin([0, 1]).all():
        raise ValueError("Identifiability audit requires binary labels")
    train[rule_column] = train[rule_column].astype(str)
    train[label_column] = train[label_column].astype(np.int8)

    grouped = train.groupby(rule_column, observed=True)[label_column].agg(
        full_positive="sum", full_total="count"
    )
    grouped["full_negative"] = grouped["full_total"] - grouped["full_positive"]
    grouped["full_incident_rate"] = grouped["full_positive"] / grouped["full_total"]
    grouped["full_binary_entropy"] = _binary_entropy(grouped["full_incident_rate"])
    grouped["mixed_full_truth"] = (
        (grouped["full_positive"] >= minimum_full_per_class)
        & (grouped["full_negative"] >= minimum_full_per_class)
        & (grouped["full_incident_rate"] > lower_full_incident_rate)
        & (grouped["full_incident_rate"] < upper_full_incident_rate)
    )

    revealed_frame = metadata.loc[
        revealed.indices, [rule_column, label_column, group_column]
    ].copy()
    revealed_frame[rule_column] = revealed_frame[rule_column].astype(str)
    revealed_frame[label_column] = revealed_frame[label_column].astype(np.int8)
    revealed_counts = revealed_frame.groupby(rule_column, observed=True)[label_column].agg(
        revealed_positive="sum", revealed_total="count"
    )
    revealed_counts["revealed_negative"] = (
        revealed_counts["revealed_total"] - revealed_counts["revealed_positive"]
    )
    positive_groups = (
        revealed_frame[revealed_frame[label_column] == 1]
        .groupby(rule_column, observed=True)[group_column]
        .nunique(dropna=True)
        .rename("revealed_positive_groups")
    )

    table = grouped.join(revealed_counts, how="left").join(positive_groups, how="left")
    count_columns = [
        "revealed_positive",
        "revealed_total",
        "revealed_negative",
        "revealed_positive_groups",
    ]
    table[count_columns] = table[count_columns].fillna(0).astype(np.int64)
    table["pairable_revealed"] = (
        (table["revealed_positive"] >= minimum_revealed_per_class)
        & (table["revealed_negative"] >= minimum_revealed_per_class)
    )
    # This column is diagnostic-only: full truth determines whether a rule is
    # genuinely ambiguous, while the revealed counts determine whether the
    # proposed learner could form legal same-rule pairs.  It must never be used
    # to filter training examples.
    table["identifiable_revealed"] = (
        table["mixed_full_truth"] & table["pairable_revealed"]
    )
    table.index.name = rule_column
    return table.reset_index().sort_values(
        ["pairable_revealed", "full_binary_entropy", "full_total"],
        ascending=[False, False, False],
        kind="stable",
    ).reset_index(drop=True)


def summarize_identifiability(
    table: pd.DataFrame,
    *,
    minimum_pairable_rules: int,
    minimum_pairable_positive_groups: int,
) -> IdentifiabilitySummary:
    """Apply an explicit feasibility gate to a rule table."""

    if minimum_pairable_rules < 1 or minimum_pairable_positive_groups < 1:
        raise ValueError("Feasibility thresholds must be positive")
    mixed = table["mixed_full_truth"].astype(bool)
    pairable = table["identifiable_revealed"].astype(bool)
    total_rows = int(table["full_total"].sum())
    total_positive = int(table["full_positive"].sum())
    revealed_positive = int(table["revealed_positive"].sum())
    pairable_groups = int(table.loc[pairable, "revealed_positive_groups"].sum())
    pairable_rules = int(pairable.sum())
    return IdentifiabilitySummary(
        rules_total=int(len(table)),
        rules_mixed_full_truth=int(mixed.sum()),
        rules_pairable_revealed=pairable_rules,
        full_truth_rows_in_mixed_rules=int(table.loc[mixed, "full_total"].sum()),
        full_truth_positive_in_mixed_rules=int(
            table.loc[mixed, "full_positive"].sum()
        ),
        revealed_positive_pairable=int(table.loc[pairable, "revealed_positive"].sum()),
        revealed_negative_pairable=int(table.loc[pairable, "revealed_negative"].sum()),
        revealed_positive_groups_pairable=pairable_groups,
        fraction_train_rows_in_mixed_rules=(
            float(table.loc[mixed, "full_total"].sum() / total_rows)
            if total_rows
            else float("nan")
        ),
        fraction_train_positives_in_mixed_rules=(
            float(table.loc[mixed, "full_positive"].sum() / total_positive)
            if total_positive
            else float("nan")
        ),
        fraction_revealed_positives_pairable=(
            float(table.loc[pairable, "revealed_positive"].sum() / revealed_positive)
            if revealed_positive
            else float("nan")
        ),
        pairability_pass=(
            pairable_rules >= minimum_pairable_rules
            and pairable_groups >= minimum_pairable_positive_groups
        ),
    )
