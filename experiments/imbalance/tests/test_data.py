import numpy as np
import pandas as pd

from prepare_data import build_strict_past_contexts
from src.data import define_ambiguous_rules, reveal_training_labels, scenario_fold
from src.identifiability import rule_identifiability_table, summarize_identifiability


def test_strict_past_context_excludes_same_timestamp():
    frame = pd.DataFrame(
        {
            "machine": ["s::h", "s::h", "s::h", "s::h"],
            "timestamp": [1, 1, 2, 4],
            "event_id": [0, 1, 2, 3],
        }
    )
    context = build_strict_past_contexts(frame, event_count=4, length=3, timeout=2)
    pad = 4
    assert context.tolist() == [
        [pad, pad, pad],
        [pad, pad, pad],
        [pad, 0, 1],
        [pad, pad, 2],
    ]


def test_scenario_fold_is_disjoint_and_deterministic():
    fold = scenario_fold(["a", "b", "c", "d"], "b")
    assert fold.validation_scenario == "c"
    assert fold.train_scenarios == ("a", "d")


def test_reveal_labels_has_exact_ratio_and_group_coverage():
    metadata = pd.DataFrame(
        {
            "label": [1] * 30 + [0] * 100,
            "group": [f"g{i % 3}" for i in range(30)] + [None] * 100,
        }
    )
    revealed = reveal_training_labels(
        metadata=metadata,
        train_indices=np.arange(len(metadata)),
        label_column="label",
        group_column="group",
        label_budget=22,
        negative_to_positive_ratio=10,
        seed=7,
    )
    assert len(revealed.positive_indices) == 2
    assert len(revealed.negative_indices) == 20
    assert revealed.realized_ratio == 10
    assert np.all(metadata.loc[revealed.positive_indices, "label"].to_numpy() == 1)
    assert np.all(metadata.loc[revealed.negative_indices, "label"].to_numpy() == 0)


def test_ambiguous_rules_use_training_rows_only():
    metadata = pd.DataFrame(
        {
            "rule": ["mixed"] * 4 + ["pure"] * 4 + ["test_only"] * 2,
            "label": [1, 1, 0, 0] + [1, 1, 1, 1] + [1, 0],
        }
    )
    definition = define_ambiguous_rules(
        metadata,
        np.arange(8),
        "label",
        rule_column="rule",
        lower_incident_rate=0.1,
        upper_incident_rate=0.9,
        minimum_per_class=2,
    )
    assert definition.rules == ("mixed",)
    assert definition.training_positive == 2
    assert definition.training_negative == 2


def test_identifiability_separates_full_truth_from_revealed_pairability():
    metadata = pd.DataFrame(
        {
            "rule": ["a"] * 6 + ["b"] * 4 + ["test_only"] * 2,
            "label": [1, 1, 1, 0, 0, 0] + [1, 1, 1, 1] + [1, 0],
            "group": ["g1", "g2", "g3", None, None, None]
            + ["g4"] * 4
            + ["g5", None],
        }
    )
    revealed = reveal_training_labels(
        metadata,
        np.arange(10),
        "label",
        "group",
        label_budget=6,
        negative_to_positive_ratio=1,
        seed=9,
    )
    table = rule_identifiability_table(
        metadata,
        np.arange(10),
        revealed,
        "label",
        "group",
        rule_column="rule",
        minimum_full_per_class=2,
    )
    assert set(table["rule"]) == {"a", "b"}
    assert table.set_index("rule").loc["a", "mixed_full_truth"]
    assert not table.set_index("rule").loc["b", "mixed_full_truth"]
    assert table.set_index("rule").loc["a", "identifiable_revealed"]
    summary = summarize_identifiability(
        table,
        minimum_pairable_rules=1,
        minimum_pairable_positive_groups=1,
    )
    assert summary.rules_mixed_full_truth == 1
    assert summary.rules_pairable_revealed == 1
    assert summary.pairability_pass
