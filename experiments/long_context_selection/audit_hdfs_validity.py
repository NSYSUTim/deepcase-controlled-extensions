"""Post-unblinding validity audit; cannot change the frozen HDFS decision."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_hdfs_external import _bootstrap_grouped_ratios, point_index
from run_hdfs_external import labeled_metrics
from src.common import EXPERIMENT, write_json


ARTIFACTS = EXPERIMENT / "hdfs_artifacts"
RESULTS = EXPERIMENT / "results" / "hdfs_external"


def analyze() -> dict:
    decision_path = RESULTS / "final_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    labels = np.load(ARTIFACTS / "labels.npy", mmap_mode="r")
    splits = np.load(ARTIFACTS / "split_codes.npy", mmap_mode="r")
    row_sequence = np.load(ARTIFACTS / "sequence_ids.npy", mmap_mode="r")
    test_rows = np.flatnonzero(splits == 3)
    truth = np.asarray(labels[test_rows], dtype=np.int8)
    test_sequence = np.asarray(row_sequence[test_rows], dtype=np.int32)
    sequence_ids, starts = np.unique(test_sequence, return_index=True)
    lengths = np.diff(np.r_[starts, len(test_rows)]).astype(np.int64)
    with np.load(ARTIFACTS / "sequences.npz") as table:
        overlap = table["train_pattern_overlap"][sequence_ids]
        sequence_labels = table["label"][sequence_ids]
    # Final-sequence IDs are global IDs and need not be contiguous because the
    # deterministic hash split interleaves dev/validation/final sequences.
    nonoverlap_lookup = np.zeros(int(sequence_ids.max()) + 1, dtype=bool)
    nonoverlap_lookup[sequence_ids] = ~overlap
    nonoverlap_row_mask = nonoverlap_lookup[test_sequence]

    selected_by_seed = {int(row["seed"]): row for row in decision["blind_rows"]}
    baseline_sequence_counts = np.zeros(len(sequence_ids), dtype=np.int64)
    selector_sequence_counts = np.zeros(len(sequence_ids), dtype=np.int64)
    rows = []
    for seed, selected in selected_by_seed.items():
        base_record = json.loads(
            (RESULTS / "last10" / f"seed_{seed}.json").read_text(encoding="utf-8")
        )
        selector_record = json.loads(
            (RESULTS / "anchor5_assoc5" / f"seed_{seed}.json").read_text(
                encoding="utf-8"
            )
        )
        base_index = point_index(base_record, 0.1, 0.2)
        selector_index = int(selected["selected_grid_index"])
        base_prediction = np.load(
            base_record["predictions"], mmap_mode="r"
        )[base_index]
        selector_prediction = np.load(
            selector_record["predictions"], mmap_mode="r"
        )[selector_index]
        baseline_sequence_counts += np.add.reduceat(
            (base_prediction > 0).astype(np.int64), starts
        )
        selector_sequence_counts += np.add.reduceat(
            (selector_prediction > 0).astype(np.int64), starts
        )
        base_metrics = labeled_metrics(
            truth[nonoverlap_row_mask], base_prediction[nonoverlap_row_mask]
        )
        selector_metrics = labeled_metrics(
            truth[nonoverlap_row_mask], selector_prediction[nonoverlap_row_mask]
        )
        rows.append(
            {
                "seed": seed,
                "nonoverlap_baseline_recall": base_metrics[
                    "automatic_incident_recall"
                ],
                "nonoverlap_selector_recall": selector_metrics[
                    "automatic_incident_recall"
                ],
                "nonoverlap_recall_gain": selector_metrics[
                    "automatic_incident_recall"
                ]
                - base_metrics["automatic_incident_recall"],
                "nonoverlap_baseline_false_escalations_per_1000": base_metrics[
                    "false_escalations_per_1000_nonincidents"
                ],
                "nonoverlap_selector_false_escalations_per_1000": selector_metrics[
                    "false_escalations_per_1000_nonincidents"
                ],
            }
        )

    nonoverlap_incident = (sequence_labels == 1) & ~overlap
    nonoverlap_normal = (sequence_labels == 0) & ~overlap
    evaluation = {
        "model_seed_count": len(rows),
        "safety_switch_rate_per_1000": 1.0,
        "low_rate_absolute_margin_per_1000": 1.0,
        "high_rate_relative_margin": 0.05,
    }
    recall_bootstrap = _bootstrap_grouped_ratios(
        lengths[nonoverlap_incident],
        baseline_sequence_counts[nonoverlap_incident],
        selector_sequence_counts[nonoverlap_incident],
        iterations=20_000,
        seed=260908,
        scale=1.0,
        statistic="gain",
        evaluation=evaluation,
    )
    safety_bootstrap = _bootstrap_grouped_ratios(
        lengths[nonoverlap_normal],
        baseline_sequence_counts[nonoverlap_normal],
        selector_sequence_counts[nonoverlap_normal],
        iterations=20_000,
        seed=260909,
        scale=1000.0,
        statistic="safety_violation",
        evaluation=evaluation,
    )
    mean_workload = float(
        np.mean([row["baseline_workload"] for row in decision["blind_rows"]])
    )
    safety_limit = float(
        decision["mean_baseline_false_escalations_per_1000"] * 1.05
    )
    overlap_fraction = float(overlap.mean())
    reasons = []
    if safety_limit > 1000.0:
        reasons.append(
            "relative safety limit exceeds the physical maximum of 1000/1000"
        )
    if mean_workload > 0.95:
        reasons.append("baseline sends more than 95% of final events to review")
    if decision["mean_baseline_false_escalations_per_1000"] > 900.0:
        reasons.append("baseline automatically escalates more than 90% of normal events")
    if overlap_fraction > 0.5:
        reasons.append("most final sequences exactly match an official-train event pattern")
    reasons.extend(
        [
            "HDFS labels are sequence-level rather than AIT alert-level incident labels",
            "HDFS has no timestamps for the one-day candidate rule",
        ]
    )
    return {
        "status": "post_unblinding_validity_audit_cannot_change_frozen_decision",
        "formal_frozen_external_decision": decision["external_claim_succeeds"],
        "operational_safe_superiority_supported": False,
        "reason": reasons,
        "baseline_mean_review_workload": mean_workload,
        "baseline_false_escalations_per_1000": decision[
            "mean_baseline_false_escalations_per_1000"
        ],
        "frozen_relative_safety_limit_per_1000": safety_limit,
        "physical_maximum_false_escalations_per_1000": 1000.0,
        "safety_criterion_is_nonbinding": safety_limit > 1000.0,
        "final_train_pattern_overlap_sequences": int(overlap.sum()),
        "final_sequence_count": int(len(overlap)),
        "final_train_pattern_overlap_fraction": overlap_fraction,
        "nonoverlap_final_sequences": int((~overlap).sum()),
        "nonoverlap_anomalous_sequences": int(nonoverlap_incident.sum()),
        "nonoverlap_normal_sequences": int(nonoverlap_normal.sum()),
        "nonoverlap_seed_metrics": rows,
        "nonoverlap_mean_recall_gain": float(
            np.mean([row["nonoverlap_recall_gain"] for row in rows])
        ),
        "nonoverlap_recall_gain_sequence_bootstrap_95_percent_ci": np.quantile(
            recall_bootstrap, [0.025, 0.975]
        ).tolist(),
        "nonoverlap_safety_violation_sequence_bootstrap_95_percent_ci": np.quantile(
            safety_bootstrap, [0.025, 0.975]
        ).tolist(),
        "interpretation": (
            "The positive recall signal is real for this benchmark encoding, but the "
            "benchmark cannot support a useful or safe alert-triage superiority claim."
        ),
    }


def markdown(result: dict) -> str:
    lines = [
        "# HDFS post-unblinding validity audit",
        "",
        f"- Frozen numerical decision: `{result['formal_frozen_external_decision']}`",
        f"- Operational safe-superiority evidence: `{result['operational_safe_superiority_supported']}`",
        f"- Baseline review workload: `{result['baseline_mean_review_workload']:.6f}`",
        "- Baseline false escalation / 1000: "
        f"`{result['baseline_false_escalations_per_1000']:.6f}`",
        "- Frozen safety limit / 1000: "
        f"`{result['frozen_relative_safety_limit_per_1000']:.6f}` "
        "(physical maximum `1000`)",
        "- Exact-pattern overlap: "
        f"`{result['final_train_pattern_overlap_sequences']}/"
        f"{result['final_sequence_count']}` "
        f"(`{result['final_train_pattern_overlap_fraction']:.2%}`)",
        "- Non-overlap recall-gain 95% CI: "
        f"`[{result['nonoverlap_recall_gain_sequence_bootstrap_95_percent_ci'][0]:.6f}, "
        f"{result['nonoverlap_recall_gain_sequence_bootstrap_95_percent_ci'][1]:.6f}]`",
        "- Non-overlap safety-violation 95% CI: "
        f"`[{result['nonoverlap_safety_violation_sequence_bootstrap_95_percent_ci'][0]:.6f}, "
        f"{result['nonoverlap_safety_violation_sequence_bootstrap_95_percent_ci'][1]:.6f}]` "
        "(positive values fail non-inferiority)",
        "",
        "The formal pass is retained for auditability, but its safety condition is "
        "non-binding at the observed baseline. It is not evidence of a usable safer system.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    result = analyze()
    output = RESULTS / "validity_audit.json"
    write_json(output, result)
    output.with_suffix(".md").write_text(markdown(result), encoding="utf-8")
    print(markdown(result))


if __name__ == "__main__":
    main()
