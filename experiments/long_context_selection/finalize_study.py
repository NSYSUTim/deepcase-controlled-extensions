"""Create a fail-closed, machine-readable decision for the complete study."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.common import EXPERIMENT, write_json


RESULTS = EXPERIMENT / "results"
ARTIFACTS = EXPERIMENT / "artifacts"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_decision() -> dict:
    retrieval = load(ARTIFACTS / "retrieval_audit.json")
    gate_b = load(RESULTS / "gate_b" / "gate_b_full_decision.json")
    gate_c = load(RESULTS / "gate_c" / "gate_c_design_decision.json")
    confirmation = load(RESULTS / "confirmation_decision.json")
    anchor = load(RESULTS / "anchor_design_decision.json")
    hdfs = load(RESULTS / "hdfs_external" / "final_decision.json")
    hdfs_audit = load(RESULTS / "hdfs_external" / "validity_audit.json")

    primary = confirmation["primary"]
    all_audit = retrieval["by_scenario"]["__all__"]

    # Fail loudly if the prose-level conclusion could drift away from the
    # frozen analyzers or if a diagnostic result is accidentally promoted.
    assert gate_b["gate_b_passed"] is True
    assert gate_c["advance_to_confirmatory_scenarios"] is True
    assert confirmation["confirmatory_claim_succeeds"] is False
    assert confirmation["frozen_criteria"]["false_escalation_pass"] is False
    assert anchor["status"] == (
        "post_failure_method_development_not_independent_confirmation"
    )
    assert hdfs["external_claim_succeeds"] is True
    assert hdfs_audit["operational_safe_superiority_supported"] is False
    assert hdfs_audit["safety_criterion_is_nonbinding"] is True

    return {
        "decision_semantics": (
            "fail_closed: a numerical or exploratory pass cannot override a failed "
            "confirmatory safety criterion or an invalid external operational claim"
        ),
        "research_question": (
            "Can a test-label-free selector choose exactly 10 positions from a "
            "strictly-past last-100 pool and outperform DeepCASE Last10 without an "
            "unacceptable false-escalation increase?"
        ),
        "gate_a_retrieval_audit": {
            "incident_targets": all_audit["incident_targets"],
            "last10_redundancy": all_audit["last10_redundancy_rate"],
            "mean_last10_unique_event_types": all_audit["last10_mean_unique_types"],
            "any_incident_displacement_at_10": all_audit[
                "any_incident_displacement_at_10"
            ],
            "same_group_displacement_at_10": all_audit[
                "same_group_displacement_at_10"
            ],
            "any_incident_predecessor_coverage_at_10": all_audit[
                "any_incident_predecessor_coverage"
            ]["10"],
            "any_incident_predecessor_coverage_at_100": all_audit[
                "any_incident_predecessor_coverage"
            ]["100"],
            "interpretation": (
                "Last10 is highly redundant, but labelled incident-predecessor "
                "displacement is too rare to support that specific mechanism."
            ),
        },
        "gate_b_fixed_budget_upper_bound": {
            "passed": gate_b["gate_b_passed"],
            "fox_oracle10_mean_recall_gain": gate_b["effects"]["fox"][
                "oracle10"
            ]["mean_recall_gain"],
            "russellmitchell_oracle10_mean_recall_gain": gate_b["effects"][
                "russellmitchell"
            ]["oracle10"]["mean_recall_gain"],
            "fox_class_oracle10_mean_recall_gain": gate_b["effects"]["fox"][
                "class_oracle10"
            ]["mean_recall_gain"],
            "russellmitchell_class_oracle10_mean_recall_gain": gate_b["effects"][
                "russellmitchell"
            ]["class_oracle10"]["mean_recall_gain"],
            "last100_model_status": (
                "excluded_before upper-bound outcomes: not budget matched, length and "
                "capacity confounded, and CPU smoke cache took about 40 minutes"
            ),
        },
        "gate_c_design": {
            "passed": gate_c["advance_to_confirmatory_scenarios"],
            "selector": gate_c["selector"],
            "uses_test_labels": gate_c["selector_uses_test_labels"],
            "context_positions": gate_c["context_positions"],
            "fox_mean_recall_gain": gate_c["effects"]["fox"]["mean_recall_gain"],
            "russellmitchell_mean_recall_gain": gate_c["effects"][
                "russellmitchell"
            ]["mean_recall_gain"],
        },
        "primary_ait_confirmation": {
            "claim_succeeds": confirmation["confirmatory_claim_succeeds"],
            "selector": confirmation["selector"],
            "mean_recall_gain": primary["paired_mean_recall_gain"],
            "scenario_bootstrap_95_percent_ci": primary[
                "scenario_bootstrap_95_percent_ci"
            ],
            "positive_scenarios": primary["positive_scenarios"],
            "scenario_count": len(primary["scenario_effects"]),
            "pooled_baseline_false_escalations_per_1000": primary[
                "pooled_baseline_false_escalations_per_1000"
            ],
            "pooled_selector_false_escalations_per_1000": primary[
                "pooled_selector_false_escalations_per_1000"
            ],
            "relative_false_escalation_increase": primary[
                "pooled_false_escalation_relative_increase"
            ],
            "failed_criterion": "false escalation increased by more than 5%",
        },
        "post_failure_ait_development": {
            "method": anchor["method"],
            "independent_confirmation": False,
            "v1_passed": anchor["v1"]["passed"],
            "v2_development_gate_passed": anchor["v2"]["passed"],
            "v2_safe_seed_count": anchor["v2"]["safe_seed_count"],
            "v2_seed_count": anchor["v2"]["seed_count"],
            "interpretation": (
                "recency anchoring is a candidate for a future preregistered study, "
                "not a rescue of the failed primary confirmation"
            ),
        },
        "hdfs_external": {
            "frozen_numerical_pass": hdfs["external_claim_succeeds"],
            "mean_recall_gain": hdfs["mean_recall_gain"],
            "recall_gain_sequence_bootstrap_95_percent_ci": hdfs[
                "recall_gain_sequence_bootstrap_95_percent_ci"
            ],
            "baseline_review_workload": hdfs_audit[
                "baseline_mean_review_workload"
            ],
            "baseline_false_escalations_per_1000": hdfs[
                "mean_baseline_false_escalations_per_1000"
            ],
            "frozen_safety_limit_per_1000": hdfs_audit[
                "frozen_relative_safety_limit_per_1000"
            ],
            "physical_maximum_per_1000": hdfs_audit[
                "physical_maximum_false_escalations_per_1000"
            ],
            "train_pattern_overlap_fraction": hdfs_audit[
                "final_train_pattern_overlap_fraction"
            ],
            "nonoverlap_recall_gain": hdfs_audit[
                "nonoverlap_mean_recall_gain"
            ],
            "nonoverlap_recall_gain_95_percent_ci": hdfs_audit[
                "nonoverlap_recall_gain_sequence_bootstrap_95_percent_ci"
            ],
            "nonoverlap_safety_violation_95_percent_ci": hdfs_audit[
                "nonoverlap_safety_violation_sequence_bootstrap_95_percent_ci"
            ],
            "nonoverlap_safety_violation_95_percent_ci": hdfs_audit[
                "nonoverlap_safety_violation_sequence_bootstrap_95_percent_ci"
            ],
            "operational_safe_superiority_supported": hdfs_audit[
                "operational_safe_superiority_supported"
            ],
            "interpretation": hdfs_audit["interpretation"],
        },
        "protocol_deviations_and_scope_limits": [
            (
                "Registered random10, unique-recent10, rarity10, and fixed "
                "relevance-diversity Gate-C controls were not completed before the "
                "association selector advanced. This weakens mechanism attribution "
                "but cannot reverse the already failed primary safety decision."
            ),
            (
                "The Last100 recurrent model was removed by a pre-outcome feasibility "
                "amendment; all decisive comparisons are the fair fixed-10 estimand."
            ),
            (
                "AIT-ADS is a synthetic eight-scenario testbed with unusually high "
                "event-level incident prevalence; seeds are technical repeats and "
                "scenarios are the independent units."
            ),
            (
                "HDFS has sequence-level anomaly labels, no timestamps, 88.42% exact "
                "official-train-pattern overlap in the final split, and cannot validate "
                "AIT alert-triage safety."
            ),
        ],
        "overall_conclusion": {
            "safe_superiority_over_last10_supported": False,
            "better_than_prior_overall_supported": False,
            "fixed_budget_recall_signal_supported": True,
            "definitive_outcome": "primary hypothesis failed",
            "defensible_thesis_contribution": (
                "A pre-outcome-frozen negative-result study: a training-only long-pool "
                "selector improves strict incident recall, but the gain is not safe "
                "because false escalations rise sharply; public HDFS reproduces a "
                "recall signal but is structurally incapable of validating the SOC "
                "safety claim."
            ),
        },
    }


def markdown(result: dict) -> str:
    primary = result["primary_ait_confirmation"]
    hdfs = result["hdfs_external"]
    return "\n".join(
        [
            "# Overall fail-closed study decision",
            "",
            "- Primary AIT safe-superiority claim: "
            f"`{primary['claim_succeeds']}`",
            "- AIT strict-recall gain: "
            f"`{primary['mean_recall_gain']:.6f}`; scenario-bootstrap 95% CI "
            f"`[{primary['scenario_bootstrap_95_percent_ci'][0]:.6f}, "
            f"{primary['scenario_bootstrap_95_percent_ci'][1]:.6f}]`",
            "- AIT false escalations / 1000: "
            f"`{primary['pooled_baseline_false_escalations_per_1000']:.6f}` -> "
            f"`{primary['pooled_selector_false_escalations_per_1000']:.6f}`",
            "- Frozen HDFS numerical pass: "
            f"`{hdfs['frozen_numerical_pass']}`",
            "- HDFS operational safe-superiority support: "
            f"`{hdfs['operational_safe_superiority_supported']}`",
            "- Overall better-than-prior claim: `False`",
            "",
            result["overall_conclusion"]["defensible_thesis_contribution"],
            "",
        ]
    )


def main() -> None:
    result = build_decision()
    output = RESULTS / "overall_decision.json"
    write_json(output, result)
    output.with_suffix(".md").write_text(markdown(result), encoding="utf-8")
    print(markdown(result))


if __name__ == "__main__":
    main()
