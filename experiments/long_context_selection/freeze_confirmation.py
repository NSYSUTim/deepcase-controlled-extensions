"""Freeze the selected method before generating confirmatory selector contexts."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.common import EXPERIMENT, load_config, sha256, write_json


def main() -> None:
    config = load_config()
    gate_b = EXPERIMENT / "results" / "gate_b" / "gate_b_full_decision.json"
    gate_c = EXPERIMENT / "results" / "gate_c" / "gate_c_design_decision.json"
    for path in (gate_b, gate_c):
        if not path.exists():
            raise FileNotFoundError(path)
    decision_b = json.loads(gate_b.read_text(encoding="utf-8"))
    decision_c = json.loads(gate_c.read_text(encoding="utf-8"))
    if not decision_b["gate_b_passed"] or not decision_c["gate_c_passed"]:
        raise RuntimeError("both design gates must pass before confirmation")
    confirmation_contexts = [
        EXPERIMENT / "artifacts" / f"contexts_association_abs10_{scenario}.npy"
        for scenario in config["design"]["confirmatory_scenarios"]
    ]
    if any(path.exists() for path in confirmation_contexts):
        raise RuntimeError(
            "confirmatory selector artifacts already exist; cannot claim a new pre-outcome freeze"
        )
    output = EXPERIMENT / "results" / "confirmation_freeze.json"
    write_json(output, {
        "status": "frozen_before_confirmatory_selector_generation",
        "selector": "association_abs10",
        "selector_budget": 10,
        "candidate_pool": 100,
        "retrieval_uses_test_labels": False,
        "teacher_fit_scope": "training_scenarios_only_for_each_held_out_fold",
        "feature_table": "target_event_x_candidate_event_x_recency_band_log_odds",
        "ranking": "absolute_training_evidence_then_recency",
        "operating_point_rule": config["gate_b"]["workload_match"],
        "baseline_operating_point": {"eps": 0.10, "threshold": 0.20},
        "model_seeds": config["design"]["seeds"],
        "confirmatory_scenarios": config["design"]["confirmatory_scenarios"],
        "primary_metric": "strict_automatic_incident_recall",
        "final_success_criteria": {
            "scenario_bootstrap_ci_excludes_zero": True,
            "minimum_positive_scenarios": config["gates"]["final_min_positive_scenarios"],
            "maximum_false_escalation_relative_increase": config["gates"][
                "final_max_false_escalation_relative_increase"
            ],
            "context_positions": 10,
        },
        "hashes": {
            "protocol": sha256(EXPERIMENT / "protocol.md"),
            "config": sha256(EXPERIMENT / "config.yaml"),
            "gate_b_decision": sha256(gate_b),
            "gate_c_decision": sha256(gate_c),
        },
    })
    print(output)


if __name__ == "__main__":
    main()
