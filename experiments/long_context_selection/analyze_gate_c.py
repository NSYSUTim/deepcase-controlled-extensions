"""Decide whether the label-free selector advances beyond design scenarios."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_gate_b import default_point, load_result
from src.common import EXPERIMENT, load_config, write_json
from src.evaluation import choose_workload_matched_point


def analyze() -> dict:
    config = load_config()
    scenarios = list(config["design"]["design_scenarios"])
    seeds = [int(value) for value in config["design"]["seeds"]]
    rows = []
    for scenario in scenarios:
        receipt_path = EXPERIMENT / "artifacts" / f"association_{scenario}.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["association_abs10_uses_test_labels"]:
            raise AssertionError("deployable selector receipt reports test-label use")
        for seed in seeds:
            baseline = load_result("full", scenario, "last10", seed)
            selector = load_result("full", scenario, "association_abs10", seed)
            _, baseline_point = default_point(baseline)
            target_workload = baseline_point["test"]["review_workload"]
            index = choose_workload_matched_point(
                selector["operating_points"], target_workload
            )
            selected = selector["operating_points"][index]
            if baseline["context_length"] != 10 or selector["context_length"] != 10:
                raise AssertionError("Gate C is not budget matched at ten positions")
            base_metrics = baseline_point["test"]
            select_metrics = selected["test"]
            rows.append({
                "scenario": scenario,
                "seed": seed,
                "selected_eps": selected["eps"],
                "selected_threshold": selected["threshold"],
                "baseline_workload": target_workload,
                "selector_workload": select_metrics["review_workload"],
                "workload_difference": select_metrics["review_workload"] - target_workload,
                "baseline_automatic_incident_recall": base_metrics[
                    "automatic_incident_recall"
                ],
                "selector_automatic_incident_recall": select_metrics[
                    "automatic_incident_recall"
                ],
                "recall_gain": select_metrics["automatic_incident_recall"]
                - base_metrics["automatic_incident_recall"],
                "baseline_false_escalations_per_1000": base_metrics[
                    "false_escalations_per_1000_nonincidents"
                ],
                "selector_false_escalations_per_1000": select_metrics[
                    "false_escalations_per_1000_nonincidents"
                ],
                "false_escalation_difference_per_1000": select_metrics[
                    "false_escalations_per_1000_nonincidents"
                ] - base_metrics["false_escalations_per_1000_nonincidents"],
                "baseline_strict_macro_f1": base_metrics["strict_macro_f1"],
                "selector_strict_macro_f1": select_metrics["strict_macro_f1"],
            })
    effects = {}
    for scenario in scenarios:
        subset = [row for row in rows if row["scenario"] == scenario]
        effects[scenario] = {
            "mean_recall_gain": float(np.mean([row["recall_gain"] for row in subset])),
            "seed_recall_gains": [row["recall_gain"] for row in subset],
            "mean_workload_difference": float(
                np.mean([row["workload_difference"] for row in subset])
            ),
            "mean_false_escalation_difference_per_1000": float(
                np.mean([row["false_escalation_difference_per_1000"] for row in subset])
            ),
        }
    gains = [effects[scenario]["mean_recall_gain"] for scenario in scenarios]
    each = float(config["gates"]["selector_gain_each"])
    one = float(config["gates"]["selector_gain_one"])
    passed = all(value >= each for value in gains) or (
        max(gains) >= one and min(gains) >= 0.0
    )
    return {
        "gate_c_passed": bool(passed),
        "selector": "association_abs10",
        "selector_uses_test_labels": False,
        "context_positions": 10,
        "effects": effects,
        "matched_rows": rows,
        "advance_to_confirmatory_scenarios": bool(passed),
    }


def markdown(result: dict) -> str:
    lines = [
        "# Gate C: label-free selector feasibility",
        "",
        f"- Gate passed: `{result['gate_c_passed']}`",
        f"- Selector: `{result['selector']}`",
        f"- Uses held-out labels at retrieval: `{result['selector_uses_test_labels']}`",
        f"- Context positions: `{result['context_positions']}`",
        "",
        "| Scenario | Mean strict-recall gain | Mean workload difference | Mean false-escalation difference / 1000 |",
        "|---|---:|---:|---:|",
    ]
    for scenario, values in result["effects"].items():
        lines.append(
            f"| {scenario} | {values['mean_recall_gain']:.6f} | "
            f"{values['mean_workload_difference']:.6f} | "
            f"{values['mean_false_escalation_difference_per_1000']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    result = analyze()
    output = EXPERIMENT / "results" / "gate_c" / "gate_c_design_decision.json"
    write_json(output, result)
    output.with_suffix(".md").write_text(markdown(result), encoding="utf-8")
    print(markdown(result))


if __name__ == "__main__":
    main()
