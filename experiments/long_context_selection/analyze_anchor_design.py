"""Record v1 failure and v2 development result for anchor5_assoc5."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_gate_b import default_point, load_result
from src.common import EXPERIMENT, write_json


SCENARIOS = ["fox", "russellmitchell"]
SEEDS = [1729, 2718, 31415]


def safety_limit(baseline_rate: float, version: str) -> float:
    if version == "v1_relative_only":
        return 1.05 * baseline_rate
    if version == "v2_hybrid_noninferiority":
        return baseline_rate + 1.0 if baseline_rate < 1.0 else 1.05 * baseline_rate
    raise ValueError(version)


def choose_safe_point(record: dict, target_workload: float, limit: float) -> int | None:
    eligible = []
    for order, point in enumerate(record["operating_points"]):
        validation_false = float(
            point["validation"]["false_escalations_per_1000_nonincidents"]
        )
        if validation_false <= limit:
            workload = float(point["test"]["review_workload"])
            eligible.append((abs(workload - target_workload), workload, order))
    return min(eligible)[2] if eligible else None


def analyze_version(version: str) -> dict:
    rows = []
    for scenario in SCENARIOS:
        for seed in SEEDS:
            baseline = load_result("full", scenario, "last10", seed)
            selector = load_result("full", scenario, "anchor5_assoc5", seed)
            _, baseline_point = default_point(baseline)
            base = baseline_point["test"]
            baseline_validation_false = float(
                baseline_point["validation"][
                    "false_escalations_per_1000_nonincidents"
                ]
            )
            limit = safety_limit(baseline_validation_false, version)
            selected_index = choose_safe_point(
                selector, float(base["review_workload"]), limit
            )
            row = {
                "scenario": scenario,
                "seed": seed,
                "baseline_validation_false_escalations_per_1000": baseline_validation_false,
                "validation_safety_limit_per_1000": limit,
                "safe_point_exists": selected_index is not None,
            }
            if selected_index is not None:
                point = selector["operating_points"][selected_index]
                metrics = point["test"]
                row.update(
                    {
                        "selected_eps": point["eps"],
                        "selected_threshold": point["threshold"],
                        "baseline_recall": base["automatic_incident_recall"],
                        "selector_recall": metrics["automatic_incident_recall"],
                        "recall_gain": metrics["automatic_incident_recall"]
                        - base["automatic_incident_recall"],
                        "baseline_workload": base["review_workload"],
                        "selector_workload": metrics["review_workload"],
                        "workload_difference": metrics["review_workload"]
                        - base["review_workload"],
                        "nonincidents": base["nonincidents"],
                        "baseline_auto_false_incident": base[
                            "auto_false_incident"
                        ],
                        "selector_auto_false_incident": metrics[
                            "auto_false_incident"
                        ],
                    }
                )
            rows.append(row)

    all_safe = all(row["safe_point_exists"] for row in rows)
    effects = {}
    if all_safe:
        for scenario in SCENARIOS:
            subset = [row for row in rows if row["scenario"] == scenario]
            effects[scenario] = {
                "mean_recall_gain": float(
                    np.mean([row["recall_gain"] for row in subset])
                ),
                "mean_workload_difference": float(
                    np.mean([row["workload_difference"] for row in subset])
                ),
            }
        nonincidents = sum(row["nonincidents"] for row in rows)
        baseline_false = 1000.0 * sum(
            row["baseline_auto_false_incident"] for row in rows
        ) / nonincidents
        selector_false = 1000.0 * sum(
            row["selector_auto_false_incident"] for row in rows
        ) / nonincidents
        test_limit = safety_limit(baseline_false, version)
        passed = (
            all(values["mean_recall_gain"] > 0.0 for values in effects.values())
            and selector_false <= test_limit
        )
    else:
        baseline_false = selector_false = test_limit = None
        passed = False
    return {
        "version": version,
        "passed": bool(passed),
        "all_seed_safety_sets_nonempty": all_safe,
        "safe_seed_count": sum(row["safe_point_exists"] for row in rows),
        "seed_count": len(rows),
        "scenario_effects": effects,
        "pooled_baseline_false_escalations_per_1000": baseline_false,
        "pooled_selector_false_escalations_per_1000": selector_false,
        "pooled_test_safety_limit_per_1000": test_limit,
        "rows": rows,
    }


def markdown(result: dict) -> str:
    lines = [
        "# anchor5_assoc5 AIT development gates",
        "",
        "The six earlier confirmation scenarios are not used in this table.",
        "",
        "| Version | Pass | Safe seeds | fox gain | russellmitchell gain | "
        "Pooled false escalation / 1000 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("v1", "v2"):
        value = result[name]
        effects = value["scenario_effects"]
        fox = effects.get("fox", {}).get("mean_recall_gain")
        russell = effects.get("russellmitchell", {}).get("mean_recall_gain")
        false_pair = (
            "not evaluated"
            if value["pooled_selector_false_escalations_per_1000"] is None
            else f"{value['pooled_baseline_false_escalations_per_1000']:.6f} -> "
            f"{value['pooled_selector_false_escalations_per_1000']:.6f}"
        )
        lines.append(
            f"| {value['version']} | {value['passed']} | "
            f"{value['safe_seed_count']}/{value['seed_count']} | "
            f"{fox if fox is not None else 'N/A'} | "
            f"{russell if russell is not None else 'N/A'} | {false_pair} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    result = {
        "status": "post_failure_method_development_not_independent_confirmation",
        "method": "anchor5_assoc5",
        "v1": analyze_version("v1_relative_only"),
        "v2": analyze_version("v2_hybrid_noninferiority"),
    }
    output = EXPERIMENT / "results" / "anchor_design_decision.json"
    write_json(output, result)
    output.with_suffix(".md").write_text(markdown(result), encoding="utf-8")
    print(markdown(result))


if __name__ == "__main__":
    main()
