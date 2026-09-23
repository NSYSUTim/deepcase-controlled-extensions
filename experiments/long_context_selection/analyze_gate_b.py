"""Apply the frozen workload match and decide the Gate-B launch criterion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.common import EXPERIMENT, load_config, write_json
from src.evaluation import choose_workload_matched_point


def load_result(stage: str, scenario: str, method: str, seed: int) -> dict:
    path = EXPERIMENT / "results" / "gate_b" / stage / scenario / method / f"seed_{seed}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("status") != "complete":
        raise RuntimeError(f"incomplete result: {path}")
    return record


def default_point(record: dict) -> tuple[int, dict]:
    for index, point in enumerate(record["operating_points"]):
        if point["eps"] == 0.10 and point["threshold"] == 0.20:
            return index, point
    raise ValueError("last10 default operating point is absent")


def decide(config: dict, effects: dict) -> tuple[bool, list[str]]:
    passed = []
    scenarios = list(config["design"]["design_scenarios"])
    each = float(config["gates"]["upper_bound_gain_each"])
    one = float(config["gates"]["upper_bound_gain_one"])
    for method in config["gate_b"]["decision_upper_bounds"]:
        delta = [effects[scenario][method]["mean_recall_gain"] for scenario in scenarios]
        pass_each = all(value >= each for value in delta)
        pass_one = max(delta) >= one and min(delta) >= 0.0
        if pass_each or pass_one:
            passed.append(method)
    return bool(passed), passed


def analyze(stage: str) -> dict:
    config = load_config()
    scenarios = list(config["design"]["design_scenarios"])
    seeds = [int(value) for value in config["design"]["seeds"]]
    if stage in {"smoke", "pilot"}:
        seeds = seeds[:1]
        if stage == "smoke":
            scenarios = ["russellmitchell"]
    effects: dict[str, dict] = {}
    rows = []
    methods = ["last10", *config["gate_b"]["decision_upper_bounds"]]
    for scenario in scenarios:
        effects[scenario] = {}
        for seed in seeds:
            baseline_record = load_result(stage, scenario, "last10", seed)
            baseline_index, baseline_point = default_point(baseline_record)
            target_workload = baseline_point["test"]["review_workload"]
            baseline_recall = baseline_point["test"]["automatic_incident_recall"]
            for method in methods:
                record = load_result(stage, scenario, method, seed)
                if method == "last10":
                    chosen_index = baseline_index
                else:
                    chosen_index = choose_workload_matched_point(
                        record["operating_points"], target_workload
                    )
                chosen = record["operating_points"][chosen_index]
                metrics = chosen["test"]
                rows.append({
                    "scenario": scenario,
                    "seed": seed,
                    "method": method,
                    "baseline_workload": target_workload,
                    "eps": chosen["eps"],
                    "threshold": chosen["threshold"],
                    "workload": metrics["review_workload"],
                    "workload_difference": metrics["review_workload"] - target_workload,
                    "automatic_incident_recall": metrics["automatic_incident_recall"],
                    "recall_gain": metrics["automatic_incident_recall"] - baseline_recall,
                    "false_escalations_per_1000_nonincidents": metrics[
                        "false_escalations_per_1000_nonincidents"
                    ],
                    "strict_macro_f1": metrics["strict_macro_f1"],
                })
        for method in methods:
            subset = [row for row in rows if row["scenario"] == scenario and row["method"] == method]
            effects[scenario][method] = {
                "mean_recall_gain": float(np.mean([row["recall_gain"] for row in subset])),
                "mean_workload_difference": float(np.mean([row["workload_difference"] for row in subset])),
                "seed_recall_gains": [row["recall_gain"] for row in subset],
            }
    if stage == "full":
        gate_passed, passing_methods = decide(config, effects)
    else:
        gate_passed, passing_methods = None, []
    return {
        "stage": stage,
        "gate_b_passed": gate_passed,
        "passing_upper_bounds": passing_methods,
        "effects": effects,
        "matched_rows": rows,
        "decision_note": (
            "Only full-stage, three-seed results trigger the frozen gate."
            if stage != "full" else
            "A true value launches Gate C; false records the natural selector hypothesis as failed."
        ),
    }


def markdown(result: dict) -> str:
    lines = [
        "# Gate B: model upper-bound result",
        "",
        f"- Stage: `{result['stage']}`",
        f"- Frozen gate passed: `{result['gate_b_passed']}`",
        f"- Passing upper bounds: `{result['passing_upper_bounds']}`",
        "",
        "| Scenario | Method | Mean strict-recall gain | Mean workload difference |",
        "|---|---:|---:|---:|",
    ]
    for scenario, methods in result["effects"].items():
        for method, values in methods.items():
            lines.append(
                f"| {scenario} | {method} | {values['mean_recall_gain']:.6f} | "
                f"{values['mean_workload_difference']:.6f} |"
            )
    lines.extend([
        "",
        "Strict recall credits only automatic Incident decisions. Reject is unresolved",
        "and contributes to review workload, never to a successful detection.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["smoke", "pilot", "full"], default="full")
    args = parser.parse_args()
    result = analyze(args.stage)
    output = EXPERIMENT / "results" / "gate_b" / f"gate_b_{args.stage}_decision.json"
    write_json(output, result)
    output.with_suffix(".md").write_text(markdown(result), encoding="utf-8")
    print(markdown(result))


if __name__ == "__main__":
    main()
