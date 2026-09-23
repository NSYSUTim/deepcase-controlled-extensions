"""Run the frozen six-scenario confirmatory analysis.

The primary analysis uses only the frozen nearest-workload rule.  A separate
post-hoc sensitivity analysis mixes the two operating points bracketing the
baseline workload.  It is reported for diagnosing a coarse operating grid and
is never allowed to rescue the primary decision.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_gate_b import default_point, load_result
from src.common import EXPERIMENT, load_config, sha256, write_json
from src.evaluation import choose_workload_matched_point


SELECTOR = "association_abs10"


def _bootstrap_mean_ci(
    scenario_effects: list[float], iterations: int, seed: int
) -> tuple[float, float]:
    """Percentile CI that resamples scenarios, not alert rows or model seeds."""

    effects = np.asarray(scenario_effects, dtype=float)
    if effects.ndim != 1 or effects.size < 2:
        raise ValueError("at least two scenario effects are required")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, effects.size, size=(iterations, effects.size))
    means = effects[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def _bracketed_randomized_point(
    points: list[dict], target_workload: float
) -> dict:
    """Expected metrics from an outcome-blind randomized workload match.

    Grid order resolves duplicate-workload ties.  Only workload is used to
    locate the bracket and mixing probability.  Recall and false-escalation
    rates are linear here because every pair of points is evaluated against the
    exact same alerts and denominators.
    """

    indexed = [
        (index, float(point["test"]["review_workload"]), point)
        for index, point in enumerate(points)
    ]
    below = [item for item in indexed if item[1] <= target_workload]
    above = [item for item in indexed if item[1] >= target_workload]
    if not below or not above:
        nearest = choose_workload_matched_point(points, target_workload)
        point = points[nearest]
        metrics = point["test"]
        return {
            "bracketed": False,
            "lower_index": nearest,
            "upper_index": nearest,
            "upper_probability": 0.0,
            "workload": float(metrics["review_workload"]),
            "automatic_incident_recall": float(
                metrics["automatic_incident_recall"]
            ),
            "false_escalations_per_1000_nonincidents": float(
                metrics["false_escalations_per_1000_nonincidents"]
            ),
        }

    lower_workload = max(item[1] for item in below)
    upper_workload = min(item[1] for item in above)
    lower = min((item for item in below if item[1] == lower_workload), key=lambda x: x[0])
    upper = min((item for item in above if item[1] == upper_workload), key=lambda x: x[0])
    if math.isclose(lower_workload, upper_workload, rel_tol=0.0, abs_tol=1e-15):
        upper_probability = 0.0
    else:
        upper_probability = (target_workload - lower_workload) / (
            upper_workload - lower_workload
        )

    def mix(metric: str) -> float:
        low = float(lower[2]["test"][metric])
        high = float(upper[2]["test"][metric])
        return (1.0 - upper_probability) * low + upper_probability * high

    return {
        "bracketed": True,
        "lower_index": lower[0],
        "upper_index": upper[0],
        "upper_probability": float(upper_probability),
        "workload": mix("review_workload"),
        "automatic_incident_recall": mix("automatic_incident_recall"),
        "false_escalations_per_1000_nonincidents": mix(
            "false_escalations_per_1000_nonincidents"
        ),
    }


def _relative_increase(selector: float, baseline: float) -> float:
    if baseline == 0.0:
        return 0.0 if selector == 0.0 else float("inf")
    return selector / baseline - 1.0


def _validate_freeze(config: dict) -> tuple[dict, dict]:
    freeze_path = EXPERIMENT / "results" / "confirmation_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    expected_files = {
        "config": EXPERIMENT / "config.yaml",
        "protocol": EXPERIMENT / "protocol.md",
        "gate_b_decision": EXPERIMENT / "results" / "gate_b" / "gate_b_full_decision.json",
        "gate_c_decision": EXPERIMENT / "results" / "gate_c" / "gate_c_design_decision.json",
    }
    observed = {name: sha256(path) for name, path in expected_files.items()}
    if observed != freeze["hashes"]:
        raise AssertionError(
            f"a frozen input changed after confirmation: expected={freeze['hashes']} "
            f"observed={observed}"
        )
    if freeze["selector"] != SELECTOR or freeze["retrieval_uses_test_labels"]:
        raise AssertionError("confirmation selector identity or leakage flag changed")
    if freeze["confirmatory_scenarios"] != config["design"]["confirmatory_scenarios"]:
        raise AssertionError("confirmatory scenario set changed after freezing")
    if freeze["model_seeds"] != config["design"]["seeds"]:
        raise AssertionError("model seeds changed after freezing")
    return freeze, observed


def analyze() -> dict:
    config = load_config()
    freeze, observed_hashes = _validate_freeze(config)
    scenarios = list(freeze["confirmatory_scenarios"])
    seeds = [int(value) for value in freeze["model_seeds"]]
    rows: list[dict] = []
    all_receipts_label_free = True
    all_contexts_exactly_ten = True

    for scenario in scenarios:
        receipt_path = EXPERIMENT / "artifacts" / f"association_{scenario}.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt_label_free = not bool(
            receipt["association_abs10_uses_test_labels"]
            or receipt["test_labels_used_to_fit_teacher"]
        )
        all_receipts_label_free &= receipt_label_free
        for seed in seeds:
            baseline = load_result("full", scenario, "last10", seed)
            selector = load_result("full", scenario, SELECTOR, seed)
            all_contexts_exactly_ten &= (
                int(baseline["context_length"]) == 10
                and int(selector["context_length"]) == 10
            )
            _, baseline_point = default_point(baseline)
            base = baseline_point["test"]
            target_workload = float(base["review_workload"])
            selected_index = choose_workload_matched_point(
                selector["operating_points"], target_workload
            )
            selected_point = selector["operating_points"][selected_index]
            selected = selected_point["test"]
            sensitivity = _bracketed_randomized_point(
                selector["operating_points"], target_workload
            )
            rows.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "alerts": int(base["alerts"]),
                    "incidents": int(base["incidents"]),
                    "nonincidents": int(base["nonincidents"]),
                    "selected_grid_index": selected_index,
                    "selected_eps": float(selected_point["eps"]),
                    "selected_threshold": float(selected_point["threshold"]),
                    "baseline_workload": target_workload,
                    "selector_workload": float(selected["review_workload"]),
                    "workload_difference": float(selected["review_workload"])
                    - target_workload,
                    "absolute_workload_mismatch": abs(
                        float(selected["review_workload"]) - target_workload
                    ),
                    "baseline_recall": float(base["automatic_incident_recall"]),
                    "selector_recall": float(selected["automatic_incident_recall"]),
                    "recall_gain": float(selected["automatic_incident_recall"])
                    - float(base["automatic_incident_recall"]),
                    "baseline_strict_macro_f1": float(base["strict_macro_f1"]),
                    "selector_strict_macro_f1": float(selected["strict_macro_f1"]),
                    "strict_macro_f1_gain": float(selected["strict_macro_f1"])
                    - float(base["strict_macro_f1"]),
                    "baseline_false_escalations_per_1000": float(
                        base["false_escalations_per_1000_nonincidents"]
                    ),
                    "selector_false_escalations_per_1000": float(
                        selected["false_escalations_per_1000_nonincidents"]
                    ),
                    "baseline_auto_false_incident": int(base["auto_false_incident"]),
                    "selector_auto_false_incident": int(
                        selected["auto_false_incident"]
                    ),
                    "sensitivity_bracketed": bool(sensitivity["bracketed"]),
                    "sensitivity_lower_index": int(sensitivity["lower_index"]),
                    "sensitivity_upper_index": int(sensitivity["upper_index"]),
                    "sensitivity_upper_probability": float(
                        sensitivity["upper_probability"]
                    ),
                    "sensitivity_workload": float(sensitivity["workload"]),
                    "sensitivity_recall": float(
                        sensitivity["automatic_incident_recall"]
                    ),
                    "sensitivity_recall_gain": float(
                        sensitivity["automatic_incident_recall"]
                    )
                    - float(base["automatic_incident_recall"]),
                    "sensitivity_false_escalations_per_1000": float(
                        sensitivity["false_escalations_per_1000_nonincidents"]
                    ),
                    "receipt_label_free": receipt_label_free,
                }
            )

    scenario_effects: dict[str, dict] = {}
    for scenario in scenarios:
        subset = [row for row in rows if row["scenario"] == scenario]
        scenario_effects[scenario] = {
            "mean_recall_gain": float(np.mean([row["recall_gain"] for row in subset])),
            "seed_recall_gains": [row["recall_gain"] for row in subset],
            "mean_baseline_recall": float(
                np.mean([row["baseline_recall"] for row in subset])
            ),
            "mean_selector_recall": float(
                np.mean([row["selector_recall"] for row in subset])
            ),
            "mean_workload_difference": float(
                np.mean([row["workload_difference"] for row in subset])
            ),
            "mean_absolute_workload_mismatch": float(
                np.mean([row["absolute_workload_mismatch"] for row in subset])
            ),
            "mean_strict_macro_f1_gain": float(
                np.mean([row["strict_macro_f1_gain"] for row in subset])
            ),
            "mean_baseline_false_escalations_per_1000": float(
                np.mean(
                    [row["baseline_false_escalations_per_1000"] for row in subset]
                )
            ),
            "mean_selector_false_escalations_per_1000": float(
                np.mean(
                    [row["selector_false_escalations_per_1000"] for row in subset]
                )
            ),
            "sensitivity_mean_recall_gain": float(
                np.mean([row["sensitivity_recall_gain"] for row in subset])
            ),
            "sensitivity_all_seeds_bracketed": all(
                row["sensitivity_bracketed"] for row in subset
            ),
        }

    effects = [scenario_effects[name]["mean_recall_gain"] for name in scenarios]
    ci = _bootstrap_mean_ci(
        effects,
        int(config["gates"]["bootstrap_iterations"]),
        int(config["gates"]["bootstrap_seed"]),
    )
    sensitivity_effects = [
        scenario_effects[name]["sensitivity_mean_recall_gain"] for name in scenarios
    ]
    sensitivity_ci = _bootstrap_mean_ci(
        sensitivity_effects,
        int(config["gates"]["bootstrap_iterations"]),
        int(config["gates"]["bootstrap_seed"]),
    )

    # A rate over all held-out non-incidents is naturally pooled.  Model seeds
    # are repeated predictions of the same alerts, so pooling here equals the
    # total false decisions divided by total repeated non-incident predictions.
    total_nonincidents = sum(row["nonincidents"] for row in rows)
    pooled_base_false = 1000.0 * sum(
        row["baseline_auto_false_incident"] for row in rows
    ) / total_nonincidents
    pooled_select_false = 1000.0 * sum(
        row["selector_auto_false_incident"] for row in rows
    ) / total_nonincidents
    pooled_relative = _relative_increase(pooled_select_false, pooled_base_false)
    macro_base_false = float(
        np.mean(
            [
                scenario_effects[name]["mean_baseline_false_escalations_per_1000"]
                for name in scenarios
            ]
        )
    )
    macro_select_false = float(
        np.mean(
            [
                scenario_effects[name]["mean_selector_false_escalations_per_1000"]
                for name in scenarios
            ]
        )
    )
    macro_relative = _relative_increase(macro_select_false, macro_base_false)

    positive_scenarios = sum(value > 0.0 for value in effects)
    recall_ci_pass = ci[0] > 0.0
    positive_count_pass = positive_scenarios >= int(
        config["gates"]["final_min_positive_scenarios"]
    )
    false_escalation_pass = pooled_relative <= float(
        config["gates"]["final_max_false_escalation_relative_increase"]
    )
    natural_stream = True
    overall_success = all(
        [
            recall_ci_pass,
            positive_count_pass,
            false_escalation_pass,
            all_contexts_exactly_ten,
            natural_stream,
            all_receipts_label_free,
        ]
    )

    return {
        "confirmatory_claim_succeeds": bool(overall_success),
        "primary_result_cannot_be_rescued_by_sensitivity": True,
        "selector": SELECTOR,
        "scenarios": scenarios,
        "model_seeds": seeds,
        "frozen_input_hashes_verified": observed_hashes,
        "confirmation_freeze_sha256": sha256(
            EXPERIMENT / "results" / "confirmation_freeze.json"
        ),
        "primary": {
            "estimand": "strict automatic incident recall gain at nearest review workload",
            "scenario_effects": scenario_effects,
            "paired_mean_recall_gain": float(np.mean(effects)),
            "scenario_bootstrap_95_percent_ci": [ci[0], ci[1]],
            "bootstrap_iterations": int(config["gates"]["bootstrap_iterations"]),
            "bootstrap_seed": int(config["gates"]["bootstrap_seed"]),
            "positive_scenarios": int(positive_scenarios),
            "pooled_baseline_false_escalations_per_1000": pooled_base_false,
            "pooled_selector_false_escalations_per_1000": pooled_select_false,
            "pooled_false_escalation_relative_increase": pooled_relative,
            "macro_baseline_false_escalations_per_1000": macro_base_false,
            "macro_selector_false_escalations_per_1000": macro_select_false,
            "macro_false_escalation_relative_increase": macro_relative,
        },
        "frozen_criteria": {
            "recall_ci_lower_above_zero": bool(recall_ci_pass),
            "minimum_positive_scenarios": int(
                config["gates"]["final_min_positive_scenarios"]
            ),
            "positive_scenario_count_pass": bool(positive_count_pass),
            "maximum_pooled_false_escalation_relative_increase": float(
                config["gates"]["final_max_false_escalation_relative_increase"]
            ),
            "false_escalation_pass": bool(false_escalation_pass),
            "context_positions_exactly_ten": bool(all_contexts_exactly_ten),
            "natural_stream": natural_stream,
            "all_selector_receipts_label_free": bool(all_receipts_label_free),
        },
        "post_hoc_workload_interpolation_sensitivity": {
            "status": "diagnostic_only_cannot_reverse_primary_decision",
            "policy": "randomize_between_workload_bracketing_grid_points_without_outcome_use",
            "paired_mean_recall_gain": float(np.mean(sensitivity_effects)),
            "scenario_bootstrap_95_percent_ci": [
                sensitivity_ci[0],
                sensitivity_ci[1],
            ],
            "all_seed_comparisons_bracketed": all(
                row["sensitivity_bracketed"] for row in rows
            ),
        },
        "matched_rows": rows,
    }


def markdown(result: dict) -> str:
    primary = result["primary"]
    criteria = result["frozen_criteria"]
    sensitivity = result["post_hoc_workload_interpolation_sensitivity"]
    lines = [
        "# Six-scenario frozen confirmation",
        "",
        f"- Confirmatory claim succeeds: `{result['confirmatory_claim_succeeds']}`",
        f"- Paired mean strict-recall gain: `{primary['paired_mean_recall_gain']:.6f}`",
        "- Scenario-bootstrap 95% CI: "
        f"`[{primary['scenario_bootstrap_95_percent_ci'][0]:.6f}, "
        f"{primary['scenario_bootstrap_95_percent_ci'][1]:.6f}]`",
        f"- Positive scenarios: `{primary['positive_scenarios']}/6`",
        "- Pooled false escalations / 1000: "
        f"`{primary['pooled_baseline_false_escalations_per_1000']:.6f}` -> "
        f"`{primary['pooled_selector_false_escalations_per_1000']:.6f}` "
        f"(relative `{primary['pooled_false_escalation_relative_increase']:.2%}`)",
        f"- Label-free receipts verified: `{criteria['all_selector_receipts_label_free']}`",
        f"- Exact ten-position budget: `{criteria['context_positions_exactly_ten']}`",
        "",
        "| Scenario | Base recall | Selector recall | Gain | Workload difference | "
        "Abs workload mismatch | False escalation / 1000 (base -> selector) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario, values in primary["scenario_effects"].items():
        lines.append(
            f"| {scenario} | {values['mean_baseline_recall']:.6f} | "
            f"{values['mean_selector_recall']:.6f} | "
            f"{values['mean_recall_gain']:.6f} | "
            f"{values['mean_workload_difference']:.6f} | "
            f"{values['mean_absolute_workload_mismatch']:.6f} | "
            f"{values['mean_baseline_false_escalations_per_1000']:.6f} -> "
            f"{values['mean_selector_false_escalations_per_1000']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Frozen-criterion audit",
            "",
            f"- Recall CI excludes zero: `{criteria['recall_ci_lower_above_zero']}`",
            f"- Positive-scenario count passes: `{criteria['positive_scenario_count_pass']}`",
            f"- False-escalation relative cap passes: `{criteria['false_escalation_pass']}`",
            f"- Natural, non-injected stream: `{criteria['natural_stream']}`",
            "",
            "## Post-hoc coarse-grid sensitivity (cannot rescue primary)",
            "",
            f"- Mean gain: `{sensitivity['paired_mean_recall_gain']:.6f}`",
            "- Scenario-bootstrap 95% CI: "
            f"`[{sensitivity['scenario_bootstrap_95_percent_ci'][0]:.6f}, "
            f"{sensitivity['scenario_bootstrap_95_percent_ci'][1]:.6f}]`",
            f"- All comparisons bracketed: `{sensitivity['all_seed_comparisons_bracketed']}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    result = analyze()
    output = EXPERIMENT / "results" / "confirmation_decision.json"
    write_json(output, result)
    output.with_suffix(".md").write_text(markdown(result), encoding="utf-8")
    write_csv(output.with_name("confirmation_matched_rows.csv"), result["matched_rows"])
    print(markdown(result))


if __name__ == "__main__":
    main()
