import numpy as np
import pandas as pd

from analyze_results import exact_sign_flip_pvalue, paired_inference, select_point


def _point(workload, recall, precision, threshold):
    metrics = {
        "analyst_workload": workload,
        "triage_incident_recall": recall,
        "triage_precision": precision,
    }
    return {
        "eps": 0.1,
        "threshold": threshold,
        "validation": metrics.copy(),
        "test": metrics.copy(),
    }


def test_validation_selection_treats_undefined_precision_as_worst_tie_breaker():
    points = [
        _point(0.0, 0.0, None, 0.9),
        _point(0.1, 0.0, 0.5, 0.2),
    ]
    selected, _ = select_point(points, budget=0.1, policy="validation")
    assert selected["threshold"] == 0.2


def test_calibrated_selection_uses_largest_feasible_workload():
    points = [
        _point(0.10, 0.9, 0.8, 0.1),
        _point(0.20, 0.1, 0.2, 0.2),
        _point(0.21, 1.0, 1.0, 0.3),
    ]
    selected, _ = select_point(points, budget=0.20, policy="calibrated")
    assert selected["threshold"] == 0.2


def test_exact_sign_flip_is_one_sided_and_exact():
    # With two strictly positive paired differences, one of four sign
    # assignments has a mean at least as large as the observation.
    assert np.isclose(exact_sign_flip_pvalue(np.array([1.0, 2.0])), 0.25)


def test_incomplete_confirmatory_panel_has_no_inferential_p_value():
    rows = []
    for scenario in ("a", "b", "c", "d", "e"):
        for method, recall in (("deepcase", 0.5), ("proposed", 0.6)):
            rows.append(
                {
                    "label_column": "event_incident",
                    "imbalance_ratio": 1000,
                    "budget_design": "fixed_0.5",
                    "policy": "calibrated",
                    "analysis_scope": "overall",
                    "method": method,
                    "feasible": True,
                    "test_scenario": scenario,
                    "seed": 1,
                    "test_triage_incident_recall": recall,
                    "test_analyst_workload": 0.5,
                }
            )
    result = paired_inference(
        pd.DataFrame(rows),
        "proposed",
        "deepcase",
        {"a", "b", "c", "d", "e", "f"},
    ).iloc[0]
    assert not result["complete_confirmatory_panel"]
    assert np.isnan(result["exact_one_sided_sign_flip_p"])
    assert np.isnan(result["bootstrap_ci_low"])


def test_missing_one_planned_seed_pair_has_no_confirmatory_inference():
    rows = []
    for scenario in ("a", "b", "c", "d", "e", "f"):
        for seed in (1, 2, 3):
            for method, recall in (("deepcase", 0.5), ("proposed", 0.6)):
                rows.append(
                    {
                        "label_column": "event_incident",
                        "imbalance_ratio": 1000,
                        "budget_design": "deepcase_default_matched",
                        "policy": "deepcase_default_matched",
                        "analysis_scope": "overall",
                        "method": method,
                        "feasible": not (
                            scenario == "f" and seed == 3 and method == "proposed"
                        ),
                        "test_scenario": scenario,
                        "seed": seed,
                        "test_triage_incident_recall": recall,
                        "test_analyst_workload": 0.5,
                    }
                )
    result = paired_inference(
        pd.DataFrame(rows),
        "proposed",
        "deepcase",
        {"a", "b", "c", "d", "e", "f"},
    ).iloc[0]
    assert result["scenario_n"] == 6
    assert result["seed_scenario_pairs"] == 17
    assert result["expected_seed_scenario_pairs"] == 18
    assert not result["complete_seed_panel"]
    assert not result["complete_confirmatory_panel"]
    assert np.isnan(result["exact_one_sided_sign_flip_p"])
    assert np.isnan(result["bootstrap_ci_low"])
