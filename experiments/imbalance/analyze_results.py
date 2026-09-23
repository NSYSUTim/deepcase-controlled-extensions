"""Leakage-auditable fixed-workload selection and scenario-level inference."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon


METRICS = (
    "analyst_workload",
    "triage_incident_recall",
    "triage_precision",
    "relaxed_f1",
    "reject_rate",
    "incident_group_recall",
)


def load_runs(root: Path) -> list[dict]:
    runs: list[dict] = []
    for path in sorted(root.rglob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") == "complete" and record.get("operating_points"):
            record["_path"] = str(path.resolve())
            runs.append(record)
    return runs


def point_key(point: dict) -> tuple[float, float]:
    return (
        float(point.get("eps", -1.0)),
        float(point.get("threshold", point.get("rule_threshold", -1.0))),
    )


def finite_or_negative_infinity(value: object) -> float:
    """Give undefined selection metrics the worst possible rank."""

    if value is None:
        return float("-inf")
    numeric = float(value)
    return numeric if np.isfinite(numeric) else float("-inf")


def select_point(
    points: list[dict], budget: float, policy: str
) -> tuple[dict | None, str]:
    """Select without consulting test outcomes.

    ``validation`` uses validation labels and workload only. ``calibrated``
    additionally observes the unlabeled test-stream workload so that a SOC can
    fit its known review capacity; test recall/precision are never consulted.
    """

    if policy == "validation":
        feasible = [
            point
            for point in points
            if point["validation"]["analyst_workload"] <= budget
        ]
        if not feasible:
            return None, "no validation operating point at or below budget"
        return max(
            feasible,
            key=lambda point: (
                finite_or_negative_infinity(
                    point["validation"]["triage_incident_recall"]
                ),
                finite_or_negative_infinity(point["validation"]["triage_precision"]),
                finite_or_negative_infinity(point["validation"]["analyst_workload"]),
                tuple(-item for item in point_key(point)),
            ),
        ), "validation recall maximized subject to validation workload"
    if policy == "calibrated":
        feasible = [
            point for point in points if point["test"]["analyst_workload"] <= budget
        ]
        if not feasible:
            return None, "no observed unlabeled test workload at or below budget"
        return max(
            feasible,
            key=lambda point: (
                finite_or_negative_infinity(point["test"]["analyst_workload"]),
                finite_or_negative_infinity(
                    point["validation"]["triage_incident_recall"]
                ),
                finite_or_negative_infinity(point["validation"]["triage_precision"]),
                tuple(-item for item in point_key(point)),
            ),
        ), "closest workload below budget; label-based ties use validation only"
    raise ValueError(f"Unknown selection policy: {policy}")


def flatten_selection(
    record: dict,
    budget: float,
    policy: str,
    point: dict | None,
    reason: str,
    budget_design: str | None = None,
) -> dict:
    row = {
        "stage": record["stage"],
        "result_namespace": record.get("result_namespace"),
        "scoring_index_version": record.get("scoring_index_version"),
        "interpreter_query_iterations": record.get(
            "interpreter_query_iterations",
            record.get("config_snapshot", {})
            .get("interpreter", {})
            .get("query_iterations"),
        ),
        "label_column": record["label_column"],
        "test_scenario": record["test_scenario"],
        "validation_scenario": record["validation_scenario"],
        "imbalance_ratio": int(record["imbalance_ratio"]),
        "seed": int(record["seed"]),
        "method": record["method"],
        "budget": float(budget),
        "budget_design": budget_design or f"fixed_{budget:g}",
        "policy": policy,
        "feasible": point is not None,
        "selection_reason": reason,
        "source": record["_path"],
    }
    if point is None:
        return row
    row.update(
        {
            "eps": point.get("eps"),
            "threshold": point.get("threshold", point.get("rule_threshold")),
        }
    )
    for split in ("validation", "test"):
        for metric in METRICS:
            row[f"{split}_{metric}"] = point[split].get(metric)
        diagnostic = point.get("diagnostic_strata", {}).get(split, {}).get(
            "ambiguous_rule", {}
        )
        row[f"{split}_ambiguous_rows"] = diagnostic.get("rows", 0)
        diagnostic_metrics = diagnostic.get("metrics") or {}
        for metric in METRICS:
            row[f"{split}_ambiguous_{metric}"] = diagnostic_metrics.get(metric)
    return row


def exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    if len(differences) == 0:
        return float("nan")
    observed = differences.mean()
    if len(differences) <= 20:
        signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(differences))))
        null_means = (signs * differences).mean(axis=1)
        return float(np.mean(null_means >= observed - 1e-15))
    rng = np.random.default_rng(20_260_903)
    signs = rng.choice((-1.0, 1.0), size=(100_000, len(differences)))
    null_means = (signs * differences).mean(axis=1)
    return float((1 + np.sum(null_means >= observed)) / (len(null_means) + 1))


def scenario_bootstrap_ci(differences: np.ndarray) -> tuple[float, float]:
    differences = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(20_260_903)
    samples = rng.choice(
        differences, size=(20_000, len(differences)), replace=True
    ).mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return float(lower), float(upper)


def paired_inference(
    selected: pd.DataFrame,
    proposed: str,
    baseline: str,
    confirmatory_scenarios: set[str] | None = None,
    recall_column: str = "test_triage_incident_recall",
    workload_column: str = "test_analyst_workload",
    analysis_scope: str = "overall",
) -> pd.DataFrame:
    rows: list[dict] = []
    grouping = ["label_column", "imbalance_ratio", "budget_design", "policy"]
    for keys, frame in selected.groupby(grouping, dropna=False):
        eligible = frame[frame["method"].isin([proposed, baseline])]
        if confirmatory_scenarios is not None:
            eligible = eligible[
                eligible["test_scenario"].isin(confirmatory_scenarios)
            ]
        candidates = eligible[eligible["feasible"]]
        if candidates.empty:
            continue
        seed_means = (
            candidates.groupby(["test_scenario", "seed", "method"], as_index=False)[
                [recall_column, workload_column]
            ]
            .mean()
        )
        pivot = seed_means.pivot_table(
            index=["test_scenario", "seed"],
            columns="method",
            values=[recall_column, workload_column],
        )
        required = [
            (recall_column, proposed),
            (recall_column, baseline),
            (workload_column, proposed),
            (workload_column, baseline),
        ]
        if any(column not in pivot.columns for column in required):
            continue
        complete = pivot.dropna(subset=required)
        if complete.empty:
            continue
        per_seed = pd.DataFrame(
            {
                "delta_recall": complete[(recall_column, proposed)]
                - complete[(recall_column, baseline)],
                "delta_workload": complete[(workload_column, proposed)]
                - complete[(workload_column, baseline)],
            }
        )
        # Random seeds are repeated measurements, not independent SOCs.
        by_scenario = per_seed.groupby(level="test_scenario").mean()
        difference = by_scenario["delta_recall"].to_numpy()
        expected_scenario_n = (
            len(confirmatory_scenarios)
            if confirmatory_scenarios is not None
            else len(by_scenario)
        )
        expected_seed_n = int(eligible["seed"].nunique())
        expected_seed_scenario_pairs = expected_scenario_n * expected_seed_n
        complete_seed_panel = len(per_seed) == expected_seed_scenario_pairs
        complete_confirmatory_panel = (
            len(by_scenario) == expected_scenario_n and complete_seed_panel
        )
        if complete_confirmatory_panel:
            lower, upper = scenario_bootstrap_ci(difference)
            exact_p = exact_sign_flip_pvalue(difference)
            try:
                if np.allclose(difference, 0.0):
                    raise ValueError("all paired differences are zero")
                wilcoxon_p = float(
                    wilcoxon(difference, alternative="greater", zero_method="pratt").pvalue
                )
            except ValueError:
                wilcoxon_p = float("nan")
        else:
            # A feasibility-selected subset does not estimate the predeclared
            # six-scenario confirmatory effect. Keep its descriptive mean, but
            # never attach inferential quantities to the changing subset.
            lower, upper = float("nan"), float("nan")
            exact_p, wilcoxon_p = float("nan"), float("nan")
        rows.append(
            {
                **dict(zip(grouping, keys, strict=True)),
                "analysis_scope": analysis_scope,
                "proposed": proposed,
                "baseline": baseline,
                "scenario_n": len(by_scenario),
                "expected_scenario_n": expected_scenario_n,
                "complete_confirmatory_panel": complete_confirmatory_panel,
                "seed_scenario_pairs": len(per_seed),
                "expected_seed_scenario_pairs": expected_seed_scenario_pairs,
                "complete_seed_panel": complete_seed_panel,
                "mean_delta_recall": float(difference.mean()),
                "median_delta_recall": float(np.median(difference)),
                "bootstrap_ci_low": lower,
                "bootstrap_ci_high": upper,
                "mean_delta_workload": float(by_scenario["delta_workload"].mean()),
                "exact_one_sided_sign_flip_p": exact_p,
                "one_sided_wilcoxon_p": wilcoxon_p,
                "scenario_deltas": json.dumps(
                    by_scenario["delta_recall"].to_dict(), ensure_ascii=False
                ),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["holm_family_size"] = 1
    result["exact_p_holm"] = result["exact_one_sided_sign_flip_p"]
    fixed = (
        result["budget_design"].astype(str).str.startswith("fixed_")
        & result["exact_one_sided_sign_flip_p"].notna()
    )
    family_columns = ["label_column", "imbalance_ratio", "policy", "analysis_scope"]
    for _, index in result[fixed].groupby(family_columns).groups.items():
        ordered = result.loc[index, "exact_one_sided_sign_flip_p"].sort_values()
        size = len(ordered)
        adjusted = np.maximum.accumulate(
            [(size - rank) * value for rank, value in enumerate(ordered)]
        )
        result.loc[ordered.index, "exact_p_holm"] = np.minimum(adjusted, 1.0)
        result.loc[index, "holm_family_size"] = size
    return result


def baseline_matched_selections(runs: list[dict], config: dict) -> list[dict]:
    """Match each method to the original DeepCASE test-stream review count.

    Only prediction counts are observed in the test stream.  No outcome metric
    participates in finding the closest operating point below the reference
    capacity.  This is the confirmatory interpretation of "same manpower".
    """

    reference = config["design"]["primary_reference"]
    keys = ["label_column", "test_scenario", "imbalance_ratio", "seed"]
    result: list[dict] = []
    frame = pd.DataFrame(
        [{**{key: run[key] for key in keys}, "record": run} for run in runs]
    )
    if frame.empty:
        return result
    for _, group in frame.groupby(keys, sort=True):
        reference_rows = group[group["record"].map(lambda item: item["method"] == reference["method"])]
        if len(reference_rows) != 1:
            continue
        reference_record = reference_rows.iloc[0]["record"]
        reference_points = [
            point
            for point in reference_record["operating_points"]
            if np.isclose(float(point.get("eps", np.nan)), float(reference["eps"]))
            and np.isclose(
                float(point.get("threshold", np.nan)), float(reference["threshold"])
            )
        ]
        if len(reference_points) != 1:
            continue
        reference_point = reference_points[0]
        budget = float(reference_point["test"]["analyst_workload"])
        for record in group["record"]:
            if record["method"] == reference["method"]:
                selected = reference_point
                reason = "original DeepCASE default operating point"
            else:
                feasible = [
                    point
                    for point in record["operating_points"]
                    if point["test"]["analyst_workload"] <= budget + 1e-12
                ]
                selected = (
                    max(
                        feasible,
                        key=lambda point: (
                            point["test"]["analyst_workload"],
                            tuple(-item for item in point_key(point)),
                        ),
                    )
                    if feasible
                    else None
                )
                reason = (
                    "closest workload not exceeding DeepCASE default; no test outcomes used"
                    if selected is not None
                    else "no operating point below DeepCASE default workload"
                )
            result.append(
                flatten_selection(
                    record,
                    budget,
                    "deepcase_default_matched",
                    selected,
                    reason,
                    budget_design="deepcase_default_matched",
                )
            )
    return result


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without an optional dependency."""

    if frame.empty:
        return "No rows."
    rendered = frame.copy()
    for column in rendered.columns:
        rendered[column] = rendered[column].map(
            lambda value: ""
            if pd.isna(value)
            else f"{value:.6g}"
            if isinstance(value, (float, np.floating))
            else str(value).replace("|", "\\|")
        )
    headers = [str(column) for column in rendered.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in rendered.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def write_markdown(
    path: Path,
    runs: list[dict],
    selected: pd.DataFrame,
    paired: pd.DataFrame,
    confirmatory_n: int,
) -> None:
    complete = selected[selected["feasible"]]
    lines = [
        "# Workload-constrained experiment analysis",
        "",
        f"Complete run files: **{len(runs)}**.",
        "",
        "`analyst_workload` is an alert-review-rate proxy, not measured analyst time. "
        "Incident and Reject both consume review; only Non-Incident is auto-dismissed.",
        "",
        "Validation policy never consults test workload or labels. Calibrated policy may "
        "observe only test alert counts/review rate; it never uses test outcomes. Missing "
        "operating points are reported as infeasible and are not interpolated.",
        "",
        "## Feasibility",
        "",
    ]
    feasibility = (
        selected.groupby(["policy", "budget_design", "method"], as_index=False)["feasible"]
        .agg(feasible="sum", total="count")
    )
    lines.append(markdown_table(feasibility) if len(feasibility) else "No runs.")
    lines.extend(["", "## Selected test performance", ""])
    if len(complete):
        summary = (
            complete.groupby(["policy", "budget_design", "method"], as_index=False)[
                ["test_analyst_workload", "test_triage_incident_recall", "test_relaxed_f1"]
            ]
            .mean()
        )
        lines.append(markdown_table(summary))
    else:
        lines.append("No feasible operating points.")
    lines.extend(["", "## Proposed versus DeepCASE", ""])
    lines.append(markdown_table(paired) if len(paired) else "Insufficient paired folds.")
    lines.extend(
        [
            "",
            "Inference first averages repeated seeds within each held-out scenario. "
            f"The independent unit is the {confirmatory_n} confirmatory scenario folds, "
            "never individual alerts or random seeds.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["smoke", "pilot", "full"], default="pilot")
    parser.add_argument("--label", choices=["event_incident", "window_incident"], default="event_incident")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--result-namespace", default="aligned_v2")
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    result_root = (
        args.config.parent / "results" / args.result_namespace / args.stage / args.label
    )
    runs = load_runs(result_root)
    budgets = [float(item) for item in config["design"]["workload_budgets"]]
    selections: list[dict] = []
    for record in runs:
        for budget in budgets:
            for policy in ("validation", "calibrated"):
                point, reason = select_point(record["operating_points"], budget, policy)
                selections.append(flatten_selection(record, budget, policy, point, reason))
    selections.extend(baseline_matched_selections(runs, config))
    selected = pd.DataFrame(selections)
    confirmatory = (
        set(config["design"]["confirmatory_scenarios"])
        if args.stage == "full"
        else None
    )
    if len(selected):
        paired = pd.concat(
            [
                paired_inference(selected, "proposed", "deepcase", confirmatory),
                paired_inference(selected, "proposed", "rule_prior", confirmatory),
                paired_inference(selected, "deepcase", "rule_prior", confirmatory),
                paired_inference(
                    selected,
                    "proposed",
                    "deepcase",
                    confirmatory,
                    recall_column="test_ambiguous_triage_incident_recall",
                    workload_column="test_ambiguous_analyst_workload",
                    analysis_scope="training_defined_ambiguous_rules",
                ),
                paired_inference(
                    selected,
                    "proposed",
                    "rule_prior",
                    confirmatory,
                    recall_column="test_ambiguous_triage_incident_recall",
                    workload_column="test_ambiguous_analyst_workload",
                    analysis_scope="training_defined_ambiguous_rules",
                ),
                paired_inference(
                    selected,
                    "deepcase",
                    "rule_prior",
                    confirmatory,
                    recall_column="test_ambiguous_triage_incident_recall",
                    workload_column="test_ambiguous_analyst_workload",
                    analysis_scope="training_defined_ambiguous_rules",
                ),
            ],
            ignore_index=True,
        )
    else:
        paired = pd.DataFrame()
    artifact_root = args.config.parent / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    stem = f"analysis_{args.result_namespace}_{args.stage}_{args.label}"
    selected.to_csv(artifact_root / f"{stem}_selected.csv", index=False)
    paired.to_csv(artifact_root / f"{stem}_paired.csv", index=False)
    write_markdown(
        artifact_root / f"{stem}.md",
        runs,
        selected,
        paired,
        len(config["design"]["confirmatory_scenarios"]),
    )
    print(artifact_root / f"{stem}.md")


if __name__ == "__main__":
    main()
