from __future__ import annotations

from typing import Any


def build_default_evaluation_plan(
    *,
    focus: str,
    extra_metrics: list[str] | None = None,
) -> dict[str, Any]:
    metrics = [
        "macro_f1",
        "pr_auc",
        "incident_recall",
        "reject_rate",
        "cluster_purity",
        "analyst_workload_reduction_proxy",
    ]
    if extra_metrics:
        for metric in extra_metrics:
            if metric not in metrics:
                metrics.append(metric)

    return {
        "focus": focus,
        "required_metrics": metrics,
        "minimum_comparisons": [
            "baseline_deepcase",
            "single_method_variant",
            "at_least_one_ablation",
        ],
        "analysis_views": [
            "overall",
            "subgroup",
            "case_study",
            "explanation_sanity_check",
        ],
    }
