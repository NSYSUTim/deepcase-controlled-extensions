from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .evaluation import BASELINE_PREDICTION_POLICY, build_run_evaluation
from .utils import ensure_dir, load_json, save_json, slugify


METRIC_KEYS = {
    "n_clusters": ("workload_reduction", "n_clusters"),
    "coverage": ("workload_reduction", "coverage"),
    "overall_reduction": ("workload_reduction", "overall_reduction"),
    "cluster_purity": ("cluster_quality", "cluster_purity"),
    "auto_decided_rate": ("prediction", "auto_decided_rate"),
    "reject_rate": ("prediction", "reject_rate"),
    "binary_f1": ("prediction", "binary_incident_detection", "f1"),
    "auto_decided_accuracy": ("prediction", "auto_decided_accuracy"),
    "strict_accuracy_rejects_wrong": ("prediction", "strict_accuracy_rejects_wrong"),
    "macro_f1": ("prediction", "multiclass_attack_type", "macro_f1"),
    "weighted_f1": ("prediction", "multiclass_attack_type", "weighted_f1"),
}


@dataclass(frozen=True)
class RunEvaluation:
    method: str
    run_id: str
    summary_path: Path | None
    evaluation: dict[str, Any]


def _nested_get(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _read_report_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#")


def load_baseline_evaluation(project_root: Path, run_id: str) -> RunEvaluation:
    reports_dir = project_root / "results" / "runs" / run_id / "reports"
    summary_path = reports_dir / "summary.json"
    if not reports_dir.exists():
        raise FileNotFoundError(f"Baseline reports directory not found: {reports_dir}")

    summary = load_json(summary_path) if summary_path.exists() else {}
    if "evaluation" in summary:
        evaluation = summary["evaluation"]
    else:
        clusters = _read_report_csv(reports_dir / "clusters.csv")
        predictions = _read_report_csv(reports_dir / "prediction.csv")
        evaluation = build_run_evaluation(
            clusters=clusters["cluster_id"].to_numpy(dtype=int),
            train_labels=clusters["true_label"].to_numpy(dtype=int),
            predictions=predictions["predicted_score"].to_numpy(dtype=float).astype(int),
            test_labels=predictions["true_label"].to_numpy(dtype=int),
            prediction_policy=BASELINE_PREDICTION_POLICY,
        )

    return RunEvaluation(
        method="baseline",
        run_id=run_id,
        summary_path=summary_path if summary_path.exists() else None,
        evaluation=evaluation,
    )


def load_research_evaluation(summary_path: Path) -> RunEvaluation:
    summary = load_json(summary_path)
    if "evaluation" not in summary:
        raise ValueError(f"Research summary has no evaluation section: {summary_path}")
    return RunEvaluation(
        method=str(summary.get("method_name", summary_path.parent.parent.parent.name)),
        run_id=str(summary.get("run_id", summary_path.parent.parent.name)),
        summary_path=summary_path,
        evaluation=summary["evaluation"],
    )


def resolve_research_run_summary(project_root: Path, method_name: str, run_id: str) -> Path:
    method_dir = project_root / "results" / "research" / slugify(method_name)
    resolved_run_id = run_id
    if run_id == "latest":
        latest_path = method_dir / "latest_run.txt"
        if not latest_path.exists():
            raise FileNotFoundError(f"Latest research run not found: {latest_path}")
        resolved_run_id = latest_path.read_text(encoding="utf-8").strip()
    return method_dir / "runs" / resolved_run_id / "reports" / "summary.json"


def row_from_evaluation(run: RunEvaluation) -> dict[str, Any]:
    row: dict[str, Any] = {
        "method": run.method,
        "run_id": run.run_id,
        "summary_path": str(run.summary_path) if run.summary_path is not None else "",
    }
    for column, keys in METRIC_KEYS.items():
        value = _nested_get(run.evaluation, keys)
        if isinstance(value, np.generic):
            value = value.item()
        row[column] = value
    return row


def classify_against_baseline(row: dict[str, Any], baseline: dict[str, Any]) -> str:
    binary_delta = _float(row.get("binary_f1")) - _float(baseline.get("binary_f1"))
    auto_delta = _float(row.get("auto_decided_rate")) - _float(
        baseline.get("auto_decided_rate")
    )
    reject_delta = _float(row.get("reject_rate")) - _float(baseline.get("reject_rate"))
    workload_delta = _float(row.get("overall_reduction")) - _float(
        baseline.get("overall_reduction")
    )

    if binary_delta >= 0.02 and auto_delta >= -0.05 and reject_delta <= 0.05:
        return "primary_candidate"
    if workload_delta > 0:
        return "workload_ablation"
    return "ablation_only"


def _float(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def build_comparison(
    *,
    baseline: RunEvaluation,
    research_runs: list[RunEvaluation],
) -> dict[str, Any]:
    baseline_row = row_from_evaluation(baseline)
    rows = [baseline_row]
    deltas: list[dict[str, Any]] = []

    for run in research_runs:
        row = row_from_evaluation(run)
        rows.append(row)
        delta_row: dict[str, Any] = {
            "method": row["method"],
            "run_id": row["run_id"],
            "recommended_role": classify_against_baseline(row, baseline_row),
        }
        for metric in (
            "overall_reduction",
            "coverage",
            "cluster_purity",
            "auto_decided_rate",
            "reject_rate",
            "binary_f1",
            "macro_f1",
            "weighted_f1",
        ):
            delta_row[f"delta_{metric}"] = _float(row.get(metric)) - _float(
                baseline_row.get(metric)
            )
        deltas.append(delta_row)

    return {
        "baseline_run_id": baseline.run_id,
        "metric_policy": {
            "primary_candidate": (
                "binary_f1 improves by at least 0.02 while auto decision/reject rates "
                "stay within 0.05 of baseline"
            ),
            "workload_ablation": "overall workload reduction improves but primary detection criteria do not",
            "ablation_only": "use for discussion only unless the full run improves",
        },
        "rows": rows,
        "deltas_vs_baseline": deltas,
    }


def write_comparison_outputs(comparison: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir = ensure_dir(output_dir)
    rows = comparison["rows"]
    deltas = comparison["deltas_vs_baseline"]

    json_path = save_json(output_dir / "comparison.json", comparison)
    csv_path = output_dir / "comparison.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    delta_csv_path = output_dir / "comparison_delta.csv"
    pd.DataFrame(deltas).to_csv(delta_csv_path, index=False)

    md_path = output_dir / "comparison.md"
    md_path.write_text(_build_markdown(comparison), encoding="utf-8")
    return {
        "json": json_path,
        "csv": csv_path,
        "delta_csv": delta_csv_path,
        "markdown": md_path,
    }


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return "N/A"
    if isinstance(value, str):
        return value
    return f"{float(value):.4f}"


def _build_markdown(comparison: dict[str, Any]) -> str:
    rows = comparison["rows"]
    deltas = {item["run_id"]: item for item in comparison["deltas_vs_baseline"]}
    headers = [
        "method",
        "role",
        "n_clusters",
        "overall_reduction",
        "cluster_purity",
        "auto_decided_rate",
        "reject_rate",
        "binary_f1",
        "weighted_f1",
    ]
    lines = [
        "# DeepCASE Result Comparison",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        role = "baseline"
        if row["method"] != "baseline":
            role = deltas.get(row["run_id"], {}).get("recommended_role", "N/A")
        values = [
            row["method"],
            role,
            _fmt(row.get("n_clusters")),
            _fmt(row.get("overall_reduction")),
            _fmt(row.get("cluster_purity")),
            _fmt(row.get("auto_decided_rate")),
            _fmt(row.get("reject_rate")),
            _fmt(row.get("binary_f1")),
            _fmt(row.get("weighted_f1")),
        ]
        lines.append("| " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## Report Use",
            "",
            "- Use `primary_candidate` as the main proposed method result.",
            "- Use `workload_ablation` as supporting evidence only.",
            "- Use `ablation_only` for limitations or future work unless a full run improves.",
        ]
    )
    return "\n".join(lines) + "\n"
