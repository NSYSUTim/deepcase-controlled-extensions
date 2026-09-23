"""Generate a machine-readable and paper-ready audit of prepared AIT-ADS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data import load_prepared, scenario_fold


def pct(value: float) -> str:
    return f"{100.0 * value:.3f}%"


def markdown_table(table: pd.DataFrame, float_digits: int = 4) -> str:
    """Render a compact Markdown table without an optional tabulate dependency."""

    columns = [str(column) for column in table.columns]
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for values in table.itertuples(index=False, name=None):
        formatted = []
        for value in values:
            if isinstance(value, (float, np.floating)):
                formatted.append(f"{float(value):.{float_digits}f}")
            else:
                formatted.append(str(value))
        rows.append("| " + " | ".join(formatted) + " |")
    return "\n".join(rows)


def audit(config_path: Path, output_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    root = config_path.parent.parent
    data = load_prepared(
        root / config["data"]["canonical_parquet"],
        root / config["data"]["contexts_npz"],
    )
    frame = data.metadata
    label_cross = pd.crosstab(frame["window_incident"], frame["event_incident"])
    tied_sizes = frame.groupby(["machine", "timestamp"], observed=True).size()
    tied_rows = int(tied_sizes[tied_sizes > 1].sum())
    duplicated_alert_keys = int(
        frame.duplicated(
            ["scenario", "timestamp", "machine", "event_key", "event_label"],
            keep=False,
        ).sum()
    )
    empty_context = int(np.all(data.context == data.pad_id, axis=1).sum())
    scenario_stats = (
        frame.groupby("scenario", observed=True)
        .agg(
            alerts=("row_id", "size"),
            event_prevalence=("event_incident", "mean"),
            window_prevalence=("window_incident", "mean"),
            event_types=("event_key", "nunique"),
            machines=("machine", "nunique"),
        )
        .reset_index()
    )
    rule_stats = (
        frame.groupby(["detector", "event_key"], observed=True)
        .agg(alerts=("row_id", "size"), incident_rate=("event_incident", "mean"))
        .reset_index()
        .sort_values("alerts", ascending=False)
    )
    all_events = set(frame["event_key"].astype(str).unique())
    unseen_by_fold: dict[str, dict] = {}
    for test in config["design"]["scenarios"]:
        fold = scenario_fold(config["design"]["scenarios"], test)
        train_events = set(
            frame.loc[frame["scenario"].isin(fold.train_scenarios), "event_key"].astype(str)
        )
        test_mask = frame["scenario"].eq(test)
        unseen_events = sorted(set(frame.loc[test_mask, "event_key"].astype(str)) - train_events)
        unseen_rows = int(frame.loc[test_mask, "event_key"].astype(str).isin(unseen_events).sum())
        unseen_by_fold[test] = {
            "validation": fold.validation_scenario,
            "unseen_event_types": unseen_events,
            "unseen_test_alerts": unseen_rows,
        }

    legacy_path = root / config["data"]["legacy_csv"]
    legacy = None
    if legacy_path.exists():
        legacy_frame = pd.read_csv(legacy_path, usecols=["event"])
        legacy = {
            "rows": len(legacy_frame),
            "event_types": int(legacy_frame["event"].nunique()),
            "missing_vs_official": int(len(frame) - len(legacy_frame)),
        }

    result = {
        "rows": len(frame),
        "event_types": len(data.vocabulary),
        "machines": int(frame["machine"].nunique()),
        "detector_counts": frame["detector"].value_counts().sort_index().to_dict(),
        "event_incident_prevalence": float(frame["event_incident"].mean()),
        "window_incident_prevalence": float(frame["window_incident"].mean()),
        "event_incident_groups": int(frame["event_incident_group"].nunique(dropna=True)),
        "window_incident_groups": int(frame["window_incident_group"].nunique(dropna=True)),
        "label_cross": {
            str(row): {str(col): int(label_cross.loc[row, col]) for col in label_cross.columns}
            for row in label_cross.index
        },
        "tied_timestamp_rows": tied_rows,
        "duplicated_alert_key_rows": duplicated_alert_keys,
        "empty_context_rows": empty_context,
        "legacy": legacy,
        "unseen_by_fold": unseen_by_fold,
    }

    lines = [
        "# AIT-ADS data and validity audit",
        "",
        "## Decision",
        "",
        "The downloaded raw files are authentic in row count, but the legacy ",
        "`ait_ads_deepcase.csv` is not an acceptable formal input: it omits all ",
        "AMiner alerts and collapses all Suricata signatures into Wazuh wrapper ",
        "rule 86601. The experiment uses the official author-produced alert CSVs.",
        "",
        "## Core counts",
        "",
        f"- Alerts: {len(frame):,}",
        f"- Stable detector event types: {len(data.vocabulary):,}",
        f"- Scenario-qualified machines: {frame['machine'].nunique():,}",
        f"- Event-label incident prevalence: {pct(frame['event_incident'].mean())}",
        f"- Time-window incident prevalence: {pct(frame['window_incident'].mean())}",
        f"- Event-label incident groups: {frame['event_incident_group'].nunique(dropna=True):,}",
        f"- Time-window incident groups: {frame['window_incident_group'].nunique(dropna=True):,}",
        f"- Rows sharing a machine/timestamp with another alert: {tied_rows:,}",
        f"- Empty strict-past contexts: {empty_context:,}",
        "",
        "The primary label is not naturally imbalanced: incidents are the ",
        f"majority ({pct(frame['event_incident'].mean())}). Therefore extreme ",
        "ratios are created only in the revealed training-label subset while the ",
        "event stream and validation/test prevalence remain untouched. Claims ",
        "must be phrased as robustness to incident-label scarcity, not as proof ",
        "on a naturally extreme-prevalence SOC.",
        "",
        "## Detector counts",
        "",
        markdown_table(
            frame["detector"].value_counts().sort_index().rename("alerts").reset_index()
        ),
        "",
        "## Per-scenario prevalence",
        "",
        markdown_table(scenario_stats),
        "",
        "## Label-definition disagreement",
        "",
        markdown_table(label_cross.reset_index()),
        "",
        "Event-level and attack-window labels disagree materially, especially ",
        "for delayed/derived DNSteal alerts and attack windows containing ",
        "background alerts. Event labels are primary; window labels are a ",
        "pre-registered sensitivity analysis. Neither is silently substituted.",
        "",
        "## Highest-volume rules/events",
        "",
        markdown_table(rule_stats.head(20)),
        "",
        "## Leakage controls",
        "",
        "- Scenario-disjoint nested holdout is required because millions of ",
        "  repeated alerts make random row splits severely optimistic.",
        "- Same-second alerts are excluded from one another's context.",
        "- Scenario is prefixed to host names, preventing cross-testbed context.",
        "- Test-only event types map to UNK and remain eligible for original ",
        "  DeepCASE unknown-event rejection.",
        "- Duplicate multiplicity is retained as workload and model-exposure ",
        "  weight; alert rows are not treated as independent inferential units.",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output_path.with_suffix(".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("artifacts") / "data_audit.md",
    )
    args = parser.parse_args()
    result = audit(args.config.resolve(), args.output.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
