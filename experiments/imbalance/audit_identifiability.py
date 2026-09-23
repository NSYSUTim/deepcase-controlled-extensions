"""Fail-closed AIT-ADS audit for within-rule contextual learning."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yaml

from src.data import fold_indices, load_prepared, reveal_training_labels, scenario_fold
from src.identifiability import rule_identifiability_table, summarize_identifiability


def _markdown(records: list[dict]) -> str:
    lines = [
        "# RC-CR 可識別性稽核",
        "",
        "本稽核只使用各 fold 的 training scenarios。完整 training truth 僅用於描述資料是否存在同一 rule 的正負語意；實際 pairability 僅由 revealed labels 決定。validation/test truth 未參與。",
        "",
        "| Test scenario | Seed | Mixed rules | Pairable revealed rules | Pairable positive groups | Revealed positive coverage | Gate |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in records:
        summary = item["summary"]
        lines.append(
            "| {test_scenario} | {seed} | {rules_mixed_full_truth} | "
            "{rules_pairable_revealed} | {revealed_positive_groups_pairable} | "
            "{fraction_revealed_positives_pairable:.3f} | {gate} |".format(
                **item,
                **summary,
                gate="PASS" if summary["pairability_pass"] else "STOP",
            )
        )
    lines.extend(
        [
            "",
            "`PASS` 只表示資料中存在可學習的 revealed within-rule pairs，不表示模型一定優於 DeepCASE。",
            "",
        ]
    )
    return "\n".join(lines)


def run(config_path: Path, output_dir: Path) -> None:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    root = config_path.resolve().parent.parent
    data = load_prepared(
        root / config["data"]["canonical_parquet"],
        root / config["data"]["contexts_npz"],
    )
    label_column = str(config["data"]["primary_label"])
    group_column = "event_incident_group"
    rule_column = str(config["design"]["ambiguous_rule_diagnostic"]["rule_column"])
    audit_cfg = config["rccr"]["identifiability"]
    ratio = int(config["rccr"]["imbalance_ratio"])
    label_budget = int(config["design"]["label_budget"])
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for test_scenario in config["design"]["design_scenarios"]:
        fold = scenario_fold(config["design"]["scenarios"], test_scenario)
        indices = fold_indices(data.metadata, fold)
        for seed in config["rccr"]["design_seeds"]:
            revealed = reveal_training_labels(
                data.metadata,
                indices["train"],
                label_column,
                group_column,
                label_budget,
                ratio,
                int(seed),
            )
            table = rule_identifiability_table(
                data.metadata,
                indices["train"],
                revealed,
                label_column,
                group_column,
                rule_column=rule_column,
                minimum_full_per_class=int(audit_cfg["minimum_full_per_class"]),
                minimum_revealed_per_class=int(
                    audit_cfg["minimum_revealed_per_class"]
                ),
                lower_full_incident_rate=float(
                    audit_cfg["lower_full_incident_rate"]
                ),
                upper_full_incident_rate=float(
                    audit_cfg["upper_full_incident_rate"]
                ),
            )
            summary = summarize_identifiability(
                table,
                minimum_pairable_rules=int(audit_cfg["minimum_pairable_rules"]),
                minimum_pairable_positive_groups=int(
                    audit_cfg["minimum_pairable_positive_groups"]
                ),
            )
            name = f"rules_{test_scenario}_seed_{seed}.csv"
            table.to_csv(output_dir / name, index=False)
            records.append(
                {
                    "test_scenario": str(test_scenario),
                    "validation_scenario": fold.validation_scenario,
                    "train_scenarios": list(fold.train_scenarios),
                    "seed": int(seed),
                    "imbalance_ratio": ratio,
                    "label_budget": label_budget,
                    "rule_table": name,
                    "summary": asdict(summary),
                }
            )

    payload = {
        "status": "complete",
        "scope": "design-only training-fold diagnostic",
        "label_column": label_column,
        "group_column": group_column,
        "rule_column": rule_column,
        "gate_config": audit_cfg,
        "all_records_pass": all(
            item["summary"]["pairability_pass"] for item in records
        ),
        "records": records,
    }
    (output_dir / "identifiability_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (output_dir / "identifiability_audit.md").write_text(
        _markdown(records), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.yaml")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("artifacts") / "rccr_identifiability",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.config.resolve(), arguments.output_dir.resolve())
