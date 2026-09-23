"""Measure natural last-10 displacement without fitting a selector."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.common import EXPERIMENT, load_config, resolve_repo, sha256, write_json


def safe_divide(a: int | float, b: int | float) -> float:
    return float(a / b) if b else float("nan")


def group_codes(values: pd.Series) -> np.ndarray:
    codes, _ = pd.factorize(values.astype("string"), sort=True, use_na_sentinel=True)
    return codes.astype(np.int16, copy=False)


def audit(frame: pd.DataFrame, rows_path: Path, caps: list[int], chunk: int = 100_000) -> dict:
    candidates = np.load(rows_path, mmap_mode="r")
    if candidates.shape != (len(frame), max(caps)):
        raise ValueError(f"candidate shape {candidates.shape} does not match data")
    labels = frame["event_incident"].to_numpy(dtype=np.int8, copy=False)
    events = frame["event_id"].to_numpy(dtype=np.int16, copy=False)
    timestamps = frame["timestamp"].to_numpy(dtype=np.int64, copy=False)
    groups = group_codes(frame["event_incident_group"])
    scenarios = frame["scenario"].astype(str).to_numpy()
    all_scenarios = list(dict.fromkeys(scenarios.tolist()))
    accumulators: dict[str, dict] = {}
    for scenario in ["__all__", *all_scenarios]:
        accumulators[scenario] = {
            "targets": 0,
            "incident_targets": 0,
            "last10_nonempty": 0,
            "last10_slots": 0,
            "last10_unique_sum": 0,
            "last10_duplicate_sum": 0,
            "last10_span_sum": 0.0,
            "last10_span_count": 0,
            "any_incident_opportunity_100": 0,
            "any_incident_displaced_10": 0,
            "same_group_opportunity_100": 0,
            "same_group_displaced_10": 0,
            "coverage_any": {str(cap): 0 for cap in caps},
            "coverage_same_group": {str(cap): 0 for cap in caps},
        }

    for begin in range(0, len(frame), chunk):
        end = min(begin + chunk, len(frame))
        row_block = np.asarray(candidates[begin:end])
        valid = row_block >= 0
        safe_rows = np.where(valid, row_block, 0)
        prior_labels = labels[safe_rows] * valid
        prior_events = np.where(valid, events[safe_rows], -1)
        prior_groups = np.where(valid, groups[safe_rows], -2)
        target_incident = labels[begin:end] == 1
        target_group = groups[begin:end]
        last10_valid = valid[:, -10:]
        last10_labels = prior_labels[:, -10:]
        any100 = prior_labels.any(axis=1)
        any10 = last10_labels.any(axis=1)
        same100 = (prior_groups == target_group[:, None]) & valid & (target_group[:, None] >= 0)
        same10 = same100[:, -10:].any(axis=1)
        same100_any = same100.any(axis=1)

        unique_counts = np.zeros(end - begin, dtype=np.int16)
        for local in range(end - begin):
            vals = prior_events[local, -10:][last10_valid[local]]
            unique_counts[local] = len(np.unique(vals)) if len(vals) else 0
        slots = last10_valid.sum(axis=1)
        duplicates = slots - unique_counts
        oldest = np.full(end - begin, -1, dtype=np.int64)
        has_context = slots > 0
        if has_context.any():
            first_position = np.argmax(last10_valid[has_context], axis=1)
            context_rows = row_block[has_context, -10:]
            oldest[has_context] = context_rows[np.arange(len(context_rows)), first_position]
        spans = np.zeros(end - begin, dtype=np.float64)
        spans[has_context] = timestamps[begin:end][has_context] - timestamps[oldest[has_context]]

        masks = {"__all__": np.ones(end - begin, dtype=bool)}
        for scenario in all_scenarios:
            masks[scenario] = scenarios[begin:end] == scenario
        for scenario, mask in masks.items():
            acc = accumulators[scenario]
            incident_mask = mask & target_incident
            acc["targets"] += int(mask.sum())
            acc["incident_targets"] += int(incident_mask.sum())
            acc["last10_nonempty"] += int((mask & has_context).sum())
            acc["last10_slots"] += int(slots[mask].sum())
            acc["last10_unique_sum"] += int(unique_counts[mask].sum())
            acc["last10_duplicate_sum"] += int(duplicates[mask].sum())
            acc["last10_span_sum"] += float(spans[mask & has_context].sum())
            acc["last10_span_count"] += int((mask & has_context).sum())
            opportunity_any = incident_mask & any100
            opportunity_same = incident_mask & same100_any
            acc["any_incident_opportunity_100"] += int(opportunity_any.sum())
            acc["any_incident_displaced_10"] += int((opportunity_any & ~any10).sum())
            acc["same_group_opportunity_100"] += int(opportunity_same.sum())
            acc["same_group_displaced_10"] += int((opportunity_same & ~same10).sum())
            for cap in caps:
                cap_any = prior_labels[:, -cap:].any(axis=1)
                cap_same = same100[:, -cap:].any(axis=1)
                acc["coverage_any"][str(cap)] += int((incident_mask & cap_any).sum())
                acc["coverage_same_group"][str(cap)] += int((incident_mask & cap_same).sum())

    result: dict[str, dict] = {}
    for scenario, acc in accumulators.items():
        incident = acc["incident_targets"]
        slots = acc["last10_slots"]
        result[scenario] = {
            **acc,
            "incident_prevalence": safe_divide(incident, acc["targets"]),
            "last10_mean_filled_slots": safe_divide(slots, acc["targets"]),
            "last10_redundancy_rate": safe_divide(acc["last10_duplicate_sum"], slots),
            "last10_mean_unique_types": safe_divide(acc["last10_unique_sum"], acc["targets"]),
            "last10_mean_span_seconds_nonempty": safe_divide(
                acc["last10_span_sum"], acc["last10_span_count"]
            ),
            "any_incident_displacement_at_10": safe_divide(
                acc["any_incident_displaced_10"], acc["any_incident_opportunity_100"]
            ),
            "same_group_displacement_at_10": safe_divide(
                acc["same_group_displaced_10"], acc["same_group_opportunity_100"]
            ),
            "any_incident_predecessor_coverage": {
                cap: safe_divide(count, incident) for cap, count in acc["coverage_any"].items()
            },
            "same_group_predecessor_coverage": {
                cap: safe_divide(count, incident)
                for cap, count in acc["coverage_same_group"].items()
            },
        }
    return result


def markdown(payload: dict) -> str:
    lines = [
        "# Natural context-retrieval audit",
        "",
        "This is a descriptive oracle audit. Labels were not used by a deployable selector.",
        "",
        "| scenario | alerts | incident prevalence | redundancy@10 | any displacement@10 | same-group displacement@10 | any coverage 10→100 | mean span (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scenario, item in payload["by_scenario"].items():
        coverage = item["any_incident_predecessor_coverage"]
        lines.append(
            f"| {scenario} | {item['targets']} | {item['incident_prevalence']:.4f} | "
            f"{item['last10_redundancy_rate']:.4f} | "
            f"{item['any_incident_displacement_at_10']:.4f} | "
            f"{item['same_group_displacement_at_10']:.4f} | "
            f"{coverage['10']:.4f}→{coverage['100']:.4f} | "
            f"{item['last10_mean_span_seconds_nonempty']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    config = load_config()
    source = resolve_repo(config["data"]["canonical_parquet"])
    candidates = resolve_repo(config["data"]["candidate_rows"])
    frame = pd.read_parquet(
        source,
        columns=[
            "scenario",
            "timestamp",
            "event_id",
            "event_incident",
            "event_incident_group",
        ],
    )
    by_scenario = audit(frame, candidates, list(config["data"]["candidate_caps"]))
    warning = float(config["gates"]["displacement_warning"])
    all_item = by_scenario["__all__"]
    payload = {
        "protocol": "protocol.md",
        "source_sha256": sha256(source),
        "candidate_sha256": sha256(candidates),
        "descriptive_warning_threshold": warning,
        "natural_mechanism_supported_descriptively": bool(
            max(
                all_item["any_incident_displacement_at_10"],
                all_item["same_group_displacement_at_10"],
            )
            >= warning
        ),
        "by_scenario": by_scenario,
    }
    artifact = EXPERIMENT / "artifacts" / "retrieval_audit.json"
    report = EXPERIMENT / "artifacts" / "retrieval_audit.md"
    write_json(artifact, payload)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(markdown(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
