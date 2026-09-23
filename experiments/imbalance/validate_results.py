"""Fail-closed audit of completed experiment records before reporting."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--namespace", default="aligned_v2")
    parser.add_argument("--stage", choices=["pilot", "full"], default="full")
    parser.add_argument("--scenarios", nargs="+")
    parser.add_argument("--label", default="event_incident")
    parser.add_argument("--ratio", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1729, 2718, 31415])
    parser.add_argument("--methods", nargs="+", default=["deepcase", "proposed", "rule_prior"])
    parser.add_argument("--query-iterations", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = args.config.parent / "results" / args.namespace / args.stage / args.label
    repository_root = args.config.parent.parent
    expected_config_sha256 = sha256_file(args.config)
    manifest_path = repository_root / config["data"]["manifest_json"]
    expected_manifest_sha256 = sha256_file(manifest_path)
    if args.stage == "full":
        settings = config["interpreter"]["full_settings"]
    else:
        settings = [
            (eps, threshold)
            for eps in config["interpreter"]["pilot_eps_grid"]
            for threshold in config["interpreter"]["pilot_threshold_grid"]
        ]
    expected_settings = {(float(eps), float(threshold)) for eps, threshold in settings}
    scenarios = (
        list(args.scenarios)
        if args.scenarios is not None
        else list(config["design"]["confirmatory_scenarios"])
    )
    expected_query_iterations = int(
        config["interpreter"]["query_iterations"]
        if args.query_iterations is None
        else args.query_iterations
    )
    errors: list[str] = []
    checked = 0
    checkpoint_hashes_checked = 0
    identities: dict[tuple[str, int], tuple] = {}
    for scenario in scenarios:
        for seed in args.seeds:
            for method in args.methods:
                path = root / scenario / f"ratio_{args.ratio}" / f"seed_{seed}" / f"{method}.json"
                if not path.exists():
                    errors.append(f"missing: {path}")
                    continue
                record = json.loads(path.read_text(encoding="utf-8"))
                checked += 1
                prefix = f"{scenario}/{seed}/{method}"
                if record.get("status") != "complete":
                    errors.append(f"{prefix}: status is not complete")
                if record.get("stage") != args.stage:
                    errors.append(f"{prefix}: stage identity mismatch")
                if record.get("method") != method:
                    errors.append(f"{prefix}: method identity mismatch")
                if record.get("label_column") != args.label:
                    errors.append(f"{prefix}: label identity mismatch")
                if record.get("result_namespace") != args.namespace:
                    errors.append(f"{prefix}: namespace identity mismatch")
                if record.get("scoring_index_version") != "aligned_v2":
                    errors.append(f"{prefix}: wrong scoring index version")
                if record.get("test_scenario") != scenario or record.get("seed") != seed:
                    errors.append(f"{prefix}: identity mismatch")
                if int(record.get("imbalance_ratio", -1)) != args.ratio:
                    errors.append(f"{prefix}: imbalance ratio mismatch")
                recorded_query_iterations = record.get("interpreter_query_iterations")
                if recorded_query_iterations is None:
                    recorded_query_iterations = (
                        record.get("config_snapshot", {})
                        .get("interpreter", {})
                        .get("query_iterations", -1)
                    )
                if int(recorded_query_iterations) != expected_query_iterations:
                    errors.append(f"{prefix}: unexpected Interpreter query iterations")
                if record.get("config_sha256") != expected_config_sha256:
                    errors.append(f"{prefix}: configuration hash mismatch")
                if record.get("data_manifest_sha256") != expected_manifest_sha256:
                    errors.append(f"{prefix}: data manifest hash mismatch")
                recorded_checkpoint_sha256 = record.get("checkpoint_sha256")
                if recorded_checkpoint_sha256 is not None:
                    checkpoint_path_value = record.get("checkpoint")
                    if not checkpoint_path_value:
                        errors.append(f"{prefix}: checkpoint hash recorded without path")
                    else:
                        checkpoint_path = Path(checkpoint_path_value)
                        if not checkpoint_path.exists():
                            errors.append(f"{prefix}: checkpoint file is missing")
                        else:
                            checkpoint_hashes_checked += 1
                            if sha256_file(checkpoint_path) != recorded_checkpoint_sha256:
                                errors.append(f"{prefix}: checkpoint hash mismatch")
                revealed = record.get("revealed_labels", {})
                if not np.isclose(float(revealed.get("realized_ratio", -1)), args.ratio):
                    errors.append(f"{prefix}: realized ratio mismatch")
                positive_labels = int(revealed.get("positive_labels", -1))
                negative_labels = int(revealed.get("negative_labels", -1))
                label_budget = int(revealed.get("label_budget", -1))
                if positive_labels <= 0 or negative_labels <= 0:
                    errors.append(f"{prefix}: empty or invalid revealed label class")
                if positive_labels + negative_labels != label_budget:
                    errors.append(f"{prefix}: revealed label counts do not sum to budget")
                if label_budget > int(config["design"]["label_budget"]):
                    errors.append(f"{prefix}: revealed label budget exceeds design")
                train_scenarios = list(record.get("train_scenarios", []))
                validation_scenario = record.get("validation_scenario")
                if (
                    scenario in train_scenarios
                    or validation_scenario == scenario
                    or validation_scenario in train_scenarios
                    or len(set(train_scenarios)) != len(train_scenarios)
                ):
                    errors.append(f"{prefix}: overlapping or duplicated scenario split")
                split_rows = record.get("split_rows", {})
                if any(int(split_rows.get(split, 0)) <= 0 for split in ("train", "validation", "test")):
                    errors.append(f"{prefix}: empty data split")
                identity = (
                    tuple(record.get("train_scenarios", [])),
                    record.get("validation_scenario"),
                    record.get("test_scenario"),
                    tuple(sorted(record.get("split_rows", {}).items())),
                    revealed.get("positive_labels"),
                    revealed.get("negative_labels"),
                )
                pair_key = (scenario, seed)
                if pair_key in identities and identities[pair_key] != identity:
                    errors.append(f"{prefix}: paired split/label counts differ")
                identities[pair_key] = identity
                points = record.get("operating_points", [])
                if method != "rule_prior":
                    settings = {
                        (float(point["eps"]), float(point["threshold"])) for point in points
                    }
                    if settings != expected_settings:
                        errors.append(f"{prefix}: incomplete/extra neural settings")
                    primary = config["design"]["primary_reference"]
                    primary_setting = (float(primary["eps"]), float(primary["threshold"]))
                    if method == primary["method"] and primary_setting not in settings:
                        errors.append(f"{prefix}: missing primary DeepCASE reference point")
                elif len(points) != len({float(point["rule_threshold"]) for point in points}):
                    errors.append(f"{prefix}: duplicated rule-prior threshold")
                for point in points:
                    for split in ("validation", "test"):
                        metrics = point.get(split, {})
                        expected_alerts = int(split_rows.get(split, -1))
                        if int(metrics.get("alerts", -2)) != expected_alerts:
                            errors.append(f"{prefix}: {split} alert count mismatch")
                        for name in (
                            "analyst_workload", "triage_incident_recall", "reject_rate",
                            "automation_coverage", "relaxed_f1", "incident_group_recall",
                        ):
                            value = metrics.get(name)
                            if value is None or not np.isfinite(value) or not 0 <= value <= 1:
                                errors.append(f"{prefix}: invalid {split}.{name}={value}")
                        precision = metrics.get("triage_precision")
                        workload = metrics.get("analyst_workload")
                        # Precision has a zero denominator when no alert is sent for
                        # review.  JSON null is the mathematically correct value in
                        # that case; it is not evidence of a corrupt result record.
                        if precision is None:
                            if workload is None or not np.isclose(float(workload), 0.0):
                                errors.append(
                                    f"{prefix}: undefined {split}.triage_precision "
                                    f"at nonzero workload={workload}"
                                )
                        elif not np.isfinite(precision) or not 0 <= precision <= 1:
                            errors.append(
                                f"{prefix}: invalid {split}.triage_precision={precision}"
                            )
                        workload = metrics.get("analyst_workload")
                        automation = metrics.get("automation_coverage")
                        reject_rate = metrics.get("reject_rate")
                        if (
                            automation is not None
                            and reject_rate is not None
                            and not np.isclose(float(automation) + float(reject_rate), 1.0)
                        ):
                            errors.append(f"{prefix}: {split} accepted/reject invariant failed")
                        if (
                            workload is not None
                            and reject_rate is not None
                            and float(reject_rate) > float(workload) + 1e-12
                        ):
                            errors.append(f"{prefix}: {split} reject rate exceeds review workload")
                        auto_incident = metrics.get("auto_incident_rate")
                        if (
                            workload is not None
                            and reject_rate is not None
                            and auto_incident is not None
                            and not np.isclose(
                                float(workload), float(reject_rate) + float(auto_incident)
                            )
                        ):
                            errors.append(f"{prefix}: {split} review decomposition failed")
                        reject_components = [
                            metrics.get("reject_low_confidence_rate"),
                            metrics.get("reject_unknown_event_rate"),
                            metrics.get("reject_distance_rate"),
                        ]
                        if (
                            reject_rate is not None
                            and all(value is not None for value in reject_components)
                            and not np.isclose(
                                float(reject_rate), sum(float(value) for value in reject_components)
                            )
                        ):
                            errors.append(f"{prefix}: {split} Reject decomposition failed")
    summary = {
        "namespace": args.namespace,
        "stage": args.stage,
        "scenarios": scenarios,
        "query_iterations": expected_query_iterations,
        "label": args.label,
        "ratio": int(args.ratio),
        "seeds": [int(seed) for seed in args.seeds],
        "methods": list(args.methods),
        "checked_records": checked,
        "expected_records": len(scenarios) * len(args.seeds) * len(args.methods),
        "checkpoint_hashes_checked": checkpoint_hashes_checked,
        "errors": errors,
        "source_sha256": {
            str(path.relative_to(args.config.parent.parent)).replace("\\", "/"): sha256_file(path)
            for path in (
                args.config,
                args.config.parent / "requirements-lock.txt",
                args.config.parent / "run.ps1",
                args.config.parent / "run_experiment.py",
                args.config.parent / "analyze_results.py",
                args.config.parent / "validate_results.py",
                args.config.parent / "src" / "data.py",
                args.config.parent / "src" / "model.py",
                args.config.parent / "src" / "evaluation.py",
                args.config.parent / "tests" / "test_analysis.py",
                args.config.parent / "tests" / "test_data.py",
                args.config.parent / "tests" / "test_interpreter_alignment.py",
                args.config.parent / "tests" / "test_model_evaluation.py",
                args.config.parent.parent / "deepcase" / "context_builder" / "context_builder.py",
                args.config.parent.parent / "deepcase" / "interpreter" / "interpreter.py",
                manifest_path,
            )
        },
    }
    artifact = (
        args.config.parent
        / "artifacts"
        / (
            f"validation_{args.namespace}_{args.stage}_{args.label}_"
            f"ratio_{args.ratio}_q{expected_query_iterations}.json"
        )
    )
    artifact.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
