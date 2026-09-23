"""Run paired smoke, pilot, or full AIT-ADS experiments with resumable outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
import yaml

from src.data import (
    AmbiguousRuleDefinition,
    define_ambiguous_rules,
    encode_for_fold,
    fold_indices,
    load_prepared,
    reveal_training_labels,
    scenario_fold,
)
from src.evaluation import metrics_dict
from src.model import (
    IncidentAlignedContextBuilder,
    build_attended_cache,
    compact_interpreter_fit_set,
    fit_cached_interpreter_allow_unlabeled,
    fit_interpreter_allow_unlabeled,
    fit_rule_prior,
    predict_interpreter,
    predict_cached_interpreter,
    predict_rule_prior,
    set_reproducible_seed,
    train_context_builder,
)


def emit(message: str) -> None:
    print(message, flush=True)


def package_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "scikit_learn": sklearn.__version__,
    }


def git_revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize_metrics(value: dict) -> dict:
    return {
        key: (None if isinstance(item, float) and not np.isfinite(item) else item)
        for key, item in value.items()
    }


def operating_settings(config: dict, stage: str) -> list[tuple[float, float]]:
    interpreter = config["interpreter"]
    if stage == "smoke":
        return [(0.1, 0.2)]
    if stage == "pilot":
        return [
            (float(eps), float(threshold))
            for eps in interpreter["pilot_eps_grid"]
            for threshold in interpreter["pilot_threshold_grid"]
        ]
    return [
        (float(eps), float(threshold))
        for eps, threshold in interpreter["full_settings"]
    ]


def rule_thresholds(stage: str) -> list[float]:
    return [0.5] if stage == "smoke" else [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]


def stage_model_config(config: dict, stage: str) -> dict:
    result = dict(config["model"])
    if stage == "smoke":
        result.update({"hidden_size": 16, "epochs": 1, "steps_per_epoch": 10})
    elif stage == "pilot":
        result.update({"hidden_size": 32, "epochs": 2, "steps_per_epoch": 100})
    return result


def evaluate_predictions(
    metadata: pd.DataFrame,
    indices: np.ndarray,
    label_column: str,
    group_column: str,
    predictions: np.ndarray,
) -> dict:
    truth = metadata.loc[indices, label_column].to_numpy(dtype=np.int8)
    groups = metadata.loc[indices, group_column].to_numpy(dtype=object)
    return sanitize_metrics(metrics_dict(truth, predictions, groups))


def evaluate_prediction_bundle(
    metadata: pd.DataFrame,
    indices: np.ndarray,
    label_column: str,
    group_column: str,
    predictions: np.ndarray,
    ambiguous_rules: AmbiguousRuleDefinition,
    rule_column: str,
) -> tuple[dict, dict]:
    """Return the primary natural-stream metric and a secondary diagnostic.

    The ambiguous-rule subset is defined from training scenarios before this
    function is called.  Test outcomes therefore cannot determine membership.
    """

    overall = evaluate_predictions(
        metadata, indices, label_column, group_column, predictions
    )
    rule_values = metadata.loc[indices, rule_column].astype(str).to_numpy()
    mask = np.isin(rule_values, ambiguous_rules.rules)
    subset_indices = indices[mask]
    diagnostic = {
        "rows": int(mask.sum()),
        "rule_count_observed": int(len(np.unique(rule_values[mask]))),
        "metrics": None,
    }
    if mask.any():
        diagnostic["metrics"] = evaluate_predictions(
            metadata,
            subset_indices,
            label_column,
            group_column,
            predictions[mask],
        )
    return overall, diagnostic


def run_rule_prior(
    output_file: Path,
    base_record: dict,
    encoded,
    revealed,
    indices: dict[str, np.ndarray],
    metadata: pd.DataFrame,
    label_column: str,
    group_column: str,
    ambiguous_rules: AmbiguousRuleDefinition,
    rule_column: str,
    stage: str,
) -> None:
    posterior = fit_rule_prior(encoded.target, revealed, encoded.input_size)
    points = []
    for threshold in rule_thresholds(stage):
        point = {"rule_threshold": threshold, "diagnostic_strata": {}}
        for split in ("validation", "test"):
            prediction = predict_rule_prior(
                posterior,
                encoded.target,
                indices[split],
                threshold,
                encoded.unk_id,
            )
            overall, diagnostic = evaluate_prediction_bundle(
                metadata,
                indices[split],
                label_column,
                group_column,
                prediction,
                ambiguous_rules,
                rule_column,
            )
            point[split] = overall
            point["diagnostic_strata"][split] = {
                "ambiguous_rule": diagnostic
            }
        points.append(point)
    record = {**base_record, "operating_points": points, "status": "complete"}
    atomic_json(output_file, record)


def run_neural_method(
    output_file: Path,
    checkpoint_file: Path,
    base_record: dict,
    method: str,
    encoded,
    compact_fit: tuple[np.ndarray, np.ndarray, np.ndarray, dict],
    revealed,
    indices: dict[str, np.ndarray],
    metadata: pd.DataFrame,
    label_column: str,
    group_column: str,
    ambiguous_rules: AmbiguousRuleDefinition,
    rule_column: str,
    config: dict,
    stage: str,
    seed: int,
    evaluate_only: bool,
    previous_record: dict | None,
    query_iterations_override: int | None,
) -> None:
    model_cfg = stage_model_config(config, stage)
    set_reproducible_seed(seed)
    model = IncidentAlignedContextBuilder(
        input_size=encoded.input_size,
        hidden_size=int(model_cfg["hidden_size"]),
        max_length=encoded.context.shape[1],
    )
    if evaluate_only:
        if not checkpoint_file.exists():
            raise FileNotFoundError(
                f"--evaluate-only requires an existing checkpoint: {checkpoint_file}"
            )
        checkpoint = torch.load(
            checkpoint_file, map_location="cpu", weights_only=False
        )
        expected = (encoded.input_size, int(model_cfg["hidden_size"]), method, seed)
        observed = (
            int(checkpoint["input_size"]),
            int(checkpoint["hidden_size"]),
            str(checkpoint["method"]),
            int(checkpoint["seed"]),
        )
        if observed != expected:
            raise ValueError(
                f"Checkpoint metadata mismatch; expected {expected}, observed {observed}"
            )
        model.load_state_dict(checkpoint["state_dict"])
        training_seconds = (
            None if previous_record is None else previous_record.get("training_seconds")
        )
        training_log = (
            [] if previous_record is None else previous_record.get("training_log", [])
        )
        trained_model_config = (
            model_cfg
            if previous_record is None
            else previous_record.get("model_config", model_cfg)
        )
        emit(f"{method}: loaded existing checkpoint; training was not rerun")
    else:
        start = time.perf_counter()
        logs = train_context_builder(
            model=model,
            context=encoded.context,
            target_event=encoded.target,
            train_indices=indices["train"],
            revealed=revealed,
            metadata=metadata,
            label_column=label_column,
            group_column=group_column,
            method=method,
            seed=seed,
            epochs=int(model_cfg["epochs"]),
            steps_per_epoch=int(model_cfg["steps_per_epoch"]),
            batch_size=int(model_cfg["batch_size"]),
            learning_rate=float(model_cfg["learning_rate"]),
            weight_decay=float(model_cfg["weight_decay"]),
            label_smoothing=float(model_cfg["label_smoothing"]),
            incident_lambda=float(model_cfg["incident_lambda"]),
            focal_gamma=float(model_cfg["focal_gamma"]),
            effective_beta=float(model_cfg["effective_beta"]),
            progress=lambda item: emit(f"{method}: {item}"),
        )
        training_seconds = time.perf_counter() - start
        training_log = [asdict(item) for item in logs]
        trained_model_config = model_cfg
    X_fit, y_fit, fit_scores, compact_stats = compact_fit
    emit(f"{method}: cache threshold-independent attended representations")
    query_iterations = int(
        config["interpreter"]["query_iterations"]
        if query_iterations_override is None
        else query_iterations_override
    )
    fit_cache = build_attended_cache(
        model,
        X_fit,
        y_fit,
        None,
        features=encoded.input_size,
        query_iterations=query_iterations,
        preserve_rows=True,
    )
    split_caches = {
        split: build_attended_cache(
            model,
            encoded.context,
            encoded.target,
            indices[split],
            features=encoded.input_size,
            query_iterations=query_iterations,
            preserve_rows=False,
        )
        for split in ("validation", "test")
    }
    points = []
    for eps, threshold in operating_settings(config, stage):
        emit(f"{method}: Interpreter eps={eps} threshold={threshold}")
        op_start = time.perf_counter()
        interpreter = fit_cached_interpreter_allow_unlabeled(
            model,
            fit_cache,
            fit_scores,
            features=encoded.input_size,
            eps=eps,
            min_samples=int(config["interpreter"]["min_samples"]),
            threshold=threshold,
        )
        point = {"eps": eps, "threshold": threshold, "diagnostic_strata": {}}
        for split in ("validation", "test"):
            predictions = predict_cached_interpreter(
                interpreter,
                split_caches[split],
            )
            overall, diagnostic = evaluate_prediction_bundle(
                metadata,
                indices[split],
                label_column,
                group_column,
                predictions,
                ambiguous_rules,
                rule_column,
            )
            point[split] = overall
            point["diagnostic_strata"][split] = {
                "ambiguous_rule": diagnostic
            }
        point["seconds"] = time.perf_counter() - op_start
        points.append(point)

    if not evaluate_only:
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "input_size": encoded.input_size,
                "hidden_size": int(model_cfg["hidden_size"]),
                "max_length": encoded.context.shape[1],
                "method": method,
                "seed": seed,
            },
            checkpoint_file,
        )
    record = {
        **base_record,
        "model_config": trained_model_config,
        "training_seconds": training_seconds,
        "training_log": training_log,
        "evaluation_only": evaluate_only,
        "compact_interpreter_fit": compact_stats,
        "operating_points": points,
        "checkpoint": str(checkpoint_file),
        "checkpoint_sha256": sha256_file(checkpoint_file),
        "status": "complete",
    }
    atomic_json(output_file, record)


def selected_values(all_values, selected):
    return list(all_values) if selected is None else [selected]


def run(args: argparse.Namespace) -> None:
    if args.query_iterations is not None and args.query_iterations < 0:
        raise ValueError("--query-iterations must be non-negative")
    if args.checkpoint_source_ratio is not None and args.checkpoint_source is None:
        raise ValueError("--checkpoint-source-ratio requires --checkpoint-source")
    config_path = args.config.resolve()
    root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    data = load_prepared(
        root / config["data"]["canonical_parquet"],
        root / config["data"]["contexts_npz"],
    )
    label_column = config["data"]["primary_label"] if args.label is None else args.label
    group_column = (
        "event_incident_group" if label_column == "event_incident" else "window_incident_group"
    )

    if args.stage in {"smoke", "pilot"}:
        default_scenarios = ["russellmitchell"]
        default_ratios = [1000]
        default_seeds = [int(config["design"]["seeds"][0])]
        default_methods = (
            ["deepcase"] if args.stage == "smoke" else list(config["design"]["methods"])
        )
    else:
        default_scenarios = list(
            config["design"][
                "confirmatory_scenarios" if args.confirmatory_only else "scenarios"
            ]
        )
        default_ratios = [int(item) for item in config["design"]["imbalance_ratios"]]
        default_seeds = [int(item) for item in config["design"]["seeds"]]
        default_methods = list(config["design"]["methods"])

    scenarios = selected_values(default_scenarios, args.test_scenario)
    ratios = selected_values(default_ratios, args.ratio)
    seeds = selected_values(default_seeds, args.seed)
    methods = selected_values(default_methods, args.method)
    output_root = (
        config_path.parent / "results" / args.result_namespace / args.stage / label_column
    )
    environment = package_versions()
    revision = git_revision(root)

    for test_scenario in scenarios:
        fold = scenario_fold(config["design"]["scenarios"], test_scenario)
        indices = fold_indices(data.metadata, fold)
        emit(f"Fold train={fold.train_scenarios} val={fold.validation_scenario} test={test_scenario}")
        encoded = encode_for_fold(data, indices["train"])
        diagnostic_cfg = config["design"]["ambiguous_rule_diagnostic"]
        rule_column = str(diagnostic_cfg["rule_column"])
        ambiguous_rules = define_ambiguous_rules(
            data.metadata,
            indices["train"],
            label_column,
            rule_column=rule_column,
            lower_incident_rate=float(diagnostic_cfg["lower_incident_rate"]),
            upper_incident_rate=float(diagnostic_cfg["upper_incident_rate"]),
            minimum_per_class=int(diagnostic_cfg["minimum_per_class"]),
        )
        emit(
            "Secondary oracle diagnostic: "
            f"{len(ambiguous_rules.rules)} ambiguous training-defined rules"
        )
        for ratio in ratios:
            for seed in seeds:
                revealed = reveal_training_labels(
                    metadata=data.metadata,
                    train_indices=indices["train"],
                    label_column=label_column,
                    group_column=group_column,
                    label_budget=int(config["design"]["label_budget"]),
                    negative_to_positive_ratio=int(ratio),
                    seed=int(seed),
                )
                emit(
                    f"ratio=1:{ratio} seed={seed} labels={revealed.label_budget} "
                    f"positive_groups={revealed.positive_groups_covered}/"
                    f"{revealed.positive_groups_available}"
                )
                labels_all = data.metadata[label_column].to_numpy(dtype=np.int8)
                compact_fit = compact_interpreter_fit_set(
                    encoded.context,
                    encoded.target,
                    indices["train"],
                    revealed,
                    labels_all,
                    max_multiplicity=int(config["interpreter"]["max_multiplicity_for_fit"]),
                )
                emit(f"Interpreter compression: {compact_fit[3]}")
                for method in methods:
                    run_dir = output_root / test_scenario / f"ratio_{ratio}" / f"seed_{seed}"
                    output_file = run_dir / f"{method}.json"
                    checkpoint_file = run_dir / f"{method}.pt"
                    source_record = None
                    if args.checkpoint_source is not None:
                        source_ratio = (
                            ratio
                            if args.checkpoint_source_ratio is None
                            else args.checkpoint_source_ratio
                        )
                        source_run_dir = (
                            args.checkpoint_source.resolve()
                            / test_scenario / f"ratio_{source_ratio}" / f"seed_{seed}"
                        )
                        checkpoint_file = source_run_dir / f"{method}.pt"
                        source_json = source_run_dir / f"{method}.json"
                        if source_json.exists():
                            source_record = json.loads(source_json.read_text(encoding="utf-8"))
                    existing = None
                    if output_file.exists():
                        try:
                            existing = json.loads(output_file.read_text(encoding="utf-8"))
                            if (
                                existing.get("status") == "complete"
                                and not args.force
                            ):
                                emit(f"skip complete: {output_file}")
                                continue
                        except (OSError, json.JSONDecodeError):
                            pass
                    base_record = {
                        "status": "running",
                        "stage": args.stage,
                        "method": method,
                        "label_column": label_column,
                        "group_column": group_column,
                        "test_scenario": test_scenario,
                        "validation_scenario": fold.validation_scenario,
                        "train_scenarios": list(fold.train_scenarios),
                        "imbalance_ratio": ratio,
                        "seed": seed,
                        "revealed_labels": {},
                        "ambiguous_rule_diagnostic": asdict(ambiguous_rules),
                        "split_rows": {key: int(len(value)) for key, value in indices.items()},
                        "fold_vocabulary": {
                            "input_size": encoded.input_size,
                            "seen_events": int(len(encoded.seen_global_events)),
                            "unk_id": encoded.unk_id,
                            "pad_id": encoded.pad_id,
                        },
                        "environment": environment,
                        "result_namespace": args.result_namespace,
                        "scoring_index_version": "aligned_v2",
                        "interpreter_query_iterations": int(
                            config["interpreter"]["query_iterations"]
                            if args.query_iterations is None
                            else args.query_iterations
                        ),
                        "config_sha256": sha256_file(config_path),
                        "data_manifest_sha256": sha256_file(
                            root / config["data"]["manifest_json"]
                        ),
                        "config_snapshot": config,
                        "git_revision": revision,
                        "config_path": str(config_path),
                    }
                    # Dataclass arrays are intentionally summarized, never serialized.
                    base_record["revealed_labels"] = {
                        "requested_ratio": revealed.requested_ratio,
                        "realized_ratio": revealed.realized_ratio,
                        "label_budget": revealed.label_budget,
                        "positive_labels": int(len(revealed.positive_indices)),
                        "negative_labels": int(len(revealed.negative_indices)),
                        "positive_groups_covered": revealed.positive_groups_covered,
                        "positive_groups_available": revealed.positive_groups_available,
                    }
                    atomic_json(output_file, base_record)
                    emit(f"start {method}: {output_file}")
                    if method == "rule_prior":
                        run_rule_prior(
                            output_file,
                            base_record,
                            encoded,
                            revealed,
                            indices,
                            data.metadata,
                            label_column,
                            group_column,
                            ambiguous_rules,
                            rule_column,
                            args.stage,
                        )
                    else:
                        run_neural_method(
                            output_file,
                            checkpoint_file,
                            base_record,
                            method,
                            encoded,
                            compact_fit,
                            revealed,
                            indices,
                            data.metadata,
                            label_column,
                            group_column,
                            ambiguous_rules,
                            rule_column,
                            config,
                            args.stage,
                            seed,
                            args.evaluate_only,
                            source_record if source_record is not None else existing,
                            args.query_iterations,
                        )
                    emit(f"complete {method}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--stage", choices=["smoke", "pilot", "full"], default="smoke")
    parser.add_argument("--test-scenario", choices=[
        "fox", "harrison", "russellmitchell", "santos", "shaw", "wardbeck", "wheeler", "wilson"
    ])
    parser.add_argument("--ratio", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--method", choices=[
        "rule_prior", "deepcase", "aux_bce", "aux_weighted_bce", "aux_focal",
        "aux_group_balanced_bce", "proposed"
    ])
    parser.add_argument("--label", choices=["event_incident", "window_incident"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--confirmatory-only", action="store_true")
    parser.add_argument("--result-namespace", default="aligned_v2")
    parser.add_argument("--checkpoint-source", type=Path)
    parser.add_argument("--checkpoint-source-ratio", type=int)
    parser.add_argument("--query-iterations", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
