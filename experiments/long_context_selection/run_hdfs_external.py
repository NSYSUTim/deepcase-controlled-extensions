"""Blind HDFS DeepCASE runner: final outcomes are absent from result metrics."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import sklearn
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_experiments.src.model import (
    IncidentAlignedContextBuilder,
    fit_cached_interpreter_allow_unlabeled,
    predict_cached_interpreter,
    set_reproducible_seed,
    train_context_builder,
)
from run_gate_b import (
    all_training_labels,
    atomic_json,
    compact_full_policy_set,
    event_encoding,
    file_hash,
    fit_cache_from_unique,
    split_unique_patterns,
    streaming_unique_cache,
)
from src.common import EXPERIMENT, sha256


CONFIG_PATH = EXPERIMENT / "external_config_v2.yaml"
PROTOCOL_PATH = EXPERIMENT / "post_failure_external_protocol_v2.md"
ARTIFACTS = EXPERIMENT / "hdfs_artifacts"
RESULTS = EXPERIMENT / "results" / "hdfs_external"


def emit(message: str) -> None:
    print(message, flush=True)


def _divide(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def labeled_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    truth = np.asarray(truth, dtype=np.int8)
    prediction = np.asarray(prediction, dtype=np.int8)
    incident = truth == 1
    nonincident = ~incident
    auto_incident = prediction > 0
    auto_nonincident = prediction == 0
    reject = prediction < 0
    tp1 = int(np.sum(auto_incident & incident))
    fp1 = int(np.sum(auto_incident & nonincident))
    fn1 = int(np.sum(~auto_incident & incident))
    tp0 = int(np.sum(auto_nonincident & nonincident))
    fp0 = int(np.sum(auto_nonincident & incident))
    fn0 = int(np.sum(~auto_nonincident & nonincident))
    f1_incident = _divide(2 * tp1, 2 * tp1 + fp1 + fn1)
    f1_normal = _divide(2 * tp0, 2 * tp0 + fp0 + fn0)
    return {
        "alerts": int(len(truth)),
        "incidents": int(incident.sum()),
        "nonincidents": int(nonincident.sum()),
        "automatic_incident_recall": _divide(tp1, int(incident.sum())),
        "automatic_incident_precision": _divide(tp1, tp1 + fp1),
        "strict_macro_f1": 0.5 * (f1_incident + f1_normal),
        "review_workload": float(np.mean(auto_incident | reject)),
        "automatic_incident_rate": float(np.mean(auto_incident)),
        "automatic_nonincident_rate": float(np.mean(auto_nonincident)),
        "reject_rate": float(np.mean(reject)),
        "false_escalations_per_1000_nonincidents": 1000.0
        * _divide(fp1, int(nonincident.sum())),
        "auto_true_incident": tp1,
        "auto_false_incident": fp1,
        "auto_true_nonincident": tp0,
        "auto_false_nonincident": fp0,
        "rejected_incident": int(np.sum(reject & incident)),
        "rejected_nonincident": int(np.sum(reject & nonincident)),
    }


def unlabeled_metrics(prediction: np.ndarray) -> dict:
    prediction = np.asarray(prediction, dtype=np.int8)
    return {
        "alerts": int(len(prediction)),
        "review_workload": float(np.mean(prediction != 0)),
        "automatic_incident_rate": float(np.mean(prediction > 0)),
        "automatic_nonincident_rate": float(np.mean(prediction == 0)),
        "reject_rate": float(np.mean(prediction < 0)),
        "final_outcomes_accessed": False,
    }


def run_one(method: str, seed: int, config: dict, force: bool) -> None:
    method_dir = RESULTS / method
    output = method_dir / f"seed_{seed}.json"
    checkpoint = method_dir / f"seed_{seed}.pt"
    prediction_path = method_dir / f"seed_{seed}_predictions.npy"
    if output.exists() and prediction_path.exists() and not force:
        record = json.loads(output.read_text(encoding="utf-8"))
        if record.get("status") == "complete_blind":
            emit(f"skip complete: {output}")
            return

    target_global = np.load(ARTIFACTS / "target_events.npy", mmap_mode="r")
    splits = np.load(ARTIFACTS / "split_codes.npy", mmap_mode="r")
    labels = np.load(ARTIFACTS / "labels_blind.npy", mmap_mode="r")
    groups = np.load(ARTIFACTS / "sequence_ids.npy", mmap_mode="r")
    context_path = ARTIFACTS / (
        "contexts_last10.npy"
        if method == "last10"
        else "contexts_anchor5_assoc5.npy"
    )
    context_global = np.load(context_path, mmap_mode="r")
    train = np.flatnonzero(splits <= 1)
    validation = np.flatnonzero(splits == 2)
    test = np.flatnonzero(splits == 3)
    split = {"train": train, "validation": validation, "test": test}
    manifest = json.loads((ARTIFACTS / "manifest.json").read_text(encoding="utf-8"))
    global_pad_id = int(manifest["materialized"]["pad_id"])
    context, target, vocabulary = event_encoding(
        context_global, np.asarray(target_global), train, global_pad_id
    )
    revealed = all_training_labels(train, np.asarray(labels), np.asarray(groups))
    compact = compact_full_policy_set(
        context,
        target,
        train,
        labels,
        max_multiplicity=int(config["interpreter"]["max_multiplicity_for_fit"]),
    )

    model_cfg = config["model"]
    set_reproducible_seed(seed)
    model = IncidentAlignedContextBuilder(
        input_size=vocabulary["input_size"],
        hidden_size=int(model_cfg["hidden_size"]),
        max_length=context.shape[1],
    )
    base = {
        "status": "running_blind",
        "dataset": "HDFS_DeepLog_preprocessed",
        "method": method,
        "seed": seed,
        "split_rows": {name: int(len(rows)) for name, rows in split.items()},
        "context_length": int(context.shape[1]),
        "vocabulary": vocabulary,
        "final_labels_masked_to": -9,
        "final_outcomes_accessed_for_metrics": False,
        "cluster_policy_labels": "official_train_plus_development",
        "cluster_policy_label_rows": int(len(train)),
        "model_config": model_cfg,
        "interpreter_settings": config["interpreter"]["settings"],
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "scikit_learn": sklearn.__version__,
            "platform": platform.platform(),
        },
        "external_config_sha256": sha256(CONFIG_PATH),
        "external_protocol_sha256": sha256(PROTOCOL_PATH),
        "manifest_sha256": sha256(ARTIFACTS / "manifest.json"),
        "context_sha256": sha256(context_path),
        "scoring_index_version": "aligned_v2",
    }
    atomic_json(output, base)

    if checkpoint.exists() and not force:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        expected = (method, seed, context.shape[1], vocabulary["input_size"])
        observed = (
            saved["method"],
            saved["seed"],
            saved["context_length"],
            saved["input_size"],
        )
        if observed != expected:
            raise ValueError(f"checkpoint mismatch: {observed} != {expected}")
        model.load_state_dict(saved["state_dict"])
        training_seconds = saved["training_seconds"]
        training_log = saved["training_log"]
        emit(f"hdfs/{method}/{seed}: resumed checkpoint")
    else:
        emit(f"hdfs/{method}/{seed}: train")
        started = time.perf_counter()
        logs = train_context_builder(
            model=model,
            context=context,
            target_event=target,
            train_indices=train,
            revealed=revealed,
            metadata=None,
            label_column="unused",
            group_column="unused",
            method="deepcase",
            seed=seed,
            epochs=int(model_cfg["epochs"]),
            steps_per_epoch=int(model_cfg["steps_per_epoch"]),
            batch_size=int(model_cfg["batch_size"]),
            learning_rate=float(model_cfg["learning_rate"]),
            weight_decay=float(model_cfg["weight_decay"]),
            label_smoothing=float(model_cfg["label_smoothing"]),
            incident_lambda=0.0,
            focal_gamma=2.0,
            effective_beta=0.999,
            progress=lambda item: emit(f"hdfs/{method}/{seed}: {item}"),
        )
        training_seconds = time.perf_counter() - started
        training_log = [asdict(item) for item in logs]
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "method": method,
                "seed": seed,
                "context_length": int(context.shape[1]),
                "input_size": vocabulary["input_size"],
                "training_seconds": training_seconds,
                "training_log": training_log,
            },
            checkpoint,
        )

    X_fit, y_fit, fit_pattern_scores, repeats, compact_stats = compact
    emit(f"hdfs/{method}/{seed}: attention caches; fit={compact_stats}")
    cache_started = time.perf_counter()
    fit_unique_cache = streaming_unique_cache(
        model, X_fit, y_fit, features=vocabulary["input_size"]
    )
    fit_cache, fit_scores = fit_cache_from_unique(
        fit_unique_cache, fit_pattern_scores, repeats
    )
    caches = {}
    for name in ("validation", "test"):
        X_unique, y_unique, inverse = split_unique_patterns(
            context, target, split[name]
        )
        unique_cache = streaming_unique_cache(
            model, X_unique, y_unique, features=vocabulary["input_size"]
        )
        caches[name] = type(unique_cache)(
            confidence=unique_cache.confidence,
            vectors=unique_cache.vectors,
            events=unique_cache.events,
            inverse=inverse,
        )
    cache_seconds = time.perf_counter() - cache_started

    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    predictions = np.lib.format.open_memmap(
        prediction_path,
        mode="w+",
        dtype=np.int8,
        shape=(len(config["interpreter"]["settings"]), len(test)),
    )
    points = []
    for point_index, (eps, threshold) in enumerate(
        config["interpreter"]["settings"]
    ):
        started = time.perf_counter()
        interpreter = fit_cached_interpreter_allow_unlabeled(
            model,
            fit_cache,
            fit_scores,
            features=vocabulary["input_size"],
            eps=float(eps),
            min_samples=int(config["interpreter"]["min_samples"]),
            threshold=float(threshold),
        )
        validation_prediction = predict_cached_interpreter(
            interpreter, caches["validation"]
        ).astype(np.int8)
        test_prediction = predict_cached_interpreter(
            interpreter, caches["test"]
        ).astype(np.int8)
        predictions[point_index] = test_prediction
        point = {
            "eps": float(eps),
            "threshold": float(threshold),
            "validation": labeled_metrics(
                labels[validation], validation_prediction
            ),
            "test_unlabeled": unlabeled_metrics(test_prediction),
            "seconds": time.perf_counter() - started,
        }
        points.append(point)
        emit(
            f"hdfs/{method}/{seed}: eps={eps} threshold={threshold} "
            f"validation false/1000="
            f"{point['validation']['false_escalations_per_1000_nonincidents']:.4f} "
            f"blind test workload={point['test_unlabeled']['review_workload']:.4f}"
        )
    predictions.flush()
    record = {
        **base,
        "status": "complete_blind",
        "training_seconds": training_seconds,
        "training_log": training_log,
        "cache_seconds": cache_seconds,
        "compact_interpreter_fit": compact_stats,
        "operating_points": points,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_hash(checkpoint),
        "predictions": str(prediction_path),
        "predictions_sha256": file_hash(prediction_path),
    }
    atomic_json(output, record)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["last10", "anchor5_assoc5"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    seeds = config["evaluation"]["model_seeds"]
    if args.seed is not None:
        seeds = [args.seed]
    for seed in seeds:
        run_one(args.method, int(seed), config, args.force)


if __name__ == "__main__":
    main()
