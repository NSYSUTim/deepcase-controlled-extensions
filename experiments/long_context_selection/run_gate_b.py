"""Run the preregistered Last10/Last100/oracle10 DeepCASE upper-bound gate."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_experiments.src.data import RevealedLabels, fold_indices, scenario_fold
from research_experiments.src.model import (
    AttendedCache,
    IncidentAlignedContextBuilder,
    fit_cached_interpreter_allow_unlabeled,
    predict_cached_interpreter,
    set_reproducible_seed,
    train_context_builder,
)
from src.common import EXPERIMENT, ROOT, load_config, resolve_repo
from src.evaluation import strict_metrics


def emit(message: str) -> None:
    print(message, flush=True)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def stage_model_config(config: dict, stage: str) -> dict:
    result = dict(config["model"])
    if stage == "smoke":
        result.update(hidden_size=16, epochs=1, steps_per_epoch=10)
    elif stage == "pilot":
        result.update(hidden_size=32, epochs=2, steps_per_epoch=100)
    return result


def event_encoding(
    context_global: np.ndarray,
    target_global: np.ndarray,
    train_indices: np.ndarray,
    global_pad_id: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit vocabulary on training scenarios and map every other event to UNK."""

    seen = np.unique(target_global[train_indices]).astype(np.int16)
    mapping = np.full(global_pad_id + 1, -1, dtype=np.int16)
    mapping[seen] = np.arange(len(seen), dtype=np.int16)
    unk_id = len(seen)
    local_pad_id = len(seen) + 1
    mapping[:global_pad_id][mapping[:global_pad_id] < 0] = unk_id
    mapping[global_pad_id] = local_pad_id
    local_context = mapping[context_global]
    local_target = mapping[target_global]
    return local_context, local_target, {
        "input_size": int(local_pad_id + 1),
        "seen_events": int(len(seen)),
        "unk_id": int(unk_id),
        "pad_id": int(local_pad_id),
    }


def all_training_labels(
    train_indices: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> RevealedLabels:
    train_labels = labels[train_indices]
    positive = train_indices[train_labels == 1]
    negative = train_indices[train_labels == 0]
    available_groups = np.unique(groups[positive])
    return RevealedLabels(
        indices=train_indices.copy(),
        labels=train_labels.copy(),
        positive_indices=positive,
        negative_indices=negative,
        requested_ratio=0,
        realized_ratio=float(len(negative) / len(positive)),
        label_budget=int(len(train_indices)),
        positive_groups_covered=int(len(available_groups)),
        positive_groups_available=int(len(available_groups)),
    )


def compact_full_policy_set(
    context: np.ndarray,
    target: np.ndarray,
    train_indices: np.ndarray,
    labels: np.ndarray,
    max_multiplicity: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Collapse exact patterns while retaining the DBSCAN multiplicity needed.

    Unlike a row subsample, this is lossless for DBSCAN core status after each
    exact duplicate reaches ``min_samples``.  Scores use all training labels.
    """

    patterns = np.concatenate(
        [context[train_indices], target[train_indices, None]], axis=1
    )
    unique, inverse, counts = np.unique(
        patterns, axis=0, return_inverse=True, return_counts=True
    )
    scores = np.zeros(len(unique), dtype=np.int8)
    np.maximum.at(scores, inverse, labels[train_indices])
    repeats = np.minimum(counts, max_multiplicity).astype(np.int16)
    stats = {
        "raw_train_rows": int(len(train_indices)),
        "unique_context_target_patterns": int(len(unique)),
        "expanded_fit_rows": int(repeats.sum()),
        "labeled_unique_patterns": int(len(unique)),
        "positive_unique_patterns": int(scores.sum()),
        "compression_is_row_sampling": False,
    }
    return unique[:, :-1], unique[:, -1], scores, repeats, stats


def _direct_attention_batch(
    model: IncidentAlignedContextBuilder,
    context: np.ndarray,
    target: np.ndarray,
    features: int,
) -> tuple[np.ndarray, sp.csr_matrix]:
    """Exact query-iterations=0 path without the upstream repeated torch.unique."""

    X = torch.as_tensor(np.asarray(context), dtype=torch.long)
    y = torch.as_tensor(np.asarray(target), dtype=torch.long)
    model.eval()
    with torch.no_grad():
        confidence_all, attention_all = model.forward(
            X, steps=1, teach_ratio=0.0
        )
        confidence = confidence_all[:, 0].exp()[torch.arange(len(y)), y]
        attention = attention_all[:, 0]
    row = np.repeat(np.arange(len(X), dtype=np.int32), X.shape[1])
    column = X.numpy().reshape(-1)
    weight = attention.numpy().reshape(-1)
    vectors = sp.csr_matrix(
        (weight, (row, column)), shape=(len(X), features)
    )
    vectors.sum_duplicates()
    vectors.data = np.round(vectors.data, decimals=4)
    vectors.eliminate_zeros()
    return confidence.numpy(), vectors


def streaming_unique_cache(
    model: IncidentAlignedContextBuilder,
    context: np.ndarray,
    target: np.ndarray,
    *,
    features: int,
    chunk_rows: int = 4096,
) -> AttendedCache:
    """Compute all unique-row representations with bounded resident memory."""

    confidence = np.empty(len(target), dtype=np.float32)
    vector_parts = []
    for start in range(0, len(target), chunk_rows):
        stop = min(start + chunk_rows, len(target))
        confidence[start:stop], vectors = _direct_attention_batch(
            model, context[start:stop], target[start:stop], features
        )
        vector_parts.append(vectors)
    stacked = sp.vstack(vector_parts, format="csr") if vector_parts else sp.csr_matrix((0, features))
    return AttendedCache(
        confidence=confidence,
        vectors=stacked,
        events=np.asarray(target).copy(),
        inverse=np.arange(len(target), dtype=np.int64),
    )


def fit_cache_from_unique(
    unique_cache: AttendedCache,
    unique_scores: np.ndarray,
    repeats: np.ndarray,
) -> tuple[AttendedCache, np.ndarray]:
    expansion = np.repeat(np.arange(len(repeats)), repeats.astype(np.int64))
    cache = AttendedCache(
        confidence=unique_cache.confidence[expansion],
        vectors=unique_cache.vectors[expansion],
        events=unique_cache.events[expansion],
        inverse=np.arange(len(expansion), dtype=np.int64),
    )
    return cache, unique_scores[expansion]


def split_unique_patterns(
    context: np.ndarray,
    target: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    patterns = np.concatenate([context[indices], target[indices, None]], axis=1)
    unique, inverse = np.unique(patterns, axis=0, return_inverse=True)
    return unique[:, :-1], unique[:, -1], inverse


def evaluate_cache(interpreter, cache, truth, groups) -> tuple[dict, np.ndarray]:
    prediction = predict_cached_interpreter(interpreter, cache)
    return strict_metrics(truth, prediction, groups), prediction


def model_result_path(stage: str, scenario: str, method: str, seed: int) -> Path:
    return EXPERIMENT / "results" / "gate_b" / stage / scenario / method / f"seed_{seed}.json"


def run_one(
    *,
    stage: str,
    scenario: str,
    method: str,
    seed: int,
    context: np.ndarray,
    target: np.ndarray,
    vocabulary: dict,
    split: dict[str, np.ndarray],
    metadata: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
    revealed: RevealedLabels,
    compact_fit: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict],
    config: dict,
    force: bool,
) -> None:
    output = model_result_path(stage, scenario, method, seed)
    checkpoint = output.with_suffix(".pt")
    if output.exists() and not force:
        try:
            record = json.loads(output.read_text(encoding="utf-8"))
            if record.get("status") == "complete":
                emit(f"skip complete: {output}")
                return
        except (OSError, json.JSONDecodeError):
            pass

    model_cfg = stage_model_config(config, stage)
    set_reproducible_seed(seed)
    model = IncidentAlignedContextBuilder(
        input_size=vocabulary["input_size"],
        hidden_size=int(model_cfg["hidden_size"]),
        max_length=context.shape[1],
    )
    base = {
        "status": "running",
        "gate": "B",
        "stage": stage,
        "test_scenario": scenario,
        "method": method,
        "seed": int(seed),
        "train_scenarios": sorted(metadata.loc[split["train"], "scenario"].unique().tolist()),
        "validation_scenario": str(metadata.loc[split["validation"], "scenario"].iloc[0]),
        "split_rows": {key: int(len(value)) for key, value in split.items()},
        "context_length": int(context.shape[1]),
        "vocabulary": vocabulary,
        "cluster_policy_labels": "all_training_scenario_rows",
        "cluster_policy_label_rows": int(len(revealed.indices)),
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
        "git_revision": git_revision(),
        "config_sha256": file_hash(EXPERIMENT / "config.yaml"),
        "protocol_sha256": file_hash(EXPERIMENT / "protocol.md"),
        "context_variants_sha256": file_hash(EXPERIMENT / "artifacts" / "context_variants.json"),
        "scoring_index_version": "aligned_v2",
    }
    atomic_json(output, base)

    training_log = []
    if checkpoint.exists() and not force:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        expected = (method, scenario, seed, context.shape[1], vocabulary["input_size"])
        observed = (
            saved["method"], saved["scenario"], saved["seed"],
            saved["context_length"], saved["input_size"],
        )
        if observed != expected:
            raise ValueError(f"checkpoint mismatch: {observed} != {expected}")
        model.load_state_dict(saved["state_dict"])
        training_seconds = saved["training_seconds"]
        training_log = saved["training_log"]
        emit(f"{scenario}/{method}/{seed}: resumed checkpoint")
    else:
        emit(f"{scenario}/{method}/{seed}: train")
        started = time.perf_counter()
        logs = train_context_builder(
            model=model,
            context=context,
            target_event=target,
            train_indices=split["train"],
            revealed=revealed,
            metadata=metadata,
            label_column="event_incident",
            group_column="event_incident_group",
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
            progress=lambda item: emit(f"{scenario}/{method}/{seed}: {item}"),
        )
        training_seconds = time.perf_counter() - started
        training_log = [asdict(item) for item in logs]
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "method": method,
                "scenario": scenario,
                "seed": seed,
                "context_length": int(context.shape[1]),
                "input_size": vocabulary["input_size"],
                "training_seconds": training_seconds,
                "training_log": training_log,
            },
            checkpoint,
        )

    X_fit, y_fit, unique_fit_scores, repeats, compact_stats = compact_fit
    emit(f"{scenario}/{method}/{seed}: attention caches; fit={compact_stats}")
    cache_started = time.perf_counter()
    fit_unique_cache = streaming_unique_cache(
        model, X_fit, y_fit,
        features=vocabulary["input_size"],
    )
    fit_cache, fit_scores = fit_cache_from_unique(
        fit_unique_cache, unique_fit_scores, repeats
    )
    caches = {}
    for split_name in ("validation", "test"):
        X_unique, y_unique, inverse = split_unique_patterns(
            context, target, split[split_name]
        )
        unique_cache = streaming_unique_cache(
            model, X_unique, y_unique,
            features=vocabulary["input_size"],
        )
        caches[split_name] = AttendedCache(
            confidence=unique_cache.confidence,
            vectors=unique_cache.vectors,
            events=unique_cache.events,
            inverse=inverse,
        )
    cache_seconds = time.perf_counter() - cache_started

    points = []
    for eps, threshold in config["interpreter"]["settings"]:
        started = time.perf_counter()
        interpreter = fit_cached_interpreter_allow_unlabeled(
            model, fit_cache, fit_scores,
            features=vocabulary["input_size"],
            eps=float(eps),
            min_samples=int(config["interpreter"]["min_samples"]),
            threshold=float(threshold),
        )
        point = {"eps": float(eps), "threshold": float(threshold)}
        for split_name in ("validation", "test"):
            rows = split[split_name]
            point[split_name], _ = evaluate_cache(
                interpreter,
                caches[split_name],
                labels[rows],
                groups[rows],
            )
        point["seconds"] = time.perf_counter() - started
        points.append(point)
        emit(
            f"{scenario}/{method}/{seed}: eps={eps} threshold={threshold} "
            f"test workload={point['test']['review_workload']:.4f} "
            f"strict recall={point['test']['automatic_incident_recall']:.4f}"
        )

    record = {
        **base,
        "status": "complete",
        "training_seconds": training_seconds,
        "training_log": training_log,
        "cache_seconds": cache_seconds,
        "compact_interpreter_fit": compact_stats,
        "operating_points": points,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_hash(checkpoint),
    }
    atomic_json(output, record)


def selected(values: list, one):
    return values if one is None else [one]


def run(args: argparse.Namespace) -> None:
    config = load_config()
    metadata = pd.read_parquet(resolve_repo(config["data"]["canonical_parquet"]))
    if not np.array_equal(metadata["row_id"].to_numpy(), np.arange(len(metadata))):
        raise ValueError("row_id is not a dense positional index")
    target_global = metadata["event_id"].to_numpy(dtype=np.int16, copy=True)
    labels = metadata["event_incident"].to_numpy(dtype=np.int8, copy=True)
    groups = metadata["event_incident_group"].astype(object).to_numpy(copy=True)
    scenarios_all = list(config["design"]["scenarios"])
    default_scenarios = (
        ["russellmitchell"] if args.stage == "smoke"
        else list(config["design"]["design_scenarios"])
    )
    default_seeds = (
        [int(config["design"]["seeds"][0])]
        if args.stage in {"smoke", "pilot"}
        else [int(value) for value in config["design"]["seeds"]]
    )
    methods = selected(list(config["gate_b"]["methods"]), args.method)
    for scenario in selected(default_scenarios, args.test_scenario):
        fold = scenario_fold(scenarios_all, scenario)
        split = fold_indices(metadata, fold)
        revealed = all_training_labels(split["train"], labels, groups)
        emit(
            f"fold test={scenario} validation={fold.validation_scenario} "
            f"train={fold.train_scenarios}"
        )
        for method in methods:
            if method in {
                "class_oracle10",
                "association_abs10",
                "anchor5_assoc5",
            }:
                path = (
                    EXPERIMENT
                    / "artifacts"
                    / f"contexts_{method}_{scenario}.npy"
                )
            else:
                path = EXPERIMENT / "artifacts" / f"contexts_{method}.npy"
            context_global = np.load(path, mmap_mode="r")
            context, target, vocabulary = event_encoding(
                context_global,
                target_global,
                split["train"],
                global_pad_id=int(target_global.max()) + 1,
            )
            emit(f"{scenario}/{method}: compact full training policy set")
            compact = compact_full_policy_set(
                context,
                target,
                split["train"],
                labels,
                max_multiplicity=int(
                    config["interpreter"]["max_multiplicity_for_fit"]
                ),
            )
            for seed in selected(default_seeds, args.seed):
                run_one(
                    stage=args.stage,
                    scenario=scenario,
                    method=method,
                    seed=int(seed),
                    context=context,
                    target=target,
                    vocabulary=vocabulary,
                    split=split,
                    metadata=metadata,
                    labels=labels,
                    groups=groups,
                    revealed=revealed,
                    compact_fit=compact,
                    config=config,
                    force=args.force,
                )
            del compact, context, context_global
            gc.collect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["smoke", "pilot", "full"], default="smoke")
    parser.add_argument("--test-scenario", choices=[
        "fox", "harrison", "russellmitchell", "santos", "shaw", "wardbeck", "wheeler", "wilson"
    ])
    parser.add_argument(
        "--method",
        choices=[
            "last10",
            "last100",
            "oracle10",
            "class_oracle10",
            "association_abs10",
            "anchor5_assoc5",
        ],
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
