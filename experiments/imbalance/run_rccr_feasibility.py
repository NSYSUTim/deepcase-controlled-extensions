"""Run the design-only RC-CR feasibility experiment on frozen q=100 DeepCASE.

AIT-ADS has already been inspected.  Outputs from this script are therefore
exploratory design evidence and must not be described as a fresh confirmatory
test.  Test outcomes never choose an operating point or weight decay inside a
fold; the fold's validation scenario performs that selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import yaml

from run_experiment import (
    atomic_json,
    evaluate_prediction_bundle,
    operating_settings,
    package_versions,
    sanitize_metrics,
)
from src.data import (
    define_ambiguous_rules,
    encode_for_fold,
    fold_indices,
    load_prepared,
    reveal_training_labels,
    scenario_fold,
)
from src.evaluation import metrics_dict
from src.model import (
    AttendedCache,
    IncidentAlignedContextBuilder,
    build_attended_cache,
    compact_interpreter_fit_set,
    fit_cached_interpreter_allow_unlabeled,
    predict_cached_interpreter,
    set_reproducible_seed,
)
from src.rccr import (
    matched_workload_predictions,
    score_attended_vectors,
    serialize_risk_model,
    smoothed_rule_logits,
    train_linear_risk_model,
)


def emit(message: str) -> None:
    print(message, flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_files(directory: Path, name: str) -> tuple[Path, Path, Path]:
    return (
        directory / f"{name}_vectors.npz",
        directory / f"{name}_arrays.npz",
        directory / f"{name}_manifest.json",
    )


def save_cache(directory: Path, name: str, cache: AttendedCache, identity: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    vector_path, array_path, manifest_path = _cache_files(directory, name)
    sp.save_npz(vector_path, cache.vectors.tocsr(), compressed=True)
    np.savez_compressed(
        array_path,
        confidence=cache.confidence,
        events=cache.events,
        inverse=cache.inverse,
    )
    atomic_json(
        manifest_path,
        {
            **identity,
            "name": name,
            "vector_rows": int(cache.vectors.shape[0]),
            "vector_columns": int(cache.vectors.shape[1]),
            "expanded_rows": int(len(cache.inverse)),
            "vectors_sha256": sha256_file(vector_path),
            "arrays_sha256": sha256_file(array_path),
            "status": "complete",
        },
    )


def load_cache(directory: Path, name: str, identity: dict) -> AttendedCache | None:
    vector_path, array_path, manifest_path = _cache_files(directory, name)
    if not vector_path.exists() or not array_path.exists() or not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if manifest.get("status") != "complete":
        return None
    if any(manifest.get(key) != value for key, value in identity.items()):
        return None
    if manifest.get("vectors_sha256") != sha256_file(vector_path):
        return None
    if manifest.get("arrays_sha256") != sha256_file(array_path):
        return None
    arrays = np.load(array_path, allow_pickle=False)
    cache = AttendedCache(
        confidence=arrays["confidence"],
        vectors=sp.load_npz(vector_path),
        events=arrays["events"],
        inverse=arrays["inverse"],
    )
    if cache.vectors.shape[0] != len(cache.events):
        raise ValueError(f"Corrupt {name} cache: vector/event row mismatch")
    if len(cache.confidence) != len(cache.events):
        raise ValueError(f"Corrupt {name} cache: confidence/event row mismatch")
    return cache


def cache_or_build(
    directory: Path,
    name: str,
    identity: dict,
    builder,
    *,
    force: bool,
) -> AttendedCache:
    cached = None if force else load_cache(directory, name, identity)
    if cached is not None:
        emit(f"cache hit: {name} ({cached.vectors.shape[0]} vectors)")
        return cached
    emit(f"cache build start: {name}")
    started = time.perf_counter()
    cache = builder()
    save_cache(directory, name, cache, identity)
    emit(
        f"cache build complete: {name}, vectors={cache.vectors.shape[0]}, "
        f"seconds={time.perf_counter() - started:.1f}"
    )
    return cache


def expanded_scores(cache: AttendedCache, rule_logits: np.ndarray, model) -> np.ndarray:
    unique = score_attended_vectors(cache.vectors, cache.events, rule_logits, model)
    return unique[cache.inverse]


def metric_bundle(
    metadata,
    indices,
    label_column,
    group_column,
    prediction,
    ambiguous_rules,
    rule_column,
) -> dict:
    overall, diagnostic = evaluate_prediction_bundle(
        metadata,
        indices,
        label_column,
        group_column,
        prediction,
        ambiguous_rules,
        rule_column,
    )
    return {"overall": overall, "ambiguous_rule": diagnostic}


def choose_low_workload_setting(reference_points: list[dict]) -> tuple[float, float]:
    """Choose only from DeepCASE validation workload, independent of RC-CR."""

    selected = min(
        reference_points,
        key=lambda item: (
            item["validation"]["overall"]["analyst_workload"],
            -item["validation"]["overall"]["triage_incident_recall"],
            item["eps"],
            item["threshold"],
        ),
    )
    return float(selected["eps"]), float(selected["threshold"])


def _point(points: list[dict], setting: tuple[float, float]) -> dict:
    for item in points:
        if (float(item["eps"]), float(item["threshold"])) == setting:
            return item
    raise KeyError(f"Missing operating point: {setting}")


def _selection_key(candidate: dict) -> tuple[float, float, float, float]:
    metrics = candidate["validation"]["low_workload"]["overall"]
    return (
        float(metrics["triage_incident_recall"]),
        float(metrics["incident_group_recall"]),
        float(metrics["triage_precision"]),
        -float(candidate["weight_decay"]),
    )


def run(args: argparse.Namespace) -> None:
    config_path = args.config.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    root = config_path.parent.parent
    if args.test_scenario not in config["design"]["design_scenarios"]:
        raise ValueError("RC-CR feasibility is restricted to registered design scenarios")
    cfg = config["rccr"]["feasibility"]
    ratio = int(config["rccr"]["imbalance_ratio"])
    seed = int(args.seed)
    query_iterations = int(cfg["query_iterations"])
    label_column = str(config["data"]["primary_label"])
    group_column = "event_incident_group"
    diagnostic_cfg = config["design"]["ambiguous_rule_diagnostic"]
    rule_column = str(diagnostic_cfg["rule_column"])

    data = load_prepared(
        root / config["data"]["canonical_parquet"],
        root / config["data"]["contexts_npz"],
    )
    fold = scenario_fold(config["design"]["scenarios"], args.test_scenario)
    indices = fold_indices(data.metadata, fold)
    encoded = encode_for_fold(data, indices["train"])
    revealed = reveal_training_labels(
        data.metadata,
        indices["train"],
        label_column,
        group_column,
        int(config["design"]["label_budget"]),
        ratio,
        seed,
    )
    ambiguous_rules = define_ambiguous_rules(
        data.metadata,
        indices["train"],
        label_column,
        rule_column=rule_column,
        lower_incident_rate=float(diagnostic_cfg["lower_incident_rate"]),
        upper_incident_rate=float(diagnostic_cfg["upper_incident_rate"]),
        minimum_per_class=int(diagnostic_cfg["minimum_per_class"]),
    )

    source_dir = (
        config_path.parent
        / "results"
        / args.checkpoint_namespace
        / "full"
        / label_column
        / args.test_scenario
        / f"ratio_{ratio}"
        / f"seed_{seed}"
    )
    checkpoint_path = source_dir / "deepcase.pt"
    source_json_path = source_dir / "deepcase.json"
    if not checkpoint_path.exists() or not source_json_path.exists():
        raise FileNotFoundError(f"Frozen DeepCASE source is incomplete: {source_dir}")
    source_record = json.loads(source_json_path.read_text(encoding="utf-8"))
    checkpoint_sha256 = sha256_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected = (encoded.input_size, seed, "deepcase")
    observed = (
        int(checkpoint["input_size"]),
        int(checkpoint["seed"]),
        str(checkpoint["method"]),
    )
    if observed != expected:
        raise ValueError(f"Checkpoint mismatch: expected {expected}, observed {observed}")
    set_reproducible_seed(seed)
    model = IncidentAlignedContextBuilder(
        input_size=encoded.input_size,
        hidden_size=int(checkpoint["hidden_size"]),
        max_length=int(checkpoint["max_length"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    labels_all = data.metadata[label_column].to_numpy(dtype=np.int8)
    compact_fit = compact_interpreter_fit_set(
        encoded.context,
        encoded.target,
        indices["train"],
        revealed,
        labels_all,
        max_multiplicity=int(config["interpreter"]["max_multiplicity_for_fit"]),
    )
    X_fit, y_fit, fit_scores, compact_stats = compact_fit
    cache_dir = (
        config_path.parent
        / "results"
        / args.result_namespace
        / "cache"
        / args.test_scenario
        / f"ratio_{ratio}"
        / f"seed_{seed}"
    )
    identity = {
        "test_scenario": args.test_scenario,
        "validation_scenario": fold.validation_scenario,
        "seed": seed,
        "ratio": ratio,
        "query_iterations": query_iterations,
        "checkpoint_sha256": checkpoint_sha256,
        "input_size": encoded.input_size,
    }
    fit_cache = cache_or_build(
        cache_dir,
        "fit",
        {**identity, "rows": int(len(X_fit)), "preserve_rows": True},
        lambda: build_attended_cache(
            model,
            X_fit,
            y_fit,
            None,
            features=encoded.input_size,
            query_iterations=query_iterations,
            preserve_rows=True,
        ),
        force=args.force_cache,
    )
    revealed_cache = cache_or_build(
        cache_dir,
        "revealed",
        {**identity, "rows": int(len(revealed.indices)), "preserve_rows": True},
        lambda: build_attended_cache(
            model,
            encoded.context,
            encoded.target,
            revealed.indices,
            features=encoded.input_size,
            query_iterations=query_iterations,
            preserve_rows=True,
        ),
        force=args.force_cache,
    )
    split_caches = {}
    for split in ("validation", "test"):
        split_caches[split] = cache_or_build(
            cache_dir,
            split,
            {
                **identity,
                "rows": int(len(indices[split])),
                "preserve_rows": False,
            },
            lambda split=split: build_attended_cache(
                model,
                encoded.context,
                encoded.target,
                indices[split],
                features=encoded.input_size,
                query_iterations=query_iterations,
                preserve_rows=False,
            ),
            force=args.force_cache,
        )

    reference_points: list[dict] = []
    reference_predictions: dict[tuple[float, float], dict[str, np.ndarray]] = {}
    for eps, threshold in operating_settings(config, "full"):
        emit(f"fit frozen Interpreter eps={eps} threshold={threshold}")
        interpreter = fit_cached_interpreter_allow_unlabeled(
            model,
            fit_cache,
            fit_scores,
            features=encoded.input_size,
            eps=eps,
            min_samples=int(config["interpreter"]["min_samples"]),
            threshold=threshold,
        )
        predictions = {
            split: predict_cached_interpreter(interpreter, split_caches[split])
            for split in ("validation", "test")
        }
        setting = (float(eps), float(threshold))
        reference_predictions[setting] = predictions
        reference_points.append(
            {
                "eps": float(eps),
                "threshold": float(threshold),
                **{
                    split: metric_bundle(
                        data.metadata,
                        indices[split],
                        label_column,
                        group_column,
                        predictions[split],
                        ambiguous_rules,
                        rule_column,
                    )
                    for split in ("validation", "test")
                },
            }
        )

    endpoints = {
        "low_workload": choose_low_workload_setting(reference_points),
        "deepcase_default": (
            float(config["design"]["primary_reference"]["eps"]),
            float(config["design"]["primary_reference"]["threshold"]),
        ),
    }
    emit(f"frozen endpoints: {endpoints}")
    event_ids = encoded.target[revealed.indices]
    revealed_labels = data.metadata.loc[revealed.indices, label_column].to_numpy(
        dtype=np.int8
    )
    revealed_groups = data.metadata.loc[revealed.indices, group_column].to_numpy(
        dtype=object
    )
    if not np.array_equal(revealed_cache.events, event_ids):
        raise AssertionError("Revealed cache lost row alignment")
    rule_logits = smoothed_rule_logits(
        event_ids,
        revealed_labels,
        encoded.input_size,
        alpha=float(cfg["rule_prior_alpha"]),
    )

    candidates: dict[str, dict] = {
        "rule_prior": {
            "objective": "rule_prior",
            "weight_decay": None,
            "model": None,
        }
    }
    for objective in cfg["objectives"]:
        for weight_decay in cfg["weight_decay_grid"]:
            name = f"{objective}_wd_{float(weight_decay):g}"
            emit(f"train {name}")
            fitted = train_linear_risk_model(
                revealed_cache.vectors,
                event_ids,
                revealed_labels,
                revealed_groups,
                rule_logits,
                objective=str(objective),
                seed=seed,
                epochs=int(cfg["epochs"]),
                steps_per_epoch=int(cfg["steps_per_epoch"]),
                batch_size=int(cfg["batch_size"]),
                learning_rate=float(cfg["learning_rate"]),
                weight_decay=float(weight_decay),
            )
            candidates[name] = {
                "objective": str(objective),
                "weight_decay": float(weight_decay),
                "model": fitted,
            }

    # Validation labels select weight decay inside each objective.  Test labels
    # are read only after these choices have been frozen.
    validation_scores = {
        name: expanded_scores(split_caches["validation"], rule_logits, item["model"])
        for name, item in candidates.items()
    }
    for name, item in candidates.items():
        item["validation"] = {}
        for endpoint_name, setting in endpoints.items():
            prediction = matched_workload_predictions(
                validation_scores[name],
                reference_predictions[setting]["validation"],
                seed=int(cfg["tie_break_seed"]),
            )
            item["validation"][endpoint_name] = metric_bundle(
                data.metadata,
                indices["validation"],
                label_column,
                group_column,
                prediction,
                ambiguous_rules,
                rule_column,
            )

    selected_names = {"rule_prior": "rule_prior"}
    for objective in cfg["objectives"]:
        objective_candidates = [
            (name, item)
            for name, item in candidates.items()
            if item["objective"] == objective
        ]
        selected_names[str(objective)] = max(
            objective_candidates, key=lambda pair: _selection_key(pair[1])
        )[0]
    emit(f"validation-selected candidates: {selected_names}")

    selected_results: dict[str, dict] = {}
    for family, name in selected_names.items():
        item = candidates[name]
        test_score = expanded_scores(split_caches["test"], rule_logits, item["model"])
        test_metrics = {}
        for endpoint_name, setting in endpoints.items():
            prediction = matched_workload_predictions(
                test_score,
                reference_predictions[setting]["test"],
                seed=int(cfg["tie_break_seed"]),
            )
            test_metrics[endpoint_name] = metric_bundle(
                data.metadata,
                indices["test"],
                label_column,
                group_column,
                prediction,
                ambiguous_rules,
                rule_column,
            )
        selected_results[family] = {
            "candidate_name": name,
            "objective": item["objective"],
            "weight_decay": item["weight_decay"],
            "model": (
                None if item["model"] is None else serialize_risk_model(item["model"])
            ),
            "validation": item["validation"],
            "test": test_metrics,
        }

    reference_by_endpoint = {
        endpoint_name: _point(reference_points, setting)
        for endpoint_name, setting in endpoints.items()
    }
    low_reference_recall = reference_by_endpoint["low_workload"]["test"]["overall"][
        "triage_incident_recall"
    ]
    feasibility = {}
    for family, item in selected_results.items():
        recall = item["test"]["low_workload"]["overall"]["triage_incident_recall"]
        feasibility[family] = {
            "test_recall_gain_vs_deepcase": float(recall - low_reference_recall),
            "passes_single_scenario_effect_threshold": bool(
                recall - low_reference_recall >= float(cfg["minimum_recall_gain"])
            ),
        }

    output_file = (
        config_path.parent
        / "results"
        / args.result_namespace
        / "design"
        / args.test_scenario
        / f"ratio_{ratio}"
        / f"seed_{seed}.json"
    )
    payload = {
        "status": "complete",
        "scope": "exploratory design-only; not confirmatory",
        "method_family": "RC-CR",
        "test_scenario": args.test_scenario,
        "validation_scenario": fold.validation_scenario,
        "train_scenarios": list(fold.train_scenarios),
        "seed": seed,
        "imbalance_ratio": ratio,
        "query_iterations": query_iterations,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "source_record_status": source_record.get("status"),
        "environment": package_versions(),
        "split_rows": {key: int(len(value)) for key, value in indices.items()},
        "revealed_labels": {
            "positive": int(len(revealed.positive_indices)),
            "negative": int(len(revealed.negative_indices)),
            "positive_groups_covered": revealed.positive_groups_covered,
            "positive_groups_available": revealed.positive_groups_available,
        },
        "compact_interpreter_fit": compact_stats,
        "ambiguous_rule_definition": asdict(ambiguous_rules),
        "config": cfg,
        "endpoints": {
            name: {"eps": setting[0], "threshold": setting[1]}
            for name, setting in endpoints.items()
        },
        "deepcase_reference_points": reference_points,
        "deepcase_reference_by_endpoint": reference_by_endpoint,
        "selected_models": selected_results,
        "feasibility": feasibility,
    }
    atomic_json(output_file, payload)
    emit(f"complete: {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.yaml")
    )
    parser.add_argument(
        "--test-scenario", choices=["russellmitchell", "fox"], required=True
    )
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--checkpoint-namespace", default="rescue_q100_v1")
    parser.add_argument("--result-namespace", default="rccr_feasibility_v1")
    parser.add_argument("--force-cache", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
