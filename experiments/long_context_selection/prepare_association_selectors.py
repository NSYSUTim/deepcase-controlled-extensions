"""Fit the long-pool teacher and materialize fold-specific 10-slot selectors."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_experiments.src.data import fold_indices, scenario_fold
from src.association import (
    association_scores,
    fit_association_model,
    probability_metrics,
    recency_bands,
    select_by_association,
)
from src.common import EXPERIMENT, load_config, resolve_repo, sha256, write_json
from prepare_context_variants import event_ids_from_candidate_rows


def save_model(path: Path, model) -> None:
    np.savez_compressed(
        path,
        weights=model.weights,
        base_log_odds=model.base_log_odds,
        global_to_local=model.global_to_local,
        seen_global_events=model.seen_global_events,
        unknown_id=np.asarray(model.unknown_id),
        bands=model.bands,
        smoothing=np.asarray(model.smoothing),
        maximum_absolute_log_odds=np.asarray(model.maximum_absolute_log_odds),
    )


def materialize_selectors(
    model,
    candidate_rows: np.ndarray,
    event_ids: np.ndarray,
    target: np.ndarray,
    labels: np.ndarray,
    output_abs: Path,
    output_oracle: Path,
    *,
    pad_id: int,
    budget: int,
    batch_rows: int,
) -> float:
    outputs = {
        "association_abs10": np.lib.format.open_memmap(
            output_abs, mode="w+", dtype=np.uint8, shape=(len(target), budget)
        ),
        "class_oracle10": np.lib.format.open_memmap(
            output_oracle, mode="w+", dtype=np.uint8, shape=(len(target), budget)
        ),
    }
    started = time.perf_counter()
    for start in range(0, len(target), batch_rows):
        stop = min(start + batch_rows, len(target))
        block = np.asarray(candidate_rows[start:stop])
        target_block = target[start:stop]
        selected_abs = select_by_association(
            model, block, event_ids, target_block, budget=budget
        )
        selected_oracle = select_by_association(
            model,
            block,
            event_ids,
            target_block,
            budget=budget,
            labels_for_oracle=labels[start:stop],
        )
        outputs["association_abs10"][start:stop] = event_ids_from_candidate_rows(
            selected_abs, event_ids, pad_id
        )
        outputs["class_oracle10"][start:stop] = event_ids_from_candidate_rows(
            selected_oracle, event_ids, pad_id
        )
    for output in outputs.values():
        output.flush()
    return time.perf_counter() - started


def run_scenario(scenario: str, force: bool) -> None:
    config = load_config()
    receipt = EXPERIMENT / "artifacts" / f"association_{scenario}.json"
    output_abs = EXPERIMENT / "artifacts" / f"contexts_association_abs10_{scenario}.npy"
    output_oracle = EXPERIMENT / "artifacts" / f"contexts_class_oracle10_{scenario}.npy"
    if receipt.exists() and output_abs.exists() and output_oracle.exists() and not force:
        print(f"exists: {receipt}")
        return
    frame = pd.read_parquet(
        resolve_repo(config["data"]["canonical_parquet"]),
        columns=["row_id", "scenario", "event_id", "event_incident"],
    )
    candidates = np.load(resolve_repo(config["data"]["candidate_rows"]), mmap_mode="r")
    event_ids = frame["event_id"].to_numpy(dtype=np.int16, copy=True)
    labels = frame["event_incident"].to_numpy(dtype=np.int8, copy=True)
    fold = scenario_fold(config["design"]["scenarios"], scenario)
    split = fold_indices(frame, fold)
    teacher_cfg = config["association_teacher"]
    bands100 = recency_bands(candidates.shape[1], teacher_cfg["recency_rank_bands"])
    started = time.perf_counter()
    model100 = fit_association_model(
        candidates,
        event_ids,
        event_ids,
        labels,
        split["train"],
        global_event_count=int(event_ids.max()) + 1,
        bands=bands100,
        smoothing=float(teacher_cfg["bernoulli_smoothing"]),
        maximum_absolute_log_odds=float(teacher_cfg["maximum_absolute_log_odds"]),
        batch_rows=int(teacher_cfg["fit_batch_rows"]),
    )
    fit100_seconds = time.perf_counter() - started
    model_path = EXPERIMENT / "artifacts" / f"association_model_{scenario}.npz"
    save_model(model_path, model100)

    # Fit the matched Last10 teacher independently; it cannot learn anything
    # from positions 11--100.
    started = time.perf_counter()
    model10 = fit_association_model(
        candidates[:, -10:],
        event_ids,
        event_ids,
        labels,
        split["train"],
        global_event_count=int(event_ids.max()) + 1,
        bands=np.zeros(10, dtype=np.int8),
        smoothing=float(teacher_cfg["bernoulli_smoothing"]),
        maximum_absolute_log_odds=float(teacher_cfg["maximum_absolute_log_odds"]),
        batch_rows=int(teacher_cfg["fit_batch_rows"]),
    )
    fit10_seconds = time.perf_counter() - started
    metrics = {}
    for split_name in ("validation", "test"):
        rows = split[split_name]
        score10 = association_scores(
            model10,
            candidates[:, -10:],
            event_ids,
            event_ids,
            rows,
            batch_rows=int(teacher_cfg["fit_batch_rows"]),
        )
        score100 = association_scores(
            model100,
            candidates,
            event_ids,
            event_ids,
            rows,
            batch_rows=int(teacher_cfg["fit_batch_rows"]),
        )
        metrics[split_name] = {
            "last10": probability_metrics(labels[rows], score10),
            "pool100": probability_metrics(labels[rows], score100),
        }
        metrics[split_name]["delta_pool100_minus_last10"] = {
            key: metrics[split_name]["pool100"][key] - metrics[split_name]["last10"][key]
            for key in ("roc_auc", "average_precision", "log_loss", "brier_score")
        }

    selection_seconds = materialize_selectors(
        model100,
        candidates,
        event_ids,
        event_ids,
        labels,
        output_abs,
        output_oracle,
        pad_id=int(event_ids.max()) + 1,
        budget=int(config["data"]["selector_budget"]),
        batch_rows=int(teacher_cfg["selection_batch_rows"]),
    )
    write_json(receipt, {
        "test_scenario": scenario,
        "validation_scenario": fold.validation_scenario,
        "train_scenarios": list(fold.train_scenarios),
        "test_labels_used_to_fit_teacher": False,
        "association_abs10_uses_test_labels": False,
        "class_oracle10_uses_target_labels": True,
        "class_oracle10_deployable": False,
        "teacher_metrics": metrics,
        "fit100_seconds": fit100_seconds,
        "fit10_seconds": fit10_seconds,
        "selection_seconds": selection_seconds,
        "model_path": str(model_path),
        "model_sha256": sha256(model_path),
        "outputs": {
            "association_abs10": {"path": str(output_abs), "sha256": sha256(output_abs)},
            "class_oracle10": {"path": str(output_oracle), "sha256": sha256(output_oracle)},
        },
    })
    print(metrics["test"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-scenario", required=True, choices=[
        "fox", "harrison", "russellmitchell", "santos", "shaw", "wardbeck", "wheeler", "wilson"
    ])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_scenario(args.test_scenario, args.force)


if __name__ == "__main__":
    main()
