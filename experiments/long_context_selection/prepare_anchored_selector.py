"""Materialize the fixed post-failure anchor5/association5 selector on AIT-ADS."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_context_variants import event_ids_from_candidate_rows
from src.association import AssociationModel, select_recency_anchored_association
from src.common import EXPERIMENT, load_config, resolve_repo, sha256, write_json


METHOD = "anchor5_assoc5"
ANCHORS = 5
BUDGET = 10


def load_model(path: Path) -> AssociationModel:
    with np.load(path) as data:
        return AssociationModel(
            weights=data["weights"],
            base_log_odds=data["base_log_odds"],
            global_to_local=data["global_to_local"],
            seen_global_events=data["seen_global_events"],
            unknown_id=int(data["unknown_id"]),
            bands=data["bands"],
            smoothing=float(data["smoothing"]),
            maximum_absolute_log_odds=float(data["maximum_absolute_log_odds"]),
        )


def run_scenario(scenario: str, force: bool, batch_rows: int) -> None:
    config = load_config()
    model_path = EXPERIMENT / "artifacts" / f"association_model_{scenario}.npz"
    association_receipt = (
        EXPERIMENT / "artifacts" / f"association_{scenario}.json"
    )
    output = EXPERIMENT / "artifacts" / f"contexts_{METHOD}_{scenario}.npy"
    receipt = EXPERIMENT / "artifacts" / f"{METHOD}_{scenario}.json"
    if output.exists() and receipt.exists() and not force:
        print(f"exists: {receipt}")
        return
    frame = pd.read_parquet(
        resolve_repo(config["data"]["canonical_parquet"]),
        columns=["event_id"],
    )
    event_ids = frame["event_id"].to_numpy(dtype=np.int16, copy=True)
    candidates = np.load(
        resolve_repo(config["data"]["candidate_rows"]), mmap_mode="r"
    )
    model = load_model(model_path)
    pad_id = int(event_ids.max()) + 1
    contexts = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.uint8, shape=(len(event_ids), BUDGET)
    )
    started = time.perf_counter()
    for start in range(0, len(event_ids), batch_rows):
        stop = min(start + batch_rows, len(event_ids))
        rows = select_recency_anchored_association(
            model,
            np.asarray(candidates[start:stop]),
            event_ids,
            event_ids[start:stop],
            budget=BUDGET,
            recency_anchors=ANCHORS,
        )
        contexts[start:stop] = event_ids_from_candidate_rows(
            rows, event_ids, pad_id
        )
    contexts.flush()
    write_json(
        receipt,
        {
            "status": "post_failure_exploratory_method_development",
            "test_scenario": scenario,
            "method": METHOD,
            "candidate_pool": int(candidates.shape[1]),
            "context_positions": BUDGET,
            "unconditional_recency_anchors": ANCHORS,
            "association_selected_older_positions": BUDGET - ANCHORS,
            "ranking": "absolute_training_only_association_then_recency",
            "uses_test_labels": False,
            "uses_target_or_predecessor_labels_at_retrieval": False,
            "association_model_sha256": sha256(model_path),
            "association_receipt_sha256": sha256(association_receipt),
            "candidate_rows_sha256": sha256(
                resolve_repo(config["data"]["candidate_rows"])
            ),
            "output_sha256": sha256(output),
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    print(f"wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-scenario",
        required=True,
        choices=[
            "fox",
            "harrison",
            "russellmitchell",
            "santos",
            "shaw",
            "wardbeck",
            "wheeler",
            "wilson",
        ],
    )
    parser.add_argument("--batch-rows", type=int, default=50_000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_scenario(args.test_scenario, args.force, args.batch_rows)


if __name__ == "__main__":
    main()
