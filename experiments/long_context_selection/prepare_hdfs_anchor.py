"""Fit training-only HDFS association evidence and build anchor5_assoc5."""

from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_association_selectors import save_model
from prepare_context_variants import event_ids_from_candidate_rows
from src.association import (
    fit_association_model,
    recency_bands,
    select_recency_anchored_association,
)
from src.common import EXPERIMENT, sha256, write_json


CONFIG_PATH = EXPERIMENT / "external_config_v2.yaml"
ARTIFACTS = EXPERIMENT / "hdfs_artifacts"


def main() -> None:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    manifest_path = ARTIFACTS / "manifest.json"
    target = np.load(ARTIFACTS / "target_events.npy", mmap_mode="r")
    candidates = np.load(ARTIFACTS / "candidate_events_100.npy", mmap_mode="r")
    labels = np.load(ARTIFACTS / "labels_blind.npy", mmap_mode="r")
    splits = np.load(ARTIFACTS / "split_codes.npy", mmap_mode="r")
    train = np.flatnonzero(splits <= 1)
    vocabulary_size = int(target.max()) + 1
    identity_events = np.arange(vocabulary_size, dtype=np.int16)
    teacher = config["association_teacher"]
    bands = recency_bands(candidates.shape[1], teacher["recency_rank_bands"])
    started = time.perf_counter()
    model = fit_association_model(
        candidates,
        identity_events,
        np.asarray(target),
        np.asarray(labels),
        train,
        global_event_count=vocabulary_size,
        bands=bands,
        smoothing=float(teacher["bernoulli_smoothing"]),
        maximum_absolute_log_odds=float(
            teacher["maximum_absolute_log_odds"]
        ),
        batch_rows=int(teacher["fit_batch_rows"]),
    )
    fit_seconds = time.perf_counter() - started
    model_path = ARTIFACTS / "association_model.npz"
    save_model(model_path, model)
    output_path = ARTIFACTS / "contexts_anchor5_assoc5.npy"
    budget = int(config["method"]["context_positions"])
    anchors = int(config["method"]["unconditional_newest_anchors"])
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.int8, shape=(len(target), budget)
    )
    started = time.perf_counter()
    batch_rows = int(teacher["selection_batch_rows"])
    pad_id = vocabulary_size
    for start in range(0, len(target), batch_rows):
        stop = min(start + batch_rows, len(target))
        selected = select_recency_anchored_association(
            model,
            np.asarray(candidates[start:stop]),
            identity_events,
            np.asarray(target[start:stop]),
            budget=budget,
            recency_anchors=anchors,
        )
        output[start:stop] = event_ids_from_candidate_rows(
            selected, identity_events, pad_id
        )
    output.flush()
    receipt = {
        "status": "complete_blind_to_final_labels",
        "method": "anchor5_assoc5",
        "fit_rows": int(len(train)),
        "fit_split_codes": [0, 1],
        "final_labels_masked_to": -9,
        "final_labels_used": False,
        "retrieval_labels_used": False,
        "context_positions": budget,
        "recency_anchors": anchors,
        "association_positions": budget - anchors,
        "fit_seconds": fit_seconds,
        "selection_seconds": time.perf_counter() - started,
        "manifest_sha256": sha256(manifest_path),
        "config_sha256": sha256(CONFIG_PATH),
        "model_sha256": sha256(model_path),
        "output_sha256": sha256(output_path),
    }
    write_json(ARTIFACTS / "anchor5_assoc5.json", receipt)
    print(receipt)


if __name__ == "__main__":
    main()
