"""Materialize auditable event-ID contexts for the Gate-B upper bounds."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.common import EXPERIMENT, load_config, resolve_repo, sha256, write_json


def event_ids_from_candidate_rows(
    candidate_rows: np.ndarray,
    event_ids: np.ndarray,
    pad_id: int,
) -> np.ndarray:
    """Map candidate row IDs to event IDs; -1 is a dedicated left-pad token."""

    if candidate_rows.ndim != 2:
        raise ValueError("candidate_rows must be a two-dimensional matrix")
    valid = candidate_rows >= 0
    safe_rows = np.where(valid, candidate_rows, 0)
    result = event_ids[safe_rows].astype(np.uint8, copy=False)
    result[~valid] = pad_id
    return result


def oracle_positions(
    candidate_rows: np.ndarray,
    candidate_incident: np.ndarray,
    budget: int,
) -> np.ndarray:
    """Choose incident predecessors first, then recency, without target labels.

    Returned column positions are chronological.  Because source padding is on
    the left, rows with fewer than ``budget`` candidates remain left padded.
    """

    if candidate_rows.shape != candidate_incident.shape:
        raise ValueError("candidate rows and labels must have identical shapes")
    if not 0 < budget <= candidate_rows.shape[1]:
        raise ValueError("invalid oracle budget")
    width = candidate_rows.shape[1]
    position = np.arange(width, dtype=np.int16)[None, :]
    valid = candidate_rows >= 0
    # Priority is lexicographic: incident status first, then larger (more
    # recent) source position. Invalid left padding always ranks last.
    priority = np.where(
        valid,
        candidate_incident.astype(np.int16) * (width + 1) + position,
        -1,
    )
    selected = np.argpartition(priority, width - budget, axis=1)[:, -budget:]
    selected.sort(axis=1)
    return selected


def oracle_context(
    candidate_rows: np.ndarray,
    event_ids: np.ndarray,
    incident_labels: np.ndarray,
    pad_id: int,
    budget: int = 10,
) -> np.ndarray:
    """Build the non-deployable predecessor-label oracle context."""

    valid = candidate_rows >= 0
    safe_rows = np.where(valid, candidate_rows, 0)
    predecessor_incident = incident_labels[safe_rows].astype(bool, copy=False)
    predecessor_incident &= valid
    positions = oracle_positions(candidate_rows, predecessor_incident, budget)
    selected_rows = np.take_along_axis(candidate_rows, positions, axis=1)
    return event_ids_from_candidate_rows(selected_rows, event_ids, pad_id)


def build_variants(
    candidates_path: Path,
    output_dir: Path,
    event_ids: np.ndarray,
    incident_labels: np.ndarray,
    pad_id: int,
    *,
    budget: int,
    chunk_rows: int,
) -> dict:
    candidates = np.load(candidates_path, mmap_mode="r")
    if len(candidates) != len(event_ids) or len(event_ids) != len(incident_labels):
        raise ValueError("candidate, event and label row counts differ")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "last10": output_dir / "contexts_last10.npy",
        "last100": output_dir / "contexts_last100.npy",
        "oracle10": output_dir / "contexts_oracle10.npy",
    }
    outputs = {
        "last10": np.lib.format.open_memmap(
            paths["last10"], mode="w+", dtype=np.uint8, shape=(len(event_ids), budget)
        ),
        "last100": np.lib.format.open_memmap(
            paths["last100"], mode="w+", dtype=np.uint8, shape=candidates.shape
        ),
        "oracle10": np.lib.format.open_memmap(
            paths["oracle10"], mode="w+", dtype=np.uint8, shape=(len(event_ids), budget)
        ),
    }
    started = time.perf_counter()
    for start in range(0, len(event_ids), chunk_rows):
        stop = min(start + chunk_rows, len(event_ids))
        block = np.asarray(candidates[start:stop])
        outputs["last100"][start:stop] = event_ids_from_candidate_rows(
            block, event_ids, pad_id
        )
        outputs["last10"][start:stop] = event_ids_from_candidate_rows(
            block[:, -budget:], event_ids, pad_id
        )
        outputs["oracle10"][start:stop] = oracle_context(
            block, event_ids, incident_labels, pad_id, budget
        )
    for output in outputs.values():
        output.flush()
    elapsed = time.perf_counter() - started
    return {
        "rows": int(len(event_ids)),
        "pad_id": int(pad_id),
        "budget": int(budget),
        "candidate_width": int(candidates.shape[1]),
        "oracle_uses_target_label": False,
        "oracle_uses_predecessor_event_incident": True,
        "elapsed_seconds": elapsed,
        "files": {
            name: {
                "path": str(path),
                "shape": list(outputs[name].shape),
                "dtype": str(outputs[name].dtype),
                "sha256": sha256(path),
            }
            for name, path in paths.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--chunk-rows", type=int, default=50_000)
    args = parser.parse_args()
    config = load_config()
    output_dir = EXPERIMENT / "artifacts"
    receipt = output_dir / "context_variants.json"
    expected = [
        output_dir / "contexts_last10.npy",
        output_dir / "contexts_last100.npy",
        output_dir / "contexts_oracle10.npy",
    ]
    if receipt.exists() and all(path.exists() for path in expected) and not args.force:
        print(f"exists: {receipt}")
        return
    frame = pd.read_parquet(
        resolve_repo(config["data"]["canonical_parquet"]),
        columns=["event_id", "event_incident"],
    )
    event_ids = frame["event_id"].to_numpy(dtype=np.int16, copy=True)
    labels = frame["event_incident"].to_numpy(dtype=np.int8, copy=True)
    pad_id = int(event_ids.max()) + 1
    if pad_id > np.iinfo(np.uint8).max:
        raise ValueError("event vocabulary does not fit uint8")
    metadata = build_variants(
        resolve_repo(config["data"]["candidate_rows"]),
        output_dir,
        event_ids,
        labels,
        pad_id,
        budget=int(config["data"]["selector_budget"]),
        chunk_rows=args.chunk_rows,
    )
    metadata["candidate_sha256"] = sha256(
        resolve_repo(config["data"]["candidate_rows"])
    )
    write_json(receipt, metadata)
    print(metadata)


if __name__ == "__main__":
    main()
