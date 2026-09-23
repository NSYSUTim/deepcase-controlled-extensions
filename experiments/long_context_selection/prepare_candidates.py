"""Build memory-mapped strict-past candidate row indices for AIT-ADS."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.common import load_config, resolve_repo, sha256, write_json


def build_candidates(frame: pd.DataFrame, output: Path, cap: int, timeout: int) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    candidates = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.int32, shape=(len(frame), cap)
    )
    candidates[:] = -1
    timestamp = frame["timestamp"].to_numpy(dtype=np.int64, copy=False)
    machines = frame["machine"].astype(str)
    started = time.perf_counter()

    grouped = frame.groupby(machines, sort=False, observed=True).indices
    for group_number, (_, raw_indices) in enumerate(grouped.items(), start=1):
        indices = np.asarray(raw_indices, dtype=np.int64)
        order = np.argsort(timestamp[indices], kind="stable")
        indices = indices[order]
        times = timestamp[indices]
        history: deque[tuple[int, int]] = deque(maxlen=cap)
        start = 0
        while start < len(indices):
            current = int(times[start])
            stop = start + 1
            while stop < len(indices) and int(times[stop]) == current:
                stop += 1
            while history and current - history[0][0] > timeout:
                history.popleft()
            if history:
                prior = np.fromiter((row for _, row in history), dtype=np.int32)
                candidates[indices[start:stop], -len(prior) :] = prior
            for position in range(start, stop):
                history.append((current, int(indices[position])))
            start = stop
        if group_number % 20 == 0:
            candidates.flush()
    candidates.flush()
    elapsed = time.perf_counter() - started
    return {
        "rows": int(len(frame)),
        "candidate_cap": int(cap),
        "timeout_seconds": int(timeout),
        "machine_groups": int(len(grouped)),
        "dtype": "int32",
        "shape": [int(len(frame)), int(cap)],
        "ties": "strictly earlier timestamps only",
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config()
    source = resolve_repo(config["data"]["canonical_parquet"])
    output = resolve_repo(config["data"]["candidate_rows"])
    receipt = output.with_suffix(".json")
    if output.exists() and receipt.exists() and not args.force:
        print(f"exists: {output}")
        return
    frame = pd.read_parquet(source, columns=["timestamp", "machine"])
    cap = max(config["data"]["candidate_caps"])
    metadata = build_candidates(
        frame, output, cap=cap, timeout=int(config["data"]["timeout_seconds"])
    )
    metadata["source"] = str(source)
    metadata["source_sha256"] = sha256(source)
    metadata["candidate_sha256"] = sha256(output)
    write_json(receipt, metadata)
    print(metadata)


if __name__ == "__main__":
    main()
