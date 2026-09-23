"""Parse the authors-linked HDFS files into strict sequence-boundary arrays."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.common import EXPERIMENT, ROOT, sha256, write_json
from src.hdfs_external import (
    SOURCE_LABEL,
    line_bucket,
    normalized_line,
    parse_event_line,
    preceding_windows,
    split_code,
)


CONFIG_PATH = EXPERIMENT / "external_config_v2.yaml"
RAW_DIR = ROOT / "dataset" / "raw" / "hdfs_deeplog"
OUTPUT_DIR = EXPERIMENT / "hdfs_artifacts"
SOURCES = ("hdfs_train", "hdfs_test_normal", "hdfs_test_abnormal")


def iter_nonempty_lines(path: Path):
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not normalized_line(raw).strip():
                continue
            yield line_number, raw


def scan_inputs(paths: dict[str, Path]) -> dict:
    tokens: set[int] = set()
    total_events = 0
    total_sequences = 0
    sequence_counts = {source: 0 for source in SOURCES}
    event_counts = {source: 0 for source in SOURCES}
    split_sequences = {str(code): 0 for code in range(4)}
    split_events = {str(code): 0 for code in range(4)}
    train_hashes: set[bytes] = set()
    test_hash_labels: dict[bytes, set[int]] = {}
    test_hash_counts: dict[bytes, int] = {}

    for source in SOURCES:
        for _, raw in iter_nonempty_lines(paths[source]):
            values = parse_event_line(raw)
            count = int(len(values))
            tokens.update(int(value) for value in np.unique(values))
            code = split_code(source, raw)
            digest = hashlib.sha256(normalized_line(raw)).digest()
            if source == "hdfs_train":
                train_hashes.add(digest)
            else:
                test_hash_labels.setdefault(digest, set()).add(
                    SOURCE_LABEL[source]
                )
                test_hash_counts[digest] = test_hash_counts.get(digest, 0) + 1
            total_events += count
            total_sequences += 1
            sequence_counts[source] += 1
            event_counts[source] += count
            split_sequences[str(code)] += 1
            split_events[str(code)] += count

    overlap = train_hashes.intersection(test_hash_labels)
    contradictory = sum(len(labels) > 1 for labels in test_hash_labels.values())
    duplicate_test_patterns = sum(count > 1 for count in test_hash_counts.values())
    return {
        "event_tokens": sorted(tokens),
        "total_events": total_events,
        "total_sequences": total_sequences,
        "sequence_counts_by_source": sequence_counts,
        "event_counts_by_source": event_counts,
        "sequence_counts_by_split": split_sequences,
        "event_counts_by_split": split_events,
        "official_train_pattern_overlap_count": len(overlap),
        "official_test_contradictory_pattern_count": contradictory,
        "official_test_duplicate_pattern_count": duplicate_test_patterns,
    }


def materialize(paths: dict[str, Path], audit: dict, force: bool) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "target": OUTPUT_DIR / "target_events.npy",
        "candidates": OUTPUT_DIR / "candidate_events_100.npy",
        "last10": OUTPUT_DIR / "contexts_last10.npy",
        "labels": OUTPUT_DIR / "labels.npy",
        "labels_blind": OUTPUT_DIR / "labels_blind.npy",
        "split": OUTPUT_DIR / "split_codes.npy",
        "sequence_id": OUTPUT_DIR / "sequence_ids.npy",
    }
    if not force and all(path.exists() for path in outputs.values()):
        raise FileExistsError("HDFS artifacts already exist; use --force deliberately")
    tokens = audit["event_tokens"]
    if len(tokens) + 1 > 127:
        raise ValueError("HDFS vocabulary plus padding does not fit signed int8")
    mapping = {token: index for index, token in enumerate(tokens)}
    pad_id = len(tokens)
    rows = int(audit["total_events"])
    target = np.lib.format.open_memmap(
        outputs["target"], mode="w+", dtype=np.int8, shape=(rows,)
    )
    candidates = np.lib.format.open_memmap(
        outputs["candidates"], mode="w+", dtype=np.int8, shape=(rows, 100)
    )
    last10 = np.lib.format.open_memmap(
        outputs["last10"], mode="w+", dtype=np.int8, shape=(rows, 10)
    )
    labels = np.lib.format.open_memmap(
        outputs["labels"], mode="w+", dtype=np.int8, shape=(rows,)
    )
    labels_blind = np.lib.format.open_memmap(
        outputs["labels_blind"], mode="w+", dtype=np.int8, shape=(rows,)
    )
    splits = np.lib.format.open_memmap(
        outputs["split"], mode="w+", dtype=np.uint8, shape=(rows,)
    )
    sequence_ids = np.lib.format.open_memmap(
        outputs["sequence_id"], mode="w+", dtype=np.int32, shape=(rows,)
    )
    sequence_start = np.empty(audit["total_sequences"], dtype=np.int64)
    sequence_length = np.empty(audit["total_sequences"], dtype=np.int32)
    sequence_label = np.empty(audit["total_sequences"], dtype=np.int8)
    sequence_split = np.empty(audit["total_sequences"], dtype=np.uint8)
    sequence_source = np.empty(audit["total_sequences"], dtype=np.uint8)
    sequence_bucket = np.full(audit["total_sequences"], 255, dtype=np.uint8)
    sequence_train_overlap = np.zeros(audit["total_sequences"], dtype=bool)
    train_hashes = {
        hashlib.sha256(normalized_line(raw)).digest()
        for _, raw in iter_nonempty_lines(paths["hdfs_train"])
    }

    row = 0
    sequence = 0
    started = time.perf_counter()
    for source_number, source in enumerate(SOURCES):
        for _, raw in iter_nonempty_lines(paths[source]):
            original = parse_event_line(raw)
            dense = np.fromiter(
                (mapping[int(value)] for value in original),
                dtype=np.int8,
                count=len(original),
            )
            stop = row + len(dense)
            windows = preceding_windows(dense, width=100, pad_id=-1)
            code = split_code(source, raw)
            label = SOURCE_LABEL[source]
            target[row:stop] = dense
            candidates[row:stop] = windows
            last = windows[:, -10:].copy()
            last[last < 0] = pad_id
            last10[row:stop] = last
            labels[row:stop] = label
            labels_blind[row:stop] = -9 if code == 3 else label
            splits[row:stop] = code
            sequence_ids[row:stop] = sequence
            sequence_start[sequence] = row
            sequence_length[sequence] = len(dense)
            sequence_label[sequence] = label
            sequence_split[sequence] = code
            sequence_source[sequence] = source_number
            if source != "hdfs_train":
                sequence_bucket[sequence] = line_bucket(raw)
                sequence_train_overlap[sequence] = (
                    hashlib.sha256(normalized_line(raw)).digest() in train_hashes
                )
            row = stop
            sequence += 1
            if sequence % 100_000 == 0:
                print(f"materialized sequences={sequence} rows={row}", flush=True)
    if row != rows or sequence != audit["total_sequences"]:
        raise AssertionError("materialized counts differ from first pass")
    for array in (
        target,
        candidates,
        last10,
        labels,
        labels_blind,
        splits,
        sequence_ids,
    ):
        array.flush()
    sequence_path = OUTPUT_DIR / "sequences.npz"
    np.savez_compressed(
        sequence_path,
        start=sequence_start,
        length=sequence_length,
        label=sequence_label,
        split=sequence_split,
        source=sequence_source,
        bucket=sequence_bucket,
        train_pattern_overlap=sequence_train_overlap,
    )
    return {
        "elapsed_seconds": time.perf_counter() - started,
        "dense_event_mapping": {str(token): index for token, index in mapping.items()},
        "event_vocabulary_size": len(tokens),
        "pad_id": pad_id,
        "files": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in {**outputs, "sequences": sequence_path}.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    paths = {source: RAW_DIR / source for source in SOURCES}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"download the frozen HDFS inputs first: {missing}")
    started = time.perf_counter()
    audit = scan_inputs(paths)
    built = materialize(paths, audit, args.force)
    receipt = {
        "status": "complete",
        "source_urls": config["hdfs"]["files"],
        "source_files": {
            source: {"path": str(path), "sha256": sha256(path)}
            for source, path in paths.items()
        },
        "partition": config["hdfs"]["duplicate_safe_partition"],
        "scan": audit,
        "materialized": built,
        "total_seconds": time.perf_counter() - started,
        "external_config_sha256": sha256(CONFIG_PATH),
    }
    write_json(OUTPUT_DIR / "manifest.json", receipt)
    print(audit)


if __name__ == "__main__":
    main()
