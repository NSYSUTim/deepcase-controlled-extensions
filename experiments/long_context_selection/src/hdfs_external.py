"""Pure helpers for the duplicate-safe HDFS external protocol."""

from __future__ import annotations

import hashlib

import numpy as np


SOURCE_LABEL = {
    "hdfs_train": 0,
    "hdfs_test_normal": 0,
    "hdfs_test_abnormal": 1,
}


def normalized_line(raw_line: bytes) -> bytes:
    """Remove only the record delimiter; other bytes define sequence identity."""

    return raw_line.rstrip(b"\r\n")


def line_bucket(line: bytes) -> int:
    digest = hashlib.sha256(normalized_line(line)).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % 10


def split_code(source: str, line: bytes) -> int:
    """0=official train, 1=development, 2=validation, 3=final test."""

    if source == "hdfs_train":
        return 0
    bucket = line_bucket(line)
    if bucket <= 3:
        return 1
    if bucket <= 5:
        return 2
    return 3


def parse_event_line(raw_line: bytes) -> np.ndarray:
    line = normalized_line(raw_line)
    if not line.strip():
        raise ValueError("empty HDFS sequence")
    try:
        text = line.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError("HDFS event sequence is not ASCII") from error
    values = np.fromstring(text, sep=" ", dtype=np.int32)
    if len(values) != len(line.split()):
        raise ValueError("non-integer token in HDFS sequence")
    if len(values) == 0 or np.any(values < 0):
        raise ValueError("HDFS sequence needs non-negative integer events")
    return values


def preceding_windows(events: np.ndarray, width: int, pad_id: int) -> np.ndarray:
    """Return strict-predecessor windows without crossing a sequence boundary."""

    events = np.asarray(events)
    if events.ndim != 1 or width <= 0:
        raise ValueError("events must be one-dimensional and width positive")
    padded = np.concatenate(
        [np.full(width, pad_id, dtype=events.dtype), events]
    )
    return np.lib.stride_tricks.sliding_window_view(padded, width)[: len(events)]


def safety_limit_per_1000(
    baseline_rate: float,
    *,
    switch_rate: float = 1.0,
    absolute_margin: float = 1.0,
    relative_margin: float = 0.05,
) -> float:
    if baseline_rate < 0:
        raise ValueError("rate must be non-negative")
    if baseline_rate < switch_rate:
        return baseline_rate + absolute_margin
    return baseline_rate * (1.0 + relative_margin)
