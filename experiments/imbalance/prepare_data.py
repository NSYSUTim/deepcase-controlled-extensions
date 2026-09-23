"""Build a canonical, leakage-resistant AIT-ADS table and DeepCASE contexts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


EXPECTED_SCENARIO_ROWS = {
    "fox": 473_104,
    "harrison": 593_948,
    "russellmitchell": 45_544,
    "santos": 130_779,
    "shaw": 70_782,
    "wardbeck": 91_257,
    "wheeler": 616_161,
    "wilson": 634_246,
}
EXPECTED_TOTAL_ROWS = 2_655_821

EVENT_GROUP_MAP = {
    "dns_scan": "network_scans",
    "service_scan": "service_scans",
    "wpscan": "wpscan",
    "dirb": "dirb",
    "webshell_cmd": "webshell",
    "crack_passwords": "cracking",
    "online_cracking": "cracking",
    "attacker_change_user": "privilege_escalation",
    "escalated_sudo_command": "privilege_escalation",
    "dnsteal": "dnsteal",
}


def sha256(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def detector_from_short(values: pd.Series) -> pd.Series:
    prefix = values.str.slice(0, 1)
    return prefix.map({"A": "AMiner", "S": "Suricata", "W": "Wazuh"})


def build_strict_past_contexts(
    frame: pd.DataFrame,
    event_count: int,
    length: int,
    timeout: int,
) -> np.ndarray:
    """Return last-N contexts using only timestamps strictly before the target.

    The source CSV writes timestamps at one-second resolution. Processing all
    rows tied at time t before appending any of them prevents arbitrary CSV
    ordering from becoming an artificial causal signal.
    """

    pad_id = event_count
    contexts = np.full((len(frame), length), pad_id, dtype=np.int16)
    machine_values = frame["machine"].astype(str).to_numpy()
    timestamp_values = frame["timestamp"].to_numpy(dtype=np.int64)
    event_values = frame["event_id"].to_numpy(dtype=np.int16)

    groups = frame.groupby("machine", sort=False, observed=True).indices
    for machine, raw_indices in groups.items():
        indices = np.asarray(raw_indices, dtype=np.int64)
        order = np.argsort(timestamp_values[indices], kind="stable")
        indices = indices[order]
        times = timestamp_values[indices]
        events = event_values[indices]
        history: deque[tuple[int, int]] = deque(maxlen=length)

        start = 0
        while start < len(indices):
            current_time = int(times[start])
            stop = start + 1
            while stop < len(indices) and times[stop] == current_time:
                stop += 1

            while history and current_time - history[0][0] > timeout:
                history.popleft()

            if history:
                prior = np.fromiter((item[1] for item in history), dtype=np.int16)
                contexts[indices[start:stop], -len(prior) :] = prior

            # Same-second events become visible only to strictly later seconds.
            for event in events[start:stop]:
                history.append((current_time, int(event)))
            start = stop

    if not np.all(machine_values == frame["machine"].astype(str).to_numpy()):
        raise AssertionError("machine column changed while building contexts")
    return contexts


def prepare(config_path: Path) -> None:
    config = load_config(config_path)
    root = config_path.parent.parent
    data_cfg = config["data"]
    source_dir = root / data_cfg["official_csv_dir"]
    output_table = root / data_cfg["canonical_parquet"]
    output_context = root / data_cfg["contexts_npz"]
    output_manifest = root / data_cfg["manifest_json"]
    output_table.parent.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    source_hashes: dict[str, str] = {}
    observed_rows: dict[str, int] = {}
    for scenario in config["design"]["scenarios"]:
        source = source_dir / f"{scenario}_alerts.txt"
        if not source.exists():
            raise FileNotFoundError(f"Missing official AIT-ADS CSV: {source}")
        data = pd.read_csv(source, dtype_backend="pyarrow")
        required = {"time", "name", "ip", "host", "short", "time_label", "event_label"}
        if set(data.columns) != required:
            raise ValueError(f"Unexpected columns in {source}: {list(data.columns)}")
        if data.isna().any().any():
            raise ValueError(f"Null values found in {source}")
        observed_rows[scenario] = len(data)
        if len(data) != EXPECTED_SCENARIO_ROWS[scenario]:
            raise ValueError(
                f"Row-count mismatch for {scenario}: {len(data)} != "
                f"{EXPECTED_SCENARIO_ROWS[scenario]}"
            )
        data.insert(0, "scenario", scenario)
        data.insert(1, "source_row", np.arange(len(data), dtype=np.int32))
        frames.append(data)
        source_hashes[source.name] = sha256(source)

    frame = pd.concat(frames, ignore_index=True)
    if len(frame) != EXPECTED_TOTAL_ROWS:
        raise ValueError(f"Total row mismatch: {len(frame)} != {EXPECTED_TOTAL_ROWS}")

    frame = frame.rename(columns={"time": "timestamp", "short": "event_key"})
    frame["timestamp"] = frame["timestamp"].astype("int64")
    frame["machine"] = frame["scenario"].astype(str) + "::" + frame["host"].astype(str)
    frame["detector"] = detector_from_short(frame["event_key"].astype(str))
    if frame["detector"].isna().any():
        bad = sorted(frame.loc[frame["detector"].isna(), "event_key"].unique())
        raise ValueError(f"Cannot infer detector for event keys: {bad}")

    vocabulary = sorted(frame["event_key"].astype(str).unique())
    event_to_id = {event: index for index, event in enumerate(vocabulary)}
    frame["event_id"] = frame["event_key"].astype(str).map(event_to_id).astype("int16")
    frame["event_incident"] = frame["event_label"].astype(str).ne("-").astype("int8")
    frame["window_incident"] = (
        frame["time_label"].astype(str).ne("false_positive").astype("int8")
    )
    normalized = frame["event_label"].astype(str).map(EVENT_GROUP_MAP)
    frame["event_incident_group"] = np.where(
        frame["event_incident"].eq(1),
        frame["scenario"].astype(str) + "::" + normalized.fillna("unmapped"),
        None,
    )
    frame["window_incident_group"] = np.where(
        frame["window_incident"].eq(1),
        frame["scenario"].astype(str) + "::" + frame["time_label"].astype(str),
        None,
    )
    frame.insert(0, "row_id", np.arange(len(frame), dtype=np.int64))

    contexts = build_strict_past_contexts(
        frame=frame,
        event_count=len(vocabulary),
        length=int(data_cfg["context_length"]),
        timeout=int(data_cfg["timeout_seconds"]),
    )

    keep = [
        "row_id",
        "scenario",
        "source_row",
        "timestamp",
        "machine",
        "host",
        "ip",
        "detector",
        "event_key",
        "event_id",
        "name",
        "time_label",
        "event_label",
        "event_incident",
        "window_incident",
        "event_incident_group",
        "window_incident_group",
    ]
    frame[keep].to_parquet(output_table, index=False, compression="zstd")
    np.savez_compressed(
        output_context,
        context=contexts,
        target=frame["event_id"].to_numpy(dtype=np.int16),
        vocabulary=np.asarray(vocabulary, dtype=object),
        pad_id=np.asarray(len(vocabulary), dtype=np.int16),
    )

    manifest = {
        "source": "AIT-ADS official author-produced alerts_csv archive",
        "rows": len(frame),
        "scenario_rows": observed_rows,
        "source_sha256": source_hashes,
        "event_types": len(vocabulary),
        "detectors": frame["detector"].value_counts().sort_index().to_dict(),
        "event_label_counts": frame["event_label"].value_counts().to_dict(),
        "time_label_counts": frame["time_label"].value_counts().to_dict(),
        "event_incident_prevalence": float(frame["event_incident"].mean()),
        "window_incident_prevalence": float(frame["window_incident"].mean()),
        "context_policy": {
            "length": int(data_cfg["context_length"]),
            "timeout_seconds": int(data_cfg["timeout_seconds"]),
            "ties": "strictly earlier timestamps only",
            "multiplicity": "preserved",
        },
        "vocabulary": vocabulary,
    }
    with output_manifest.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
    )
    args = parser.parse_args()
    prepare(args.config.resolve())


if __name__ == "__main__":
    main()

