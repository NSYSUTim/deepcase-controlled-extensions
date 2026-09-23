from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd
from tqdm import tqdm

REQUIRED_COLUMNS = {"timestamp", "machine", "event", "label"}


@dataclass(frozen=True)
class ConvertResult:
    output_path: Path
    reused_existing: bool
    total_events: int
    total_attack_events: int
    attack_to_id: dict[str, int]


def parse_timestamp(timestamp_str: str) -> float:
    normalized = timestamp_str.rstrip("Z") + "+00:00"
    return datetime.fromisoformat(normalized).timestamp()


def build_label_lookup(
    labels_df: pd.DataFrame,
    attack_to_id: dict[str, int],
) -> dict[str, list[tuple[float, float, int]]]:
    lookup: dict[str, list[tuple[float, float, int]]] = {}
    for scenario, group in labels_df.groupby("scenario"):
        intervals: list[tuple[float, float, int]] = []
        for _, row in group.iterrows():
            intervals.append(
                (float(row["start"]), float(row["end"]), attack_to_id[str(row["attack"])])
            )
        intervals.sort(key=lambda item: item[0])
        lookup[str(scenario)] = intervals
    return lookup


def get_attack_label_fast(
    timestamp: float,
    intervals: list[tuple[float, float, int]],
) -> int:
    for start, end, attack_id in intervals:
        if start <= timestamp <= end:
            return attack_id
    return -1


def validate_processed_csv(path: Path | str) -> bool:
    candidate = Path(path)
    if not candidate.exists() or candidate.stat().st_size == 0:
        return False
    frame = pd.read_csv(candidate, nrows=5)
    if not REQUIRED_COLUMNS.issubset(frame.columns):
        return False
    return True


def convert_wazuh_to_deepcase_csv(
    *,
    raw_dir: Path,
    labels_path: Path,
    output_path: Path,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> ConvertResult:
    if not force and validate_processed_csv(output_path):
        log(f"[prepare] 重用既有 processed CSV：{output_path}")
        labels_df = pd.read_csv(labels_path)
        attack_types = sorted(str(item) for item in labels_df["attack"].unique())
        attack_to_id = {attack: index for index, attack in enumerate(attack_types)}
        return ConvertResult(
            output_path=output_path,
            reused_existing=True,
            total_events=0,
            total_attack_events=0,
            attack_to_id=attack_to_id,
        )

    labels_df = pd.read_csv(labels_path)
    attack_types = sorted(str(item) for item in labels_df["attack"].unique())
    attack_to_id = {attack: index for index, attack in enumerate(attack_types)}
    label_lookup = build_label_lookup(labels_df, attack_to_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_events = 0
    total_attack_events = 0
    wazuh_files = sorted(raw_dir.glob("*_wazuh.json"))
    if not wazuh_files:
        raise FileNotFoundError(f"在 {raw_dir} 找不到任何 '*_wazuh.json' 檔案。")

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["timestamp", "machine", "event", "label"])

        for wazuh_file in wazuh_files:
            scenario = wazuh_file.stem.replace("_wazuh", "")
            intervals = label_lookup.get(scenario, [])
            batch: list[list[object]] = []
            file_events = 0
            file_attack_events = 0

            with wazuh_file.open("r", encoding="utf-8") as input_file:
                for line in tqdm(
                    input_file,
                    desc=f"prepare:{scenario}",
                    unit="events",
                    mininterval=5.0,
                ):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    timestamp_str = record.get("@timestamp")
                    agent = record.get("agent")
                    rule = record.get("rule")
                    if timestamp_str is None or agent is None or rule is None:
                        continue

                    agent_id = agent.get("id")
                    rule_id = rule.get("id")
                    if agent_id is None or rule_id is None:
                        continue

                    timestamp = parse_timestamp(str(timestamp_str))
                    machine = f"{scenario}_{agent_id}"
                    event = int(rule_id)
                    label = get_attack_label_fast(timestamp, intervals)

                    batch.append([timestamp, machine, event, label])
                    file_events += 1
                    if label != -1:
                        file_attack_events += 1

                    if len(batch) >= 10_000:
                        writer.writerows(batch)
                        batch.clear()

            if batch:
                writer.writerows(batch)

            total_events += file_events
            total_attack_events += file_attack_events
            log(
                f"[prepare] {scenario}: 事件數={file_events} 攻擊事件數={file_attack_events}"
            )

    return ConvertResult(
        output_path=output_path,
        reused_existing=False,
        total_events=total_events,
        total_attack_events=total_attack_events,
        attack_to_id=attack_to_id,
    )
