from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from my_capstone.utils import PROJECT_ROOT, load_json, resolve_path

from ..common.data_types import ResearchConfigBase

REQUIRED_KEYS = {
    "method_name",
    "display_name",
    "variant_name",
    "baseline_config",
    "baseline_run_id",
    "results_root",
    "dataset_name",
    "device",
    "notes",
    "hidden_size",
    "short_context_length",
    "long_memory_length",
    "landmark_strategy",
    "landmark_time_window_seconds",
    "preserve_l1_distance",
    "preserve_dbscan",
}


@dataclass(frozen=True)
class HierarchicalContextConfig(ResearchConfigBase):
    short_context_length: int
    long_memory_length: int
    landmark_strategy: str
    landmark_time_window_seconds: int
    landmark_candidate_limit: int
    preserve_l1_distance: bool
    preserve_dbscan: bool

    def as_dict(self) -> dict[str, object]:
        payload = self.common_dict()
        payload.update(
            {
                "short_context_length": self.short_context_length,
                "long_memory_length": self.long_memory_length,
                "landmark_strategy": self.landmark_strategy,
                "landmark_time_window_seconds": self.landmark_time_window_seconds,
                "landmark_candidate_limit": self.landmark_candidate_limit,
                "preserve_l1_distance": self.preserve_l1_distance,
                "preserve_dbscan": self.preserve_dbscan,
            }
        )
        return payload


def load_hierarchical_context_config(path: str | Path) -> HierarchicalContextConfig:
    config_path = resolve_path(path)
    payload = load_json(config_path)
    missing = sorted(REQUIRED_KEYS - payload.keys())
    if missing:
        raise ValueError(f"hierarchical_context 設定檔缺少欄位：{', '.join(missing)}")
    if str(payload["method_name"]) != "hierarchical_context":
        raise ValueError(
            "hierarchical_context 設定檔的 method_name 必須是 hierarchical_context。"
        )

    return HierarchicalContextConfig(
        config_path=config_path,
        method_name="hierarchical_context",
        display_name=str(payload["display_name"]),
        variant_name=str(payload["variant_name"]),
        baseline_config=resolve_path(payload["baseline_config"], base_dir=PROJECT_ROOT),
        baseline_run_id=str(payload["baseline_run_id"]),
        results_root=resolve_path(payload["results_root"], base_dir=PROJECT_ROOT),
        dataset_name=str(payload["dataset_name"]),
        device=str(payload["device"]),
        notes=str(payload["notes"]),
        hidden_size=int(payload["hidden_size"]),
        short_context_length=int(payload["short_context_length"]),
        long_memory_length=int(payload["long_memory_length"]),
        landmark_strategy=str(payload["landmark_strategy"]),
        landmark_time_window_seconds=int(payload["landmark_time_window_seconds"]),
        landmark_candidate_limit=int(
            payload.get(
                "landmark_candidate_limit",
                max(256, int(payload["long_memory_length"]) * 8),
            )
        ),
        preserve_l1_distance=bool(payload["preserve_l1_distance"]),
        preserve_dbscan=bool(payload["preserve_dbscan"]),
    )
