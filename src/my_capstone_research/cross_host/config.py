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
    "local_context_length",
    "companion_context_length",
    "companion_time_window_seconds",
    "fusion_strategy",
    "preserve_l1_distance",
    "preserve_dbscan",
}


@dataclass(frozen=True)
class CrossHostConfig(ResearchConfigBase):
    local_context_length: int
    companion_context_length: int
    companion_time_window_seconds: int
    fusion_strategy: str
    preserve_l1_distance: bool
    preserve_dbscan: bool
    token_scope: str = "machine_event"

    def as_dict(self) -> dict[str, object]:
        payload = self.common_dict()
        payload.update(
            {
                "local_context_length": self.local_context_length,
                "companion_context_length": self.companion_context_length,
                "companion_time_window_seconds": self.companion_time_window_seconds,
                "fusion_strategy": self.fusion_strategy,
                "preserve_l1_distance": self.preserve_l1_distance,
                "preserve_dbscan": self.preserve_dbscan,
                "token_scope": self.token_scope,
            }
        )
        return payload


def load_cross_host_config(path: str | Path) -> CrossHostConfig:
    config_path = resolve_path(path)
    payload = load_json(config_path)
    missing = sorted(REQUIRED_KEYS - payload.keys())
    if missing:
        raise ValueError(f"cross_host 設定檔缺少欄位：{', '.join(missing)}")
    if str(payload["method_name"]) != "cross_host":
        raise ValueError("cross_host 設定檔的 method_name 必須是 cross_host。")

    token_scope = str(payload.get("token_scope", "machine_event"))
    if token_scope not in {"event", "machine_event"}:
        raise ValueError("cross_host token_scope must be 'event' or 'machine_event'.")

    return CrossHostConfig(
        config_path=config_path,
        method_name="cross_host",
        display_name=str(payload["display_name"]),
        variant_name=str(payload["variant_name"]),
        baseline_config=resolve_path(payload["baseline_config"], base_dir=PROJECT_ROOT),
        baseline_run_id=str(payload["baseline_run_id"]),
        results_root=resolve_path(payload["results_root"], base_dir=PROJECT_ROOT),
        dataset_name=str(payload["dataset_name"]),
        device=str(payload["device"]),
        notes=str(payload["notes"]),
        hidden_size=int(payload["hidden_size"]),
        local_context_length=int(payload["local_context_length"]),
        companion_context_length=int(payload["companion_context_length"]),
        companion_time_window_seconds=int(payload["companion_time_window_seconds"]),
        fusion_strategy=str(payload["fusion_strategy"]),
        preserve_l1_distance=bool(payload["preserve_l1_distance"]),
        preserve_dbscan=bool(payload["preserve_dbscan"]),
        token_scope=token_scope,
    )
