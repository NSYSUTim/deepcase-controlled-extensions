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
    "context_length",
    "loss_type",
    "focal_gamma",
    "positive_class_weight",
    "negative_class_weight",
    "reject_low_threshold",
    "reject_high_threshold",
    "preserve_l1_distance",
    "preserve_dbscan",
}


@dataclass(frozen=True)
class RebalancedConfig(ResearchConfigBase):
    context_length: int
    loss_type: str
    focal_gamma: float
    positive_class_weight: float
    negative_class_weight: float
    reject_low_threshold: float
    reject_high_threshold: float
    preserve_l1_distance: bool
    preserve_dbscan: bool

    def as_dict(self) -> dict[str, object]:
        payload = self.common_dict()
        payload.update(
            {
                "context_length": self.context_length,
                "loss_type": self.loss_type,
                "focal_gamma": self.focal_gamma,
                "positive_class_weight": self.positive_class_weight,
                "negative_class_weight": self.negative_class_weight,
                "reject_low_threshold": self.reject_low_threshold,
                "reject_high_threshold": self.reject_high_threshold,
                "preserve_l1_distance": self.preserve_l1_distance,
                "preserve_dbscan": self.preserve_dbscan,
            }
        )
        return payload


def load_rebalanced_config(path: str | Path) -> RebalancedConfig:
    config_path = resolve_path(path)
    payload = load_json(config_path)
    missing = sorted(REQUIRED_KEYS - payload.keys())
    if missing:
        raise ValueError(f"rebalanced 設定檔缺少欄位：{', '.join(missing)}")
    if str(payload["method_name"]) != "rebalanced":
        raise ValueError("rebalanced 設定檔的 method_name 必須是 rebalanced。")

    return RebalancedConfig(
        config_path=config_path,
        method_name="rebalanced",
        display_name=str(payload["display_name"]),
        variant_name=str(payload["variant_name"]),
        baseline_config=resolve_path(payload["baseline_config"], base_dir=PROJECT_ROOT),
        baseline_run_id=str(payload["baseline_run_id"]),
        results_root=resolve_path(payload["results_root"], base_dir=PROJECT_ROOT),
        dataset_name=str(payload["dataset_name"]),
        device=str(payload["device"]),
        notes=str(payload["notes"]),
        hidden_size=int(payload["hidden_size"]),
        context_length=int(payload["context_length"]),
        loss_type=str(payload["loss_type"]),
        focal_gamma=float(payload["focal_gamma"]),
        positive_class_weight=float(payload["positive_class_weight"]),
        negative_class_weight=float(payload["negative_class_weight"]),
        reject_low_threshold=float(payload["reject_low_threshold"]),
        reject_high_threshold=float(payload["reject_high_threshold"]),
        preserve_l1_distance=bool(payload["preserve_l1_distance"]),
        preserve_dbscan=bool(payload["preserve_dbscan"]),
    )
