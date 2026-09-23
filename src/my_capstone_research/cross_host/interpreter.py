from __future__ import annotations

from typing import Any

from .config import CrossHostConfig


def build_cross_host_interpreter_plan(config: CrossHostConfig) -> dict[str, Any]:
    return {
        "baseline_policy": "延用 DeepCASE 的 attention fingerprint -> L1 distance -> DBSCAN 叢集。",
        "research_policy": "把輸入 fingerprint 改為 fused local/cross-host representation。",
        "preserve_l1_distance": config.preserve_l1_distance,
        "preserve_dbscan": config.preserve_dbscan,
        "future_variant": [
            "第二階段可再比較 HDBSCAN",
            "第二階段可再比較 Transformer companion encoder",
        ],
    }
