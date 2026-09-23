from __future__ import annotations

from typing import Any

from .config import HierarchicalContextConfig


def build_hierarchical_interpreter_plan(config: HierarchicalContextConfig) -> dict[str, Any]:
    return {
        "baseline_policy": "延用 DeepCASE 的 attention fingerprint -> L1 distance -> DBSCAN。",
        "research_policy": "把 fingerprint 來源改成 hierarchical fused representation。",
        "preserve_l1_distance": config.preserve_l1_distance,
        "preserve_dbscan": config.preserve_dbscan,
        "future_variant": [
            "Transformer long-memory encoder",
            "HDBSCAN 版本 Interpreter",
        ],
    }
