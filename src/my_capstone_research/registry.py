from __future__ import annotations

from pathlib import Path
from typing import Any

from my_capstone.utils import load_json, resolve_path

from .cross_host.pipeline import CrossHostResearchPipeline
from .hierarchical_context.pipeline import HierarchicalContextResearchPipeline
from .rebalanced.pipeline import RebalancedResearchPipeline

PIPELINES = {
    "cross_host": CrossHostResearchPipeline(),
    "hierarchical_context": HierarchicalContextResearchPipeline(),
    "rebalanced": RebalancedResearchPipeline(),
}


def get_research_pipeline(method_name: str):
    key = str(method_name).strip().lower()
    if key not in PIPELINES:
        available = ", ".join(sorted(PIPELINES))
        raise ValueError(f"未知的研究方法：{method_name}。可用方法：{available}")
    return PIPELINES[key]


def load_research_config(path: str | Path) -> Any:
    config_path = resolve_path(path)
    payload = load_json(config_path)
    method_name = payload.get("method_name")
    if not method_name:
        raise ValueError("研究設定檔缺少 method_name。")
    pipeline = get_research_pipeline(str(method_name))
    return pipeline.load_config(config_path)
