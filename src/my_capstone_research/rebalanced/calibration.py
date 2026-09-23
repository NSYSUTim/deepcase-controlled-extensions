from __future__ import annotations

import math
from typing import Any

from .config import RebalancedConfig


def calibrated_incident_posterior(
    *,
    cluster_prior: float,
    confidence: float,
    distance: float,
    distance_scale: float = 1.0,
) -> float:
    distance_penalty = math.exp(-max(distance, 0.0) / max(distance_scale, 1e-6))
    score = cluster_prior * confidence * distance_penalty
    return max(0.0, min(1.0, score))


def triage_decision(
    posterior: float,
    *,
    low_threshold: float,
    high_threshold: float,
) -> str:
    if posterior >= high_threshold:
        return "incident"
    if posterior <= low_threshold:
        return "benign"
    return "reject"


def build_calibration_plan(config: RebalancedConfig) -> dict[str, Any]:
    return {
        "inputs": [
            "cluster prior",
            "model confidence",
            "distance to cluster",
        ],
        "reject_policy": {
            "low_threshold": config.reject_low_threshold,
            "high_threshold": config.reject_high_threshold,
        },
        "difference_from_baseline": "baseline 以 cluster score 為主；研究版把 posterior calibration 與 reject-aware decision 一起納入。",
    }
