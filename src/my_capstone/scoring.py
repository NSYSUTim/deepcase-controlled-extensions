from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np

TRAIN_NO_ATTACK_LABEL = -1
BENIGN_CLUSTER_SCORE = -4
PREDICT_DISTANCE_EXCEEDED = -3
PREDICT_UNSEEN_EVENT = -2
PREDICT_NOT_CONFIDENT = -1

PREDICTION_MEANINGS = {
    BENIGN_CLUSTER_SCORE: "良性預設叢集",
    PREDICT_DISTANCE_EXCEEDED: "最接近叢集仍超出距離門檻",
    PREDICT_UNSEEN_EVENT: "預測時遇到訓練未見事件",
    PREDICT_NOT_CONFIDENT: "模型信心不足",
}


def score_meaning(score: int) -> str:
    if score in PREDICTION_MEANINGS:
        return PREDICTION_MEANINGS[score]
    if score >= 0:
        return "攻擊類別分數"
    return "未知負碼"


def build_cluster_scores(
    *,
    clusters: np.ndarray,
    labels: np.ndarray,
    strategy: str = "max",
    no_attack_label: int = TRAIN_NO_ATTACK_LABEL,
    benign_cluster_score: int = BENIGN_CLUSTER_SCORE,
) -> np.ndarray:
    if clusters.shape != labels.shape:
        raise ValueError("clusters 和 labels 的 shape 必須一致。")

    result = np.full(labels.shape[0], no_attack_label, dtype=int)
    unique_clusters = np.unique(clusters)

    for cluster in unique_clusters:
        if cluster == -1:
            continue

        indices = np.flatnonzero(clusters == cluster)
        cluster_labels = labels[indices]
        valid_labels = cluster_labels[cluster_labels != no_attack_label]

        if valid_labels.size == 0:
            result[indices] = benign_cluster_score
            continue

        if strategy == "max":
            score = int(valid_labels.max())
        elif strategy == "min":
            score = int(valid_labels.min())
        elif strategy == "avg":
            score = int(round(float(valid_labels.mean())))
        else:
            raise ValueError(f"不支援的分數策略：{strategy}")

        result[indices] = score

    return result


def distribution(values: Iterable[int]) -> dict[str, int]:
    counts = Counter(int(value) for value in values)
    return {str(key): int(count) for key, count in sorted(counts.items())}


def negative_distribution(values: Iterable[int]) -> dict[str, dict[str, object]]:
    counts = Counter(int(value) for value in values if int(value) < 0)
    return {
        str(key): {
            "count": int(count),
            "meaning": score_meaning(key),
        }
        for key, count in sorted(counts.items())
    }
