from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from deepcase.context_builder import ContextBuilder
from deepcase.interpreter import Interpreter

from my_capstone.config import RunConfig, load_run_config
from my_capstone.convert import ConvertResult
from my_capstone.evaluation import (
    BASELINE_PREDICTION_POLICY,
    PredictionPolicy,
    build_run_evaluation,
)
from my_capstone.pipeline import (
    SequenceBundle,
    SplitBundle,
    build_label_legend,
    execute_deepcase_prepare,
    save_mapping_json,
    save_sequence_bundle,
)
from my_capstone.scoring import (
    BENIGN_CLUSTER_SCORE,
    PREDICT_DISTANCE_EXCEEDED,
    PREDICT_NOT_CONFIDENT,
    PREDICT_UNSEEN_EVENT,
    TRAIN_NO_ATTACK_LABEL,
    build_cluster_scores,
    distribution,
    score_meaning,
)
from my_capstone.utils import save_annotated_csv, save_json, save_torch, tensor_to_cpu

from .data_types import ResearchRunArtifacts

NO_EVENT_ORIGINAL_ID = -1337
REBALANCED_REJECT_SCORE = -5


@dataclass
class ResearchTrainingState:
    baseline_config: RunConfig
    processed_result: ConvertResult
    split_bundle: SplitBundle
    builder: ContextBuilder
    interpreter: Interpreter
    clusters: np.ndarray
    cluster_scores: np.ndarray
    score_meaning_fn: Callable[[int], str]


def load_baseline_runtime(baseline_config_path: Path) -> tuple[RunConfig, ConvertResult]:
    baseline_config = load_run_config(baseline_config_path)
    processed_result = execute_deepcase_prepare(config=baseline_config, force_prepare=False)
    return baseline_config, processed_result


def load_processed_frame(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    frame["scenario"] = frame["machine"].astype(str).map(
        lambda value: value.rsplit("_", 1)[0] if "_" in value else value
    )
    frame["timestamp"] = frame["timestamp"].astype(float)
    frame["event"] = frame["event"].astype(int)
    frame["label"] = frame["label"].astype(int)
    return frame


def build_sequence_bundle(
    *,
    events: list[int],
    contexts: list[list[int]],
    labels: list[int],
    mapping: dict[int, int],
) -> SequenceBundle:
    return SequenceBundle(
        events=torch.as_tensor(np.asarray(events, dtype=np.int64), dtype=torch.long),
        context=torch.as_tensor(np.asarray(contexts, dtype=np.int64), dtype=torch.long),
        labels=torch.as_tensor(np.asarray(labels, dtype=np.int64), dtype=torch.long),
        mapping={int(key): int(value) for key, value in mapping.items()},
    )


def split_sequence_bundle(bundle: SequenceBundle, train_ratio: float) -> SplitBundle:
    n_samples = bundle.events.shape[0]
    split_index = max(1, int(n_samples * train_ratio))
    split_index = min(split_index, n_samples - 1)
    return SplitBundle(
        train=SequenceBundle(
            events=bundle.events[:split_index],
            context=bundle.context[:split_index],
            labels=bundle.labels[:split_index] if bundle.labels is not None else None,
            mapping=bundle.mapping,
        ),
        test=SequenceBundle(
            events=bundle.events[split_index:],
            context=bundle.context[split_index:],
            labels=bundle.labels[split_index:] if bundle.labels is not None else None,
            mapping=bundle.mapping,
        ),
    )


def to_device(bundle: SequenceBundle, device: str) -> SequenceBundle:
    return SequenceBundle(
        events=bundle.events.to(device),
        context=bundle.context.to(device),
        labels=bundle.labels.to(device) if bundle.labels is not None else None,
        mapping=bundle.mapping,
    )


def build_context_builder(
    *,
    baseline_config: RunConfig,
    n_features: int,
    max_length: int,
) -> ContextBuilder:
    return ContextBuilder(
        input_size=n_features,
        output_size=n_features,
        hidden_size=baseline_config.hidden_size,
        max_length=max_length,
    ).to(baseline_config.resolved_device)


def build_interpreter(
    *,
    baseline_config: RunConfig,
    builder: ContextBuilder,
    n_features: int,
) -> Interpreter:
    return Interpreter(
        context_builder=builder,
        features=n_features,
        eps=baseline_config.eps,
        min_samples=baseline_config.min_samples,
        threshold=baseline_config.threshold,
    )


def fit_builder(
    *,
    baseline_config: RunConfig,
    builder: ContextBuilder,
    train_bundle: SequenceBundle,
) -> None:
    device_bundle = to_device(train_bundle, baseline_config.resolved_device)
    builder.fit(
        X=device_bundle.context,
        y=device_bundle.events.reshape(-1, 1),
        epochs=baseline_config.epochs,
        batch_size=baseline_config.train_batch_size,
        learning_rate=baseline_config.learning_rate,
        verbose=True,
    )


def cluster_with_interpreter(
    *,
    baseline_config: RunConfig,
    interpreter: Interpreter,
    train_bundle: SequenceBundle,
) -> np.ndarray:
    device_bundle = to_device(train_bundle, baseline_config.resolved_device)
    return np.asarray(
        interpreter.cluster(
            X=device_bundle.context,
            y=device_bundle.events.reshape(-1, 1),
            iterations=baseline_config.query_iterations,
            batch_size=baseline_config.query_batch_size,
            verbose=True,
        )
    )


def predict_with_interpreter(
    *,
    baseline_config: RunConfig,
    interpreter: Interpreter,
    prediction_bundle: SequenceBundle,
) -> np.ndarray:
    device_bundle = to_device(prediction_bundle, baseline_config.resolved_device)
    return np.asarray(
        interpreter.predict(
            X=device_bundle.context,
            y=device_bundle.events.reshape(-1, 1),
            iterations=baseline_config.query_iterations,
            batch_size=baseline_config.query_batch_size,
            verbose=True,
        )
    )


def save_common_research_outputs(
    *,
    artifacts: ResearchRunArtifacts,
    split_bundle: SplitBundle,
    processed_result: ConvertResult,
) -> None:
    save_sequence_bundle(artifacts.train_sequences_save, split_bundle.train)
    save_sequence_bundle(artifacts.test_sequences_save, split_bundle.test)
    save_mapping_json(artifacts.mapping_json, split_bundle.train.mapping)
    save_json(artifacts.label_legend_json, build_label_legend(processed_result.attack_to_id))


def _original_event_ids(events: np.ndarray, mapping: dict[int, int]) -> list[int]:
    return [int(mapping.get(int(event_id), int(event_id))) for event_id in events.tolist()]


def save_cluster_report(
    *,
    path: Path,
    bundle: SequenceBundle,
    clusters: np.ndarray,
    cluster_scores: np.ndarray,
    score_meaning_fn: Callable[[int], str],
    extra_columns: dict[str, list[Any]] | None = None,
) -> Path:
    events = tensor_to_cpu(bundle.events).numpy()
    labels = tensor_to_cpu(bundle.labels).numpy()
    frame = pd.DataFrame(
        {
            "sequence_index": np.arange(events.shape[0], dtype=int),
            "internal_event_id": events,
            "original_event_id": _original_event_ids(events, bundle.mapping),
            "cluster_id": clusters.astype(int),
            "true_label": labels.astype(int),
            "assigned_score": cluster_scores.astype(int),
            "assigned_score_meaning": [score_meaning_fn(int(score)) for score in cluster_scores],
        }
    )
    if extra_columns:
        for key, value in extra_columns.items():
            frame[key] = value
    comments = [
        "研究版訓練資料分群結果。",
        "cluster_id 是 Interpreter 對 train sequence 的分群編號。",
        "assigned_score 是該 sequence 所在 cluster 最後被指定的分數或 triage 決策。",
    ]
    return save_annotated_csv(path, frame, comments)


def save_prediction_report(
    *,
    path: Path,
    bundle: SequenceBundle,
    predictions: np.ndarray,
    score_meaning_fn: Callable[[int], str],
    extra_columns: dict[str, list[Any]] | None = None,
) -> Path:
    events = tensor_to_cpu(bundle.events).numpy()
    labels = tensor_to_cpu(bundle.labels).numpy()
    frame = pd.DataFrame(
        {
            "sequence_index": np.arange(predictions.shape[0], dtype=int),
            "internal_event_id": events,
            "original_event_id": _original_event_ids(events, bundle.mapping),
            "predicted_score": predictions.astype(int),
            "predicted_score_meaning": [score_meaning_fn(int(score)) for score in predictions],
            "true_label": labels.astype(int),
        }
    )
    if extra_columns:
        for key, value in extra_columns.items():
            frame[key] = value
    comments = [
        "研究版測試資料預測結果。",
        "predicted_score 是研究版方法最終輸出的分數或 triage 決策。",
    ]
    return save_annotated_csv(path, frame, comments)


def save_predictions_tensor(
    *,
    path: Path,
    bundle: SequenceBundle,
    predictions: np.ndarray,
    extras: dict[str, Any] | None = None,
) -> Path:
    payload = {
        "prediction": predictions,
        "events_test": tensor_to_cpu(bundle.events),
        "context_test": tensor_to_cpu(bundle.context),
        "labels_test": tensor_to_cpu(bundle.labels),
        "mapping": bundle.mapping,
    }
    if extras:
        payload.update(extras)
    return save_torch(path, payload)


def build_research_summary(
    *,
    artifacts: ResearchRunArtifacts,
    method_name: str,
    display_name: str,
    baseline_config: RunConfig,
    split_bundle: SplitBundle,
    processed_result: ConvertResult,
    n_clusters: int | None,
    clusters: np.ndarray | None,
    cluster_scores: np.ndarray,
    predictions: np.ndarray,
    score_meaning_table: dict[str, str],
    notes: list[str],
    prediction_policy: PredictionPolicy | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "about": "研究版 DeepCASE run 摘要。",
        "run_id": artifacts.run_id,
        "method_name": method_name,
        "display_name": display_name,
        "dataset_name": baseline_config.dataset_name,
        "device": baseline_config.resolved_device,
        "processed_csv": str(processed_result.output_path),
        "processed_csv_reused": processed_result.reused_existing,
        "n_train_events": int(split_bundle.train.events.shape[0]),
        "n_test_events": int(split_bundle.test.events.shape[0]),
        "n_unique_events": int(len(split_bundle.train.mapping)),
        "n_clusters": int(n_clusters) if n_clusters is not None else None,
        "parameters": {
            "context_length": int(split_bundle.train.context.shape[1]),
            "timeout": baseline_config.timeout,
            "hidden_size": baseline_config.hidden_size,
            "epochs": baseline_config.epochs,
            "train_batch_size": baseline_config.train_batch_size,
            "learning_rate": baseline_config.learning_rate,
            "eps": baseline_config.eps,
            "min_samples": baseline_config.min_samples,
            "threshold": baseline_config.threshold,
            "query_iterations": baseline_config.query_iterations,
            "query_batch_size": baseline_config.query_batch_size,
            "train_split_ratio": baseline_config.train_split_ratio,
        },
        "score_meanings": score_meaning_table,
        "train_cluster_score_distribution": distribution(cluster_scores),
        "attack_label_legend_path": str(artifacts.label_legend_json),
        "notes": notes,
    }
    if predictions.size > 0:
        summary["prediction_score_distribution"] = distribution(predictions)
        negative_counts = Counter(int(value) for value in predictions.tolist() if int(value) < 0)
        summary["negative_code_distribution"] = {
            str(score): {
                "count": int(count),
                "meaning": score_meaning_table.get(str(score), score_meaning(score)),
            }
            for score, count in sorted(negative_counts.items())
        }

    train_labels = (
        tensor_to_cpu(split_bundle.train.labels).numpy()
        if split_bundle.train.labels is not None
        else None
    )
    test_labels = (
        tensor_to_cpu(split_bundle.test.labels).numpy()
        if split_bundle.test.labels is not None
        else None
    )
    summary["evaluation"] = build_run_evaluation(
        clusters=clusters,
        train_labels=train_labels,
        predictions=predictions if predictions.size > 0 else None,
        test_labels=test_labels,
        prediction_policy=prediction_policy or BASELINE_PREDICTION_POLICY,
    )
    return summary


def baseline_score_meaning(score: int) -> str:
    return score_meaning(score)


def baseline_score_meaning_table() -> dict[str, str]:
    return {
        str(BENIGN_CLUSTER_SCORE): "良性預設叢集",
        str(PREDICT_DISTANCE_EXCEEDED): "距離最近叢集太遠",
        str(PREDICT_UNSEEN_EVENT): "訓練沒見過的事件",
        str(PREDICT_NOT_CONFIDENT): "模型信心不足",
        ">=0": "攻擊類別 ID",
    }


def build_rebalanced_cluster_scores(
    *,
    clusters: np.ndarray,
    labels: np.ndarray,
    positive_weight: float,
    negative_weight: float,
    low_threshold: float,
    high_threshold: float,
) -> np.ndarray:
    result = np.full(labels.shape[0], REBALANCED_REJECT_SCORE, dtype=int)
    binary_labels = np.where(labels != TRAIN_NO_ATTACK_LABEL, 1, 0)

    for cluster in np.unique(clusters):
        if cluster == -1:
            continue
        indices = np.flatnonzero(clusters == cluster)
        cluster_binary = binary_labels[indices]
        positive = int(cluster_binary.sum())
        negative = int(cluster_binary.shape[0] - positive)
        numerator = positive_weight * positive
        denominator = numerator + negative_weight * negative
        posterior = 0.0 if denominator == 0 else numerator / denominator
        if posterior >= high_threshold:
            result[indices] = 1
        elif posterior <= low_threshold:
            result[indices] = 0
        else:
            result[indices] = REBALANCED_REJECT_SCORE
    return result


def rebalanced_score_meaning(score: int) -> str:
    table = {
        REBALANCED_REJECT_SCORE: "校準後仍不確定，保留 reject",
        0: "判為 benign / non-incident",
        1: "判為 incident",
        BENIGN_CLUSTER_SCORE: "良性預設叢集",
        PREDICT_DISTANCE_EXCEEDED: "距離最近叢集太遠",
        PREDICT_UNSEEN_EVENT: "訓練沒見過的事件",
        PREDICT_NOT_CONFIDENT: "模型信心不足",
    }
    return table.get(score, f"未知分數 {score}")


def rebalanced_score_meaning_table() -> dict[str, str]:
    return {
        str(REBALANCED_REJECT_SCORE): "校準後仍不確定，保留 reject",
        "0": "benign / non-incident",
        "1": "incident",
        str(PREDICT_DISTANCE_EXCEEDED): "距離最近叢集太遠",
        str(PREDICT_UNSEEN_EVENT): "訓練沒見過的事件",
        str(PREDICT_NOT_CONFIDENT): "模型信心不足",
    }


def binary_true_labels(labels: np.ndarray) -> list[int]:
    return [0 if int(value) == TRAIN_NO_ATTACK_LABEL else 1 for value in labels.tolist()]
