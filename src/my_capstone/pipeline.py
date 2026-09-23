from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

from deepcase.context_builder import ContextBuilder
from deepcase.interpreter import Interpreter
from deepcase.preprocessing import Preprocessor

from .artifacts import RunArtifacts
from .config import RunConfig
from .convert import ConvertResult, convert_wazuh_to_deepcase_csv
from .evaluation import BASELINE_PREDICTION_POLICY, build_run_evaluation
from .scoring import (
    BENIGN_CLUSTER_SCORE,
    PREDICT_DISTANCE_EXCEEDED,
    PREDICT_NOT_CONFIDENT,
    PREDICT_UNSEEN_EVENT,
    TRAIN_NO_ATTACK_LABEL,
    build_cluster_scores,
    distribution,
    negative_distribution,
    score_meaning,
)
from .utils import (
    resolve_path,
    save_annotated_csv,
    save_json,
    save_torch,
    tensor_to_cpu,
)


@dataclass
class SequenceBundle:
    events: torch.Tensor
    context: torch.Tensor
    labels: torch.Tensor | None
    mapping: dict[int, int]


@dataclass
class SplitBundle:
    train: SequenceBundle
    test: SequenceBundle


@dataclass
class TrainingState:
    split_bundle: SplitBundle
    builder: ContextBuilder
    interpreter: Interpreter
    clusters: np.ndarray
    cluster_scores: np.ndarray
    processed_result: ConvertResult


def prediction_negative_code_descriptions() -> dict[str, str]:
    return {
        str(BENIGN_CLUSTER_SCORE): "良性預設叢集",
        str(PREDICT_DISTANCE_EXCEEDED): "最接近叢集仍超出距離門檻",
        str(PREDICT_UNSEEN_EVENT): "預測時遇到訓練未見事件",
        str(PREDICT_NOT_CONFIDENT): "模型信心不足",
    }


def sequence_bundle_to_cpu(bundle: SequenceBundle) -> SequenceBundle:
    return SequenceBundle(
        events=tensor_to_cpu(bundle.events),
        context=tensor_to_cpu(bundle.context),
        labels=tensor_to_cpu(bundle.labels),
        mapping={int(key): int(value) for key, value in bundle.mapping.items()},
    )


def save_sequence_bundle(path: Path, bundle: SequenceBundle) -> Path:
    cpu_bundle = sequence_bundle_to_cpu(bundle)
    return save_torch(
        path,
        {
            "events": cpu_bundle.events,
            "context": cpu_bundle.context,
            "labels": cpu_bundle.labels,
            "mapping": cpu_bundle.mapping,
        },
    )


def load_sequence_bundle(path: Path) -> SequenceBundle:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return SequenceBundle(
        events=payload["events"],
        context=payload["context"],
        labels=payload["labels"],
        mapping={int(key): int(value) for key, value in payload["mapping"].items()},
    )


def build_label_legend(attack_to_id: dict[str, int]) -> dict[str, Any]:
    attack_id_to_name = {
        str(attack_id): attack_name for attack_name, attack_id in sorted(attack_to_id.items())
    }
    return {
        "about": (
            "本檔用來解釋 true_label 與正分數的意義。"
            "在這個專案裡，true_label 來自資料集標註；>=0 的分數代表某個 attack 類別 ID。"
        ),
        "true_label_definition": {
            "-1": "這筆 sequence 沒有對到任何攻擊區間",
            ">=0": "這筆 sequence 對應到某個 attack 類別 ID",
        },
        "prediction_special_values": prediction_negative_code_descriptions(),
        "attack_id_to_name": attack_id_to_name,
    }


def save_mapping_json(path: Path, mapping: dict[int, int]) -> Path:
    special_event_values = {}
    for internal_id, original_id in mapping.items():
        if int(original_id) == -1337:
            special_event_values[str(internal_id)] = "NO_EVENT / padding"

    payload = {
        "about": (
            "DeepCASE 會把原始 event ID 重新編成內部連續編號。"
            "報表中的 internal_event_id 請用本檔對回 original_event_id。"
        ),
        "fields_used_by_reports": {
            "internal_event_id": "DeepCASE 模型內部使用的事件編號",
            "original_event_id": "原始資料中的事件 ID，例如 Wazuh rule.id",
        },
        "special_event_values": special_event_values,
        "internal_event_id_to_original_event_id": {
            str(int(internal_id)): int(original_id)
            for internal_id, original_id in sorted(mapping.items())
        },
    }
    return save_json(path, payload)


def execute_deepcase_prepare(
    *,
    config: RunConfig,
    force_prepare: bool = False,
) -> ConvertResult:
    return convert_wazuh_to_deepcase_csv(
        raw_dir=config.raw_data_dir,
        labels_path=config.labels_csv,
        output_path=config.processed_csv,
        force=force_prepare,
    )


def load_processed_sequences(config: RunConfig, csv_path: Path) -> SequenceBundle:
    preprocessor = Preprocessor(
        length=config.context_length,
        timeout=config.timeout,
    )
    context, events, labels, mapping = preprocessor.csv(str(csv_path), verbose=True)
    if labels is None:
        labels = np.full(events.shape[0], TRAIN_NO_ATTACK_LABEL, dtype=int)
        labels = torch.as_tensor(labels, dtype=torch.long)
    mapping_int = {int(key): int(value) for key, value in mapping.items()}
    return SequenceBundle(
        events=events,
        context=context,
        labels=labels,
        mapping=mapping_int,
    )


def split_sequence_bundle(bundle: SequenceBundle, train_ratio: float) -> SplitBundle:
    n_samples = bundle.events.shape[0]
    split_index = max(1, int(n_samples * train_ratio))
    split_index = min(split_index, n_samples - 1)

    train_bundle = SequenceBundle(
        events=bundle.events[:split_index],
        context=bundle.context[:split_index],
        labels=bundle.labels[:split_index] if bundle.labels is not None else None,
        mapping=bundle.mapping,
    )
    test_bundle = SequenceBundle(
        events=bundle.events[split_index:],
        context=bundle.context[split_index:],
        labels=bundle.labels[split_index:] if bundle.labels is not None else None,
        mapping=bundle.mapping,
    )
    return SplitBundle(train=train_bundle, test=test_bundle)


def _to_device(bundle: SequenceBundle, device: str) -> SequenceBundle:
    return SequenceBundle(
        events=bundle.events.to(device),
        context=bundle.context.to(device),
        labels=bundle.labels.to(device) if bundle.labels is not None else None,
        mapping=bundle.mapping,
    )


def build_context_builder(config: RunConfig, n_features: int) -> ContextBuilder:
    return ContextBuilder(
        input_size=n_features,
        output_size=n_features,
        hidden_size=config.hidden_size,
        max_length=config.context_length,
    ).to(config.resolved_device)


def build_interpreter(config: RunConfig, builder: ContextBuilder, n_features: int) -> Interpreter:
    return Interpreter(
        context_builder=builder,
        features=n_features,
        eps=config.eps,
        min_samples=config.min_samples,
        threshold=config.threshold,
    )


def _original_event_ids(events: np.ndarray, mapping: dict[int, int]) -> list[int]:
    return [int(mapping.get(int(event_id), int(event_id))) for event_id in events.tolist()]


def save_clusters_csv(
    *,
    path: Path,
    clusters: np.ndarray,
    bundle: SequenceBundle,
    cluster_scores: np.ndarray,
) -> Path:
    events = tensor_to_cpu(bundle.events).numpy()
    labels = tensor_to_cpu(bundle.labels).numpy() if bundle.labels is not None else np.full(
        events.shape[0],
        np.nan,
    )

    frame = pd.DataFrame(
        {
            "sequence_index": np.arange(events.shape[0], dtype=int),
            "internal_event_id": events,
            "original_event_id": _original_event_ids(events, bundle.mapping),
            "cluster_id": clusters.astype(int),
            "true_label": labels,
            "assigned_score": cluster_scores,
            "assigned_score_meaning": [
                score_meaning(int(score)) for score in cluster_scores
            ],
        }
    )
    comments = [
        "DeepCASE 訓練叢集報告。每列代表 1 筆訓練 sequence。",
        "cluster_id 是 Interpreter 對訓練 sequence 指派的叢集編號；-1 表示噪音或未進入任何已知叢集。",
        "true_label 是資料集原始標註；-1 表示沒有對到攻擊區間，>=0 表示 attack 類別 ID。",
        "assigned_score 是該 sequence 最終繼承的 cluster 分數；-4 表示良性預設叢集，>=0 表示 attack 類別分數。",
        "internal_event_id 是 DeepCASE 內部事件編號；original_event_id 是原始事件 ID。",
    ]
    return save_annotated_csv(path, frame, comments)


def save_prediction_csv(
    *,
    path: Path,
    predictions: np.ndarray,
    bundle: SequenceBundle,
) -> Path:
    events = tensor_to_cpu(bundle.events).numpy()
    labels = tensor_to_cpu(bundle.labels).numpy() if bundle.labels is not None else np.full(
        predictions.shape[0],
        np.nan,
    )

    frame = pd.DataFrame(
        {
            "sequence_index": np.arange(predictions.shape[0], dtype=int),
            "internal_event_id": events,
            "original_event_id": _original_event_ids(events, bundle.mapping),
            "predicted_score": predictions,
            "predicted_score_meaning": [
                score_meaning(int(score)) for score in predictions
            ],
            "true_label": labels,
        }
    )
    comments = [
        "DeepCASE 測試預測報告。每列代表 1 筆 test sequence。",
        "predicted_score 是 Interpreter 對該 sequence 輸出的分數。",
        "predicted_score = -4 表示良性預設叢集；-3 表示離所有已知叢集太遠；-2 表示未見事件；-1 表示模型信心不足；>=0 表示 attack 類別分數。",
        "true_label 是資料集原始標註；-1 表示沒有對到攻擊區間，>=0 表示 attack 類別 ID。",
        "internal_event_id 是 DeepCASE 內部事件編號；original_event_id 是原始事件 ID。",
    ]
    return save_annotated_csv(path, frame, comments)


def build_summary(
    *,
    command: str,
    artifacts: RunArtifacts,
    config: RunConfig,
    processed_result: ConvertResult | None,
    split_bundle: SplitBundle | None,
    n_unique_events: int | None,
    n_clusters: int | None,
    clusters: np.ndarray | None = None,
    cluster_scores: np.ndarray | None = None,
    predictions: np.ndarray | None = None,
    source_run_id: str | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "about": (
            "本檔是這次 DeepCASE run 的摘要。"
            "若只想先看整體表現，請優先看 n_clusters、prediction_score_distribution、negative_code_distribution。"
        ),
        "run_id": artifacts.run_id,
        "command": command,
        "source_run_id": source_run_id,
        "dataset_name": config.dataset_name,
        "device": config.resolved_device,
        "processed_csv": str(config.processed_csv),
        "processed_csv_reused": (
            processed_result.reused_existing if processed_result is not None else None
        ),
        "parameters": {
            "context_length": config.context_length,
            "timeout": config.timeout,
            "hidden_size": config.hidden_size,
            "epochs": config.epochs,
            "train_batch_size": config.train_batch_size,
            "learning_rate": config.learning_rate,
            "eps": config.eps,
            "min_samples": config.min_samples,
            "threshold": config.threshold,
            "query_iterations": config.query_iterations,
            "query_batch_size": config.query_batch_size,
            "train_split_ratio": config.train_split_ratio,
        },
        "scoring_policy": {
            "training_no_attack_label": TRAIN_NO_ATTACK_LABEL,
            "benign_cluster_score": BENIGN_CLUSTER_SCORE,
            "prediction_negative_codes": prediction_negative_code_descriptions(),
        },
    }

    if split_bundle is not None:
        summary["n_train_events"] = int(split_bundle.train.events.shape[0])
        summary["n_test_events"] = int(split_bundle.test.events.shape[0])
    if n_unique_events is not None:
        summary["n_unique_events"] = int(n_unique_events)
    if n_clusters is not None:
        summary["n_clusters"] = int(n_clusters)
    if cluster_scores is not None:
        summary["train_cluster_score_distribution"] = distribution(cluster_scores)
    if predictions is not None:
        summary["prediction_score_distribution"] = distribution(predictions)
        summary["negative_code_distribution"] = negative_distribution(predictions)

    if split_bundle is not None and (clusters is not None or predictions is not None):
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
            predictions=predictions,
            test_labels=test_labels,
            prediction_policy=BASELINE_PREDICTION_POLICY,
        )

    if processed_result is not None:
        summary["attack_label_legend_path"] = str(artifacts.label_legend_json)

    return summary


def save_run_config(path: Path, config: RunConfig) -> Path:
    payload = {
        "about": "本檔是這次 run 實際使用的參數快照。若你想比較不同實驗設定，請看這份檔案。",
        "config": config.as_dict(),
    }
    return save_json(path, payload)


def train_deepcase(
    *,
    config: RunConfig,
    artifacts: RunArtifacts,
    force_prepare: bool,
) -> TrainingState:
    processed_result = execute_deepcase_prepare(
        config=config,
        force_prepare=force_prepare,
    )
    full_bundle = load_processed_sequences(config, processed_result.output_path)
    split_bundle = split_sequence_bundle(full_bundle, config.train_split_ratio)

    if config.save_train_sequences:
        save_sequence_bundle(artifacts.train_sequences_save, split_bundle.train)
    if config.save_test_sequences:
        save_sequence_bundle(artifacts.test_sequences_save, split_bundle.test)

    save_mapping_json(artifacts.mapping_json, split_bundle.train.mapping)
    save_json(artifacts.label_legend_json, build_label_legend(processed_result.attack_to_id))

    n_unique_events = len(split_bundle.train.mapping)
    builder = build_context_builder(config, n_unique_events)
    train_device_bundle = _to_device(split_bundle.train, config.resolved_device)

    builder.fit(
        X=train_device_bundle.context,
        y=train_device_bundle.events.reshape(-1, 1),
        epochs=config.epochs,
        batch_size=config.train_batch_size,
        learning_rate=config.learning_rate,
        verbose=True,
    )
    if config.save_builder:
        builder.save(artifacts.builder_save)

    interpreter = build_interpreter(config, builder, n_unique_events)
    clusters = interpreter.cluster(
        X=train_device_bundle.context,
        y=train_device_bundle.events.reshape(-1, 1),
        iterations=config.query_iterations,
        batch_size=config.query_batch_size,
        verbose=True,
    )

    if split_bundle.train.labels is None:
        raise ValueError("目前的分數流程需要訓練標籤，不能缺少 labels。")
    train_labels_np = tensor_to_cpu(split_bundle.train.labels).numpy()
    cluster_scores = build_cluster_scores(
        clusters=np.asarray(interpreter.clusters),
        labels=train_labels_np,
        strategy="max",
    )
    interpreter.score(cluster_scores, verbose=True)

    save_clusters_csv(
        path=artifacts.clusters_csv,
        clusters=np.asarray(interpreter.clusters),
        bundle=split_bundle.train,
        cluster_scores=cluster_scores,
    )

    if config.save_interpreter:
        interpreter.save(artifacts.interpreter_save)

    return TrainingState(
        split_bundle=split_bundle,
        builder=builder,
        interpreter=interpreter,
        clusters=np.asarray(clusters),
        cluster_scores=cluster_scores,
        processed_result=processed_result,
    )


def predict_with_interpreter(
    *,
    config: RunConfig,
    interpreter: Interpreter,
    prediction_bundle: SequenceBundle,
) -> np.ndarray:
    prediction_device_bundle = _to_device(prediction_bundle, config.resolved_device)
    predictions = interpreter.predict(
        X=prediction_device_bundle.context,
        y=prediction_device_bundle.events.reshape(-1, 1),
        iterations=config.query_iterations,
        batch_size=config.query_batch_size,
        verbose=True,
    )
    return np.asarray(predictions)


def execute_deepcase_train(
    *,
    config: RunConfig,
    artifacts: RunArtifacts,
    force_prepare: bool = False,
) -> dict[str, Any]:
    save_run_config(artifacts.run_config_json, config)
    training_state = train_deepcase(
        config=config,
        artifacts=artifacts,
        force_prepare=force_prepare,
    )
    summary = build_summary(
        command="deepcase-train",
        artifacts=artifacts,
        config=config,
        processed_result=training_state.processed_result,
        split_bundle=training_state.split_bundle,
        n_unique_events=len(training_state.split_bundle.train.mapping),
        n_clusters=max(0, int(training_state.clusters.max()) + 1)
        if training_state.clusters.size
        else 0,
        clusters=training_state.clusters,
        cluster_scores=training_state.cluster_scores,
    )
    if config.save_summary_json:
        save_json(artifacts.summary_json, summary)
    return summary


def execute_deepcase_predict(
    *,
    config: RunConfig,
    artifacts: RunArtifacts,
    source_artifacts: RunArtifacts,
    input_csv: str | None = None,
) -> dict[str, Any]:
    save_run_config(artifacts.run_config_json, config)
    builder = ContextBuilder.load(source_artifacts.builder_save, config.resolved_device)
    interpreter = Interpreter.load(
        source_artifacts.interpreter_save,
        context_builder=builder,
    )

    if input_csv is None:
        prediction_bundle = load_sequence_bundle(source_artifacts.test_sequences_save)
    else:
        prediction_bundle = load_processed_sequences(config, resolve_path(input_csv))

    if config.save_test_sequences:
        save_sequence_bundle(artifacts.test_sequences_save, prediction_bundle)
    save_mapping_json(artifacts.mapping_json, prediction_bundle.mapping)

    if source_artifacts.label_legend_json.exists():
        label_legend = source_artifacts.label_legend_json.read_text(encoding="utf-8")
        artifacts.label_legend_json.write_text(label_legend, encoding="utf-8")

    predictions = predict_with_interpreter(
        config=config,
        interpreter=interpreter,
        prediction_bundle=prediction_bundle,
    )
    if config.save_prediction_csv:
        save_prediction_csv(
            path=artifacts.prediction_csv,
            predictions=predictions,
            bundle=prediction_bundle,
        )
    if config.save_predictions_pt:
        save_torch(
            artifacts.predictions_pt,
            {
                "prediction": predictions,
                "events_test": tensor_to_cpu(prediction_bundle.events),
                "context_test": tensor_to_cpu(prediction_bundle.context),
                "labels_test": tensor_to_cpu(prediction_bundle.labels),
                "mapping": prediction_bundle.mapping,
            },
        )

    n_test_events = int(prediction_bundle.events.shape[0])
    summary = build_summary(
        command="deepcase-predict",
        artifacts=artifacts,
        config=config,
        processed_result=None,
        split_bundle=SplitBundle(train=prediction_bundle, test=prediction_bundle),
        n_unique_events=len(prediction_bundle.mapping),
        n_clusters=None,
        clusters=None,
        predictions=predictions,
        source_run_id=source_artifacts.run_id,
    )
    summary["n_train_events"] = None
    summary["n_test_events"] = n_test_events
    if config.save_summary_json:
        save_json(artifacts.summary_json, summary)
    return summary


def execute_deepcase_run(
    *,
    config: RunConfig,
    artifacts: RunArtifacts,
    force_prepare: bool = False,
) -> dict[str, Any]:
    save_run_config(artifacts.run_config_json, config)
    training_state = train_deepcase(
        config=config,
        artifacts=artifacts,
        force_prepare=force_prepare,
    )
    predictions = predict_with_interpreter(
        config=config,
        interpreter=training_state.interpreter,
        prediction_bundle=training_state.split_bundle.test,
    )

    if config.save_prediction_csv:
        save_prediction_csv(
            path=artifacts.prediction_csv,
            predictions=predictions,
            bundle=training_state.split_bundle.test,
        )
    if config.save_predictions_pt:
        save_torch(
            artifacts.predictions_pt,
            {
                "prediction": predictions,
                "events_test": tensor_to_cpu(training_state.split_bundle.test.events),
                "context_test": tensor_to_cpu(training_state.split_bundle.test.context),
                "labels_test": tensor_to_cpu(training_state.split_bundle.test.labels),
                "mapping": training_state.split_bundle.test.mapping,
            },
        )

    summary = build_summary(
        command="deepcase-run",
        artifacts=artifacts,
        config=config,
        processed_result=training_state.processed_result,
        split_bundle=training_state.split_bundle,
        n_unique_events=len(training_state.split_bundle.train.mapping),
        n_clusters=max(0, int(training_state.clusters.max()) + 1)
        if training_state.clusters.size
        else 0,
        clusters=training_state.clusters,
        cluster_scores=training_state.cluster_scores,
        predictions=predictions,
    )
    if config.save_summary_json:
        save_json(artifacts.summary_json, summary)
    return summary
