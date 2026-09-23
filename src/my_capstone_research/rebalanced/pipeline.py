from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import torch

from my_capstone.evaluation import rebalanced_prediction_policy
from my_capstone.pipeline import (
    load_processed_sequences,
    load_sequence_bundle,
    save_mapping_json,
    save_sequence_bundle,
)
from my_capstone.scoring import TRAIN_NO_ATTACK_LABEL
from my_capstone.utils import resolve_path, save_json

from ..common.data_types import ComponentChange, MethodBlueprint, ResearchRunArtifacts
from ..common.evaluation import build_default_evaluation_plan
from ..common.execution import (
    REBALANCED_REJECT_SCORE,
    ResearchTrainingState,
    binary_true_labels,
    build_research_summary,
    cluster_with_interpreter,
    load_baseline_runtime,
    predict_with_interpreter,
    rebalanced_score_meaning,
    rebalanced_score_meaning_table,
    save_cluster_report,
    save_common_research_outputs,
    save_prediction_report,
    save_predictions_tensor,
    split_sequence_bundle,
)
from .calibration import build_calibration_plan
from .config import RebalancedConfig, load_rebalanced_config
from .context_builder import (
    RebalancedContextBuilder,
    build_rebalanced_model_change_summary,
)
from .interpreter import RebalancedInterpreter, build_rebalanced_interpreter_plan


def build_rebalanced_blueprint(config: RebalancedConfig) -> MethodBlueprint:
    return MethodBlueprint(
        method_name=config.method_name,
        display_name=config.display_name,
        research_gap="原版 DeepCASE 在資料不平衡下，少數 Incident 類別容易被 majority cluster 稀釋。",
        preserved_components=[
            "GRU backbone",
            "attention fingerprint",
            "L1 distance + DBSCAN clustering",
        ],
        changed_components=[
            ComponentChange(
                component="training objective",
                baseline="only next-event prediction",
                research="next-event + supervised triage joint objective",
                rationale="讓 representation learning 直接接觸 Incident / Non-Incident 訊號。",
            ),
            ComponentChange(
                component="prediction decision",
                baseline="cluster hard score + reject code",
                research="cluster prior + confidence + distance 的 calibrated posterior",
                rationale="把 minority class triage 與 reject 納入正式決策流程。",
            ),
        ],
        pipeline_stages=[
            "joint-objective GRU ContextBuilder 訓練",
            "原版 DBSCAN clustering",
            "估計 cluster prior",
            "用 calibrated posterior 做 incident / benign / reject triage",
        ],
        future_variants=[
            "Transformer backbone",
            "learned distance metric",
        ],
    )


def build_rebalanced_builder(
    *,
    config: RebalancedConfig,
    n_features: int,
) -> RebalancedContextBuilder:
    return RebalancedContextBuilder(
        input_size=n_features,
        output_size=n_features,
        hidden_size=config.hidden_size,
        max_length=config.context_length,
        loss_type=config.loss_type,
        focal_gamma=config.focal_gamma,
        positive_class_weight=config.positive_class_weight,
        negative_class_weight=config.negative_class_weight,
    ).to(config.resolved_device)


def build_rebalanced_interpreter(
    *,
    baseline_config,
    config: RebalancedConfig,
    builder: RebalancedContextBuilder,
    n_features: int,
) -> RebalancedInterpreter:
    return RebalancedInterpreter(
        context_builder=builder,
        features=n_features,
        eps=baseline_config.eps,
        min_samples=baseline_config.min_samples,
        threshold=baseline_config.threshold,
        low_threshold=config.reject_low_threshold,
        high_threshold=config.reject_high_threshold,
        distance_scale=baseline_config.eps,
    )


def build_rebalanced_cluster_statistics(
    *,
    clusters: np.ndarray,
    labels: np.ndarray,
    positive_weight: float,
    negative_weight: float,
    low_threshold: float,
    high_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    hard_scores = np.full(labels.shape[0], REBALANCED_REJECT_SCORE, dtype=int)
    priors = np.full(labels.shape[0], 0.5, dtype=float)
    binary_labels = np.where(labels == TRAIN_NO_ATTACK_LABEL, 0, 1)

    for cluster in np.unique(clusters):
        if cluster == -1:
            continue
        indices = np.flatnonzero(clusters == cluster)
        cluster_binary = binary_labels[indices]
        positive = int(cluster_binary.sum())
        negative = int(cluster_binary.shape[0] - positive)
        numerator = positive_weight * positive
        denominator = numerator + negative_weight * negative
        posterior = 0.0 if denominator == 0 else float(numerator / denominator)
        priors[indices] = posterior

        if posterior >= high_threshold:
            hard_scores[indices] = 1
        elif posterior <= low_threshold:
            hard_scores[indices] = 0
        else:
            hard_scores[indices] = REBALANCED_REJECT_SCORE
    return hard_scores, priors


def train_rebalanced_method(
    *,
    config: RebalancedConfig,
    artifacts: ResearchRunArtifacts,
) -> ResearchTrainingState:
    baseline_config, processed_result = load_baseline_runtime(config.baseline_config)
    runtime_config = replace(baseline_config, context_length=config.context_length)
    full_bundle = load_processed_sequences(runtime_config, processed_result.output_path)
    split_bundle = split_sequence_bundle(full_bundle, runtime_config.train_split_ratio)
    save_common_research_outputs(
        artifacts=artifacts,
        split_bundle=split_bundle,
        processed_result=processed_result,
    )

    builder = build_rebalanced_builder(
        config=config,
        n_features=len(split_bundle.train.mapping),
    )
    triage_targets = np.asarray(
        binary_true_labels(split_bundle.train.labels.detach().cpu().numpy()),
        dtype=np.float32,
    )
    builder.fit(
        X=split_bundle.train.context.to(runtime_config.resolved_device),
        y=split_bundle.train.events.reshape(-1, 1).to(runtime_config.resolved_device),
        epochs=runtime_config.epochs,
        batch_size=runtime_config.train_batch_size,
        learning_rate=runtime_config.learning_rate,
        verbose=True,
        triage_targets=torch.as_tensor(triage_targets, dtype=torch.float32, device=runtime_config.resolved_device),
    )
    builder.save(artifacts.builder_save)

    interpreter = build_rebalanced_interpreter(
        baseline_config=runtime_config,
        config=config,
        builder=builder,
        n_features=len(split_bundle.train.mapping),
    )
    clusters = cluster_with_interpreter(
        baseline_config=runtime_config,
        interpreter=interpreter,
        train_bundle=split_bundle.train,
    )
    train_labels = split_bundle.train.labels.detach().cpu().numpy()
    cluster_scores, cluster_priors = build_rebalanced_cluster_statistics(
        clusters=clusters,
        labels=train_labels,
        positive_weight=config.positive_class_weight,
        negative_weight=config.negative_class_weight,
        low_threshold=config.reject_low_threshold,
        high_threshold=config.reject_high_threshold,
    )
    interpreter.score(cluster_scores, verbose=True)
    interpreter.assign_cluster_priors(cluster_priors)
    interpreter.save(artifacts.interpreter_save)

    save_cluster_report(
        path=artifacts.clusters_csv,
        bundle=split_bundle.train,
        clusters=clusters,
        cluster_scores=cluster_scores,
        score_meaning_fn=rebalanced_score_meaning,
        extra_columns={
            "binary_true_label": binary_true_labels(train_labels),
            "cluster_prior": cluster_priors.tolist(),
        },
    )
    return ResearchTrainingState(
        baseline_config=runtime_config,
        processed_result=processed_result,
        split_bundle=split_bundle,
        builder=builder,
        interpreter=interpreter,
        clusters=clusters,
        cluster_scores=cluster_scores,
        score_meaning_fn=rebalanced_score_meaning,
    )


class RebalancedResearchPipeline:
    method_name = "rebalanced"

    def load_config(self, path) -> RebalancedConfig:
        return load_rebalanced_config(path)

    def execute_prepare(
        self,
        *,
        config: RebalancedConfig,
        artifacts: ResearchRunArtifacts,
        source_artifacts=None,
        input_csv: str | None = None,
    ) -> dict[str, Any]:
        del source_artifacts, input_csv
        output_path = artifacts.specs_dir / "objective_plan.json"
        save_json(
            output_path,
            {
                "goal": "joint objective + calibrated posterior triage",
                "loss_type": config.loss_type,
                "positive_class_weight": config.positive_class_weight,
                "negative_class_weight": config.negative_class_weight,
                "focal_gamma": config.focal_gamma,
            },
        )
        return {"objective_plan": output_path}

    def execute_train(
        self,
        *,
        config: RebalancedConfig,
        artifacts: ResearchRunArtifacts,
        source_artifacts=None,
        input_csv: str | None = None,
    ) -> dict[str, Any]:
        del source_artifacts, input_csv
        training_state = train_rebalanced_method(config=config, artifacts=artifacts)
        blueprint_path = artifacts.specs_dir / "method_blueprint.json"
        model_change_path = artifacts.specs_dir / "model_change_plan.json"
        interpreter_path = artifacts.specs_dir / "interpreter_plan.json"
        calibration_path = artifacts.specs_dir / "calibration_plan.json"
        save_json(blueprint_path, build_rebalanced_blueprint(config))
        save_json(model_change_path, build_rebalanced_model_change_summary(config))
        save_json(interpreter_path, build_rebalanced_interpreter_plan(config))
        save_json(calibration_path, build_calibration_plan(config))
        save_json(
            artifacts.summary_json,
            build_research_summary(
                artifacts=artifacts,
                method_name=config.method_name,
                display_name=config.display_name,
                baseline_config=training_state.baseline_config,
                split_bundle=training_state.split_bundle,
                processed_result=training_state.processed_result,
                n_clusters=max(0, int(training_state.clusters.max()) + 1)
                if training_state.clusters.size
                else 0,
                clusters=training_state.clusters,
                cluster_scores=training_state.cluster_scores,
                predictions=np.asarray([], dtype=int),
                prediction_policy=rebalanced_prediction_policy(REBALANCED_REJECT_SCORE),
                score_meaning_table=rebalanced_score_meaning_table(),
                notes=[
                    "rebalanced 完整版已啟用 joint-objective builder。",
                    "prediction 改用 calibrated posterior triage。",
                ],
            ),
        )
        return {
            "method_blueprint": blueprint_path,
            "model_change_plan": model_change_path,
            "interpreter_plan": interpreter_path,
            "calibration_plan": calibration_path,
            "builder_save": artifacts.builder_save,
            "interpreter_save": artifacts.interpreter_save,
            "clusters_csv": artifacts.clusters_csv,
            "summary_json": artifacts.summary_json,
        }

    def execute_predict(
        self,
        *,
        config: RebalancedConfig,
        artifacts: ResearchRunArtifacts,
        source_artifacts: ResearchRunArtifacts | None = None,
        input_csv: str | None = None,
    ) -> dict[str, Any]:
        if source_artifacts is None:
            raise ValueError("rebalanced predict 需要 source_artifacts。")

        baseline_config, processed_result = load_baseline_runtime(config.baseline_config)
        runtime_config = replace(baseline_config, context_length=config.context_length)
        builder = RebalancedContextBuilder.load(
            source_artifacts.builder_save,
            device=config.resolved_device,
        )
        interpreter = RebalancedInterpreter.load(
            source_artifacts.interpreter_save,
            context_builder=builder,
        )
        if input_csv is None:
            prediction_bundle = load_sequence_bundle(source_artifacts.test_sequences_save)
        else:
            raise ValueError(
                "rebalanced 研究版目前不支援直接用 --input-csv 重建 sequence bundle。"
                "原因是 baseline preprocessor 會重新編 internal event id，"
                "可能和已訓練 builder/interpreter 的編碼不一致。"
                "請先使用既有 run 的 test_sequences.save 做 predict，"
                "或在需要新資料時重新 train/run。"
            )
        save_sequence_bundle(artifacts.test_sequences_save, prediction_bundle)
        save_mapping_json(artifacts.mapping_json, prediction_bundle.mapping)
        if source_artifacts.label_legend_json.exists():
            artifacts.label_legend_json.write_text(
                source_artifacts.label_legend_json.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

        predictions = predict_with_interpreter(
            baseline_config=runtime_config,
            interpreter=interpreter,
            prediction_bundle=prediction_bundle,
        )
        binary_test_labels = binary_true_labels(
            prediction_bundle.labels.detach().cpu().numpy()
        )
        save_prediction_report(
            path=artifacts.prediction_csv,
            bundle=prediction_bundle,
            predictions=predictions,
            score_meaning_fn=rebalanced_score_meaning,
            extra_columns={"binary_true_label": binary_test_labels},
        )
        save_predictions_tensor(
            path=artifacts.predictions_pt,
            bundle=prediction_bundle,
            predictions=predictions,
            extras={
                "method_name": config.method_name,
                "binary_true_label": np.asarray(binary_test_labels, dtype=np.int64),
            },
        )
        evaluation_path = artifacts.reports_dir / "evaluation_plan.json"
        save_json(
            evaluation_path,
            build_default_evaluation_plan(
                focus="imbalance_aware_soc_triage",
                extra_metrics=["incident_fnr", "rejected_incident_ratio"],
            ),
        )
        save_json(
            artifacts.summary_json,
            build_research_summary(
                artifacts=artifacts,
                method_name=config.method_name,
                display_name=config.display_name,
                baseline_config=runtime_config,
                split_bundle=type("SplitProxy", (), {"train": prediction_bundle, "test": prediction_bundle})(),
                processed_result=processed_result,
                n_clusters=None,
                clusters=None,
                cluster_scores=np.asarray([], dtype=int),
                predictions=predictions,
                prediction_policy=rebalanced_prediction_policy(REBALANCED_REJECT_SCORE),
                score_meaning_table=rebalanced_score_meaning_table(),
                notes=[f"rebalanced predict 來源 run_id = {source_artifacts.run_id}"],
            ),
        )
        return {
            "prediction_csv": artifacts.prediction_csv,
            "predictions_pt": artifacts.predictions_pt,
            "summary_json": artifacts.summary_json,
            "evaluation_plan": evaluation_path,
        }

    def execute_run(
        self,
        *,
        config: RebalancedConfig,
        artifacts: ResearchRunArtifacts,
        source_artifacts=None,
        input_csv: str | None = None,
    ) -> dict[str, Any]:
        del source_artifacts, input_csv
        training_state = train_rebalanced_method(config=config, artifacts=artifacts)
        blueprint_path = artifacts.specs_dir / "method_blueprint.json"
        model_change_path = artifacts.specs_dir / "model_change_plan.json"
        interpreter_path = artifacts.specs_dir / "interpreter_plan.json"
        calibration_path = artifacts.specs_dir / "calibration_plan.json"
        save_json(blueprint_path, build_rebalanced_blueprint(config))
        save_json(model_change_path, build_rebalanced_model_change_summary(config))
        save_json(interpreter_path, build_rebalanced_interpreter_plan(config))
        save_json(calibration_path, build_calibration_plan(config))

        predictions = predict_with_interpreter(
            baseline_config=training_state.baseline_config,
            interpreter=training_state.interpreter,
            prediction_bundle=training_state.split_bundle.test,
        )
        binary_test_labels = binary_true_labels(
            training_state.split_bundle.test.labels.detach().cpu().numpy()
        )
        save_prediction_report(
            path=artifacts.prediction_csv,
            bundle=training_state.split_bundle.test,
            predictions=predictions,
            score_meaning_fn=rebalanced_score_meaning,
            extra_columns={"binary_true_label": binary_test_labels},
        )
        save_predictions_tensor(
            path=artifacts.predictions_pt,
            bundle=training_state.split_bundle.test,
            predictions=predictions,
            extras={
                "method_name": config.method_name,
                "binary_true_label": np.asarray(binary_test_labels, dtype=np.int64),
            },
        )
        evaluation_path = artifacts.reports_dir / "evaluation_plan.json"
        save_json(
            evaluation_path,
            build_default_evaluation_plan(
                focus="imbalance_aware_soc_triage",
                extra_metrics=["incident_fnr", "rejected_incident_ratio"],
            ),
        )
        save_json(
            artifacts.summary_json,
            build_research_summary(
                artifacts=artifacts,
                method_name=config.method_name,
                display_name=config.display_name,
                baseline_config=training_state.baseline_config,
                split_bundle=training_state.split_bundle,
                processed_result=training_state.processed_result,
                n_clusters=max(0, int(training_state.clusters.max()) + 1)
                if training_state.clusters.size
                else 0,
                clusters=training_state.clusters,
                cluster_scores=training_state.cluster_scores,
                predictions=predictions,
                prediction_policy=rebalanced_prediction_policy(REBALANCED_REJECT_SCORE),
                score_meaning_table=rebalanced_score_meaning_table(),
                notes=["rebalanced 完整版：joint-objective builder + calibrated posterior interpreter。"],
            ),
        )
        return {
            "method_blueprint": blueprint_path,
            "model_change_plan": model_change_path,
            "interpreter_plan": interpreter_path,
            "calibration_plan": calibration_path,
            "builder_save": artifacts.builder_save,
            "interpreter_save": artifacts.interpreter_save,
            "clusters_csv": artifacts.clusters_csv,
            "prediction_csv": artifacts.prediction_csv,
            "predictions_pt": artifacts.predictions_pt,
            "summary_json": artifacts.summary_json,
            "evaluation_plan": evaluation_path,
        }
