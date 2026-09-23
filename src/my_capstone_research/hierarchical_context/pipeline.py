from __future__ import annotations

from typing import Any

import numpy as np
from deepcase.interpreter import Interpreter

from my_capstone.pipeline import load_sequence_bundle, save_mapping_json, save_sequence_bundle
from my_capstone.scoring import build_cluster_scores
from my_capstone.utils import load_json, resolve_path, save_json

from ..common.data_types import ComponentChange, MethodBlueprint, ResearchRunArtifacts
from ..common.evaluation import build_default_evaluation_plan
from ..common.execution import (
    ResearchTrainingState,
    baseline_score_meaning,
    baseline_score_meaning_table,
    build_research_summary,
    cluster_with_interpreter,
    fit_builder,
    load_baseline_runtime,
    load_processed_frame,
    predict_with_interpreter,
    save_cluster_report,
    save_common_research_outputs,
    save_prediction_report,
    save_predictions_tensor,
    split_sequence_bundle,
)
from .config import HierarchicalContextConfig, load_hierarchical_context_config
from .context_builder import (
    HierarchicalContextBuilder,
    build_hierarchical_model_change_summary,
)
from .interpreter import build_hierarchical_interpreter_plan
from .preprocessing import (
    build_hierarchical_preprocessing_plan,
    build_hierarchical_sequence_bundle,
)


def build_hierarchical_blueprint(
    config: HierarchicalContextConfig,
) -> MethodBlueprint:
    return MethodBlueprint(
        method_name=config.method_name,
        display_name=config.display_name,
        research_gap="原版 DeepCASE 的固定短窗 context 不利於 long-horizon 事件關聯。",
        preserved_components=[
            "Interpreter 的 L1 distance + DBSCAN",
            "attention fingerprint clustering 主幹",
        ],
        changed_components=[
            ComponentChange(
                component="context organization",
                baseline="固定長度短窗 context",
                research="short-term context + long-term memory",
                rationale="把短期關鍵事件與長期前導事件分開建模。",
            ),
            ComponentChange(
                component="context builder",
                baseline="單流 GRU + attention",
                research="雙分支 GRU + hierarchical fusion",
                rationale="讓模型在內部明確區分 short-term 與 long-term 訊號。",
            ),
        ],
        pipeline_stages=[
            "建 short-term context",
            "選 long-term landmarks",
            "用 hierarchical GRU ContextBuilder 訓練 next-event prediction",
            "用原版 Interpreter 做 query / clustering / reject",
        ],
        future_variants=[
            "hierarchical Transformer encoder",
            "memory-aware landmark selector",
        ],
    )


def build_hierarchical_builder(
    *,
    config: HierarchicalContextConfig,
    n_features: int,
) -> HierarchicalContextBuilder:
    return HierarchicalContextBuilder(
        input_size=n_features,
        output_size=n_features,
        hidden_size=config.hidden_size,
        max_length=config.short_context_length + config.long_memory_length,
        short_context_length=config.short_context_length,
        long_memory_length=config.long_memory_length,
    ).to(config.resolved_device)


def build_hierarchical_interpreter(
    *,
    baseline_config,
    builder: HierarchicalContextBuilder,
    n_features: int,
) -> Interpreter:
    return Interpreter(
        context_builder=builder,
        features=n_features,
        eps=baseline_config.eps,
        min_samples=baseline_config.min_samples,
        threshold=baseline_config.threshold,
    )


def train_hierarchical_method(
    *,
    config: HierarchicalContextConfig,
    artifacts: ResearchRunArtifacts,
) -> ResearchTrainingState:
    baseline_config, processed_result = load_baseline_runtime(config.baseline_config)
    frame = load_processed_frame(processed_result.output_path)
    print("[hierarchical_context] 開始建立 hierarchical sequence bundle")
    token_lookup: dict[int, int] = {}
    full_bundle = build_hierarchical_sequence_bundle(
        frame=frame,
        config=config,
        token_to_internal=token_lookup,
    )
    save_json(
        artifacts.data_dir / "token_lookup.json",
        {
            "about": "hierarchical_context 研究版訓練時使用的 original event -> internal token 映射",
            "token_to_internal": {
                str(key): int(value) for key, value in token_lookup.items()
            },
        },
    )
    split_bundle = split_sequence_bundle(full_bundle, baseline_config.train_split_ratio)
    save_common_research_outputs(
        artifacts=artifacts,
        split_bundle=split_bundle,
        processed_result=processed_result,
    )

    builder = build_hierarchical_builder(
        config=config,
        n_features=len(split_bundle.train.mapping),
    )
    print("[hierarchical_context] sequence bundle 建立完成，開始訓練 builder")
    fit_builder(
        baseline_config=baseline_config,
        builder=builder,
        train_bundle=split_bundle.train,
    )
    builder.save(artifacts.builder_save)

    interpreter = build_hierarchical_interpreter(
        baseline_config=baseline_config,
        builder=builder,
        n_features=len(split_bundle.train.mapping),
    )
    clusters = cluster_with_interpreter(
        baseline_config=baseline_config,
        interpreter=interpreter,
        train_bundle=split_bundle.train,
    )
    train_labels = split_bundle.train.labels.detach().cpu().numpy()
    cluster_scores = build_cluster_scores(
        clusters=clusters,
        labels=train_labels,
        strategy="max",
    )
    interpreter.score(cluster_scores, verbose=True)
    interpreter.save(artifacts.interpreter_save)

    save_cluster_report(
        path=artifacts.clusters_csv,
        bundle=split_bundle.train,
        clusters=clusters,
        cluster_scores=cluster_scores,
        score_meaning_fn=baseline_score_meaning,
    )
    return ResearchTrainingState(
        baseline_config=baseline_config,
        processed_result=processed_result,
        split_bundle=split_bundle,
        builder=builder,
        interpreter=interpreter,
        clusters=clusters,
        cluster_scores=cluster_scores,
        score_meaning_fn=baseline_score_meaning,
    )


def load_hierarchical_predict_bundle(
    *,
    config: HierarchicalContextConfig,
    source_artifacts: ResearchRunArtifacts,
    input_csv: str | None,
):
    if input_csv is None:
        return load_sequence_bundle(source_artifacts.test_sequences_save)
    frame = load_processed_frame(resolve_path(input_csv))
    token_lookup_payload = load_json(source_artifacts.data_dir / "token_lookup.json")
    token_lookup = {
        int(key): int(value)
        for key, value in token_lookup_payload["token_to_internal"].items()
    }
    return build_hierarchical_sequence_bundle(
        frame=frame,
        config=config,
        token_to_internal=token_lookup,
        allow_new_tokens=False,
    )


class HierarchicalContextResearchPipeline:
    method_name = "hierarchical_context"

    def load_config(self, path) -> HierarchicalContextConfig:
        return load_hierarchical_context_config(path)

    def execute_prepare(
        self,
        *,
        config: HierarchicalContextConfig,
        artifacts: ResearchRunArtifacts,
        source_artifacts=None,
        input_csv: str | None = None,
    ) -> dict[str, Any]:
        del source_artifacts, input_csv
        output_path = artifacts.specs_dir / "memory_plan.json"
        save_json(output_path, build_hierarchical_preprocessing_plan(config))
        return {"memory_plan": output_path}

    def execute_train(
        self,
        *,
        config: HierarchicalContextConfig,
        artifacts: ResearchRunArtifacts,
        source_artifacts=None,
        input_csv: str | None = None,
    ) -> dict[str, Any]:
        del source_artifacts, input_csv
        training_state = train_hierarchical_method(config=config, artifacts=artifacts)
        blueprint_path = artifacts.specs_dir / "method_blueprint.json"
        model_change_path = artifacts.specs_dir / "model_change_plan.json"
        interpreter_path = artifacts.specs_dir / "interpreter_plan.json"
        save_json(blueprint_path, build_hierarchical_blueprint(config))
        save_json(model_change_path, build_hierarchical_model_change_summary(config))
        save_json(interpreter_path, build_hierarchical_interpreter_plan(config))
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
                score_meaning_table=baseline_score_meaning_table(),
                notes=[
                    "hierarchical_context 完整版已啟用雙分支 GRU builder。",
                    "Interpreter 仍維持原版 L1 / DBSCAN，方便和 baseline 公平比較。",
                ],
            ),
        )
        return {
            "method_blueprint": blueprint_path,
            "model_change_plan": model_change_path,
            "interpreter_plan": interpreter_path,
            "builder_save": artifacts.builder_save,
            "interpreter_save": artifacts.interpreter_save,
            "clusters_csv": artifacts.clusters_csv,
            "summary_json": artifacts.summary_json,
        }

    def execute_predict(
        self,
        *,
        config: HierarchicalContextConfig,
        artifacts: ResearchRunArtifacts,
        source_artifacts: ResearchRunArtifacts | None = None,
        input_csv: str | None = None,
    ) -> dict[str, Any]:
        if source_artifacts is None:
            raise ValueError("hierarchical_context predict 需要 source_artifacts。")

        baseline_config, processed_result = load_baseline_runtime(config.baseline_config)
        builder = HierarchicalContextBuilder.load(
            source_artifacts.builder_save,
            device=config.resolved_device,
        )
        interpreter = Interpreter.load(
            source_artifacts.interpreter_save,
            context_builder=builder,
        )
        prediction_bundle = load_hierarchical_predict_bundle(
            config=config,
            source_artifacts=source_artifacts,
            input_csv=input_csv,
        )
        save_sequence_bundle(artifacts.test_sequences_save, prediction_bundle)
        save_mapping_json(artifacts.mapping_json, prediction_bundle.mapping)
        if source_artifacts.label_legend_json.exists():
            artifacts.label_legend_json.write_text(
                source_artifacts.label_legend_json.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

        predictions = predict_with_interpreter(
            baseline_config=baseline_config,
            interpreter=interpreter,
            prediction_bundle=prediction_bundle,
        )
        save_prediction_report(
            path=artifacts.prediction_csv,
            bundle=prediction_bundle,
            predictions=predictions,
            score_meaning_fn=baseline_score_meaning,
        )
        save_predictions_tensor(
            path=artifacts.predictions_pt,
            bundle=prediction_bundle,
            predictions=predictions,
            extras={"method_name": config.method_name},
        )
        evaluation_path = artifacts.reports_dir / "evaluation_plan.json"
        save_json(
            evaluation_path,
            build_default_evaluation_plan(
                focus="long_horizon_event_correlation",
                extra_metrics=["slow_attack_subgroup_recall"],
            ),
        )
        save_json(
            artifacts.summary_json,
            build_research_summary(
                artifacts=artifacts,
                method_name=config.method_name,
                display_name=config.display_name,
                baseline_config=baseline_config,
                split_bundle=type("SplitProxy", (), {"train": prediction_bundle, "test": prediction_bundle})(),
                processed_result=processed_result,
                n_clusters=None,
                clusters=None,
                cluster_scores=np.asarray([], dtype=int),
                predictions=predictions,
                score_meaning_table=baseline_score_meaning_table(),
                notes=[f"hierarchical_context predict 來源 run_id = {source_artifacts.run_id}"],
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
        config: HierarchicalContextConfig,
        artifacts: ResearchRunArtifacts,
        source_artifacts=None,
        input_csv: str | None = None,
    ) -> dict[str, Any]:
        del source_artifacts, input_csv
        training_state = train_hierarchical_method(config=config, artifacts=artifacts)
        blueprint_path = artifacts.specs_dir / "method_blueprint.json"
        model_change_path = artifacts.specs_dir / "model_change_plan.json"
        interpreter_path = artifacts.specs_dir / "interpreter_plan.json"
        save_json(blueprint_path, build_hierarchical_blueprint(config))
        save_json(model_change_path, build_hierarchical_model_change_summary(config))
        save_json(interpreter_path, build_hierarchical_interpreter_plan(config))

        predictions = predict_with_interpreter(
            baseline_config=training_state.baseline_config,
            interpreter=training_state.interpreter,
            prediction_bundle=training_state.split_bundle.test,
        )
        save_prediction_report(
            path=artifacts.prediction_csv,
            bundle=training_state.split_bundle.test,
            predictions=predictions,
            score_meaning_fn=baseline_score_meaning,
        )
        save_predictions_tensor(
            path=artifacts.predictions_pt,
            bundle=training_state.split_bundle.test,
            predictions=predictions,
            extras={"method_name": config.method_name},
        )
        evaluation_path = artifacts.reports_dir / "evaluation_plan.json"
        save_json(
            evaluation_path,
            build_default_evaluation_plan(
                focus="long_horizon_event_correlation",
                extra_metrics=["slow_attack_subgroup_recall"],
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
                score_meaning_table=baseline_score_meaning_table(),
                notes=["hierarchical_context 完整版：雙分支 GRU builder + 原版 Interpreter。"],
            ),
        )
        return {
            "method_blueprint": blueprint_path,
            "model_change_plan": model_change_path,
            "interpreter_plan": interpreter_path,
            "builder_save": artifacts.builder_save,
            "interpreter_save": artifacts.interpreter_save,
            "clusters_csv": artifacts.clusters_csv,
            "prediction_csv": artifacts.prediction_csv,
            "predictions_pt": artifacts.predictions_pt,
            "summary_json": artifacts.summary_json,
            "evaluation_plan": evaluation_path,
        }
