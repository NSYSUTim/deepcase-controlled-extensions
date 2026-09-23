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
from .config import CrossHostConfig, load_cross_host_config
from .context_builder import (
    CrossHostContextBuilder,
    build_cross_host_model_change_summary,
)
from .interpreter import build_cross_host_interpreter_plan
from .preprocessing import (
    build_cross_host_preprocessing_plan,
    build_cross_host_sequence_bundle,
)


def build_cross_host_blueprint(config: CrossHostConfig) -> MethodBlueprint:
    return MethodBlueprint(
        method_name=config.method_name,
        display_name=config.display_name,
        research_gap="原版 DeepCASE 只看 same-host context，跨主機攻擊鏈容易被切碎。",
        preserved_components=[
            "Interpreter 的 L1 distance + DBSCAN",
            "attention fingerprint clustering 主幹",
        ],
        changed_components=[
            ComponentChange(
                component="context source",
                baseline="單一主機的前序事件",
                research="本機 local context + 同 scenario 其他主機 companion context",
                rationale="讓 lateral movement 與跨主機共現事件可以直接進入 ContextBuilder。",
            ),
            ComponentChange(
                component="context builder",
                baseline="單流 GRU + attention",
                research="雙流 GRU + fusion gate + joint attention",
                rationale="分開建模 local / companion stream，再融合成最終 attention fingerprint。",
            ),
        ],
        pipeline_stages=[
            "建 local context",
            "建 cross-host companion context",
            "用雙流 GRU ContextBuilder 訓練 next-event prediction",
            "用原版 Interpreter 做 query / clustering / reject",
        ],
        future_variants=[
            "cross-host Transformer encoder",
            "HDBSCAN research variant",
        ],
    )


def build_cross_host_builder(
    *,
    config: CrossHostConfig,
    n_features: int,
) -> CrossHostContextBuilder:
    return CrossHostContextBuilder(
        input_size=n_features,
        output_size=n_features,
        hidden_size=config.hidden_size,
        max_length=config.local_context_length + config.companion_context_length,
        local_context_length=config.local_context_length,
        companion_context_length=config.companion_context_length,
    ).to(config.resolved_device)


def build_cross_host_interpreter(
    *,
    baseline_config,
    builder: CrossHostContextBuilder,
    n_features: int,
) -> Interpreter:
    return Interpreter(
        context_builder=builder,
        features=n_features,
        eps=baseline_config.eps,
        min_samples=baseline_config.min_samples,
        threshold=baseline_config.threshold,
    )


def train_cross_host_method(
    *,
    config: CrossHostConfig,
    artifacts: ResearchRunArtifacts,
) -> ResearchTrainingState:
    baseline_config, processed_result = load_baseline_runtime(config.baseline_config)
    frame = load_processed_frame(processed_result.output_path)
    print("[cross_host] 開始建立 cross-host sequence bundle")
    token_lookup: dict[str, int] = {}
    full_bundle = build_cross_host_sequence_bundle(
        frame=frame,
        config=config,
        token_to_internal=token_lookup,
    )
    save_json(
        artifacts.data_dir / "token_lookup.json",
        {
            "about": "cross_host 研究版訓練時使用的 machine|event -> internal token 映射",
            "token_scope": config.token_scope,
            "token_to_internal": token_lookup,
        },
    )
    split_bundle = split_sequence_bundle(full_bundle, baseline_config.train_split_ratio)
    save_common_research_outputs(
        artifacts=artifacts,
        split_bundle=split_bundle,
        processed_result=processed_result,
    )

    builder = build_cross_host_builder(
        config=config,
        n_features=len(split_bundle.train.mapping),
    )
    print("[cross_host] sequence bundle 建立完成，開始訓練 builder")
    fit_builder(
        baseline_config=baseline_config,
        builder=builder,
        train_bundle=split_bundle.train,
    )
    builder.save(artifacts.builder_save)

    interpreter = build_cross_host_interpreter(
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


def load_cross_host_predict_bundle(
    *,
    config: CrossHostConfig,
    source_artifacts: ResearchRunArtifacts,
    input_csv: str | None,
):
    if input_csv is None:
        return load_sequence_bundle(source_artifacts.test_sequences_save)

    frame = load_processed_frame(resolve_path(input_csv))
    token_lookup_payload = load_json(source_artifacts.data_dir / "token_lookup.json")
    token_lookup = {
        str(key): int(value)
        for key, value in token_lookup_payload["token_to_internal"].items()
    }
    return build_cross_host_sequence_bundle(
        frame=frame,
        config=config,
        token_to_internal=token_lookup,
        allow_new_tokens=False,
    )


class CrossHostResearchPipeline:
    method_name = "cross_host"

    def load_config(self, path) -> CrossHostConfig:
        return load_cross_host_config(path)

    def execute_prepare(
        self,
        *,
        config: CrossHostConfig,
        artifacts: ResearchRunArtifacts,
        source_artifacts=None,
        input_csv: str | None = None,
    ) -> dict[str, Any]:
        del source_artifacts, input_csv
        output_path = artifacts.specs_dir / "context_source_plan.json"
        save_json(output_path, build_cross_host_preprocessing_plan(config))
        return {"context_source_plan": output_path}

    def execute_train(
        self,
        *,
        config: CrossHostConfig,
        artifacts: ResearchRunArtifacts,
        source_artifacts=None,
        input_csv: str | None = None,
    ) -> dict[str, Any]:
        del source_artifacts, input_csv
        training_state = train_cross_host_method(config=config, artifacts=artifacts)
        blueprint_path = artifacts.specs_dir / "method_blueprint.json"
        model_change_path = artifacts.specs_dir / "model_change_plan.json"
        interpreter_path = artifacts.specs_dir / "interpreter_plan.json"
        save_json(blueprint_path, build_cross_host_blueprint(config))
        save_json(model_change_path, build_cross_host_model_change_summary(config))
        save_json(interpreter_path, build_cross_host_interpreter_plan(config))
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
                    "cross_host 完整版已啟用雙流 GRU builder。",
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
        config: CrossHostConfig,
        artifacts: ResearchRunArtifacts,
        source_artifacts: ResearchRunArtifacts | None = None,
        input_csv: str | None = None,
    ) -> dict[str, Any]:
        if source_artifacts is None:
            raise ValueError("cross_host predict 需要 source_artifacts。")

        baseline_config, processed_result = load_baseline_runtime(config.baseline_config)
        builder = CrossHostContextBuilder.load(
            source_artifacts.builder_save,
            device=config.resolved_device,
        )
        interpreter = Interpreter.load(
            source_artifacts.interpreter_save,
            context_builder=builder,
        )
        prediction_bundle = load_cross_host_predict_bundle(
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
                focus="cross_host_soc_correlation",
                extra_metrics=["cross_host_subgroup_recall"],
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
                notes=[
                    f"cross_host predict 來源 run_id = {source_artifacts.run_id}",
                ],
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
        config: CrossHostConfig,
        artifacts: ResearchRunArtifacts,
        source_artifacts=None,
        input_csv: str | None = None,
    ) -> dict[str, Any]:
        del source_artifacts, input_csv
        training_state = train_cross_host_method(config=config, artifacts=artifacts)
        blueprint_path = artifacts.specs_dir / "method_blueprint.json"
        model_change_path = artifacts.specs_dir / "model_change_plan.json"
        interpreter_path = artifacts.specs_dir / "interpreter_plan.json"
        save_json(blueprint_path, build_cross_host_blueprint(config))
        save_json(model_change_path, build_cross_host_model_change_summary(config))
        save_json(interpreter_path, build_cross_host_interpreter_plan(config))
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
                focus="cross_host_soc_correlation",
                extra_metrics=["cross_host_subgroup_recall"],
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
                notes=[
                    "cross_host 完整版：雙流 GRU builder + 原版 Interpreter。",
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
            "prediction_csv": artifacts.prediction_csv,
            "predictions_pt": artifacts.predictions_pt,
            "summary_json": artifacts.summary_json,
            "evaluation_plan": evaluation_path,
        }
