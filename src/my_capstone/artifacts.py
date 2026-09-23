from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import current_timestamp, ensure_dir, save_json, slugify


@dataclass(frozen=True)
class RunArtifacts:
    run_id: str
    run_dir: Path
    stdout_log: Path
    run_config_json: Path
    artifact_manifest_json: Path
    summary_json: Path
    mapping_json: Path
    label_legend_json: Path
    train_sequences_save: Path
    test_sequences_save: Path
    builder_save: Path
    interpreter_save: Path
    clusters_csv: Path
    prediction_csv: Path
    predictions_pt: Path


def _grouped_paths_for_run(run_dir: Path, run_id: str) -> RunArtifacts:
    logs_dir = ensure_dir(run_dir / "logs")
    config_dir = ensure_dir(run_dir / "config")
    models_dir = ensure_dir(run_dir / "models")
    data_dir = ensure_dir(run_dir / "data")
    reports_dir = ensure_dir(run_dir / "reports")

    return RunArtifacts(
        run_id=run_id,
        run_dir=run_dir,
        stdout_log=logs_dir / "stdout.log",
        run_config_json=config_dir / "run_config.json",
        artifact_manifest_json=reports_dir / "artifact_manifest.json",
        summary_json=reports_dir / "summary.json",
        mapping_json=data_dir / "mapping.json",
        label_legend_json=data_dir / "label_legend.json",
        train_sequences_save=data_dir / "train_sequences.save",
        test_sequences_save=data_dir / "test_sequences.save",
        builder_save=models_dir / "builder.save",
        interpreter_save=models_dir / "interpreter.save",
        clusters_csv=reports_dir / "clusters.csv",
        prediction_csv=reports_dir / "prediction.csv",
        predictions_pt=reports_dir / "predictions.pt",
    )


def _flat_paths_for_run(run_dir: Path, run_id: str) -> RunArtifacts:
    return RunArtifacts(
        run_id=run_id,
        run_dir=run_dir,
        stdout_log=run_dir / "stdout.log",
        run_config_json=run_dir / "run_config.json",
        artifact_manifest_json=run_dir / "artifact_manifest.json",
        summary_json=run_dir / "summary.json",
        mapping_json=run_dir / "mapping.json",
        label_legend_json=run_dir / "label_legend.json",
        train_sequences_save=run_dir / "train_sequences.save",
        test_sequences_save=run_dir / "test_sequences.save",
        builder_save=run_dir / "builder.save",
        interpreter_save=run_dir / "interpreter.save",
        clusters_csv=run_dir / "clusters.csv",
        prediction_csv=run_dir / "prediction.csv",
        predictions_pt=run_dir / "predictions.pt",
    )


def _paths_for_existing_run(run_dir: Path, run_id: str) -> RunArtifacts:
    if (run_dir / "models").exists() or (run_dir / "reports").exists():
        return _grouped_paths_for_run(run_dir, run_id)
    return _flat_paths_for_run(run_dir, run_id)


def create_run_artifacts(
    *,
    results_root: Path,
    dataset_name: str,
    device_name: str,
    mode_name: str,
) -> RunArtifacts:
    slug = f"{slugify(dataset_name)}_{slugify(device_name)}_{slugify(mode_name)}"
    run_id = f"{current_timestamp()}_{slug}"
    run_dir = ensure_dir(results_root / "runs" / run_id)
    return _grouped_paths_for_run(run_dir, run_id)


def load_run_artifacts(results_root: Path, run_id: str) -> RunArtifacts:
    resolved_run_id = run_id
    if run_id == "latest":
        latest_path = results_root / "latest_run.txt"
        if not latest_path.exists():
            raise FileNotFoundError("找不到 results/latest_run.txt。")
        resolved_run_id = latest_path.read_text(encoding="utf-8").strip()

    run_dir = results_root / "runs" / resolved_run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"找不到對應的 run 目錄：{run_dir}")
    return _paths_for_existing_run(run_dir, resolved_run_id)


def write_latest_run(results_root: Path, run_id: str) -> Path:
    latest_path = ensure_dir(results_root) / "latest_run.txt"
    latest_path.write_text(run_id, encoding="utf-8")
    return latest_path


def write_artifact_manifest(
    *,
    artifacts: RunArtifacts,
    command: str,
    config_path: Path,
    summary: dict[str, Any],
    source_run_id: str | None = None,
) -> Path:
    entries = [
        {
            "name": "stdout_log",
            "category": "logs",
            "path": str(artifacts.stdout_log),
            "exists": artifacts.stdout_log.exists(),
            "description": "本次執行的主控台輸出紀錄。",
        },
        {
            "name": "run_config_json",
            "category": "config",
            "path": str(artifacts.run_config_json),
            "exists": artifacts.run_config_json.exists(),
            "description": "本次實際使用的設定快照。",
        },
        {
            "name": "artifact_manifest_json",
            "category": "reports",
            "path": str(artifacts.artifact_manifest_json),
            "exists": artifacts.artifact_manifest_json.exists(),
            "description": "本次 run 的索引檔，不是模型結果本身。",
        },
        {
            "name": "summary_json",
            "category": "reports",
            "path": str(artifacts.summary_json),
            "exists": artifacts.summary_json.exists(),
            "description": "最適合先看的摘要報告。",
        },
        {
            "name": "mapping_json",
            "category": "data",
            "path": str(artifacts.mapping_json),
            "exists": artifacts.mapping_json.exists(),
            "description": "DeepCASE 內部事件編號與原始事件 ID 的對照表。",
        },
        {
            "name": "label_legend_json",
            "category": "data",
            "path": str(artifacts.label_legend_json),
            "exists": artifacts.label_legend_json.exists(),
            "description": "true_label 與正分數所代表的 attack 類別對照表。",
        },
        {
            "name": "train_sequences_save",
            "category": "data",
            "path": str(artifacts.train_sequences_save),
            "exists": artifacts.train_sequences_save.exists(),
            "description": "保存下來的訓練序列資料，可供之後重用。",
        },
        {
            "name": "test_sequences_save",
            "category": "data",
            "path": str(artifacts.test_sequences_save),
            "exists": artifacts.test_sequences_save.exists(),
            "description": "保存下來的測試序列資料，可供之後重用。",
        },
        {
            "name": "builder_save",
            "category": "models",
            "path": str(artifacts.builder_save),
            "exists": artifacts.builder_save.exists(),
            "description": "已訓練的 ContextBuilder 模型。",
        },
        {
            "name": "interpreter_save",
            "category": "models",
            "path": str(artifacts.interpreter_save),
            "exists": artifacts.interpreter_save.exists(),
            "description": "已訓練的 Interpreter 與叢集狀態。",
        },
        {
            "name": "clusters_csv",
            "category": "reports",
            "path": str(artifacts.clusters_csv),
            "exists": artifacts.clusters_csv.exists(),
            "description": "訓練資料逐筆叢集與分數資訊。",
        },
        {
            "name": "prediction_csv",
            "category": "reports",
            "path": str(artifacts.prediction_csv),
            "exists": artifacts.prediction_csv.exists(),
            "description": "測試資料逐筆預測結果，最適合人工檢查。",
        },
        {
            "name": "predictions_pt",
            "category": "reports",
            "path": str(artifacts.predictions_pt),
            "exists": artifacts.predictions_pt.exists(),
            "description": "完整的 torch 預測封包，適合後續分析腳本使用。",
        },
    ]

    payload = {
        "about": "本檔是 run 目錄索引。它告訴你有哪些輸出檔、各自屬於哪一類，以及應該先看哪一些；它不是模型結果本身。",
        "run_id": artifacts.run_id,
        "command": command,
        "config_path": str(config_path),
        "source_run_id": source_run_id,
        "run_dir": str(artifacts.run_dir),
        "summary_path": str(artifacts.summary_json),
        "categories": {
            "models": "模型檔",
            "reports": "報表與分析結果",
            "data": "可重用的序列與對照資料",
            "logs": "執行紀錄",
            "config": "本次執行設定",
        },
        "artifacts": entries,
        "summary_snapshot": summary,
    }
    return save_json(artifacts.artifact_manifest_json, payload)
