from __future__ import annotations

from pathlib import Path
from typing import Any

from my_capstone.utils import ensure_dir, save_json, slugify

from .data_types import ResearchRunArtifacts
from .utils import build_research_run_id


def create_research_artifacts(
    *,
    results_root: Path,
    method_name: str,
    variant_name: str,
) -> ResearchRunArtifacts:
    run_id = build_research_run_id(method_name, variant_name)
    method_slug = slugify(method_name)
    run_dir = ensure_dir(results_root / "research" / method_slug / "runs" / run_id)
    specs_dir = ensure_dir(run_dir / "specs")
    models_dir = ensure_dir(run_dir / "models")
    data_dir = ensure_dir(run_dir / "data")
    reports_dir = ensure_dir(run_dir / "reports")
    logs_dir = ensure_dir(run_dir / "logs")
    config_dir = ensure_dir(run_dir / "config")
    stdout_log = logs_dir / "stdout.log"
    return ResearchRunArtifacts(
        method_name=method_name,
        run_id=run_id,
        run_dir=run_dir,
        specs_dir=specs_dir,
        models_dir=models_dir,
        data_dir=data_dir,
        reports_dir=reports_dir,
        logs_dir=logs_dir,
        config_dir=config_dir,
        stdout_log=stdout_log,
        research_config_json=config_dir / "research_config.json",
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


def write_latest_research_run(results_root: Path, method_name: str, run_id: str) -> Path:
    output_path = ensure_dir(results_root / "research" / slugify(method_name)) / "latest_run.txt"
    output_path.write_text(run_id, encoding="utf-8")
    return output_path


def load_research_artifacts(
    *,
    results_root: Path,
    method_name: str,
    run_id: str,
) -> ResearchRunArtifacts:
    resolved_run_id = run_id
    method_root = results_root / "research" / slugify(method_name)
    if run_id == "latest":
        latest_path = method_root / "latest_run.txt"
        if not latest_path.exists():
            raise FileNotFoundError(f"找不到 {latest_path}")
        resolved_run_id = latest_path.read_text(encoding="utf-8").strip()

    run_dir = method_root / "runs" / resolved_run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"找不到 research run：{run_dir}")

    specs_dir = ensure_dir(run_dir / "specs")
    models_dir = ensure_dir(run_dir / "models")
    data_dir = ensure_dir(run_dir / "data")
    reports_dir = ensure_dir(run_dir / "reports")
    logs_dir = ensure_dir(run_dir / "logs")
    config_dir = ensure_dir(run_dir / "config")
    stdout_log = logs_dir / "stdout.log"
    return ResearchRunArtifacts(
        method_name=method_name,
        run_id=resolved_run_id,
        run_dir=run_dir,
        specs_dir=specs_dir,
        models_dir=models_dir,
        data_dir=data_dir,
        reports_dir=reports_dir,
        logs_dir=logs_dir,
        config_dir=config_dir,
        stdout_log=stdout_log,
        research_config_json=config_dir / "research_config.json",
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


def write_research_manifest(
    *,
    artifacts: ResearchRunArtifacts,
    command: str,
    config_path: Path,
    generated_files: dict[str, Path],
) -> Path:
    payload: dict[str, Any] = {
        "about": "研究版 run 的產物索引。",
        "method_name": artifacts.method_name,
        "run_id": artifacts.run_id,
        "command": command,
        "config_path": str(config_path),
        "run_dir": str(artifacts.run_dir),
        "categories": {
            "models": "研究版模型與 Interpreter 存檔",
            "data": "研究版 sequence、mapping、label legend",
            "reports": "研究版分群、預測、summary 與評估輸出",
            "specs": "方法藍圖、設計與比較規格",
            "logs": "執行過程記錄",
            "config": "研究版實際使用參數快照",
        },
        "generated_files": {key: str(value) for key, value in generated_files.items()},
    }
    return save_json(artifacts.artifact_manifest_json, payload)
