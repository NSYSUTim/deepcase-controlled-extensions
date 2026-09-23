from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class ComponentChange:
    component: str
    baseline: str
    research: str
    rationale: str


@dataclass(frozen=True)
class MethodBlueprint:
    method_name: str
    display_name: str
    research_gap: str
    preserved_components: list[str]
    changed_components: list[ComponentChange]
    pipeline_stages: list[str]
    future_variants: list[str]


@dataclass(frozen=True)
class ResearchConfigBase:
    config_path: Path
    method_name: str
    display_name: str
    variant_name: str
    baseline_config: Path
    baseline_run_id: str
    results_root: Path
    dataset_name: str
    device: str
    notes: str
    hidden_size: int

    @property
    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def common_dict(self) -> dict[str, Any]:
        return {
            "method_name": self.method_name,
            "display_name": self.display_name,
            "variant_name": self.variant_name,
            "baseline_config": str(self.baseline_config),
            "baseline_run_id": self.baseline_run_id,
            "results_root": str(self.results_root),
            "dataset_name": self.dataset_name,
            "device": self.device,
            "resolved_device": self.resolved_device,
            "notes": self.notes,
            "hidden_size": self.hidden_size,
        }


@dataclass(frozen=True)
class ResearchRunArtifacts:
    method_name: str
    run_id: str
    run_dir: Path
    specs_dir: Path
    models_dir: Path
    data_dir: Path
    reports_dir: Path
    logs_dir: Path
    config_dir: Path
    stdout_log: Path
    research_config_json: Path
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


@dataclass(frozen=True)
class ResearchRunSummary:
    run_id: str
    method_name: str
    command: str
    config_path: Path
    output_files: dict[str, str]
    notes: list[str] = field(default_factory=list)
