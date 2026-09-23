from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .utils import PROJECT_ROOT, load_json, resolve_path

REQUIRED_KEYS = {
    "project_name",
    "dataset_name",
    "raw_data_dir",
    "labels_csv",
    "processed_csv",
    "results_root",
    "context_length",
    "timeout",
    "hidden_size",
    "epochs",
    "train_batch_size",
    "learning_rate",
    "eps",
    "min_samples",
    "threshold",
    "query_iterations",
    "query_batch_size",
    "train_split_ratio",
    "device",
    "save_train_sequences",
    "save_test_sequences",
    "save_builder",
    "save_interpreter",
    "save_predictions_pt",
    "save_prediction_csv",
    "save_summary_json",
}


@dataclass(frozen=True)
class RunConfig:
    config_path: Path
    project_name: str
    dataset_name: str
    raw_data_dir: Path
    labels_csv: Path
    processed_csv: Path
    results_root: Path
    context_length: int
    timeout: float
    hidden_size: int
    epochs: int
    train_batch_size: int
    learning_rate: float
    eps: float
    min_samples: int
    threshold: float
    query_iterations: int
    query_batch_size: int
    train_split_ratio: float
    device: str
    save_train_sequences: bool
    save_test_sequences: bool
    save_builder: bool
    save_interpreter: bool
    save_predictions_pt: bool
    save_prediction_csv: bool
    save_summary_json: bool

    @property
    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def as_dict(self) -> dict[str, Any]:
        return {
            "project_name": self.project_name,
            "dataset_name": self.dataset_name,
            "raw_data_dir": str(self.raw_data_dir),
            "labels_csv": str(self.labels_csv),
            "processed_csv": str(self.processed_csv),
            "results_root": str(self.results_root),
            "context_length": self.context_length,
            "timeout": self.timeout,
            "hidden_size": self.hidden_size,
            "epochs": self.epochs,
            "train_batch_size": self.train_batch_size,
            "learning_rate": self.learning_rate,
            "eps": self.eps,
            "min_samples": self.min_samples,
            "threshold": self.threshold,
            "query_iterations": self.query_iterations,
            "query_batch_size": self.query_batch_size,
            "train_split_ratio": self.train_split_ratio,
            "device": self.device,
            "resolved_device": self.resolved_device,
            "save_train_sequences": self.save_train_sequences,
            "save_test_sequences": self.save_test_sequences,
            "save_builder": self.save_builder,
            "save_interpreter": self.save_interpreter,
            "save_predictions_pt": self.save_predictions_pt,
            "save_prediction_csv": self.save_prediction_csv,
            "save_summary_json": self.save_summary_json,
        }


def load_run_config(path: str | Path) -> RunConfig:
    config_path = resolve_path(path)
    payload = load_json(config_path)
    missing = sorted(REQUIRED_KEYS - payload.keys())
    if missing:
        raise ValueError(f"設定檔缺少必要欄位：{', '.join(missing)}")

    train_split_ratio = float(payload["train_split_ratio"])
    if not 0 < train_split_ratio < 1:
        raise ValueError("train_split_ratio 必須介於 0 和 1 之間。")

    device = str(payload["device"])
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device 必須是 auto、cpu、cuda 其中之一。")

    return RunConfig(
        config_path=config_path,
        project_name=str(payload["project_name"]),
        dataset_name=str(payload["dataset_name"]),
        raw_data_dir=resolve_path(payload["raw_data_dir"], base_dir=PROJECT_ROOT),
        labels_csv=resolve_path(payload["labels_csv"], base_dir=PROJECT_ROOT),
        processed_csv=resolve_path(payload["processed_csv"], base_dir=PROJECT_ROOT),
        results_root=resolve_path(payload["results_root"], base_dir=PROJECT_ROOT),
        context_length=int(payload["context_length"]),
        timeout=float(payload["timeout"]),
        hidden_size=int(payload["hidden_size"]),
        epochs=int(payload["epochs"]),
        train_batch_size=int(payload["train_batch_size"]),
        learning_rate=float(payload["learning_rate"]),
        eps=float(payload["eps"]),
        min_samples=int(payload["min_samples"]),
        threshold=float(payload["threshold"]),
        query_iterations=int(payload["query_iterations"]),
        query_batch_size=int(payload["query_batch_size"]),
        train_split_ratio=train_split_ratio,
        device=device,
        save_train_sequences=bool(payload["save_train_sequences"]),
        save_test_sequences=bool(payload["save_test_sequences"]),
        save_builder=bool(payload["save_builder"]),
        save_interpreter=bool(payload["save_interpreter"]),
        save_predictions_pt=bool(payload["save_predictions_pt"]),
        save_prediction_csv=bool(payload["save_prediction_csv"]),
        save_summary_json=bool(payload["save_summary_json"]),
    )
