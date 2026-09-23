from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from my_capstone.artifacts import create_run_artifacts, load_run_artifacts
from my_capstone.config import load_run_config
from my_capstone.pipeline import execute_deepcase_predict, execute_deepcase_run


def build_tiny_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "tiny_config.json"
    payload = {
        "project_name": "My_Capstone",
        "dataset_name": "tiny-demo",
        "raw_data_dir": str(PROJECT_ROOT / "tests" / "fixtures" / "raw" / "ait_ads"),
        "labels_csv": str(PROJECT_ROOT / "tests" / "fixtures" / "raw" / "labels.csv"),
        "processed_csv": str(tmp_path / "tiny_processed.csv"),
        "results_root": str(tmp_path / "results"),
        "context_length": 2,
        "timeout": 86400,
        "hidden_size": 16,
        "epochs": 1,
        "train_batch_size": 2,
        "learning_rate": 0.01,
        "eps": 0.5,
        "min_samples": 1,
        "threshold": 0.0,
        "query_iterations": 1,
        "query_batch_size": 4,
        "train_split_ratio": 0.5,
        "device": "cpu",
        "save_train_sequences": True,
        "save_test_sequences": True,
        "save_builder": True,
        "save_interpreter": True,
        "save_predictions_pt": True,
        "save_prediction_csv": True,
        "save_summary_json": True,
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path


class PipelineSmokeTests(unittest.TestCase):
    def test_run_and_predict_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = build_tiny_config(tmp_path)
            config = load_run_config(config_path)

            train_artifacts = create_run_artifacts(
                results_root=config.results_root,
                dataset_name=config.dataset_name,
                device_name=config.resolved_device,
                mode_name="full",
            )
            execute_deepcase_run(
                config=config,
                artifacts=train_artifacts,
                force_prepare=True,
            )

            self.assertTrue(train_artifacts.builder_save.exists())
            self.assertTrue(train_artifacts.interpreter_save.exists())
            self.assertTrue(train_artifacts.summary_json.exists())
            self.assertTrue(train_artifacts.prediction_csv.exists())
            self.assertTrue(train_artifacts.predictions_pt.exists())

            predict_artifacts = create_run_artifacts(
                results_root=config.results_root,
                dataset_name=config.dataset_name,
                device_name=config.resolved_device,
                mode_name="predict",
            )
            source_artifacts = load_run_artifacts(
                config.results_root,
                train_artifacts.run_id,
            )
            execute_deepcase_predict(
                config=config,
                artifacts=predict_artifacts,
                source_artifacts=source_artifacts,
            )

            self.assertTrue(predict_artifacts.prediction_csv.exists())
            self.assertTrue(predict_artifacts.predictions_pt.exists())


if __name__ == "__main__":
    unittest.main()
