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

from my_capstone.config import load_run_config


class ConfigTests(unittest.TestCase):
    def test_load_run_config_resolves_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "project_name": "My_Capstone",
                        "dataset_name": "tiny",
                        "raw_data_dir": "data/raw/ait_ads",
                        "labels_csv": "data/raw/labels.csv",
                        "processed_csv": "data/processed/ait_ads_deepcase.csv",
                        "results_root": "results",
                        "context_length": 10,
                        "timeout": 86400,
                        "hidden_size": 128,
                        "epochs": 10,
                        "train_batch_size": 128,
                        "learning_rate": 0.01,
                        "eps": 0.1,
                        "min_samples": 5,
                        "threshold": 0.2,
                        "query_iterations": 100,
                        "query_batch_size": 1024,
                        "train_split_ratio": 0.2,
                        "device": "auto",
                        "save_train_sequences": True,
                        "save_test_sequences": True,
                        "save_builder": True,
                        "save_interpreter": True,
                        "save_predictions_pt": True,
                        "save_prediction_csv": True,
                        "save_summary_json": True,
                    }
                ),
                encoding="utf-8",
            )

            config = load_run_config(config_path)

            self.assertEqual(config.device, "auto")
            self.assertEqual(config.resolved_device, "cpu")
            self.assertEqual(config.train_split_ratio, 0.2)


if __name__ == "__main__":
    unittest.main()
