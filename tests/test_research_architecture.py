from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from main_research import main
from my_capstone_research.cross_host.config import CrossHostConfig
from my_capstone_research.cross_host.preprocessing import build_cross_host_sequence_bundle


def write_config(tmp_path: Path, payload: dict) -> Path:
    config_path = tmp_path / f"{payload['method_name']}.json"
    config_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return config_path


def write_baseline_config(tmp_path: Path) -> Path:
    fixtures = PROJECT_ROOT / "tests" / "fixtures" / "raw"
    payload = {
        "project_name": "My_Capstone",
        "dataset_name": "tiny-demo",
        "raw_data_dir": str(fixtures / "ait_ads"),
        "labels_csv": str(fixtures / "labels.csv"),
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
    config_path = tmp_path / "baseline_tiny.json"
    config_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return config_path


def base_payload(tmp_path: Path, method_name: str) -> dict:
    baseline_config_path = write_baseline_config(tmp_path)
    return {
        "method_name": method_name,
        "display_name": method_name,
        "variant_name": "test-variant",
        "baseline_config": str(baseline_config_path),
        "baseline_run_id": "",
        "results_root": str(tmp_path / "results"),
        "dataset_name": "demo",
        "device": "cpu",
        "notes": "test",
        "hidden_size": 16,
    }


class ResearchArchitectureTests(unittest.TestCase):
    def test_cross_host_event_scope_keeps_event_token_space_compact(self) -> None:
        config = CrossHostConfig(
            config_path=Path("cross_host_event.json"),
            method_name="cross_host",
            display_name="cross_host",
            variant_name="event-scope",
            baseline_config=Path("baseline.json"),
            baseline_run_id="",
            results_root=Path("results"),
            dataset_name="demo",
            device="cpu",
            notes="test",
            hidden_size=16,
            local_context_length=1,
            companion_context_length=1,
            companion_time_window_seconds=3600,
            fusion_strategy="gate_then_attention",
            preserve_l1_distance=True,
            preserve_dbscan=True,
            token_scope="event",
        )
        frame = pd.DataFrame(
            [
                {
                    "timestamp": 1.0,
                    "machine": "host_a",
                    "scenario": "scenario",
                    "event": 100,
                    "label": -1,
                },
                {
                    "timestamp": 2.0,
                    "machine": "host_b",
                    "scenario": "scenario",
                    "event": 100,
                    "label": 1,
                },
            ]
        )

        bundle = build_cross_host_sequence_bundle(frame=frame, config=config)

        self.assertEqual(bundle.events.tolist(), [1, 1])
        self.assertEqual(bundle.mapping, {0: -1337, 1: 100})

    def test_cross_host_scaffold_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            payload = base_payload(tmp_path, "cross_host")
            payload.update(
                {
                    "local_context_length": 10,
                    "companion_context_length": 10,
                    "companion_time_window_seconds": 3600,
                    "fusion_strategy": "gate_then_attention",
                    "preserve_l1_distance": True,
                    "preserve_dbscan": True,
                }
            )
            config_path = write_config(tmp_path, payload)
            self.assertEqual(main(["run", "--config", str(config_path)]), 0)
            report_files = list((tmp_path / "results").rglob("summary.json"))
            self.assertTrue(report_files)

    def test_hierarchical_scaffold_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            payload = base_payload(tmp_path, "hierarchical_context")
            payload.update(
                {
                    "short_context_length": 10,
                    "long_memory_length": 32,
                    "landmark_strategy": "rarity_topk",
                    "landmark_time_window_seconds": 86400,
                    "preserve_l1_distance": True,
                    "preserve_dbscan": True,
                }
            )
            config_path = write_config(tmp_path, payload)
            self.assertEqual(main(["run", "--config", str(config_path)]), 0)
            report_files = list((tmp_path / "results").rglob("summary.json"))
            self.assertTrue(report_files)

    def test_rebalanced_scaffold_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            payload = base_payload(tmp_path, "rebalanced")
            payload.update(
                {
                    "context_length": 10,
                    "loss_type": "focal",
                    "focal_gamma": 2.0,
                    "positive_class_weight": 3.0,
                    "negative_class_weight": 1.0,
                    "reject_low_threshold": 0.25,
                    "reject_high_threshold": 0.70,
                    "preserve_l1_distance": True,
                    "preserve_dbscan": True,
                }
            )
            config_path = write_config(tmp_path, payload)
            self.assertEqual(main(["run", "--config", str(config_path)]), 0)
            report_files = list((tmp_path / "results").rglob("summary.json"))
            self.assertTrue(report_files)


if __name__ == "__main__":
    unittest.main()
