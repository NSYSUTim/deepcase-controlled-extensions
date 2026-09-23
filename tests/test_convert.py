from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from my_capstone.convert import (
    build_label_lookup,
    convert_wazuh_to_deepcase_csv,
    parse_timestamp,
    validate_processed_csv,
)


class ConvertTests(unittest.TestCase):
    def test_parse_timestamp_returns_epoch(self) -> None:
        self.assertEqual(parse_timestamp("2026-01-01T00:00:00Z"), 1767225600.0)

    def test_build_label_lookup_orders_intervals(self) -> None:
        frame = pd.DataFrame(
            [
                {"scenario": "demo", "attack": "alpha", "start": 10, "end": 20},
                {"scenario": "demo", "attack": "alpha", "start": 1, "end": 5},
            ]
        )
        lookup = build_label_lookup(frame, {"alpha": 0})
        self.assertEqual(lookup["demo"][0], (1.0, 5.0, 0))

    def test_convert_and_validate_fixture(self) -> None:
        raw_dir = Path(__file__).parent / "fixtures" / "raw" / "ait_ads"
        labels_csv = Path(__file__).parent / "fixtures" / "raw" / "labels.csv"
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "processed.csv"
            result = convert_wazuh_to_deepcase_csv(
                raw_dir=raw_dir,
                labels_path=labels_csv,
                output_path=output_path,
                force=True,
                log=lambda _: None,
            )

            self.assertTrue(result.output_path.exists())
            self.assertTrue(validate_processed_csv(result.output_path))


if __name__ == "__main__":
    unittest.main()
