from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from my_capstone.artifacts import create_run_artifacts, load_run_artifacts, write_latest_run


class ArtifactTests(unittest.TestCase):
    def test_artifact_paths_and_latest_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifacts = create_run_artifacts(
                results_root=root,
                dataset_name="ait-ads",
                device_name="cpu",
                mode_name="full",
            )

            self.assertTrue(artifacts.run_dir.exists())
            self.assertTrue(artifacts.run_id.endswith("ait-ads_cpu_full"))

            write_latest_run(root, artifacts.run_id)
            loaded = load_run_artifacts(root, "latest")
            self.assertEqual(loaded.run_id, artifacts.run_id)


if __name__ == "__main__":
    unittest.main()
