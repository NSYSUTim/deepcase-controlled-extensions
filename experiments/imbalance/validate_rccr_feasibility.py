"""Independent fail-closed validation of the RC-CR design result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_point(points: list[dict], eps: float, threshold: float) -> dict:
    matches = [
        point
        for point in points
        if float(point["eps"]) == eps and float(point["threshold"]) == threshold
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one point for eps={eps}, threshold={threshold}")
    return matches[0]


def run(result_path: Path, output_path: Path) -> dict:
    errors: list[str] = []
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "complete":
        errors.append("RC-CR result is not complete")
    if result.get("scope") != "exploratory design-only; not confirmatory":
        errors.append("RC-CR scope marker is missing")
    if int(result.get("query_iterations", -1)) != 100:
        errors.append("RC-CR result is not q=100")

    checkpoint = Path(result["checkpoint"])
    if not checkpoint.exists():
        errors.append("Frozen checkpoint is missing")
    elif sha256_file(checkpoint) != result.get("checkpoint_sha256"):
        errors.append("Frozen checkpoint hash mismatch")
    source_path = checkpoint.with_name("deepcase.json")
    if not source_path.exists():
        errors.append("Frozen source JSON is missing")
        source = None
    else:
        source = json.loads(source_path.read_text(encoding="utf-8"))

    compared = 0
    maximum_difference = 0.0
    if source is not None:
        for current in result["deepcase_reference_points"]:
            original = _find_point(
                source["operating_points"],
                float(current["eps"]),
                float(current["threshold"]),
            )
            for split in ("validation", "test"):
                for metric, value in current[split]["overall"].items():
                    original_value = original[split].get(metric)
                    if value is None or original_value is None:
                        if value != original_value:
                            errors.append(f"Null mismatch: {split}/{metric}")
                        continue
                    difference = abs(float(value) - float(original_value))
                    maximum_difference = max(maximum_difference, difference)
                    compared += 1
                    if difference != 0.0:
                        errors.append(
                            f"Source reproduction mismatch: {current['eps']}/"
                            f"{current['threshold']}/{split}/{metric}={difference}"
                        )

    for endpoint in ("low_workload", "deepcase_default"):
        reference = result["deepcase_reference_by_endpoint"][endpoint]["test"][
            "overall"
        ]
        for family, item in result["selected_models"].items():
            metrics = item["test"][endpoint]["overall"]
            if float(metrics["analyst_workload"]) != float(
                reference["analyst_workload"]
            ):
                errors.append(f"Workload mismatch: {family}/{endpoint}")
            if float(metrics["reject_rate"]) != float(reference["reject_rate"]):
                errors.append(f"Reject mismatch: {family}/{endpoint}")

    pairwise_gate = result["feasibility"]["within_rule_pairwise"][
        "passes_single_scenario_effect_threshold"
    ]
    if pairwise_gate:
        errors.append("Expected first-fold stop rule did not fire")

    result_root = result_path.parents[3]
    cache_dir = (
        result_root
        / "cache"
        / result["test_scenario"]
        / f"ratio_{result['imbalance_ratio']}"
        / f"seed_{result['seed']}"
    )
    cache_files_verified = 0
    for name in ("fit", "revealed", "validation", "test"):
        manifest_path = cache_dir / f"{name}_manifest.json"
        if not manifest_path.exists():
            errors.append(f"Missing cache manifest: {name}")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        vectors = cache_dir / f"{name}_vectors.npz"
        arrays = cache_dir / f"{name}_arrays.npz"
        if manifest.get("status") != "complete":
            errors.append(f"Incomplete cache: {name}")
            continue
        if not vectors.exists() or not arrays.exists():
            errors.append(f"Missing cache payload: {name}")
            continue
        if sha256_file(vectors) != manifest.get("vectors_sha256"):
            errors.append(f"Vector cache hash mismatch: {name}")
            continue
        if sha256_file(arrays) != manifest.get("arrays_sha256"):
            errors.append(f"Array cache hash mismatch: {name}")
            continue
        cache_files_verified += 2

    fox_path = (
        result_path.parents[2]
        / "fox"
        / f"ratio_{result['imbalance_ratio']}"
        / f"seed_{result['seed']}.json"
    )
    if fox_path.exists():
        errors.append("fox result exists despite first-fold stop rule")

    receipt = {
        "status": "valid" if not errors else "invalid",
        "errors": errors,
        "result": str(result_path),
        "checkpoint_sha256": result.get("checkpoint_sha256"),
        "source_metric_values_compared": compared,
        "source_maximum_absolute_difference": maximum_difference,
        "cache_payload_files_verified": cache_files_verified,
        "pairwise_gate_passed": bool(pairwise_gate),
        "fox_run_absent": not fox_path.exists(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    if errors:
        raise SystemExit("RC-CR validation failed:\n- " + "\n- ".join(errors))
    return receipt


def parse_args() -> argparse.Namespace:
    base = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        type=Path,
        default=(
            base
            / "results"
            / "rccr_feasibility_v1"
            / "design"
            / "russellmitchell"
            / "ratio_1000"
            / "seed_1729.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=base / "artifacts" / "validation_rccr_feasibility_v1.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    receipt = run(arguments.result.resolve(), arguments.output.resolve())
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
