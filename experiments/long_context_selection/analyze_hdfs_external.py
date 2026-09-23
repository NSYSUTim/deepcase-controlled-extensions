"""Unblind HDFS once and apply the frozen sequence-bootstrap decision."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_hdfs_external import labeled_metrics
from src.common import EXPERIMENT, sha256, write_json
from src.hdfs_external import safety_limit_per_1000


CONFIG_PATH = EXPERIMENT / "external_config_v2.yaml"
PROTOCOL_PATH = EXPERIMENT / "post_failure_external_protocol_v2.md"
FREEZE_PATH = EXPERIMENT / "results" / "external_track_v2_freeze.json"
ARTIFACTS = EXPERIMENT / "hdfs_artifacts"
RESULTS = EXPERIMENT / "results" / "hdfs_external"


def load_result(method: str, seed: int) -> dict:
    path = RESULTS / method / f"seed_{seed}.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("status") != "complete_blind":
        raise RuntimeError(f"blind model result is incomplete: {path}")
    if record.get("final_outcomes_accessed_for_metrics") is not False:
        raise AssertionError(f"runner was not blind: {path}")
    return record


def point_index(record: dict, eps: float, threshold: float) -> int:
    for index, point in enumerate(record["operating_points"]):
        if point["eps"] == eps and point["threshold"] == threshold:
            return index
    raise ValueError("frozen baseline point absent")


def choose_selector_point(
    record: dict, target_workload: float, safety_limit: float
) -> int | None:
    ranked = []
    for order, point in enumerate(record["operating_points"]):
        false_rate = float(
            point["validation"]["false_escalations_per_1000_nonincidents"]
        )
        if false_rate <= safety_limit:
            workload = float(point["test_unlabeled"]["review_workload"])
            ranked.append((abs(workload - target_workload), workload, order))
    return min(ranked)[2] if ranked else None


def _bootstrap_grouped_ratios(
    lengths: np.ndarray,
    baseline_seed_counts: np.ndarray,
    selector_seed_counts: np.ndarray,
    *,
    iterations: int,
    seed: int,
    scale: float,
    statistic: str,
    evaluation: dict,
) -> np.ndarray:
    """Exact cluster bootstrap after grouping identical sufficient statistics."""

    triples, frequencies = np.unique(
        np.column_stack(
            [lengths, baseline_seed_counts, selector_seed_counts]
        ).astype(np.int64),
        axis=0,
        return_counts=True,
    )
    probabilities = frequencies / frequencies.sum()
    rng = np.random.default_rng(seed)
    result = np.empty(iterations, dtype=float)
    # Bound the multinomial count matrix to roughly 32 MB per chunk.
    batch_size = max(1, min(iterations, 4_000_000 // max(1, len(triples))))
    seed_count = int(evaluation["model_seed_count"])
    for start in range(0, iterations, batch_size):
        stop = min(start + batch_size, iterations)
        draws = rng.multinomial(
            int(frequencies.sum()), probabilities, size=stop - start
        )
        denominator = draws @ triples[:, 0]
        baseline = scale * (draws @ triples[:, 1]) / (
            seed_count * denominator
        )
        selector = scale * (draws @ triples[:, 2]) / (
            seed_count * denominator
        )
        if statistic == "gain":
            result[start:stop] = selector - baseline
        elif statistic == "safety_violation":
            limit = np.where(
                baseline < evaluation["safety_switch_rate_per_1000"],
                baseline + evaluation["low_rate_absolute_margin_per_1000"],
                baseline * (1.0 + evaluation["high_rate_relative_margin"]),
            )
            result[start:stop] = selector - limit
        else:
            raise ValueError(statistic)
    return result


def verify_freeze() -> dict:
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    for name, item in freeze["files"].items():
        path = EXPERIMENT.parent / item["repo_relative_path"]
        observed = sha256(path)
        if observed != item["sha256"]:
            raise AssertionError(
                f"frozen external source changed: {name}: {observed} != {item['sha256']}"
            )
    return freeze


def analyze() -> dict:
    freeze = verify_freeze()
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    evaluation = dict(config["evaluation"])
    seeds = [int(value) for value in evaluation["model_seeds"]]
    evaluation["model_seed_count"] = len(seeds)
    labels_all = np.load(ARTIFACTS / "labels.npy", mmap_mode="r")
    splits = np.load(ARTIFACTS / "split_codes.npy", mmap_mode="r")
    sequence_ids_all = np.load(ARTIFACTS / "sequence_ids.npy", mmap_mode="r")
    test_rows = np.flatnonzero(splits == 3)
    truth = np.asarray(labels_all[test_rows], dtype=np.int8)
    test_sequence_ids = np.asarray(sequence_ids_all[test_rows], dtype=np.int32)
    if len(test_rows) != int(np.sum(splits == 3)):
        raise AssertionError("not all final rows were selected")
    sequence_ids, starts = np.unique(test_sequence_ids, return_index=True)
    if not np.array_equal(sequence_ids, test_sequence_ids[starts]):
        raise AssertionError("test rows are not sequence ordered")
    lengths = np.diff(np.r_[starts, len(test_rows)]).astype(np.int64)
    with np.load(ARTIFACTS / "sequences.npz") as sequence_table:
        sequence_labels = sequence_table["label"][sequence_ids]
        train_overlap = sequence_table["train_pattern_overlap"][sequence_ids]
        expected_lengths = sequence_table["length"][sequence_ids]
    if not np.array_equal(lengths, expected_lengths):
        raise AssertionError("row and sequence lengths disagree")

    baseline_sequence_counts = np.zeros(len(sequence_ids), dtype=np.int64)
    selector_sequence_counts = np.zeros(len(sequence_ids), dtype=np.int64)
    rows = []
    all_safe = True
    exact_ten = True
    for seed in seeds:
        baseline = load_result("last10", seed)
        selector = load_result("anchor5_assoc5", seed)
        exact_ten &= (
            int(baseline["context_length"]) == 10
            and int(selector["context_length"]) == 10
        )
        baseline_index = point_index(
            baseline,
            float(config["interpreter"]["default_eps"]),
            float(config["interpreter"]["default_threshold"]),
        )
        baseline_point = baseline["operating_points"][baseline_index]
        base_validation_false = float(
            baseline_point["validation"][
                "false_escalations_per_1000_nonincidents"
            ]
        )
        limit = safety_limit_per_1000(
            base_validation_false,
            switch_rate=float(evaluation["safety_switch_rate_per_1000"]),
            absolute_margin=float(
                evaluation["low_rate_absolute_margin_per_1000"]
            ),
            relative_margin=float(evaluation["high_rate_relative_margin"]),
        )
        target_workload = float(
            baseline_point["test_unlabeled"]["review_workload"]
        )
        selected_index = choose_selector_point(selector, target_workload, limit)
        all_safe &= selected_index is not None
        if selected_index is None:
            rows.append(
                {
                    "seed": seed,
                    "safe_point_exists": False,
                    "baseline_validation_false_escalations_per_1000": base_validation_false,
                    "validation_safety_limit_per_1000": limit,
                }
            )
            continue
        selector_point = selector["operating_points"][selected_index]
        baseline_predictions = np.load(
            baseline["predictions"], mmap_mode="r"
        )[baseline_index]
        selector_predictions = np.load(
            selector["predictions"], mmap_mode="r"
        )[selected_index]
        base_metrics = labeled_metrics(truth, baseline_predictions)
        select_metrics = labeled_metrics(truth, selector_predictions)
        baseline_sequence_counts += np.add.reduceat(
            (baseline_predictions > 0).astype(np.int64), starts
        )
        selector_sequence_counts += np.add.reduceat(
            (selector_predictions > 0).astype(np.int64), starts
        )
        rows.append(
            {
                "seed": seed,
                "safe_point_exists": True,
                "selected_grid_index": selected_index,
                "selected_eps": selector_point["eps"],
                "selected_threshold": selector_point["threshold"],
                "baseline_validation_false_escalations_per_1000": base_validation_false,
                "validation_safety_limit_per_1000": limit,
                "baseline_workload": base_metrics["review_workload"],
                "selector_workload": select_metrics["review_workload"],
                "workload_difference": select_metrics["review_workload"]
                - base_metrics["review_workload"],
                "baseline_recall": base_metrics["automatic_incident_recall"],
                "selector_recall": select_metrics["automatic_incident_recall"],
                "recall_gain": select_metrics["automatic_incident_recall"]
                - base_metrics["automatic_incident_recall"],
                "baseline_false_escalations_per_1000": base_metrics[
                    "false_escalations_per_1000_nonincidents"
                ],
                "selector_false_escalations_per_1000": select_metrics[
                    "false_escalations_per_1000_nonincidents"
                ],
                "baseline_strict_macro_f1": base_metrics["strict_macro_f1"],
                "selector_strict_macro_f1": select_metrics["strict_macro_f1"],
            }
        )

    if not all_safe:
        return {
            "external_claim_succeeds": False,
            "failure_reason": "one_or_more_seeds_have_no_validation_safe_point",
            "blind_rows": rows,
            "freeze_verified": True,
        }

    incident_sequences = sequence_labels == 1
    normal_sequences = sequence_labels == 0
    iterations = int(evaluation["bootstrap_iterations"])
    bootstrap_seed = int(evaluation["bootstrap_seed"])
    recall_bootstrap = _bootstrap_grouped_ratios(
        lengths[incident_sequences],
        baseline_sequence_counts[incident_sequences],
        selector_sequence_counts[incident_sequences],
        iterations=iterations,
        seed=bootstrap_seed,
        scale=1.0,
        statistic="gain",
        evaluation=evaluation,
    )
    safety_bootstrap = _bootstrap_grouped_ratios(
        lengths[normal_sequences],
        baseline_sequence_counts[normal_sequences],
        selector_sequence_counts[normal_sequences],
        iterations=iterations,
        seed=bootstrap_seed + 1,
        scale=1000.0,
        statistic="safety_violation",
        evaluation=evaluation,
    )
    recall_ci = np.quantile(recall_bootstrap, [0.025, 0.975])
    safety_ci = np.quantile(safety_bootstrap, [0.025, 0.975])
    total_incident_events = int(lengths[incident_sequences].sum())
    total_normal_events = int(lengths[normal_sequences].sum())
    mean_base_recall = float(
        baseline_sequence_counts[incident_sequences].sum()
        / (len(seeds) * total_incident_events)
    )
    mean_select_recall = float(
        selector_sequence_counts[incident_sequences].sum()
        / (len(seeds) * total_incident_events)
    )
    mean_base_false = float(
        1000.0
        * baseline_sequence_counts[normal_sequences].sum()
        / (len(seeds) * total_normal_events)
    )
    mean_select_false = float(
        1000.0
        * selector_sequence_counts[normal_sequences].sum()
        / (len(seeds) * total_normal_events)
    )
    observed_safety_limit = safety_limit_per_1000(
        mean_base_false,
        switch_rate=float(evaluation["safety_switch_rate_per_1000"]),
        absolute_margin=float(evaluation["low_rate_absolute_margin_per_1000"]),
        relative_margin=float(evaluation["high_rate_relative_margin"]),
    )
    recall_pass = float(recall_ci[0]) > 0.0
    safety_pass = float(safety_ci[1]) <= 0.0
    success = recall_pass and safety_pass and all_safe and exact_ten

    return {
        "external_claim_succeeds": bool(success),
        "dataset": "HDFS_DeepLog_preprocessed",
        "freeze_verified": True,
        "freeze_sha256": sha256(FREEZE_PATH),
        "all_seed_safety_sets_nonempty": bool(all_safe),
        "context_positions_exactly_ten": bool(exact_ten),
        "all_final_sequences_used": True,
        "final_sequences": int(len(sequence_ids)),
        "final_events": int(len(test_rows)),
        "final_anomalous_sequences": int(incident_sequences.sum()),
        "final_normal_sequences": int(normal_sequences.sum()),
        "mean_baseline_recall": mean_base_recall,
        "mean_selector_recall": mean_select_recall,
        "mean_recall_gain": mean_select_recall - mean_base_recall,
        "recall_gain_sequence_bootstrap_95_percent_ci": recall_ci.tolist(),
        "recall_ci_pass": bool(recall_pass),
        "mean_baseline_false_escalations_per_1000": mean_base_false,
        "mean_selector_false_escalations_per_1000": mean_select_false,
        "observed_false_escalation_safety_limit_per_1000": observed_safety_limit,
        "safety_violation_sequence_bootstrap_95_percent_ci": safety_ci.tolist(),
        "safety_noninferiority_ci_pass": bool(safety_pass),
        "bootstrap_iterations": iterations,
        "bootstrap_seed": bootstrap_seed,
        "official_train_pattern_overlap_final_sequences": int(
            np.sum(train_overlap)
        ),
        "blind_rows": rows,
        "limitations": [
            "HDFS has no timestamps; this cannot validate the one-day timeout.",
            "HDFS anomaly labels are sequence-level and are inherited by event rows.",
            "One public external dataset does not establish universal generalization.",
        ],
    }


def markdown(result: dict) -> str:
    if "failure_reason" in result:
        return (
            "# HDFS external confirmation\n\n"
            f"- Success: `{result['external_claim_succeeds']}`\n"
            f"- Failure: `{result['failure_reason']}`\n"
        )
    return "\n".join(
        [
            "# HDFS external confirmation",
            "",
            f"- Success: `{result['external_claim_succeeds']}`",
            f"- Final sequences/events: `{result['final_sequences']}` / `{result['final_events']}`",
            f"- Strict automatic anomaly recall: `{result['mean_baseline_recall']:.6f}` -> "
            f"`{result['mean_selector_recall']:.6f}` (gain `{result['mean_recall_gain']:.6f}`)",
            "- Recall-gain sequence-bootstrap 95% CI: "
            f"`[{result['recall_gain_sequence_bootstrap_95_percent_ci'][0]:.6f}, "
            f"{result['recall_gain_sequence_bootstrap_95_percent_ci'][1]:.6f}]`",
            "- False escalations / 1000: "
            f"`{result['mean_baseline_false_escalations_per_1000']:.6f}` -> "
            f"`{result['mean_selector_false_escalations_per_1000']:.6f}`",
            "- Safety-violation sequence-bootstrap 95% CI: "
            f"`[{result['safety_violation_sequence_bootstrap_95_percent_ci'][0]:.6f}, "
            f"{result['safety_violation_sequence_bootstrap_95_percent_ci'][1]:.6f}]`",
            f"- Recall criterion passes: `{result['recall_ci_pass']}`",
            f"- Safety criterion passes: `{result['safety_noninferiority_ci_pass']}`",
            f"- Exact ten-position budget: `{result['context_positions_exactly_ten']}`",
            "",
        ]
    )


def main() -> None:
    result = analyze()
    output = RESULTS / "final_decision.json"
    write_json(output, result)
    output.with_suffix(".md").write_text(markdown(result), encoding="utf-8")
    print(markdown(result))


if __name__ == "__main__":
    main()
