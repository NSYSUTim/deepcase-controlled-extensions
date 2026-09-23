from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from my_capstone.result_comparison import (
    build_comparison,
    load_baseline_evaluation,
    load_research_evaluation,
    resolve_research_run_summary,
    write_comparison_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare baseline and research DeepCASE result summaries."
    )
    parser.add_argument(
        "--baseline-run",
        default="2026-05-19_18-00-28_ait-ads_cpu_full",
        help="Baseline run ID under results/runs.",
    )
    parser.add_argument(
        "--research-summary",
        action="append",
        default=[],
        help="Path to a research reports/summary.json. Can be repeated.",
    )
    parser.add_argument(
        "--research-run",
        action="append",
        default=[],
        help="Research run in METHOD:RUN_ID form, e.g. rebalanced:latest.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/comparison",
        help="Directory for comparison.json/csv/md.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    baseline = load_baseline_evaluation(PROJECT_ROOT, args.baseline_run)
    summary_paths = [Path(path) for path in args.research_summary]
    for research_run in args.research_run:
        if ":" not in research_run:
            raise ValueError("--research-run must use METHOD:RUN_ID format.")
        method, run_id = research_run.split(":", 1)
        summary_paths.append(resolve_research_run_summary(PROJECT_ROOT, method, run_id))

    if not summary_paths:
        raise ValueError("Provide at least one --research-summary or --research-run.")

    research_runs = [load_research_evaluation(path) for path in summary_paths]
    comparison = build_comparison(baseline=baseline, research_runs=research_runs)
    outputs = write_comparison_outputs(
        comparison,
        PROJECT_ROOT / args.output_dir,
    )
    print("[compare_results] wrote:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
