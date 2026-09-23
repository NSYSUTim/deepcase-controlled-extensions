"""Generate publication-ready figures from frozen result JSON files."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def main() -> None:
    result = json.loads(
        (RESULTS / "confirmation_decision.json").read_text(encoding="utf-8")
    )
    primary = result["primary"]
    effects = primary["scenario_effects"]
    scenarios = list(effects)
    recall_gain = np.array([effects[name]["mean_recall_gain"] for name in scenarios])
    baseline_false = np.array(
        [effects[name]["mean_baseline_false_escalations_per_1000"] for name in scenarios]
    )
    selector_false = np.array(
        [effects[name]["mean_selector_false_escalations_per_1000"] for name in scenarios]
    )

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )
    color_baseline = "#4C78A8"
    color_selector = "#E45756"
    color_gain = "#2A9D8F"

    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), constrained_layout=True)
    positions = np.arange(len(scenarios))

    ax = axes[0]
    ax.axvline(0, color="0.35", linewidth=0.9)
    ax.scatter(recall_gain, positions, s=35, color=color_gain, zorder=3)
    for y, value in zip(positions, recall_gain):
        ax.text(
            value + 0.003,
            y,
            f"{value:+.3f}",
            va="center",
            ha="left",
        )
    overall = primary["paired_mean_recall_gain"]
    low, high = primary["scenario_bootstrap_95_percent_ci"]
    overall_y = len(scenarios) + 0.35
    ax.errorbar(
        overall,
        overall_y,
        xerr=np.array([[overall - low], [high - overall]]),
        fmt="D",
        color="0.15",
        capsize=4,
        markersize=5,
    )
    ax.text(overall + 0.004, overall_y, f"{overall:+.3f} (95% CI)", va="center")
    ax.set_yticks(np.r_[positions, overall_y], scenarios + ["overall"])
    ax.invert_yaxis()
    ax.set_xlabel("Strict automatic incident-recall gain")
    ax.set_title("A. Recall effect by held-out scenario")
    ax.set_xlim(-0.025, 0.105)
    ax.grid(axis="x", color="0.88", linewidth=0.6)

    ax = axes[1]
    width = 0.36
    ax.bar(
        positions - width / 2,
        baseline_false,
        width,
        label="Last10",
        color=color_baseline,
    )
    ax.bar(
        positions + width / 2,
        selector_false,
        width,
        label="association_abs10",
        color=color_selector,
    )
    ax.set_yscale("symlog", linthresh=0.1, linscale=0.8)
    ax.set_xticks(positions, scenarios, rotation=32, ha="right")
    ax.set_ylabel("False escalations per 1,000 non-incidents")
    ax.set_title("B. Safety cost at matched review workload")
    ax.grid(axis="y", color="0.88", linewidth=0.6)
    ax.legend(frameon=False, loc="upper left", fontsize=8)

    figure.suptitle(
        "AIT-ADS confirmation: recall improved, but the safety criterion failed",
        fontsize=12,
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(FIGURES / f"ait_confirmation_tradeoff.{suffix}")
    plt.close(figure)


if __name__ == "__main__":
    main()
