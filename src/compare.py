"""
Phase 5 - Q4: Comparative Performance Analysis.

Builds grouped-bar comparison charts for the regression and classification results,
plus a combined confusion-matrix figure, from the tables already written by
regression.py and classification.py (re-run here so the figures always match the
latest metrics).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from classification import TIER_LABELS, run as run_classification
from regression import run as run_regression

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "outputs" / "figures"
TAB_DIR = ROOT / "outputs" / "tables"


def plot_regression_comparison(results_df) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    results_df["RMSE (INR)"].plot(kind="bar", ax=axes[0], color="#C44E52")
    axes[0].set_title("Regression: RMSE by model (lower is better)")
    axes[0].set_ylabel("RMSE (INR)")
    axes[0].tick_params(axis="x", rotation=25)

    results_df["R2 (rupee space)"].plot(kind="bar", ax=axes[1], color="#4C72B0")
    axes[1].set_title("Regression: R² by model (higher is better)")
    axes[1].set_ylabel("R² (rupee space)")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "regression_comparison.png", dpi=130)
    plt.close(fig)


def plot_classification_comparison(results_df) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    results_df.plot(kind="bar", ax=ax)
    ax.set_title("Classification: metric comparison by model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=15)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "classification_comparison.png", dpi=130)
    plt.close(fig)


def plot_confusion_matrices(confmats: dict) -> None:
    fig, axes = plt.subplots(1, len(confmats), figsize=(5 * len(confmats), 4.3))
    for ax, (name, cm) in zip(axes, confmats.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=TIER_LABELS, yticklabels=TIER_LABELS, ax=ax)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrices.png", dpi=130)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    reg_results, *_ = run_regression()
    plot_regression_comparison(reg_results)

    clf_results, confmats, *_ = run_classification()
    plot_classification_comparison(clf_results)
    plot_confusion_matrices(confmats)

    print("Regression comparison:")
    print(reg_results.round(4).to_string())
    print("\nClassification comparison:")
    print(clf_results.round(4).to_string())
    print(f"\nFigures saved -> {FIG_DIR}")


if __name__ == "__main__":
    main()
