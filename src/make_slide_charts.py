"""
Presentation charts for the PPT deck. One focused idea per PNG (not the multi-panel
report figures in outputs/figures/) - built for a projector: large fonts, generous
margins, 16:9-friendly sizing.

Every number comes from the same pipeline already verified in the notebook/report
(data_prep.load_clean, regression.run, classification.run, interpret.*) - nothing
here is invented or re-derived differently.

Palette: the validated default from the dataviz skill (references/palette.md) -
categorical hues assigned in fixed order, sequential = one hue (blue) light->dark,
diverging = blue<->red with a neutral gray midpoint. Saved to images/ at project root.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from classification import TIER_LABELS, run as run_classification
from data_prep import load_clean
from eda import NUMERIC_COLS, missing_value_report
from interpret import linear_regression_coefficients, logistic_regression_coefficients
from regression import run as run_regression

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "images"

# ---------------------------------------------------------------- palette (validated default)
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE, ORANGE, AQUA, YELLOW = CAT[0], CAT[1], CAT[2], CAT[3]
RED = CAT[7]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
NEUTRAL_MID = "#f0efec"

SEQ_BLUE = ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
SEQ_BLUE_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
DIVERGE_CMAP = LinearSegmentedColormap.from_list("diverge_blue_red", [RED, NEUTRAL_MID, BLUE])

# fixed identity mapping - same category always gets the same color across every chart
FUEL_COLORS = {"Diesel": CAT[0], "Petrol": CAT[1], "CNG": CAT[2], "LPG": CAT[3]}
TRANS_COLORS = {"Manual": CAT[0], "Automatic": CAT[1]}
SELLER_COLORS = {"Individual": CAT[0], "Dealer": CAT[1], "Trustmark Dealer": CAT[2]}
OWNER_COLORS = {"First Owner": CAT[0], "Second Owner": CAT[1], "Third Owner": CAT[2],
                 "Fourth & Above Owner": CAT[3], "Test Drive Car": CAT[4]}
REG_MODEL_COLORS = {"Linear Regression": CAT[0], "Polynomial Regression (deg=2)": CAT[1],
                     "Ridge Regression (RidgeCV)": CAT[2], "Bayesian Ridge Regression": CAT[3]}
CLF_MODEL_COLORS = {"Logistic Regression": CAT[0], "Linear SVM": CAT[1], "RBF-Kernel SVM": CAT[2]}


def _style() -> None:
    sns.set_theme(style="white")
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK_PRIMARY,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": AXIS,
        "axes.titlecolor": INK_PRIMARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 15,
        "axes.titlesize": 18,
        "axes.titleweight": "bold",
        "axes.labelsize": 14,
        "xtick.labelsize": 12.5,
        "ytick.labelsize": 12.5,
        "legend.fontsize": 12.5,
        "legend.frameon": False,
    })


def _save(fig, name: str) -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    path = IMG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"  {name}")


def _kfmt(x, _pos=None):
    if abs(x) >= 1e7:
        return f"{x/1e7:.1f}Cr"
    if abs(x) >= 1e5:
        return f"{x/1e5:.1f}L"
    if abs(x) >= 1e3:
        return f"{x/1e3:.0f}k"
    return f"{x:.0f}"


# ================================================================== Q2 EDA charts

def chart_price_distribution_raw(df):
    fig, ax = plt.subplots(figsize=(10, 5.6))
    sns.histplot(df["selling_price"], bins=50, ax=ax, color=BLUE, edgecolor=SURFACE, linewidth=0.4)
    ax.set_title("Selling Price Distribution (Raw)")
    ax.set_xlabel("Selling price (INR)")
    ax.set_ylabel("Number of listings")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_kfmt))
    ax.grid(axis="y")
    _save(fig, "01_price_distribution_raw.png")


def chart_price_distribution_log(df):
    fig, ax = plt.subplots(figsize=(10, 5.6))
    sns.histplot(np.log(df["selling_price"]), bins=50, ax=ax, color=BLUE, edgecolor=SURFACE, linewidth=0.4)
    ax.set_title("Selling Price Distribution (Log-Transformed)")
    ax.set_xlabel("log(Selling price)")
    ax.set_ylabel("Number of listings")
    ax.grid(axis="y")
    _save(fig, "02_price_distribution_log.png")


def chart_missing_values():
    mv = missing_value_report()
    mv = mv[mv["missing_count"] > 0].sort_values("missing_pct")
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.barh(mv.index, mv["missing_pct"], color=BLUE, height=0.6)
    for y, v in enumerate(mv["missing_pct"]):
        ax.text(v + 0.05, y, f"{v:.2f}%", va="center", fontsize=12.5, color=INK_SECONDARY)
    ax.set_title("Missing Values in Raw Dataset")
    ax.set_xlabel("% of rows missing")
    ax.set_xlim(0, mv["missing_pct"].max() * 1.25)
    ax.grid(axis="x")
    _save(fig, "03_missing_values.png")


def chart_correlation_heatmap(df):
    corr = df[NUMERIC_COLS].corr()
    fig, ax = plt.subplots(figsize=(9.5, 8))
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=DIVERGE_CMAP, norm=norm,
                square=True, linewidths=1.5, linecolor=SURFACE,
                cbar_kws={"label": "Correlation"}, ax=ax, annot_kws={"fontsize": 11})
    ax.set_title("Correlation Matrix — Numeric Features")
    _save(fig, "04_correlation_heatmap.png")


def chart_price_vs_car_age(df):
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.scatter(df["car_age"], df["selling_price"], alpha=0.25, s=22, color=BLUE, linewidths=0)
    ax.set_yscale("log")
    ax.set_title("Selling Price vs Car Age")
    ax.set_xlabel("Car age (years)")
    ax.set_ylabel("Selling price (INR, log scale)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_kfmt))
    ax.grid(axis="both")
    _save(fig, "05_price_vs_car_age.png")


def chart_price_vs_km_driven(df):
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.scatter(df["km_driven"], df["selling_price"], alpha=0.25, s=22, color=BLUE, linewidths=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Selling Price vs Kilometers Driven")
    ax.set_xlabel("km driven (log scale)")
    ax.set_ylabel("Selling price (INR, log scale)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_kfmt))
    ax.grid(axis="both")
    _save(fig, "06_price_vs_km_driven.png")


def _price_by(df, col, colors, fname, title):
    order = df.groupby(col)["selling_price"].median().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(10, 5.6))
    sns.boxplot(data=df, x=col, y="selling_price", order=order, ax=ax,
                palette=[colors[c] for c in order], showfliers=False, width=0.55,
                linewidth=1.3, fliersize=0)
    for patch in ax.patches:
        patch.set_edgecolor(INK_PRIMARY)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Selling price (INR, log scale)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_kfmt))
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y")
    _save(fig, fname)


def chart_price_by_fuel(df):
    _price_by(df, "fuel", FUEL_COLORS, "07_price_by_fuel.png", "Selling Price by Fuel Type")


def chart_price_by_transmission(df):
    _price_by(df, "transmission", TRANS_COLORS, "08_price_by_transmission.png",
              "Selling Price by Transmission")


def chart_price_by_seller_type(df):
    _price_by(df, "seller_type", SELLER_COLORS, "09_price_by_seller_type.png",
              "Selling Price by Seller Type")


def chart_price_by_owner(df):
    _price_by(df, "owner", OWNER_COLORS, "10_price_by_owner.png", "Selling Price by Ownership History")


def chart_brand_median_price(df):
    top_brands = df["brand"].value_counts().head(15).index
    brand_med = (df[df["brand"].isin(top_brands)].groupby("brand")["selling_price"]
                 .median().sort_values(ascending=True))
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.barh(brand_med.index, brand_med.values, color=BLUE, height=0.65)
    ax.set_title("Median Selling Price by Brand (Top 15 by Listing Count)")
    ax.set_xlabel("Median selling price (INR)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_kfmt))
    ax.grid(axis="x")
    _save(fig, "11_brand_median_price.png")


# ================================================================== Q3/Q4 model charts

def chart_regression_rmse(reg_results):
    order = list(REG_MODEL_COLORS)
    vals = reg_results.loc[order, "RMSE (INR)"]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    bars = ax.bar(order, vals, color=[REG_MODEL_COLORS[m] for m in order], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2500, f"INR {v:,.0f}",
                ha="center", fontsize=12, color=INK_SECONDARY)
    ax.set_title("Regression: RMSE by Model (lower is better)")
    ax.set_ylabel("RMSE (INR)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_kfmt))
    ax.set_xticklabels(order, rotation=12, ha="right")
    ax.grid(axis="y")
    _save(fig, "12_regression_rmse_comparison.png")


def chart_regression_r2(reg_results):
    order = list(REG_MODEL_COLORS)
    vals = reg_results.loc[order, "R2 (rupee space)"]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    bars = ax.bar(order, vals, color=[REG_MODEL_COLORS[m] for m in order], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                ha="center", fontsize=12, color=INK_SECONDARY)
    ax.set_title("Regression: R² by Model (higher is better)")
    ax.set_ylabel("R² (rupee space)")
    ax.set_ylim(0, 1)
    ax.set_xticklabels(order, rotation=12, ha="right")
    ax.grid(axis="y")
    _save(fig, "13_regression_r2_comparison.png")


def chart_classification_metrics(clf_results):
    order = list(CLF_MODEL_COLORS)
    metrics = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"]
    x = np.arange(len(metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, model in enumerate(order):
        vals = clf_results.loc[model, metrics]
        ax.bar(x + (i - 1) * width, vals, width=width, label=model, color=CLF_MODEL_COLORS[model])
    ax.set_title("Classification: Metric Comparison by Model")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", ncol=1)
    ax.grid(axis="y")
    _save(fig, "14_classification_metrics_comparison.png")


def chart_confusion_matrix(name, cm, fname):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap=SEQ_BLUE_CMAP, cbar=False,
                xticklabels=TIER_LABELS, yticklabels=TIER_LABELS, ax=ax,
                linewidths=1.5, linecolor=SURFACE, annot_kws={"fontsize": 15, "weight": "bold"})
    for text in ax.texts:
        val = int(text.get_text())
        text.set_color(INK_PRIMARY if val < cm.max() * 0.55 else "white")
    ax.set_title(f"Confusion Matrix — {name}")
    ax.set_xlabel("Predicted tier")
    ax.set_ylabel("Actual tier")
    _save(fig, fname)


# ================================================================== Q5 interpretation charts

def chart_linear_coefficients(reg_models):
    coef = linear_regression_coefficients(reg_models).head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [BLUE if v >= 0 else RED for v in coef["coefficient"]]
    ax.barh(coef["feature"], coef["coefficient"], color=colors, height=0.6)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_title("Top Drivers of log(Selling Price) — Linear Regression")
    ax.set_xlabel("Standardized coefficient (effect per 1 std-dev)")
    ax.grid(axis="x")
    _save(fig, "15_linear_regression_top_coefficients.png")


def chart_logistic_coefficients(clf_models):
    coef = logistic_regression_coefficients(clf_models).head(12)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    vmax = coef.abs().values.max()
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    sns.heatmap(coef, annot=True, fmt=".2f", cmap=DIVERGE_CMAP, norm=norm,
                linewidths=1.5, linecolor=SURFACE, cbar_kws={"label": "Coefficient"},
                ax=ax, annot_kws={"fontsize": 11})
    ax.set_title("Top Drivers of Price Tier — Logistic Regression")
    ax.set_xlabel("Tier")
    ax.set_ylabel("")
    _save(fig, "16_logistic_regression_top_coefficients.png")


def main() -> None:
    _style()
    print("Loading data and re-running the verified pipeline...")
    df = load_clean()
    reg_results, reg_models, train_idx, test_idx = run_regression()
    clf_results, confmats, reports, thresholds, clf_models, *_ = run_classification()

    print(f"\nSaving presentation charts -> {IMG_DIR}")
    chart_price_distribution_raw(df)
    chart_price_distribution_log(df)
    chart_missing_values()
    chart_correlation_heatmap(df)
    chart_price_vs_car_age(df)
    chart_price_vs_km_driven(df)
    chart_price_by_fuel(df)
    chart_price_by_transmission(df)
    chart_price_by_seller_type(df)
    chart_price_by_owner(df)
    chart_brand_median_price(df)
    chart_regression_rmse(reg_results)
    chart_regression_r2(reg_results)
    chart_classification_metrics(clf_results)
    for name, cm in confmats.items():
        fname = f"{ {'Logistic Regression': '17a', 'Linear SVM': '17b', 'RBF-Kernel SVM': '17c'}[name] }_confusion_matrix_{name.replace(' ', '_')}.png"
        chart_confusion_matrix(name, cm, fname)
    chart_linear_coefficients(reg_models)
    chart_logistic_coefficients(clf_models)

    n = len(list(IMG_DIR.glob("*.png")))
    print(f"\nDone. {n} charts in {IMG_DIR}")


if __name__ == "__main__":
    main()
