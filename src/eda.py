"""
Phase 2 - Exploratory Data Analysis & Insights (Q1/Q2 support material).

Produces, under outputs/:
  tables/missing_values.csv        - missingness in the RAW dataset (pre-cleaning)
  tables/descriptive_stats.csv     - mean, std, var, quantiles for numeric columns (cleaned data)
  tables/correlation_matrix.csv    - Pearson correlation, numeric columns
  tables/covariance_matrix.csv     - covariance, numeric columns
  tables/outlier_summary.csv       - IQR-rule outlier counts per numeric column
  figures/*.png                    - distribution, relationship and category plots

Run directly; also exposes helper functions for reuse from the notebook.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
if __name__ == "__main__":
    # Only force a headless backend for standalone script runs; when this module is
    # imported (e.g. from the notebook), leave whatever backend the caller set
    # ('%matplotlib inline' in Jupyter) alone so figures display inline.
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data_prep import RAW_CSV, load_clean

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "outputs" / "figures"
TAB_DIR = ROOT / "outputs" / "tables"

NUMERIC_COLS = [
    "selling_price", "km_driven", "car_age", "mileage", "engine",
    "max_power", "torque_nm", "torque_rpm", "seats",
]

sns.set_theme(style="whitegrid")


def missing_value_report() -> pd.DataFrame:
    """Missingness in the RAW (pre-cleaning) dataset - what Q2 asks us to report."""
    raw = pd.read_csv(RAW_CSV)
    n = len(raw)
    blank_mask = raw.apply(lambda s: s.astype(str).str.strip().eq("") if s.dtype == object else s.isna())
    counts = blank_mask.sum()
    report = pd.DataFrame({
        "missing_count": counts,
        "missing_pct": (counts / n * 100).round(2),
    }).sort_values("missing_count", ascending=False)
    return report


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = df[NUMERIC_COLS].agg(["mean", "std", "var", "min",
                                   lambda s: s.quantile(0.25),
                                   "median",
                                   lambda s: s.quantile(0.75),
                                   lambda s: s.quantile(0.95),
                                   "max"]).T
    stats.columns = ["mean", "std", "var", "min", "q25", "median", "q75", "q95", "max"]
    return stats.round(3)


def outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    """IQR-rule outlier counts: value outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]."""
    rows = []
    for col in NUMERIC_COLS:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((df[col] < lo) | (df[col] > hi)).sum())
        rows.append({"column": col, "lower_fence": round(lo, 2), "upper_fence": round(hi, 2),
                      "n_outliers": n_out, "pct_outliers": round(n_out / len(df) * 100, 2)})
    return pd.DataFrame(rows)


def make_figures(df: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Price distribution: raw vs log
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df["selling_price"], bins=50, ax=axes[0], color="#4C72B0")
    axes[0].set_title("Selling price (raw)")
    axes[0].set_xlabel("Selling price (INR)")
    sns.histplot(np.log(df["selling_price"]), bins=50, ax=axes[1], color="#55A868")
    axes[1].set_title("Selling price (log)")
    axes[1].set_xlabel("log(Selling price)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "price_distribution.png", dpi=130)
    plt.close(fig)

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(8, 6.5))
    corr = df[NUMERIC_COLS].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation matrix (numeric features)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "correlation_heatmap.png", dpi=130)
    plt.close(fig)

    # Price vs age / km_driven scatter
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    sns.scatterplot(data=df, x="car_age", y="selling_price", alpha=0.3, ax=axes[0], color="#4C72B0")
    axes[0].set_yscale("log")
    axes[0].set_title("Price vs car age")
    sns.scatterplot(data=df, x="km_driven", y="selling_price", alpha=0.3, ax=axes[1], color="#C44E52")
    axes[1].set_yscale("log")
    axes[1].set_xscale("log")
    axes[1].set_title("Price vs km driven")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "price_vs_age_km.png", dpi=130)
    plt.close(fig)

    # Boxplots by categorical
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    for ax, col in zip(axes.flat, ["fuel", "transmission", "seller_type", "owner"]):
        order = df.groupby(col)["selling_price"].median().sort_values(ascending=False).index
        sns.boxplot(data=df, x=col, y="selling_price", order=order, ax=ax, showfliers=False)
        ax.set_yscale("log")
        ax.set_title(f"Selling price by {col}")
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "price_by_category.png", dpi=130)
    plt.close(fig)

    # Brand-wise median price (top 15 by count)
    top_brands = df["brand"].value_counts().head(15).index
    brand_med = (df[df["brand"].isin(top_brands)]
                 .groupby("brand")["selling_price"].median()
                 .sort_values(ascending=False))
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=brand_med.values, y=brand_med.index, ax=ax, color="#4C72B0")
    ax.set_xlabel("Median selling price (INR)")
    ax.set_title("Median price by brand (top 15 by listing count)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "brand_median_price.png", dpi=130)
    plt.close(fig)


INSIGHTS = [
    "Selling price is strongly right-skewed (mean > median, long tail past INR 20L); "
    "regression targets log(selling_price) rather than the raw rupee value.",
    "car_age and km_driven are the two strongest negative correlates of price - older, "
    "higher-mileage cars sell for systematically less, as expected from depreciation.",
    "Automatic-transmission and Diesel-fuel listings carry a substantially higher median "
    "price than Manual/Petrol, reflecting both a segment effect (automatics skew toward "
    "premium trims) and a fuel-type effect (diesel engines cost more new).",
    "max_power and engine size are highly collinear (both describe engine displacement/output), "
    "which motivates using Ridge regression to stabilise coefficients rather than relying on "
    "plain OLS with both features present.",
]


def main() -> None:
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    df = load_clean()

    mv = missing_value_report()
    mv.to_csv(TAB_DIR / "missing_values.csv")
    print("Missing values (raw dataset), top columns:")
    print(mv.head(6).to_string())

    stats = descriptive_stats(df)
    stats.to_csv(TAB_DIR / "descriptive_stats.csv")
    print("\nDescriptive statistics (cleaned data):")
    print(stats.to_string())

    corr = df[NUMERIC_COLS].corr()
    corr.to_csv(TAB_DIR / "correlation_matrix.csv")
    cov = df[NUMERIC_COLS].cov()
    cov.to_csv(TAB_DIR / "covariance_matrix.csv")
    print("\nCorrelation with selling_price:")
    print(corr["selling_price"].sort_values(ascending=False).round(3).to_string())

    outliers = outlier_summary(df)
    outliers.to_csv(TAB_DIR / "outlier_summary.csv", index=False)
    print("\nOutlier summary (IQR rule):")
    print(outliers.to_string(index=False))

    make_figures(df)
    print(f"\nFigures saved -> {FIG_DIR}")

    print("\nKey insights:")
    for i, s in enumerate(INSIGHTS, 1):
        print(f"  {i}. {s}")


if __name__ == "__main__":
    main()
