"""
Phase 6 - Q5 support: pull out interpretable coefficients from the linear models
(Linear Regression for price, Logistic Regression for tier) so Q5 can point to
concrete drivers rather than just metric numbers.

Coefficients are on STANDARDIZED features (StandardScaler is the first step of every
pipeline), so they are directly comparable to each other as "effect per one std-dev
change in that feature" - the conventional way to read relative feature importance
out of a linear model with mixed-unit inputs.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from classification import run as run_classification
from regression import run as run_regression

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "outputs" / "figures"
TAB_DIR = ROOT / "outputs" / "tables"


def _feature_names(preprocessor) -> list[str]:
    """Strip the ColumnTransformer's 'num__'/'cat__' prefixes for readable labels."""
    raw = preprocessor.get_feature_names_out()
    return [n.split("__", 1)[1] if "__" in n else n for n in raw]


def linear_regression_coefficients(models) -> pd.DataFrame:
    pipe = models["Linear Regression"]
    names = _feature_names(pipe.named_steps["prep"])
    coefs = pipe.named_steps["reg"].coef_
    df = pd.DataFrame({"feature": names, "coefficient": coefs})
    df["abs_coefficient"] = df["coefficient"].abs()
    return df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def logistic_regression_coefficients(models) -> pd.DataFrame:
    pipe = models["Logistic Regression"]
    names = _feature_names(pipe.named_steps["prep"])
    clf = pipe.named_steps["clf"]
    df = pd.DataFrame(clf.coef_.T, index=names, columns=clf.classes_)
    df["max_abs"] = df.abs().max(axis=1)
    return df.sort_values("max_abs", ascending=False).drop(columns="max_abs")


def plot_top_coefficients(coef_df: pd.DataFrame, title: str, path: Path, n: int = 15) -> None:
    top = coef_df.head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#C44E52" if v < 0 else "#4C72B0" for v in top["coefficient"]]
    ax.barh(top["feature"], top["coefficient"], color=colors)
    ax.set_title(title)
    ax.set_xlabel("Standardized coefficient (effect on log(price) per 1 std-dev)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    _, reg_models, *_ = run_regression()
    lin_coef = linear_regression_coefficients(reg_models)
    lin_coef.to_csv(TAB_DIR / "linear_regression_coefficients.csv", index=False)
    plot_top_coefficients(lin_coef, "Top drivers of log(selling_price) - Linear Regression",
                           FIG_DIR / "linear_regression_top_coefficients.png")

    print("Top 10 Linear Regression coefficients (standardized):")
    print(lin_coef.head(10).to_string(index=False))

    _, _, _, _, clf_models, *_ = run_classification()
    log_coef = logistic_regression_coefficients(clf_models)
    log_coef.to_csv(TAB_DIR / "logistic_regression_coefficients.csv")
    print("\nTop 10 Logistic Regression coefficients (by max |coef| across classes):")
    print(log_coef.head(10).round(3).to_string())

    print(f"\nSaved -> {TAB_DIR}")


if __name__ == "__main__":
    main()
