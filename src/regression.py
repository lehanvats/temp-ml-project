"""
Phase 3 - Q3a: Regression (Linear, Polynomial, Ridge, Bayesian Ridge), per PDF Q3's
named model list. All four share one ColumnTransformer pattern (StandardScaler on
numerics, OneHotEncoder(drop='first') on categoricals); Polynomial additionally
expands ONLY the numeric block (degree=2) before the categoricals are concatenated,
so the one-hot columns are never polynomially expanded.

Target is log(selling_price); every reported metric is converted back to rupees
(predictions exponentiated) before scoring, so MAE/RMSE are directly interpretable
and comparable across models - log-space R^2 is reported alongside as a diagnostic.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import BayesianRidge, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from data_prep import load_clean
from split import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, get_train_test_idx

ROOT = Path(__file__).resolve().parent.parent
TAB_DIR = ROOT / "outputs" / "tables"

RIDGE_ALPHAS = np.logspace(-3, 3, 13)


def _ohe() -> OneHotEncoder:
    # sparse_output=False: BayesianRidge (and a dense round-trip generally) needs a dense array.
    return OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)


def _base_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", _ohe(), CATEGORICAL_FEATURES),
    ])


def _poly_preprocessor() -> ColumnTransformer:
    poly_numeric = Pipeline([
        ("scale", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ])
    return ColumnTransformer([
        ("num_poly", poly_numeric, NUMERIC_FEATURES),
        ("cat", _ohe(), CATEGORICAL_FEATURES),
    ])


def build_models() -> dict[str, Pipeline]:
    return {
        "Linear Regression": Pipeline([("prep", _base_preprocessor()), ("reg", LinearRegression())]),
        "Polynomial Regression (deg=2)": Pipeline([("prep", _poly_preprocessor()), ("reg", LinearRegression())]),
        "Ridge Regression (RidgeCV)": Pipeline([("prep", _base_preprocessor()),
                                                  ("reg", RidgeCV(alphas=RIDGE_ALPHAS))]),
        "Bayesian Ridge Regression": Pipeline([("prep", _base_preprocessor()), ("reg", BayesianRidge())]),
    }


def evaluate(model: Pipeline, X_train, y_log_train, X_test, y_test_rupees) -> dict:
    model.fit(X_train, y_log_train)
    pred_log = model.predict(X_test)
    pred_rupees = np.exp(pred_log)

    return {
        "MAE (INR)": mean_absolute_error(y_test_rupees, pred_rupees),
        "MSE (INR^2)": mean_squared_error(y_test_rupees, pred_rupees),
        "RMSE (INR)": mean_squared_error(y_test_rupees, pred_rupees) ** 0.5,
        "R2 (rupee space)": r2_score(y_test_rupees, pred_rupees),
        "R2 (log space)": r2_score(np.log(y_test_rupees), pred_log),
    }


def run() -> tuple[pd.DataFrame, dict[str, Pipeline], pd.Index, pd.Index]:
    df = load_clean()
    train_idx, test_idx = get_train_test_idx(df)

    X_train, X_test = df.loc[train_idx, ALL_FEATURES], df.loc[test_idx, ALL_FEATURES]
    y_train_log = np.log(df.loc[train_idx, "selling_price"])
    y_test_rupees = df.loc[test_idx, "selling_price"]

    models = build_models()
    results = {}
    for name, pipe in models.items():
        results[name] = evaluate(pipe, X_train, y_train_log, X_test, y_test_rupees)

    results_df = pd.DataFrame(results).T
    return results_df, models, train_idx, test_idx


def main() -> None:
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    results_df, models, train_idx, test_idx = run()

    ridge_alpha = models["Ridge Regression (RidgeCV)"].named_steps["reg"].alpha_
    print(f"RidgeCV selected alpha = {ridge_alpha:.4g}\n")

    print("Regression results (metrics in rupee space unless noted):")
    print(results_df.round(4).to_string())

    results_df.to_csv(TAB_DIR / "regression_results.csv")
    print(f"\nSaved -> {TAB_DIR / 'regression_results.csv'}")


if __name__ == "__main__":
    main()
