"""
Phase 4 - Q3b: Classification into price tiers (Budget / Mid / Premium).

Tier thresholds are the 33rd/67th percentiles of selling_price computed on the
TRAIN split ONLY, then frozen and applied to both train and test - this is the
leakage-sensitive step flagged in the plan: fitting thresholds on the full dataset
would let the test-set price distribution influence the label definition itself.

Features are exactly the regression feature set (selling_price is never a feature
here - the whole point is to predict the tier from the car's attributes).

Models per PDF Q3: Logistic Regression (multinomial), a linear SVM, and an
RBF-kernel SVM (the PDF's "Kernel methods and/or SVM").
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                              f1_score, precision_score, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, LinearSVC

from data_prep import load_clean
from split import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, get_train_test_idx

ROOT = Path(__file__).resolve().parent.parent
TAB_DIR = ROOT / "outputs" / "tables"
FIG_DIR = ROOT / "outputs" / "figures"

TIER_LABELS = ["Budget", "Mid", "Premium"]


def assign_tiers(prices: pd.Series, thresholds: tuple[float, float]) -> pd.Series:
    lo, hi = thresholds
    bins = [-np.inf, lo, hi, np.inf]
    return pd.cut(prices, bins=bins, labels=TIER_LABELS)


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), CATEGORICAL_FEATURES),
    ])


def build_models() -> dict[str, Pipeline]:
    return {
        # sklearn >=1.7 always fits a true multinomial model for lbfgs/multi-class
        # targets; the old multi_class= kwarg was removed, so this is multinomial
        # logistic regression by default already.
        "Logistic Regression": Pipeline([
            ("prep", _preprocessor()),
            ("clf", LogisticRegression(max_iter=2000)),
        ]),
        "Linear SVM": Pipeline([
            ("prep", _preprocessor()),
            ("clf", LinearSVC(max_iter=10000)),
        ]),
        "RBF-Kernel SVM": Pipeline([
            ("prep", _preprocessor()),
            ("clf", SVC(kernel="rbf")),
        ]),
    }


def evaluate(model: Pipeline, X_train, y_train, X_test, y_test) -> tuple[dict, np.ndarray]:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision (macro)": precision_score(y_test, pred, average="macro"),
        "Recall (macro)": recall_score(y_test, pred, average="macro"),
        "F1 (macro)": f1_score(y_test, pred, average="macro"),
    }
    cm = confusion_matrix(y_test, pred, labels=TIER_LABELS)
    return metrics, cm, pred


def run():
    df = load_clean()
    train_idx, test_idx = get_train_test_idx(df)

    train_prices = df.loc[train_idx, "selling_price"]
    thresholds = tuple(train_prices.quantile([1 / 3, 2 / 3]).values)

    y_train = assign_tiers(train_prices, thresholds)
    y_test = assign_tiers(df.loc[test_idx, "selling_price"], thresholds)

    X_train, X_test = df.loc[train_idx, ALL_FEATURES], df.loc[test_idx, ALL_FEATURES]

    models = build_models()
    results, confmats, reports = {}, {}, {}
    for name, pipe in models.items():
        metrics, cm, pred = evaluate(pipe, X_train, y_train, X_test, y_test)
        results[name] = metrics
        confmats[name] = cm
        reports[name] = classification_report(y_test, pred, labels=TIER_LABELS, digits=3)

    results_df = pd.DataFrame(results).T
    return results_df, confmats, reports, thresholds, models, train_idx, test_idx, y_train, y_test


def main() -> None:
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    results_df, confmats, reports, thresholds, *_ , y_train, y_test = run()

    print(f"Tier thresholds (from TRAIN prices only): "
          f"Budget < INR {thresholds[0]:,.0f} <= Mid < INR {thresholds[1]:,.0f} <= Premium\n")
    print("Train tier distribution:\n", y_train.value_counts().to_string())
    print("\nTest tier distribution:\n", y_test.value_counts().to_string())

    print("\nClassification results:")
    print(results_df.round(4).to_string())
    results_df.to_csv(TAB_DIR / "classification_results.csv")

    for name, cm in confmats.items():
        print(f"\nConfusion matrix - {name} (rows=actual, cols=predicted, order={TIER_LABELS}):")
        print(cm)
        pd.DataFrame(cm, index=TIER_LABELS, columns=TIER_LABELS).to_csv(
            TAB_DIR / f"confusion_matrix_{name.replace(' ', '_')}.csv")

    for name, rep in reports.items():
        print(f"\nClassification report - {name}:\n{rep}")

    print(f"\nSaved -> {TAB_DIR / 'classification_results.csv'} and confusion matrices")


if __name__ == "__main__":
    main()
