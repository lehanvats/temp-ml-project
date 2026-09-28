"""
Shared train/test split, used identically by regression.py and classification.py so
Q4's model comparison is apples-to-apples across both tasks.

The split is stratified on a ROUGH price-tertile bin computed on the full dataset -
this only balances the split, it is NOT the classification label itself. The actual
class label thresholds are frozen from TRAIN-only price quantiles inside
classification.py, so no test-set information leaks into the label definition.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.2

NUMERIC_FEATURES = [
    "car_age", "km_driven_capped", "mileage", "engine",
    "max_power", "torque_nm", "torque_rpm", "seats",
]
CATEGORICAL_FEATURES = ["brand", "fuel", "seller_type", "transmission", "owner"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def get_train_test_idx(df: pd.DataFrame):
    rough_bin = pd.qcut(df["selling_price"], q=3, labels=False, duplicates="drop")
    train_idx, test_idx = train_test_split(
        df.index, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=rough_bin
    )
    return train_idx, test_idx
