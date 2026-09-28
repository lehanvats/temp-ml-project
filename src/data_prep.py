"""
Phase 1 - Data cleaning for PBL-I Group 4 (Used Car Price Prediction & Condition Classification).

Handles, in order:
  1. Exact-duplicate removal (BEFORE any split, so duplicates never straddle train/test).
  2. Unit stripping for mileage / engine / max_power / torque / seats.
  3. Torque parsing (many free-text formats -> torque_nm, torque_rpm).
  4. Dropping rows with no spec information at all; median-imputing the few residual NaNs.
  5. Feature engineering: brand (from name), car_age (from year).
  6. A capped km_driven column for modelling (raw km_driven is kept for EDA/outlier reporting).

Run directly to produce outputs/cleaned_dataset.csv and print a shape/ null-check summary.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "Group_4_Dataset.csv"
CLEAN_CSV = ROOT / "outputs" / "cleaned_dataset.csv"

REFERENCE_YEAR = 2020  # latest model year present in the dataset
KGM_TO_NM = 9.80665

SPEC_COLS = ["mileage", "engine", "max_power", "torque_nm", "seats"]


def _strip_unit(series: pd.Series, unit_pattern: str) -> pd.Series:
    """Remove a trailing unit string and coerce to float; blanks/garbage -> NaN."""
    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.str.replace(unit_pattern, "", regex=True).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


_NUM_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
_NM_LABEL_RE = re.compile(r"([\d.]+)\s*nm", re.IGNORECASE)
_KGM_LABEL_RE = re.compile(r"([\d.]+)\s*kgm", re.IGNORECASE)


def _parse_torque(raw: str) -> tuple[float, float]:
    """
    Parse the many torque text formats into (torque_nm, torque_rpm).

    Observed formats include:
      '190Nm@ 2000rpm'            -> Nm, single rpm
      '250Nm@ 1500-2500rpm'       -> Nm, rpm range (take the max)
      '380Nm(38.7kgm)@ 2500rpm'   -> BOTH units given; the explicit 'Nm' label wins
      '12.7@ 2,700(kgm@ rpm)'     -> bare number + trailing '(kgm@ rpm)' annotation -> kgm
      '22.4 kgm at 1750-2750rpm'  -> kgm label attached to the value, rpm range
      '113.75nm@ 4000rpm'         -> nm, lowercase

    Unit resolution (in priority order, since some strings carry both an Nm and
    a kgm figure e.g. '380Nm(38.7kgm)@ 2500rpm' and must not double-count):
      1. A number directly labelled 'Nm'  -> use it as-is.
      2. A number directly labelled 'kgm' -> convert to Nm.
      3. A trailing '(kgm@ rpm)' annotation with no attached unit label on the
         value itself -> the first number in the string is kgm, convert.
    rpm is taken from the remaining numbers in the string (max, for ranges).
    """
    if not isinstance(raw, str) or not raw.strip():
        return np.nan, np.nan

    nums = [float(n.replace(",", "")) for n in _NUM_RE.findall(raw)]
    if not nums:
        return np.nan, np.nan
    rpm = max(nums[1:]) if len(nums) > 1 else np.nan

    nm_match = _NM_LABEL_RE.search(raw)
    if nm_match:
        return float(nm_match.group(1)), rpm

    kgm_match = _KGM_LABEL_RE.search(raw)
    if kgm_match:
        return float(kgm_match.group(1)) * KGM_TO_NM, rpm

    if "kgm" in raw.lower():
        # e.g. '12.7@ 2,700(kgm@ rpm)': unit stated only in the trailing annotation.
        # This dataset applies the '(kgm@ rpm)' tag as a fixed template regardless
        # of the true unit - values cluster <=53 (genuine kgm readings, max ~53 kgm
        # = ~520 Nm) with a clear gap before the next cluster at >=110 (already Nm,
        # mislabeled). 60 sits in that gap, so it separates the two groups cleanly.
        value = nums[0]
        return (value * KGM_TO_NM if value <= 60 else value), rpm

    return nums[0], rpm


def load_clean() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV)
    n_start = len(df)

    # 1. Exact duplicates - drop before anything else touches the data.
    df = df.drop_duplicates().reset_index(drop=True)
    n_after_dedupe = len(df)

    # 2. Unit stripping.
    df["mileage"] = _strip_unit(df["mileage"], r"\s*(kmpl|km/kg)\s*$")
    df.loc[df["mileage"] == 0, "mileage"] = np.nan  # '0' mileage on a non-electric car is an undisclosed reading
    df["engine"] = _strip_unit(df["engine"], r"\s*CC\s*$")
    df["max_power"] = _strip_unit(df["max_power"], r"\s*bhp\s*$")
    df.loc[df["max_power"] == 0, "max_power"] = np.nan  # '0' bhp is a data-entry error, not a real reading
    df["seats"] = pd.to_numeric(df["seats"], errors="coerce")

    # 3. Torque parsing.
    torque_parsed = df["torque"].apply(_parse_torque)
    df["torque_nm"] = torque_parsed.apply(lambda t: t[0])
    df["torque_rpm"] = torque_parsed.apply(lambda t: t[1])
    df = df.drop(columns=["torque"])

    # A handful of source rows carry physically-implausible readings (e.g. one row
    # parses to 190 kgm @ 21,800 rpm - no production car engine revs anywhere near
    # that high). Treat values outside a generous plausible range as data-entry
    # errors -> NaN, to be median-imputed below rather than left to distort fits.
    implausible_torque = df["torque_nm"].notna() & ~df["torque_nm"].between(20, 700)
    implausible_rpm = df["torque_rpm"].notna() & ~df["torque_rpm"].between(500, 8000)
    n_implausible = int((implausible_torque | implausible_rpm).sum())
    df.loc[implausible_torque, "torque_nm"] = np.nan
    df.loc[implausible_rpm, "torque_rpm"] = np.nan

    # 4. Drop rows with no spec information at all; median-impute the small residual.
    all_missing = df[SPEC_COLS].isna().all(axis=1)
    n_dropped_no_spec = int(all_missing.sum())
    df = df.loc[~all_missing].reset_index(drop=True)

    residual_na_before = df[SPEC_COLS + ["torque_rpm"]].isna().sum()
    for col in SPEC_COLS + ["torque_rpm"]:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # 5. Feature engineering.
    df["brand"] = df["name"].str.split().str[0]
    df["car_age"] = REFERENCE_YEAR - df["year"]
    df = df.drop(columns=["name"])

    # 6. Capped km_driven for modelling (raw km_driven kept for EDA/outlier reporting).
    q99 = df["km_driven"].quantile(0.99)
    df["km_driven_capped"] = df["km_driven"].clip(upper=q99)

    print(f"Raw rows:                {n_start}")
    print(f"After exact dedupe:      {n_after_dedupe}  (-{n_start - n_after_dedupe})")
    print(f"Dropped (no spec data):  {n_dropped_no_spec}")
    print(f"Implausible torque/rpm readings -> NaN'd: {n_implausible}")
    print(f"Final cleaned rows:      {len(df)}")
    print("\nResidual NaNs before median-impute (per column):")
    print(residual_na_before.to_string())
    print(f"\nkm_driven q99 cap used for modelling: {q99:,.0f} km")

    assert df[SPEC_COLS + ["torque_rpm", "brand", "car_age"]].isna().sum().sum() == 0, \
        "Unexpected NaNs remain in modelling columns after cleaning."

    return df


if __name__ == "__main__":
    cleaned = load_clean()
    CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(CLEAN_CSV, index=False)
    print(f"\nSaved -> {CLEAN_CSV}  shape={cleaned.shape}")
