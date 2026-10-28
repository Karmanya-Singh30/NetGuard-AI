"""Cleaning, target construction, de-duplication and leak-free splitting.

Order of operations (see docs/METHODOLOGY.md for the reasoning):
  1. tidy columns / labels             (data_loading.tidy_frame)
  2. +/-inf  -> NaN                     (replace_infinite)
  3. add binary + family targets        (add_targets)
  4. remove exact duplicates and rows whose identical features carry
     conflicting labels                 (deduplicate)
  5. stratified train/test split        (split_train_test)
  6. everything *learned* (imputation medians, scaling, constant-feature
     removal, feature selection, resampling) is fitted inside a sklearn
     Pipeline on the training data only  (build_preprocessor)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from . import config
from .data_loading import family_of

log = logging.getLogger(__name__)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric columns that are allowed as model inputs."""
    return [c for c in df.columns
            if c not in config.NON_FEATURE_COLS
            and c not in config.EXCLUDED_FEATURES
            and pd.api.types.is_numeric_dtype(df[c])]


def replace_infinite(df: pd.DataFrame, cols: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    cols = cols or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    counts = {}
    for c in cols:
        mask = np.isinf(df[c].to_numpy())
        n = int(mask.sum())
        if n:
            counts[c] = n
            df.loc[mask, c] = np.nan
    return df, counts


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df[config.BINARY_TARGET] = (df[config.LABEL_COL] != config.NEGATIVE_LABEL).astype(np.int8)
    df[config.FAMILY_TARGET] = df[config.LABEL_COL].map(family_of)
    unknown = df.loc[df[config.FAMILY_TARGET] == "Unknown", config.LABEL_COL].unique()
    if len(unknown):
        log.warning("Labels without a family mapping (kept as 'Unknown'): %s", list(unknown))
    return df


def _row_hash(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    # 64-bit row hashes are far cheaper than DataFrame.duplicated on 2.8M x 78
    # floats; NaN hashes consistently so NaN rows compare equal.
    return pd.util.hash_pandas_object(df[cols], index=False)


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove exact duplicates and label-conflicting duplicates.

    * Exact duplicates (same features AND same label) are collapsed to one row.
      Kept, they would appear on both sides of a random split and inflate test
      scores (the model would be 'tested' on rows it memorised).
    * Rows whose features are identical but whose labels differ cannot be
      separated by any model and indicate labelling noise; all such rows are
      removed and counted.
    Duplicates are judged on *all* numeric flow features, including
    Destination Port, so two flows are only 'the same' if every field matches.
    """
    cols = [c for c in df.columns if c not in config.NON_FEATURE_COLS
            and pd.api.types.is_numeric_dtype(df[c])]
    n0 = len(df)
    h = _row_hash(df, cols)
    key = pd.DataFrame({"h": h.values, "y": df[config.LABEL_COL].values})

    exact_dup = key.duplicated(keep="first").to_numpy()
    by_label_before = df[config.LABEL_COL].value_counts()
    removed_exact_by_label = df.loc[exact_dup, config.LABEL_COL].value_counts()
    df = df.loc[~exact_dup]
    key = key.loc[~exact_dup]

    n_labels_per_hash = key.groupby("h")["y"].transform("nunique").to_numpy()
    conflict = n_labels_per_hash > 1
    removed_conflict_by_label = df.loc[conflict, config.LABEL_COL].value_counts()
    df = df.loc[~conflict].reset_index(drop=True)

    stats = {
        "rows_before": int(n0),
        "exact_duplicates_removed": int(exact_dup.sum()),
        "conflicting_label_rows_removed": int(conflict.sum()),
        "rows_after": int(len(df)),
        "rows_before_by_label": {k: int(v) for k, v in by_label_before.items()},
        "exact_duplicates_removed_by_label": {k: int(v) for k, v in removed_exact_by_label.items()},
        "conflicting_removed_by_label": {k: int(v) for k, v in removed_conflict_by_label.items()},
        "rows_after_by_label": {k: int(v) for k, v in df[config.LABEL_COL].value_counts().items()},
    }
    return df, stats


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Steps 2–4 above. Returns the cleaned frame and a JSON-able report."""
    numeric = [c for c in df.columns if c not in config.NON_FEATURE_COLS
               and pd.api.types.is_numeric_dtype(df[c])]
    df, inf_counts = replace_infinite(df, numeric)
    df = add_targets(df)
    df, dedup = deduplicate(df)
    report = {"infinite_values_replaced_with_nan": inf_counts,
              "missing_after_inf_replacement": {c: int(v) for c, v in df[numeric].isna().sum().items() if v},
              "deduplication": dedup}
    return df, report


def stratify_key(df: pd.DataFrame) -> pd.Series:
    """Stratify on the fine label so even tiny classes appear in both splits.
    Classes with a single row cannot be stratified; they are merged into their
    family for stratification purposes only (labels are unchanged)."""
    key = df[config.LABEL_COL].astype(str)
    vc = key.value_counts()
    singletons = vc[vc < 2].index
    if len(singletons):
        key = key.where(~key.isin(singletons), df[config.FAMILY_TARGET].astype(str))
        vc = key.value_counts()
        key = key.where(~key.isin(vc[vc < 2].index), "__rare__")
    return key


def split_train_test(df: pd.DataFrame, test_size: float = config.TEST_SIZE,
                     seed: int = config.RANDOM_STATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr_idx, te_idx = train_test_split(df.index, test_size=test_size, random_state=seed,
                                      stratify=stratify_key(df))
    return df.loc[tr_idx].reset_index(drop=True), df.loc[te_idx].reset_index(drop=True)


def stratified_cap(df: pd.DataFrame, max_rows: int | None, seed: int = config.RANDOM_STATE,
                   min_per_class: int = 2000) -> pd.DataFrame:
    """Reproducible subsample for tuning / low-memory training.

    Every class keeps at least `min_per_class` rows (or all of them if it has
    fewer), so rare attacks are never discarded; the remaining budget is shared
    proportionally. Returns df unchanged if it is already small enough.
    """
    if max_rows is None or len(df) <= max_rows:
        return df
    y = df[config.LABEL_COL]
    counts = y.value_counts()
    keep = counts.clip(upper=min_per_class)
    budget = max(max_rows - int(keep.sum()), 0)
    extra = (counts - keep)
    if extra.sum() > 0 and budget > 0:
        keep = keep + np.floor(extra / extra.sum() * budget).astype(int)
    parts = [g.sample(n=int(min(keep[lbl], len(g))), random_state=seed)
             for lbl, g in df.groupby(config.LABEL_COL, observed=True)]
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Learned preprocessing (always inside a Pipeline)
# ----------------------------------------------------------------------------
def signed_log1p(X):
    """sign(x) * log(1 + |x|): compresses heavy-tailed byte/time features while
    keeping the sign (some CICIDS fields such as Init_Win_bytes can be -1)."""
    X = np.asarray(X)
    if X.dtype not in (np.float32, np.float64):
        X = X.astype(np.float64)
    # In place and dtype-preserving: float32 input stays float32 (half the memory
    # on millions of rows) and no full-size temporaries are created.
    out = np.abs(X)
    np.log1p(out, out=out)
    np.copysign(out, X, out=out)
    return out


def build_preprocessor(scale: bool) -> Pipeline:
    """Imputer -> drop constant columns -> (log + standardise, for linear models)."""
    steps = [
        ("impute", SimpleImputer(strategy="median")),
        ("drop_constant", VarianceThreshold(threshold=0.0)),
    ]
    if scale:
        steps += [("log", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")),
                  ("scale", StandardScaler())]
    return Pipeline(steps)


def smote_floor_strategy(y, floor: int = 20_000) -> dict:
    """SMOTE sampling strategy: raise every class below `floor` rows up to `floor`
    and leave larger classes untouched. imblearn's default would copy every class up
    to the majority size, i.e. tens of millions of synthetic rows on CICIDS2017.
    Lives here (an importable module) so pipelines that contain it can be loaded."""
    counts = pd.Series(np.asarray(y)).value_counts()
    return {c: int(floor) for c, n in counts.items() if n < floor}
