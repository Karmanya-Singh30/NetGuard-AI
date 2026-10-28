"""Feature selection — every method here is fitted on training data only.

CICIDS2017 already ships ~78 engineered flow statistics (durations, packet
counts, byte counts, packet-length statistics, inter-arrival times, flag
counts, rates, bulk/subflow/active/idle statistics). We do not invent fields
that are absent from the source; "feature engineering" here means choosing a
compact, non-redundant subset, which the comparison in `compare_feature_sets`
justifies with cross-validation on the training split.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate

from . import config

log = logging.getLogger(__name__)


class ColumnSelector(BaseEstimator, TransformerMixin):
    """First pipeline step: select (and order) the schema columns by name.

    Guarantees that inference never silently reorders or misreads features:
    a missing column raises, extra columns are ignored.
    """

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X, y=None):
        self._check(X)
        self.n_features_in_ = len(self.columns)
        self.feature_names_in_ = np.array(self.columns, dtype=object)
        return self

    def _check(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ColumnSelector expects a pandas DataFrame with named columns")
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise KeyError(f"Missing required feature columns: {missing}")

    def transform(self, X):
        self._check(X)
        return X[self.columns]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns, dtype=object)


def correlation_filter(X: pd.DataFrame, threshold: float = 0.98) -> list[str]:
    """Greedy removal of one feature from every pair with |Pearson r| > threshold.

    Columns are visited in their original order; a column is kept unless it is
    highly correlated with an already-kept column. Constant columns are dropped.
    """
    Xf = X.astype(np.float64)
    Xf = Xf.fillna(Xf.median())
    std = Xf.std()
    Xf = Xf.loc[:, std > 0]
    corr = np.abs(np.corrcoef(Xf.to_numpy(), rowvar=False))
    cols = list(Xf.columns)
    kept: list[int] = []
    for j in range(len(cols)):
        if all(corr[j, k] <= threshold for k in kept):
            kept.append(j)
    return [cols[j] for j in kept]


def mutual_info_ranking(X: pd.DataFrame, y: pd.Series, seed: int = config.RANDOM_STATE,
                        max_rows: int = 50_000) -> pd.Series:
    if len(X) > max_rows:
        idx = X.sample(n=max_rows, random_state=seed).index
        X, y = X.loc[idx], y.loc[idx]
    Xi = SimpleImputer(strategy="median").fit_transform(X)
    mi = mutual_info_classif(Xi, y, random_state=seed)
    return pd.Series(mi, index=X.columns).sort_values(ascending=False)


def rf_importance_ranking(X: pd.DataFrame, y: pd.Series, seed: int = config.RANDOM_STATE,
                          n_estimators: int = 50) -> pd.Series:
    Xi = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(X)
    rf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=seed,
                                min_samples_leaf=2)
    rf.fit(Xi, y)
    return pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)


def compare_feature_sets(X: pd.DataFrame, y: pd.Series, top_k: int = 30,
                         seed: int = config.RANDOM_STATE, n_estimators: int = 50,
                         tolerance: float = 0.002, cv_folds: int = config.CV_FOLDS) -> dict:
    """Compare candidate feature sets with stratified CV on TRAINING data only.

    Candidates: all eligible features, correlation-filtered, top-k mutual
    information, top-k random-forest importance. The rankings themselves are
    computed on the same training subsample (never on test data).

    Selection rule (documented, simple to defend): pick the *smallest* set whose
    mean CV F1 (ATTACK) is within `tolerance` of the best set.
    """
    from sklearn.pipeline import make_pipeline

    sets: dict[str, list[str]] = {"all_features": list(X.columns)}
    sets["correlation_filtered"] = correlation_filter(X)
    mi = mutual_info_ranking(X, y, seed)
    sets[f"top{top_k}_mutual_info"] = list(mi.index[:top_k])
    rfi = rf_importance_ranking(X, y, seed)
    sets[f"top{top_k}_rf_importance"] = list(rfi.index[:top_k])

    base = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1,
                                                random_state=seed, min_samples_leaf=2))
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    rows = []
    for name, cols in sets.items():
        t0 = time.perf_counter()
        res = cross_validate(clone(base), X[cols], y, cv=cv,
                             scoring={"f1": "f1", "recall": "recall", "precision": "precision"},
                             n_jobs=1)
        rows.append({"feature_set": name, "n_features": len(cols),
                     "cv_f1_mean": float(res["test_f1"].mean()),
                     "cv_f1_std": float(res["test_f1"].std()),
                     "cv_recall_mean": float(res["test_recall"].mean()),
                     "cv_precision_mean": float(res["test_precision"].mean()),
                     "seconds": round(time.perf_counter() - t0, 1)})
        log.info("feature set %-28s n=%3d  F1=%.4f", name, len(cols), rows[-1]["cv_f1_mean"])

    best = max(r["cv_f1_mean"] for r in rows)
    eligible = [r for r in rows if r["cv_f1_mean"] >= best - tolerance]
    chosen = min(eligible, key=lambda r: (r["n_features"], -r["cv_f1_mean"]))
    return {"comparison": rows, "chosen": chosen["feature_set"],
            "chosen_features": sets[chosen["feature_set"]],
            "rule": f"smallest set with CV F1 within {tolerance} of the best",
            "mutual_info_ranking": {k: float(v) for k, v in mi.items()},
            "rf_importance_ranking": {k: float(v) for k, v in rfi.items()},
            "sets": sets}
