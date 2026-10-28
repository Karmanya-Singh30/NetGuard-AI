"""Model explanations.

Important: feature importance describes what the *model* relies on, not what
*causes* an attack. A feature can rank highly because it correlates with the
way this lab testbed generated traffic.
"""
from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from . import config
from .evaluate import PALETTE, _finish

log = logging.getLogger(__name__)


def model_feature_names(pipeline) -> list[str]:
    """Names of the columns that actually reach the final estimator."""
    pre = pipeline[:-1]
    try:
        return list(pre.get_feature_names_out())
    except Exception:  # pragma: no cover
        return list(pipeline.named_steps["select"].columns)


def builtin_importance(pipeline) -> pd.Series | None:
    est = pipeline[-1]
    names = model_feature_names(pipeline)
    if hasattr(est, "feature_importances_"):
        return pd.Series(est.feature_importances_, index=names).sort_values(ascending=False)
    if hasattr(est, "coef_"):
        coef = np.ravel(est.coef_) if est.coef_.shape[0] == 1 else np.abs(est.coef_).mean(0)
        return pd.Series(coef, index=names).sort_values(key=np.abs, ascending=False)
    return None


def permutation_ranking(pipeline, X: pd.DataFrame, y, n_repeats: int = 5,
                        max_rows: int = 20_000, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Drop in test F1 when one input column is shuffled (on held-out data)."""
    y = np.asarray(y)
    if len(X) > max_rows:
        pos = np.random.default_rng(seed).choice(len(X), size=max_rows, replace=False)
        X, y = X.iloc[pos], y[pos]
    r = permutation_importance(pipeline, X, y, scoring="f1", n_repeats=n_repeats,
                               random_state=seed, n_jobs=1)
    cols = list(X.columns)
    return (pd.DataFrame({"feature": cols, "importance_mean": r.importances_mean,
                          "importance_std": r.importances_std})
            .sort_values("importance_mean", ascending=False).reset_index(drop=True))


def plot_importance(series: pd.Series, title: str, xlabel: str, path, top=15, err=None):
    s = series.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.35 * len(s) + 1.5))
    ax.barh(s.index, s.values, xerr=None if err is None else err.reindex(s.index).values,
            color=PALETTE["primary"])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    return _finish(fig, path)


# ----------------------------------------------------------------------------
# SHAP (optional — only used if installed and compatible with the model)
# ----------------------------------------------------------------------------
def shap_available() -> bool:
    try:
        import shap  # noqa: F401
        return True
    except Exception:
        return False


def _tree_explainer(pipeline):
    import shap
    est = pipeline[-1]
    return shap.TreeExplainer(est)


def linear_contributions(pipeline, X: pd.DataFrame) -> pd.DataFrame | None:
    """Exact per-row contributions for a linear model: coef_j * x_j (on the
    transformed, standardised inputs). They sum, with the intercept, to the
    model's log-odds of ATTACK."""
    est = pipeline[-1]
    if not hasattr(est, "coef_") or est.coef_.shape[0] != 1:
        return None
    Xt = np.asarray(pipeline[:-1].transform(X), dtype=float)
    out = pd.DataFrame(Xt * est.coef_[0], columns=model_feature_names(pipeline), index=X.index)
    out.attrs["method"] = "coefficient × standardised value"
    return out


def explain_rows(pipeline, X: pd.DataFrame) -> pd.DataFrame | None:
    """Per-row contributions toward ATTACK.

    Linear models: exact coefficient contributions. Tree models: SHAP values
    (if the shap package supports the model). Returns a DataFrame
    (rows x model features) in units of the model's raw output, or None.
    """
    lin = linear_contributions(pipeline, X)
    if lin is not None:
        return lin
    if not shap_available():
        return None
    try:
        Xt = pipeline[:-1].transform(X)
        expl = _tree_explainer(pipeline)
        sv = expl.shap_values(np.asarray(Xt), check_additivity=False)
        sv = np.asarray(sv)
        if sv.ndim == 3:            # (rows, features, classes) -> ATTACK column
            sv = sv[:, :, 1]
        elif isinstance(expl.shap_values, list):  # pragma: no cover
            sv = sv[1]
        out = pd.DataFrame(sv, columns=model_feature_names(pipeline), index=X.index)
        out.attrs["method"] = "SHAP"
        return out
    except Exception as exc:  # unsupported model/version -> no explanation, no crash
        log.warning("SHAP explanation unavailable: %s", exc)
        return None


def shap_global(pipeline, X: pd.DataFrame, path, max_rows: int = 2000,
                seed: int = config.RANDOM_STATE) -> pd.Series | None:
    Xs = X.sample(n=min(max_rows, len(X)), random_state=seed)
    if linear_contributions(pipeline, Xs.head(1)) is not None:
        return None   # linear model: coefficients already give the exact global picture
    contrib = explain_rows(pipeline, Xs)
    if contrib is None:
        return None
    mean_abs = contrib.abs().mean().sort_values(ascending=False)
    plot_importance(mean_abs, f"SHAP: mean |contribution| to P(ATTACK) ({len(Xs):,} test flows)",
                    "Mean |SHAP value|", path)
    return mean_abs
