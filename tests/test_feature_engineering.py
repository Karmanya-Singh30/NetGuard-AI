import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (ColumnSelector, compare_feature_sets, correlation_filter,
                                     mutual_info_ranking, rf_importance_ranking)


def toy(n=200, seed=0):
    """'signal' separates the classes, 'copy' duplicates it, 'noise' is random."""
    rng = np.random.default_rng(seed)
    y = pd.Series(np.repeat([0, 1], n // 2))
    signal = y * 5.0 + rng.normal(0, 0.5, n)
    X = pd.DataFrame({"signal": signal, "copy": signal * 2 + 1,
                      "noise": rng.normal(0, 1, n), "const": 3.0})
    return X, y


# ColumnSelector ---------------------------------------------------------------
def test_column_selector_orders_and_drops_extras():
    X = pd.DataFrame({"b": [2], "extra": [9], "a": [1]})
    sel = ColumnSelector(["a", "b"]).fit(X)
    assert list(sel.transform(X).columns) == ["a", "b"]
    assert sel.n_features_in_ == 2
    assert list(sel.get_feature_names_out()) == ["a", "b"]


def test_column_selector_missing_column_raises_on_fit_and_transform():
    sel = ColumnSelector(["a", "b"])
    with pytest.raises(KeyError, match=r"\['b'\]"):
        sel.fit(pd.DataFrame({"a": [1]}))
    sel.fit(pd.DataFrame({"a": [1], "b": [2]}))
    with pytest.raises(KeyError, match="Missing required"):
        sel.transform(pd.DataFrame({"b": [1]}))


def test_column_selector_rejects_unnamed_arrays():
    with pytest.raises(TypeError, match="DataFrame"):
        ColumnSelector(["a"]).fit(np.zeros((2, 1)))


# Rankings / filters -----------------------------------------------------------
def test_correlation_filter_drops_copy_and_constant():
    X, _ = toy()
    assert correlation_filter(X) == ["signal", "noise"]


def test_correlation_filter_handles_nan():
    X, _ = toy()
    X.loc[:10, "noise"] = np.nan
    assert correlation_filter(X) == ["signal", "noise"]


def test_mutual_info_ranking_puts_signal_above_noise():
    X, y = toy()
    mi = mutual_info_ranking(X, y, seed=0)
    assert set(mi.index) == set(X.columns)
    assert mi.is_monotonic_decreasing
    assert mi["signal"] > mi["noise"]


def test_rf_importance_ranking_is_deterministic_and_sums_to_one():
    X, y = toy()
    a = rf_importance_ranking(X, y, seed=0, n_estimators=10)
    b = rf_importance_ranking(X, y, seed=0, n_estimators=10)
    pd.testing.assert_series_equal(a, b)
    assert np.isclose(a.sum(), 1.0) and a.is_monotonic_decreasing
    assert a["const"] == 0.0


def test_compare_feature_sets_picks_smallest_set_within_tolerance():
    X, y = toy(120)
    res = compare_feature_sets(X, y, top_k=1, seed=0, n_estimators=5, tolerance=1.0, cv_folds=2)
    sizes = {r["feature_set"]: r["n_features"] for r in res["comparison"]}
    assert sizes[res["chosen"]] == min(sizes.values())
    assert res["chosen_features"] == res["sets"][res["chosen"]]
    assert res["sets"]["all_features"] == list(X.columns)
    assert len(res["sets"]["top1_mutual_info"]) == 1
