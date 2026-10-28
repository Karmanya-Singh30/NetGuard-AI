import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src import config
from src.data_preprocessing import build_preprocessor, signed_log1p
from src.predict import predict, predict_family, validate_frame
from src.train import (candidate_models, fit_large, make_pipeline, multiclass_weight_param,
                       resolve_fit_cap)


def small_xy(n=80, seed=0):
    rng = np.random.default_rng(seed)
    y = pd.Series(np.tile([0, 1], n // 2))
    X = pd.DataFrame({"a": rng.normal(size=n) + y * 3, "b": rng.lognormal(size=n),
                      "c": rng.normal(size=n)})
    return X, y


# Pipeline fit / predict -------------------------------------------------------
@pytest.mark.parametrize("est,scale", [
    (RandomForestClassifier(n_estimators=10, random_state=0), False),
    (LogisticRegression(max_iter=500), True),
])
def test_pipeline_fit_predict_proba_shape_and_range(est, scale):
    X, y = small_xy()
    pipe = make_pipeline(["a", "b", "c"], est, scale=scale).fit(X, y)
    p = pipe.predict_proba(X)
    assert p.shape == (len(X), 2)
    assert ((p >= 0) & (p <= 1)).all() and np.allclose(p.sum(axis=1), 1)
    assert list(pipe.classes_) == [0, 1]
    assert set(pipe.predict(X)) <= {0, 1}


def test_fixed_seed_gives_identical_models():
    X, y = small_xy()
    fit = lambda: make_pipeline(["a", "b", "c"], RandomForestClassifier(
        n_estimators=10, random_state=3), scale=False).fit(X, y).predict_proba(X)
    assert np.array_equal(fit(), fit())


def test_candidate_models_are_seeded():
    for name, (est, _, _, _) in candidate_models(quick=True, n_jobs=1, seed=5).items():
        # Dummy and lbfgs LogisticRegression are deterministic and ignore random_state
        if name in ("Decision Tree", "Random Forest", "HistGradientBoosting"):
            assert est.get_params()["random_state"] == 5, name


@pytest.mark.filterwarnings("ignore:Skipping features without any observed values")
def test_pipeline_handles_nan_rows_and_all_nan_training_column():
    X, y = small_xy()
    X["empty"] = np.nan
    pipe = make_pipeline(["a", "b", "c", "empty"],
                         RandomForestClassifier(n_estimators=5, random_state=0), scale=False)
    pipe.fit(X, y)
    row = X.head(1).copy()
    row.loc[:, :] = np.nan
    assert pipe.predict_proba(row).shape == (1, 2)


# Leakage guards ---------------------------------------------------------------
def test_imputer_medians_come_from_training_data_only(trained):
    pipe, train, test, feats, _ = trained
    imp = pipe.named_steps["pre"].named_steps["impute"]
    expected = train[feats].median(skipna=True).to_numpy()
    keep = ~np.isnan(expected)
    assert np.allclose(imp.statistics_[keep], expected[keep])
    assert not np.allclose(imp.statistics_[keep], test[feats].median().to_numpy()[keep])


def test_constant_filter_learned_on_train_not_inference_data():
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "k": [7.0] * 4})
    pre = build_preprocessor(scale=False).fit(train)
    test = pd.DataFrame({"a": [1.0, 2.0], "k": [100.0, -5.0]})   # k varies at inference
    assert pre.transform(test).shape == (2, 1)


def test_inference_is_row_independent(trained):
    """Predicting a row alone or inside a batch gives the same score (no
    statistics are recomputed from inference data)."""
    pipe, _, test, feats, _ = trained
    batch = pipe.predict_proba(test[feats].head(20))[:, 1]
    single = [pipe.predict_proba(test[feats].iloc[[i]])[0, 1] for i in range(3)]
    assert np.allclose(batch[:3], single)


# Schema enforcement in predict.py ----------------------------------------------
def test_validated_frame_feeds_pipeline_in_schema_order(trained):
    pipe, _, test, feats, schema = trained
    raw = test[feats].head(10)
    shuffled = raw[feats[::-1]].assign(**{"Label": "BENIGN", "unused": 1})
    v = validate_frame(shuffled, schema)
    assert v.ok and list(v.X.columns) == feats and v.X.dtypes.eq(np.float32).all()
    assert np.allclose(pipe.predict_proba(v.X), pipe.predict_proba(raw.astype(np.float32)))


def test_validate_frame_lists_every_missing_column(trained):
    _, _, test, feats, schema = trained
    v = validate_frame(test[feats[2:]].head(3), schema)
    assert not v.ok and feats[0] in v.errors[0] and feats[1] in v.errors[0]
    assert "2 required" in v.errors[0]


def test_validate_frame_warns_on_duplicate_headers():
    df = pd.DataFrame([[1.0, 2.0]], columns=["a", " a"])
    v = validate_frame(df, {"features": ["a"]})
    assert v.ok and v.X["a"].iloc[0] == 1.0
    assert any("Duplicate column" in w for w in v.warnings)


def test_validate_frame_does_not_mutate_input():
    df = pd.DataFrame({" a": ["1", "Infinity"]})
    before = df.copy()
    validate_frame(df, {"features": ["a"]})
    pd.testing.assert_frame_equal(df, before)


def test_validate_frame_negative_infinity_becomes_missing():
    v = validate_frame(pd.DataFrame({"a": ["-Infinity", "-inf", "3"]}), {"features": ["a"]})
    assert v.ok and v.X["a"].isna().tolist() == [True, True, False]


def test_predict_threshold_and_index(trained):
    pipe, _, test, feats, _ = trained
    X = test[feats].head(30)
    X.index = range(100, 130)
    p = pipe.predict_proba(X)[:, 1]
    t = float(np.median(p))
    out = predict(pipe, X, threshold=t)
    assert list(out.index) == list(X.index)
    assert ((out["prediction"] == config.POSITIVE_LABEL) == (p >= t)).all()
    assert (predict(pipe, X, threshold=0.0)["prediction"] == config.POSITIVE_LABEL).all()


def test_predict_family_returns_top_class(cleaned):
    from src.data_preprocessing import feature_columns
    df, _ = cleaned
    feats = feature_columns(df)
    pipe = make_pipeline(feats, RandomForestClassifier(n_estimators=10, random_state=0),
                         scale=False).fit(df[feats], df[config.FAMILY_TARGET])
    X = df[feats].head(15)
    out = predict_family(pipe, X)
    proba = pipe.predict_proba(X)
    assert (out["predicted_family"].to_numpy() == pipe.classes_[proba.argmax(1)]).all()
    assert np.allclose(out["family_confidence"], proba.max(1), atol=1e-6)


# Training helpers --------------------------------------------------------------
def test_fit_large_caps_rf_bootstrap_only_above_threshold(monkeypatch):
    import src.train as tr
    X, y = small_xy()
    pipe = make_pipeline(["a", "b", "c"], RandomForestClassifier(n_estimators=3, random_state=0),
                         scale=False)
    fit_large(pipe, X, y)
    assert pipe[-1].max_samples is None
    monkeypatch.setattr(tr, "RF_MAX_SAMPLES", 40)
    fit_large(pipe, X, y)
    assert pipe[-1].max_samples == pytest.approx(40 / len(X))


def test_resolve_fit_cap(monkeypatch):
    import src.train as tr
    assert resolve_fit_cap("all")[0] is None
    assert resolve_fit_cap("5000")[0] == 5000
    monkeypatch.setattr(tr, "total_ram_gb", lambda: 8.0)
    assert resolve_fit_cap("auto")[0] == tr.LOW_RAM_FIT_CAP
    monkeypatch.setattr(tr, "total_ram_gb", lambda: 64.0)
    assert resolve_fit_cap("auto")[0] is None


def test_multiclass_weight_param():
    assert multiclass_weight_param(RandomForestClassifier()) == {"class_weight": "balanced_subsample"}
    assert multiclass_weight_param(LogisticRegression()) == {"class_weight": "balanced"}
    from sklearn.dummy import DummyClassifier
    assert multiclass_weight_param(DummyClassifier()) == {}


def test_signed_log1p_casts_ints_and_is_odd():
    out = signed_log1p(np.array([[-3, 0, 3]]))
    assert out.dtype == np.float64
    assert np.allclose(out[0, 0], -out[0, 2])


# Saved artifacts (skip if training has not been run) ---------------------------
@pytest.fixture(scope="module")
def saved():
    if not (config.FINAL_PIPELINE_PATH.exists() and config.SCHEMA_PATH.exists()):
        pytest.skip("models/*.joblib not found - run `python -m src.train` first")
    from src.predict import load_pipeline, load_schema
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # sklearn version mismatch warnings
        return load_pipeline(), load_schema()


def test_saved_pipeline_matches_schema(saved):
    pipe, schema = saved
    assert list(pipe.named_steps["select"].columns) == schema["features"]
    assert not set(schema["features"]) & (config.NON_FEATURE_COLS | config.EXCLUDED_FEATURES)
    assert list(pipe.classes_) == [0, 1]


def test_saved_pipeline_predicts_on_schema_medians(saved):
    pipe, schema = saved
    stats = schema["training_feature_stats"]
    X = pd.DataFrame([{f: stats[f]["median"] for f in schema["features"]}] * 2)
    v = validate_frame(X, schema)
    assert v.ok
    out = predict(pipe, v.X)
    assert len(out) == 2 and out["p_attack"].between(0, 1).all()
    with pytest.raises(KeyError, match="Missing required"):
        pipe.predict_proba(v.X.iloc[:, 1:])
