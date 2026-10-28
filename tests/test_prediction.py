import joblib
import numpy as np
import pandas as pd

from src import config
from src.evaluate import binary_metrics
from src.predict import predict


def test_save_and_reload_gives_identical_predictions(trained, tmp_path):
    pipe, _, test, feats, _ = trained
    path = tmp_path / "pipe.joblib"
    joblib.dump(pipe, path, compress=3)
    reloaded = joblib.load(path)
    a = pipe.predict_proba(test[feats])
    b = reloaded.predict_proba(test[feats])
    assert np.allclose(a, b)


def test_prediction_shape_labels_and_probabilities(trained):
    pipe, _, test, feats, _ = trained
    out = predict(pipe, test[feats])
    assert len(out) == len(test)
    assert set(out["prediction"]) <= {config.POSITIVE_LABEL, config.NEGATIVE_LABEL}
    assert out["p_attack"].between(0, 1).all()


def test_column_order_does_not_change_predictions(trained):
    pipe, _, test, feats, _ = trained
    X = test[feats].head(50)
    shuffled = X[list(reversed(feats))].assign(extra_column=1.0)
    assert np.allclose(pipe.predict_proba(X), pipe.predict_proba(shuffled))


def test_missing_feature_raises(trained):
    import pytest
    pipe, _, test, feats, _ = trained
    with pytest.raises(KeyError, match="Missing required"):
        pipe.predict_proba(test[feats[1:]].head(3))


def test_pipeline_trains_on_small_valid_sample(cleaned):
    from sklearn.linear_model import LogisticRegression
    from src.data_preprocessing import feature_columns
    from src.train import make_pipeline
    df, _ = cleaned
    feats = feature_columns(df)
    sample = df.groupby(config.BINARY_TARGET).head(50)
    pipe = make_pipeline(feats, LogisticRegression(max_iter=500), scale=True)
    pipe.fit(sample[feats], sample[config.BINARY_TARGET])
    assert pipe.predict(sample[feats]).shape == (len(sample),)


def test_metric_label_mapping_positive_is_attack():
    y_true = np.array([1, 1, 1, 0, 0])   # 1 = ATTACK
    y_pred = np.array([1, 0, 1, 0, 1])
    m = binary_metrics(y_true, y_pred, np.array([.9, .2, .8, .1, .7]))
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 1)
    assert m["recall"] == 2 / 3
    assert m["fpr"] == 0.5
    assert m["specificity"] == 0.5


def test_undefined_metrics_are_none_not_zero():
    m = binary_metrics(np.zeros(4), np.zeros(4))   # no attacks at all
    assert m["recall"] is None and m["fnr"] is None and m["roc_auc"] is None
    assert m["specificity"] == 1.0


def test_tree_and_linear_explanations(trained, cleaned):
    from sklearn.linear_model import LogisticRegression
    from src.explainability import explain_rows
    from src.train import make_pipeline
    pipe, train, test, feats, _ = trained
    rows = test[feats].head(3)
    tree = explain_rows(pipe, rows)
    if tree is not None:            # SHAP installed
        assert tree.shape[0] == 3 and np.isfinite(tree.to_numpy()).all()
    lin = make_pipeline(feats, LogisticRegression(max_iter=500), scale=True).fit(
        train[feats], train[config.BINARY_TARGET])
    c = explain_rows(lin, rows)
    est = lin[-1]
    logit = c.sum(axis=1) + est.intercept_[0]
    p = 1 / (1 + np.exp(-logit))
    assert np.allclose(p, lin.predict_proba(rows)[:, 1], atol=1e-6)


def test_objects_inside_saved_pipelines_are_importable():
    """Regression test: functions stored in a pickled pipeline must not live in
    src.train, which runs as __main__ under `python -m src.train`."""
    from src.data_preprocessing import signed_log1p, smote_floor_strategy
    for fn in (signed_log1p, smote_floor_strategy):
        assert fn.__module__ == "src.data_preprocessing"
    assert smote_floor_strategy(["a"] * 5 + ["b"] * 50, floor=10) == {"a": 10}


def test_signed_log1p_keeps_float32_and_sign():
    from src.data_preprocessing import signed_log1p
    x = np.array([[-1.0, 0.0, 9.0]], dtype=np.float32)
    out = signed_log1p(x)
    assert out.dtype == np.float32
    assert np.allclose(out, [[-np.log(2), 0.0, np.log(10)]])
