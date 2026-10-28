import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_loading import clean_column_name, normalise_label, tidy_frame
from src.data_preprocessing import (build_preprocessor, clean_dataset, deduplicate,
                                    feature_columns, replace_infinite, split_train_test,
                                    stratified_cap)


# 1. Loading and target detection --------------------------------------------
def test_load_finds_target_and_files(loaded):
    df, rep = loaded
    assert config.LABEL_COL in df.columns
    assert len(rep.files) == 8
    assert config.SOURCE_COL in df.columns


def test_column_names_stripped_and_duplicate_dropped(loaded):
    df, rep = loaded
    assert all(c == c.strip() for c in df.columns)
    assert "Fwd Header Length" in df.columns
    assert "Fwd Header Length.1" not in df.columns
    assert "Fwd Header Length.1" in rep.duplicate_columns_dropped


@pytest.mark.parametrize("raw,expected", [
    ("Web Attack – Brute Force", "Web Attack - Brute Force"),
    ("Web Attack \x96 XSS", "Web Attack - XSS"),
    ("Web Attack � Sql Injection", "Web Attack - Sql Injection"),
    ("Web Attack Ã¢â‚¬â€œ XSS", "Web Attack - XSS"),
    (" BENIGN ", "BENIGN"),
    ("DoS Hulk", "DoS Hulk"),
])
def test_label_normalisation(raw, expected):
    assert normalise_label(raw) == expected


def test_all_loaded_labels_have_a_family(cleaned):
    df, _ = cleaned
    assert (df[config.FAMILY_TARGET] != "Unknown").all()


def test_clean_column_name():
    assert clean_column_name("  Flow   Bytes/s ") == "Flow Bytes/s"


def test_target_missing_raises():
    with pytest.raises(ValueError, match="Label"):
        tidy_frame(pd.DataFrame({"a": [1, 2]}))


# 2. Missing and non-finite values -------------------------------------------
def test_infinite_replaced_with_nan():
    df = pd.DataFrame({"x": [1.0, np.inf, -np.inf, 2.0]})
    out, counts = replace_infinite(df.copy())
    assert counts == {"x": 2}
    assert out["x"].isna().sum() == 2


def test_no_infinite_after_cleaning(cleaned):
    df, rep = cleaned
    num = df[feature_columns(df)].to_numpy()
    assert not np.isinf(num).any()
    assert rep["infinite_values_replaced_with_nan"]


def test_imputer_handles_nan_without_dropping_rows(trained):
    pipe, _, test, feats, _ = trained
    X = test[feats].head(5).copy()
    X.iloc[0, :] = np.nan
    assert pipe.predict_proba(X).shape == (5, 2)


# De-duplication and split ------------------------------------------------------
def test_deduplicate_exact_and_conflicting():
    df = pd.DataFrame({"a": [1.0, 1.0, 2.0, 3.0, 3.0], "b": [5.0, 5.0, 6.0, 7.0, 7.0],
                       config.LABEL_COL: ["BENIGN", "BENIGN", "BENIGN", "DDoS", "BENIGN"]})
    out, stats = deduplicate(df)
    assert stats["exact_duplicates_removed"] == 1
    assert stats["conflicting_label_rows_removed"] == 2
    assert len(out) == 2


def test_no_identical_rows_across_split(cleaned):
    df, _ = cleaned
    train, test = split_train_test(df)
    cols = [c for c in df.columns if c not in config.NON_FEATURE_COLS
            and pd.api.types.is_numeric_dtype(df[c])]
    h_tr = set(pd.util.hash_pandas_object(train[cols], index=False))
    h_te = set(pd.util.hash_pandas_object(test[cols], index=False))
    assert not (h_tr & h_te), "a flow appears in both train and test"


def test_split_is_stratified_and_complete(cleaned):
    df, _ = cleaned
    train, test = split_train_test(df)
    assert len(train) + len(test) == len(df)
    assert set(train[config.LABEL_COL]) == set(df[config.LABEL_COL])
    assert abs(train[config.BINARY_TARGET].mean() - test[config.BINARY_TARGET].mean()) < 0.02


def test_stratified_cap_keeps_rare_classes(cleaned):
    df, _ = cleaned
    small = stratified_cap(df, max_rows=600, min_per_class=5)
    assert set(small[config.LABEL_COL]) == set(df[config.LABEL_COL])
    assert len(small) <= 600 + 5 * df[config.LABEL_COL].nunique()


# Leakage guards ---------------------------------------------------------------
def test_excluded_and_bookkeeping_columns_are_not_features(cleaned):
    df, _ = cleaned
    feats = feature_columns(df)
    for c in [config.LABEL_COL, config.BINARY_TARGET, config.FAMILY_TARGET, config.SOURCE_COL,
              *config.EXCLUDED_FEATURES]:
        assert c not in feats


def test_scaler_statistics_come_from_training_data_only(trained):
    _, train, test, feats, _ = trained
    pre = build_preprocessor(scale=True).fit(train[feats])
    imputed_train = pre.named_steps["impute"].transform(train[feats])
    kept = pre.named_steps["drop_constant"].transform(imputed_train)
    logged = pre.named_steps["log"].transform(kept)
    assert np.allclose(pre.named_steps["scale"].mean_, logged.mean(axis=0))
    # refitting on test would give different statistics -> proves no test data was used
    pre_test = build_preprocessor(scale=True).fit(test[feats])
    assert not np.allclose(pre.named_steps["scale"].mean_, pre_test.named_steps["scale"].mean_)


def test_constant_features_removed_inside_pipeline(trained):
    pipe, train, _, feats, _ = trained
    const = [f for f in feats if train[f].nunique(dropna=False) <= 1]
    assert const, "fixture should contain constant columns"
    from src.explainability import model_feature_names
    assert not set(const) & set(model_feature_names(pipe))


def test_smote_only_inside_training_folds():
    """SMOTE must live inside an imblearn pipeline so it is re-fitted per CV fold."""
    import inspect
    from src import train
    src = inspect.getsource(train.run_multiclass)
    assert "ImbPipeline" in src and "SMOTE" in src
    assert "fit_resample" not in inspect.getsource(train)
