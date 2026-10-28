"""Unit tests for loading/cleaning helpers on tiny hand-built frames."""
import zipfile

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_loading import LoadReport, find_sources, load_cicids, tidy_frame
from src.data_preprocessing import (add_targets, clean_dataset, deduplicate, feature_columns,
                                    replace_infinite, split_train_test, stratified_cap,
                                    stratify_key)


def raw_frame():
    return pd.DataFrame({" Flow Duration": [1, 2, 3, 4],
                         "Flow Bytes/s": ["5", "Infinity", "x", "7"],
                         " Destination Port": [80, 443, 80, 22],
                         " Label": [" BENIGN", "DDoS", "benign", "Web Attack \x96 XSS"]})


def clean_frame():
    return pd.DataFrame({"a": [1.0, np.inf, 1.0, 2.0, -np.inf, 3.0],
                         "b": [0.0, 1.0, 0.0, np.nan, 2.0, 3.0],
                         config.LABEL_COL: ["BENIGN", "DDoS", "BENIGN", "BENIGN", "PortScan", "Bot"]})


# tidy_frame -------------------------------------------------------------------
def test_tidy_frame_strips_names_coerces_float32_and_normalises_labels():
    rep = LoadReport()
    out = tidy_frame(raw_frame(), rep)
    assert list(out.columns) == ["Flow Duration", "Flow Bytes/s", "Destination Port", "Label"]
    assert out["Label"].tolist() == ["BENIGN", "DDoS", "BENIGN", "Web Attack - XSS"]
    for c in ["Flow Duration", "Flow Bytes/s", "Destination Port"]:
        assert out[c].dtype == np.float32
    # "Infinity" parses as inf; "x" is genuinely non-numeric -> NaN and counted
    assert np.isinf(out.loc[1, "Flow Bytes/s"]) and np.isnan(out.loc[2, "Flow Bytes/s"])
    assert rep.coerced_non_numeric == {"Flow Bytes/s": 1}


def test_tidy_frame_does_not_mutate_input():
    raw = raw_frame()
    before = raw.copy()
    tidy_frame(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_tidy_frame_keeps_non_identical_dot_suffixed_copy():
    df = pd.DataFrame({"X": [1, 2], "X.1": [1, 3], "Label": ["BENIGN", "BENIGN"]})
    rep = LoadReport()
    out = tidy_frame(df, rep)
    assert "X.1" in out.columns and rep.duplicate_columns_dropped == []


def test_tidy_frame_drops_names_duplicated_after_stripping():
    df = pd.DataFrame([[1, 2, "BENIGN"]], columns=["A", " A", "Label"])
    rep = LoadReport()
    out = tidy_frame(df, rep)
    assert list(out.columns) == ["A", "Label"] and out["A"].iloc[0] == 1
    assert rep.duplicate_columns_dropped == ["A"]


# Loading ----------------------------------------------------------------------
def _write_csv(path, label="BENIGN"):
    path.write_text(f" Flow Duration, Label\n1,{label}\n2,{label}\n", encoding="latin-1")


def test_find_sources_reads_zip_and_skips_macosx(tmp_path):
    zpath = tmp_path / "MachineLearningCSV.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("MachineLearningCVE/Monday.csv", " Flow Duration, Label\n1,BENIGN\n")
        zf.writestr("__MACOSX/._Monday.csv", "junk")
    names = [n for n, _ in find_sources(zpath)]
    assert names == ["Monday.csv"]
    df, rep = load_cicids(zpath)
    assert len(df) == 1 and rep.files == ["Monday.csv"]


def test_find_sources_empty_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No CICIDS2017"):
        find_sources(tmp_path)


def test_load_files_filter_and_source_column(tmp_path):
    _write_csv(tmp_path / "a.csv")
    _write_csv(tmp_path / "b.csv", "DDoS")
    df, rep = load_cicids(tmp_path, files=["b.csv"])
    assert rep.files == ["b.csv"] and rep.rows_per_file == {"b.csv": 2}
    assert set(df[config.SOURCE_COL]) == {"b.csv"}
    assert isinstance(df[config.SOURCE_COL].dtype, pd.CategoricalDtype)


def test_load_keeps_only_common_columns(tmp_path):
    (tmp_path / "a.csv").write_text("x,y,Label\n1,2,BENIGN\n", encoding="latin-1")
    (tmp_path / "b.csv").write_text("x,Label\n3,DDoS\n", encoding="latin-1")
    df, _ = load_cicids(tmp_path)
    assert "y" not in df.columns and len(df) == 2


# Inf / NaN --------------------------------------------------------------------
def test_replace_infinite_leaves_nan_and_finite_values_alone():
    df = pd.DataFrame({"x": [1.0, np.nan, np.inf], "y": [2.0, 3.0, 4.0]})
    out, counts = replace_infinite(df.copy())
    assert counts == {"x": 1}
    assert out["x"].isna().tolist() == [False, True, True]
    assert out["y"].tolist() == [2.0, 3.0, 4.0]


def test_replace_infinite_skips_non_numeric_columns():
    df = pd.DataFrame({"x": [np.inf], "s": ["inf"]})
    out, counts = replace_infinite(df.copy())
    assert counts == {"x": 1} and out["s"].iloc[0] == "inf"


@pytest.mark.xfail(reason="BUG: replace_infinite uses `cols or <all numeric>`, so an explicit "
                          "empty column list is treated as 'all columns'", strict=True)
def test_replace_infinite_with_empty_column_list_changes_nothing():
    df = pd.DataFrame({"x": [np.inf]})
    _, counts = replace_infinite(df.copy(), [])
    assert counts == {}


def test_clean_dataset_reports_inf_and_remaining_missing():
    out, rep = clean_dataset(clean_frame())
    assert rep["infinite_values_replaced_with_nan"] == {"a": 2}
    assert not np.isinf(out[["a", "b"]].to_numpy()).any()
    # b's NaN survives (imputation happens later, inside the pipeline)
    assert rep["missing_after_inf_replacement"]["b"] == 1
    assert out["b"].isna().sum() == 1


@pytest.mark.xfail(reason="BUG: clean_dataset mutates its argument (replace_infinite writes "
                          "NaN in place, add_targets adds columns); callers must pass a copy",
                   strict=True)
def test_clean_dataset_does_not_mutate_input():
    df = clean_frame()
    before = df.copy()
    clean_dataset(df)
    pd.testing.assert_frame_equal(df, before)


# Targets ----------------------------------------------------------------------
def test_add_targets_binarises_and_maps_families():
    df = pd.DataFrame({config.LABEL_COL: ["BENIGN", "DoS Hulk", "Web Attack - XSS", "Mystery"]})
    out = add_targets(df)
    assert out[config.BINARY_TARGET].tolist() == [0, 1, 1, 1]
    assert out[config.BINARY_TARGET].dtype == np.int8
    assert out[config.FAMILY_TARGET].tolist() == ["BENIGN", "DoS", "WebAttack", "Unknown"]


def test_every_family_map_label_except_benign_is_an_attack():
    labels = list(config.FAMILY_MAP)
    out = add_targets(pd.DataFrame({config.LABEL_COL: labels}))
    expected = [0 if lbl == config.NEGATIVE_LABEL else 1 for lbl in labels]
    assert out[config.BINARY_TARGET].tolist() == expected


# Duplicates -------------------------------------------------------------------
def test_deduplicate_treats_nan_rows_as_equal_and_reports_by_label():
    df = pd.DataFrame({"a": [np.nan, np.nan, 1.0], "b": [1.0, 1.0, 1.0],
                       config.LABEL_COL: ["BENIGN", "BENIGN", "DDoS"]})
    out, stats = deduplicate(df)
    assert stats["exact_duplicates_removed"] == 1
    assert stats["exact_duplicates_removed_by_label"] == {"BENIGN": 1}
    assert stats["rows_before"] == 3 and stats["rows_after"] == 2 == len(out)
    assert out.index.tolist() == [0, 1]


def test_deduplicate_ignores_bookkeeping_columns():
    # same flow in two day-files is still a duplicate
    df = pd.DataFrame({"a": [1.0, 1.0], config.LABEL_COL: ["BENIGN", "BENIGN"],
                       config.SOURCE_COL: ["mon.csv", "tue.csv"]})
    out, stats = deduplicate(df)
    assert stats["exact_duplicates_removed"] == 1 and len(out) == 1


def test_deduplicate_keeps_rows_differing_in_any_feature():
    df = pd.DataFrame({"a": [1.0, 1.0], "Destination Port": [80.0, 443.0],
                       config.LABEL_COL: ["BENIGN", "DDoS"]})
    out, stats = deduplicate(df)
    assert len(out) == 2 and stats["conflicting_label_rows_removed"] == 0


# Feature selection by column -------------------------------------------------
def test_feature_columns_drops_targets_excluded_and_non_numeric():
    df = pd.DataFrame({"a": [1.0], "Destination Port": [80.0], "text": ["x"],
                       config.LABEL_COL: ["BENIGN"], config.BINARY_TARGET: [0],
                       config.FAMILY_TARGET: ["BENIGN"], config.SOURCE_COL: ["m.csv"]})
    assert feature_columns(df) == ["a"]


def test_feature_columns_preserves_column_order():
    df = pd.DataFrame({"z": [1.0], "a": [2.0], "m": [3.0]})
    assert feature_columns(df) == ["z", "a", "m"]


# Split / sampling determinism -------------------------------------------------
def test_split_is_deterministic_for_a_seed(cleaned):
    df, _ = cleaned
    a, _ = split_train_test(df, seed=7)
    b, _ = split_train_test(df, seed=7)
    c, _ = split_train_test(df, seed=8)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)


def test_stratify_key_merges_singleton_labels():
    df = pd.DataFrame({config.LABEL_COL: ["BENIGN"] * 4 + ["DoS Hulk", "DoS GoldenEye", "Heartbleed"],
                       config.FAMILY_TARGET: ["BENIGN"] * 4 + ["DoS", "DoS", "Heartbleed"]})
    key = stratify_key(df)
    assert key.tolist() == ["BENIGN"] * 4 + ["DoS", "DoS", "__rare__"]
    # labels themselves are not changed
    assert df[config.LABEL_COL].iloc[4] == "DoS Hulk"


def test_stratified_cap_no_op_and_deterministic(cleaned):
    df, _ = cleaned
    assert stratified_cap(df, None) is df
    assert stratified_cap(df, len(df) + 1) is df
    a = stratified_cap(df, 300, min_per_class=3)
    b = stratified_cap(df, 300, min_per_class=3)
    pd.testing.assert_frame_equal(a, b)
