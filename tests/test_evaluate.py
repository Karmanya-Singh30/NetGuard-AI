"""Metric correctness on hand-computed examples (positive class = ATTACK = 1)."""
import json

import numpy as np
import pandas as pd
import pytest

from src import config
from src.evaluate import (binary_metrics, detection_by_family, multiclass_report, save_json,
                          plot_threshold_sweep)


def test_binary_metrics_hand_computed():
    # TP=3, FN=1, FP=2, TN=4
    y_true = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    y_pred = [1, 1, 1, 0, 1, 1, 0, 0, 0, 0]
    m = binary_metrics(y_true, y_pred)
    assert (m["tp"], m["fn"], m["fp"], m["tn"], m["n"]) == (3, 1, 2, 4, 10)
    assert m["accuracy"] == pytest.approx(0.7)
    assert m["precision"] == pytest.approx(3 / 5)
    assert m["recall"] == pytest.approx(3 / 4)
    assert m["f1"] == pytest.approx(2 * 0.6 * 0.75 / (0.6 + 0.75))
    assert m["specificity"] == pytest.approx(4 / 6)
    assert m["fpr"] == pytest.approx(2 / 6)
    assert m["fnr"] == pytest.approx(1 / 4)
    assert m["roc_auc"] is None and m["pr_auc"] is None   # no scores given


def test_binary_metrics_auc_hand_computed():
    y_true = [0, 0, 1, 1]
    score = [0.1, 0.4, 0.35, 0.8]
    m = binary_metrics(y_true, [0, 0, 0, 1], score)
    # 4 (pos, neg) pairs, 3 ranked correctly
    assert m["roc_auc"] == pytest.approx(0.75)
    # ranked: 0.8(+) 0.4(-) 0.35(+) 0.1(-): AP = 0.5*1 + 0.5*(2/3)
    assert m["pr_auc"] == pytest.approx(0.5 + 1 / 3)


def test_perfect_and_all_wrong_predictions():
    y = np.array([0, 1, 0, 1])
    perfect = binary_metrics(y, y, y.astype(float))
    assert perfect["f1"] == 1.0 and perfect["fpr"] == 0.0 and perfect["roc_auc"] == 1.0
    wrong = binary_metrics(y, 1 - y)
    assert wrong["accuracy"] == 0.0 and wrong["recall"] == 0.0 and wrong["precision"] == 0.0


def test_no_positive_predictions_gives_undefined_precision():
    m = binary_metrics([1, 0, 0], [0, 0, 0])
    assert m["precision"] is None
    assert m["recall"] == 0.0 and m["f1"] == 0.0


def test_binary_metrics_accepts_bools_and_floats():
    a = binary_metrics([True, False], [1.0, 0.0])
    assert (a["tp"], a["tn"]) == (1, 1)


def test_detection_by_family_rates_and_missed():
    fam = pd.Series(["BENIGN"] * 4 + ["DoS"] * 3 + ["Bot"] * 2)
    y_true = [0, 0, 0, 0, 1, 1, 1, 1, 1]
    y_pred = [0, 1, 0, 0, 1, 1, 0, 0, 0]
    out = detection_by_family(fam, y_true, y_pred).set_index("family")
    assert out.loc["BENIGN", "flagged_rate"] == pytest.approx(0.25)   # = FPR
    assert out.loc["BENIGN", "missed"] == 0
    assert out.loc["DoS", "flagged_rate"] == pytest.approx(2 / 3)     # = recall
    assert out.loc["DoS", "missed"] == 1
    assert out.loc["Bot", "missed"] == 2
    assert list(out.index) == ["BENIGN", "DoS", "Bot"]               # sorted by n


def test_multiclass_report_hand_computed():
    y_true = ["a", "a", "b", "b"]
    y_pred = ["a", "b", "b", "b"]
    rep = multiclass_report(y_true, y_pred, labels=["a", "b"])
    assert rep["a"]["precision"] == 1.0 and rep["a"]["recall"] == 0.5
    assert rep["b"]["precision"] == pytest.approx(2 / 3) and rep["b"]["recall"] == 1.0
    assert rep["accuracy"] == 0.75
    assert all(isinstance(v, float) for v in rep["macro avg"].values())


def test_threshold_sweep_matches_binary_metrics(tmp_path):
    y = np.array([0, 0, 1, 1, 1])
    s = np.array([0.1, 0.6, 0.3, 0.7, 0.9])
    _, rows = plot_threshold_sweep(y, s, tmp_path / "sweep.png")
    row = next(r for r in rows if abs(r["threshold"] - 0.5) < 1e-9)
    m = binary_metrics(y, (s >= 0.5).astype(int))
    assert row["precision"] == pytest.approx(m["precision"])
    assert row["recall"] == pytest.approx(m["recall"])
    assert row["fpr"] == pytest.approx(m["fpr"])
    assert (tmp_path / "sweep.png").exists()


def test_save_json_is_strict_json(tmp_path):
    p = save_json({"a": np.float32(0.5), "b": float("nan"), "c": np.int64(3),
                   "d": [np.inf, 1.0], 1: "key"}, tmp_path / "x" / "m.json")
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data == {"a": 0.5, "b": None, "c": 3, "d": [None, 1.0], "1": "key"}


def test_positive_label_constants():
    assert config.BINARY_CLASS_NAMES == {0: config.NEGATIVE_LABEL, 1: config.POSITIVE_LABEL}
