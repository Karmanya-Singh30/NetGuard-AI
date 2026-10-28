"""Input-realism (out-of-distribution) check: records unlike the training flows
must be flagged, real flows must not be."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.predict import realism_check, realism_summary

ROOT = Path(__file__).resolve().parents[1]


def _schema():
    stats = {
        "Flow Duration": {"min": 0.0, "p01": 3.0, "median": 3e4, "p99": 1.1e8, "max": 1.2e8},
        "Total Fwd Packets": {"min": 1.0, "p01": 1.0, "median": 2.0, "p99": 200.0, "max": 2e5},
        "PSH Flag Count": {"min": 0.0, "p01": 0.0, "median": 0.0, "p99": 1.0, "max": 1.0},
    }
    return {"features": list(stats), "training_feature_stats": stats}


def test_typical_rows_pass():
    X = pd.DataFrame({"Flow Duration": [5e4, 1e6], "Total Fwd Packets": [3, 10],
                      "PSH Flag Count": [0, 1]})
    chk = realism_check(X, _schema())
    assert (chk["input_check"] == "ok").all()
    assert not realism_summary(chk)["warn"]


def test_non_binary_flag_is_out_of_range():
    X = pd.DataFrame({"Flow Duration": [5e4], "Total Fwd Packets": [3], "PSH Flag Count": [7]})
    chk = realism_check(X, _schema())
    assert chk.loc[0, "input_check"] == "out of range"
    assert chk.loc[0, "worst_feature"] == "PSH Flag Count"
    assert realism_summary(chk)["warn"]


def test_fractional_flag_and_beyond_max_flagged():
    X = pd.DataFrame({"Flow Duration": [5e4, 5e9], "Total Fwd Packets": [3, 3],
                      "PSH Flag Count": [0.5, 0]})
    assert (realism_check(X, _schema())["input_check"] == "out of range").all()


def test_missing_values_are_ignored():
    X = pd.DataFrame({"Flow Duration": [np.nan], "Total Fwd Packets": [3.0],
                      "PSH Flag Count": [np.nan]})
    assert realism_check(X, _schema()).loc[0, "input_check"] == "ok"


def test_one_rare_row_in_a_large_file_does_not_warn():
    n = 1000
    X = pd.DataFrame({"Flow Duration": [5e4] * n, "Total Fwd Packets": [3] * n,
                      "PSH Flag Count": [0] * n})
    X.loc[0, "Flow Duration"] = 2e8        # one genuine outlier (0.1 %)
    s = realism_summary(realism_check(X, _schema()))
    assert s["out_of_range"] == 1 and not s["warn"]


# --- against the real saved artifacts (skipped if the model has not been trained) ---
real = pytest.mark.skipif(not (ROOT / "data/samples/demo_traffic_sample.csv").exists()
                          or not (ROOT / "models/feature_schema.json").exists(),
                          reason="trained artifacts not present")


@real
def test_real_demo_sample_is_not_flagged_but_fabricated_rows_are():
    from src.predict import load_schema, validate_frame
    sch = load_schema()
    demo = validate_frame(pd.read_csv(ROOT / "data/samples/demo_traffic_sample.csv"), sch).X
    assert not realism_summary(realism_check(demo, sch))["warn"]

    fake = demo.copy()
    fake["PSH Flag Count"] = np.random.default_rng(0).integers(0, 15, len(fake))
    s = realism_summary(realism_check(fake, sch))
    assert s["warn"] and s["out_of_range"] > len(fake) * 0.5
