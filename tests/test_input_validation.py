import io

import numpy as np
import pandas as pd

from src.predict import read_uploaded_csv, validate_frame

SCHEMA = {"features": ["Flow Duration", "Total Fwd Packets", "Flow Bytes/s"]}


def frame(**over):
    d = {"Flow Duration": [10.0, 20.0], "Total Fwd Packets": [1, 2], "Flow Bytes/s": [5.0, 6.0]}
    d.update(over)
    return pd.DataFrame(d)


def test_valid_frame_passes():
    v = validate_frame(frame(), SCHEMA)
    assert v.ok and v.X.shape == (2, 3) and list(v.X.columns) == SCHEMA["features"]


def test_missing_required_column_is_rejected():
    v = validate_frame(frame().drop(columns="Flow Bytes/s"), SCHEMA)
    assert not v.ok and "Flow Bytes/s" in v.errors[0]


def test_non_numeric_values_are_rejected_with_column_name():
    v = validate_frame(frame(**{"Total Fwd Packets": ["1", "abc"]}), SCHEMA)
    assert not v.ok and "Total Fwd Packets" in v.errors[0] and "abc" in v.errors[0]


def test_infinity_strings_are_accepted_and_treated_as_missing():
    v = validate_frame(frame(**{"Flow Bytes/s": ["Infinity", "NaN"]}), SCHEMA)
    assert v.ok and v.X["Flow Bytes/s"].isna().all()
    assert any("infinite" in w for w in v.warnings)


def test_leading_spaces_in_headers_and_reordering():
    df = frame()[["Flow Bytes/s", "Flow Duration", "Total Fwd Packets"]]
    df.columns = [" Flow Bytes/s", " Flow Duration", " Total Fwd Packets"]
    v = validate_frame(df, SCHEMA)
    assert v.ok and list(v.X.columns) == SCHEMA["features"]
    assert v.X["Flow Duration"].tolist() == [10.0, 20.0]


def test_extra_columns_and_label_are_ignored_but_reported():
    df = frame().assign(**{" Label": ["BENIGN", "DDoS"], "Destination Port": [80, 443]})
    v = validate_frame(df, SCHEMA)
    assert v.ok and "Label" not in v.X.columns and "Destination Port" not in v.X.columns
    assert v.reference_labels.tolist() == ["BENIGN", "DDoS"]
    assert any("extra column" in w for w in v.warnings)


def test_empty_and_oversized_inputs():
    assert not validate_frame(frame().iloc[0:0], SCHEMA).ok
    assert not validate_frame(frame(), SCHEMA, max_rows=1).ok


def test_read_uploaded_csv_handles_garbage_and_empty():
    df, err = read_uploaded_csv(io.BytesIO(b""))
    assert df is None and "empty" in err
    df, err = read_uploaded_csv(io.BytesIO(b"\x00\x01\x02binary\xff\xfe"))
    # either rejected outright or parsed into something validate_frame then rejects
    assert err is not None or not validate_frame(df, SCHEMA).ok


def test_read_uploaded_csv_parses_cicids_style_file():
    raw = " Flow Duration, Total Fwd Packets,Flow Bytes/s, Label\n1,2,Infinity,BENIGN\n".encode()
    df, err = read_uploaded_csv(io.BytesIO(raw))
    assert err is None
    v = validate_frame(df, SCHEMA)
    assert v.ok and np.isnan(v.X.iloc[0]["Flow Bytes/s"])
