"""Inference: load saved artifacts, validate input records, predict.

Kept separate from training so the website never trains anything. Uploaded
files are only ever parsed as CSV text — never unpickled or executed.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config
from .data_loading import clean_column_name, normalise_label

_INF_STRINGS = {"inf", "+inf", "-inf", "infinity", "+infinity", "-infinity", "nan", ""}


class ArtifactsMissing(FileNotFoundError):
    pass


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    X: pd.DataFrame | None = None
    reference_labels: pd.Series | None = None   # only if the file happened to contain Label
    n_rows: int = 0


# ----------------------------------------------------------------------------
# Artifacts
# ----------------------------------------------------------------------------
def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_schema(path: Path = config.SCHEMA_PATH) -> dict:
    if not Path(path).exists():
        raise ArtifactsMissing(f"{path} not found — run `python -m src.train` first.")
    return load_json(path)


@lru_cache(maxsize=4)
def _load_pipeline_cached(path: str, mtime: float):
    return joblib.load(path)


def load_pipeline(path: Path = config.FINAL_PIPELINE_PATH):
    """Load a pipeline produced by *this project's* training script.
    (joblib/pickle must never be used on user-supplied files.)"""
    path = Path(path)
    if not path.exists():
        raise ArtifactsMissing(f"{path} not found — run `python -m src.train` first.")
    return _load_pipeline_cached(str(path), path.stat().st_mtime)


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------
def _coerce_numeric(col: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return (numeric values, mask of cells that were genuinely non-numeric)."""
    if pd.api.types.is_numeric_dtype(col):
        return col.astype(np.float64), pd.Series(False, index=col.index)
    s = col.astype(str).str.strip()
    num = pd.to_numeric(s, errors="coerce")
    lowered = s.str.lower()
    bad = num.isna() & ~lowered.isin(_INF_STRINGS) & col.notna()
    num = num.where(~lowered.isin({"inf", "+inf", "infinity", "+infinity"}), np.inf)
    num = num.where(~lowered.isin({"-inf", "-infinity"}), -np.inf)
    return num.astype(np.float64), bad


def validate_frame(df: pd.DataFrame, schema: dict,
                   max_rows: int = config.MAX_UPLOAD_ROWS) -> ValidationResult:
    res = ValidationResult(ok=False)
    if df is None or df.shape[1] == 0:
        res.errors.append("The file contains no columns.")
        return res
    if len(df) == 0:
        res.errors.append("The file contains a header but no data rows.")
        return res
    if len(df) > max_rows:
        res.errors.append(f"Too many rows ({len(df):,}). The limit is {max_rows:,}; split the file.")
        return res

    df = df.copy()
    df.columns = [clean_column_name(c) for c in df.columns]
    if df.columns.duplicated().any():
        dups = sorted(set(df.columns[df.columns.duplicated()]))
        res.warnings.append(f"Duplicate column names {dups}: the first occurrence is used.")
        df = df.loc[:, ~df.columns.duplicated()]

    required: list[str] = schema["features"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        res.errors.append(f"Missing {len(missing)} required column(s): {missing}")
        return res

    label_col = next((c for c in df.columns if c.lower() == config.LABEL_COL.lower()), None)
    if label_col is not None:
        res.reference_labels = df[label_col].astype(str).map(normalise_label)
        res.warnings.append("The file contains a 'Label' column. It is NOT used for prediction; "
                            "it is only shown alongside the predictions for comparison.")
    extra = [c for c in df.columns if c not in required and c != label_col]
    if extra:
        res.warnings.append(f"{len(extra)} extra column(s) ignored (not used by the model): "
                            f"{extra[:8]}{' …' if len(extra) > 8 else ''}")

    X = pd.DataFrame(index=df.index)
    bad_cols = {}
    for c in required:
        num, bad = _coerce_numeric(df[c])
        if bad.any():
            bad_cols[c] = df.loc[bad, c].astype(str).head(3).tolist()
        X[c] = num
    if bad_cols:
        res.errors.append("Non-numeric values found in numeric feature column(s): " +
                          "; ".join(f"{k} (e.g. {v})" for k, v in list(bad_cols.items())[:6]))
        return res

    n_inf = int(np.isinf(X.to_numpy()).sum())
    X = X.replace([np.inf, -np.inf], np.nan)   # identical to the training treatment
    n_nan = int(X.isna().sum().sum())
    if n_inf:
        res.warnings.append(f"{n_inf} infinite value(s) were treated as missing (as in training).")
    if n_nan:
        res.warnings.append(f"{n_nan} missing value(s) will be filled with the training-set "
                            "median of that feature (as in training).")
    res.X = X[required].astype(np.float32)
    res.ok, res.n_rows = True, len(X)
    return res


def read_uploaded_csv(file_obj, max_mb: int = config.MAX_UPLOAD_MB) -> tuple[pd.DataFrame | None, str | None]:
    """Parse an uploaded file as CSV only. Returns (frame, error_message)."""
    try:
        size = getattr(file_obj, "size", None)
        if size is not None and size > max_mb * 2**20:
            return None, f"File is larger than {max_mb} MB."
        df = pd.read_csv(file_obj, encoding="latin-1", low_memory=False)
        return df, None
    except pd.errors.EmptyDataError:
        return None, "The file is empty."
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        msg = re.sub(r"\s+", " ", str(exc))[:200]
        return None, f"Could not parse the file as CSV ({msg})."


# ----------------------------------------------------------------------------
# Input realism (out-of-distribution) check
# ----------------------------------------------------------------------------
# A classifier can only be trusted on inputs that resemble its training data.
# Calibrated on the 499,621 real held-out test flows: only 2 (0.0004 %) have a
# value outside the range seen in training, and 2.0 % have 5+ features in the
# extreme 1 % tails (4.5 % in the attack-rich demo sample). Hand-made or synthetic rows typically fail the first test (e.g. a
# PSH Flag Count of 7, when CICFlowMeter only ever writes 0 or 1).
EXTREME_TAIL_FEATURES = 5
OUT_OF_RANGE_SHARE_WARN = 0.01
FLAGGED_SHARE_WARN = 0.10


def realism_check(X: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Per-row check of how much each record resembles the training flows.

    Returns columns: n_impossible (values outside the training min–max, or a
    non-0/1 value in a 0/1 flag), n_extreme (values outside the training
    1st–99th percentile), worst_feature, and input_check:
    'out of range' > 'unusual' > 'ok'. Missing values are ignored.
    """
    stats = schema["training_feature_stats"]
    cols = list(X.columns)
    V = X.to_numpy(dtype=np.float64)
    mn = np.array([stats[c]["min"] for c in cols])
    mx = np.array([stats[c]["max"] for c in cols])
    lo = np.array([stats[c]["p01"] for c in cols])
    hi = np.array([stats[c]["p99"] for c in cols])
    with np.errstate(invalid="ignore"):
        impossible = (V < mn) | (V > mx)
        binary = np.array([stats[c]["min"] == 0 and stats[c]["max"] == 1 for c in cols])
        impossible[:, binary] |= ~np.isin(V[:, binary], [0.0, 1.0]) & ~np.isnan(V[:, binary])
        extreme = (V < lo) | (V > hi)
    n_imp, n_ext = impossible.sum(1), extreme.sum(1)
    names = np.array(cols, dtype=object)
    worst = np.where(impossible.any(1), names[impossible.argmax(1)],
                     np.where(extreme.any(1), names[extreme.argmax(1)], ""))
    status = np.where(n_imp > 0, "out of range",
                      np.where(n_ext >= EXTREME_TAIL_FEATURES, "unusual", "ok"))
    return pd.DataFrame({"input_check": status, "n_impossible": n_imp, "n_extreme": n_ext,
                         "worst_feature": worst}, index=X.index)


def realism_summary(chk: pd.DataFrame) -> dict:
    """File-level summary used by the website to decide whether to warn."""
    n = len(chk)
    out_of_range = int((chk["input_check"] == "out of range").sum())
    unusual = int((chk["input_check"] == "unusual").sum())
    bad = chk.loc[chk["input_check"] != "ok", "worst_feature"]
    share = (out_of_range + unusual) / n if n else 0.0
    return {"n": n, "out_of_range": out_of_range, "unusual": unusual, "share_flagged": share,
            "top_features": bad[bad != ""].value_counts().head(5).to_dict(),
            "warn": n > 0 and (out_of_range / n > OUT_OF_RANGE_SHARE_WARN or share > FLAGGED_SHARE_WARN)}


# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
def predict(pipeline, X: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    proba = pipeline.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return pd.DataFrame({
        "prediction": np.where(pred == 1, config.POSITIVE_LABEL, config.NEGATIVE_LABEL),
        "p_attack": np.round(proba, 6),
    }, index=X.index)


def predict_family(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    proba = pipeline.predict_proba(X)
    classes = pipeline.classes_
    top = proba.argmax(1)
    return pd.DataFrame({"predicted_family": classes[top],
                         "family_confidence": np.round(proba[np.arange(len(X)), top], 6)},
                        index=X.index)


def predict_records(df: pd.DataFrame) -> tuple[ValidationResult, pd.DataFrame | None]:
    """Convenience: validate a raw frame and predict with the saved artifacts."""
    schema = load_schema()
    v = validate_frame(df, schema)
    if not v.ok:
        return v, None
    meta = load_json(config.METADATA_PATH)
    out = predict(load_pipeline(), v.X, threshold=meta.get("decision_threshold", 0.5))
    return v, out
