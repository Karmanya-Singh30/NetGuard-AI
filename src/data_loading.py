"""Load the CICIDS2017 MachineLearningCSV files.

Accepted inputs (searched in this order when no path is given):
  1. data/raw/*.csv
  2. data/*.csv                       (the project's original layout)
  3. data/raw/MachineLearningCSV.zip  (read directly, no unzipping needed)
A path to a folder, a single CSV, or the zip can also be passed explicitly.
"""
from __future__ import annotations

import logging
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)


@dataclass
class LoadReport:
    """Facts recorded while loading, reused by the dataset profile."""
    files: list[str] = field(default_factory=list)
    rows_per_file: dict[str, int] = field(default_factory=dict)
    duplicate_columns_dropped: list[str] = field(default_factory=list)
    coerced_non_numeric: dict[str, int] = field(default_factory=dict)
    raw_label_variants: dict[str, str] = field(default_factory=dict)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def clean_column_name(name: str) -> str:
    """CICIDS2017 headers have stray leading spaces (' Flow Duration')."""
    return re.sub(r"\s+", " ", str(name)).strip()


_WEB_ATTACK_RE = re.compile(r"^Web Attack[^A-Za-z]+", re.IGNORECASE)


def normalise_label(raw: str) -> str:
    """Map encoding variants of a label to one canonical spelling.

    'Web Attack – Brute Force', 'Web Attack � Brute Force' and
    'Web Attack \x96 Brute Force' all become 'Web Attack - Brute Force'.
    """
    s = str(raw).strip()
    if _WEB_ATTACK_RE.match(s):
        s = _WEB_ATTACK_RE.sub("Web Attack - ", s)
    if s.upper() == "BENIGN":
        s = "BENIGN"
    return s


def family_of(label: str) -> str:
    return config.FAMILY_MAP.get(label, "Unknown")


def find_sources(path: str | Path | None = None) -> list[tuple[str, object]]:
    """Return a list of (display_name, opener) pairs for every day-CSV.

    `opener` is either a Path or a (ZipFile, member_name) tuple.
    """
    candidates: list[Path] = []
    if path is not None:
        candidates = [Path(path)]
    else:
        candidates = [config.RAW_DIR, config.DATA_DIR,
                      config.RAW_DIR / "MachineLearningCSV.zip",
                      config.DATA_DIR / "MachineLearningCSV.zip"]

    for c in candidates:
        if c.is_file() and c.suffix.lower() == ".zip":
            zf = zipfile.ZipFile(c)
            members = sorted(m for m in zf.namelist()
                             if m.lower().endswith(".csv") and not m.startswith("__MACOSX"))
            if members:
                return [(Path(m).name, (zf, m)) for m in members]
        elif c.is_file() and c.suffix.lower() == ".csv":
            return [(c.name, c)]
        elif c.is_dir():
            csvs = sorted(p for p in c.glob("*.csv"))
            if csvs:
                return [(p.name, p) for p in csvs]
    raise FileNotFoundError(
        "No CICIDS2017 CSV files found. Put the 8 *.pcap_ISCX.csv files (or "
        "MachineLearningCSV.zip) in data/raw/ — see README 'Dataset'.")


def _read_one(opener) -> pd.DataFrame:
    # latin-1 never fails to decode; label variants are fixed afterwards.
    kw = dict(encoding="latin-1", low_memory=False)
    if isinstance(opener, tuple):
        zf, member = opener
        with zf.open(member) as fh:
            return pd.read_csv(fh, **kw)
    return pd.read_csv(opener, **kw)


def tidy_frame(df: pd.DataFrame, report: LoadReport | None = None) -> pd.DataFrame:
    """Clean column names, drop duplicated columns, coerce features to float32."""
    df = df.copy()
    df.columns = [clean_column_name(c) for c in df.columns]

    # Exact duplicate *names* after stripping (rare) -> keep the first.
    dup_names = df.columns[df.columns.duplicated()].tolist()
    if dup_names:
        df = df.loc[:, ~df.columns.duplicated()]
    # pandas renames a repeated header to 'X.1'. CICIDS2017 repeats
    # 'Fwd Header Length'; drop the copy only if it is truly identical.
    dropped = []
    for c in list(df.columns):
        m = re.match(r"^(.*)\.(\d+)$", c)
        if m and m.group(1) in df.columns:
            a, b = df[m.group(1)], df[c]
            same = (pd.to_numeric(a, errors="coerce").fillna(-1).values ==
                    pd.to_numeric(b, errors="coerce").fillna(-1).values).all()
            if same:
                df = df.drop(columns=c)
                dropped.append(c)
    if report is not None:
        for c in dup_names + dropped:
            if c not in report.duplicate_columns_dropped:
                report.duplicate_columns_dropped.append(c)

    if config.LABEL_COL not in df.columns:
        raise ValueError(f"Target column '{config.LABEL_COL}' not found. "
                         f"Columns: {list(df.columns)[:10]}...")

    raw_labels = df[config.LABEL_COL].astype(str)
    norm = raw_labels.map(normalise_label)
    if report is not None:
        for r, n in zip(raw_labels.unique(), pd.Series(raw_labels.unique()).map(normalise_label)):
            report.raw_label_variants[r] = n
    df[config.LABEL_COL] = norm

    feature_cols = [c for c in df.columns if c not in config.NON_FEATURE_COLS]
    for c in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            before = df[c].isna().sum()
            df[c] = pd.to_numeric(df[c], errors="coerce")
            coerced = int(df[c].isna().sum() - before)
            if coerced and report is not None:
                report.coerced_non_numeric[c] = report.coerced_non_numeric.get(c, 0) + coerced
        # float32 halves memory; values are counts/rates, float32 is ample.
        # Values beyond float32 range would become inf -> handled as NaN later.
        with np.errstate(over="ignore"):
            df[c] = df[c].astype(np.float32)
    return df


def load_cicids(path: str | Path | None = None,
                files: Iterable[str] | None = None) -> tuple[pd.DataFrame, LoadReport]:
    """Load and concatenate all day files, adding a `source_file` column."""
    report = LoadReport()
    sources = find_sources(path)
    if files is not None:
        wanted = set(files)
        sources = [s for s in sources if s[0] in wanted]
    frames = []
    for name, opener in sources:
        log.info("Loading %s", name)
        df = tidy_frame(_read_one(opener), report)
        df[config.SOURCE_COL] = name
        report.files.append(name)
        report.rows_per_file[name] = len(df)
        frames.append(df)
    common = set.intersection(*(set(f.columns) for f in frames))
    if any(set(f.columns) != common for f in frames):
        log.warning("Files have differing columns; keeping the %d common ones", len(common))
    cols = [c for c in frames[0].columns if c in common]
    data = pd.concat([f[cols] for f in frames], ignore_index=True)
    data[config.SOURCE_COL] = data[config.SOURCE_COL].astype("category")
    return data, report
