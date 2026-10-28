import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from synthetic_data import write_synthetic_dataset  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory) -> Path:
    """Small SYNTHETIC dataset in CICIDS2017 format (not real traffic)."""
    d = tmp_path_factory.mktemp("synthetic")
    write_synthetic_dataset(d, rows_per_file=400, seed=1)
    return d


@pytest.fixture(scope="session")
def loaded(synthetic_dir):
    from src.data_loading import load_cicids
    return load_cicids(synthetic_dir)


@pytest.fixture(scope="session")
def cleaned(loaded):
    from src.data_preprocessing import clean_dataset
    df, report = clean_dataset(loaded[0].copy())
    return df, report


@pytest.fixture(scope="session")
def trained(cleaned):
    """A small fitted pipeline built exactly like the training script builds them."""
    from sklearn.ensemble import RandomForestClassifier
    from src import config
    from src.data_preprocessing import feature_columns, split_train_test
    from src.train import make_pipeline
    df, _ = cleaned
    train, test = split_train_test(df)
    feats = feature_columns(df)
    pipe = make_pipeline(feats, RandomForestClassifier(n_estimators=20, random_state=0), scale=False)
    pipe.fit(train[feats], train[config.BINARY_TARGET])
    schema = {"features": feats}
    return pipe, train, test, feats, schema
