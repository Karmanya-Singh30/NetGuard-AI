"""Central configuration: paths, seeds, label definitions.

Every other module imports from here so names, paths and label mappings
stay identical across training, evaluation, the website and the tests.
"""
from __future__ import annotations

import os
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
# NETGUARD_ROOT lets the tests run the whole project inside a temporary folder.
ROOT = Path(os.environ.get("NETGUARD_ROOT", Path(__file__).resolve().parents[1])).resolve()
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"

FINAL_PIPELINE_PATH = MODELS_DIR / "final_pipeline.joblib"
MULTICLASS_PIPELINE_PATH = MODELS_DIR / "multiclass_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
SCHEMA_PATH = MODELS_DIR / "feature_schema.json"
PROFILE_PATH = RESULTS_DIR / "dataset_profile.json"
DEMO_SAMPLE_PATH = SAMPLES_DIR / "demo_traffic_sample.csv"

# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 3

# ----------------------------------------------------------------------------
# Columns
# ----------------------------------------------------------------------------
LABEL_COL = "Label"           # original multiclass label (after normalisation)
BINARY_TARGET = "is_attack"   # 1 = ATTACK (positive class), 0 = BENIGN
FAMILY_TARGET = "attack_family"
SOURCE_COL = "source_file"    # which day-file a row came from (never a feature)

POSITIVE_LABEL = "ATTACK"
NEGATIVE_LABEL = "BENIGN"
BINARY_CLASS_NAMES = {0: NEGATIVE_LABEL, 1: POSITIVE_LABEL}

# Columns that are never used as model inputs.
# - Label/targets: they are what we predict.
# - source_file: bookkeeping; using it would leak the day (= attack type).
# - Destination Port: excluded by default. In CICIDS2017 specific attacks were
#   launched against specific ports of a lab testbed, so the port acts as a
#   shortcut that is unlikely to generalise. See docs/METHODOLOGY.md.
NON_FEATURE_COLS = {LABEL_COL, BINARY_TARGET, FAMILY_TARGET, SOURCE_COL}
EXCLUDED_FEATURES = {"Destination Port"}

# ----------------------------------------------------------------------------
# Label normalisation and attack families
# ----------------------------------------------------------------------------
# The raw CSVs contain an en-dash in "Web Attack – Brute Force" that is often
# mis-decoded (e.g. "Web Attack � Brute Force"). data_loading.normalise_label
# maps every variant to the canonical names below.
FAMILY_MAP = {
    "BENIGN": "BENIGN",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "DDoS": "DDoS",
    "PortScan": "PortScan",
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",
    "Web Attack - Brute Force": "WebAttack",
    "Web Attack - XSS": "WebAttack",
    "Web Attack - Sql Injection": "WebAttack",
    "Bot": "Bot",
    "Infiltration": "Infiltration",
    "Heartbleed": "Heartbleed",
}

# Classes with fewer training rows than this are kept in the data and in the
# binary task, but flagged in reports as too rare for reliable multiclass
# learning. Nothing is silently dropped.
RARE_CLASS_THRESHOLD = 100

# Day files, in chronological order (used for the held-out-day experiment).
DAY_ORDER = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]
# Held-out-day experiment: train on Monday–Thursday, test on Friday.
HOLDOUT_TEST_PREFIX = "Friday"

# ----------------------------------------------------------------------------
# Website limits
# ----------------------------------------------------------------------------
MAX_UPLOAD_ROWS = 200_000
MAX_UPLOAD_MB = 200


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, SAMPLES_DIR, MODELS_DIR, FIGURES_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
