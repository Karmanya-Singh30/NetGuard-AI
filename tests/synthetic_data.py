"""SYNTHETIC test fixture — NOT real network traffic, NOT for reporting.

Generates small CSV files that mimic the *format* of the CICIDS2017
MachineLearningCSV release (header spellings with leading spaces, the
duplicated 'Fwd Header Length' column, 'Infinity'/NaN values, a cp1252
en-dash in web-attack labels, duplicate rows) so the pipeline, website and
tests can be exercised without the 2.8M-row dataset.

Values are random draws with class-dependent shifts; any metric obtained on
this data is meaningless and must never appear in the report.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Header exactly as spelled in the MachineLearningCSV files (incl. spaces).
RAW_HEADER = [
    " Destination Port", " Flow Duration", " Total Fwd Packets", " Total Backward Packets",
    "Total Length of Fwd Packets", " Total Length of Bwd Packets", " Fwd Packet Length Max",
    " Fwd Packet Length Min", " Fwd Packet Length Mean", " Fwd Packet Length Std",
    "Bwd Packet Length Max", " Bwd Packet Length Min", " Bwd Packet Length Mean",
    " Bwd Packet Length Std", "Flow Bytes/s", " Flow Packets/s", " Flow IAT Mean",
    " Flow IAT Std", " Flow IAT Max", " Flow IAT Min", "Fwd IAT Total", " Fwd IAT Mean",
    " Fwd IAT Std", " Fwd IAT Max", " Fwd IAT Min", "Bwd IAT Total", " Bwd IAT Mean",
    " Bwd IAT Std", " Bwd IAT Max", " Bwd IAT Min", "Fwd PSH Flags", " Bwd PSH Flags",
    " Fwd URG Flags", " Bwd URG Flags", " Fwd Header Length", " Bwd Header Length",
    "Fwd Packets/s", " Bwd Packets/s", " Min Packet Length", " Max Packet Length",
    " Packet Length Mean", " Packet Length Std", " Packet Length Variance", "FIN Flag Count",
    " SYN Flag Count", " RST Flag Count", " PSH Flag Count", " ACK Flag Count",
    " URG Flag Count", " CWE Flag Count", " ECE Flag Count", " Down/Up Ratio",
    " Average Packet Size", " Avg Fwd Segment Size", " Avg Bwd Segment Size",
    " Fwd Header Length", "Fwd Avg Bytes/Bulk", " Fwd Avg Packets/Bulk",
    " Fwd Avg Bulk Rate", " Bwd Avg Bytes/Bulk", " Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate", "Subflow Fwd Packets", " Subflow Fwd Bytes",
    " Subflow Bwd Packets", " Subflow Bwd Bytes", "Init_Win_bytes_forward",
    " Init_Win_bytes_backward", " act_data_pkt_fwd", " min_seg_size_forward",
    "Active Mean", " Active Std", " Active Max", " Active Min", "Idle Mean", " Idle Std",
    " Idle Max", " Idle Min", " Label",
]

DAY_LABELS = {
    "Monday-WorkingHours.pcap_ISCX.csv": {"BENIGN": 1.0},
    "Tuesday-WorkingHours.pcap_ISCX.csv": {"BENIGN": .8, "FTP-Patator": .1, "SSH-Patator": .1},
    "Wednesday-workingHours.pcap_ISCX.csv": {"BENIGN": .6, "DoS Hulk": .25, "DoS GoldenEye": .05,
                                             "DoS slowloris": .05, "DoS Slowhttptest": .045,
                                             "Heartbleed": .005},
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv": {
        "BENIGN": .9, "Web Attack \u2013 Brute Force": .05, "Web Attack \u2013 XSS": .03,
        "Web Attack \u2013 Sql Injection": .02},
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv": {"BENIGN": .97, "Infiltration": .03},
    "Friday-WorkingHours-Morning.pcap_ISCX.csv": {"BENIGN": .9, "Bot": .1},
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv": {"BENIGN": .5, "PortScan": .5},
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv": {"BENIGN": .45, "DDoS": .55},
}


def _rows(label: str, n: int, rng: np.random.Generator) -> pd.DataFrame:
    k = len(RAW_HEADER) - 1
    shift = 0.0 if label == "BENIGN" else 1.0 + (hash(label) % 7) / 3.0
    base = rng.lognormal(mean=2.0 + shift * 0.6, sigma=1.2, size=(n, k))
    df = pd.DataFrame(base, columns=RAW_HEADER[:-1])
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    df[" Fwd Header Length"] = np.round(df[" Fwd Header Length"])
    df[" Destination Port"] = rng.choice([80, 443, 22, 21, 8080, 53], size=n)
    for c in ["Fwd PSH Flags", " Bwd PSH Flags", " Fwd URG Flags", " Bwd URG Flags",
              "Fwd Avg Bytes/Bulk", " Fwd Avg Packets/Bulk", " Fwd Avg Bulk Rate",
              " Bwd Avg Bytes/Bulk", " Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
              " CWE Flag Count"]:
        df[c] = 0  # constant columns, as in the real files
    df["Init_Win_bytes_forward"] = np.where(rng.random(n) < 0.2, -1, df["Init_Win_bytes_forward"])
    out = df.reindex(columns=list(dict.fromkeys(RAW_HEADER[:-1])))
    out.insert(55, " Fwd Header Length.dup", out[" Fwd Header Length"])  # duplicated header copy
    out[" Label"] = label
    return out


def write_synthetic_dataset(folder: Path, rows_per_file: int = 1500, seed: int = 0) -> list[Path]:
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for fname, mix in DAY_LABELS.items():
        parts = []
        for lab, share in mix.items():
            n = max(12, int(rows_per_file * share))
            parts.append(_rows(lab, n, rng))
        df = pd.concat(parts, ignore_index=True)
        # Infinity / NaN in the rate columns, as in the real data
        idx = rng.choice(len(df), size=max(2, len(df) // 200), replace=False)
        df.loc[idx[: len(idx) // 2], "Flow Bytes/s"] = np.inf
        df.loc[idx[len(idx) // 2:], "Flow Bytes/s"] = np.nan
        df.loc[idx, " Flow Packets/s"] = np.inf
        # exact duplicate rows
        df = pd.concat([df, df.sample(n=max(5, len(df) // 20), random_state=seed)], ignore_index=True)
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        header = [c if c != " Fwd Header Length.dup" else " Fwd Header Length" for c in df.columns]
        p = folder / fname
        df.to_csv(p, index=False, header=header, encoding="cp1252", na_rep="NaN")
        txt = p.read_text(encoding="cp1252").replace("inf", "Infinity")
        p.write_text(txt, encoding="cp1252")
        paths.append(p)
    return paths


if __name__ == "__main__":
    import sys
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/synthetic_fixture")
    print(write_synthetic_dataset(out))
