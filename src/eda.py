"""Dataset profile and exploratory figures, computed from the actual data.

The profile is computed on the *raw* (tidied, not yet cleaned) data so it
documents the problems the cleaning step must solve. Figures that describe
classes use the cleaned, de-duplicated data. EDA is descriptive only: no
modelling decision is fitted on it (all learned steps use the training split).
"""
from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config
from .data_loading import LoadReport
from .evaluate import PALETTE, _finish

log = logging.getLogger(__name__)

KEY_FEATURES = ["Flow Duration", "Total Fwd Packets", "Total Backward Packets",
                "Flow Bytes/s", "Flow Packets/s", "Packet Length Mean",
                "Flow IAT Mean", "Init_Win_bytes_forward", "Destination Port"]


def profile_dataset(df: pd.DataFrame, load: LoadReport) -> dict:
    numeric = [c for c in df.columns if c not in config.NON_FEATURE_COLS
               and pd.api.types.is_numeric_dtype(df[c])]
    non_numeric = [c for c in df.columns if c not in numeric]
    arr = df[numeric]
    n = len(df)

    missing = arr.isna().sum()
    inf = pd.Series({c: int(np.isinf(arr[c].to_numpy()).sum()) for c in numeric})
    nunique = arr.nunique(dropna=False)
    constant = nunique[nunique <= 1].index.tolist()
    top_freq = {}
    for c in numeric:
        if c in constant:
            continue
        vc = arr[c].value_counts(dropna=False, normalize=True)
        top_freq[c] = float(vc.iloc[0])
    near_constant = sorted([c for c, f in top_freq.items() if f >= 0.999])

    h = pd.util.hash_pandas_object(df[numeric + [config.LABEL_COL]], index=False)
    dup_rows = int(h.duplicated().sum())
    hf = pd.util.hash_pandas_object(df[numeric], index=False)
    feat_dup = int(hf.duplicated().sum())

    labels = df[config.LABEL_COL].value_counts()
    per_file = (df.groupby([config.SOURCE_COL, config.LABEL_COL], observed=True).size()
                .unstack(fill_value=0))
    neg = df[numeric].lt(0).sum()

    return {
        "files": load.files,
        "rows_per_file": load.rows_per_file,
        "n_rows": int(n),
        "n_columns_after_tidy": int(df.shape[1] - 1),  # minus our source_file column
        "n_numeric_features": len(numeric),
        "non_numeric_columns": [c for c in non_numeric if c != config.SOURCE_COL],
        "duplicate_columns_dropped": load.duplicate_columns_dropped,
        "values_coerced_to_nan_non_numeric": load.coerced_non_numeric,
        "label_spelling_variants": load.raw_label_variants,
        "memory_mb_float32": round(float(df.memory_usage(deep=True).sum()) / 2**20, 1),
        "class_distribution": {k: int(v) for k, v in labels.items()},
        "class_distribution_pct": {k: round(100 * v / n, 4) for k, v in labels.items()},
        "binary_distribution": {
            config.NEGATIVE_LABEL: int((df[config.LABEL_COL] == config.NEGATIVE_LABEL).sum()),
            config.POSITIVE_LABEL: int((df[config.LABEL_COL] != config.NEGATIVE_LABEL).sum())},
        "imbalance_ratio_largest_to_smallest": float(labels.iloc[0] / labels.iloc[-1]),
        "labels_per_file": {f: {k: int(v) for k, v in row.items() if v}
                            for f, row in per_file.iterrows()},
        "missing_values": {c: int(v) for c, v in missing.items() if v},
        "missing_pct": {c: round(100 * v / n, 4) for c, v in missing.items() if v},
        "infinite_values": {c: int(v) for c, v in inf.items() if v},
        "duplicate_rows_including_label": dup_rows,
        "duplicate_rows_features_only": feat_dup,
        "constant_columns": constant,
        "near_constant_columns_top_value_ge_99.9pct": near_constant,
        "columns_with_negative_values": {c: int(v) for c, v in neg.items() if v},
        "identifier_review": {
            "Destination Port": "Present. Not an identifier of a flow, but in this lab dataset "
                                "attacks target fixed ports; excluded from model inputs by default "
                                "to avoid a testbed-specific shortcut (config.EXCLUDED_FEATURES).",
            "IPs/Flow ID/Timestamp": "Not present in the MachineLearningCSV release" if not any(
                k in df.columns for k in ("Source IP", "Flow ID", "Timestamp")) else "PRESENT - review!",
        },
        "describe": arr.describe().T[["mean", "std", "min", "50%", "max"]]
            .replace([np.inf, -np.inf], np.nan).round(4).to_dict(orient="index"),
    }


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def fig_class_distribution(df: pd.DataFrame, path, title_suffix=""):
    vc = df[config.LABEL_COL].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(8, 0.38 * len(vc) + 1.5))
    colors = [PALETTE["warn"] if i == config.NEGATIVE_LABEL else PALETTE["primary"] for i in vc.index]
    ax.barh(vc.index, vc.values, color=colors)
    ax.set_xscale("log")
    for i, v in enumerate(vc.values):
        ax.text(v * 1.08, i, f"{v:,}", va="center", fontsize=8)
    ax.set_xlabel("Number of flows (log scale)")
    ax.set_title(f"Class distribution{title_suffix}")
    return _finish(fig, path)


def fig_binary_by_day(df: pd.DataFrame, path):
    t = (df.assign(cls=np.where(df[config.BINARY_TARGET] == 1, "ATTACK", "BENIGN"))
         .groupby([config.SOURCE_COL, "cls"], observed=True).size().unstack(fill_value=0))
    order = [d for d in config.DAY_ORDER if d in t.index] + [d for d in t.index if d not in config.DAY_ORDER]
    t = t.loc[order]
    t.index = [i.replace(".pcap_ISCX.csv", "").replace("WorkingHours", "WH") for i in t.index]
    fig, ax = plt.subplots(figsize=(8, 4))
    t.plot(kind="bar", stacked=True, ax=ax, color={"BENIGN": PALETTE["warn"], "ATTACK": PALETTE["primary"]})
    ax.set_ylabel("Flows (after cleaning)")
    ax.set_xlabel("Capture file (day)")
    ax.legend(title="")
    ax.set_title("Benign vs attack flows per capture day")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return _finish(fig, path)


def fig_data_quality(profile: dict, dedup: dict, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    inf = profile["infinite_values"]
    miss = profile["missing_values"]
    cols = sorted(set(inf) | set(miss))
    if cols:
        x = np.arange(len(cols))
        axes[0].bar(x - 0.2, [miss.get(c, 0) for c in cols], 0.4, label="Missing (NaN)")
        axes[0].bar(x + 0.2, [inf.get(c, 0) for c in cols], 0.4, label="Infinite", color=PALETTE["bad"])
        axes[0].set_xticks(x, cols, rotation=30, ha="right")
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, "No missing or infinite values", ha="center")
    axes[0].set_ylabel("Rows")
    axes[0].set_title("Non-finite values by column (raw data)")

    rem = pd.Series(dedup["exact_duplicates_removed_by_label"]).reindex(
        dedup["rows_before_by_label"].keys()).fillna(0)
    before = pd.Series(dedup["rows_before_by_label"])
    share = (rem / before).sort_values()
    axes[1].barh(share.index, share.values, color=PALETTE["primary"])
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Share of rows that were exact duplicates")
    axes[1].set_title("Exact duplicates by class (removed before split)")
    return _finish(fig, path)


def fig_feature_distributions(df: pd.DataFrame, path, max_rows=200_000):
    s = df.sample(n=min(max_rows, len(df)), random_state=config.RANDOM_STATE)
    feats = [f for f in KEY_FEATURES if f in s.columns][:6]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, f in zip(axes.ravel(), feats):
        for cls, color in ((0, PALETTE["warn"]), (1, PALETTE["primary"])):
            v = s.loc[s[config.BINARY_TARGET] == cls, f].dropna()
            v = np.sign(v) * np.log10(1 + np.abs(v))
            ax.hist(v, bins=60, alpha=0.55, density=True, color=color,
                    label="BENIGN" if cls == 0 else "ATTACK")
        ax.set_title(f)
        ax.set_xlabel("sign(x)·log10(1+|x|)")
        ax.set_ylabel("Density")
    for ax in axes.ravel()[len(feats):]:
        ax.remove()
    axes.ravel()[0].legend()
    fig.suptitle(f"Feature distributions by class (random sample of {len(s):,} flows)")
    return _finish(fig, path)


def fig_correlation(df: pd.DataFrame, features: list[str], path, max_rows=200_000):
    s = df[features].sample(n=min(max_rows, len(df)), random_state=config.RANDOM_STATE)
    s = s.loc[:, s.std() > 0]
    corr = s.corr()
    fig, ax = plt.subplots(figsize=(10, 8.5))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, cmap="vlag", vmin=-1, vmax=1, center=0, square=True,
                linewidths=0.3, cbar_kws={"label": "Pearson r", "shrink": 0.7}, ax=ax)
    ax.set_title(f"Correlation of {corr.shape[0]} selected flow features")
    plt.setp(ax.get_xticklabels(), fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)
    return _finish(fig, path)


def fig_family_boxplots(df: pd.DataFrame, path, max_per_family=20_000):
    feats = [f for f in ["Flow Duration", "Packet Length Mean", "Flow IAT Mean", "Total Fwd Packets"]
             if f in df.columns]
    s = (df.sample(frac=1.0, random_state=config.RANDOM_STATE)
           .groupby(config.FAMILY_TARGET, observed=True).head(max_per_family))
    order = s[config.FAMILY_TARGET].value_counts().index.tolist()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, f in zip(axes.ravel(), feats):
        v = s[[config.FAMILY_TARGET, f]].dropna().copy()
        v[f] = np.log10(1 + v[f].clip(lower=0))
        sns.boxplot(data=v, x=config.FAMILY_TARGET, y=f, order=order, ax=ax, fliersize=0.5,
                    color=PALETTE["accent"])
        ax.set_ylabel(f"log10(1 + {f})")
        ax.set_xlabel("")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
    fig.suptitle(f"Class-wise feature comparison (≤{max_per_family:,} flows per family)")
    return _finish(fig, path)


def fig_scatter(df: pd.DataFrame, path, max_rows=60_000):
    x, y = "Flow Duration", "Total Fwd Packets"
    if x not in df.columns or y not in df.columns:
        return None
    s = df.sample(n=min(max_rows, len(df)), random_state=config.RANDOM_STATE)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for fam in s[config.FAMILY_TARGET].value_counts().index:
        g = s[s[config.FAMILY_TARGET] == fam]
        ax.scatter(np.log10(1 + g[x].clip(lower=0)), np.log10(1 + g[y].clip(lower=0)),
                   s=3, alpha=0.4, label=f"{fam} (n={len(g):,})")
    ax.set_xlabel(f"log10(1 + {x} [µs])")
    ax.set_ylabel(f"log10(1 + {y})")
    ax.set_title("Flow duration vs forward packets by family (sample)")
    ax.legend(markerscale=4, fontsize=7, ncol=2)
    return _finish(fig, path)


def make_eda_figures(raw_profile: dict, clean_df: pd.DataFrame, dedup: dict,
                     corr_features: list[str]) -> list[str]:
    F = config.FIGURES_DIR
    out = [
        fig_class_distribution(clean_df, F / "eda_class_distribution.png", " (after cleaning)"),
        fig_binary_by_day(clean_df, F / "eda_binary_by_day.png"),
        fig_data_quality(raw_profile, dedup, F / "eda_data_quality.png"),
        fig_feature_distributions(clean_df, F / "eda_feature_distributions.png"),
        fig_correlation(clean_df, corr_features, F / "eda_correlation_heatmap.png"),
        fig_family_boxplots(clean_df, F / "eda_family_boxplots.png"),
        fig_scatter(clean_df, F / "eda_duration_vs_packets.png"),
    ]
    return [str(p.relative_to(config.ROOT)) for p in out if p is not None]
