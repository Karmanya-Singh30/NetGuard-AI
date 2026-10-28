"""Evaluation metrics and plots.

Positive class = ATTACK (label 1). Therefore:
  TP = attack flow predicted ATTACK      FN = attack flow predicted BENIGN (missed attack)
  TN = benign flow predicted BENIGN      FP = benign flow predicted ATTACK (false alarm)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)

from . import config

PALETTE = {"primary": "#0e7490", "accent": "#22d3ee", "warn": "#f59e0b", "bad": "#dc2626",
           "grid": "#cbd5e1"}


def _safe_div(a: float, b: float) -> float | None:
    return float(a) / float(b) if b else None   # None = undefined, never a fake 0


def binary_metrics(y_true, y_pred, y_score=None) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    m = {
        "n": int(len(y_true)), "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)) if (tp + fp) else None,
        "recall": _safe_div(tp, tp + fn),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)) if (tp + fp + fn) else None,
        "specificity": _safe_div(tn, tn + fp),
        "fpr": _safe_div(fp, fp + tn),
        "fnr": _safe_div(fn, fn + tp),
        "roc_auc": None, "pr_auc": None,
    }
    both_classes = len(np.unique(y_true)) == 2
    if y_score is not None and both_classes:
        m["roc_auc"] = float(roc_auc_score(y_true, y_score))
        m["pr_auc"] = float(average_precision_score(y_true, y_score))
    return m


def detection_by_family(families: pd.Series, y_true, y_pred) -> pd.DataFrame:
    """Per attack family: share of flows flagged ATTACK.
    For attack families this is the detection rate (recall); for BENIGN it is the FPR."""
    df = pd.DataFrame({"family": np.asarray(families), "y": np.asarray(y_true),
                       "pred": np.asarray(y_pred)})
    out = df.groupby("family").agg(n=("y", "size"), flagged_attack=("pred", "sum"))
    out["flagged_rate"] = out["flagged_attack"] / out["n"]
    out["missed"] = np.where(out.index == config.NEGATIVE_LABEL, 0, out["n"] - out["flagged_attack"])
    return out.sort_values("n", ascending=False).reset_index()


def multiclass_report(y_true, y_pred, labels: list[str]) -> dict:
    rep = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return {k: ({kk: float(vv) for kk, vv in v.items()} if isinstance(v, dict) else float(v))
            for k, v in rep.items()}


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def _finish(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confusion(y_true, y_pred, labels, display, title, path: Path, normalize=False):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    data = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1) if normalize else cm
    size = max(4.5, 0.7 * len(labels) + 2)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    if normalize:
        # Percentages with enough precision to show small error rates (0.07% is not "0.00"),
        # plus the raw count underneath.
        annot = np.array([[f"{100 * data[i, j]:.2f}%\n({cm[i, j]:,})" for j in range(len(labels))]
                          for i in range(len(labels))])
        sns.heatmap(data, annot=annot, fmt="", cmap="Blues", cbar=False, vmin=0, vmax=1,
                    xticklabels=display, yticklabels=display, ax=ax,
                    annot_kws={"size": 8 if len(labels) > 4 else 11})
    else:
        sns.heatmap(data, annot=True, fmt=",d", cmap="Blues", cbar=False,
                    xticklabels=display, yticklabels=display, ax=ax, annot_kws={"size": 9})
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    return _finish(fig, path)


def plot_roc_pr(curves: dict[str, tuple], path_roc: Path, path_pr: Path):
    """curves: model_name -> (y_true, y_score).
    ROC uses a log false-positive axis: at the error rates of good IDS models
    (<1 % FPR) a linear axis hides every difference in the top-left corner."""
    fig, ax = plt.subplots(figsize=(6.4, 5))
    for name, (yt, ys) in curves.items():
        fpr, tpr, _ = roc_curve(yt, ys)
        ax.plot(np.clip(fpr, 1e-6, 1), tpr, label=f"{name} (AUC={roc_auc_score(yt, ys):.4f})")
    ax.set_xscale("log")
    ax.set_xlim(1e-5, 1)
    ax.set_xlabel("False positive rate, log scale (benign flagged as attack)")
    ax.set_ylabel("True positive rate (attacks detected)")
    ax.set_title("ROC curves — test set (positive class = ATTACK)")
    ax.grid(True, which="major", color=PALETTE["grid"], lw=0.5)
    ax.legend(fontsize=8, loc="lower right")
    _finish(fig, path_roc)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    base = float(np.mean(list(curves.values())[0][0]))
    for ax, zoom in zip(axes, (False, True)):
        for name, (yt, ys) in curves.items():
            p, r, _ = precision_recall_curve(yt, ys)
            ax.plot(r, p, label=f"{name} (AP={average_precision_score(yt, ys):.4f})")
        if zoom:
            ax.set_xlim(0.95, 1.0005)
            ax.set_ylim(0.95, 1.0005)
            ax.set_title("Zoom: recall and precision ≥ 0.95")
        else:
            ax.axhline(base, ls="--", color="grey", lw=1, label=f"Chance (attack share={base:.3f})")
            ax.set_title("Precision–recall curves — test set")
            ax.legend(fontsize=8, loc="lower left")
        ax.set_xlabel("Recall (attacks detected)")
        ax.set_ylabel("Precision (alerts that are real attacks)")
        ax.grid(True, color=PALETTE["grid"], lw=0.5)
    _finish(fig, path_pr)


def plot_model_comparison(df: pd.DataFrame, path: Path):
    """Left: precision/recall/F1. Right: the error rates (FPR, FNR) on a log
    scale, where the differences between strong models are actually visible."""
    d = df[df["model"] != "Dummy (majority)"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.2, 1]})
    long = d.melt(id_vars="model", value_vars=["precision", "recall", "f1"],
                  var_name="metric", value_name="value")
    sns.barplot(data=long, x="metric", y="value", hue="model", ax=axes[0])
    lo = max(0.0, float(np.nanmin(long["value"])) - 0.05)
    axes[0].set_ylim(lo, 1.005)
    axes[0].set_ylabel(f"Score (axis starts at {lo:.2f})")
    axes[0].set_xlabel("")
    axes[0].set_title("Detection quality (higher is better)")
    axes[0].legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    err = d.melt(id_vars="model", value_vars=["fpr", "fnr"], var_name="metric", value_name="value")
    err["metric"] = err["metric"].map({"fpr": "False-alarm rate (FPR)", "fnr": "Missed-attack rate (FNR)"})
    sns.barplot(data=err, x="metric", y="value", hue="model", ax=axes[1], legend=False)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Rate, log scale (lower is better)")
    axes[1].set_xlabel("")
    axes[1].set_title("Error rates")
    for c in axes[1].containers:
        axes[1].bar_label(c, fmt="%.4f", fontsize=7, padding=2)
    fig.suptitle("Model comparison on the held-out test set (positive class = ATTACK; "
                 "Dummy baseline omitted: recall 0)", fontsize=11)
    return _finish(fig, path)


def plot_threshold_sweep(y_true, y_score, path: Path):
    ts = np.linspace(0.05, 0.95, 19)
    rows = []
    for t in ts:
        m = binary_metrics(y_true, (y_score >= t).astype(int))
        rows.append((t, m["precision"] or 0, m["recall"] or 0, m["fpr"] or 0))
    a = np.array(rows)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(a[:, 0], a[:, 1], marker="o", ms=3, label="Precision")
    ax.plot(a[:, 0], a[:, 2], marker="o", ms=3, label="Recall")
    ax2 = ax.twinx()
    ax2.plot(a[:, 0], a[:, 3], color=PALETTE["bad"], ls="--", label="FPR (right axis)")
    ax2.set_ylabel("False positive rate")
    ax.axvline(0.5, color="grey", lw=1, ls=":")
    ax.set_xlabel("Decision threshold on P(ATTACK)")
    ax.set_ylabel("Precision / recall")
    ax.set_title("Effect of the decision threshold (test set, analysis only)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower left")
    return _finish(fig, path), [dict(threshold=float(t), precision=float(p), recall=float(r), fpr=float(f))
                                for t, p, r, f in rows]


def plot_family_detection(fam: pd.DataFrame, title: str, path: Path):
    d = fam.copy()
    d["kind"] = np.where(d["family"] == config.NEGATIVE_LABEL, "BENIGN (false-alarm rate)",
                         "Attack (detection rate)")
    d = d.sort_values("flagged_rate")
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(d) + 1.5))
    colors = [PALETTE["warn"] if k.startswith("BENIGN") else PALETTE["primary"] for k in d["kind"]]
    ax.barh(d["family"] + "  (n=" + d["n"].astype(str) + ")", d["flagged_rate"], color=colors)
    for i, v in enumerate(d["flagged_rate"]):
        ax.text(min(v + 0.01, 0.93), i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Share of flows flagged as ATTACK")
    ax.set_title(title)
    return _finish(fig, path)


def _sanitize(o):
    """Recursively turn NaN/inf into None so the JSON is strictly valid."""
    if isinstance(o, dict):
        return {str(k): _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(type(o))
    path.write_text(json.dumps(_sanitize(obj), indent=2, default=conv, allow_nan=False),
                    encoding="utf-8")
    return path
