"""Redraw the evaluation figures from the SAVED models, without retraining.

    python scripts/regenerate_figures.py

Re-creates the identical train/test split (same cleaning, same seed), reloads
models/candidates/*.joblib and models/final_pipeline.joblib, and first checks
that the recomputed test metrics equal the saved ones exactly — so a redrawn
figure can never show different numbers from reports/results/.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src import eda, evaluate as ev  # noqa: E402
from src.data_loading import load_cicids  # noqa: E402
from src.data_preprocessing import clean_dataset, split_train_test  # noqa: E402


def main():
    comp = json.loads((config.RESULTS_DIR / "model_comparison.json").read_text())
    meta = json.loads(config.METADATA_PATH.read_text())
    raw, _ = load_cicids()
    df, _ = clean_dataset(raw)
    del raw
    eda.fig_binary_by_day(df, config.FIGURES_DIR / "eda_binary_by_day.png")
    _, test = split_train_test(df)
    del df
    y = test[config.BINARY_TARGET].to_numpy()
    curves = {}
    for row in comp:
        name = row["model"]
        path = config.MODELS_DIR / "candidates" / f"{name.split()[0].lower()}.joblib"
        pipe = joblib.load(path)
        proba = pipe.predict_proba(test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        m = ev.binary_metrics(y, pred, proba)
        for k in ("tp", "tn", "fp", "fn"):
            assert m[k] == row[k], f"{name}: {k} {m[k]} != saved {row[k]} — split or model differs"
        ev.plot_confusion(y, pred, [0, 1], ["BENIGN", "ATTACK"], f"{name} — test confusion matrix",
                          config.FIGURES_DIR / f"cm_{name.split()[0].lower()}.png")
        if name != "Dummy (majority)":
            curves[name] = (y, proba)
        if name == meta["final_model"]:
            ev.plot_confusion(y, pred, [0, 1], ["BENIGN", "ATTACK"],
                              f"Final model ({name}) — normalised",
                              config.FIGURES_DIR / "cm_final_normalised.png", normalize=True)
            ev.plot_threshold_sweep(y, proba, config.FIGURES_DIR / "threshold_sweep.png")
        print(f"verified {name}: metrics identical to saved results")
    ev.plot_roc_pr(curves, config.FIGURES_DIR / "roc_curves.png", config.FIGURES_DIR / "pr_curves.png")
    ev.plot_model_comparison(pd.DataFrame(comp), config.FIGURES_DIR / "model_comparison.png")
    mc = joblib.load(config.MULTICLASS_PIPELINE_PATH)
    labels = json.loads((config.RESULTS_DIR / "multiclass_results.json").read_text())["labels"]
    fam_pred = mc.predict(test)
    for norm, fname in ((True, "cm_multiclass_normalised.png"), (False, "cm_multiclass_counts.png")):
        ev.plot_confusion(test[config.FAMILY_TARGET], fam_pred, labels, labels,
                          f"Attack-family model ({meta['final_model']}) — "
                          + ("row-normalised" if norm else "counts"),
                          config.FIGURES_DIR / fname, normalize=norm)
    print("figures regenerated")


if __name__ == "__main__":
    main()
