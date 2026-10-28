"""End-to-end training and evaluation for NetGuard AI.

Run from the project root:
    python -m src.train                      # full run on data/raw (or data/)
    python -m src.train --data path/to/MachineLearningCSV.zip
    python -m src.train --max-train-rows 800000   # low-RAM laptops
    python -m src.train --quick              # fast smoke test (tiny grids)

Everything the website shows is written by this script to models/ and
reports/. Nothing here is ever run by the website.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import platform
import sys
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, recall_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
import joblib

from . import config
from . import eda, evaluate as ev, explainability as xai
from .data_loading import load_cicids
from .data_preprocessing import (build_preprocessor, clean_dataset, feature_columns,
                                 smote_floor_strategy, split_train_test, stratified_cap)
from .feature_engineering import ColumnSelector, compare_feature_sets

log = logging.getLogger("netguard.train")

SCORING = {"f1": "f1", "recall": "recall", "precision": "precision",
           "specificity": make_scorer(recall_score, pos_label=0)}


# ----------------------------------------------------------------------------
# Model zoo
# ----------------------------------------------------------------------------
def candidate_models(quick: bool, n_jobs: int, seed: int = config.RANDOM_STATE) -> dict:
    """name -> (estimator, needs_scaling, param_distributions, n_iter)"""
    rf_trees = 30 if quick else 100
    return {
        "Dummy (majority)": (DummyClassifier(strategy="most_frequent"), False, {}, 1),
        "Logistic Regression": (
            LogisticRegression(max_iter=300 if quick else 1000, class_weight="balanced"),
            True, {"model__C": [0.01, 0.1, 1.0, 10.0]}, 2 if quick else 4),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=seed),
            False, {"model__max_depth": [10, 20, None],
                    "model__min_samples_leaf": [1, 5, 20],
                    "model__criterion": ["gini", "entropy"],
                    "model__class_weight": [None, "balanced"]}, 2 if quick else 8),
        "Random Forest": (
            RandomForestClassifier(n_estimators=rf_trees, n_jobs=n_jobs, random_state=seed),
            False, {"model__max_depth": [20, 30, None],
                    "model__min_samples_leaf": [2, 5, 10],
                    "model__max_features": ["sqrt", 0.3],
                    "model__class_weight": [None, "balanced_subsample"]}, 2 if quick else 6),
        "HistGradientBoosting": (
            HistGradientBoostingClassifier(max_iter=60 if quick else 300, early_stopping=True,
                                           validation_fraction=0.1, n_iter_no_change=15,
                                           random_state=seed),
            False, {"model__learning_rate": [0.05, 0.1, 0.2],
                    "model__max_leaf_nodes": [31, 63, 127],
                    "model__l2_regularization": [0.0, 1.0],
                    "model__class_weight": [None, "balanced"]}, 2 if quick else 6),
    }


RF_MAX_SAMPLES = 400_000   # bootstrap rows per tree when fitting on large data


def fit_large(pipe, X, y):
    """Fit a pipeline on the full (large) training data.

    For Random Forest each tree is grown on a bootstrap sample of at most
    RF_MAX_SAMPLES rows (`max_samples`). This is standard RF subsampling: it bounds
    memory and model size (full-depth trees on millions of rows can need several
    GB) and cuts fit time, with little effect on accuracy. During CV tuning the
    folds are smaller than the cap, so trees there use full bootstrap samples."""
    est = pipe[-1]
    if isinstance(est, RandomForestClassifier) and len(X) > RF_MAX_SAMPLES:
        est.set_params(max_samples=RF_MAX_SAMPLES / len(X))
    return pipe.fit(X, y)


def make_pipeline(features: list[str], estimator, scale: bool) -> Pipeline:
    return Pipeline([("select", ColumnSelector(features)),
                     ("pre", build_preprocessor(scale)),
                     ("model", estimator)])


MULTICLASS_FIT_ROWS = 500_000   # family model: stratified cap, all rare-class rows kept
SMOTE_FLOOR = 20_000            # SMOTE tops minority classes up to this many rows (never down)
# NOTE: anything stored inside a saved pipeline must be defined in an importable
# module (not here): `python -m src.train` runs this file as __main__, and the
# website could not unpickle a function from __main__.


def multiclass_weight_param(est) -> dict:
    if isinstance(est, RandomForestClassifier):
        return {"class_weight": "balanced_subsample"}
    if isinstance(est, (HistGradientBoostingClassifier, LogisticRegression, DecisionTreeClassifier)):
        return {"class_weight": "balanced"}
    return {}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def total_ram_gb() -> float | None:
    """Physical RAM of this machine in GB (Windows, Linux, macOS), or None."""
    try:
        if sys.platform.startswith("win"):
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return st.ullTotalPhys / 2**30
        import os
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except Exception:
        return None


LOW_RAM_GB = 12
LOW_RAM_FIT_CAP = 1_000_000


def resolve_fit_cap(arg: str) -> tuple[int | None, str]:
    """--max-train-rows: 'auto' (default), 'all', or a number."""
    if arg in ("all", "0", "none"):
        return None, "all training rows (requested)"
    if arg != "auto":
        return int(arg), f"capped at {int(arg):,} rows (requested)"
    ram = total_ram_gb()
    if ram is not None and ram < LOW_RAM_GB:
        return LOW_RAM_FIT_CAP, (f"capped at {LOW_RAM_FIT_CAP:,} rows automatically "
                                 f"({ram:.1f} GB RAM < {LOW_RAM_GB} GB)")
    return None, f"all training rows ({'unknown' if ram is None else f'{ram:.1f} GB'} RAM)"


def timed(fn, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    return out, time.perf_counter() - t0


def feature_stats(X: pd.DataFrame) -> dict:
    q = X.quantile([0.0, 0.01, 0.5, 0.99, 1.0])
    return {c: {"min": float(q.loc[0.0, c]), "p01": float(q.loc[0.01, c]),
                "median": float(q.loc[0.5, c]), "p99": float(q.loc[0.99, c]),
                "max": float(q.loc[1.0, c])} for c in X.columns}


def fmt(v, nd=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:.{nd}f}" if isinstance(v, (float, np.floating)) else str(v)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def run(args) -> dict:
    config.ensure_dirs()
    seed = config.RANDOM_STATE
    np.random.seed(seed)
    t_start = time.perf_counter()
    R = config.RESULTS_DIR
    F = config.FIGURES_DIR

    # 1. Load + profile ---------------------------------------------------
    log.info("Step 1/9  loading data")
    raw, load_report = load_cicids(args.data)
    log.info("Loaded %s rows x %s columns from %d file(s)", f"{len(raw):,}", raw.shape[1],
             len(load_report.files))
    profile = eda.profile_dataset(raw, load_report)
    ev.save_json(profile, config.PROFILE_PATH)

    # 2. Clean -------------------------------------------------------------
    log.info("Step 2/9  cleaning (inf->NaN, targets, de-duplication)")
    df, clean_report = clean_dataset(raw)
    del raw
    ev.save_json(clean_report, R / "cleaning_report.json")
    feats_all = feature_columns(df)
    log.info("After cleaning: %s rows; %d eligible features", f"{len(df):,}", len(feats_all))

    if not args.skip_eda:
        corr_feats = [c for c in feats_all if df[c].std() > 0][:22]
        figs = eda.make_eda_figures(profile, df, clean_report["deduplication"], corr_feats)
        log.info("EDA figures: %s", figs)

    # 3. Split -------------------------------------------------------------
    log.info("Step 3/9  stratified %d/%d split (seed %d)", 100 - 100 * config.TEST_SIZE,
             100 * config.TEST_SIZE, seed)
    train, test = split_train_test(df, config.TEST_SIZE, seed)
    del df
    gc.collect()
    split_info = {
        "strategy": "stratified random split on the fine-grained label after de-duplication",
        "test_size": config.TEST_SIZE, "random_state": seed,
        "train_rows": len(train), "test_rows": len(test),
        "train_by_label": train[config.LABEL_COL].value_counts().to_dict(),
        "test_by_label": test[config.LABEL_COL].value_counts().to_dict(),
        "train_attack_share": float(train[config.BINARY_TARGET].mean()),
        "test_attack_share": float(test[config.BINARY_TARGET].mean()),
    }
    rare = [k for k, v in split_info["train_by_label"].items() if v < config.RARE_CLASS_THRESHOLD]
    split_info["rare_classes_lt_threshold"] = rare
    ev.save_json(split_info, R / "split_info.json")

    # Training rows actually used for fitting (all of them unless capped)
    fit_cap, fit_cap_reason = resolve_fit_cap(str(args.max_train_rows).lower())
    args.max_train_rows = fit_cap
    log.info("Final-model fitting rows: %s", fit_cap_reason)
    fit_df = stratified_cap(train, fit_cap, seed)
    tune_df = stratified_cap(train, args.tune_rows, seed)
    n_fit, n_tune = len(fit_df), len(tune_df)
    log.info("Fitting rows: %s | tuning subsample: %s", f"{n_fit:,}", f"{n_tune:,}")

    # 4. Feature selection (training subsample, CV) -------------------------
    log.info("Step 4/9  feature-set comparison (CV on training subsample)")
    fs = compare_feature_sets(tune_df[feats_all], tune_df[config.BINARY_TARGET],
                              top_k=min(30, len(feats_all)), seed=seed,
                              n_estimators=20 if args.quick else 50)
    features = fs["chosen_features"]
    ev.save_json({k: v for k, v in fs.items() if k != "sets"} | {"sets": fs["sets"]},
                 R / "feature_selection.json")
    log.info("Chosen feature set: %s (%d features)", fs["chosen"], len(features))

    # 5. Tune + fit + evaluate each model ------------------------------------
    log.info("Step 5/9  model tuning and comparison")
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=seed)
    X_tune, y_tune = tune_df[features], tune_df[config.BINARY_TARGET]
    X_fit, y_fit = fit_df[features], fit_df[config.BINARY_TARGET]
    X_test, y_test = test[features], test[config.BINARY_TARGET].to_numpy()

    models = candidate_models(args.quick, args.n_jobs, seed)
    results, fitted, curves = [], {}, {}
    for name, (est, scale, params, n_iter) in models.items():
        log.info("  %s", name)
        pipe = make_pipeline(features, est, scale)
        if params:
            # refit=False: the search must not refit the winner on the tuning sample and
            # keep it in memory — the final model is refitted on the full training data
            # below. (Saves one fit per model and, for Random Forest, several GB of RAM.)
            search = RandomizedSearchCV(pipe, params, n_iter=n_iter, scoring=SCORING, refit=False,
                                        cv=cv, random_state=seed, n_jobs=1,
                                        return_train_score=True)
            _, t_search = timed(search.fit, X_tune, y_tune)
            cvr = search.cv_results_
            i = int(np.argmin(cvr["rank_test_f1"]))          # best mean CV F1
            best_raw = cvr["params"][i]
            best_params = {k.replace("model__", ""): v for k, v in best_raw.items()}
            cv_scores = {m: float(cvr[f"mean_test_{m}"][i]) for m in SCORING}
            cv_std_f1 = float(cvr["std_test_f1"][i])
            train_f1 = float(cvr["mean_train_f1"][i])
            final = clone(pipe).set_params(**best_raw)
            search_table = pd.DataFrame(cvr)[["params", "mean_test_f1", "std_test_f1",
                                              "mean_test_recall", "mean_test_specificity",
                                              "mean_train_f1", "mean_fit_time"]]
            search_table["params"] = search_table["params"].astype(str)
            search_table.to_csv(R / f"cv_search_{name.split()[0].lower()}.csv", index=False)
            del search
            gc.collect()
        else:
            res, t_search = timed(cross_validate, pipe, X_tune, y_tune, cv=cv, scoring=SCORING,
                                  return_train_score=True)
            best_params, cv_std_f1 = {}, float(res["test_f1"].std())
            cv_scores = {m: float(res[f"test_{m}"].mean()) for m in SCORING}
            train_f1 = float(res["train_f1"].mean())
            final = clone(pipe)

        _, t_fit = timed(fit_large, final, X_fit, y_fit)
        t0 = time.perf_counter()
        proba = final.predict_proba(X_test)[:, 1]
        t_pred = time.perf_counter() - t0
        pred = (proba >= 0.5).astype(int)
        m = ev.binary_metrics(y_test, pred, proba if name != "Dummy (majority)" else None)
        row = {"model": name, **{k: m[k] for k in ("accuracy", "precision", "recall", "f1",
                                                   "specificity", "fpr", "fnr", "roc_auc",
                                                   "pr_auc", "tp", "tn", "fp", "fn")},
               "cv_f1": cv_scores["f1"], "cv_f1_std": cv_std_f1, "cv_recall": cv_scores["recall"],
               "cv_specificity": cv_scores["specificity"], "cv_train_f1": train_f1,
               "overfit_gap_f1": train_f1 - cv_scores["f1"],
               "tuning_seconds": round(t_search, 1), "fit_seconds": round(t_fit, 1),
               "predict_ms_per_1000_rows": round(1000 * t_pred / max(len(X_test), 1) * 1000, 3),
               "best_params": best_params}
        results.append(row)
        # Keep memory flat: park each fitted candidate on disk, reload only the winner.
        cand_dir = config.MODELS_DIR / "candidates"
        cand_dir.mkdir(parents=True, exist_ok=True)
        cand_path = cand_dir / f"{name.split()[0].lower()}.joblib"
        joblib.dump(final, cand_path)
        row["model_file_mb"] = round(cand_path.stat().st_size / 2**20, 1)
        fitted[name] = cand_path
        if name != "Dummy (majority)":
            curves[name] = (y_test, proba)
        log.info("    CV F1=%.4f | test F1=%s recall=%s FPR=%s | fit %.1fs", cv_scores["f1"],
                 fmt(m["f1"]), fmt(m["recall"]), fmt(m["fpr"]), t_fit)
        ev.plot_confusion(y_test, pred, [0, 1], ["BENIGN", "ATTACK"],
                          f"{name} — test confusion matrix", F / f"cm_{name.split()[0].lower()}.png")
        del final, proba, pred
        gc.collect()

    comp = pd.DataFrame(results)
    comp.drop(columns=["best_params"]).to_csv(R / "model_comparison.csv", index=False)
    ev.save_json(results, R / "model_comparison.json")
    ev.plot_model_comparison(comp, F / "model_comparison.png")
    ev.plot_roc_pr(curves, F / "roc_curves.png", F / "pr_curves.png")

    # 6. Choose the final model WITHOUT looking at test results ---------------
    cand = comp[comp["model"] != "Dummy (majority)"].copy()
    best_cv = cand["cv_f1"].max()
    tied = cand[cand["cv_f1"] >= best_cv - 0.001]
    chosen = tied.sort_values(["cv_specificity", "predict_ms_per_1000_rows"],
                              ascending=[False, True]).iloc[0]
    final_name = chosen["model"]
    final_pipe = joblib.load(fitted[final_name])
    selection = {"final_model": final_name,
                 "rule": "highest mean CV F1 (ATTACK) on the training subsample; models within "
                         "0.001 are tie-broken by higher CV specificity, then faster prediction. "
                         "Test-set results were NOT used for this choice.",
                 "cv_f1_by_model": dict(zip(cand["model"], cand["cv_f1"]))}
    log.info("Step 6/9  final model: %s", final_name)

    proba = final_pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    final_metrics = ev.binary_metrics(y_test, pred, proba)
    ev.plot_confusion(y_test, pred, [0, 1], ["BENIGN", "ATTACK"],
                      f"Final model ({final_name}) — normalised", F / "cm_final_normalised.png",
                      normalize=True)
    _, sweep = ev.plot_threshold_sweep(y_test, proba, F / "threshold_sweep.png")

    # Error analysis
    fam = ev.detection_by_family(test[config.FAMILY_TARGET], y_test, pred)
    fine = ev.detection_by_family(test[config.LABEL_COL], y_test, pred)
    ev.plot_family_detection(fam, f"Final model: share flagged ATTACK per family (test)",
                             F / "detection_by_family.png")
    err = test.assign(pred=pred, p_attack=proba)
    fn_rows = err[(err[config.BINARY_TARGET] == 1) & (err.pred == 0)]
    fp_rows = err[(err[config.BINARY_TARGET] == 0) & (err.pred == 1)]
    tn_rows = err[(err[config.BINARY_TARGET] == 0) & (err.pred == 0)]
    tp_rows = err[(err[config.BINARY_TARGET] == 1) & (err.pred == 1)]
    imp = xai.builtin_importance(final_pipe)
    top_feats = [f for f in (imp.index[:8] if imp is not None else features[:8]) if f in features]

    def med(d):
        return {f: (float(d[f].median()) if len(d) else None) for f in top_feats}
    error_analysis = {
        "false_negatives": int(len(fn_rows)), "false_positives": int(len(fp_rows)),
        "false_negatives_by_label": fn_rows[config.LABEL_COL].value_counts().to_dict(),
        "fn_p_attack_quantiles": fn_rows["p_attack"].quantile([.1, .5, .9]).to_dict() if len(fn_rows) else {},
        "fp_p_attack_quantiles": fp_rows["p_attack"].quantile([.1, .5, .9]).to_dict() if len(fp_rows) else {},
        "median_top_features": {"false_positive": med(fp_rows), "true_negative": med(tn_rows),
                                "false_negative": med(fn_rows), "true_positive": med(tp_rows)},
        "detection_by_family": fam.to_dict(orient="records"),
        "detection_by_label": fine.to_dict(orient="records"),
        "threshold_sweep": sweep,
    }
    ev.save_json(error_analysis, R / "error_analysis.json")

    # 7. Explainability ------------------------------------------------------
    log.info("Step 7/9  explainability")
    expl = {"method_notes": "Built-in importance = impurity decrease (trees) or coefficients "
                            "(linear). Permutation importance = drop in test F1 when a column is "
                            "shuffled. SHAP = per-flow additive contributions (tree models)."}
    if imp is not None:
        expl["builtin_importance"] = imp.to_dict()
        xai.plot_importance(imp.abs() if imp.lt(0).any() else imp,
                            f"{final_name}: built-in feature importance",
                            "Importance (normalised)" if imp.ge(0).all() else "|coefficient|",
                            F / "importance_builtin.png")
    perm = xai.permutation_ranking(final_pipe, X_test, y_test,
                                   n_repeats=2 if args.quick else 5,
                                   max_rows=3000 if args.quick else 20_000)
    expl["permutation_importance"] = perm.to_dict(orient="records")
    xai.plot_importance(perm.set_index("feature")["importance_mean"],
                        "Permutation importance (drop in test F1)", "Mean decrease in F1",
                        F / "importance_permutation.png",
                        err=perm.set_index("feature")["importance_std"])
    shap_rank = xai.shap_global(final_pipe, X_test, F / "importance_shap.png",
                                max_rows=300 if args.quick else 2000)
    expl["shap_global"] = shap_rank.to_dict() if shap_rank is not None else None
    expl["shap_available_for_final_model"] = shap_rank is not None
    ev.save_json(expl, R / "explainability.json")

    # 8. Multiclass family model + imbalance strategy comparison ---------------
    multi = None
    if not args.skip_multiclass:
        log.info("Step 8/9  multiclass attack-family model")
        multi = run_multiclass(final_name, final_pipe, features, tune_df, fit_df, test, args, cv)

    # Held-out-day experiment ---------------------------------------------------
    holdout = None
    if not args.skip_holdout:
        log.info("Step 8b    held-out-day experiment (train Mon–Thu, test Friday)")
        cols = [config.SOURCE_COL, config.LABEL_COL, config.FAMILY_TARGET, config.BINARY_TARGET] + features
        del tune_df
        gc.collect()
        holdout = run_holdout(final_pipe, features, pd.concat([train[cols], test[cols]],
                                                               ignore_index=True), args)
        gc.collect()

    # 9. Save artifacts ---------------------------------------------------------
    log.info("Step 9/9  saving artifacts")
    joblib.dump(final_pipe, config.FINAL_PIPELINE_PATH, compress=3)
    schema = {
        "features": features,
        "dtype": "float (numeric); inf treated as missing; missing imputed with training median",
        "positive_class": config.POSITIVE_LABEL,
        "class_mapping": {str(k): v for k, v in config.BINARY_CLASS_NAMES.items()},
        "excluded_features": sorted(config.EXCLUDED_FEATURES),
        "training_feature_stats": feature_stats(fit_df[features]),
        "families": sorted(test[config.FAMILY_TARGET].unique().tolist()),
    }
    ev.save_json(schema, config.SCHEMA_PATH)

    # Demo file: real held-out TEST records (never used for fitting or tuning)
    shuffled = test.sample(frac=1.0, random_state=seed)
    demo = pd.concat([shuffled[shuffled[config.FAMILY_TARGET] == "BENIGN"].head(300),
                      shuffled[shuffled[config.FAMILY_TARGET] != "BENIGN"]
                      .groupby(config.FAMILY_TARGET, observed=True).head(60)])
    demo_cols = [c for c in feats_all if c in test.columns] + sorted(config.EXCLUDED_FEATURES & set(test.columns)) \
        + [config.LABEL_COL]
    demo[[c for c in dict.fromkeys(demo_cols)]].sample(frac=1, random_state=seed) \
        .to_csv(config.DEMO_SAMPLE_PATH, index=False)

    meta = {
        "project": "NetGuard AI", "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "final_model": final_name, "best_params": chosen["best_params"],
        "decision_threshold": 0.5, "task": "binary: ATTACK (1) vs BENIGN (0)",
        "selection": selection, "feature_set": fs["chosen"], "n_features": len(features),
        "dataset": {"name": "CIC-IDS2017 (MachineLearningCSV)",
                    "source": "https://www.unb.ca/cic/datasets/ids-2017.html",
                    "files": load_report.files, "rows_raw": profile["n_rows"],
                    "rows_after_cleaning": clean_report["deduplication"]["rows_after"]},
        "split": {k: split_info[k] for k in ("strategy", "test_size", "random_state",
                                             "train_rows", "test_rows")},
        "rows_used_for_fitting": n_fit, "rows_used_for_tuning": n_tune,
        "max_train_rows_arg": args.max_train_rows, "fit_rows_policy": fit_cap_reason,
        "ram_gb": round(total_ram_gb() or 0, 1), "quick_mode": bool(args.quick),
        "data_path_arg": str(args.data) if args.data else None,
        "reportable": (not args.quick) and profile["n_rows"] > 1_000_000,
        "test_metrics": final_metrics,
        "multiclass_available": multi is not None,
        "holdout_day": holdout["metrics"] if holdout else None,
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                        "pandas": pd.__version__, "numpy": np.__version__,
                        "platform": platform.platform()},
        "total_runtime_minutes": round((time.perf_counter() - t_start) / 60, 1),
    }
    ev.save_json(meta, config.METADATA_PATH)
    write_summary(meta, comp, fs, error_analysis, multi, holdout, split_info, clean_report, profile)
    log.info("Done in %.1f min. Start the website with:  streamlit run app/app.py",
             meta["total_runtime_minutes"])
    return meta


def run_multiclass(final_name, final_pipe, features, tune_df, fit_df, test, args, cv) -> dict:
    R, F, seed = config.RESULTS_DIR, config.FIGURES_DIR, config.RANDOM_STATE
    base_est = clone(final_pipe.named_steps["model"])
    if isinstance(base_est, DummyClassifier):
        return None
    scale = "log" in final_pipe.named_steps["pre"].named_steps
    yk = config.FAMILY_TARGET

    # Imbalance strategy comparison (CV, training subsample only)
    small = stratified_cap(tune_df, 30_000 if args.quick else 100_000, seed, min_per_class=1500)
    min_count = int(small[yk].value_counts().min())
    strategies = {"none": make_pipeline(features, clone(base_est), scale),
                  "class_weight": make_pipeline(features, clone(base_est).set_params(
                      **multiclass_weight_param(base_est)), scale)}
    smote_note = None
    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        k = min(5, max(1, int(min_count * (cv.n_splits - 1) / cv.n_splits) - 1))
        # imblearn pipelines cannot nest a Pipeline, so the preprocessing steps
        # are flattened; SMOTE runs after imputation (it cannot handle NaN).
        from functools import partial
        floor = 2_000 if args.quick else SMOTE_FLOOR
        strategies["smote_in_fold"] = ImbPipeline(
            [("select", ColumnSelector(features))] + build_preprocessor(scale).steps +
            [("smote", SMOTE(k_neighbors=k, random_state=seed,
                             sampling_strategy=partial(smote_floor_strategy, floor=floor))),
             ("model", clone(base_est))])
        smote_note = (f"SMOTE applied inside each training fold only (k_neighbors={k}); classes "
                      f"below {floor:,} rows are topped up to {floor:,}")
    except ImportError:
        smote_note = "imbalanced-learn not installed; SMOTE strategy skipped"
    imb_rows = []
    for sname, p in strategies.items():
        r = cross_validate(p, small, small[yk], cv=cv, n_jobs=1,
                           scoring={"macro_f1": "f1_macro", "weighted_f1": "f1_weighted",
                                    "balanced_acc": "balanced_accuracy"})
        imb_rows.append({"strategy": sname, "cv_macro_f1": float(r["test_macro_f1"].mean()),
                         "cv_weighted_f1": float(r["test_weighted_f1"].mean()),
                         "cv_balanced_accuracy": float(r["test_balanced_acc"].mean())})
        log.info("    imbalance strategy %-14s macro-F1=%.4f", sname, imb_rows[-1]["cv_macro_f1"])
    best_s = max(imb_rows, key=lambda r: r["cv_macro_f1"])["strategy"]
    mpipe = clone(strategies[best_s])
    m_fit = stratified_cap(fit_df, MULTICLASS_FIT_ROWS, seed)
    _, t_fit = timed(fit_large, mpipe, m_fit, m_fit[yk])
    pred = mpipe.predict(test)
    labels = [l for l in ["BENIGN", "DoS", "DDoS", "PortScan", "BruteForce", "WebAttack", "Bot",
                          "Infiltration", "Heartbleed", "Unknown"] if l in set(test[yk])]
    report = ev.multiclass_report(test[yk], pred, labels)
    ev.plot_confusion(test[yk], pred, labels, labels,
                      f"Attack-family model ({final_name}) — row-normalised",
                      F / "cm_multiclass_normalised.png", normalize=True)
    ev.plot_confusion(test[yk], pred, labels, labels,
                      f"Attack-family model ({final_name}) — counts", F / "cm_multiclass_counts.png")
    joblib.dump(mpipe, config.MULTICLASS_PIPELINE_PATH, compress=3)
    out = {"estimator": final_name, "imbalance_comparison": imb_rows, "chosen_strategy": best_s,
           "smote_note": smote_note, "selection_rule": "highest CV macro-F1 on training subsample",
           "fit_seconds": round(t_fit, 1), "fit_rows": len(m_fit),
           "test_report": report, "labels": labels,
           "rare_classes_note": f"Classes with < {config.RARE_CLASS_THRESHOLD} training rows are "
                                "kept but their per-class scores rest on very few test rows."}
    ev.save_json(out, R / "multiclass_results.json")
    return out


def run_holdout(final_pipe, features, src: pd.DataFrame, args) -> dict | None:
    is_test = src[config.SOURCE_COL].astype(str).str.startswith(config.HOLDOUT_TEST_PREFIX)
    tr, te = src[~is_test], src[is_test]
    if len(tr) == 0 or len(te) == 0 or tr[config.BINARY_TARGET].nunique() < 2:
        log.warning("Held-out-day experiment skipped (needs Friday and non-Friday files)")
        return None
    tr = stratified_cap(tr, args.max_train_rows, config.RANDOM_STATE)
    p = clone(final_pipe)
    fit_large(p, tr[features], tr[config.BINARY_TARGET])
    proba = p.predict_proba(te[features])[:, 1]
    pred = (proba >= 0.5).astype(int)
    m = ev.binary_metrics(te[config.BINARY_TARGET], pred, proba)
    fam = ev.detection_by_family(te[config.LABEL_COL], te[config.BINARY_TARGET], pred)
    ev.plot_family_detection(fam, "Held-out day: train Mon–Thu, test Friday",
                             config.FIGURES_DIR / "holdout_day_detection.png")
    seen = set(tr[config.LABEL_COL])
    out = {"train_files": sorted(tr[config.SOURCE_COL].astype(str).unique()),
           "test_files": sorted(te[config.SOURCE_COL].astype(str).unique()),
           "train_rows": len(tr), "test_rows": len(te), "metrics": m,
           "detection_by_label": fam.to_dict(orient="records"),
           "test_labels_unseen_in_training": sorted(set(te[config.LABEL_COL]) - seen)}
    ev.save_json(out, config.RESULTS_DIR / "holdout_day_results.json")
    log.info("    held-out day: recall=%s FPR=%s", fmt(m["recall"]), fmt(m["fpr"]))
    return out


def write_summary(meta, comp, fs, err, multi, holdout, split, clean, profile):
    """Human-readable results file generated from the actual run."""
    L = [f"# NetGuard AI — results summary", "",
         f"_Generated automatically by `src/train.py` on {meta['created_utc']}"
         f"{'' if meta['reportable'] else ' — NOT REPORTABLE: quick mode or a small/synthetic dataset (smoke test only)'}._", "",
         "## Data", "",
         f"- Raw rows: {profile['n_rows']:,} from {len(profile['files'])} file(s)",
         f"- Exact duplicates removed: {clean['deduplication']['exact_duplicates_removed']:,}; "
         f"label-conflicting rows removed: {clean['deduplication']['conflicting_label_rows_removed']:,}",
         f"- Rows after cleaning: {clean['deduplication']['rows_after']:,}",
         f"- Train / test rows: {split['train_rows']:,} / {split['test_rows']:,} "
         f"(attack share {split['train_attack_share']:.4f} / {split['test_attack_share']:.4f})",
         f"- Rows used to fit final models: {meta['rows_used_for_fitting']:,} ({meta['fit_rows_policy']}); "
         f"tuning subsample: {meta['rows_used_for_tuning']:,}", "",
         f"## Feature selection ({fs['rule']})", "",
         "| Feature set | n | CV F1 | CV recall |", "|---|---:|---:|---:|"]
    for r in fs["comparison"]:
        L.append(f"| {r['feature_set']} | {r['n_features']} | {r['cv_f1_mean']:.4f} | {r['cv_recall_mean']:.4f} |")
    L += ["", f"Chosen: **{fs['chosen']}**", "", "## Binary model comparison (test set, positive = ATTACK)", "",
          "| Model | CV F1 | Accuracy | Precision | Recall | F1 | Specificity | FPR | FNR | ROC-AUC | PR-AUC | Fit s |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in comp.iterrows():
        L.append(f"| {r['model']} | {fmt(r['cv_f1'])} | {fmt(r['accuracy'])} | {fmt(r['precision'])} | "
                 f"{fmt(r['recall'])} | {fmt(r['f1'])} | {fmt(r['specificity'])} | {fmt(r['fpr'], 5)} | "
                 f"{fmt(r['fnr'], 5)} | {fmt(r['roc_auc'])} | {fmt(r['pr_auc'])} | {r['fit_seconds']} |")
    L += ["", f"**Final model:** {meta['final_model']} — {meta['selection']['rule']}", "",
          f"False negatives (missed attacks): {err['false_negatives']:,}; "
          f"false positives (false alarms): {err['false_positives']:,}", "",
          "Missed attacks by label: " + ", ".join(f"{k}: {v}" for k, v in err["false_negatives_by_label"].items()), ""]
    if multi:
        L += ["## Attack-family model", "",
              "| Imbalance strategy | CV macro-F1 | CV balanced acc. |", "|---|---:|---:|"]
        for r in multi["imbalance_comparison"]:
            L.append(f"| {r['strategy']} | {r['cv_macro_f1']:.4f} | {r['cv_balanced_accuracy']:.4f} |")
        rep = multi["test_report"]
        L += ["", f"Chosen: {multi['chosen_strategy']}. Test macro-F1 = {rep['macro avg']['f1-score']:.4f}, "
              f"weighted-F1 = {rep['weighted avg']['f1-score']:.4f}", "",
              "| Family | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
        for lab in multi["labels"]:
            r = rep[lab]
            L.append(f"| {lab} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1-score']:.4f} | {int(r['support'])} |")
        L.append("")
    if holdout:
        m = holdout["metrics"]
        L += ["## Held-out day (generalisation check)", "",
              f"Train on {len(holdout['train_files'])} Mon–Thu files, test on Friday "
              f"({holdout['test_rows']:,} flows). Labels unseen in training: "
              f"{holdout['test_labels_unseen_in_training']}", "",
              f"Recall {fmt(m['recall'])}, precision {fmt(m['precision'])}, FPR {fmt(m['fpr'], 5)}, "
              f"F1 {fmt(m['f1'])}, ROC-AUC {fmt(m['roc_auc'])}", ""]
    (config.RESULTS_DIR / "results_summary.md").write_text("\n".join(L), encoding="utf-8")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train and evaluate NetGuard AI models")
    p.add_argument("--data", default=None, help="Folder with the CSVs, one CSV, or the zip")
    p.add_argument("--max-train-rows", default="auto",
                   help="Rows used to fit the final models: 'auto' (default: all rows, or a "
                        f"{LOW_RAM_FIT_CAP:,}-row stratified cap on machines with < {LOW_RAM_GB} GB RAM), "
                        "'all', or a number. Stratified; rare classes are always kept in full.")
    p.add_argument("--tune-rows", type=int, default=200_000,
                   help="Size of the stratified training subsample used for CV tuning")
    p.add_argument("--n-jobs", type=int, default=-1, help="CPU cores for Random Forest")
    p.add_argument("--quick", action="store_true", help="Tiny grids — smoke test only")
    p.add_argument("--skip-eda", action="store_true")
    p.add_argument("--skip-holdout", action="store_true")
    p.add_argument("--skip-multiclass", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("once", category=ConvergenceWarning)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    args = parse_args(argv)
    try:
        run(args)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        sys.exit(2)


if __name__ == "__main__":
    main()
