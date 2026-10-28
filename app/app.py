"""NetGuard AI — Streamlit website.

Start from the project root:   streamlit run app/app.py
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import components as ui
from components import C, PLOTLY_LAYOUT
from src import config
from src.predict import (predict, predict_family, read_uploaded_csv, realism_check,
                         realism_summary, validate_frame)

st.set_page_config(page_title="NetGuard AI", page_icon="🛡️", layout="wide")
ui.inject_css()

DISCLAIMER = ("NetGuard AI is an educational prototype. It analyses flow records you supply "
              "(offline CSV analysis); it does not capture or monitor live network traffic, and "
              "its output is a model estimate, not a security determination.")


def plot(fig, height=None):
    fig.update_layout(**PLOTLY_LAYOUT)
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


# ============================================================================
# A. Dashboard
# ============================================================================
def page_home():
    st.title("NetGuard AI")
    st.markdown('<p class="ng-lede">Machine-learning classification of network flows as benign '
                'or malicious, trained and evaluated on the CIC-IDS2017 intrusion dataset.</p>',
                unsafe_allow_html=True)
    ui.note(DISCLAIMER)
    if not ui.require_artifacts():
        return
    ui.reportability_banner()
    meta, m = ui.metadata(), ui.metadata()["test_metrics"]

    st.markdown(f"## {meta['final_model']} on the held-out test set")
    ui.ledger(m)
    st.caption("Flows never used for training or tuning. Positive class = ATTACK. The two small "
               "outcomes are drawn at least 2 px wide so they stay visible.")
    c = st.columns(5)
    c[0].metric("Recall", ui.pct(m["recall"]), help="Share of attack flows that were caught")
    c[1].metric("Precision", ui.pct(m["precision"]), help="Share of ATTACK alerts that were real attacks")
    c[2].metric("F1 score", ui.num(m["f1"]), help="Harmonic mean of precision and recall")
    c[3].metric("False-alarm rate", ui.pct(m["fpr"], 3), help="Share of benign flows wrongly flagged (FPR)")
    c[4].metric("Missed attacks", f"{m['fn']:,}", help="False negatives on the test set")

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("## What it does")
        st.markdown(
            "- Reads flow records with the CICIDS2017 feature columns (CSV upload or a form).\n"
            "- Predicts **ATTACK** or **BENIGN** with the probability P(attack).\n"
            "- Optionally estimates the attack family (DoS, DDoS, PortScan, brute force, …).\n"
            "- Explains which features drove the model, globally and per flow.")
    with right:
        st.markdown("## This build")
        d = meta["dataset"]
        ui.kv_table([
            ("Dataset", f'<a href="{d["source"]}">{d["name"]}</a>'),
            ("Flows (raw → cleaned)", f'{d["rows_raw"]:,} → {d["rows_after_cleaning"]:,}'),
            ("Split", f'{meta["split"]["train_rows"]:,} train / {meta["split"]["test_rows"]:,} test, '
                      f'stratified, seed {meta["split"]["random_state"]}'),
            ("Model", f'{meta["final_model"]} ({meta["n_features"]} features)'),
            ("Decision threshold", f'P(attack) ≥ {meta["decision_threshold"]}'),
            ("Trained", meta["created_utc"].replace("T", " ")),
        ])
    st.markdown("## Known limits")
    hold = meta.get("holdout_day")
    hold_line = (f"- **New attack types are largely missed.** Trained on Monday–Thursday only, the "
                 f"same model caught **{ui.pct(hold['recall'], 1)}** of Friday's attacks (Bot, "
                 "PortScan, DDoS — never seen in training). High random-split scores do not mean "
                 "unseen attacks are detected; see Model performance.\n") if hold else (
                 "- High test scores on a random split do not prove the model catches new attack "
                 "types.\n")
    st.markdown(
        hold_line +
        "- Trained on a 2017 laboratory testbed; real networks differ (distribution shift).\n"
        "- Published studies report labelling and flow-construction errors in CICIDS2017.")


# ============================================================================
# B. Dataset explorer
# ============================================================================
def page_dataset():
    st.title("Dataset explorer")
    st.markdown('<p class="ng-lede">The training data as it was measured during training. '
                'Files you upload for prediction are not part of it.</p>', unsafe_allow_html=True)
    if not ui.require_artifacts():
        return
    prof = ui.require_results("dataset_profile.json", "Dataset profile")
    clean = ui.results("cleaning_report.json")
    if prof is None:
        return
    ui.reportability_banner()
    c = st.columns(4)
    c[0].metric("Raw flows", f"{prof['n_rows']:,}")
    c[1].metric("Numeric features", prof["n_numeric_features"])
    c[2].metric("Classes", len(prof["class_distribution"]))
    if clean:
        c[3].metric("Flows after cleaning", f"{clean['deduplication']['rows_after']:,}")

    st.markdown("## Class distribution (raw data)")
    cd = pd.DataFrame({"label": list(prof["class_distribution"]),
                       "flows": list(prof["class_distribution"].values())}).sort_values("flows")
    cd["kind"] = np.where(cd["label"] == "BENIGN", "Benign", "Attack")
    fig = px.bar(cd, x="flows", y="label", orientation="h", color="kind", log_x=True,
                 color_discrete_map={"Benign": C["ok"], "Attack": C["attack"]},
                 labels={"flows": "Flows (log scale)", "label": ""})
    fig.update_traces(hovertemplate="<b>%{y}</b><br>%{x:,} flows<extra></extra>")
    fig.update_layout(legend_title_text="")
    plot(fig, height=40 + 26 * len(cd))
    st.caption(f"Imbalance: the largest class is {prof['imbalance_ratio_largest_to_smallest']:,.0f}× "
               "the smallest. Accuracy alone would therefore be misleading.")

    st.markdown("## Data-quality findings")
    q1, q2 = st.columns(2)
    with q1:
        st.markdown("**Non-finite values**")
        cols = sorted(set(prof["missing_values"]) | set(prof["infinite_values"]))
        if cols:
            st.dataframe(pd.DataFrame({"column": cols,
                                       "missing": [prof["missing_values"].get(x, 0) for x in cols],
                                       "infinite": [prof["infinite_values"].get(x, 0) for x in cols]}),
                         hide_index=True, width="stretch")
        else:
            st.write("None found.")
        st.markdown(f"**Constant columns** ({len(prof['constant_columns'])}): "
                    + (", ".join(prof["constant_columns"]) or "none"))
        st.markdown(f"**Duplicated column dropped:** {', '.join(prof['duplicate_columns_dropped']) or 'none'}")
    with q2:
        if clean:
            d = clean["deduplication"]
            ui.kv_table([("Exact duplicate rows removed", f"{d['exact_duplicates_removed']:,}"),
                         ("Rows with conflicting labels removed", f"{d['conflicting_label_rows_removed']:,}"),
                         ("Rows kept", f"{d['rows_after']:,}")])
            st.caption("Duplicates were removed before splitting so the same flow cannot appear "
                       "in both the training and the test set.")
        st.markdown("**Label spellings normalised**")
        variants = {k: v for k, v in prof["label_spelling_variants"].items() if k != v}
        st.write(variants or "No variants.")

    st.markdown("## Flows per capture day")
    lpf = pd.DataFrame(prof["labels_per_file"]).T.fillna(0).astype(int)
    lpf.index = [i.replace(".pcap_ISCX.csv", "") for i in lpf.index]
    st.dataframe(lpf, width="stretch")
    st.caption("Each attack type was run on a specific day, which is why a held-out-day test is a "
               "much harder check than a random split.")

    st.markdown("## Exploratory figures")
    tabs = st.tabs(["Distributions", "Families", "Correlation", "Per day", "Quality", "Scatter"])
    with tabs[0]:
        ui.figure("eda_feature_distributions.png", "Log-transformed distributions of key features, by class")
    with tabs[1]:
        ui.figure("eda_family_boxplots.png", "Feature ranges per attack family")
    with tabs[2]:
        ui.figure("eda_correlation_heatmap.png", "Pearson correlation of selected features")
    with tabs[3]:
        ui.figure("eda_binary_by_day.png", "Benign and attack flows per capture day (after cleaning)")
    with tabs[4]:
        ui.figure("eda_data_quality.png", "Non-finite values and duplicate share per class")
    with tabs[5]:
        ui.figure("eda_duration_vs_packets.png", "Flow duration vs forward packets, by family")

    st.markdown("## Feature reference")
    sch = ui.schema()
    if sch:
        stats = pd.DataFrame(sch["training_feature_stats"]).T
        stats.insert(0, "used by model", True)
        st.dataframe(stats, width="stretch")
        st.caption(f"Statistics of the {len(sch['features'])} model inputs on the training rows. "
                   f"Excluded on purpose: {', '.join(sch['excluded_features'])}.")


# ============================================================================
# C. Model performance
# ============================================================================
def confusion_fig(tn, fp, fn, tp, title):
    z = [[tn, fp], [fn, tp]]
    tot = [[tn + fp] * 2, [fn + tp] * 2]
    text = [[f"{z[i][j]:,}<br>{z[i][j] / max(tot[i][j], 1):.2%}" for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(z=z, x=["Predicted BENIGN", "Predicted ATTACK"],
                               y=["True BENIGN", "True ATTACK"], text=text, texttemplate="%{text}",
                               colorscale=[[0, C["raise"]], [1, "#1D5E86"]], showscale=False,
                               textfont=dict(color=C["text"]),
                               hovertemplate="%{y} → %{x}<br>%{z:,} flows<extra></extra>"))
    fig.update_layout(title=title, yaxis_autorange="reversed")
    return fig


def page_performance():
    st.title("Model performance")
    if not ui.require_artifacts():
        return
    ui.reportability_banner()
    ui.note("Positive class = <b>ATTACK</b>. TP = attack correctly flagged · FN = attack missed · "
            "FP = benign flow wrongly flagged (false alarm) · TN = benign correctly passed.")
    meta = ui.metadata()

    st.markdown("## Comparison on the held-out test set")
    comp = ui.require_results("model_comparison.json", "Model comparison")
    if comp is not None:
        model_comparison(comp, meta)
    final_model_section(meta)
    later_sections()


def model_comparison(comp, meta):
    df = pd.DataFrame(comp)
    num_cols = [c for c in df.columns if c not in ("model", "best_params")]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    show = df[["model", "cv_f1", "accuracy", "precision", "recall", "f1", "specificity", "fpr",
               "fnr", "roc_auc", "pr_auc", "fit_seconds", "predict_ms_per_1000_rows"]].rename(columns={
        "cv_f1": "CV F1 (train)", "fit_seconds": "fit (s)", "predict_ms_per_1000_rows": "predict ms / 1k"})
    for c in show.columns[1:]:
        nd = 1 if c in ("fit (s)",) else 3 if c == "predict ms / 1k" else 4
        show[c] = show[c].map(lambda v, nd=nd: "n/a" if pd.isna(v) else f"{v:.{nd}f}")
    st.dataframe(show, hide_index=True, width="stretch")
    st.caption("Selection rule: " + meta["selection"]["rule"])

    metric = st.segmented_control("Metric", ["f1", "recall", "precision", "specificity", "fpr"],
                                  default="f1")
    if metric:
        fig = px.bar(df, x="model", y=metric, text_auto=".4f", labels={"model": "", metric: metric})
        fig.update_traces(marker_color=[C["info"] if m == meta["final_model"] else "#3A5874"
                                        for m in df["model"]],
                          hovertemplate="<b>%{x}</b><br>" + metric + " %{y:.4f}<extra></extra>")
        st.caption(f"Highlighted: the final model ({meta['final_model']}).")
        plot(fig, height=340)


def final_model_section(meta):
    st.markdown(f"## Final model: {meta['final_model']}")
    m = meta["test_metrics"]
    a, b = st.columns([1, 1.2])
    with a:
        plot(confusion_fig(m["tn"], m["fp"], m["fn"], m["tp"], "Confusion matrix (test)"), height=380)
    with b:
        t1, t2, t3 = st.tabs(["ROC", "Precision–recall", "Threshold"])
        with t1:
            ui.figure("roc_curves.png", "ROC curves, all models")
        with t2:
            ui.figure("pr_curves.png", "Precision–recall curves, all models")
        with t3:
            ui.figure("threshold_sweep.png", "How precision, recall and FPR move with the threshold "
                                             "(analysis only; the threshold was fixed at 0.5)")
    st.markdown(f"Best hyperparameters (chosen by cross-validation): `{meta['best_params']}`")


def later_sections():
    err = ui.results("error_analysis.json")
    if err:
        st.markdown("## Error analysis")
        st.markdown(f"The model missed **{err['false_negatives']:,}** attack flows and raised "
                    f"**{err['false_positives']:,}** false alarms on the test set.")
        lab = pd.DataFrame(err["detection_by_label"])
        lab["role"] = np.where(lab["family"] == "BENIGN", "false-alarm rate", "detection rate")
        st.dataframe(lab.rename(columns={"family": "label", "flagged_rate": "share flagged ATTACK"})
                     .style.format({"share flagged ATTACK": "{:.4f}"}), hide_index=True, width="stretch")
        med = pd.DataFrame(err["median_top_features"])
        if not med.empty:
            st.markdown("**Median values of the most important features, by outcome**")
            st.dataframe(med.style.format(precision=2, na_rep="n/a"), width="stretch")
            st.caption("Compare the false-negative column with the true-positive column: missed "
                       "attacks usually look more like benign traffic on these features.")

    multi = ui.results("multiclass_results.json")
    if multi:
        st.markdown("## Attack-family classification")
        rep = multi["test_report"]
        rows = [{"family": k, "precision": rep[k]["precision"], "recall": rep[k]["recall"],
                 "f1": rep[k]["f1-score"], "test flows": int(rep[k]["support"])} for k in multi["labels"]]
        x, y = st.columns([1, 1.1])
        with x:
            st.dataframe(pd.DataFrame(rows).style.format(precision=4), hide_index=True, width="stretch")
            st.caption(f"Macro-F1 {rep['macro avg']['f1-score']:.4f} · weighted-F1 "
                       f"{rep['weighted avg']['f1-score']:.4f}. Imbalance strategy chosen by CV: "
                       f"{multi['chosen_strategy']}. {multi['rare_classes_note']}")
            st.dataframe(pd.DataFrame(multi["imbalance_comparison"]).style.format(precision=4),
                         hide_index=True, width="stretch")
        with y:
            ui.figure("cm_multiclass_normalised.png", "Row-normalised confusion matrix")

    hold = ui.results("holdout_day_results.json")
    if hold:
        st.markdown("## Held-out day: a harder generalisation test")
        hm = hold["metrics"]
        st.markdown(f"Trained only on Monday–Thursday, tested on all **{hold['test_rows']:,}** Friday "
                    f"flows. Attack types never seen in training: "
                    f"**{', '.join(hold['test_labels_unseen_in_training']) or 'none'}**.")
        c = st.columns(4)
        c[0].metric("Recall", ui.pct(hm["recall"]))
        c[1].metric("Precision", ui.pct(hm["precision"]))
        c[2].metric("FPR", ui.pct(hm["fpr"], 3))
        c[3].metric("F1", ui.num(hm["f1"]))
        ui.figure("holdout_day_detection.png", "Share of Friday flows flagged, per label")


# ============================================================================
# D. Intrusion detection
# ============================================================================
def verdict(out: pd.DataFrame):
    n, k = len(out), int((out["prediction"] == config.POSITIVE_LABEL).sum())
    cls = "ng-attack" if k else "ng-ok"
    st.markdown(f"""<div class="ng-verdict">
<div class="big"><span class="{cls}">{k:,}</span> of {n:,} flow{'s' if n != 1 else ''} flagged as attack</div>
<div class="sub">{(k / n if n else 0):.2%} of the records you supplied. Model estimate, not a
security determination.</div>
<div class="ng-bar" role="img" aria-label="{k:,} flagged, {n - k:,} passed"><span style="flex:{k} 0 0;
background:{C['attack']}"></span><span style="flex:{n - k} 0 0;background:{C['ok']}"></span></div>
</div>""", unsafe_allow_html=True)
    if n > 1:
        fig = px.histogram(out, x="p_attack", nbins=40, color="prediction",
                           color_discrete_map={"ATTACK": C["attack"], "BENIGN": C["ok"]},
                           labels={"p_attack": "P(attack)", "count": "flows"})
        fig.update_traces(hovertemplate="P(attack) %{x}<br>%{y:,} flows<extra></extra>")
        fig.update_layout(bargap=0.05, legend_title_text="", height=230)
        plot(fig)


def results_table(out: pd.DataFrame, X: pd.DataFrame, source: str):
    """Filterable predictions; selecting a row opens its feature values next to training ranges."""
    view = st.segmented_control("Show", ["All flows", "Flagged ATTACK", "Passed BENIGN"],
                                default="All flows", key=f"view_{source}")
    shown = out
    if view == "Flagged ATTACK":
        shown = out[out["prediction"] == config.POSITIVE_LABEL]
    elif view == "Passed BENIGN":
        shown = out[out["prediction"] != config.POSITIVE_LABEL]
    if shown.empty:
        ui.empty_state("No flows in this view", "None of the supplied records fall in this group. "
                       "Choose another filter above.")
        return
    shown = shown.head(1000)
    cfg = {"p_attack": st.column_config.NumberColumn("P(attack)", format="%.4f"),
           "family_confidence": st.column_config.ProgressColumn("family confidence", min_value=0.0,
                                                                max_value=1.0, format="%.3f")}
    if len(out) == 1:
        st.dataframe(shown, width="stretch", column_config=cfg)
        return
    event = st.dataframe(shown, width="stretch", column_config=cfg, on_select="rerun",
                         selection_mode="single-row", key=f"tbl_{source}")
    st.caption(("Showing the first 1,000 rows of this view; the download contains all of them. "
                if len(shown) == 1000 else "") + "Select a row to inspect its feature values.")
    rows = event.selection.rows if event else []
    if rows:
        idx = shown.index[rows[0]]
        r = out.loc[idx]
        stats = ui.schema()["training_feature_stats"]
        detail = pd.DataFrame({"value": X.loc[idx],
                               "training median": [stats[f]["median"] for f in X.columns],
                               "training 1st–99th pct": [f"{stats[f]['p01']:.4g} to {stats[f]['p99']:.4g}"
                                                         for f in X.columns]})
        detail["outside 1–99 pct"] = [not (stats[f]["p01"] <= v <= stats[f]["p99"]) if pd.notna(v)
                                      else None for f, v in X.loc[idx].items()]
        with st.container(border=True):
            st.markdown(f"### Flow {idx}: {r['prediction']}, P(attack) {r['p_attack']:.4f}")
            st.dataframe(detail, width="stretch", column_config={
                "outside 1–99 pct": st.column_config.CheckboxColumn("outside 1–99 pct")})


def run_prediction(df_in: pd.DataFrame, source: str):
    sch, meta = ui.schema(), ui.metadata()
    v = validate_frame(df_in, sch)
    for e in v.errors:
        st.error(e)
    if not v.ok:
        return
    if v.warnings:
        st.info("**Input checks passed.**\n\n" + "\n".join(f"- {w}" for w in v.warnings))
    with st.spinner("Classifying flows…"):
        out = predict(ui.model(), v.X, threshold=meta["decision_threshold"])
    chk = realism_check(v.X, sch)
    summ = realism_summary(chk)
    out = out.join(chk[["input_check"]])
    if summ["warn"]:
        feats = ", ".join(f"{f} ({n})" for f, n in summ["top_features"].items())
        st.warning(
            f"**These records do not look like the traffic the model was trained on.** "
            f"{summ['out_of_range']:,} of {summ['n']:,} have a value outside anything seen in "
            f"training, and {summ['unusual']:,} more have several features in the extreme 1 % "
            f"tails. Most affected: {feats}.\n\n"
            "Predictions on such rows are not reliable. For a meaningful test, use real "
            "CICFlowMeter output (for example the demo sample, which holds real held-out flows "
            "the model never saw). Hand-made or randomly generated rows rarely match how real "
            "flow features relate to each other; for instance, CICFlowMeter only ever writes 0 "
            "or 1 in PSH Flag Count. See the `input_check` column below.")
    fam_model = ui.model("multiclass")
    if fam_model is not None and st.toggle("Also estimate the attack family", value=True,
                                           key=f"fam_{source}"):
        out = out.join(predict_family(fam_model, v.X))
        st.caption("The attack-family estimate comes from a separate model and can disagree "
                   "with the ATTACK/BENIGN verdict.")
    if v.reference_labels is not None:
        out["label_in_file"] = v.reference_labels.values
        ref = (v.reference_labels != "BENIGN").astype(int).values
        agree = float(((out["prediction"] == "ATTACK").astype(int).values == ref).mean())
        st.caption(f"Agreement with the Label column found in the file: {agree:.2%} "
                   "(the label was not used to make the predictions).")
    verdict(out)
    results_table(out, v.X, source)
    full = pd.concat([out.reset_index(drop=True), v.X.reset_index(drop=True)], axis=1)
    buf = io.StringIO()
    full.to_csv(buf, index=False)
    st.download_button("Download predictions (CSV)", buf.getvalue(), "netguard_predictions.csv",
                       "text/csv", key=f"dl_{source}")
    return v, out


def page_detect():
    st.title("Intrusion detection")
    st.markdown('<p class="ng-lede">Classify flow records offline with the trained model.</p>',
                unsafe_allow_html=True)
    if not ui.require_artifacts():
        return
    ui.reportability_banner()
    ui.note("Predictions are model estimates and can be wrong. This tool analyses records you "
            "provide; it does not monitor a live network.", "warn")
    sch = ui.schema()
    t_upload, t_demo, t_form = st.tabs(["Upload a CSV", "Try the demo sample", "Enter one flow"])

    with t_upload:
        st.markdown(f"The file needs these **{len(sch['features'])}** numeric columns (CICIDS2017 "
                    "names; leading spaces are fine). Other columns are ignored, and a `Label` "
                    "column, if present, is never used for prediction.")
        with st.expander("Required columns"):
            st.code(", ".join(sch["features"]), language=None)
        st.download_button("Download an empty template", ",".join(sch["features"]) + "\n",
                           "netguard_template.csv", "text/csv")
        up = st.file_uploader("CSV file", type=["csv"],
                              help=f"Up to {config.MAX_UPLOAD_ROWS:,} rows / {config.MAX_UPLOAD_MB} MB")
        if up is not None:
            df, err = read_uploaded_csv(up)
            if err:
                st.error(err)
            else:
                run_prediction(df, "upload")

    with t_demo:
        if config.DEMO_SAMPLE_PATH.exists():
            demo = pd.read_csv(config.DEMO_SAMPLE_PATH)
            st.markdown(f"{len(demo):,} real flows drawn from the **held-out test set** (never used "
                        "for training or tuning), with their true labels for comparison.")
            if st.button("Classify the demo sample", type="primary"):
                st.session_state["demo_run"] = True
            if st.session_state.get("demo_run"):
                run_prediction(demo, "demo")
        else:
            st.caption("The demo sample is created by the training script.")

    with t_form:
        st.markdown("Values start at the training-set median for each feature. Change any of them "
                    "and classify.")
        stats = sch["training_feature_stats"]
        ex = ui.results("explainability.json") or {}
        ranked = [r["feature"] for r in ex.get("permutation_importance", [])
                  if r["feature"] in sch["features"]]
        ranked += [f for f in sch["features"] if f not in ranked]
        key_feats, other_feats = ranked[:12], ranked[12:]

        def inputs(feats, prefix):
            cols = st.columns(3)
            for i, f in enumerate(feats):
                s = stats[f]
                vals[f] = cols[i % 3].number_input(
                    f, value=float(s["median"]), format="%.4f", key=f"{prefix}_{f}",
                    help=f"Training range {s['min']:.4g} to {s['max']:.4g} "
                         f"(1st–99th percentile {s['p01']:.4g} to {s['p99']:.4g})")

        vals = {}
        with st.form("single"):
            st.markdown("**Most influential features** (by permutation importance)")
            inputs(key_feats, "k")
            if other_feats:
                with st.expander(f"Other {len(other_feats)} features"):
                    inputs(other_feats, "o")
            submitted = st.form_submit_button("Classify this flow", type="primary")
        if submitted:
            rec = pd.DataFrame([vals])
            res = run_prediction(rec, "form")   # includes the input-realism warning
            if res:
                explain_one(rec)


def explain_one(rec: pd.DataFrame):
    from src.explainability import explain_rows
    contrib = explain_rows(ui.model(), rec)
    st.markdown("### Why this prediction")
    if contrib is None:
        st.caption("A per-flow explanation is not available for this model type; see the "
                   "Explainability page for global importance instead.")
        return
    s = contrib.iloc[0].sort_values(key=np.abs, ascending=False).head(10)[::-1]
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h",
                           marker_color=[C["attack"] if v > 0 else C["ok"] for v in s.values],
                           hovertemplate="<b>%{y}</b><br>value %{customdata:.4g}<br>"
                                         "contribution %{x:+.4f}<extra></extra>",
                           customdata=rec.iloc[0].reindex(s.index).values))
    method = contrib.attrs.get("method", "SHAP")
    fig.update_layout(title=f"Top contributions ({method}): right pushes toward ATTACK, left toward BENIGN",
                      height=360, xaxis_title="Contribution to the model's raw output")
    plot(fig)
    st.caption("A contribution says how the model used a value for this flow. It is not evidence "
               "that the feature causes an attack, nor proof that the prediction is right.")


# ============================================================================
# E. Explainability
# ============================================================================
def page_explain():
    st.title("Explainability")
    if not ui.require_artifacts():
        return
    ui.reportability_banner()
    ex = ui.results("explainability.json")
    meta = ui.metadata()
    st.markdown(f"Which inputs does **{meta['final_model']}** rely on? Three complementary views:")
    t1, t2, t3 = st.tabs(["Permutation importance", "Built-in importance", "SHAP"])
    with t1:
        st.markdown("Shuffle one column of the test data and measure how much F1 drops. "
                    "Model-agnostic and measured on held-out data.")
        if ex and ex.get("permutation_importance"):
            pi = pd.DataFrame(ex["permutation_importance"]).head(20)[::-1]
            fig = go.Figure(go.Bar(x=pi["importance_mean"], y=pi["feature"], orientation="h",
                                   error_x=dict(array=pi["importance_std"], color=C["muted"]),
                                   marker_color=C["info"],
                                   hovertemplate="<b>%{y}</b><br>F1 drop %{x:.4f}<extra></extra>"))
            fig.update_layout(height=520, xaxis_title="Mean drop in test F1")
            plot(fig)
            st.caption("Correlated features share credit: shuffling one while its near-copy remains "
                       "can make both look unimportant.")
    with t2:
        st.markdown("For trees: total impurity reduction from splits on each feature (training "
                    "data). For logistic regression: coefficient size on standardised inputs.")
        ui.figure("importance_builtin.png", "Built-in importance of the final model")
    with t3:
        if ex and ex.get("shap_available_for_final_model"):
            st.markdown("Average size of each feature's SHAP contribution over a sample of test flows. "
                        "Per-flow SHAP explanations appear on the Intrusion detection page.")
            ui.figure("importance_shap.png", "Mean |SHAP| contribution")
        else:
            st.caption("SHAP was not computed: it is not installed or does not support the final "
                       "model type.")
    ui.note("Importance shows what the model uses, not what causes an attack. A feature can rank "
            "high because of how the 2017 testbed generated traffic.")


# ============================================================================
# F. Methodology and G. About
# ============================================================================
def page_method():
    st.title("Methodology")
    meta = ui.metadata() or {}
    fs = ui.results("feature_selection.json")
    st.markdown("""
## Pipeline
1. **Load** the eight CICIDS2017 day files; strip header spaces; drop the duplicated
   `Fwd Header Length` column; normalise label spellings.
2. **Clean**: ±infinity → missing; build the binary target (ATTACK = any non-BENIGN label) and
   attack-family target.
3. **De-duplicate before splitting**: exact duplicates collapsed to one row; rows whose identical
   features carry different labels removed.
4. **Split** 80 / 20, stratified on the fine-grained label, seed 42. The test set is untouched
   until final evaluation.
5. **Learned steps inside a scikit-learn Pipeline**, fitted on training data only: median
   imputation → constant-feature removal → (logistic regression only) signed log transform and
   standardisation → model.
6. **Feature selection** by cross-validation on a training subsample.
7. **Tuning** with randomised search, 3-fold stratified CV, scored by F1 of the ATTACK class.
8. **Model choice** by CV score only; test metrics are reported afterwards.
9. **Extra checks**: per-family detection, error analysis, threshold analysis, attack-family model
   with in-fold imbalance comparison, and a held-out-day test.
""")
    if fs:
        st.markdown("## Feature-set comparison (training data, CV)")
        st.dataframe(pd.DataFrame(fs["comparison"]).style.format(precision=4), hide_index=True,
                     width="stretch")
        st.caption(f"Rule: {fs['rule']}. Chosen: {fs['chosen']}.")
    st.markdown("""
## Metrics
| Metric | Formula | Meaning here |
|---|---|---|
| Precision | TP / (TP + FP) | Share of alerts that were real attacks |
| Recall | TP / (TP + FN) | Share of attacks that were caught |
| F1 | 2·P·R / (P + R) | Balance of precision and recall |
| Specificity | TN / (TN + FP) | Share of benign traffic passed |
| FPR | FP / (FP + TN) | False-alarm rate |
| FNR | FN / (FN + TP) | Missed-attack rate |
| ROC-AUC | area under TPR vs FPR | Ranking quality across all thresholds |
| PR-AUC | average precision | Ranking quality focused on the attack class |

## Class imbalance
Class weights are compared with SMOTE applied **inside each training fold only**, never before the
split. The binary task is moderately imbalanced; the attack-family task is extremely imbalanced.

## Reproducibility
Fixed seed (42), pinned requirements, and one command (`python -m src.train`) regenerates every
number and figure on this site.
""")
    if meta.get("environment"):
        st.caption("Trained with " + ", ".join(f"{k} {v}" for k, v in meta["environment"].items()))


def page_about():
    st.title("About")
    st.markdown("""
**NetGuard AI** — *Machine Learning-Based Network Intrusion Detection and Attack Classification
System*, a college project in cybersecurity and machine learning.

## Objectives
Understand a real intrusion dataset; build a leak-free, reproducible pipeline; compare classical
classifiers with metrics that suit security work; study false positives and false negatives; and
present the results in an explainable, usable interface.

## Intended users
Students, instructors and analysts who want to explore how flow-based ML detection behaves. It is
not meant for protecting a production network.

## SDG alignment (potential contribution)
- **SDG 9 — Industry, Innovation and Infrastructure:** explores ML support for detecting malicious
  traffic, a building block of more resilient digital infrastructure.
- **SDG 16 — Peace, Justice and Strong Institutions:** supports cybersecurity awareness and safer
  digital environments.

A student prototype does not measurably achieve these goals; it contributes to learning and
research toward them.

## Scope and limitations
- Offline analysis of CICIDS2017-format flow records; no packet capture or live monitoring.
- Inputs must be flow features computed the way CICFlowMeter computes them.
- Dataset from 2017 lab traffic; known labelling issues; attack types tied to specific days.
- Random-split scores overstate performance on unseen attacks (see held-out-day results).

## Future work
Live flow extraction (CICFlowMeter) feeding the model, evaluation on newer datasets, drift
monitoring, cost-sensitive thresholds chosen with analysts, and anomaly detection for unseen
attack types.
""")
    ui.note(DISCLAIMER)


pages = [
    st.Page(page_home, title="Dashboard", icon=":material/shield:", default=True),
    st.Page(page_dataset, title="Dataset explorer", icon=":material/table_chart:", url_path="dataset"),
    st.Page(page_performance, title="Model performance", icon=":material/monitoring:",
            url_path="performance"),
    st.Page(page_detect, title="Intrusion detection", icon=":material/radar:", url_path="detect"),
    st.Page(page_explain, title="Explainability", icon=":material/insights:", url_path="explain"),
    st.Page(page_method, title="Methodology", icon=":material/schema:", url_path="methodology"),
    st.Page(page_about, title="About & limitations", icon=":material/info:", url_path="about"),
]
ui.sidebar_status()
st.navigation(pages).run()
