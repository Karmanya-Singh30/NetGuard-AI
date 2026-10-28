"""Generate the two project notebooks (run once; they call the src/ package so
notebook and script results are always identical).  python notebooks/build_notebooks.py"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
SETUP = """import sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
import pandas as pd, numpy as np
from IPython.display import Image, display, Markdown
from src import config
pd.set_option("display.max_columns", 50)"""


def md(t):
    return nbf.v4.new_markdown_cell(t)


def code(t):
    return nbf.v4.new_code_cell(t)


def nb01():
    n = nbf.v4.new_notebook()
    n.cells = [
        md("# 01 — Dataset exploration (CIC-IDS2017)\n\nThis notebook profiles the **real** data "
           "with the same functions the training script uses (`src/eda.py`). Run it after placing "
           "the CSVs in `data/raw/`. Interpretations go under each figure **after** you have "
           "looked at the actual output."),
        code(SETUP),
        md("## Load"),
        code("from src.data_loading import load_cicids\nraw, load_report = load_cicids()\n"
             "print(raw.shape)\nload_report.rows_per_file"),
        md("## Profile: size, types, missing/infinite values, duplicates, constant columns"),
        code("from src.eda import profile_dataset\nprofile = profile_dataset(raw, load_report)\n"
             "{k: profile[k] for k in ['n_rows','n_numeric_features','memory_mb_float32',"
             "'duplicate_rows_including_label','duplicate_rows_features_only',"
             "'constant_columns','near_constant_columns_top_value_ge_99.9pct']}"),
        code("pd.DataFrame({'missing': pd.Series(profile['missing_values']),"
             " 'infinite': pd.Series(profile['infinite_values'])}).fillna(0).astype(int)"),
        code("pd.Series(profile['class_distribution']).to_frame('flows')"
             ".assign(pct=lambda d: (100*d.flows/d.flows.sum()).round(4))"),
        code("pd.DataFrame(profile['labels_per_file']).fillna(0).astype(int)"),
        md("**Interpretation (write after viewing):** what is the class imbalance? Which columns "
           "contain inf/NaN and why (zero-duration flows)? How many exact duplicates?"),
        md("## Clean (inf → NaN, targets, de-duplication)"),
        code("from src.data_preprocessing import clean_dataset, feature_columns\n"
             "df, clean_report = clean_dataset(raw)\nclean_report['deduplication']['rows_after'], "
             "clean_report['deduplication']['exact_duplicates_removed'], "
             "clean_report['deduplication']['conflicting_label_rows_removed']"),
        code("pd.DataFrame({'before': clean_report['deduplication']['rows_before_by_label'],"
             " 'removed_exact': clean_report['deduplication']['exact_duplicates_removed_by_label'],"
             " 'after': clean_report['deduplication']['rows_after_by_label']}).fillna(0).astype(int)"),
        md("## Figures"),
        code("from src.eda import make_eda_figures\nconfig.ensure_dirs()\n"
             "feats = [c for c in feature_columns(df) if df[c].std() > 0][:22]\n"
             "paths = make_eda_figures(profile, df, clean_report['deduplication'], feats)\n"
             "for p in paths: display(Markdown(f'### {p}')); display(Image(str(ROOT/p)))"),
        md("**Interpretation (write after viewing each figure):** what does it suggest, and what "
           "does it *not* prove? (e.g. separable distributions on one feature do not mean the "
           "feature causes the attack; lab-specific patterns may not generalise.)"),
    ]
    return n


def nb02():
    n = nbf.v4.new_notebook()
    n.cells = [
        md("# 02 — Preprocessing and modelling\n\nThe full experiment is `python -m src.train` "
           "(it writes every result file). This notebook **reads those results** and shows the "
           "key steps, so the notebook can never disagree with the scripts. Run the training "
           "script first."),
        code(SETUP),
        code("R = config.RESULTS_DIR\nload = lambda n: json.loads((R/n).read_text())\n"
             "meta = json.loads(config.METADATA_PATH.read_text())\n"
             "print('reportable:', meta.get('reportable'), '| final model:', meta['final_model'])"),
        md("## Split"),
        code("split = load('split_info.json')\n{k: split[k] for k in ['train_rows','test_rows',"
             "'train_attack_share','test_attack_share','rare_classes_lt_threshold']}"),
        md("## Feature-set comparison (cross-validation on the training subsample)"),
        code("fs = load('feature_selection.json')\nprint(fs['rule'], '->', fs['chosen'])\n"
             "pd.DataFrame(fs['comparison'])"),
        md("## Model comparison (test set; positive class = ATTACK)"),
        code("comp = pd.read_csv(R/'model_comparison.csv')\ncomp"),
        code("display(Image(str(config.FIGURES_DIR/'model_comparison.png')))\n"
             "display(Image(str(config.FIGURES_DIR/'roc_curves.png')))\n"
             "display(Image(str(config.FIGURES_DIR/'pr_curves.png')))"),
        md("## Overfitting check: training-fold vs validation-fold F1"),
        code("comp[['model','cv_train_f1','cv_f1','overfit_gap_f1']]"),
        md("## Error analysis"),
        code("err = load('error_analysis.json')\nprint('FN', err['false_negatives'], '| FP', "
             "err['false_positives'])\npd.DataFrame(err['detection_by_label'])"),
        code("display(Image(str(config.FIGURES_DIR/'detection_by_family.png')))\n"
             "display(Image(str(config.FIGURES_DIR/'threshold_sweep.png')))"),
        md("## Attack-family model and imbalance strategies"),
        code("mc = load('multiclass_results.json')\ndisplay(pd.DataFrame(mc['imbalance_comparison']))\n"
             "pd.DataFrame({k: mc['test_report'][k] for k in mc['labels']}).T"),
        md("## Held-out day"),
        code("ho = load('holdout_day_results.json')\nprint(ho['test_labels_unseen_in_training'])\n"
             "pd.Series(ho['metrics'])"),
        md("## Explainability"),
        code("ex = load('explainability.json')\npd.DataFrame(ex['permutation_importance']).head(15)"),
        md("## Reload the saved pipeline and predict on the demo sample"),
        code("from src.predict import load_pipeline, load_schema, validate_frame, predict\n"
             "demo = pd.read_csv(config.DEMO_SAMPLE_PATH)\nv = validate_frame(demo, load_schema())\n"
             "out = predict(load_pipeline(), v.X)\npd.crosstab(v.reference_labels, out['prediction'])"),
    ]
    return n


if __name__ == "__main__":
    nbf.write(nb01(), HERE / "01_dataset_exploration.ipynb")
    nbf.write(nb02(), HERE / "02_preprocessing_and_modeling.ipynb")
    print("written")
