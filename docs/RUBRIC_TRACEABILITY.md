# Rubric traceability (AUCML ESE, 30 marks)

All items below are implemented, and the results come from the real training run on the full
CIC-IDS2017 data (2,830,743 flows; 9.7 min on a Lenovo LOQ laptop).

| Component (marks) | Evidence | Where |
|---|---|---|
| Problem refinement & use-case relevance (3) | Problem, objectives, scope, users, real-world relevance, SDGs | Report §I; README; app *About*; METHODOLOGY §1 |
| Dataset understanding, preprocessing, feature handling (4) | Measured profile; 7 EDA figures with interpretations; cleaning (inf, 331,200 duplicates, 1,438 conflicts); leakage controls; CV feature-set comparison | `src/eda.py`, `data_loading.py`, `data_preprocessing.py`, `feature_engineering.py`; `reports/results/dataset_profile.json`, `cleaning_report.json`, `feature_selection.json`; `reports/figures/eda_*.png`; Report §III.B–C; `notebooks/01_*` |
| Methodology & ML pipeline design (5) | Pipeline select→impute→drop const→[scale]→model; 5 models; CV tuning; CV-only selection rule; in-fold SMOTE comparison; held-out day | `src/train.py`; METHODOLOGY §4–7; Report §III.D, Table 3 |
| Implementation & execution (5) | Working code; trained models; 7-page website; 108 tests passing; one-command reproducibility; verified determinism | `src/`, `app/`, `tests/`, `models/`, `scripts/regenerate_figures.py`; Report §IV; `docs/screenshots/` |
| Results, metrics, comparative analysis (5) | 5-model comparison; confusion matrices; ROC/PR; threshold sweep; per-family detection; error analysis; family model; held-out day | `reports/results/*`, `results_summary.md`; `reports/figures/*`; Report §V, Tables 4–6 |
| Final report & documentation (3) | Course-template report (12 pp.), full Markdown report, README, methodology, dataset notes, run guide | `reports/NetGuard_AI_Report.docx`, `reports/project_report.md`, `README.md`, `docs/` |
| Communication & presentation (2) | 14-slide plan with visuals, numbers and speaker notes | `docs/PRESENTATION_PLAN.md` |
| Viva & defence (3) | 58 Q&A with real numbers, revision sheet, demo script | `docs/VIVA_QUESTIONS.md` |

## Deliverables checklist

- [x] Working project directory and structure
- [x] Dataset acquisition instructions and documentation
- [x] EDA notebook and saved figures (real data)
- [x] Reproducible preprocessing pipeline
- [x] Feature selection (CV-based)
- [x] Baseline model (Dummy)
- [x] Multiple trained and compared models, with tuning
- [x] Final saved inference pipeline (`models/final_pipeline.joblib`)
- [x] Actual evaluation metrics and result files
- [x] Confusion matrices and ROC/PR curves
- [x] Error analysis
- [x] Explainability output (permutation, built-in, SHAP)
- [x] Functional Streamlit website (verified on the real model)
- [x] Input validation and CSV prediction
- [x] Tests (108 passing, 2 expected failures)
- [x] requirements.txt (pinned)
- [x] Complete README with real results and screenshots
- [x] Complete academic report (course template + Markdown)
- [x] Presentation outline and speaker notes
- [x] ≥ 40 viva questions and answers, revision sheet, demo script
- [x] Rubric mapping and final audit
- [x] Known limitations and future scope

## Still needed from you
- Fill in the report's author block (mobile number, team members), guide name and GitHub URL.
- Build the actual slides from `docs/PRESENTATION_PLAN.md` (or ask for a deck).
- Optional: re-run `python -m src.train` on your laptop to reproduce the results yourself.
