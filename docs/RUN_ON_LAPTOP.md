# Running NetGuard AI on your Windows laptop

The trained models and all results are already in the project folder, so **you can open the
website right away** (steps 1, 3 and 6). Training (step 5) is only needed to reproduce the
results yourself. Every command runs in **PowerShell**, from inside the project folder.

## 1. Open the project folder

```powershell
cd $env:USERPROFILE\network-intrusion-detection
```

## 2. Put the dataset in place

You already have the eight CSV files in `data\`. The training script finds them there
automatically. Alternatively, put `MachineLearningCSV.zip` in `data\raw\`; the zip is read
directly, so there's no need to unzip it.

```powershell
mkdir data\raw -Force
Move-Item data\*.pcap_ISCX.csv data\raw\      # optional tidy-up; data\ also works
```

## 3. Create a virtual environment and install the libraries (once)

Python **3.11 or newer** is required (`python --version` to check).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell refuses to run `Activate.ps1`, first run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` and answer `Y`.

## 4. Check that the code works (≈1 minute)

```powershell
python -m pytest -q
```

All tests should pass. They use a small synthetic sample, so this works even before training.

## 5. Train on the real data

```powershell
python -m src.train
```

The script prints its progress as **Step 1/9 … Step 9/9**. Expect roughly **10–30 minutes**,
depending on the laptop. Keep it plugged in and don't let it sleep.

**Memory is handled automatically.** On a laptop with less than 12 GB of RAM, the final models
are fitted on a 1,000,000-row stratified subsample of the training data, which keeps every row of
the rare attack classes. With 12 GB or more, all training rows are used. The log line
`Final-model fitting rows: …` and `model_metadata.json` record which applied. To override:

```powershell
python -m src.train --max-train-rows all       # force all rows (needs ~6 GB free RAM)
python -m src.train --max-train-rows 500000    # smaller/faster
```

(Measured: 9.7 minutes on a Lenovo LOQ — Intel Core i5, 24 GB RAM — with all rows.)

To skip the slower extra experiments: `--skip-holdout`, `--skip-multiclass`. The report is better
with them included.

## 6. Open the website

```powershell
streamlit run app/app.py
```

Your browser opens at http://localhost:8501. Stop the website with `Ctrl+C`.

## 7. What to send back for the report

After training finishes, everything the report needs is in two small folders:

- `reports\results\`: JSON/CSV/Markdown result files, including `results_summary.md`
- `reports\figures\`: PNG charts
- `models\model_metadata.json` and `models\feature_schema.json`

These are small, so they transfer easily. The large `*.joblib` model files stay on the laptop.
