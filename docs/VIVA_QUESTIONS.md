# NetGuard AI — viva preparation

Answers describe **what this project's code actually does**. Every number comes from the real
training run on the full CIC-IDS2017 data (`reports/results/results_summary.md` and the JSON
files next to it). If you retrain, re-check them there.

---

## A. Problem formulation and dataset

**1. What problem does your project solve?**
It classifies network flow records as benign or malicious, and estimates the attack family, using
supervised ML trained on CIC-IDS2017. It supports an analyst's triage. It is not a production IDS
and does not monitor live traffic.

**2. Why is this framed as classification and not anomaly detection?**
CIC-IDS2017 provides labels for every flow, so supervised learning can use them. Anomaly detection
would be the choice for unseen attacks without labels. The held-out-day experiment shows the
limits of the supervised approach, and anomaly detection is listed as future work.

**3. What is a "flow"?**
A bidirectional sequence of packets between two endpoints (same 5-tuple) within a timeout,
summarised by CICFlowMeter into statistics: duration, packet/byte counts per direction,
packet-length statistics, inter-arrival times, flag counts, rates, and active/idle times.

**4. Why CIC-IDS2017?**
It's public, widely cited and documented, and has modern attack types (DoS, DDoS, brute force,
web attacks, botnet, port scan, infiltration, Heartbleed) with rich flow features. I also had
access to all eight files. UNSW-NB15 was the alternative.

**5. What are the dataset's weaknesses?**
It's 2017 lab traffic, so real networks will differ. Published work reports labelling and flow
errors (Engelen et al. 2021; Lanvin et al. 2022). Each attack ran on a specific day. There are
many duplicate flows, inf/NaN values from zero-duration flows, and some classes are tiny.

**6. What is your target variable and positive class?**
`Label`. The binary target is ATTACK = any label other than BENIGN, and the **positive class is
ATTACK**. The family target groups the labels into 8 attack families plus BENIGN.

**7. How large is the dataset and how imbalanced is it?**
2,830,743 flows with 78 features (77 numeric after dropping the duplicated column) plus Label.
80.3 % are BENIGN (2,273,097) and 19.7 % ATTACK (557,646). Across the 15 labels it's extremely
imbalanced: BENIGN is ~206,600× larger than Heartbleed (11 flows). SQL injection has 21 flows and
Infiltration 36. After cleaning: 2,498,105 flows, 17.0 % attack.

**8. Did you check for identifier or leakage columns?**
Yes. This release has no IPs, timestamps or flow IDs. `Destination Port` is present. I excluded it
from the inputs because attacks in the testbed target fixed ports, so it's a shortcut that
wouldn't generalise. `source_file` (the day) is bookkeeping only, because it would reveal the
attack type.

## B. Preprocessing and leakage

**9. How did you handle infinite and missing values?**
±inf is replaced with NaN, and NaN is then filled with the **training-set median** by a
`SimpleImputer` inside the pipeline. The median is robust to the heavy-tailed rates. The website
applies exactly the same treatment.

**10. Why did you remove duplicates, and when?**
Exact duplicates were removed **before** the train/test split. Otherwise the same flow could land
in both sets, and the model would be "tested" on rows it memorised, which inflates scores. Rows
with identical features but different labels were also removed, because they're unlearnable
labelling noise. How many: 331,200 exact duplicates (11.7 %; up to ~45 % of SSH-Patator and ~43 % of PortScan
rows) and 1,438 label-conflicting rows.

**11. Isn't removing duplicates changing the data distribution?**
Yes, and that's documented. Some attacks, such as DoS, generate many identical flows. The goal is
an honest test set. Removing duplicates isn't done to raise scores; it usually makes them more
realistic.

**12. What is data leakage? Give three ways you prevented it.**
Leakage is information from the evaluation data influencing training. Prevention:
(a) de-duplicating before the split; (b) fitting imputation, constant-feature removal, scaling
and feature selection on training data only, inside a Pipeline; (c) SMOTE only inside training
folds; (d) choosing the model by CV score, not test score; (e) excluding `source_file` and the
port.

**13. Why stratify the split?**
To keep each class's proportion the same in train and test. I stratified on the *fine* label so
that even tiny classes like Heartbleed appear in both.

**14. Why 80/20 and a fixed seed?**
80/20 leaves plenty of training data and a large test set. The fixed seed (42) makes the split,
subsamples, CV folds and models reproducible.

**15. Why didn't you scale the features for tree models?**
Trees split on thresholds of one feature at a time, so any monotonic rescaling gives the same
splits. Logistic regression is gradient-based and penalises coefficients, so it needs comparable
scales. It gets a signed log transform plus standardisation.

**16. Why a log transform?**
Byte counts, durations and rates span many orders of magnitude. sign(x)·log(1+|x|) compresses the
tails and keeps the sign, since some fields can be −1.

**17. What does a Pipeline guarantee?**
That the same fitted steps are applied in the same order during CV, final training, testing and
on the website. Inside cross-validation every fold refits the whole chain, so there's no leakage
between folds.

## C. Features

**18. How did you select features?**
I compared four sets with 3-fold CV on the training subsample: all features, a correlation filter
(|r|>0.98), the top 30 by mutual information, and the top 30 by RF importance. Rule: the smallest
set within 0.002 F1 of the best. Chosen: the top 30 by RF importance. It had the best CV F1 (0.9954, vs 0.9937 for all 76) and is
also among the smallest sets. Mostly packet-length statistics, TCP window and flag counts, and
inter-arrival times.

**19. What is mutual information?**
How much knowing a feature reduces uncertainty about the class:
I(X;Y)=Σ p(x,y) log[p(x,y)/(p(x)p(y))]. It is zero when they're independent, and it catches
non-linear relationships.

**20. Why remove highly correlated features?**
They carry nearly the same information (e.g. `Total Fwd Packets` and `Subflow Fwd Packets`).
Removing them simplifies the model and the input form, and makes importance easier to interpret.

**21. Which features matter most?**
By permutation importance on test data: PSH Flag Count, Init_Win_bytes_forward, Flow Bytes/s,
Init_Win_bytes_backward, Fwd IAT Min. By built-in impurity importance and SHAP: packet-length
statistics (Packet Length Variance, Bwd Packet Length Std, Average Packet Size). The rankings
differ because impurity importance spreads credit over the many correlated length features,
while permutation importance measures the loss when one column is broken and its near-copies
remain. These are what the model uses, not causes. The TCP initial-window bytes reflect the
testbed machines' OS settings and are a plausible shortcut.

## D. Algorithms

**22. Explain logistic regression.**
z = β₀ + Σβᵢxᵢ; P(attack) = 1/(1+e^(−z)). It's trained by minimising binary cross-entropy with L2
regularisation (strength 1/C). It predicts ATTACK when P ≥ 0.5, i.e. z ≥ 0. The coefficients show
each feature's direction and strength.

**23. Explain a decision tree, entropy and Gini.**
It splits the data recursively on the feature threshold that reduces impurity most. Entropy
H = −Σpᵢlog₂pᵢ; Gini = 1 − Σpᵢ²; information gain = parent impurity minus the weighted child
impurity. Deep trees overfit, so I tuned max_depth and min_samples_leaf.

**24. Explain Random Forest.**
It's an ensemble of trees, each trained on a bootstrap sample and considering a random subset of
features at each split. sklearn averages the trees' predicted probabilities (soft voting). The
randomness decorrelates the trees, so averaging reduces variance.

**25. Explain gradient boosting.**
F_m(x) = F_{m−1}(x) + η·h_m(x): each new small tree fits the gradient of the loss, i.e. the
current errors. The learning rate η shrinks each step. Overfitting is controlled by leaf limits,
L2 regularisation and early stopping. HistGradientBoosting bins features, which makes it fast on
millions of rows.

**26. Why include a Dummy classifier?**
It always predicts BENIGN, so it gets high accuracy with zero recall. It shows why accuracy alone
is misleading. Dummy accuracy: **82.98 %**, with recall 0.

**27. Which model did you choose, and why?**
**Random Forest** (100 trees, min_samples_leaf 2, max_features √p, balanced_subsample). CV F1:
RF 0.9957, HistGB 0.9963, DT 0.9952, LR 0.8545. RF and HistGB were within the 0.001 tie margin,
and RF had the higher CV specificity. On the test set they are practically tied (HistGB 246 vs RF
308 false alarms; RF 199 vs HistGB 206 misses), which is expected for differences this small. Rule: the highest mean CV F1 on
training data; models within 0.001 are tie-broken by CV specificity, then speed. The test set
wasn't used to choose.

**28. Why not deep learning?**
On tabular flow features, tree ensembles are strong, fast, explainable and easy to defend. Deep
learning adds complexity without a clear benefit here, and the brief asked to avoid unnecessary
complexity.

## E. Tuning, validation, overfitting

**29. How did you tune hyperparameters?**
RandomizedSearchCV with 3-fold stratified CV on a stratified training subsample (rare classes
kept in full), scored by F1 of ATTACK. The best settings were then refit on the full training
split.

**30. Why tune on a subsample?**
Compute: there are millions of rows. Every class keeps at least 2,000 rows (or all of them), so
rare attacks aren't lost. The final models use all training rows, unless the machine has less than 12 GB RAM; then
they use a 1M-row stratified subsample that keeps every rare-class row. Which rule applied: all 1,998,484 training rows (`--max-train-rows all`) in the reported run.

**31. Cross-validation versus the test set?**
CV estimates performance *for making choices* using only training data. The test set is touched
once, at the end, to report performance on data never used for any choice.

**32. How did you check for overfitting?**
I recorded training-fold F1 next to validation-fold F1 for each model (`overfit_gap_f1`). The gaps are small: DT 0.0040, RF 0.0027, HistGB 0.0020, LR −0.0002. So
there's no serious overfitting on this split, and the train and test scores are almost equal. A large gap means memorisation. Tree depth and leaf size limits address it.

**33. What is underfitting? Did any model underfit?**
A model too simple to capture the patterns: low training *and* validation scores. Logistic regression underfits: training F1 (0.855) ≈ validation F1 (0.855), both far below the
trees. Its 6.2 % false-alarm rate shows a linear boundary can't separate these classes, even
after log-scaling.

## F. Metrics

**34. Define TP, FP, FN, TN here.**
TP = attack flagged; FN = attack missed; FP = benign flagged (false alarm); TN = benign passed.

**35. Precision vs recall? Which matters more?**
Precision = TP/(TP+FP), the share of alerts that are real. Recall = TP/(TP+FN), the share of
attacks caught. In security a missed attack (FN) is usually costlier, so recall is critical. But
too many false alarms cause alert fatigue, so precision and FPR matter too. F1 balances them.

**36. Specificity, FPR, FNR?**
Specificity = TN/(TN+FP); FPR = FP/(FP+TN) = 1 − specificity (the false-alarm rate);
FNR = FN/(FN+TP) = 1 − recall (the missed-attack rate).

**37. ROC-AUC vs PR-AUC?**
ROC-AUC = the probability that a random attack scores above a random benign flow, across all
thresholds. PR-AUC focuses on the positive class and isn't flattered by many true negatives, so
it's more informative under imbalance.

**38. Why is accuracy insufficient?**
With mostly benign traffic, predicting "benign" always gives high accuracy and catches nothing.
The Dummy model demonstrates this.

**39. What threshold did you use? Would you change it?**
0.5, fixed in advance. The threshold sweep shows that lowering it raises recall and FPR. In
deployment you'd pick it on validation data from the relative cost of misses versus false alarms,
never on the test set.

**40. How did you handle undefined metrics?**
If a denominator is zero (e.g. no ATTACK predictions), the metric is reported as n/a, never as a
fake 0 or 1.

**41. Macro vs weighted average?**
Macro averages per-class scores equally, which exposes weak rare classes. Weighted averages by
support, which is dominated by large classes. For the family model I report both and chose the
imbalance strategy by macro-F1.

## G. Imbalance and SMOTE

**42. How did you handle class imbalance?**
Class weights (`balanced`: w_c = N/(K·N_c)) are part of each model's search space. For the family
model I compared none, class weights and in-fold SMOTE by CV macro-F1. Chosen: in-fold SMOTE (CV macro-F1 0.954 vs 0.940 for none or class weights). The trade-off on
test: Bot recall rose, but Bot precision is only 0.52, because synthetic Bot points overlap
benign traffic and 356 benign flows were labelled Bot.

**43. Explain SMOTE. Why never before the split?**
x_new = x_i + λ(x_j − x_i), λ∈[0,1], where x_j is a same-class nearest neighbour. Before the split,
synthetic points built from test rows would leak into training, and test rows would be near-copies
of training data. I use an imblearn pipeline, so SMOTE is refit inside each training fold only. It only tops
classes below 20,000 rows up to 20,000, because the default (match the majority class) would
create tens of millions of synthetic rows.

## H. Results and errors

**44. What are your final test results?**
Random Forest on 499,621 test flows: recall 0.9977, precision 0.9964, F1 0.9970, FPR 0.00074,
FNR 0.0023, ROC-AUC 0.99996, PR-AUC 0.99985. TP 84,830, FN 199, FP 308, TN 414,284.

**45. Which attacks are missed most?**
By count: DoS Hulk 61, Bot 59, DDoS 20, SSH-Patator 11. By rate, the worst are the rarest
classes: all 4 SQL-injection flows, 5 of 7 Infiltration, 1 of 2 Heartbleed, and 15 % of Bot. Most
misses are borderline (median P(attack) 0.36). Engelen et al. likewise report a Random Forest
F1 of only 0.11 for SQL injection on the original data.

**46. What do your false positives look like?**
They look attack-like on the top features: very small packets (median average packet size 3
bytes vs 78.5 for correctly passed benign flows, packet-length variance 5 vs 945). Short, near-empty
benign flows resemble scans and failed connections. That's also where Engelen et al. found
mislabelled "attempted" flows.

**47. What is the held-out-day experiment and what did it show?**
Train Monday–Thursday, test Friday. Bot, PortScan and DDoS are never seen in training. Recall collapsed to **0.101** (DDoS 17.4 %, PortScan 0.07 %, Bot 0 %) at FPR 0.0003, versus
recall 0.998 on the random split. It measures generalisation to *new* attack types, which random
splits overstate.

**48. Why might your scores be very high, and is that a concern?**
CIC-IDS2017 classes are often separable in a lab setting, and a random split tests known attack
types. High scores don't prove real-world security. That's why I report de-duplication, per-family
errors, the held-out day and the dataset's known errors.

## I. Explainability

**49. Which explanation methods did you use?**
Permutation importance on test data (model-agnostic), built-in importance (impurity/coefficients),
and per-flow explanations: SHAP for trees, exact coefficient×value for logistic regression.

**50. Does feature importance prove causation?**
No. It shows what the model relies on. A feature can be important because of how the testbed
generated traffic. An explanation also doesn't prove that a prediction is correct.

## J. Website and engineering

**51. Describe the website architecture.**
Streamlit with 7 pages. It loads the saved pipeline, schema, metadata and result files (cached).
UI code (`app/`) is separate from inference (`src/predict.py`) and training (`src/train.py`). It
never trains and has no hard-coded numbers.

**52. How does the prediction pipeline handle a CSV upload?**
The file is parsed as CSV only (never unpickled). Then: headers are stripped; required columns are
checked; extra columns are ignored and reported; a Label column is reported but never used;
numeric types are validated, with errors naming the column; Infinity → NaN; columns are reordered
by name; the saved pipeline is applied; the results can be downloaded. There are row and size
limits.

**53. Why not ask the user for a label?**
The label is what we're predicting. If a file happens to contain one, it's shown only for
comparison.

**54. How is the project reproducible?**
One command (`python -m src.train`), seed 42, pinned requirements, recorded versions in
`model_metadata.json`, and tests.

**55. What do your tests cover?**
108 tests covering loading, label/header normalisation, inf handling, de-duplication, no shared
rows across the split, scaler fitted on train only, SMOTE in-fold only, save/reload, prediction
shape/probabilities, column-order invariance, metric mapping, CSV validation, the input-realism check, and an end-to-end
train + website run on a synthetic fixture.

**55b. Your model gave all-BENIGN on a CSV I made up. Is it broken?**
No. On the 499,621 real held-out flows the website reproduces the reported numbers exactly
(84,830 TP, 308 FP, 199 FN), and on a random 1,000 of them it gets every one right. Made-up rows
usually break how real flow features fit together; for example CICFlowMeter only writes 0 or 1
in `PSH Flag Count`, the model's most important feature. The model has never seen such inputs,
so its output means nothing. The website therefore runs an input-realism check. It counts values
outside anything seen in training, plus features in the extreme 1 % tails, and warns when a file
does not resemble real traffic. It was calibrated on real data: only 2 of 499,621 real test flows
fall outside the training range, against 72 of 100 rows in a synthetic file. Truly new *attack
types* in real traffic are a separate, genuine limitation (held-out-day recall 0.101).

## K. Ethics, SDGs, future

**56. Ethical considerations?**
Use only data you're authorised to analyse. Flow data can reveal behaviour, so privacy matters.
False positives can burden or wrongly suspect users, and false negatives give false security, so
the tool is a decision aid and not an authority. No raw traffic is committed to the repository.

**57. How does it relate to the SDGs?**
SDG 9 (resilient digital infrastructure) and SDG 16 (safer digital environments), as potential
contributions of an educational prototype, not measured impact.

**58. What would you do next?**
Live flow extraction with CICFlowMeter; newer datasets and cross-dataset tests; drift monitoring;
cost-based thresholds; calibration; anomaly detection for unseen attacks.

---

# Quick revision sheet

- **Task:** binary ATTACK(1)/BENIGN(0); positive = ATTACK. Second model: 9-class family.
- **Data:** CIC-IDS2017 MachineLearningCSV, 8 day files, ~78 flow features, `Label`.
- **Cleaning order:** strip headers → drop duplicate column → normalise labels → inf→NaN →
  targets → **de-duplicate** → split.
- **Split:** stratified 80/20 on the fine label, seed 42. Test locked.
- **Pipeline:** ColumnSelector → median imputer → VarianceThreshold(0) → [log+scaler for LR] →
  model.
- **Excluded:** Destination Port (testbed shortcut), source_file (leaks the day).
- **Features:** 4 sets compared by CV; smallest within 0.002 F1 of the best.
- **Models:** Dummy, LR, DT, RF, HistGB. RandomizedSearchCV, 3-fold, F1 (ATTACK).
- **Choice:** best CV F1 → tie-break CV specificity → speed. Never the test set.
- **Imbalance:** class_weight in the search; family model compares none/weights/in-fold SMOTE.
- **Metrics:** P=TP/(TP+FP), R=TP/(TP+FN), F1=2PR/(P+R), Spec=TN/(TN+FP), FPR=FP/(FP+TN),
  FNR=FN/(FN+TP), ROC-AUC, PR-AUC.
- **Extra evaluation:** per-family detection, error medians, threshold sweep, held-out Friday.
- **Explainability:** permutation (test), built-in, SHAP / coef×value per flow. Not causal.
- **Key numbers to memorise:** 2,830,743 flows → 2,498,105 after removing 331,200 duplicates.
  Final model Random Forest, 30 features. Test recall 99.77 %, precision 99.64 %, F1 0.9970,
  FPR 0.074 %, 199 missed of 85,029 attacks, 308 false alarms. Dummy accuracy 82.98 %. LR F1 0.846.
  Held-out Friday recall **10.1 %**. Family model macro-F1 0.893. Training time 9.7 min on my laptop.

---

# Website demonstration script (about 5 minutes)

1. **Dashboard (30 s).** "NetGuard AI classifies flow records as attack or benign. It's offline
   analysis, not live monitoring." Point to the disclaimer, the dataset and split, and the five
   test metrics. Explain recall and the false-alarm rate in one sentence each.
2. **Dataset explorer (60 s).** Show the class distribution on a log scale ("this is why accuracy
   misleads"). Show the data-quality findings: infinite values, the duplicated column, how many
   duplicates were removed and why that matters for leakage. Show flows per day ("attacks are tied
   to days, so I also test on a held-out day").
3. **Model performance (90 s).** Show the comparison table: the Dummy has high accuracy but zero
   recall. Read the selection rule aloud: "chosen by cross-validation, not the test set." Show the
   confusion matrix and name FN and FP in plain words. Open the threshold tab: "lower threshold,
   more attacks caught, more false alarms." Scroll to the error analysis (which labels are missed)
   and the held-out day (performance on unseen attack types).
4. **Intrusion detection (60 s).** Open *Try the demo sample* → *Classify the demo sample*. Read
   the verdict ("N of M flows flagged"), show the P(attack) distribution, and compare with the true
   labels ("the label wasn't used"). Download the CSV. Then *Enter one flow* → *Classify this flow*
   → show "Why this prediction".
5. **Robustness (30 s).** Upload a CSV with a column removed (prepare it in advance), and show the
   clear error naming the missing column.
6. **Explainability and limits (30 s).** Show permutation importance: "what the model uses, not
   what causes attacks." Finish on *About & limitations*.

**Before the viva:** run `python -m pytest -q` and `python -m src.train` on the demo laptop. Open
the site once so the model is cached. Prepare the broken CSV for step 5.
