# NetGuard AI — results summary

_Generated automatically by `src/train.py` on 2026-10-28T20:29:21+00:00._

## Data

- Raw rows: 2,830,743 from 8 file(s)
- Exact duplicates removed: 331,200; label-conflicting rows removed: 1,438
- Rows after cleaning: 2,498,105
- Train / test rows: 1,998,484 / 499,621 (attack share 0.1702 / 0.1702)
- Rows used to fit final models: 1,998,484 (all training rows (requested)); tuning subsample: 199,995

## Feature selection (smallest set with CV F1 within 0.002 of the best)

| Feature set | n | CV F1 | CV recall |
|---|---:|---:|---:|
| all_features | 76 | 0.9937 | 0.9908 |
| correlation_filtered | 48 | 0.9944 | 0.9920 |
| top30_mutual_info | 30 | 0.9930 | 0.9922 |
| top30_rf_importance | 30 | 0.9954 | 0.9942 |

Chosen: **top30_rf_importance**

## Binary model comparison (test set, positive = ATTACK)

| Model | CV F1 | Accuracy | Precision | Recall | F1 | Specificity | FPR | FNR | ROC-AUC | PR-AUC | Fit s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dummy (majority) | 0.0000 | 0.8298 | n/a | 0.0000 | 0.0000 | 1.0000 | 0.00000 | 1.00000 | n/a | n/a | 6.1 |
| Logistic Regression | 0.8545 | 0.9407 | 0.7591 | 0.9542 | 0.8456 | 0.9379 | 0.06209 | 0.04576 | 0.9849 | 0.9368 | 35.5 |
| Decision Tree | 0.9952 | 0.9991 | 0.9955 | 0.9990 | 0.9972 | 0.9991 | 0.00094 | 0.00101 | 0.9994 | 0.9980 | 37.4 |
| Random Forest | 0.9957 | 0.9990 | 0.9964 | 0.9977 | 0.9970 | 0.9993 | 0.00074 | 0.00234 | 1.0000 | 0.9999 | 27.7 |
| HistGradientBoosting | 0.9963 | 0.9991 | 0.9971 | 0.9976 | 0.9973 | 0.9994 | 0.00059 | 0.00242 | 1.0000 | 0.9999 | 20.9 |

**Final model:** Random Forest — highest mean CV F1 (ATTACK) on the training subsample; models within 0.001 are tie-broken by higher CV specificity, then faster prediction. Test-set results were NOT used for this choice.

False negatives (missed attacks): 199; false positives (false alarms): 308

Missed attacks by label: DoS Hulk: 61, Bot: 59, DDoS: 20, SSH-Patator: 11, DoS GoldenEye: 10, DoS Slowhttptest: 8, Web Attack - Brute Force: 6, Infiltration: 5, Web Attack - Sql Injection: 4, DoS slowloris: 4, Web Attack - XSS: 4, PortScan: 3, FTP-Patator: 3, Heartbleed: 1

## Attack-family model

| Imbalance strategy | CV macro-F1 | CV balanced acc. |
|---|---:|---:|
| none | 0.9397 | 0.9302 |
| class_weight | 0.9397 | 0.9302 |
| smote_in_fold | 0.9539 | 0.9688 |

Chosen: smote_in_fold. Test macro-F1 = 0.8930, weighted-F1 = 0.9986

| Family | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| BENIGN | 0.9997 | 0.9985 | 0.9991 | 414592 |
| DoS | 0.9960 | 0.9975 | 0.9968 | 38716 |
| DDoS | 0.9996 | 0.9990 | 0.9993 | 25603 |
| PortScan | 0.9953 | 0.9996 | 0.9974 | 18051 |
| BruteForce | 0.9956 | 0.9967 | 0.9962 | 1831 |
| WebAttack | 0.9816 | 0.9953 | 0.9884 | 428 |
| Bot | 0.5189 | 0.9821 | 0.6790 | 391 |
| Infiltration | 0.7143 | 0.7143 | 0.7143 | 7 |
| Heartbleed | 1.0000 | 0.5000 | 0.6667 | 2 |

## Held-out day (generalisation check)

Train on 5 Mon–Thu files, test on Friday (615,293 flows). Labels unseen in training: ['Bot', 'DDoS', 'PortScan']

Recall 0.1014, precision 0.9951, FPR 0.00028, F1 0.1841, ROC-AUC 0.7680
