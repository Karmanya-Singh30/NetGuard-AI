# CICIDS2017 — dataset notes

_Corrected version. The previous draft overstated the size ("~50 million records") and listed
"SSH"/"FTP" as classes; the real labels are `SSH-Patator` and `FTP-Patator`. **Measured**
statistics (row counts, missing/infinite values, duplicates, class distribution) are produced by
`python -m src.train` in `reports/results/dataset_profile.json`; they are deliberately not
copied here by hand._

## Source
- **CIC-IDS2017**, Canadian Institute for Cybersecurity, University of New Brunswick:
  https://www.unb.ca/cic/datasets/ids-2017.html
- Release used: **MachineLearningCSV** (`MachineLearningCSV.zip`), flow features extracted with
  CICFlowMeter.
- Citation: I. Sharafaldin, A. H. Lashkari, A. A. Ghorbani, "Toward Generating a New Intrusion
  Detection Dataset and Intrusion Traffic Characterization," *Proc. ICISSP 2018*, pp. 108–116.
- Licence/usage: the CIC publishes the dataset for research; cite the paper above. Check the
  official page for current terms before redistributing. The raw files are **not** committed to
  this repository.

## Files (one per capture session, July 2017)
| File | Traffic it contains (per CIC documentation) |
|---|---|
| Monday-WorkingHours | Benign only |
| Tuesday-WorkingHours | Benign, FTP-Patator, SSH-Patator |
| Wednesday-workingHours | Benign, DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest, Heartbleed |
| Thursday-…-Morning-WebAttacks | Benign, Web Attack – Brute Force, – XSS, – Sql Injection |
| Thursday-…-Afternoon-Infilteration | Benign, Infiltration |
| Friday-…-Morning | Benign, Bot |
| Friday-…-Afternoon-PortScan | Benign, PortScan |
| Friday-…-Afternoon-DDos | Benign, DDoS |

## Schema
- ~78 numeric flow features plus `Label`. Header names carry leading spaces, and
  `Fwd Header Length` appears twice. Both are handled in `src/data_loading.py`.
- No IP addresses, timestamps or flow IDs in this release. `Destination Port` is present and is
  excluded from model inputs (see `docs/METHODOLOGY.md` §3).

## Targets used in this project
- Binary: `is_attack` = 1 for every non-BENIGN label (positive class = ATTACK).
- Family: DoS (4 DoS labels), DDoS, PortScan, BruteForce (FTP/SSH-Patator), WebAttack (3 labels),
  Bot, Infiltration, Heartbleed, BENIGN.

## Known issues to discuss in the report
- Lab testbed traffic from 2017, so it differs from today's real networks.
- Published labelling and flow-construction errors: Engelen, Rimmer & Joosen (IEEE SPW 2021);
  Lanvin et al. (CRiSIS 2022).
- Infinite/NaN rates when a flow duration is 0; many exact duplicate flows; several all-zero
  columns; some extremely rare classes (Heartbleed, Infiltration, SQL injection).
- Attack type is confounded with capture day.
