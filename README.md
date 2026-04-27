# AI-Driven UEBA + DLP Risk Scoring System Using Genetic Algorithm for Anomaly Detection

**University project — CERT Insider Threat Dataset r4.2**

A complete User and Entity Behaviour Analytics (UEBA) and Data Loss Prevention (DLP) pipeline that combines unsupervised deep learning, rule-based content sensitivity classification, and Genetic Algorithm weight optimisation to detect insider threats. Built on the Carnegie Mellon CERT Insider Threat Dataset r4.2.

---

## Overview

Insider threats are among the hardest security problems to detect: the attacker already has legitimate access and their malicious activity is interleaved with normal work. This system takes a layered approach:

1. A **temporal deep learning model** (LSTM Autoencoder) learns each user's normal behavioural profile from sequences of daily activity features and flags days whose reconstruction error is anomalously high.
2. An **Isolation Forest** provides a shallow-learning baseline for comparison.
3. A **rule-based DLP content sensitivity classifier** scores email and file events as PUBLIC / INTERNAL / SENSITIVE / RESTRICTED, feeding a seventh risk signal into the scoring layer.
4. A **weighted multi-signal risk scorer** fuses all signals into a single ranked investigation queue.
5. A **Genetic Algorithm** evolves the signal weights and flag thresholds without retraining any model, maximising detection F1 at the user level.
6. A **Streamlit dashboard** provides interactive exploration, per-user explanation, and a live view of GA-optimised vs baseline metrics.

---

## Dataset

**CERT Insider Threat Dataset r4.2** — Carnegie Mellon University / SEI.

| Property | Value |
|---|---|
| Users | 1,000 |
| User-day records | 326,985 |
| Date range | 2010-01-02 — 2011-05-16 |
| Ground-truth insiders | 70 users |
| Train split | 261,220 rows (through 2011-02-05) |
| Test split | 65,765 rows (2011-02-06 onward) |

Raw source files (`archive/`) contain email logs, logon events, USB device connections, file system activity, and Big Five psychometric scores per employee. They are not included in the repository due to size and licence restrictions. Obtain them from the [CERT dataset page](https://resources.sei.cmu.edu/library/asset-view.cfm?assetid=508099) and place them under `archive/` before running the cleaning stage.

---

## Architecture

```
archive/
  email.csv  logon.csv  device.csv  file.csv  psychometric.csv
       |
       v
scripts/clean_cert_email_data.py
       |
       v
cleaned/email_user_daily_with_psychometric.csv   (326,985 rows x 32 features)
       |
       +----------+----------+
       |                     |
       v                     v
colab/train_isolation_    colab/train_lstm_
forest_cert.py            autoencoder_cert.py
       |                     |
       v                     v
cleaned/email_user_        cleaned/email_user_
daily_scored.csv           daily_lstm_scored.csv
(IF anomaly scores)        (LSTM reconstruction error scores)
                                    |
scripts/score_content_sensitivity.py
  (email_cleaned.csv + file_cleaned.csv)
       |
       v
cleaned/content_sensitivity_daily.csv
       |
       v
colab/risk_scorer.py
  - 7-signal weighted aggregation
  - per-user normalisation + ranking
  - rule-based XAI explanations
       |
       v
colab/ga_optimizer.py
  - evolves signal weights + flag thresholds
  - fitness = 0.8 * F1@50 + 0.2 * flag coverage@50
  - outputs models/ga_optimized_config.json
       |
       v
app/ueba_dashboard.py  (Streamlit)
```

---

## Pipeline Stages

### 1. Data Cleaning and Feature Engineering

`scripts/clean_cert_email_data.py` merges the five raw source tables into a single per-user-per-day feature matrix. Processing is chunked to handle the ~3 GB of raw CSV files within normal memory constraints.

**Features produced per user-day (32 total):**

| Category | Features |
|---|---|
| Email activity (16) | email count, size stats, attachment counts and ratio, after-hours ratio, BCC / CC counts and ratios, unique PCs, recipient counts |
| Logon / session (4) | logon count, logoff count, after-hours logons, unique workstations |
| USB / removable media (2) | connect events, disconnect events |
| File system (5) | total file ops, files to removable media, files from removable media, write count, after-hours file activity |
| Psychometric (5) | Big Five personality scores (O, C, E, A, N) |

### 2. Isolation Forest Baseline

`colab/train_isolation_forest_cert.py` trains a scikit-learn Isolation Forest on the training split.

| Parameter | Value |
|---|---|
| n_estimators | 300 |
| contamination | 0.03 |
| Input features | 32 |
| Suspicious threshold (p95) | 0.496 |
| High-risk threshold (p99) | 0.642 |

**Evaluation finding:** The Isolation Forest assigns lower anomaly scores to ground-truth insiders than to normal users on all aggregation strategies (ROC AUC < 0.5). The model is effectively inverted for this task and is retained only as a comparative baseline. This inversion is a known failure mode of Isolation Forest on datasets where outliers cluster together or overlap with the inlier distribution.

### 3. LSTM Autoencoder

`colab/train_lstm_autoencoder_cert.py` trains a sequence autoencoder on 7-day sliding windows of per-user daily features.

**Why LSTM for behavioural DLP?** Insider threat activity has a temporal character — a user's risk is not captured by a single day's behaviour but by shifts in their pattern over time. An LSTM Autoencoder trained only on normal behaviour learns the expected temporal structure; days that deviate from a user's historical pattern produce high reconstruction error. This makes the model sensitive to gradual behavioural drift, which is characteristic of the planning and execution phases of insider incidents.

| Parameter | Value |
|---|---|
| Window size | 7 days |
| Hidden dimension | 32 |
| Latent dimension | 16 |
| Training epochs | 20 |
| Batch size | 256 |
| Optimiser | Adam, lr = 1e-3 |
| Loss | Mean squared reconstruction error |
| Suspicious threshold (p95) | 0.723 |
| High-risk threshold (p99) | 0.929 |

User-level scores are aggregated from daily reconstruction errors. Three aggregations are compared: `score_max`, `score_mean`, and `score_p95` (95th percentile of a user's daily scores). `score_p95` provides the strongest insider / normal separation because `score_max` saturates at 1.0 for nearly every user.

**Top anomalies identified (test split, score = 1.0):** BER0314 (2011-03-22), DKB0631 (2011-03-28 through 2011-03-31), WLV0566 (2011-04-18 through 2011-04-20).

### 4. DLP Content Sensitivity Scoring

`scripts/score_content_sensitivity.py` applies a lightweight, fully vectorised keyword and rule-based classifier to raw email and file events in chunks, without requiring a language model.

**Sensitivity tiers:**

| Label | Score | Signals |
|---|---|---|
| RESTRICTED | 3 | Credentials, PII (SSN, passport, routing numbers), PE executables written to USB, medical records |
| SENSITIVE | 2 | Salary/payroll/budget, merger/acquisition/strategy, confidential/NDA/trade secret, office documents written to USB, PDF/archive files |
| INTERNAL | 1 | General internal documents, any removable-media read activity |
| PUBLIC | 0 | No indicators triggered |

File events use both keyword matching on content and structural signals: file extension tier table, magic-byte detection from the hex prefix in the content field (OLE2, PDF, ZIP, PE), and activity type combined with destination (write to removable media upgrades the tier). Email events additionally boost score for large attachments.

Output: `cleaned/content_sensitivity_daily.csv` — one row per user per day with mean sensitivity score, max sensitivity score, sensitive event count, restricted event count, and top label.

### 5. Weighted Risk Scoring

`colab/risk_scorer.py` fuses the seven signals into a single composite risk score per user.

**Default signal weights (sum = 1.0):**

| Signal | Default weight | Description |
|---|---|---|
| `lstm_p95` | 0.45 | LSTM reconstruction-error 95th percentile |
| `after_hours` | 0.13 | Mean after-hours email / logon fraction |
| `bcc_usage` | 0.09 | Mean BCC email ratio |
| `file_exfil` | 0.09 | Files written to removable media (rate) |
| `usb_activity` | 0.09 | Total USB connect events |
| `multi_pc` | 0.05 | Max distinct workstations accessed |
| `content_sensitivity` | 0.10 | Mean daily max DLP sensitivity score |

Each signal is independently normalised to [0, 1] using min-max scaling across all users before weighting. The final risk score is the weighted sum. Users are ranked descending to form a best-first investigation queue: investigators work from rank 1 downward, maximising expected insider detections per unit of investigation effort.

**Rule-based XAI explanations** are generated per user by applying per-signal thresholds to the normalised values. Each triggered threshold produces a plain-English sentence identifying the specific behavioural indicator (e.g., "Files copied to removable media (possible data exfiltration)").

### 6. Genetic Algorithm Weight Optimisation

`colab/ga_optimizer.py` evolves the risk-score weights and flag thresholds without retraining any model. It reads only the existing scored CSVs and the ground-truth insiders file.

**Chromosome (12 genes):**
- Genes 0–5: raw weights for the six core signals (normalised to sum = 1 during evaluation)
- Genes 6–11: flag thresholds per signal (range [0.10, 0.95])

**Fitness function:**

```
fitness = 0.8 * F1@50 + 0.2 * flag_coverage@50
```

where `flag_coverage@50` is the fraction of the top-50 risk-ranked users for whom at least one behavioural flag is triggered — rewarding explainability alongside detection quality.

**GA parameters:**

| Parameter | Value |
|---|---|
| Population size | 60 |
| Max generations | 100 |
| Early-stop patience | 25 generations |
| Selection | Tournament (k = 5) |
| Crossover | Uniform (p = 0.5 per gene) |
| Mutation | Gaussian (σ = 0.04, rate = 0.25 per gene) |
| Elitism | Top 4 individuals copied unchanged |

The population is seeded with the domain-knowledge baseline chromosome so the GA refines rather than searches from scratch.

**GA results:**

| Metric | Baseline | GA-Optimised | Delta |
|---|---|---|---|
| F1 @ K=50 | 0.2500 | 0.2667 | +0.0167 |
| Precision @ K=50 | 0.3000 | 0.3200 | +0.0200 |
| Recall @ K=50 | 0.2143 | 0.2286 | +0.0143 |
| TP (of 70 insiders) | 15 | 16 | +1 |
| Flag coverage @ K=50 | — | 1.0000 | — |

The GA redistributed weight from `lstm_p95` toward `file_exfil` and `usb_activity`, consistent with the CERT r4.2 dataset's insider population, which is dominated by data exfiltration via removable media.

The optimised config is saved to `models/ga_optimized_config.json` and loaded automatically by `risk_scorer.py` at import time. Deleting the file reverts to domain-knowledge defaults. A 6-signal config from an older run is handled transparently: the six weights are scaled by 0.9 and `content_sensitivity` receives the remaining 0.10.

---

## Evaluation Results

| Configuration | Aggregation | Threshold | K | Precision | Recall | F1 | TP |
|---|---|---|---|---|---|---|---|
| LSTM Autoencoder (GA weights) | score_p95 | 90th pct | 50 | 0.32 | 0.23 | **0.267** | **16** |
| LSTM Autoencoder (domain weights) | score_p95 | 90th pct | 50 | 0.30 | 0.21 | 0.250 | 15 |
| LSTM Autoencoder | score_p95 | 90th pct | 20 | **0.50** | 0.14 | 0.222 | 10 |
| Isolation Forest | any | any | any | < baseline | < baseline | < baseline | — |

The LSTM with `score_p95` aggregation and a threshold derived from the 90th percentile of train-user scores gives the best overall F1. The Isolation Forest consistently underperforms (ROC AUC < 0.5) and is not suitable as a primary detector on this dataset.

### Limitations

- **Dataset scope:** The CERT r4.2 dataset is synthetic. Email content and filenames are randomised, which significantly reduces the signal from the DLP content sensitivity classifier. On real enterprise data with unredacted content, the sensitivity signal would be far more informative.
- **Class imbalance:** 70 insiders among 1,000 users (7%) means absolute TP counts are small; F1 improvements of a few hundredths can represent meaningful operational gains but should be interpreted carefully.
- **GA convergence:** The GA converged in 26 generations (early-stop patience = 25). The search space is well-conditioned by the warm start from domain weights, but the improvement is incremental rather than transformative. A larger population or more generations may find further gains.
- **Temporal leakage guard:** All threshold calibration uses train-user score distributions only. Test-period behaviour is never used to set decision boundaries.
- **Isolation Forest inversion:** The model inversion (lower scores for insiders) is not corrected in this system. The IF output is preserved as a comparison baseline rather than an operational detector.

---

## Repository Structure

```
dlp-project/
├── app/
│   ├── ueba_dashboard.py         Streamlit dashboard (main UI)
│   └── monitoring_app.py         Lightweight monitoring view
├── archive/                      Raw CERT source files (NOT included — see Dataset)
│   ├── email.csv
│   ├── logon.csv
│   ├── device.csv
│   ├── file.csv
│   ├── psychometric.csv
│   └── answers/answers/insiders.csv
├── cleaned/                      Generated outputs — regenerate after clone
│   ├── email_user_daily_with_psychometric.csv
│   ├── email_user_daily_scored.csv          (Isolation Forest scores)
│   ├── email_user_daily_lstm_scored.csv     (LSTM scores)
│   └── content_sensitivity_daily.csv        (DLP sensitivity scores)
├── colab/
│   ├── train_isolation_forest_cert.py
│   ├── train_lstm_autoencoder_cert.py
│   ├── inference_isolation_forest_cert.py
│   ├── inference_lstm_autoencoder_cert.py
│   ├── risk_scorer.py                       7-signal weighted scorer + XAI
│   ├── ga_optimizer.py                      Genetic Algorithm weight optimiser
│   ├── user_level_eval.py                   User-level aggregation + evaluation
│   ├── evaluate_cert.py
│   ├── threshold_analysis.py
│   └── visualize_*.py
├── models/                       Generated model artifacts — regenerate after clone
│   ├── lstm_autoencoder_cert.pkl            (60 MB)
│   ├── isolation_forest_cert.pkl            (3.5 MB)
│   ├── lstm_autoencoder_summary.json
│   ├── isolation_forest_summary.json
│   ├── ga_optimized_config.json             GA-optimised weights + thresholds
│   └── ga_optimization_report.json         Convergence history + baseline comparison
├── plots/                        Generated visualisation outputs
├── scripts/
│   ├── clean_cert_email_data.py             Chunked data cleaning + feature eng.
│   ├── score_content_sensitivity.py         DLP content sensitivity scorer
│   ├── run_local_pipeline.py
│   └── evaluate_models.py
├── notebooks/
│   └── DLP_Pipeline.ipynb
├── config.py                     Central path configuration (local + Colab)
└── requirements.txt
```

> **Note on generated directories:** `cleaned/`, `models/`, and `plots/` contain outputs produced by the pipeline. They may not be present after a fresh clone. Follow the Quick Start steps below to regenerate them. The `archive/` directory must be populated manually with the CERT dataset files.

---

## Quick Start

### One-command Colab pipeline

For Google Colab, use the pipeline runner after cloning the repo and uploading
`kaggle.json`:

```python
REPO_URL = "https://github.com/Taha-coder-star/DLP-PROJECt.git"
!git clone {REPO_URL} /content/dlp-project
%cd /content/dlp-project
```

```python
from google.colab import files
import os, shutil

uploaded = files.upload()  # upload kaggle.json
os.makedirs("/root/.kaggle", exist_ok=True)
shutil.copy("kaggle.json", "/root/.kaggle/kaggle.json")
!chmod 600 /root/.kaggle/kaggle.json
```

Full run:

```python
!python /content/dlp-project/colab/run_full_pipeline.py --install-deps --download-data
```

Fast validation run using quick GA and 50k-row sensitivity scoring:

```python
!python /content/dlp-project/colab/run_full_pipeline.py --install-deps --download-data --smoke
```

Launch the dashboard at the end:

```python
!python /content/dlp-project/colab/run_full_pipeline.py --skip-clean --skip-iforest --skip-lstm --launch-dashboard
```

The Colab runner writes generated artifacts to `/content/dlp-data` and sets
`DLP_ROOT` / `DLP_REPO` for every stage automatically. Useful flags include
`--skip-clean`, `--skip-lstm`, `--skip-sensitivity`, `--skip-ga`, and
`--sensitivity-limit 50000`.

### Requirements

Python 3.10 or later is recommended.

```bash
pip install -r requirements.txt
```

`requirements.txt` installs: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `streamlit`, `torch`.

### Step 1 — Clean and engineer features

Processes all raw CSVs in chunks (~3 GB input, ~60 MB output). Expects `archive/` to be populated.

```bash
python scripts/clean_cert_email_data.py
```

Output: `cleaned/email_user_daily_with_psychometric.csv`

### Step 2 — Train Isolation Forest

```bash
python colab/train_isolation_forest_cert.py
```

Outputs: `cleaned/email_user_daily_scored.csv`, `models/isolation_forest_cert.pkl`, `models/isolation_forest_summary.json`

### Step 3 — Train LSTM Autoencoder

Requires a GPU for reasonable training time, but will run on CPU.

```bash
python colab/train_lstm_autoencoder_cert.py
```

Outputs: `cleaned/email_user_daily_lstm_scored.csv`, `models/lstm_autoencoder_cert.pkl`, `models/lstm_autoencoder_summary.json`

### Step 4 — Generate DLP content sensitivity signals

Processes ~4.6 M email and file events in chunks. Use `--limit` for a quick smoke test.

```bash
# Full run (~10 minutes)
python scripts/score_content_sensitivity.py

# Smoke test (50,000 rows per source, ~30 seconds)
python scripts/score_content_sensitivity.py --limit 50000
```

Output: `cleaned/content_sensitivity_daily.csv`

### Step 5 — Run the Genetic Algorithm optimiser

Reads the scored CSVs from Steps 2–4; does not retrain any model.

```bash
# Full optimisation (100 generations, ~1 minute)
python colab/ga_optimizer.py

# Quick smoke test (20 generations)
python colab/ga_optimizer.py --quick
```

Outputs: `models/ga_optimized_config.json`, `models/ga_optimization_report.json`

The dashboard and risk scorer load these automatically on next import.

### Step 6 — Launch the Streamlit dashboard

```bash
streamlit run app/ueba_dashboard.py
```

Open `http://localhost:8501` in a browser. Select settings in the sidebar and click **Run Analysis**.

---

## Dashboard Features

- **GA status badge** — sidebar shows whether GA-optimised weights are active
- **GA Optimisation Summary** — expandable panel with weight comparison table (default vs GA), baseline vs GA detection metrics, and fitness convergence chart
- **Detection metrics** — Precision, Recall, F1, TP, FP, FN for the selected model / aggregation / threshold / K
- **Top Suspicious Users table** — includes `content_sensitivity_norm` column alongside LSTM, behavioural, and risk-score columns; score column is labelled "GA-Optimized DLP Risk Score" when GA config is active
- **Sensitivity warning** — banner shown when `content_sensitivity_daily.csv` is missing, with the signal treated as zero
- **Charts** — P/R/F1 vs K sweep, score distribution (insiders vs normals), top-users risk bar, TP/FP/FN breakdown by K
- **User Investigation** — select any flagged user to see per-signal breakdown (7 bars with individual flag thresholds) and plain-English XAI explanation of which signals triggered the alert
- **AI algorithm documentation** — inline explanation of each pipeline stage including GA optimisation

---

## References

- Glasser, J. and Lindauer, B. (2013). *Bridging the gap: A pragmatic approach to generating insider threat data*. IEEE S&P Workshops.
- Greitzer, F. L. et al. (2010). *Combining traditional cyber security audit data with psychosocial data: Towards predictive modeling for insider threat identification*. CEUR Workshop Proceedings.
- Liu, F. T., Ting, K. M., and Zhou, Z.-H. (2008). *Isolation Forest*. ICDM 2008.
- Hochreiter, S. and Schmidhuber, J. (1997). *Long short-term memory*. Neural Computation.
- Malhotra, P. et al. (2016). *LSTM-based encoder-decoder for multi-sensor anomaly detection*. ICML Anomaly Detection Workshop.
