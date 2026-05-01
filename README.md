# AI-Driven UEBA + DLP Risk Scoring System

University project using the CERT Insider Threat Dataset r4.2.

This repo contains one clean pipeline for insider-threat detection:

1. Clean CERT r4.2 raw logs into daily per-user features.
2. Train an Isolation Forest baseline.
3. Train an LSTM Autoencoder for temporal behavior anomalies.
4. Score DLP content sensitivity from email and file activity.
5. Evaluate against r4.2 ground truth.
6. Optionally run a Genetic Algorithm to optimize risk weights and alert thresholds.
7. Explore results in a Streamlit dashboard.

## Current Canonical Workflow

Use this notebook in Colab:

```text
colab/Final_R42_UEBA_DLP_Pipeline.ipynb
```

Or run the pipeline script directly:

```bash
python colab/run_full_pipeline.py --install-deps --download-data
```

For a quick run:

```bash
python colab/run_full_pipeline.py --install-deps --download-data --smoke
```

To launch the dashboard after artifacts already exist:

```bash
DLP_ROOT=/content/dlp-data DLP_REPO=/content/dlp-project streamlit run /content/dlp-project/app/ueba_dashboard_tabs.py
```

On Windows/local PowerShell, run from the repo root after placing raw CERT CSVs under `archive/`:

```powershell
python colab/run_full_pipeline.py
streamlit run app/ueba_dashboard_tabs.py
```

## Required Raw Data

Raw files are intentionally ignored by Git because they are large:

```text
archive/email.csv
archive/file.csv
archive/logon.csv
archive/device.csv
archive/psychometric.csv
```

Optional files such as `users.csv`, `ldap.csv`, and `decoy_file.csv` are used if present and skipped if missing.

The small ground-truth index is kept in Git:

```text
archive/answers/answers/insiders.csv
```

## Main Files

```text
app/
  ueba_dashboard.py           Shared Streamlit dashboard logic
  ueba_dashboard_tabs.py      Main tabbed dashboard UI

colab/
  Final_R42_UEBA_DLP_Pipeline.ipynb
  run_full_pipeline.py        One-command Colab/local runner
  ground_truth.py             Selects the matching CERT release labels
  train_isolation_forest_cert.py
  train_lstm_autoencoder_cert.py
  evaluate_cert.py
  user_level_eval.py
  visualize_isolation_forest_cert.py
  visualize_lstm_autoencoder_cert.py
  visualize_user_level.py
  risk_scorer.py              Composite UEBA + DLP risk scoring
  ga_optimizer.py             GA weight and threshold optimization

scripts/
  clean_cert_email_data.py
  score_content_sensitivity.py

reports/
  Final_UEBA_DLP_R42_Report.md
```

## Pipeline Outputs

Generated files are ignored by Git and can be regenerated:

```text
cleaned/email_user_daily_with_psychometric.csv
cleaned/email_user_daily_scored.csv
cleaned/email_user_daily_lstm_scored.csv
cleaned/content_sensitivity_daily.csv
models/isolation_forest_cert.pkl
models/lstm_autoencoder_cert.pkl
models/ga_optimized_config.json
models/evaluation_report.json
plots/
```

## Risk Scoring

The final risk score combines seven normalized signals:

```text
lstm_p95
after_hours
bcc_usage
file_exfil
usb_activity
multi_pc
content_sensitivity
```

`colab/ga_optimizer.py` optimizes the same seven signal weights plus seven explanation thresholds. It does not retrain the Isolation Forest or LSTM; it only searches for better risk-scoring weights using existing scored CSVs.

## Dashboard

Run:

```bash
streamlit run app/ueba_dashboard_tabs.py
```

The dashboard includes separate pages for:

- Pipeline overview
- Isolation Forest
- LSTM Autoencoder
- Risk queue
- User investigation
- GA explanation
- Evaluation summary

## Cleanup Notes

Large raw data, generated models, generated plots, Python caches, virtual environments, and JavaScript dependencies are ignored. If they appear locally, they are not required in Git.
