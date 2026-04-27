"""Generate evaluation_report.json for the dashboard.

This script reuses existing scored CSVs, auto-selects the CERT answer-key
release that matches those scored users, and evaluates IF/LSTM without
redownloading data or retraining models.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

REPO_DIR = Path(os.environ.get("DLP_REPO", str(Path(__file__).resolve().parent.parent)))
sys.path.insert(0, str(REPO_DIR))
from config import CLEANED_DIR, MODELS_DIR  # noqa: E402
from ground_truth import describe_selection, load_day_labels  # noqa: E402

IFOREST_CSV = CLEANED_DIR / "email_user_daily_scored.csv"
LSTM_CSV = CLEANED_DIR / "email_user_daily_lstm_scored.csv"
REPORT_PATH = MODELS_DIR / "evaluation_report.json"


def evaluate(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> dict:
    if y_true.sum() == 0:
        return {}
    roc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "roc_auc": round(roc, 4),
        "avg_precision": round(ap, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "n_insiders": int(y_true.sum()),
        "n_total": int(len(y_true)),
    }


def main() -> None:
    day_labels, gt = load_day_labels([IFOREST_CSV, LSTM_CSV])
    insider_users = gt.matching_users

    print("Loading ground truth:")
    print(f"  {describe_selection(gt)}")
    print(f"  Insider user-days in selected release: {len(day_labels)}")

    report: dict = {
        "ground_truth": {
            "cert_release": gt.dataset,
            "insiders_csv": str(gt.insiders_path),
            "matching_insider_users": gt.match_count,
            "total_release_insider_users": gt.total_release_users,
            "scored_users": gt.scored_user_count,
            "release_match_counts": gt.release_match_counts,
        }
    }

    if IFOREST_CSV.exists():
        print(f"Loading IF scored: {IFOREST_CSV}")
        idf = pd.read_csv(
            IFOREST_CSV,
            usecols=["user", "email_day", "iforest_score", "risk_severity", "dataset_split"],
        )
        idf["user"] = idf["user"].astype(str)
        idf["email_day"] = pd.to_datetime(idf["email_day"], errors="coerce").dt.normalize()

        test_idf = idf[idf["dataset_split"] == "test"].merge(
            day_labels, on=["user", "email_day"], how="left"
        )
        test_idf["is_insider"] = test_idf["is_insider"].fillna(0).astype(int)
        y_true = test_idf["is_insider"].values
        y_score = test_idf["iforest_score"].values
        y_pred = test_idf["risk_severity"].isin(["suspicious", "high"]).astype(int).values
        report["if_day_test"] = evaluate(y_true, y_score, y_pred)
        print(
            "  IF day test  : "
            f"ROC={report['if_day_test'].get('roc_auc', 'n/a')} "
            f"positives={report['if_day_test'].get('n_insiders', 0)}"
        )

        user_max = idf.groupby("user")["iforest_score"].max().reset_index()
        labels = pd.DataFrame({"user": sorted(insider_users), "is_insider": 1})
        user_max = user_max.merge(labels, on="user", how="left")
        user_max["is_insider"] = user_max["is_insider"].fillna(0).astype(int)
        threshold = idf["iforest_score"].quantile(0.93)
        y_true = user_max["is_insider"].values
        y_score = user_max["iforest_score"].values
        y_pred = (y_score >= threshold).astype(int)
        report["if_user_all"] = evaluate(y_true, y_score, y_pred)
        print(
            "  IF user all  : "
            f"ROC={report['if_user_all'].get('roc_auc', 'n/a')} "
            f"positives={report['if_user_all'].get('n_insiders', 0)}"
        )
    else:
        print(f"  [SKIP] IF scored file not found: {IFOREST_CSV}")

    if LSTM_CSV.exists():
        print(f"Loading LSTM scored: {LSTM_CSV}")
        ldf = pd.read_csv(
            LSTM_CSV,
            usecols=["user", "email_day", "lstm_score", "lstm_risk_severity", "dataset_split"],
        )
        ldf["user"] = ldf["user"].astype(str)
        ldf["email_day"] = pd.to_datetime(ldf["email_day"], errors="coerce").dt.normalize()
        ldf = ldf[ldf["lstm_risk_severity"] != "undetermined"]

        test_ldf = ldf[ldf["dataset_split"] == "test"].merge(
            day_labels, on=["user", "email_day"], how="left"
        )
        test_ldf["is_insider"] = test_ldf["is_insider"].fillna(0).astype(int)
        y_true = test_ldf["is_insider"].values
        y_score = test_ldf["lstm_score"].fillna(0).values
        y_pred = test_ldf["lstm_risk_severity"].isin(["suspicious", "high"]).astype(int).values
        report["lstm_day_test"] = evaluate(y_true, y_score, y_pred)
        print(
            "  LSTM day test: "
            f"ROC={report['lstm_day_test'].get('roc_auc', 'n/a')} "
            f"positives={report['lstm_day_test'].get('n_insiders', 0)}"
        )

        user_max = ldf.groupby("user")["lstm_score"].max().reset_index()
        labels = pd.DataFrame({"user": sorted(insider_users), "is_insider": 1})
        user_max = user_max.merge(labels, on="user", how="left")
        user_max["is_insider"] = user_max["is_insider"].fillna(0).astype(int)
        threshold = ldf["lstm_score"].quantile(0.93)
        y_true = user_max["is_insider"].values
        y_score = user_max["lstm_score"].fillna(0).values
        y_pred = (y_score >= threshold).astype(int)
        report["lstm_user_all"] = evaluate(y_true, y_score, y_pred)
        print(
            "  LSTM user all: "
            f"ROC={report['lstm_user_all'].get('roc_auc', 'n/a')} "
            f"positives={report['lstm_user_all'].get('n_insiders', 0)}"
        )
    else:
        print(f"  [SKIP] LSTM scored file not found: {LSTM_CSV}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved evaluation report to: {REPORT_PATH}")

    print("\nSUMMARY")
    print(f"{'Metric':<18} {'IF Day':>10} {'IF User':>10} {'LSTM Day':>10} {'LSTM User':>10}")
    print("-" * 60)
    for key in ("roc_auc", "avg_precision", "precision", "recall", "f1"):
        vals = [
            report.get(section, {}).get(key, "-")
            for section in ("if_day_test", "if_user_all", "lstm_day_test", "lstm_user_all")
        ]
        print(f"  {key:<16} " + "  ".join(f"{str(value):>10}" for value in vals))


if __name__ == "__main__":
    main()
