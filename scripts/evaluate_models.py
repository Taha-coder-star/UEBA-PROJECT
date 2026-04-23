"""Evaluate Isolation Forest and LSTM Autoencoder against CERT r5.2 ground truth.

Usage:
    python scripts/evaluate_models.py --answers archive/answers.csv

The answers file (from the CERT r5.2 download package) must have at minimum:
    user, start, end  columns  (with optional: scenario, dataset, details)
    where start/end are dates bounding each insider's malicious activity window.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import CLEANED_DIR, MODELS_DIR


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_answers(answers_path: Path) -> pd.DataFrame:
    """Load CERT answers file and return normalised (user, date) insider labels."""
    df = pd.read_csv(answers_path)
    df.columns = df.columns.str.strip().str.lower()

    # Identify user column
    user_col = next((c for c in df.columns if c in ("user", "user_id", "userid")), None)
    if user_col is None:
        raise ValueError(f"No user column found in answers file. Columns: {df.columns.tolist()}")

    # Expand date ranges → one row per (user, date)
    if "start" in df.columns and "end" in df.columns:
        df["start"] = pd.to_datetime(df["start"], errors="coerce").dt.normalize()
        df["end"]   = pd.to_datetime(df["end"],   errors="coerce").dt.normalize()
        rows = []
        for _, row in df.iterrows():
            if pd.isna(row["start"]) or pd.isna(row["end"]):
                continue
            dates = pd.date_range(row["start"], row["end"], freq="D")
            for d in dates:
                rows.append({"user": row[user_col], "date": d})
        labels = pd.DataFrame(rows).drop_duplicates()
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        labels = df[[user_col, "date"]].rename(columns={user_col: "user"}).drop_duplicates()
    else:
        # No date info — user-level only
        labels = pd.DataFrame({"user": df[user_col].unique()})

    labels["is_insider"] = 1
    return labels


def load_scored(path: Path, score_col: str, severity_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["user", "email_day", score_col, severity_col, "dataset_split"])
    df["email_day"] = pd.to_datetime(df["email_day"], errors="coerce").dt.normalize()
    return df


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def add_labels(scored: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Merge ground-truth insider labels into the scored dataframe."""
    if "date" in labels.columns:
        # Day-level labels
        day_labels = labels[["user", "date", "is_insider"]].rename(columns={"date": "email_day"})
        scored = scored.merge(day_labels, on=["user", "email_day"], how="left")
    else:
        # User-level labels only — mark all days for that user
        user_labels = labels[["user", "is_insider"]]
        scored = scored.merge(user_labels, on="user", how="left")

    scored["is_insider"] = scored["is_insider"].fillna(0).astype(int)
    return scored


def user_level_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to user level: insider if any day is labelled insider."""
    return df.groupby("user")["is_insider"].max().reset_index()


def evaluate(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    """Return a dict of metrics for one model."""
    if y_true.sum() == 0:
        print(f"  [WARN] No positive labels found for {label} — check answers file user IDs match scored file.")
        return {}

    roc   = roc_auc_score(y_true, y_score)
    ap    = average_precision_score(y_true, y_score)
    cm    = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    prec  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "model": label,
        "roc_auc": round(roc, 4),
        "avg_precision": round(ap, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "n_positives": int(y_true.sum()),
        "n_total": int(len(y_true)),
    }


def print_result(r: dict, title: str) -> None:
    if not r:
        return
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")
    print(f"  ROC AUC         : {r['roc_auc']}")
    print(f"  Avg Precision   : {r['avg_precision']}")
    print(f"  Precision       : {r['precision']}")
    print(f"  Recall          : {r['recall']}")
    print(f"  F1              : {r['f1']}")
    print(f"  TP/FP/TN/FN     : {r['tp']} / {r['fp']} / {r['tn']} / {r['fn']}")
    print(f"  Insiders found  : {r['tp']} / {r['n_positives']} labelled")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(answers_path: Path, split: str | None) -> None:
    print(f"\nLoading answers from: {answers_path}")
    labels = load_answers(answers_path)
    has_dates = "date" in labels.columns

    insider_users = labels["user"].unique()
    print(f"  Insider users in answers : {len(insider_users)}")
    if has_dates:
        print(f"  Insider user-days        : {len(labels)}")

    # --- Isolation Forest ---
    if_path = CLEANED_DIR / "email_user_daily_scored.csv"
    if if_path.exists():
        if_df = load_scored(if_path, "iforest_score", "risk_severity")
        if split:
            if_df = if_df[if_df["dataset_split"] == split]
        if_df = add_labels(if_df, labels)

        # Day-level
        if has_dates:
            y_true  = if_df["is_insider"].values
            y_score = if_df["iforest_score"].values
            y_pred  = (if_df["risk_severity"].isin(["suspicious", "high"])).astype(int).values
            if_day  = evaluate(y_true, y_score, y_pred, "Isolation Forest (day-level)")
            print_result(if_day, "Isolation Forest — Day Level")

        # User-level
        if_users = user_level_labels(if_df)
        if_users = if_users.merge(
            if_df.groupby("user")["iforest_score"].max().reset_index(),
            on="user"
        )
        y_true  = if_users["is_insider"].values
        y_score = if_users["iforest_score"].values
        y_pred  = (y_score >= 0.5).astype(int)
        if_user = evaluate(y_true, y_score, y_pred, "Isolation Forest (user-level)")
        print_result(if_user, "Isolation Forest — User Level (max score per user)")
    else:
        print("  [SKIP] IF scored file not found")
        if_day, if_user = {}, {}

    # --- LSTM Autoencoder ---
    lstm_path = CLEANED_DIR / "email_user_daily_lstm_scored.csv"
    if lstm_path.exists():
        lstm_df = load_scored(lstm_path, "lstm_score", "lstm_risk_severity")
        if split:
            lstm_df = lstm_df[lstm_df["dataset_split"] == split]
        # Drop undetermined rows (no score assigned)
        lstm_df = lstm_df[lstm_df["lstm_risk_severity"] != "undetermined"]
        lstm_df = add_labels(lstm_df, labels)

        if has_dates:
            y_true   = lstm_df["is_insider"].values
            y_score  = lstm_df["lstm_score"].fillna(0).values
            y_pred   = (lstm_df["lstm_risk_severity"].isin(["suspicious", "high"])).astype(int).values
            lstm_day = evaluate(y_true, y_score, y_pred, "LSTM (day-level)")
            print_result(lstm_day, "LSTM Autoencoder — Day Level")

        lstm_users = user_level_labels(lstm_df)
        lstm_users = lstm_users.merge(
            lstm_df.groupby("user")["lstm_score"].max().reset_index(),
            on="user"
        )
        y_true   = lstm_users["is_insider"].values
        y_score  = lstm_users["lstm_score"].fillna(0).values
        y_pred   = (y_score >= 0.5).astype(int)
        lstm_user = evaluate(y_true, y_score, y_pred, "LSTM (user-level)")
        print_result(lstm_user, "LSTM Autoencoder — User Level (max score per user)")
    else:
        print("  [SKIP] LSTM scored file not found")
        lstm_day, lstm_user = {}, {}

    # --- Save report ---
    report = {
        "answers_file": str(answers_path),
        "split_evaluated": split or "all",
        "insider_users_in_answers": len(insider_users),
        "isolation_forest": {
            "day_level": if_day if has_dates else "n/a (no dates in answers)",
            "user_level": if_user,
        },
        "lstm_autoencoder": {
            "day_level": lstm_day if has_dates else "n/a (no dates in answers)",
            "user_level": lstm_user,
        },
    }
    out_path = MODELS_DIR / "evaluation_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport saved to: {out_path}\n")

    # --- Quick comparison ---
    print("SUMMARY COMPARISON")
    print(f"{'Metric':<18} {'IF User':>10} {'LSTM User':>10}")
    print("-" * 40)
    for key in ("roc_auc", "avg_precision", "precision", "recall", "f1"):
        iv = if_user.get(key, "-")
        lv = lstm_user.get(key, "-")
        print(f"  {key:<16} {str(iv):>10} {str(lv):>10}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DLP models against CERT ground truth")
    parser.add_argument("--answers", required=True, help="Path to CERT answers CSV file")
    parser.add_argument(
        "--split", default=None, choices=["train", "test", None],
        help="Evaluate only on this dataset split (default: all rows)"
    )
    args = parser.parse_args()
    main(Path(args.answers), args.split)
