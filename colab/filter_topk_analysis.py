"""Filter-then-rank analysis — no retraining.

Stage 1: keep only users whose max anomaly score meets a percentile threshold
         (95th or 97th, derived from the train split).
Stage 2: rank the surviving users by score and flag the top K as anomalies.

Evaluated for every (threshold %, K) combination.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_DIR = Path(os.environ.get("DLP_REPO", str(Path(__file__).resolve().parent.parent)))
sys.path.insert(0, str(REPO_DIR))
from config import CLEANED_DIR  # noqa: E402
from ground_truth import describe_selection, select_ground_truth_release  # noqa: E402

IFOREST_CSV  = CLEANED_DIR / "email_user_daily_scored.csv"
LSTM_CSV     = CLEANED_DIR / "email_user_daily_lstm_scored.csv"

THRESHOLDS = [95, 97]
K_VALUES   = [10, 20, 50]


# ── Ground truth ──────────────────────────────────────────────────────────────

def load_insider_users() -> set[str]:
    return select_ground_truth_release([IFOREST_CSV, LSTM_CSV]).matching_users


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(top_k_users: set[str], insider_users: set[str], k: int) -> dict:
    tp = len(top_k_users & insider_users)
    fp = k - tp
    fn = len(insider_users) - tp
    prec   = tp / k if k else 0.0
    recall = tp / len(insider_users) if insider_users else 0.0
    f1     = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
    return {"precision": prec, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


# ── Print ─────────────────────────────────────────────────────────────────────

def print_table(rows: list[dict], title: str, n_total: int, n_insiders: int) -> None:
    header = (
        f"  {'Thresh':>7}  {'Filtered':>8}  {'K':>4}"
        f"  {'Precision':>10}  {'Recall':>8}  {'F1':>8}"
        f"  {'TP':>4}  {'FP':>5}  {'FN':>4}"
    )
    width = len(header)
    print(f"\n{'─' * width}")
    print(f"  {title}  ({n_insiders} insiders / {n_total} users total)")
    print(f"{'─' * width}")
    print(header)
    print(f"{'─' * width}")
    last_thresh = None
    for r in rows:
        sep = "  " if r["thresh_pct"] == last_thresh else ""
        if r["thresh_pct"] != last_thresh and last_thresh is not None:
            print(f"  {'·' * (width - 2)}")
        last_thresh = r["thresh_pct"]

        filtered_str = f"{r['filtered_users']}"
        if r["k"] > r["filtered_users"]:
            filtered_str += "*"   # K exceeds pool size

        print(
            f"  {r['thresh_pct']:>5}%  {filtered_str:>8}  {r['k']:>4}"
            f"  {r['precision']:>10.4f}  {r['recall']:>8.4f}  {r['f1']:>8.4f}"
            f"  {r['tp']:>4}  {r['fp']:>5}  {r['fn']:>4}"
        )
    print(f"{'─' * width}")
    print("  * K exceeds filtered pool — all survivors flagged.\n")


# ── Per-model analysis ────────────────────────────────────────────────────────

def analyse(
    df: pd.DataFrame,
    score_col: str,
    insider_users: set[str],
    model_name: str,
) -> None:
    # Thresholds derived from train split only
    train_scores = df.loc[df["dataset_split"] == "train", score_col].dropna()
    cutoffs = {p: float(np.percentile(train_scores, p)) for p in THRESHOLDS}

    # Max score per user across all splits
    user_scores = (
        df.groupby("user")[score_col]
        .max()
        .reset_index()
        .rename(columns={score_col: "score"})
    )
    n_total    = len(user_scores)
    n_insiders = len(insider_users & set(user_scores["user"]))

    rows = []
    for pct in THRESHOLDS:
        cutoff   = cutoffs[pct]
        filtered = (
            user_scores[user_scores["score"] >= cutoff]
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
        )
        pool_size = len(filtered)

        for k in K_VALUES:
            actual_k  = min(k, pool_size)
            top_k_set = set(filtered.head(actual_k)["user"])
            m = compute_metrics(top_k_set, insider_users, actual_k)
            rows.append({
                "thresh_pct":     pct,
                "cutoff":         cutoff,
                "filtered_users": pool_size,
                "k":              k,
                **m,
            })

    print_table(rows, model_name, n_total, n_insiders)

    # Insider breakdown per threshold
    user_scores["is_insider"] = user_scores["user"].isin(insider_users)
    for pct in THRESHOLDS:
        cutoff   = cutoffs[pct]
        filtered = user_scores[user_scores["score"] >= cutoff]
        ins_in   = filtered["is_insider"].sum()
        print(
            f"  {pct}th pct (cutoff={cutoff:.4f}): "
            f"{len(filtered)} users pass filter  —  "
            f"{ins_in} insiders retained ({ins_in/n_insiders*100 if n_insiders else 0.0:.1f}% recall)"
        )
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    gt = select_ground_truth_release([IFOREST_CSV, LSTM_CSV])
    insider_users = gt.matching_users
    print("Ground truth:")
    print(f"  {describe_selection(gt)}\n")

    if IFOREST_CSV.exists():
        idf = pd.read_csv(IFOREST_CSV, usecols=["user", "iforest_score", "dataset_split"])
        analyse(idf, "iforest_score", insider_users, "Isolation Forest")
    else:
        print(f"[SKIP] {IFOREST_CSV} not found")

    if LSTM_CSV.exists():
        ldf = pd.read_csv(LSTM_CSV, usecols=["user", "lstm_score", "lstm_risk_severity", "dataset_split"])
        ldf = ldf[ldf["lstm_risk_severity"] != "undetermined"]
        analyse(ldf, "lstm_score", insider_users, "LSTM Autoencoder")
    else:
        print(f"[SKIP] {LSTM_CSV} not found")


if __name__ == "__main__":
    main()
