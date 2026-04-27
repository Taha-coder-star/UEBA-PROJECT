"""Top-K ranking analysis — no retraining.

Ranks all users by their maximum anomaly score (descending) and evaluates
Precision / Recall / F1 when the top K users are flagged as anomalies.
K values: 10, 20, 50, 100.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

REPO_DIR = Path(os.environ.get("DLP_REPO", str(Path(__file__).resolve().parent.parent)))
sys.path.insert(0, str(REPO_DIR))
from config import CLEANED_DIR  # noqa: E402
from ground_truth import describe_selection, select_ground_truth_release  # noqa: E402

IFOREST_CSV  = CLEANED_DIR / "email_user_daily_scored.csv"
LSTM_CSV     = CLEANED_DIR / "email_user_daily_lstm_scored.csv"

K_VALUES = [10, 20, 50, 100]


# ── Ground truth ──────────────────────────────────────────────────────────────

def load_insider_users() -> set[str]:
    return select_ground_truth_release([IFOREST_CSV, LSTM_CSV]).matching_users


# ── Ranking + metrics ─────────────────────────────────────────────────────────

def topk_metrics(
    user_scores: pd.DataFrame,   # columns: user, score
    insider_users: set[str],
    k: int,
) -> dict:
    top_k = set(user_scores.head(k)["user"])
    tp = len(top_k & insider_users)
    fp = k - tp
    fn = len(insider_users) - tp
    prec   = tp / k
    recall = tp / len(insider_users)
    f1     = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
    return {"k": k, "precision": prec, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def print_table(rows: list[dict], title: str, n_users: int, n_insiders: int) -> None:
    header = (f"{'K':>5}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}"
              f"  {'TP':>5}  {'FP':>5}  {'FN':>5}")
    width = len(header)
    print(f"\n{'─' * width}")
    print(f"  {title}  ({n_insiders} insiders / {n_users} users)")
    print(f"{'─' * width}")
    print(header)
    print(f"{'─' * width}")
    for r in rows:
        print(
            f"  {r['k']:>3}  {r['precision']:>10.4f}  {r['recall']:>8.4f}"
            f"  {r['f1']:>8.4f}  {r['tp']:>5}  {r['fp']:>5}  {r['fn']:>5}"
        )
    print(f"{'─' * width}\n")


# ── Per-model analysis ────────────────────────────────────────────────────────

def analyse(
    df: pd.DataFrame,
    score_col: str,
    insider_users: set[str],
    model_name: str,
) -> None:
    user_scores = (
        df.groupby("user")[score_col]
        .max()
        .reset_index()
        .rename(columns={score_col: "score"})
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )

    n_users    = len(user_scores)
    n_insiders = len(insider_users & set(user_scores["user"]))

    rows = [topk_metrics(user_scores, insider_users, k) for k in K_VALUES]
    print_table(rows, model_name, n_users, n_insiders)

    # Show where insiders actually rank
    user_scores["rank"] = range(1, n_users + 1)
    user_scores["is_insider"] = user_scores["user"].isin(insider_users)
    insider_ranks = user_scores.loc[user_scores["is_insider"], ["user", "rank", "score"]]
    print(f"  Insider rank distribution (out of {n_users}):")
    desc = insider_ranks["rank"].describe()
    print(f"    min={int(desc['min'])}  median={int(desc['50%'])}  "
          f"max={int(desc['max'])}  mean={desc['mean']:.1f}")
    top10  = (insider_ranks["rank"] <= 10).sum()
    top20  = (insider_ranks["rank"] <= 20).sum()
    top50  = (insider_ranks["rank"] <= 50).sum()
    top100 = (insider_ranks["rank"] <= 100).sum()
    print(f"    In top-10: {top10}  top-20: {top20}  top-50: {top50}  top-100: {top100}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    gt = select_ground_truth_release([IFOREST_CSV, LSTM_CSV])
    insider_users = gt.matching_users
    print("Ground truth:")
    print(f"  {describe_selection(gt)}\n")

    if IFOREST_CSV.exists():
        idf = pd.read_csv(IFOREST_CSV, usecols=["user", "iforest_score"])
        analyse(idf, "iforest_score", insider_users, "Isolation Forest")
    else:
        print(f"[SKIP] {IFOREST_CSV} not found")

    if LSTM_CSV.exists():
        ldf = pd.read_csv(LSTM_CSV, usecols=["user", "lstm_score", "lstm_risk_severity"])
        ldf = ldf[ldf["lstm_risk_severity"] != "undetermined"]
        analyse(ldf, "lstm_score", insider_users, "LSTM Autoencoder")
    else:
        print(f"[SKIP] {LSTM_CSV} not found")


if __name__ == "__main__":
    main()
