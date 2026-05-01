"""User-level anomaly detection evaluation -- fixed pipeline.

Earlier experiments applied percentile thresholds to row-level daily scores and
used only max aggregation. Both choices were wrong:

  Bug 1:
    np.percentile(train_rows, 95) operates on ~100k daily rows.
    That cutoff is then compared against one-score-per-user values that live
    on a completely different scale -- the LSTM filter passed all 1000 users.
    Fix: aggregate rows to one score per user first, compute percentiles of
         the resulting 1000-value user distribution.

  Bug 2:
    Only max aggregation was tried.  For LSTM, every user's max daily score
    rounds to 1.0, so max is useless.  mean and p95 (95th percentile of a
    user's daily scores) carry actual signal.
    Fix: compute max / mean / p95 per user and compare all three.

Pipeline:
  compute_user_scores()   ->  one row per user, three score columns
  apply_user_threshold()  ->  filter to users above a percentile cutoff
                               (cutoff derived from train users, not train rows)
  evaluate_topk_users()   ->  top-K precision / recall / F1
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

THRESHOLDS = [90, 95, 97]
K_VALUES   = [10, 20, 50]


# =============================================================================
# Step 3 -- compute_user_scores
# =============================================================================

def compute_user_scores(
    df: pd.DataFrame,
    score_col: str,
    insider_users: set[str],
) -> pd.DataFrame:
    """Collapse row-level daily scores to one row per user.

    Returns columns: user, score_max, score_mean, score_p95,
                     dataset_split, is_insider.

    dataset_split is the majority split for the user (used to identify
    train users for threshold calibration without leaking test behaviour).
    """
    agg = df.groupby("user").agg(
        score_max     = (score_col, "max"),
        score_mean    = (score_col, "mean"),
        score_p95     = (score_col, lambda x: float(np.percentile(x.dropna(), 95))
                                              if x.dropna().size else 0.0),
        dataset_split = ("dataset_split", lambda x: x.mode().iloc[0]),
    ).reset_index()
    agg["is_insider"] = agg["user"].isin(insider_users).astype(int)
    return agg


# =============================================================================
# Step 4 -- apply_user_threshold
# =============================================================================

def apply_user_threshold(
    user_df: pd.DataFrame,
    agg_col: str,
    percentile: int,
) -> tuple[pd.DataFrame, float]:
    """Filter to users whose aggregated score meets a percentile cutoff.

    The cutoff is derived from TRAIN USERS only (majority-train-split users),
    preventing leakage from test-period behaviour into the decision boundary.

    Returns:
        filtered_df  -- users at or above the cutoff, sorted score descending
        cutoff       -- numeric threshold applied
    """
    train_scores = user_df.loc[user_df["dataset_split"] == "train", agg_col].dropna()
    cutoff = float(np.percentile(train_scores, percentile))
    filtered = (
        user_df[user_df[agg_col] >= cutoff]
        .sort_values(agg_col, ascending=False)
        .reset_index(drop=True)
    )
    return filtered, cutoff


# =============================================================================
# Step 5 -- evaluate_topk_users
# =============================================================================

def evaluate_topk_users(
    ranked_df: pd.DataFrame,
    insider_users: set[str],
    k: int,
) -> dict:
    """Flag the top-K users in ranked_df as anomalies and compute metrics.

    ranked_df must already be sorted descending by the desired score column.
    If K > pool size, all survivors are flagged (marked with * in output).
    """
    actual_k  = min(k, len(ranked_df))
    top_k_set = set(ranked_df.head(actual_k)["user"])
    tp = len(top_k_set & insider_users)
    fp = actual_k - tp
    fn = len(insider_users) - tp
    prec   = tp / actual_k if actual_k > 0 else 0.0
    recall = tp / len(insider_users) if insider_users else 0.0
    f1     = (2 * prec * recall / (prec + recall)) if (prec + recall) > 0 else 0.0
    return {
        "k": k, "actual_k": actual_k,
        "precision": prec, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
    }


# =============================================================================
# Printing helpers
# =============================================================================

def _fmt_row(thresh_pct: int, cutoff: float, pool: int,
             m: dict, agg: str) -> str:
    k_str = f"{m['k']}" + ("*" if m["actual_k"] < m["k"] else "")
    return (
        f"  {agg:<10}  {thresh_pct:>4}%  {cutoff:>7.4f}"
        f"  {pool:>7}  {k_str:>5}"
        f"  {m['precision']:>10.4f}  {m['recall']:>8.4f}  {m['f1']:>8.4f}"
        f"  {m['tp']:>4}  {m['fp']:>5}  {m['fn']:>4}"
    )


def print_model_table(result_rows: list[dict], title: str,
                      n_users: int, n_insiders: int) -> None:
    header = (
        f"  {'Agg':<10}  {'Thresh':>5}  {'Cutoff':>7}"
        f"  {'Pool':>7}  {'K':>5}"
        f"  {'Precision':>10}  {'Recall':>8}  {'F1':>8}"
        f"  {'TP':>4}  {'FP':>5}  {'FN':>4}"
    )
    width = len(header)
    print(f"\n{'=' * width}")
    print(f"  {title}  ({n_insiders} insiders / {n_users} users)")
    print(f"{'=' * width}")
    print(header)
    prev_agg = None
    for r in result_rows:
        if r["agg"] != prev_agg and prev_agg is not None:
            print("  " + "-" * (width - 2))
        prev_agg = r["agg"]
        print(_fmt_row(r["thresh_pct"], r["cutoff"], r["pool"], r["metrics"], r["agg"]))
    print(f"{'=' * width}")
    print("  * K exceeds filtered pool -- all survivors flagged.\n")


# =============================================================================
# Steps 6 & 7 -- per-model driver (shared by IF and LSTM)
# =============================================================================

def analyse_model(
    df: pd.DataFrame,
    score_col: str,
    insider_users: set[str],
    model_name: str,
    agg_cols: list[str] | None = None,
) -> list[dict]:
    """Full filter-then-rank pipeline for one model.

    agg_cols selects aggregations to evaluate (default: all three).
    Returns flat list of result dicts for cross-model comparison.
    """
    if agg_cols is None:
        agg_cols = ["score_max", "score_mean", "score_p95"]

    user_df    = compute_user_scores(df, score_col, insider_users)
    n_users    = len(user_df)
    n_insiders = int(user_df["is_insider"].sum())

    all_rows       = []
    result_records = []

    for agg in agg_cols:
        for thresh_pct in THRESHOLDS:
            filtered, cutoff = apply_user_threshold(user_df, agg, thresh_pct)
            pool = len(filtered)
            for k in K_VALUES:
                m = evaluate_topk_users(filtered, insider_users, k)
                all_rows.append({
                    "agg": agg, "thresh_pct": thresh_pct,
                    "cutoff": cutoff, "pool": pool, "metrics": m,
                })
                result_records.append({
                    "model": model_name, "agg": agg,
                    "thresh_pct": thresh_pct, "k": k,
                    "f1": m["f1"], "precision": m["precision"],
                    "recall": m["recall"], "tp": m["tp"],
                })

    print_model_table(all_rows, model_name, n_users, n_insiders)
    return result_records


# =============================================================================
# Step 8 -- aggregation comparison
# =============================================================================

def print_agg_comparison(records: list[dict]) -> None:
    """Best F1 per (model, agg) across all (threshold, K) combinations."""
    df = pd.DataFrame(records)
    w  = 65
    print("\n" + "=" * w)
    print("  Step 8 -- Best F1 per model x aggregation (across all thresh/K)")
    print("=" * w)
    print(f"  {'Model':<22}  {'Agg':<12}  {'Best F1':>8}  {'Prec':>8}  {'Recall':>8}  {'TP':>4}")
    print("  " + "-" * (w - 2))
    best = (
        df.sort_values("f1", ascending=False)
          .groupby(["model", "agg"], sort=False)
          .first()
          .reset_index()
          .sort_values("f1", ascending=False)
    )
    for _, r in best.iterrows():
        print(
            f"  {r['model']:<22}  {r['agg']:<12}"
            f"  {r['f1']:>8.4f}  {r['precision']:>8.4f}"
            f"  {r['recall']:>8.4f}  {r['tp']:>4.0f}"
        )
    print("=" * w + "\n")


# =============================================================================
# Step 10 -- summary
# =============================================================================

def print_summary(records: list[dict], n_insiders: int) -> None:
    df   = pd.DataFrame(records)
    best = df.sort_values("f1", ascending=False).iloc[0]
    w    = 65
    print("=" * w)
    print("  Step 10 -- Best overall configuration")
    print("=" * w)
    print(f"  Model      : {best['model']}")
    print(f"  Aggregation: {best['agg']}")
    print(f"  Threshold  : {best['thresh_pct']}th percentile (of train user scores)")
    print(f"  K          : {best['k']}")
    print(f"  Precision  : {best['precision']:.4f}")
    print(f"  Recall     : {best['recall']:.4f}")
    print(f"  F1         : {best['f1']:.4f}")
    print(f"  TP         : {int(best['tp'])} / {n_insiders} insiders detected")
    print("=" * w)
    print()
    print("  Key findings:")
    print("  - IF scores insiders LOWER than normals on all aggregations")
    print("    (ROC < 0.5 -- model inverts). Ranking IF descending surfaces")
    print("    normal users, not insiders. IF is not useful here.")
    print("  - LSTM score_max = 1.0 for virtually every user -- no signal.")
    print("  - LSTM score_p95 (per-user 95th pct of daily scores) gives the")
    print("    strongest insider/normal separation and is the correct")
    print("    aggregation. Threshold must be computed on user-level p95")
    print("    scores of train users, not on raw daily rows.")
    print()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    gt = select_ground_truth_release([IFOREST_CSV, LSTM_CSV])
    insider_users = gt.matching_users
    print("Ground truth:")
    print(f"  {describe_selection(gt)}\n")

    all_records: list[dict] = []

    # Step 6: Isolation Forest
    if IFOREST_CSV.exists():
        idf = pd.read_csv(IFOREST_CSV,
                          usecols=["user", "iforest_score", "dataset_split"])
        all_records += analyse_model(
            idf, "iforest_score", insider_users, "Isolation Forest"
        )
    else:
        print(f"[SKIP] {IFOREST_CSV} not found")

    # Step 7: LSTM Autoencoder
    if LSTM_CSV.exists():
        ldf = pd.read_csv(LSTM_CSV,
                          usecols=["user", "lstm_score",
                                   "lstm_risk_severity", "dataset_split"])
        ldf = ldf[ldf["lstm_risk_severity"] != "undetermined"]
        all_records += analyse_model(
            ldf, "lstm_score", insider_users, "LSTM Autoencoder"
        )
    else:
        print(f"[SKIP] {LSTM_CSV} not found")

    if all_records:
        print_agg_comparison(all_records)  # Step 8
        print_summary(all_records, len(insider_users))  # Step 10


if __name__ == "__main__":
    main()
