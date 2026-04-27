"""Presentation-ready visualizations for the user-level evaluation pipeline.

Generates 6 plots and saves them to plots/user_level/.
Run standalone:  python colab/visualize_user_level.py

Plots produced:
  1. pr_f1_vs_k.png            -- Precision / Recall / F1 vs K (LSTM p95)
  2. aggregation_comparison.png -- Best F1 by aggregation method
  3. model_comparison.png       -- IF vs LSTM side-by-side at K = 20 / 50
  4. score_distribution.png     -- User score_p95 histogram: insider vs normal
  5. top_users_risk.png         -- Top 20 suspicious users by risk score
  6. tp_fp_fn_summary.png       -- TP / FP / FN bars across K values
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

REPO_DIR = Path(os.environ.get("DLP_REPO", str(Path(__file__).resolve().parent.parent)))
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "colab"))
from config import CLEANED_DIR  # noqa: E402
from user_level_eval import (  # noqa: E402
    compute_user_scores, apply_user_threshold, evaluate_topk_users,
)
from risk_scorer import (  # noqa: E402
    compute_behavioral_signals, compute_risk_scores,
)
from ground_truth import describe_selection, select_ground_truth_release  # noqa: E402

IFOREST_CSV  = CLEANED_DIR / "email_user_daily_scored.csv"
LSTM_CSV     = CLEANED_DIR / "email_user_daily_lstm_scored.csv"
OUT_DIR      = REPO_DIR / "plots" / "user_level"

INSIDER_COLOR = "#E85454"
NORMAL_COLOR  = "#4C9BE8"
IF_COLOR      = "#F4A83A"
LSTM_COLOR    = "#5CB85C"
K_VALUES      = [10, 20, 30, 50, 75, 100]
THRESHOLD_PCT = 90   # fixed for K-sweep plots
STYLE         = {"fontsize": 11, "fontweight": "bold"}


# ---------------------------------------------------------------------------
# Data helpers (cached in module scope after first call)
# ---------------------------------------------------------------------------

_cache: dict = {}


def _load_all() -> None:
    if _cache:
        return
    gt = select_ground_truth_release([IFOREST_CSV, LSTM_CSV])
    _cache["ground_truth"] = gt
    _cache["insider_users"] = gt.matching_users
    print(f"Ground truth: {describe_selection(gt)}")

    ldf = pd.read_csv(LSTM_CSV,
                      usecols=["user", "lstm_score", "lstm_risk_severity", "dataset_split"])
    ldf = ldf[ldf["lstm_risk_severity"] != "undetermined"]
    _cache["lstm_user"] = compute_user_scores(ldf, "lstm_score", _cache["insider_users"])

    idf_full = pd.read_csv(IFOREST_CSV)
    _cache["idf_full"] = idf_full
    _cache["if_user"]  = compute_user_scores(
        idf_full[["user", "iforest_score", "dataset_split"]],
        "iforest_score", _cache["insider_users"]
    )

    behav = compute_behavioral_signals(idf_full)
    _cache["risk_df"] = compute_risk_scores(
        _cache["lstm_user"], behav, _cache["insider_users"]
    )


def _metrics_for_k(user_df: pd.DataFrame, agg: str, thresh_pct: int,
                   insider_users: set, k: int) -> dict:
    filtered, _ = apply_user_threshold(user_df, agg, thresh_pct)
    return evaluate_topk_users(filtered, insider_users, k)


# ---------------------------------------------------------------------------
# Plot 1 -- Precision / Recall / F1 vs K
# ---------------------------------------------------------------------------

def plot_prf1_vs_k() -> Path:
    _load_all()
    iu    = _cache["insider_users"]
    lu    = _cache["lstm_user"]

    rows = []
    for k in K_VALUES:
        m = _metrics_for_k(lu, "score_p95", THRESHOLD_PCT, iu, k)
        rows.append({"K": k, "Precision": m["precision"],
                     "Recall": m["recall"], "F1": m["f1"]})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df["K"], df["Precision"], "o-", color="#E85454", linewidth=2, label="Precision")
    ax.plot(df["K"], df["Recall"],    "s-", color="#4C9BE8", linewidth=2, label="Recall")
    ax.plot(df["K"], df["F1"],        "^-", color="#5CB85C", linewidth=2.5, label="F1")
    ax.set_xlabel("K  (number of top users flagged)", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Precision / Recall / F1 vs K\n"
                 f"LSTM Autoencoder  |  score_p95 aggregation  |  {THRESHOLD_PCT}th pct threshold",
                 **STYLE)
    ax.set_xticks(K_VALUES)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    out = OUT_DIR / "pr_f1_vs_k.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Plot 2 -- Aggregation method comparison (best F1 per agg at K=50)
# ---------------------------------------------------------------------------

def plot_aggregation_comparison() -> Path:
    _load_all()
    iu = _cache["insider_users"]
    lu = _cache["lstm_user"]

    aggs   = ["score_max", "score_mean", "score_p95"]
    labels = ["Max", "Mean", "P95"]
    colors = ["#F4A83A", "#4C9BE8", "#5CB85C"]
    k      = 50

    prec_vals, rec_vals, f1_vals = [], [], []
    for agg in aggs:
        m = _metrics_for_k(lu, agg, THRESHOLD_PCT, iu, k)
        prec_vals.append(m["precision"])
        rec_vals.append(m["recall"])
        f1_vals.append(m["f1"])

    x   = np.arange(len(aggs))
    w   = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w, prec_vals, w, label="Precision", color="#E85454", alpha=0.85)
    ax.bar(x,     rec_vals,  w, label="Recall",    color="#4C9BE8", alpha=0.85)
    ax.bar(x + w, f1_vals,   w, label="F1",        color="#5CB85C", alpha=0.85)

    for i, (p, r, f) in enumerate(zip(prec_vals, rec_vals, f1_vals)):
        ax.text(i - w, p + 0.01, f"{p:.2f}", ha="center", fontsize=8)
        ax.text(i,     r + 0.01, f"{r:.2f}", ha="center", fontsize=8)
        ax.text(i + w, f + 0.01, f"{f:.2f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlabel("User Score Aggregation Method", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(f"Aggregation Method Comparison  |  LSTM  |  K={k}  |  {THRESHOLD_PCT}th pct threshold",
                 **STYLE)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    out = OUT_DIR / "aggregation_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Plot 3 -- IF vs LSTM comparison at K = 20 and K = 50
# ---------------------------------------------------------------------------

def plot_model_comparison() -> Path:
    _load_all()
    iu = _cache["insider_users"]
    lu = _cache["lstm_user"]
    iu_if = _cache["if_user"]

    k_vals  = [20, 50]
    metrics = ["precision", "recall", "f1"]
    mlabels = ["Precision", "Recall", "F1"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, k in zip(axes, k_vals):
        m_if   = _metrics_for_k(iu_if, "score_max",  THRESHOLD_PCT, iu, k)
        m_lstm = _metrics_for_k(lu,    "score_p95",  THRESHOLD_PCT, iu, k)

        x = np.arange(len(metrics))
        w = 0.35
        b1 = ax.bar(x - w/2, [m_if[m]   for m in metrics], w,
                    label="Isolation Forest", color=IF_COLOR,   alpha=0.85)
        b2 = ax.bar(x + w/2, [m_lstm[m] for m in metrics], w,
                    label="LSTM Autoencoder",  color=LSTM_COLOR, alpha=0.85)

        for bar in [*b1, *b2]:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f"{h:.2f}", ha="center", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(mlabels, fontsize=10)
        ax.set_title(f"K = {k}", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 0.7)
        ax.grid(axis="y", alpha=0.3)
        if k == k_vals[0]:
            ax.set_ylabel("Score", fontsize=11)
            ax.legend(fontsize=9)

    fig.suptitle(f"Isolation Forest vs LSTM Autoencoder  |  {THRESHOLD_PCT}th pct threshold",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "model_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Plot 4 -- User score_p95 distribution: insider vs normal
# ---------------------------------------------------------------------------

def plot_score_distribution() -> Path:
    _load_all()
    lu = _cache["lstm_user"]

    insiders = lu[lu["is_insider"] == 1]["score_p95"].dropna()
    normals  = lu[lu["is_insider"] == 0]["score_p95"].dropna()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(normals,  bins=40, alpha=0.6, color=NORMAL_COLOR,
            label=f"Normal users  (n={len(normals):,})", density=True)
    ax.hist(insiders, bins=40, alpha=0.75, color=INSIDER_COLOR,
            label=f"Insider users (n={len(insiders):,})", density=True)

    ax.axvline(normals.mean(),  color=NORMAL_COLOR,  linestyle="--", linewidth=1.5,
               label=f"Normal mean = {normals.mean():.3f}")
    ax.axvline(insiders.mean(), color=INSIDER_COLOR, linestyle="--", linewidth=1.5,
               label=f"Insider mean = {insiders.mean():.3f}")

    ax.set_xlabel("LSTM User Score (P95 of daily scores)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Distribution of User Anomaly Scores: Insiders vs Normal Users\n"
                 "LSTM Autoencoder  |  score_p95 aggregation", **STYLE)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "score_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Plot 5 -- Top 20 suspicious users by risk score
# ---------------------------------------------------------------------------

def plot_top_users_risk() -> Path:
    _load_all()
    risk_df = _cache["risk_df"]

    top20 = risk_df.head(20).copy()
    top20["label"] = top20["user"].apply(
        lambda u: f"* {u}" if u in _cache["insider_users"] else u
    )
    colors = [INSIDER_COLOR if u in _cache["insider_users"] else NORMAL_COLOR
              for u in top20["user"]]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(top20["label"][::-1], top20["risk_score"][::-1],
                   color=colors[::-1], alpha=0.85)
    for bar, val in zip(bars, top20["risk_score"][::-1]):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=8)

    ax.set_xlabel("Weighted Risk Score", fontsize=11)
    ax.set_title("Top 20 Suspicious Users by Risk Score\n"
                 f"* = confirmed CERT {_cache['ground_truth'].dataset} insider", **STYLE)
    ax.set_xlim(0, top20["risk_score"].max() + 0.07)
    ax.grid(axis="x", alpha=0.3)

    insider_patch = mpatches.Patch(color=INSIDER_COLOR, label="Confirmed insider")
    normal_patch  = mpatches.Patch(color=NORMAL_COLOR,  label="Normal user")
    ax.legend(handles=[insider_patch, normal_patch], fontsize=9)
    fig.tight_layout()
    out = OUT_DIR / "top_users_risk.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Plot 6 -- TP / FP / FN summary across K values
# ---------------------------------------------------------------------------

def plot_tp_fp_fn_summary() -> Path:
    _load_all()
    iu = _cache["insider_users"]
    lu = _cache["lstm_user"]

    k_plot = [10, 20, 50, 100]
    tps, fps, fns = [], [], []
    for k in k_plot:
        m = _metrics_for_k(lu, "score_p95", THRESHOLD_PCT, iu, k)
        tps.append(m["tp"])
        fps.append(m["fp"])
        fns.append(m["fn"])

    x = np.arange(len(k_plot))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar(x - w, tps, w, label="TP (insiders caught)",  color="#5CB85C", alpha=0.85)
    b2 = ax.bar(x,     fps, w, label="FP (false alarms)",     color="#E85454", alpha=0.85)
    b3 = ax.bar(x + w, fns, w, label="FN (missed insiders)",  color="#AAAAAA", alpha=0.85)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                    str(int(h)), ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in k_plot], fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Detection Outcomes at Different K Values\n"
                 f"LSTM Autoencoder  |  score_p95  |  {THRESHOLD_PCT}th pct threshold",
                 **STYLE)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = OUT_DIR / "tp_fp_fn_summary.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _load_all()
    print(f"Saving plots to {OUT_DIR}\n")
    plot_prf1_vs_k()
    plot_aggregation_comparison()
    plot_model_comparison()
    plot_score_distribution()
    plot_top_users_risk()
    plot_tp_fp_fn_summary()
    print("\nAll 6 plots saved.")


if __name__ == "__main__":
    main()
