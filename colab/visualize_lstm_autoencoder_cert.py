"""Visualize LSTM autoencoder anomaly detection results on CERT daily email features."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CLEANED_DIR, PLOTS_DIR  # noqa: E402

INPUT_PATH = CLEANED_DIR / "email_user_daily_lstm_scored.csv"


def save_plot(fig: plt.Figure, name: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    sns.set_theme(style="whitegrid")
    df = pd.read_csv(INPUT_PATH)
    df["email_day"] = pd.to_datetime(df["email_day"], errors="coerce")

    valid = df.dropna(subset=["lstm_score"])

    # --- Score distribution ---
    suspicious_min = valid.loc[valid["lstm_risk_severity"] == "suspicious", "lstm_score"].min() if (valid["lstm_risk_severity"] == "suspicious").any() else None
    high_min = valid.loc[valid["lstm_risk_severity"] == "high", "lstm_score"].min() if (valid["lstm_risk_severity"] == "high").any() else None

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(valid["lstm_score"], bins=60, kde=True, ax=ax, color="#1f77b4")
    if suspicious_min is not None:
        ax.axvline(suspicious_min, color="#ff7f0e", linestyle="--", label=f"Suspicious threshold ({suspicious_min:.3f})")
    if high_min is not None:
        ax.axvline(high_min, color="#d62728", linestyle="--", label=f"High threshold ({high_min:.3f})")
    ax.set_title("LSTM Autoencoder Anomaly Score Distribution")
    ax.set_xlabel("Normalized reconstruction error (anomaly score)")
    ax.legend()
    save_plot(fig, "lstm_score_distribution.png")

    # --- Top users by mean anomaly score ---
    top_users = (
        valid.groupby("user", as_index=False)["lstm_score"]
        .mean()
        .sort_values("lstm_score", ascending=False)
        .head(15)
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=top_users, x="lstm_score", y="user", ax=ax, color="#d62728")
    ax.set_title("Top 15 Users by Mean LSTM Anomaly Score")
    ax.set_xlabel("Mean anomaly score")
    ax.set_ylabel("User")
    save_plot(fig, "lstm_top_users_by_score.png")

    # --- Daily max anomaly score over time ---
    daily_max = (
        valid.groupby("email_day", as_index=False)["lstm_score"]
        .max()
        .sort_values("email_day")
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=daily_max, x="email_day", y="lstm_score", ax=ax, color="#1f77b4")
    ax.set_title("Maximum Daily LSTM Anomaly Score Over Time")
    ax.set_xlabel("Day")
    ax.set_ylabel("Max anomaly score")
    ax.tick_params(axis="x", rotation=45)
    save_plot(fig, "lstm_daily_max_score.png")

    # --- Email count vs after-hours coloured by LSTM severity ---
    plot_df = valid.sample(min(15000, len(valid)), random_state=42)
    severity_order = [s for s in ["normal", "suspicious", "high"] if s in plot_df["lstm_risk_severity"].unique()]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=plot_df,
        x="email_count",
        y="after_hours_emails",
        hue="lstm_risk_severity",
        hue_order=severity_order,
        palette={"normal": "#4daf4a", "suspicious": "#ff7f0e", "high": "#e41a1c"},
        alpha=0.5,
        ax=ax,
    )
    ax.set_title("Email Count vs After-Hours Emails (LSTM Severity)")
    ax.set_xlabel("Daily email count")
    ax.set_ylabel("After-hours emails")
    save_plot(fig, "lstm_email_count_vs_after_hours.png")

    # --- Per-user anomaly score timeline for top 5 users ---
    top5 = top_users.head(5)["user"].tolist()
    top5_df = valid[valid["user"].isin(top5)].sort_values("email_day")
    fig, ax = plt.subplots(figsize=(13, 5))
    for user in top5:
        user_data = top5_df[top5_df["user"] == user]
        ax.plot(user_data["email_day"], user_data["lstm_score"], label=user, alpha=0.8, linewidth=1.2)
    ax.set_title("LSTM Anomaly Score Timeline — Top 5 Users")
    ax.set_xlabel("Day")
    ax.set_ylabel("Anomaly score")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=45)
    save_plot(fig, "lstm_top5_user_timelines.png")

    # --- Top 20 anomalies CSV ---
    top_anomalies = valid.sort_values("lstm_score", ascending=False).head(20)
    top_anomalies.to_csv(PLOTS_DIR / "lstm_top_20_anomalies.csv", index=False)

    print("Saved LSTM plots to:", PLOTS_DIR)
    print(
        top_anomalies[
            ["user", "email_day", "dataset_split", "lstm_score", "lstm_risk_severity",
             "email_count", "after_hours_emails", "total_size"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
