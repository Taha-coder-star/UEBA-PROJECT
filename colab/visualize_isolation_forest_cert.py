"""Visualize app-ready Isolation Forest results on cleaned CERT daily email features."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CLEANED_DIR, PLOTS_DIR  # noqa: E402

INPUT_PATH = CLEANED_DIR / "email_user_daily_scored.csv"


def save_plot(fig, name: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    sns.set_theme(style="whitegrid")
    df = pd.read_csv(INPUT_PATH)

    suspicious_threshold = df.loc[df["risk_severity"] != "normal", "iforest_score"].min()
    high_threshold = df.loc[df["risk_severity"] == "high", "iforest_score"].min() if (df["risk_severity"] == "high").any() else None

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df["iforest_score"], bins=50, kde=True, ax=ax, color="#1f77b4")
    if pd.notna(suspicious_threshold):
        ax.axvline(suspicious_threshold, color="#ff7f0e", linestyle="--", label="Suspicious threshold")
    if high_threshold is not None and pd.notna(high_threshold):
        ax.axvline(high_threshold, color="#d62728", linestyle="--", label="High threshold")
    ax.set_title("Isolation Forest Anomaly Score Distribution")
    ax.set_xlabel("Normalized anomaly score")
    ax.legend()
    save_plot(fig, "iforest_score_distribution.png")

    top_users = (
        df.groupby("user", as_index=False)["iforest_score"]
        .mean()
        .sort_values("iforest_score", ascending=False)
        .head(15)
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=top_users, x="iforest_score", y="user", ax=ax, color="#d62728")
    ax.set_title("Top 15 Users by Average Anomaly Score")
    ax.set_xlabel("Average anomaly score")
    ax.set_ylabel("User")
    save_plot(fig, "top_users_by_anomaly_score.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=df.sample(min(15000, len(df)), random_state=42),
        x="email_count",
        y="after_hours_emails",
        hue="risk_severity",
        hue_order=["normal", "suspicious", "high"],
        palette={"normal": "#4daf4a", "suspicious": "#ff7f0e", "high": "#e41a1c"},
        alpha=0.6,
        ax=ax,
    )
    ax.set_title("Email Count vs After-Hours Emails")
    ax.set_xlabel("Daily email count")
    ax.set_ylabel("After-hours emails")
    save_plot(fig, "email_count_vs_after_hours.png")

    daily_risk = (
        df.groupby("email_day", as_index=False)["iforest_score"]
        .max()
        .sort_values("email_day")
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=daily_risk, x="email_day", y="iforest_score", ax=ax, color="#1f77b4")
    ax.set_title("Maximum Daily Anomaly Score Over Time")
    ax.set_xlabel("Day")
    ax.set_ylabel("Max anomaly score")
    ax.tick_params(axis="x", rotation=45)
    save_plot(fig, "daily_max_anomaly_score.png")

    top_anomalies = df.sort_values("iforest_score", ascending=False).head(20)
    top_anomalies.to_csv(PLOTS_DIR / "top_20_anomalies.csv", index=False)

    print("Saved plots to:", PLOTS_DIR)
    print(top_anomalies[["user", "email_day", "dataset_split", "iforest_score", "risk_severity", "email_count", "after_hours_emails", "total_size"]].to_string(index=False))


if __name__ == "__main__":
    main()
