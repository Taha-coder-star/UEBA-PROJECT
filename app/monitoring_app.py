"""Streamlit monitoring app for CERT email insider threat detection."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "colab") not in sys.path:
    sys.path.insert(0, str(ROOT / "colab"))

from inference_isolation_forest_cert import load_artifacts as load_iforest_artifacts, score_single_row

IFOREST_SCORED_PATH = ROOT / "cleaned" / "email_user_daily_scored.csv"
IFOREST_SUMMARY_PATH = ROOT / "models" / "isolation_forest_summary.json"
LSTM_SCORED_PATH = ROOT / "cleaned" / "email_user_daily_lstm_scored.csv"
LSTM_SUMMARY_PATH = ROOT / "models" / "lstm_autoencoder_summary.json"

FEATURE_COLUMNS = [
    "email_count", "unique_pcs", "total_size", "avg_size",
    "total_attachments", "emails_with_attachments", "after_hours_emails",
    "avg_recipients", "max_recipients", "avg_content_words", "max_content_words",
    "bcc_email_count", "cc_email_count", "attachment_email_ratio",
    "after_hours_ratio", "bcc_ratio", "o", "c", "e", "a", "n",
]


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_data
def load_iforest_scored() -> pd.DataFrame:
    df = pd.read_csv(IFOREST_SCORED_PATH)
    df["email_day"] = pd.to_datetime(df["email_day"], errors="coerce")
    return df.sort_values(["email_day", "iforest_score"], ascending=[True, False]).reset_index(drop=True)


@st.cache_data
def load_iforest_summary() -> dict:
    return json.loads(IFOREST_SUMMARY_PATH.read_text(encoding="utf-8"))


@st.cache_resource
def get_iforest_artifacts() -> dict:
    return load_iforest_artifacts()


@st.cache_data
def load_lstm_scored() -> pd.DataFrame:
    df = pd.read_csv(LSTM_SCORED_PATH)
    df["email_day"] = pd.to_datetime(df["email_day"], errors="coerce")
    return df.sort_values(["email_day", "lstm_score"], ascending=[True, False]).reset_index(drop=True)


@st.cache_data
def load_lstm_summary() -> dict:
    return json.loads(LSTM_SUMMARY_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------

def render_iforest_overview(df: pd.DataFrame, summary: dict) -> None:
    st.subheader("Model Overview — Isolation Forest")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Users", summary["users"])
    col2.metric("Rows", summary["rows"])
    col3.metric("Suspicious Rows", summary["suspicious_rows"])
    col4.metric("High Rows", summary["high_rows"])

    st.write(f"Training window: **{summary['train_start_day']}** to **{summary['train_end_day']}**")
    st.write(f"Monitoring window: **{summary['test_start_day']}** to **{summary['test_end_day']}**")
    st.write(
        f"Thresholds: suspicious >= **{summary['suspicious_threshold']:.3f}**, "
        f"high >= **{summary['high_threshold']:.3f}**"
    )

    st.subheader("Top 20 Anomalies")
    top = df.sort_values("iforest_score", ascending=False).head(20)
    st.dataframe(
        top[["user", "email_day", "dataset_split", "iforest_score", "risk_severity",
             "email_count", "after_hours_emails", "total_size"]],
        use_container_width=True,
    )

    user_scores = (
        df.groupby("user", as_index=False)["iforest_score"]
        .mean()
        .sort_values("iforest_score", ascending=False)
        .head(15)
    )
    st.subheader("Top 15 Users by Average Isolation Forest Score")
    st.bar_chart(user_scores.set_index("user"))


def render_iforest_replay(df: pd.DataFrame) -> None:
    st.subheader("Live Replay — Isolation Forest")
    unique_days = sorted(df["email_day"].dropna().dt.strftime("%Y-%m-%d").unique().tolist())
    selected_day = st.select_slider(
        "Replay day",
        options=unique_days,
        value=unique_days[min(len(unique_days) - 1, len(unique_days) // 2)],
    )
    day_df = df[df["email_day"].dt.strftime("%Y-%m-%d") == selected_day]

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows on Day", len(day_df))
    c2.metric("Suspicious Users", day_df.loc[day_df["risk_severity"] != "normal", "user"].nunique())
    c3.metric("Highest Score", f"{day_df['iforest_score'].max():.3f}")

    st.dataframe(
        day_df[["user", "iforest_score", "risk_severity", "email_count",
                "after_hours_emails", "total_size"]].head(50),
        use_container_width=True,
    )


def render_manual_scoring(artifacts: dict) -> None:
    st.subheader("Manual Feature Scoring — Isolation Forest")
    st.write("Score a new daily feature row against the saved Isolation Forest model.")
    defaults = {
        "email_count": 15.0, "unique_pcs": 1.0, "total_size": 400000.0,
        "avg_size": 25000.0, "total_attachments": 2.0, "emails_with_attachments": 1.0,
        "after_hours_emails": 0.0, "avg_recipients": 2.0, "max_recipients": 4.0,
        "avg_content_words": 55.0, "max_content_words": 90.0, "bcc_email_count": 0.0,
        "cc_email_count": 4.0, "attachment_email_ratio": 0.10, "after_hours_ratio": 0.00,
        "bcc_ratio": 0.00, "o": 30.0, "c": 30.0, "e": 30.0, "a": 30.0, "n": 30.0,
    }
    row = {}
    cols = st.columns(3)
    for i, feat in enumerate(FEATURE_COLUMNS):
        row[feat] = cols[i % 3].number_input(feat, value=float(defaults[feat]))

    if st.button("Score This Row"):
        result = score_single_row(row, artifacts=artifacts)
        st.json({
            "iforest_score": round(float(result["iforest_score"]), 4),
            "iforest_flag": int(result["iforest_flag"]),
            "risk_severity": result["risk_severity"],
        })


def render_lstm_overview(df: pd.DataFrame, summary: dict) -> None:
    st.subheader("Model Overview — LSTM Autoencoder")
    st.info(
        "Each user has their own LSTM autoencoder trained on their first 80% of activity days. "
        "Anomaly score = normalized reconstruction error on a 7-day sliding window."
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Users", summary["users"])
    col2.metric("Models Trained", summary["trained_users"])
    col3.metric("Skipped (low data)", summary["skipped_users"])
    col4.metric("Suspicious Rows", summary["suspicious_rows"])
    col5.metric("High Rows", summary["high_rows"])

    st.write(
        f"Window size: **{summary['window_size']} days** | "
        f"Hidden dim: **{summary['hidden_dim']}** | "
        f"Latent dim: **{summary['latent_dim']}**"
    )

    valid = df.dropna(subset=["lstm_score"])

    st.subheader("Top 20 Anomalies — LSTM")
    top = valid.sort_values("lstm_score", ascending=False).head(20)
    st.dataframe(
        top[["user", "email_day", "dataset_split", "lstm_score", "lstm_risk_severity",
             "email_count", "after_hours_emails", "bcc_ratio"]],
        use_container_width=True,
    )

    user_scores = (
        valid.groupby("user", as_index=False)["lstm_score"]
        .mean()
        .sort_values("lstm_score", ascending=False)
        .head(15)
    )
    st.subheader("Top 15 Users by Mean LSTM Anomaly Score")
    st.bar_chart(user_scores.set_index("user"))


def render_lstm_user_timeline(df: pd.DataFrame) -> None:
    st.subheader("Per-User Anomaly Timeline — LSTM")
    valid = df.dropna(subset=["lstm_score"])
    users = sorted(valid["user"].unique().tolist())
    selected_user = st.selectbox("Select user", users)

    user_df = valid[valid["user"] == selected_user].sort_values("email_day")

    c1, c2, c3 = st.columns(3)
    c1.metric("Active Days", len(user_df))
    c2.metric("Peak LSTM Score", f"{user_df['lstm_score'].max():.3f}")
    high_days = (user_df["lstm_risk_severity"] == "high").sum()
    c3.metric("High-Risk Days", int(high_days))

    st.line_chart(user_df.set_index("email_day")["lstm_score"])

    st.write("All flagged days for this user:")
    flagged = user_df[user_df["lstm_flag"] == 1].sort_values("lstm_score", ascending=False)
    if flagged.empty:
        st.success("No anomalous days detected for this user.")
    else:
        st.dataframe(
            flagged[["email_day", "lstm_score", "lstm_risk_severity",
                     "email_count", "after_hours_emails", "bcc_ratio", "total_size"]],
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="CERT Email Monitoring", layout="wide")
    st.title("CERT Email Insider Threat Monitoring")

    iforest_ready = IFOREST_SCORED_PATH.exists() and IFOREST_SUMMARY_PATH.exists()
    lstm_ready = LSTM_SCORED_PATH.exists() and LSTM_SUMMARY_PATH.exists()

    if not iforest_ready:
        st.error("Run the Isolation Forest training script first: `python scripts/run_local_pipeline.py --skip-lstm`")
        st.stop()

    iforest_df = load_iforest_scored()
    iforest_summary = load_iforest_summary()
    iforest_artifacts = get_iforest_artifacts()

    tabs = st.tabs(["IForest Overview", "IForest Replay", "Manual Scoring", "LSTM Overview", "LSTM User Timeline"])

    with tabs[0]:
        render_iforest_overview(iforest_df, iforest_summary)

    with tabs[1]:
        render_iforest_replay(iforest_df)

    with tabs[2]:
        render_manual_scoring(iforest_artifacts)

    with tabs[3]:
        if not lstm_ready:
            st.warning(
                "LSTM model outputs not found. "
                "Run: `python scripts/run_local_pipeline.py --lstm-only`"
            )
        else:
            lstm_df = load_lstm_scored()
            lstm_summary = load_lstm_summary()
            render_lstm_overview(lstm_df, lstm_summary)

    with tabs[4]:
        if not lstm_ready:
            st.warning(
                "LSTM model outputs not found. "
                "Run: `python scripts/run_local_pipeline.py --lstm-only`"
            )
        else:
            lstm_df = load_lstm_scored()
            render_lstm_user_timeline(lstm_df)


if __name__ == "__main__":
    main()
