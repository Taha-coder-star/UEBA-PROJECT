"""Streamlit monitoring app — CERT insider threat detection pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

_CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_ROOT))
if str(_CODE_ROOT / "colab") not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT / "colab"))

from config import CLEANED_DIR, MODELS_DIR  # noqa: E402
from inference_isolation_forest_cert import (  # noqa: E402
    load_artifacts as load_iforest_artifacts,
    score_single_row,
)

IFOREST_SCORED_PATH  = CLEANED_DIR / "email_user_daily_scored.csv"
IFOREST_SUMMARY_PATH = MODELS_DIR  / "isolation_forest_summary.json"
LSTM_SCORED_PATH     = CLEANED_DIR / "email_user_daily_lstm_scored.csv"
LSTM_SUMMARY_PATH    = MODELS_DIR  / "lstm_autoencoder_summary.json"

IFOREST_FEATURES = [
    "email_count", "unique_pcs", "total_size", "avg_size",
    "total_attachments", "emails_with_attachments", "after_hours_emails",
    "avg_recipients", "max_recipients", "avg_content_words", "max_content_words",
    "bcc_email_count", "cc_email_count", "attachment_email_ratio",
    "after_hours_ratio", "bcc_ratio", "o", "c", "e", "a", "n",
]

SPLIT_COLORS    = {"train": "#4C9BE8", "val": "#F4A83A", "test": "#E85454"}
EVAL_REPORT_PATH = MODELS_DIR / "evaluation_report.json"
INSIDERS_PATH    = ARCHIVE_DIR / "answers" / "answers" / "insiders.csv"


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_data
def load_iforest_scored() -> pd.DataFrame:
    df = pd.read_csv(IFOREST_SCORED_PATH)
    df["email_day"] = pd.to_datetime(df["email_day"], errors="coerce")
    return df.sort_values(["user", "email_day"]).reset_index(drop=True)


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
    return df.sort_values(["user", "email_day"]).reset_index(drop=True)


@st.cache_data
def load_lstm_summary() -> dict:
    return json.loads(LSTM_SUMMARY_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_eval_report() -> dict | None:
    if not EVAL_REPORT_PATH.exists():
        return None
    return json.loads(EVAL_REPORT_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_insider_labels() -> pd.DataFrame | None:
    if not INSIDERS_PATH.exists():
        return None
    ins = pd.read_csv(INSIDERS_PATH)
    ins = ins[ins["dataset"] == 4.2].copy()
    ins["start"] = pd.to_datetime(ins["start"], errors="coerce").dt.normalize()
    ins["end"]   = pd.to_datetime(ins["end"],   errors="coerce").dt.normalize()
    rows = []
    for _, r in ins.iterrows():
        if pd.isna(r["start"]) or pd.isna(r["end"]):
            continue
        for d in pd.date_range(r["start"], r["end"], freq="D"):
            rows.append({"user": r["user"], "email_day": d})
    day_labels = pd.DataFrame(rows).drop_duplicates()
    day_labels["is_insider"] = 1
    return day_labels


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def split_metrics(df: pd.DataFrame, score_col: str, severity_col: str) -> None:
    """Show anomaly rate and score stats for each split side by side."""
    for split in ["train", "val", "test"]:
        sdf = df[df["dataset_split"] == split]
        if sdf.empty:
            continue
        flagged = sdf[severity_col].isin(["suspicious", "high"]).sum()
        rate    = flagged / len(sdf) * 100
        st.markdown(
            f"**{split.upper()}** — {len(sdf):,} rows | "
            f"Flagged: {flagged:,} ({rate:.1f}%) | "
            f"Mean score: {sdf[score_col].mean():.4f} | "
            f"Max score: {sdf[score_col].max():.4f}"
        )


def score_histogram(df: pd.DataFrame, score_col: str, title: str) -> plt.Figure:
    """Overlapping score distribution histogram for train / val / test."""
    fig, ax = plt.subplots(figsize=(9, 3.5))
    for split, color in SPLIT_COLORS.items():
        subset = df[df["dataset_split"] == split][score_col].dropna()
        if not subset.empty:
            ax.hist(subset, bins=60, alpha=0.55, color=color, label=split, density=True)
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def top_users_bar(df: pd.DataFrame, score_col: str, title: str, n: int = 20) -> plt.Figure:
    """Horizontal bar chart of top-N users by mean anomaly score."""
    user_scores = (
        df.dropna(subset=[score_col])
        .groupby("user")[score_col].mean()
        .sort_values(ascending=False)
        .head(n)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#E85454" if v > 0.6 else "#F4A83A" if v > 0.4 else "#4C9BE8"
              for v in user_scores.values]
    ax.barh(user_scores.index[::-1], user_scores.values[::-1], color=colors[::-1])
    ax.set_xlabel("Mean Anomaly Score")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Tab 1 — Pipeline Overview
# ---------------------------------------------------------------------------

def render_overview(iforest_df: pd.DataFrame, iforest_summary: dict,
                    lstm_df: pd.DataFrame | None, lstm_summary: dict | None) -> None:
    st.header("Pipeline Overview")

    # Dataset split summary
    st.subheader("Dataset Split")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows", f"{len(iforest_df):,}")
    c2.metric("Train Rows", f"{(iforest_df['dataset_split']=='train').sum():,}")
    c3.metric("Val Rows",   f"{(iforest_df['dataset_split']=='val').sum():,}")
    c4.metric("Test Rows",  f"{(iforest_df['dataset_split']=='test').sum():,}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Users", iforest_df["user"].nunique())
    date_range = f"{iforest_df['email_day'].min().date()} → {iforest_df['email_day'].max().date()}"
    c2.metric("Date Range", date_range)
    c3.metric("Features Used", len(IFOREST_FEATURES))

    # Daily activity timeline
    st.subheader("Daily Row Count by Split")
    daily = (
        iforest_df.groupby(["email_day", "dataset_split"])
        .size().reset_index(name="rows")
        .pivot(index="email_day", columns="dataset_split", values="rows")
        .fillna(0)
    )
    st.area_chart(daily)

    # Model configs side by side
    st.subheader("Model Configurations")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Isolation Forest")
        st.json({
            "n_estimators": 300,
            "contamination": 0.03,
            "train_rows": iforest_summary.get("train_rows"),
            "val_rows": iforest_summary.get("val_rows"),
            "test_rows": iforest_summary.get("test_rows"),
            "suspicious_threshold": round(iforest_summary.get("suspicious_threshold", 0), 4),
            "high_threshold": round(iforest_summary.get("high_threshold", 0), 4),
        })

    with col_b:
        st.markdown("#### LSTM Autoencoder")
        if lstm_summary:
            st.json({
                "model_type": lstm_summary.get("model_type", "global"),
                "window_size": lstm_summary.get("window_size"),
                "hidden_dim": lstm_summary.get("hidden_dim"),
                "latent_dim": lstm_summary.get("latent_dim"),
                "epochs": lstm_summary.get("epochs"),
                "batch_size": lstm_summary.get("batch_size"),
                "suspicious_rows": lstm_summary.get("suspicious_rows"),
                "high_rows": lstm_summary.get("high_rows"),
            })
        else:
            st.info("LSTM not yet trained.")


# ---------------------------------------------------------------------------
# Tab 2 — Isolation Forest
# ---------------------------------------------------------------------------

def render_iforest(df: pd.DataFrame, summary: dict) -> None:
    st.header("Isolation Forest — Results")

    # Top-level metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Users Scored", summary["users"])
    c2.metric("Total Rows",   summary["rows"])
    c3.metric("Suspicious",   summary["suspicious_rows"])
    c4.metric("High Risk",    summary["high_rows"])

    st.markdown(
        f"Thresholds: suspicious ≥ **{summary['suspicious_threshold']:.4f}** | "
        f"high ≥ **{summary['high_threshold']:.4f}**"
    )

    # Score distribution per split
    st.subheader("Score Distribution by Split")
    st.pyplot(score_histogram(df, "iforest_score", "Isolation Forest Score Distribution"))
    split_metrics(df, "iforest_score", "risk_severity")

    # Score over time
    st.subheader("Mean Daily Anomaly Score Over Time")
    daily_scores = (
        df.groupby(["email_day", "dataset_split"])["iforest_score"]
        .mean().reset_index()
        .pivot(index="email_day", columns="dataset_split", values="iforest_score")
    )
    st.line_chart(daily_scores)

    # Top users
    st.subheader("Top 20 Users by Mean Score")
    st.pyplot(top_users_bar(df, "iforest_score", "Isolation Forest — Top 20 Risky Users"))

    # Severity breakdown per split
    st.subheader("Severity Breakdown per Split")
    sev_counts = (
        df.groupby(["dataset_split", "risk_severity"])
        .size().reset_index(name="count")
    )
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for i, split in enumerate(["train", "val", "test"]):
        subset = sev_counts[sev_counts["dataset_split"] == split].set_index("risk_severity")["count"]
        bottom = 0
        for sev, color in [("normal", "#6FCF97"), ("suspicious", "#F4A83A"), ("high", "#E85454")]:
            val = int(subset.get(sev, 0))
            ax.bar(split, val, bottom=bottom, color=color, label=sev if i == 0 else "")
            if val > 0:
                ax.text(i, bottom + val / 2, str(val), ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
            bottom += val
    ax.set_ylabel("Row Count")
    ax.set_title("Row Severity by Split")
    ax.legend(loc="upper right")
    fig.tight_layout()
    st.pyplot(fig)

    # Anomaly table with filters
    st.subheader("Anomaly Table")
    split_filter = st.selectbox("Filter by split", ["all", "train", "val", "test"], key="if_split")
    sev_filter   = st.selectbox("Filter by severity", ["all", "high", "suspicious"], key="if_sev")
    view = df.copy()
    if split_filter != "all":
        view = view[view["dataset_split"] == split_filter]
    if sev_filter != "all":
        view = view[view["risk_severity"] == sev_filter]
    view = view.sort_values("iforest_score", ascending=False).head(100)
    st.dataframe(
        view[["user", "email_day", "dataset_split", "iforest_score", "risk_severity",
              "email_count", "after_hours_emails", "bcc_ratio", "total_size"]],
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Tab 3 — LSTM Autoencoder
# ---------------------------------------------------------------------------

def render_lstm(df: pd.DataFrame, summary: dict) -> None:
    st.header("LSTM Autoencoder — Results")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Users",       summary["users"])
    c2.metric("Total Rows",  summary["rows"])
    c3.metric("Suspicious",  summary["suspicious_rows"])
    c4.metric("High Risk",   summary["high_rows"])

    st.info(
        f"Single global model | Window: {summary['window_size']} days | "
        f"Hidden: {summary['hidden_dim']} | Latent: {summary['latent_dim']} | "
        f"Epochs: {summary.get('epochs', 20)} | Batch: {summary.get('batch_size', 256)}"
    )

    st.subheader("Score Distribution by Split")
    st.pyplot(score_histogram(df, "lstm_score", "LSTM Anomaly Score Distribution"))
    split_metrics(df, "lstm_score", "lstm_risk_severity")

    st.subheader("Mean Daily LSTM Score Over Time")
    daily_scores = (
        df.dropna(subset=["lstm_score"])
        .groupby(["email_day", "dataset_split"])["lstm_score"]
        .mean().reset_index()
        .pivot(index="email_day", columns="dataset_split", values="lstm_score")
    )
    st.line_chart(daily_scores)

    st.subheader("Top 20 Users by Mean LSTM Score")
    st.pyplot(top_users_bar(df, "lstm_score", "LSTM Autoencoder — Top 20 Risky Users"))

    st.subheader("Per-User LSTM Timeline")
    users = sorted(df.dropna(subset=["lstm_score"])["user"].unique())
    user  = st.selectbox("Select user", users, key="lstm_user")
    udf   = df[df["user"] == user].sort_values("email_day")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active Days", len(udf))
    c2.metric("Peak Score",  f"{udf['lstm_score'].max():.4f}")
    c3.metric("High Days",   int((udf["lstm_risk_severity"] == "high").sum()))
    c4.metric("Split",       udf["dataset_split"].value_counts().idxmax())

    # Annotated timeline
    fig, ax = plt.subplots(figsize=(12, 3.5))
    for split, color in SPLIT_COLORS.items():
        mask = udf["dataset_split"] == split
        ax.scatter(udf.loc[mask, "email_day"], udf.loc[mask, "lstm_score"],
                   color=color, s=15, label=split, zorder=3)
    ax.plot(udf["email_day"], udf["lstm_score"], color="grey", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("LSTM Score")
    ax.set_title(f"LSTM Anomaly Score — {user}")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

    flagged = udf[udf["lstm_flag"] == 1].sort_values("lstm_score", ascending=False)
    if flagged.empty:
        st.success("No anomalous days detected.")
    else:
        st.dataframe(
            flagged[["email_day", "dataset_split", "lstm_score", "lstm_risk_severity",
                     "email_count", "after_hours_emails", "bcc_ratio", "total_size"]],
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Tab 4 — Model Comparison
# ---------------------------------------------------------------------------

def render_comparison(iforest_df: pd.DataFrame, lstm_df: pd.DataFrame) -> None:
    st.header("Model Comparison — Isolation Forest vs LSTM")

    # Merge on user + email_day
    merged = iforest_df[["user", "email_day", "dataset_split", "iforest_score", "risk_severity"]].merge(
        lstm_df[["user", "email_day", "lstm_score", "lstm_risk_severity"]],
        on=["user", "email_day"], how="inner",
    ).dropna(subset=["iforest_score", "lstm_score"])

    st.markdown(f"**{len(merged):,} rows** have scores from both models.")

    # Correlation per split
    st.subheader("Score Correlation per Split")
    cols = st.columns(3)
    for i, split in enumerate(["train", "val", "test"]):
        sdf = merged[merged["dataset_split"] == split]
        if not sdf.empty:
            corr = sdf["iforest_score"].corr(sdf["lstm_score"])
            cols[i].metric(f"{split.upper()} Correlation", f"{corr:.3f}")

    # Scatter: IForest vs LSTM
    st.subheader("IForest Score vs LSTM Score")
    split_sel = st.selectbox("Split to display", ["all", "train", "val", "test"], key="cmp_split")
    plot_df   = merged if split_sel == "all" else merged[merged["dataset_split"] == split_sel]
    sample    = plot_df.sample(min(5000, len(plot_df)), random_state=42)

    fig, ax = plt.subplots(figsize=(7, 5))
    for split, color in SPLIT_COLORS.items():
        mask = sample["dataset_split"] == split
        ax.scatter(sample.loc[mask, "iforest_score"], sample.loc[mask, "lstm_score"],
                   alpha=0.3, s=8, color=color, label=split)
    ax.set_xlabel("Isolation Forest Score")
    ax.set_ylabel("LSTM Score")
    ax.set_title("IForest vs LSTM Anomaly Scores")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

    # Agreement analysis — per user
    st.subheader("Per-User Agreement")
    user_if   = merged.groupby("user")["iforest_score"].mean().rename("iforest_mean")
    user_lstm = merged.groupby("user")["lstm_score"].mean().rename("lstm_mean")
    user_cmp  = pd.concat([user_if, user_lstm], axis=1).dropna()
    user_cmp["if_flagged"]   = user_cmp["iforest_mean"] > user_cmp["iforest_mean"].quantile(0.90)
    user_cmp["lstm_flagged"] = user_cmp["lstm_mean"]    > user_cmp["lstm_mean"].quantile(0.90)
    user_cmp["agreement"]    = user_cmp["if_flagged"] & user_cmp["lstm_flagged"]

    both   = user_cmp["agreement"].sum()
    only_if  = (user_cmp["if_flagged"] & ~user_cmp["lstm_flagged"]).sum()
    only_lstm = (~user_cmp["if_flagged"] & user_cmp["lstm_flagged"]).sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Flagged by BOTH (top 10%)", int(both))
    c2.metric("IForest only",  int(only_if))
    c3.metric("LSTM only",     int(only_lstm))

    # Venn-like bar
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.barh(["Agreement"], [both],     color="#E85454", label="Both")
    ax.barh(["Agreement"], [only_if],  left=both, color="#4C9BE8", label="IForest only")
    ax.barh(["Agreement"], [only_lstm], left=both+only_if, color="#F4A83A", label="LSTM only")
    ax.legend(loc="lower right")
    ax.set_xlabel("Number of Users")
    ax.set_title("Top-10% Flagging Agreement Between Models")
    fig.tight_layout()
    st.pyplot(fig)

    # Users flagged by both — highest combined risk
    st.subheader("Highest Combined Risk Users (flagged by both models)")
    user_cmp["combined_score"] = user_cmp["iforest_mean"] + user_cmp["lstm_mean"]
    top_combined = (
        user_cmp[user_cmp["agreement"]]
        .sort_values("combined_score", ascending=False)
        .head(20)
        .reset_index()
    )
    st.dataframe(
        top_combined[["user", "iforest_mean", "lstm_mean", "combined_score"]].round(4),
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Tab 5 — User Investigation
# ---------------------------------------------------------------------------

def render_user_investigation(iforest_df: pd.DataFrame, lstm_df: pd.DataFrame) -> None:
    st.header("User Investigation")
    st.write("Deep-dive into any user — both model scores, split timeline, and feature breakdown.")

    all_users = sorted(iforest_df["user"].unique())
    user = st.selectbox("Select user to investigate", all_users, key="inv_user")

    idf = iforest_df[iforest_df["user"] == user].sort_values("email_day").reset_index(drop=True)
    ldf = lstm_df[lstm_df["user"] == user].sort_values("email_day").reset_index(drop=True) \
          if lstm_df is not None else pd.DataFrame()

    # Summary metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Active Days",     len(idf))
    c2.metric("Peak IForest",    f"{idf['iforest_score'].max():.4f}")
    c3.metric("Peak LSTM",       f"{ldf['lstm_score'].max():.4f}" if not ldf.empty else "N/A")
    c4.metric("IForest High Days", int((idf["risk_severity"] == "high").sum()))
    c5.metric("LSTM High Days",    int((ldf["lstm_risk_severity"] == "high").sum()) if not ldf.empty else 0)

    # Dual score timeline
    st.subheader("Anomaly Score Timeline")
    fig, axes = plt.subplots(2, 1, figsize=(13, 5), sharex=True)

    for split, color in SPLIT_COLORS.items():
        mask = idf["dataset_split"] == split
        axes[0].scatter(idf.loc[mask, "email_day"], idf.loc[mask, "iforest_score"],
                        s=12, color=color, label=split, zorder=3)
    axes[0].plot(idf["email_day"], idf["iforest_score"], color="grey", lw=0.5, alpha=0.4)
    axes[0].axhline(idf["iforest_score"].quantile(0.95), color="#E85454", lw=0.8, linestyle="--", label="p95")
    axes[0].set_ylabel("IForest Score")
    axes[0].set_title(f"Isolation Forest — {user}")
    axes[0].legend(fontsize=7)

    if not ldf.empty:
        for split, color in SPLIT_COLORS.items():
            mask = ldf["dataset_split"] == split
            axes[1].scatter(ldf.loc[mask, "email_day"], ldf.loc[mask, "lstm_score"],
                            s=12, color=color, zorder=3)
        axes[1].plot(ldf["email_day"], ldf["lstm_score"], color="grey", lw=0.5, alpha=0.4)
        axes[1].axhline(ldf["lstm_score"].quantile(0.95), color="#E85454", lw=0.8, linestyle="--")
        axes[1].set_ylabel("LSTM Score")
        axes[1].set_title(f"LSTM Autoencoder — {user}")

    fig.tight_layout()
    st.pyplot(fig)

    # Feature breakdown over time
    st.subheader("Behavioral Features Over Time")
    features = st.multiselect(
        "Select features to plot",
        ["email_count", "after_hours_emails", "bcc_ratio", "total_size",
         "total_attachments", "avg_recipients", "unique_pcs"],
        default=["email_count", "after_hours_emails", "bcc_ratio"],
        key="feat_sel",
    )
    if features:
        st.line_chart(idf.set_index("email_day")[features])

    # Raw data table
    with st.expander("Raw daily rows for this user"):
        st.dataframe(idf, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 6 — Live Scoring
# ---------------------------------------------------------------------------

def render_live_scoring(artifacts: dict) -> None:
    st.header("Live Scoring — Isolation Forest")
    st.write("Enter a user's daily feature values and get an instant anomaly score.")

    defaults = {
        "email_count": 15.0, "unique_pcs": 1.0, "total_size": 400000.0,
        "avg_size": 25000.0, "total_attachments": 2.0, "emails_with_attachments": 1.0,
        "after_hours_emails": 0.0, "avg_recipients": 2.0, "max_recipients": 4.0,
        "avg_content_words": 55.0, "max_content_words": 90.0, "bcc_email_count": 0.0,
        "cc_email_count": 4.0, "attachment_email_ratio": 0.10, "after_hours_ratio": 0.00,
        "bcc_ratio": 0.00, "o": 30.0, "c": 30.0, "e": 30.0, "a": 30.0, "n": 30.0,
    }

    st.markdown("#### Behavioral Features")
    row = {}
    cols = st.columns(4)
    behavioral = [f for f in IFOREST_FEATURES if f not in ["o", "c", "e", "a", "n"]]
    for i, feat in enumerate(behavioral):
        row[feat] = cols[i % 4].number_input(feat, value=float(defaults[feat]), key=f"ls_{feat}")

    st.markdown("#### Psychometric Scores (OCEAN)")
    pcols = st.columns(5)
    for i, feat in enumerate(["o", "c", "e", "a", "n"]):
        row[feat] = pcols[i].number_input(feat.upper(), value=float(defaults[feat]),
                                          min_value=0.0, max_value=100.0, key=f"ls_{feat}")

    if st.button("Score This Row", type="primary"):
        result = score_single_row(row, artifacts=artifacts)
        score  = float(result["iforest_score"])
        sev    = result["risk_severity"]

        color = {"high": "red", "suspicious": "orange", "normal": "green"}.get(sev, "grey")
        st.markdown(
            f"<h2 style='color:{color}'>Risk: {sev.upper()} &nbsp;|&nbsp; Score: {score:.4f}</h2>",
            unsafe_allow_html=True,
        )

        # Score gauge
        fig, ax = plt.subplots(figsize=(7, 1.2))
        ax.barh(["Score"], [score], color=color, height=0.4)
        ax.barh(["Score"], [1 - score], left=score, color="#eee", height=0.4)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Anomaly Score (0 = normal, 1 = highly anomalous)")
        ax.set_title(f"Score: {score:.4f} — {sev.upper()}")
        ax.axvline(artifacts.get("suspicious_threshold", 0.5),
                   color="orange", lw=1.5, linestyle="--", label="suspicious")
        ax.axvline(artifacts.get("high_threshold", 0.7),
                   color="red", lw=1.5, linestyle="--", label="high")
        ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)

        st.json({
            "iforest_score": round(score, 4),
            "iforest_flag": int(result["iforest_flag"]),
            "risk_severity": sev,
        })


# ---------------------------------------------------------------------------
# Tab 7 — Ground Truth Evaluation
# ---------------------------------------------------------------------------

def _confusion_heatmap(tp: int, fp: int, tn: int, fn: int, title: str) -> plt.Figure:
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(4, 3.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Normal", "Pred Anomaly"])
    ax.set_yticklabels(["Actual Normal", "Actual Insider"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def render_evaluation(iforest_df: pd.DataFrame, lstm_df: pd.DataFrame | None) -> None:
    st.header("Ground Truth Evaluation — CERT r4.2")
    st.caption(
        "70 known insider users (dataset 4.2) evaluated against both models. "
        "Metrics computed on the held-out **test split** at day level, and across "
        "all data at user level (max score per user)."
    )

    report = load_eval_report()
    if report is None:
        st.warning(
            "Evaluation report not found. Run:\n\n"
            "```\npython scripts/evaluate_models.py "
            "--answers archive/answers/answers/insiders.csv\n```"
        )
        return

    # ── 1. Summary metric table ──────────────────────────────────────────────
    st.subheader("Metrics Summary")

    keys    = ["roc_auc", "avg_precision", "precision", "recall", "f1"]
    labels  = ["ROC AUC", "Avg Precision (AP)", "Precision", "Recall", "F1"]
    sections = {
        "IF — Day (test)":   report.get("if_day_test",  {}),
        "IF — User (all)":   report.get("if_user_all",  {}),
        "LSTM — Day (test)": report.get("lstm_day_test",{}),
        "LSTM — User (all)": report.get("lstm_user_all",{}),
    }
    table_rows = []
    for label, data in sections.items():
        row = {"Model / Level": label}
        row.update({lbl: data.get(k, "—") for k, lbl in zip(keys, labels)})
        table_rows.append(row)
    st.dataframe(pd.DataFrame(table_rows).set_index("Model / Level"), use_container_width=True)

    st.info(
        "**Interpretation:** IF day-level ROC AUC < 0.5 means insider days look "
        "*more normal* to the email-based IF model — insiders are behaviorally subtle. "
        "LSTM (0.60) better captures temporal shifts in behavior over its 7-day window."
    )

    # ── 2. Confusion matrices ────────────────────────────────────────────────
    st.subheader("Confusion Matrices")
    col1, col2, col3, col4 = st.columns(4)
    for col, (title, key) in zip(
        [col1, col2, col3, col4],
        [("IF Day (test)", "if_day_test"), ("IF User (all)", "if_user_all"),
         ("LSTM Day (test)", "lstm_day_test"), ("LSTM User (all)", "lstm_user_all")],
    ):
        d = report.get(key, {})
        if d:
            fig = _confusion_heatmap(d["tp"], d["fp"], d["tn"], d["fn"], title)
            col.pyplot(fig)

    # ── 3. Score distributions: insiders vs normals ─────────────────────────
    day_labels = load_insider_labels()
    if day_labels is not None and iforest_df is not None:
        st.subheader("Score Distributions: Insiders vs Normal Users")

        if_merged = iforest_df[iforest_df["dataset_split"] == "test"].merge(
            day_labels, on=["user", "email_day"], how="left"
        )
        if_merged["is_insider"] = if_merged["is_insider"].fillna(0).astype(int)

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            for label, mask, color in [
                ("Normal",  if_merged["is_insider"] == 0, "#4C9BE8"),
                ("Insider", if_merged["is_insider"] == 1, "#E85454"),
            ]:
                scores = if_merged.loc[mask, "iforest_score"].dropna()
                ax.hist(scores, bins=50, alpha=0.6, color=color, label=f"{label} (n={len(scores):,})", density=True)
            ax.set_xlabel("Isolation Forest Score")
            ax.set_ylabel("Density")
            ax.set_title("IF Score — Insider vs Normal (test)")
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)

        if lstm_df is not None:
            lstm_merged = lstm_df[
                (lstm_df["dataset_split"] == "test") &
                (lstm_df["lstm_risk_severity"] != "undetermined")
            ].merge(day_labels, on=["user", "email_day"], how="left")
            lstm_merged["is_insider"] = lstm_merged["is_insider"].fillna(0).astype(int)

            with col2:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                for label, mask, color in [
                    ("Normal",  lstm_merged["is_insider"] == 0, "#4C9BE8"),
                    ("Insider", lstm_merged["is_insider"] == 1, "#E85454"),
                ]:
                    scores = lstm_merged.loc[mask, "lstm_score"].dropna()
                    ax.hist(scores, bins=50, alpha=0.6, color=color,
                            label=f"{label} (n={len(scores):,})", density=True)
                ax.set_xlabel("LSTM Score")
                ax.set_ylabel("Density")
                ax.set_title("LSTM Score — Insider vs Normal (test)")
                ax.legend()
                fig.tight_layout()
                st.pyplot(fig)

    # ── 4. Per-insider user scores ───────────────────────────────────────────
    if day_labels is not None and iforest_df is not None:
        st.subheader("Insider Users — Model Scores")
        insider_users = day_labels["user"].unique()

        if_user_scores = (
            iforest_df[iforest_df["user"].isin(insider_users)]
            .groupby("user")["iforest_score"].max()
            .rename("if_max_score")
        )
        rows_data = []
        for u in insider_users:
            row_d = {"user": u, "if_max_score": round(float(if_user_scores.get(u, 0)), 4)}
            if lstm_df is not None:
                lu = lstm_df[(lstm_df["user"] == u) & (lstm_df["lstm_risk_severity"] != "undetermined")]
                row_d["lstm_max_score"] = round(float(lu["lstm_score"].max()) if not lu.empty else 0, 4)
            rows_data.append(row_d)

        insider_table = pd.DataFrame(rows_data).sort_values("if_max_score", ascending=False)
        insider_table["if_caught"]   = insider_table["if_max_score"] >= 0.496
        if "lstm_max_score" in insider_table.columns:
            insider_table["lstm_caught"] = insider_table["lstm_max_score"] >= 0.723

        st.dataframe(insider_table, use_container_width=True)

        # Summary caught/missed
        st.subheader("Detection Summary")
        c1, c2, c3, c4 = st.columns(4)
        n = len(insider_users)
        if_caught   = int(insider_table["if_caught"].sum())
        lstm_caught = int(insider_table["lstm_caught"].sum()) if "lstm_caught" in insider_table.columns else 0
        c1.metric("Total Insiders", n)
        c2.metric("IF Caught (user-level)", f"{if_caught} / {n}")
        c3.metric("LSTM Caught (user-level)", f"{lstm_caught} / {n}")
        both = int((insider_table.get("if_caught", False) & insider_table.get("lstm_caught", False)).sum()) \
               if "lstm_caught" in insider_table.columns else 0
        c4.metric("Caught by Both", f"{both} / {n}")

        # Bar: caught vs missed per model
        fig, ax = plt.subplots(figsize=(6, 2.5))
        models  = ["Isolation Forest", "LSTM Autoencoder"]
        caught  = [if_caught, lstm_caught]
        missed  = [n - if_caught, n - lstm_caught]
        ax.barh(models, caught, color="#6FCF97", label="Caught")
        ax.barh(models, missed, left=caught, color="#E85454", label="Missed")
        for i, (c, m) in enumerate(zip(caught, missed)):
            ax.text(c / 2, i, str(c), ha="center", va="center", color="white", fontweight="bold")
            ax.text(c + m / 2, i, str(m), ha="center", va="center", color="white", fontweight="bold")
        ax.set_xlabel("Number of Insider Users")
        ax.set_title(f"Insider Detection: Caught vs Missed (out of {n})")
        ax.legend(loc="lower right")
        fig.tight_layout()
        st.pyplot(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="CERT Insider Threat Dashboard",
        page_icon="🔍",
        layout="wide",
    )
    st.title("CERT Insider Threat Detection Dashboard")
    st.caption("Isolation Forest + LSTM Autoencoder — Train / Val / Test Analysis")

    iforest_ready = IFOREST_SCORED_PATH.exists() and IFOREST_SUMMARY_PATH.exists()
    lstm_ready    = LSTM_SCORED_PATH.exists() and LSTM_SUMMARY_PATH.exists()

    if not iforest_ready:
        st.error("Isolation Forest results not found. Run Step 7 first.")
        st.stop()

    iforest_df      = load_iforest_scored()
    iforest_summary = load_iforest_summary()
    iforest_artifacts = get_iforest_artifacts()

    lstm_df      = load_lstm_scored()      if lstm_ready else None
    lstm_summary = load_lstm_summary()     if lstm_ready else None

    tabs = st.tabs([
        "Pipeline Overview",
        "Isolation Forest",
        "LSTM Autoencoder",
        "Model Comparison",
        "User Investigation",
        "Live Scoring",
        "Evaluation (Ground Truth)",
    ])

    with tabs[0]:
        render_overview(iforest_df, iforest_summary, lstm_df, lstm_summary)

    with tabs[1]:
        render_iforest(iforest_df, iforest_summary)

    with tabs[2]:
        if lstm_ready:
            render_lstm(lstm_df, lstm_summary)
        else:
            st.warning("LSTM results not found. Run Step 9 first.")

    with tabs[3]:
        if lstm_ready:
            render_comparison(iforest_df, lstm_df)
        else:
            st.warning("LSTM results needed for comparison. Run Step 9 first.")

    with tabs[4]:
        render_user_investigation(iforest_df, lstm_df)

    with tabs[5]:
        render_live_scoring(iforest_artifacts)

    with tabs[6]:
        render_evaluation(iforest_df, lstm_df)


if __name__ == "__main__":
    main()
