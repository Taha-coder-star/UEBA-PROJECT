"""Tabbed UEBA + DLP dashboard.

Run with:
    streamlit run app/ueba_dashboard_tabs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
for path in [str(ROOT), str(ROOT / "colab"), str(ROOT / "app")]:
    if path not in sys.path:
        sys.path.insert(0, path)

import ueba_dashboard as base  # noqa: E402
from risk_scorer import explain_user  # noqa: E402
from user_level_eval import apply_user_threshold, evaluate_topk_users  # noqa: E402


def _metric_row(metrics: dict) -> None:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Precision", f"{metrics['precision']:.3f}")
    c2.metric("Recall", f"{metrics['recall']:.3f}")
    c3.metric("F1", f"{metrics['f1']:.3f}")
    c4.metric("TP", metrics["tp"])
    c5.metric("FP", metrics["fp"])
    c6.metric("FN", metrics["fn"])


def _top_table(risk_df: pd.DataFrame, ranked_df: pd.DataFrame, k: int, show_truth: bool) -> None:
    top_users = list(ranked_df.head(k)["user"])
    display_df = risk_df[risk_df["user"].isin(top_users)].copy()
    rank_map = {user: i + 1 for i, user in enumerate(top_users)}
    display_df["rank"] = display_df["user"].map(rank_map)
    display_df = display_df.sort_values("rank")

    show_cols = [
        "rank", "user", "employee_name", "risk_score", "lstm_p95_norm",
        "after_hours_norm", "bcc_usage_norm", "file_exfil_norm",
        "usb_activity_norm", "multi_pc_norm", "content_sensitivity_norm",
    ]
    if show_truth and "is_insider" in display_df.columns:
        show_cols.append("is_insider")
    show_cols = [c for c in show_cols if c in display_df.columns]

    names = {
        "rank": "Rank",
        "user": "User",
        "employee_name": "Name",
        "risk_score": base._ga_score_label(),
        "lstm_p95_norm": "LSTM P95",
        "after_hours_norm": "After Hours",
        "bcc_usage_norm": "BCC",
        "file_exfil_norm": "File Exfil",
        "usb_activity_norm": "USB",
        "multi_pc_norm": "Multi-PC",
        "content_sensitivity_norm": "Content Sensitivity",
        "is_insider": "Insider?",
    }
    st.dataframe(
        display_df[show_cols].rename(columns=names).set_index("Rank"),
        use_container_width=True,
    )


def _model_page(title: str, user_df: pd.DataFrame, agg: str, thresh_pct: int,
                k: int, insider_users: set[str], model_note: str) -> None:
    st.header(title)
    st.caption(model_note)

    filtered, cutoff = apply_user_threshold(user_df, agg, thresh_pct)
    metrics = evaluate_topk_users(filtered, insider_users, k)
    _metric_row(metrics)
    st.caption(
        f"Threshold = train-user {thresh_pct}th percentile of `{agg}` "
        f"({cutoff:.4f}); pool after filter = {len(filtered):,}; top-K = {k}."
    )

    tab_a, tab_b, tab_c = st.tabs(["Precision / Recall", "Score Distribution", "Outcomes"])
    with tab_a:
        st.pyplot(base.fig_prf1_vs_k(user_df, agg, thresh_pct, insider_users, [5, 10, 20, 30, 50, 75, 100]))
        plt.close()
    with tab_b:
        st.pyplot(base.fig_score_distribution(user_df, insider_users))
        plt.close()
    with tab_c:
        st.pyplot(base.fig_tp_fp_fn(user_df, agg, thresh_pct, insider_users))
        plt.close()


def _investigation_page(risk_df: pd.DataFrame, ranked_df: pd.DataFrame, k: int) -> None:
    st.header("User Investigation")
    users = list(ranked_df.head(k)["user"])
    if not users:
        st.info("No users are available for the selected ranking.")
        return

    selected_user = st.selectbox("Select a flagged user", users)
    row = risk_df[risk_df["user"] == selected_user].iloc[0]
    st.subheader(f"{row.get('employee_name', selected_user)} ({selected_user})")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(base._ga_score_label(), f"{row['risk_score']:.4f}")
    c2.metric("LSTM P95", f"{row.get('score_p95', 0):.4f}")
    c3.metric("Content Sensitivity", f"{row.get('content_sensitivity_norm', 0):.3f}")
    c4.metric("Ground Truth", "Insider" if row.get("is_insider", 0) else "Normal")

    st.markdown("**Why this user was flagged:**")
    for flag in explain_user(row):
        st.markdown(f"- {flag}")

    signals = {}
    mapping = {
        "lstm_p95": "lstm_p95_norm",
        "after_hours": "after_hours_norm",
        "bcc_usage": "bcc_usage_norm",
        "file_exfil": "file_exfil_norm",
        "usb_activity": "usb_activity_norm",
        "multi_pc": "multi_pc_norm",
        "content_sensitivity": "content_sensitivity_norm",
    }
    for key, col in mapping.items():
        if col in row.index:
            label, color = base._SIGNAL_DISPLAY[key]
            signals[label] = float(row[col])

    sig_df = pd.DataFrame({"Signal": list(signals), "Value": list(signals.values())})
    st.bar_chart(sig_df.set_index("Signal"))


def _ga_page() -> None:
    st.header("Genetic Algorithm Explained")
    st.markdown(
        """
The GA is **not retraining the LSTM or Isolation Forest**. Those models are already trained.

What the GA does here:

1. It treats the risk scorer as a weighted formula.
2. Each candidate solution is a chromosome containing signal weights and flag thresholds.
3. Signals include LSTM P95, after-hours activity, BCC usage, file exfiltration, USB activity, multi-PC access, and content sensitivity.
4. The GA ranks users using those weights.
5. It scores each candidate by how well the top-K investigation queue catches known insiders.
6. Better candidates are selected, crossed over, mutated, and carried into the next generation.

In plain English: **GA searches for better risk-score settings**, so the dashboard's final investigation queue is more aligned with the ground-truth insider labels.

Important limitation: if the labeled insider set is small, GA can overfit. That is why we first fixed the ground-truth release mismatch and prefer r4.2 with about 70 insiders.
        """
    )
    base.render_ga_summary()


def main() -> None:
    st.title("UEBA + DLP Insider Threat Dashboard")
    st.caption(base.load_ground_truth_description())

    sensitivity_available = base.SENSITIVITY_CSV.exists()
    if not sensitivity_available:
        st.warning("DLP content sensitivity file is missing; content_sensitivity will be zero.")

    with st.spinner("Loading scored artifacts..."):
        insider_users = base.load_insider_users()
        lstm_user_df = base.load_lstm_user_df()
        if_user_df = base.load_if_user_df()
        risk_df = base.load_risk_df(sensitivity_available)

    st.sidebar.header("Controls")
    model_choice = st.sidebar.selectbox("Primary model for top-K evaluation", ["LSTM Autoencoder", "Isolation Forest"])
    agg_choice = st.sidebar.selectbox("Aggregation", ["score_p95", "score_mean", "score_max"])
    thresh_pct = st.sidebar.select_slider("Threshold percentile", [80, 85, 90, 95, 97, 99], value=90)
    k_choice = st.sidebar.selectbox("Top-K", [10, 20, 30, 50, 75, 100], index=3)

    selected_user_df = lstm_user_df if model_choice == "LSTM Autoencoder" else if_user_df
    ranked_df, cutoff = apply_user_threshold(selected_user_df, agg_choice, thresh_pct)
    metrics = evaluate_topk_users(ranked_df, insider_users, k_choice)

    pages = st.tabs([
        "Overview",
        "Isolation Forest",
        "LSTM Autoencoder",
        "Risk Queue",
        "User Investigation",
        "GA Explained",
        "Evaluation",
    ])

    with pages[0]:
        st.header("Pipeline Overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Scored LSTM users", f"{len(lstm_user_df):,}")
        c2.metric("Scored IF users", f"{len(if_user_df):,}")
        c3.metric("Ground-truth insiders", f"{len(insider_users):,}")
        c4.metric("DLP sensitivity", "Available" if sensitivity_available else "Missing")
        st.markdown(
            """
This dashboard combines UEBA and DLP signals:

- **Isolation Forest** detects unusual daily feature vectors.
- **LSTM Autoencoder** detects unusual temporal behavior through reconstruction error.
- **DLP Content Sensitivity** scores email/file content as internal, sensitive, or restricted.
- **Risk Scoring** combines all signals into a ranked investigation queue.
- **GA Optimization** optionally tunes risk weights and flag thresholds after models are trained.
            """
        )
        st.pyplot(base.fig_weights_pie())
        plt.close()

    with pages[1]:
        _model_page(
            "Isolation Forest Page",
            if_user_df,
            agg_choice,
            thresh_pct,
            k_choice,
            insider_users,
            "Isolation Forest is a secondary unsupervised anomaly detector over daily UEBA features.",
        )

    with pages[2]:
        _model_page(
            "LSTM Autoencoder Page",
            lstm_user_df,
            agg_choice,
            thresh_pct,
            k_choice,
            insider_users,
            "LSTM is the main temporal behavior model. It learns normal user-day sequences and flags high reconstruction error.",
        )

    with pages[3]:
        st.header(f"Top Suspicious Users - {base._ga_score_label()}")
        _metric_row(metrics)
        _top_table(risk_df, ranked_df, k_choice, show_truth=True)
        st.pyplot(base.fig_top_users_bar(risk_df, insider_users, min(k_choice, 20), base._ga_score_label()))
        plt.close()

    with pages[4]:
        _investigation_page(risk_df, ranked_df, k_choice)

    with pages[5]:
        _ga_page()

    with pages[6]:
        st.header("Evaluation Summary")
        _metric_row(metrics)
        st.caption(
            f"Model={model_choice}, aggregation={agg_choice}, threshold percentile={thresh_pct}, "
            f"cutoff={cutoff:.4f}, top-K={k_choice}."
        )
        if base.GA_REPORT_JSON.exists():
            st.json(base._load_json_safe(base.GA_REPORT_JSON))
        else:
            st.info("GA report is not available. This is fine if you skipped GA for the final run.")


if __name__ == "__main__":
    main()
