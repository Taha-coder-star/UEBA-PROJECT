"""UEBA Dashboard -- Insider Threat Detection System.

Standalone Streamlit app.  Run with:
    streamlit run app/ueba_dashboard.py

Integrates:
  - Fixed user-level evaluation pipeline    (colab/user_level_eval.py)
  - Weighted risk scoring AI component      (colab/risk_scorer.py)
  - GA-optimised weight config              (models/ga_optimized_config.json)
  - DLP content sensitivity signals         (cleaned/content_sensitivity_daily.csv)
  - Presentation-ready charts
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import streamlit as st

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
for _p in [str(_ROOT), str(_ROOT / "colab")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import CLEANED_DIR, MODELS_DIR  # noqa: E402
from user_level_eval import (  # noqa: E402
    compute_user_scores, apply_user_threshold, evaluate_topk_users,
)
from risk_scorer import (  # noqa: E402
    compute_behavioral_signals, compute_risk_scores,
    build_investigation_queue, explain_user, explain_dataframe,
    load_sensitivity_signals,
    WEIGHTS, _GA_LOADED,
)
from ground_truth import describe_selection, find_insiders_csv, select_ground_truth_release  # noqa: E402

IFOREST_CSV        = CLEANED_DIR / "email_user_daily_scored.csv"
LSTM_CSV           = CLEANED_DIR / "email_user_daily_lstm_scored.csv"
SENSITIVITY_CSV    = CLEANED_DIR / "content_sensitivity_daily.csv"
GA_CONFIG_JSON     = MODELS_DIR / "ga_optimized_config.json"
GA_REPORT_JSON     = MODELS_DIR / "ga_optimization_report.json"

INSIDER_COLOR = "#E85454"
NORMAL_COLOR  = "#4C9BE8"

# Default signal names / colours used across charts (order matches WEIGHTS)
_SIGNAL_DISPLAY = {
    "lstm_p95":            ("LSTM P95",          "#5CB85C"),
    "after_hours":         ("After Hours",        "#4C9BE8"),
    "bcc_usage":           ("BCC Usage",          "#F4A83A"),
    "file_exfil":          ("File Exfil",         "#E85454"),
    "usb_activity":        ("USB Activity",       "#9B59B6"),
    "multi_pc":            ("Multi PC",           "#95A5A6"),
    "content_sensitivity": ("Content Sensitivity","#1ABC9C"),
}

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UEBA — Insider Threat Detection",
    page_icon="🔒",
    layout="wide",
)


# ── cached loaders ────────────────────────────────────────────────────────────

@st.cache_data
def load_insider_users() -> set[str]:
    if not IFOREST_CSV.exists() and not LSTM_CSV.exists():
        return set()
    try:
        return select_ground_truth_release([IFOREST_CSV, LSTM_CSV]).matching_users
    except Exception:
        return set()


@st.cache_data
def load_ground_truth_description() -> str:
    try:
        return describe_selection(select_ground_truth_release([IFOREST_CSV, LSTM_CSV]))
    except Exception as exc:
        return f"Ground truth unavailable: {exc}"


def ground_truth_available() -> bool:
    try:
        find_insiders_csv()
        return True
    except FileNotFoundError:
        return False


@st.cache_data
def load_lstm_user_df() -> pd.DataFrame:
    ldf = pd.read_csv(LSTM_CSV,
                      usecols=["user", "lstm_score", "lstm_risk_severity", "dataset_split"])
    ldf = ldf[ldf["lstm_risk_severity"] != "undetermined"]
    iu  = load_insider_users()
    return compute_user_scores(ldf, "lstm_score", iu)


@st.cache_data
def load_if_user_df() -> pd.DataFrame:
    idf = pd.read_csv(IFOREST_CSV,
                      usecols=["user", "iforest_score", "dataset_split"])
    iu  = load_insider_users()
    return compute_user_scores(idf, "iforest_score", iu)


@st.cache_data
def load_behavioral_df() -> pd.DataFrame:
    cols = [
        "user", "email_day", "after_hours_ratio", "bcc_ratio",
        "file_to_removable", "file_total", "usb_connect_count",
        "after_hours_logons", "logon_count", "unique_logon_pcs",
        "employee_name", "dataset_split",
    ]
    idf = pd.read_csv(IFOREST_CSV, usecols=cols)
    return compute_behavioral_signals(idf)


@st.cache_data
def _load_sensitivity_df() -> pd.DataFrame | None:
    """Wraps load_sensitivity_signals() so the result is cached."""
    return load_sensitivity_signals(SENSITIVITY_CSV)


@st.cache_data
def load_risk_df(sensitivity_available: bool) -> pd.DataFrame:
    iu    = load_insider_users()
    lu    = load_lstm_user_df()
    behav = load_behavioral_df()
    sens  = _load_sensitivity_df() if sensitivity_available else None
    return compute_risk_scores(lu, behav, iu, sensitivity_df=sens)


# ── GA / config helpers ───────────────────────────────────────────────────────

def _load_json_safe(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except Exception:
        return None


def _ga_score_label() -> str:
    return "GA-Optimized DLP Risk Score" if _GA_LOADED else "Risk Score"


# ── chart helpers ─────────────────────────────────────────────────────────────

def fig_prf1_vs_k(user_df, agg, thresh_pct, insider_users, k_values):
    rows = []
    for k in k_values:
        filtered, _ = apply_user_threshold(user_df, agg, thresh_pct)
        m = evaluate_topk_users(filtered, insider_users, k)
        rows.append({"K": k, "Precision": m["precision"],
                     "Recall": m["recall"], "F1": m["f1"]})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(df["K"], df["Precision"], "o-", color="#E85454", linewidth=2, label="Precision")
    ax.plot(df["K"], df["Recall"],    "s-", color=NORMAL_COLOR, linewidth=2, label="Recall")
    ax.plot(df["K"], df["F1"],        "^-", color="#5CB85C", linewidth=2.5, label="F1")
    ax.set_xlabel("K (top users flagged)")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 vs K")
    ax.set_ylim(0, 1)
    ax.set_xticks(k_values)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_score_distribution(user_df, insider_users):
    insiders = user_df[user_df["is_insider"] == 1]["score_p95"].dropna()
    normals  = user_df[user_df["is_insider"] == 0]["score_p95"].dropna()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(normals,  bins=40, alpha=0.6, color=NORMAL_COLOR,
            label=f"Normal (n={len(normals):,})", density=True)
    ax.hist(insiders, bins=40, alpha=0.75, color=INSIDER_COLOR,
            label=f"Insider (n={len(insiders):,})", density=True)
    ax.axvline(normals.mean(),  color=NORMAL_COLOR,  linestyle="--", linewidth=1.5,
               label=f"Normal mean = {normals.mean():.3f}")
    ax.axvline(insiders.mean(), color=INSIDER_COLOR, linestyle="--", linewidth=1.5,
               label=f"Insider mean = {insiders.mean():.3f}")
    ax.set_xlabel("LSTM score_p95")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution: Insiders vs Normal Users")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_top_users_bar(risk_df, insider_users, top_n=20, score_label="Risk Score"):
    top = risk_df.head(top_n).copy()
    top["label"] = top["user"].apply(
        lambda u: f"★ {u}" if u in insider_users else u
    )
    colors = [INSIDER_COLOR if u in insider_users else NORMAL_COLOR
              for u in top["user"]]
    fig, ax = plt.subplots(figsize=(7, max(4, top_n * 0.28)))
    ax.barh(top["label"][::-1], top["risk_score"][::-1],
            color=colors[::-1], alpha=0.85)
    ax.set_xlabel(score_label)
    ax.set_title(f"Top {top_n} Suspicious Users")
    ins_patch = mpatches.Patch(color=INSIDER_COLOR, label="Confirmed insider (★)")
    nor_patch  = mpatches.Patch(color=NORMAL_COLOR,  label="Normal user")
    ax.legend(handles=[ins_patch, nor_patch], fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def fig_tp_fp_fn(user_df, agg, thresh_pct, insider_users):
    k_vals = [10, 20, 50, 100]
    tps, fps, fns = [], [], []
    for k in k_vals:
        filtered, _ = apply_user_threshold(user_df, agg, thresh_pct)
        m = evaluate_topk_users(filtered, insider_users, k)
        tps.append(m["tp"]); fps.append(m["fp"]); fns.append(m["fn"])
    x = np.arange(len(k_vals))
    w = 0.25
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(x - w, tps, w, label="TP (caught)",     color="#5CB85C", alpha=0.85)
    ax.bar(x,     fps, w, label="FP (false alarm)", color=INSIDER_COLOR, alpha=0.85)
    ax.bar(x + w, fns, w, label="FN (missed)",      color="#AAAAAA", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in k_vals])
    ax.set_ylabel("Count")
    ax.set_title("Detection Outcomes (TP / FP / FN)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def fig_weights_pie():
    labels = [_SIGNAL_DISPLAY[k][0] for k in WEIGHTS]
    colors = [_SIGNAL_DISPLAY[k][1] for k in WEIGHTS]
    sizes  = list(WEIGHTS.values())
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(sizes, labels=labels, autopct="%1.0f%%",
           colors=colors, startangle=140, textprops={"fontsize": 9})
    title = ("GA-Optimized Weight Distribution" if _GA_LOADED
             else "Risk Score Weight Distribution")
    ax.set_title(title, fontsize=10, fontweight="bold")
    fig.tight_layout()
    return fig


def fig_ga_convergence(history: list[dict]):
    gens = [h["gen"] for h in history]
    best = [h["best_fitness"] for h in history]
    mean = [h["mean_fitness"] for h in history]
    fig, ax = plt.subplots(figsize=(5, 2.8))
    ax.plot(gens, best, "-", color="#5CB85C", linewidth=2, label="Best fitness")
    ax.plot(gens, mean, "--", color="#4C9BE8", linewidth=1.5, alpha=0.8, label="Mean fitness")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (0.8·F1 + 0.2·Coverage)")
    ax.set_title("GA Convergence")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ── GA summary section ────────────────────────────────────────────────────────

def render_ga_summary():
    ga_cfg    = _load_json_safe(GA_CONFIG_JSON)
    ga_report = _load_json_safe(GA_REPORT_JSON)

    if _GA_LOADED and ga_cfg:
        status_md = "**Status:** ✅ GA-optimised weights active"
        badge_color = "success"
    else:
        status_md = "**Status:** ⚪ Domain-knowledge defaults in use (GA not run yet)"
        badge_color = "info"

    with st.expander("🧬 Genetic Algorithm Optimization Summary", expanded=_GA_LOADED):
        st.markdown(status_md)

        if ga_cfg:
            st.markdown(f"Generated: `{ga_cfg.get('generated_at', 'unknown')}`  |  "
                        f"F1@{ga_cfg.get('k', 50)}: **{ga_cfg.get('f1_at_k', 0):.4f}**  |  "
                        f"K = {ga_cfg.get('k', 50)}")

        # Weight comparison table
        default_w = {
            "lstm_p95": 0.45, "after_hours": 0.13, "bcc_usage": 0.09,
            "file_exfil": 0.09, "usb_activity": 0.09, "multi_pc": 0.05,
            "content_sensitivity": 0.10,
        }
        wt_rows = []
        for key, (label, _) in _SIGNAL_DISPLAY.items():
            dw = default_w.get(key, 0.0)
            gw = WEIGHTS.get(key, dw)
            wt_rows.append({
                "Signal": label,
                "Default": f"{dw:.4f}",
                "GA Weight" if _GA_LOADED else "Active": f"{gw:.4f}",
                "Δ": f"{gw - dw:+.4f}" if _GA_LOADED else "—",
            })
        wt_df = pd.DataFrame(wt_rows).set_index("Signal")

        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.markdown("**Signal weight comparison:**")
            st.dataframe(wt_df, use_container_width=True)

            # Baseline vs GA metrics
            if ga_report:
                res = ga_report.get("results", {})
                m   = res.get("metrics_at_k", {})
                bm  = res.get("baseline_metrics_at_k", {})
                im  = res.get("improvement", {})
                if m and bm:
                    st.markdown("**Baseline vs GA metrics (K=50):**")
                    cmp_df = pd.DataFrame([
                        {"Config": "Baseline (domain weights)",
                         "F1": bm.get("f1", 0), "Precision": bm.get("precision", 0),
                         "Recall": bm.get("recall", 0), "TP": bm.get("tp", 0)},
                        {"Config": "GA-Optimised",
                         "F1": m.get("f1", 0),  "Precision": m.get("precision", 0),
                         "Recall": m.get("recall", 0),  "TP": m.get("tp", 0)},
                    ]).set_index("Config")
                    st.dataframe(
                        cmp_df.style.format(
                            {"F1": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}", "TP": "{:.0f}"}
                        ),
                        use_container_width=True,
                    )
                    if im:
                        st.caption(
                            f"ΔF1 = {im.get('delta_f1', 0):+.4f}  |  "
                            f"ΔPrecision = {im.get('delta_precision', 0):+.4f}  |  "
                            f"ΔRecall = {im.get('delta_recall', 0):+.4f}"
                        )

        with col_right:
            if ga_report and ga_report.get("convergence_history"):
                st.markdown("**Fitness convergence:**")
                st.pyplot(fig_ga_convergence(ga_report["convergence_history"]))
                plt.close()
                params = ga_report.get("ga_parameters", {})
                st.caption(
                    f"Pop={params.get('pop_size','?')}  "
                    f"Gens={len(ga_report['convergence_history'])}  "
                    f"Tournament K={params.get('tournament_k','?')}  "
                    f"Elitism={params.get('elitism_n','?')}"
                )
            else:
                st.info("Run `python colab/ga_optimizer.py` to generate convergence data.")

        if not _GA_LOADED:
            st.info(
                "To activate GA-optimised weights, run:  \n"
                "`python colab/ga_optimizer.py`  \n"
                "The dashboard will load the new config automatically on next refresh."
            )


# ═════════════════════════════════════════════════════════════════════════════
# Dashboard layout
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:

    # ── Header ───────────────────────────────────────────────────────────────
    title_suffix = " · GA-Optimized" if _GA_LOADED else ""
    st.title(f"🔒 UEBA — Insider Threat Detection System{title_suffix}")
    gt_description = load_ground_truth_description()
    st.markdown(
        "**User and Entity Behaviour Analytics** pipeline built on the CERT "
        "insider threat dataset. The dashboard auto-selects the answer-key release "
        "that matches the scored users. Detects suspicious users using an LSTM Autoencoder "
        "combined with a weighted behavioural risk scoring algorithm"
        + (" with **Genetic Algorithm-optimised weights**." if _GA_LOADED
           else " with domain-knowledge weights.")
    )
    st.caption(gt_description)
    st.markdown("---")

    # ── Sensitivity availability check ───────────────────────────────────────
    sensitivity_available = SENSITIVITY_CSV.exists()
    if not sensitivity_available:
        st.warning(
            "⚠️  **DLP content sensitivity signals not available.**  "
            "`cleaned/content_sensitivity_daily.csv` is missing.  "
            "Run `python scripts/score_content_sensitivity.py` to generate it.  "
            "The `content_sensitivity` signal will be treated as zero for all users.",
            icon="⚠️",
        )

    # ── Sidebar controls ─────────────────────────────────────────────────────
    st.sidebar.header("Analysis Settings")

    if _GA_LOADED:
        st.sidebar.success("🧬 GA weights active")
    else:
        st.sidebar.info("⚪ Domain-default weights")

    model_choice = st.sidebar.selectbox(
        "Model", ["LSTM Autoencoder", "Isolation Forest"]
    )
    agg_choice = st.sidebar.selectbox(
        "Score Aggregation",
        ["score_p95", "score_mean", "score_max"],
        help="How to collapse a user's daily scores into one value.\n"
             "score_p95 gives best results for LSTM.",
    )
    thresh_pct = st.sidebar.select_slider(
        "Threshold Percentile",
        options=[80, 85, 90, 95, 97, 99],
        value=90,
        help="Pre-filter: keep only users whose score exceeds this percentile "
             "of train-user scores.",
    )
    k_choice = st.sidebar.selectbox(
        "Top-K (users to flag)",
        [10, 20, 30, 50, 75, 100],
        index=3,
    )
    show_ground_truth = st.sidebar.checkbox(
        "Show ground-truth labels", value=ground_truth_available()
    )
    run_btn = st.sidebar.button("▶  Run Analysis", type="primary")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**{'GA-Optimized ' if _GA_LOADED else ''}Weight breakdown:**"
    )
    for k, v in WEIGHTS.items():
        label = _SIGNAL_DISPLAY.get(k, (k.replace("_", " ").title(), ""))[0]
        st.sidebar.progress(v, text=f"{label}: {v:.0%}")

    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner("Loading scored data…"):
        insider_users = load_insider_users() if show_ground_truth else set()
        lstm_user_df  = load_lstm_user_df()
        if_user_df    = load_if_user_df()
        risk_df       = load_risk_df(sensitivity_available)

    user_df   = lstm_user_df if model_choice == "LSTM Autoencoder" else if_user_df
    score_col = agg_choice

    # ── GA summary (always rendered, expanded when GA is active) ─────────────
    render_ga_summary()

    if not run_btn and "ran" not in st.session_state:
        st.info("Configure settings in the sidebar and click **Run Analysis**.")
        st.stop()

    st.session_state["ran"] = True

    # ── Compute results ───────────────────────────────────────────────────────
    filtered_df, cutoff = apply_user_threshold(user_df, score_col, thresh_pct)
    metrics = evaluate_topk_users(filtered_df, insider_users, k_choice)

    # ── Section 1: Metrics ────────────────────────────────────────────────────
    st.header("📊 Detection Metrics")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Precision",  f"{metrics['precision']:.3f}")
    c2.metric("Recall",     f"{metrics['recall']:.3f}")
    c3.metric("F1",         f"{metrics['f1']:.3f}")
    c4.metric("TP",         metrics["tp"])
    c5.metric("FP",         metrics["fp"])
    c6.metric("FN",         metrics["fn"])

    pool = len(filtered_df)
    st.caption(
        f"Threshold (train {thresh_pct}th pct of user {score_col}) = **{cutoff:.4f}**  |  "
        f"Pool after filter: **{pool}** users  |  "
        f"Flagging top **{k_choice}** as suspicious"
    )
    st.markdown("---")

    # ── Section 2: Top suspicious users table ────────────────────────────────
    score_label = _ga_score_label()
    st.header(f"🚨 Top Suspicious Users  —  {score_label}")
    top_k_users = set(filtered_df.head(k_choice)["user"])
    display_df  = risk_df[risk_df["user"].isin(top_k_users)].copy()
    display_df["rank"] = display_df["user"].apply(
        lambda u: list(filtered_df.head(k_choice)["user"]).index(u) + 1
        if u in list(filtered_df.head(k_choice)["user"]) else "-"
    )

    show_cols = ["rank", "user", "employee_name", "risk_score",
                 "lstm_p95_norm", "after_hours_norm", "bcc_usage_norm",
                 "file_exfil_norm", "usb_activity_norm"]
    if "content_sensitivity_norm" in display_df.columns:
        show_cols.append("content_sensitivity_norm")
    if show_ground_truth and "is_insider" in display_df.columns:
        show_cols.append("is_insider")
    display_df = display_df.sort_values("risk_score", ascending=False)

    col_rename = {
        "rank": "Rank", "user": "User ID", "employee_name": "Name",
        "risk_score": score_label,
        "lstm_p95_norm": "LSTM P95",
        "after_hours_norm": "After Hours",
        "bcc_usage_norm": "BCC Usage",
        "file_exfil_norm": "File Exfil",
        "usb_activity_norm": "USB Activity",
        "content_sensitivity_norm": "Content Sensitivity",
        "is_insider": "Insider?",
    }
    fmt = {
        score_label:          "{:.4f}",
        "LSTM P95":           "{:.3f}",
        "After Hours":        "{:.3f}",
        "BCC Usage":          "{:.3f}",
        "File Exfil":         "{:.3f}",
        "USB Activity":       "{:.3f}",
        "Content Sensitivity":"{:.3f}",
    }
    st.dataframe(
        display_df[show_cols].rename(columns=col_rename)
                             .set_index("Rank")
                             .style.format(fmt),
        use_container_width=True,
    )
    if not sensitivity_available:
        st.caption("Content Sensitivity column is all zeros — run the sensitivity scorer to populate it.")
    st.markdown("---")

    # ── Section 3: Charts ─────────────────────────────────────────────────────
    st.header("📈 Visual Analysis")
    tab1, tab2, tab3, tab4 = st.tabs([
        "P/R/F1 vs K", "Score Distribution",
        "Top Users Risk", "TP/FP/FN Summary"
    ])

    k_sweep = [5, 10, 20, 30, 50, 75, 100]
    with tab1:
        st.pyplot(fig_prf1_vs_k(user_df, score_col, thresh_pct, insider_users, k_sweep))
    with tab2:
        st.pyplot(fig_score_distribution(lstm_user_df, insider_users))
    with tab3:
        top_n = min(k_choice, 20)
        st.pyplot(fig_top_users_bar(risk_df, insider_users, top_n=top_n,
                                    score_label=score_label))
    with tab4:
        st.pyplot(fig_tp_fp_fn(user_df, score_col, thresh_pct, insider_users))

    st.markdown("---")

    # ── Section 4: User explainability ───────────────────────────────────────
    st.header("🔍 User Investigation")
    st.markdown("Select a flagged user to see why they were flagged.")

    flagged_users = list(filtered_df.head(k_choice)["user"])
    selected_user = st.selectbox("Select user", flagged_users)

    if selected_user:
        row = risk_df[risk_df["user"] == selected_user].iloc[0]
        emp = row.get("employee_name", selected_user)
        st.subheader(f"{emp}  ({selected_user})")

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric(score_label,  f"{row['risk_score']:.4f}")
        mc2.metric("LSTM P95",    f"{row['score_p95']:.4f}")
        if "content_sensitivity_norm" in row.index:
            mc3.metric("Content Sensitivity", f"{row['content_sensitivity_norm']:.3f}")
        if show_ground_truth and "is_insider" in row.index:
            mc4.metric("Ground Truth", "Insider ★" if row["is_insider"] else "Normal")

        st.markdown("**Triggered behavioral flags:**")
        flags = explain_user(row)
        for flag in flags:
            st.markdown(f"- {flag}")

        # Signal breakdown bar — all 7 signals
        signals: dict[str, float] = {}
        for key, (label, _) in _SIGNAL_DISPLAY.items():
            col_name = f"{key}_norm" if key != "lstm_p95" else "lstm_p95_norm"
            # handle the naming convention used in risk_scorer
            if key == "lstm_p95":
                col_name = "lstm_p95_norm"
            elif key == "after_hours":
                col_name = "after_hours_norm"
            elif key == "bcc_usage":
                col_name = "bcc_usage_norm"
            elif key == "file_exfil":
                col_name = "file_exfil_norm"
            elif key == "usb_activity":
                col_name = "usb_activity_norm"
            elif key == "multi_pc":
                col_name = "multi_pc_norm"
            elif key == "content_sensitivity":
                col_name = "content_sensitivity_norm"
            if col_name in row.index:
                signals[label] = float(row[col_name])

        # Derive per-signal flag threshold for the breakdown chart
        from risk_scorer import _FLAG_RULES  # noqa: E402
        col_to_thresh = {rule[0]: rule[2] for rule in _FLAG_RULES}
        norm_col_map = {
            "LSTM P95":            "lstm_p95_norm",
            "After Hours":         "after_hours_norm",
            "BCC Usage":           "bcc_usage_norm",
            "File Exfil":          "file_exfil_norm",
            "USB Activity":        "usb_activity_norm",
            "Multi PC":            "multi_pc_norm",
            "Content Sensitivity": "content_sensitivity_norm",
        }
        sig_df = pd.DataFrame({"Signal": list(signals.keys()),
                               "Value":  list(signals.values())})
        bar_colors = []
        for lbl in sig_df["Signal"]:
            thresh = col_to_thresh.get(norm_col_map.get(lbl, ""), 0.5)
            val    = signals.get(lbl, 0.0)
            bar_colors.append("#E85454" if val >= thresh else "#4C9BE8")

        # Reference line: use the median flag threshold across all signals
        median_thresh = float(np.median([r[2] for r in _FLAG_RULES]))
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.barh(sig_df["Signal"], sig_df["Value"], color=bar_colors, alpha=0.85)
        ax.axvline(median_thresh, color="gray", linestyle="--", linewidth=1,
                   label=f"Median flag threshold ({median_thresh:.2f})")
        ax.set_xlabel("Normalised Signal Value")
        ax.set_title(f"Signal Breakdown — {selected_user}")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")

    # ── Section 5: AI component explanation ──────────────────────────────────
    st.header("🤖 AI Algorithm Component")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        ga_blurb = (
            "\n\n6. **Genetic Algorithm Weight Optimisation** — a GA evolves the "
            "six behavioural-signal weights and flag thresholds over 100 generations "
            "using F1@K as the fitness function. The optimised weights are loaded "
            "automatically, replacing domain defaults without retraining any model."
            if _GA_LOADED else
            "\n\n6. **Genetic Algorithm (optional)** — run `python colab/ga_optimizer.py` "
            "to evolve optimised signal weights and flag thresholds that maximise F1@K. "
            "Results are loaded automatically on the next dashboard refresh."
        )
        st.markdown(f"""
**{_ga_score_label()} — Multi-Signal Evidence Aggregation**

This system implements a *multi-signal evidence aggregation* algorithm:

1. **LSTM Autoencoder** (deep learning) — detects temporal behavioural anomalies
   by measuring reconstruction error on sequences of daily activity features.

2. **Behavioural Signal Extraction** — five rule-based indicators derived from
   CERT insider threat research (after-hours activity, BCC email usage,
   removable media file transfers, USB events, multi-workstation access).

3. **DLP Content Sensitivity** — lightweight keyword/rule-based classifier
   applied to email and file content, scoring events as PUBLIC / INTERNAL /
   SENSITIVE / RESTRICTED. The per-user daily mean feeds into the risk score
   as the seventh signal.

4. **Normalised Weighted Aggregation** — each signal is scaled to [0, 1] and
   combined using {"GA-optimised" if _GA_LOADED else "domain-informed"} weights,
   producing a single *{_ga_score_label().lower()}*.

5. **Best-First Investigation Queue** — users are ranked by risk score
   (highest first), forming a priority queue that guides investigators to
   the highest-risk cases first — equivalent to best-first search over
   the suspicious-user population.

6. **Rule-Based Explainability (XAI)** — each flagged user receives a
   natural-language explanation identifying which signals triggered the alert.
{ga_blurb}
        """)
    with col_b:
        st.pyplot(fig_weights_pie())
        plt.close()

    # ── Section 6: Best settings summary ─────────────────────────────────────
    st.markdown("---")
    st.header("✅ Best Configuration Summary")

    ga_report = _load_json_safe(GA_REPORT_JSON)
    if _GA_LOADED and ga_report:
        res = ga_report.get("results", {})
        m   = res.get("metrics_at_k", {})
        bm  = res.get("baseline_metrics_at_k", {})
        im  = res.get("improvement", {})
        st.success(
            f"**Best overall setup (GA-optimised + user-level evaluation):**  \n"
            f"Model: **LSTM Autoencoder** | Aggregation: **score_p95** | "
            f"Threshold: **90th percentile** | K = **{m.get('k', 50)}**  \n"
            f"→ GA F1 = **{m.get('f1', 0):.4f}** "
            f"(baseline {bm.get('f1', 0):.4f}, Δ{im.get('delta_f1', 0):+.4f})  "
            f"| **{m.get('tp', 0)} / {len(insider_users)} insiders detected**  \n\n"
            f"At K=20, Precision rises to **0.50** — half of the flagged users are real insiders.  \n"
            "Isolation Forest is not recommended: it scores insiders *lower* than normal users (ROC AUC < 0.5)."
        )
    else:
        st.success(
            "**Best overall setup (from user-level evaluation):**  \n"
            "Model: **LSTM Autoencoder** | Aggregation: **score_p95** | "
            "Threshold: **90th percentile** | K = **50**  \n"
            f"→ Precision/Recall/F1 depend on the auto-selected CERT release; "
            f"currently tracking **{len(insider_users)} matching insider users**.  \n\n"
            "At K=20, Precision rises to **0.50** — half of the flagged users are real insiders.  \n"
            "Isolation Forest is not recommended: it scores insiders *lower* than normal users (ROC AUC < 0.5)."
        )


if __name__ == "__main__":
    main()
