"""Weighted Risk Scoring -- AI Algorithm Component.

This module implements a multi-signal risk scoring system for insider threat
detection.  It combines the LSTM autoencoder anomaly signal with six
behavioral / content indicators drawn from CERT insider threat research.

Algorithm class: Weighted Evidence Aggregation with Best-First Investigation Queue

Why this qualifies as an AI component:
  - Learns nothing from labels (unsupervised, deployable with zero ground truth)
  - Normalises heterogeneous signals to a common [0, 1] scale (feature engineering)
  - Combines signals using domain-informed or GA-optimised weights
  - Produces a ranked priority queue -- equivalent to best-first search over
    the space of suspicious users
  - Generates per-user natural-language explanations (rule-based XAI)

Weights (must sum to 1.0) — default domain-knowledge baseline:
  lstm_p95             0.45  -- LSTM reconstruction-error p95 (main model)
  after_hours          0.13  -- fraction of emails / logons after business hours
  bcc_usage            0.09  -- BCC email ratio (hiding recipients)
  file_exfil           0.09  -- files copied to removable media
  usb_activity         0.09  -- USB connect events (removable storage use)
  multi_pc             0.05  -- distinct workstations accessed
  content_sensitivity  0.10  -- DLP content sensitivity score (keyword / rule)

When models/ga_optimized_config.json exists (produced by colab/ga_optimizer.py),
weights and flag thresholds are replaced with GA-optimised values at import time.
A 6-signal GA config (no content_sensitivity key) is accepted: the six weights
are scaled by 0.9 and content_sensitivity receives its default weight of 0.10.

Reference: Greitzer et al. (2010), "Insider Threat Indicators", CERT/CC.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONTENT_SENSITIVITY_DEFAULT_W = 0.10    # weight added when GA config is 6-signal

# Default (domain-knowledge) weights — 7 signals, sum = 1.0
_DEFAULT_WEIGHTS: dict[str, float] = {
    "lstm_p95":            0.45,
    "after_hours":         0.13,
    "bcc_usage":           0.09,
    "file_exfil":          0.09,
    "usb_activity":        0.09,
    "multi_pc":            0.05,
    "content_sensitivity": 0.10,
}

_DEFAULT_THRESHOLDS: dict[str, float] = {
    "lstm_p95":            0.70,
    "after_hours":         0.50,
    "bcc_usage":           0.50,
    "file_exfil":          0.50,
    "usb_activity":        0.50,
    "multi_pc":            0.50,
    "content_sensitivity": 0.50,
}

_6_SIGNAL_KEYS = frozenset({
    "lstm_p95", "after_hours", "bcc_usage",
    "file_exfil", "usb_activity", "multi_pc",
})

# ---------------------------------------------------------------------------
# GA config loader — tries models/ga_optimized_config.json at import time
# ---------------------------------------------------------------------------

_REPO_DIR = Path(os.environ.get("DLP_ROOT",
                                str(Path(__file__).resolve().parent.parent)))
_GA_CONFIG_PATH = _REPO_DIR / "models" / "ga_optimized_config.json"


def _load_ga_config() -> tuple[dict[str, float], dict[str, float], bool]:
    """Return (weights, thresholds, ga_loaded).

    Handles two GA config formats:
      7-signal: has 'content_sensitivity' key — used directly.
      6-signal: legacy format without content_sensitivity — the six weights
                are scaled by (1 - _CONTENT_SENSITIVITY_DEFAULT_W) and
                content_sensitivity receives _CONTENT_SENSITIVITY_DEFAULT_W.
    Falls back to domain defaults if the file is absent or invalid.
    """
    if not _GA_CONFIG_PATH.exists():
        return _DEFAULT_WEIGHTS.copy(), _DEFAULT_THRESHOLDS.copy(), False
    try:
        cfg = json.loads(_GA_CONFIG_PATH.read_text())
        w = cfg.get("weights", {})
        t = cfg.get("thresholds", {})

        if not (_6_SIGNAL_KEYS <= set(w)):
            raise ValueError("GA config missing core signal keys")

        if "content_sensitivity" not in w:
            # Backward-compat: scale 6 signals down to leave room for the 7th
            scale = 1.0 - _CONTENT_SENSITIVITY_DEFAULT_W
            w = {k: v * scale for k, v in w.items()}
            w["content_sensitivity"] = _CONTENT_SENSITIVITY_DEFAULT_W
            t["content_sensitivity"] = _DEFAULT_THRESHOLDS["content_sensitivity"]

        # Re-normalise to guard against rounding drift
        total = sum(w.values())
        w = {k: v / total for k, v in w.items()}
        return w, t, True

    except Exception as exc:  # noqa: BLE001
        import warnings
        warnings.warn(
            f"[risk_scorer] Could not load GA config: {exc} — using defaults.",
            stacklevel=2,
        )
        return _DEFAULT_WEIGHTS.copy(), _DEFAULT_THRESHOLDS.copy(), False


WEIGHTS, _ga_thresholds, _GA_LOADED = _load_ga_config()

if _GA_LOADED:
    import warnings
    warnings.warn(
        f"[risk_scorer] Loaded GA-optimised weights from {_GA_CONFIG_PATH.name}. "
        "Delete that file to revert to domain defaults.",
        stacklevel=1,
    )

# ---------------------------------------------------------------------------
# Flag rules — drive human-readable explanation text
# (thresholds come from GA config when available)
# ---------------------------------------------------------------------------

_FLAG_RULES: list[tuple[str, str, float, str]] = [
    ("lstm_p95_norm",              ">=", _ga_thresholds["lstm_p95"],
     "High LSTM anomaly score -- unusual behavioural patterns over time"),
    ("after_hours_norm",           ">=", _ga_thresholds["after_hours"],
     "Elevated after-hours email / login activity"),
    ("bcc_usage_norm",             ">=", _ga_thresholds["bcc_usage"],
     "Abnormal BCC email usage (potential hidden recipients)"),
    ("file_exfil_norm",            ">=", _ga_thresholds["file_exfil"],
     "Files copied to removable media (possible data exfiltration)"),
    ("usb_activity_norm",          ">=", _ga_thresholds["usb_activity"],
     "Frequent USB device connections"),
    ("multi_pc_norm",              ">=", _ga_thresholds["multi_pc"],
     "Logging in from an unusual number of workstations"),
    ("content_sensitivity_norm",   ">=", _ga_thresholds["content_sensitivity"],
     "High DLP content sensitivity -- sensitive/restricted data in emails or files"),
]


# ---------------------------------------------------------------------------
# Step 0 (optional) -- load content sensitivity signals
# ---------------------------------------------------------------------------

def load_sensitivity_signals(
    sensitivity_csv: Path | str | None = None,
) -> pd.DataFrame | None:
    """Load cleaned/content_sensitivity_daily.csv and aggregate to user level.

    Returns a DataFrame with columns: user, content_sensitivity_rate
    (mean daily max_sensitivity_score, range 0–3), or None if the file
    is not found.

    sensitivity_csv -- explicit path; defaults to
                       <repo>/cleaned/content_sensitivity_daily.csv
    """
    if sensitivity_csv is None:
        sensitivity_csv = _REPO_DIR / "cleaned" / "content_sensitivity_daily.csv"
    else:
        sensitivity_csv = Path(sensitivity_csv)

    if not sensitivity_csv.exists():
        return None

    try:
        sdf = pd.read_csv(sensitivity_csv,
                          usecols=["user", "max_sensitivity_score",
                                   "sensitive_event_count",
                                   "restricted_event_count"])
        user_agg = sdf.groupby("user").agg(
            content_sensitivity_rate=("max_sensitivity_score", "mean"),
            total_sensitive_events=("sensitive_event_count", "sum"),
            total_restricted_events=("restricted_event_count", "sum"),
        ).reset_index()
        return user_agg
    except Exception as exc:  # noqa: BLE001
        import warnings
        warnings.warn(
            f"[risk_scorer] Could not load sensitivity CSV: {exc} — signal set to 0.",
            stacklevel=2,
        )
        return None


# ---------------------------------------------------------------------------
# Step 1 -- compute behavioral signals from the IF-scored CSV (row level)
# ---------------------------------------------------------------------------

def compute_behavioral_signals(idf: pd.DataFrame) -> pd.DataFrame:
    """Aggregate row-level behavioral features to one row per user.

    idf must contain columns:
        user, after_hours_ratio, bcc_ratio, file_to_removable, file_total,
        usb_connect_count, after_hours_logons, logon_count, unique_logon_pcs,
        employee_name (optional), dataset_split

    Returns one row per user with raw (un-normalised) signal values.
    """
    required = [
        "user", "after_hours_ratio", "bcc_ratio",
        "file_to_removable", "file_total",
        "usb_connect_count", "after_hours_logons",
        "logon_count", "unique_logon_pcs",
    ]
    missing = [c for c in required if c not in idf.columns]
    if missing:
        raise ValueError(f"Missing columns in idf: {missing}")

    agg = idf.groupby("user").agg(
        after_hours_rate   = ("after_hours_ratio",  "mean"),
        bcc_rate           = ("bcc_ratio",           "mean"),
        total_file_exfil   = ("file_to_removable",   "sum"),
        total_file_ops     = ("file_total",           "sum"),
        total_usb          = ("usb_connect_count",    "sum"),
        total_ah_logons    = ("after_hours_logons",   "sum"),
        total_logons       = ("logon_count",           "sum"),
        max_unique_pcs     = ("unique_logon_pcs",      "max"),
        dataset_split      = ("dataset_split",         lambda x: x.mode().iloc[0]),
    ).reset_index()

    if "employee_name" in idf.columns:
        names = idf.groupby("user")["employee_name"].first().reset_index()
        agg = agg.merge(names, on="user", how="left")
    else:
        agg["employee_name"] = agg["user"]

    agg["file_exfil_rate"] = agg["total_file_exfil"] / (agg["total_file_ops"] + 1)
    agg["ah_logon_rate"]   = agg["total_ah_logons"]  / (agg["total_logons"] + 1)
    return agg


# ---------------------------------------------------------------------------
# Step 2 -- normalise signals and compute weighted risk score
# ---------------------------------------------------------------------------

def _minmax(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


def compute_risk_scores(
    lstm_user_df: pd.DataFrame,
    behavioral_df: pd.DataFrame,
    insider_users: set[str] | None = None,
    sensitivity_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Combine LSTM p95 score, behavioural signals, and DLP content sensitivity
    into a final risk score.

    lstm_user_df  -- output of compute_user_scores() from user_level_eval.py;
                     must contain columns: user, score_p95
    behavioral_df -- output of compute_behavioral_signals()
    insider_users -- optional ground-truth set for labelling
    sensitivity_df-- optional output of load_sensitivity_signals();
                     when None the content_sensitivity signal is zeroed out

    Returns a DataFrame with one row per user, sorted by risk_score descending.
    """
    df = lstm_user_df[["user", "score_p95", "dataset_split"]].merge(
        behavioral_df[[
            "user", "after_hours_rate", "bcc_rate",
            "file_exfil_rate", "total_usb", "max_unique_pcs", "employee_name",
        ]],
        on="user", how="left",
    ).fillna(0)

    # Optionally merge content sensitivity
    if sensitivity_df is not None and not sensitivity_df.empty:
        df = df.merge(
            sensitivity_df[["user", "content_sensitivity_rate"]],
            on="user", how="left",
        ).fillna(0)
    else:
        df["content_sensitivity_rate"] = 0.0

    # Normalise each signal independently to [0, 1]
    df["lstm_p95_norm"]             = _minmax(df["score_p95"])
    df["after_hours_norm"]          = _minmax(df["after_hours_rate"])
    df["bcc_usage_norm"]            = _minmax(df["bcc_rate"])
    df["file_exfil_norm"]           = _minmax(df["file_exfil_rate"])
    df["usb_activity_norm"]         = _minmax(df["total_usb"])
    df["multi_pc_norm"]             = _minmax(df["max_unique_pcs"])
    # content_sensitivity_rate is 0–3; divide by 3 before minmax to keep scale
    df["content_sensitivity_norm"]  = _minmax(
        (df["content_sensitivity_rate"] / 3.0).clip(0, 1)
    )

    # Weighted aggregation (weights sum to 1.0)
    df["risk_score"] = (
        WEIGHTS["lstm_p95"]             * df["lstm_p95_norm"]
      + WEIGHTS["after_hours"]          * df["after_hours_norm"]
      + WEIGHTS["bcc_usage"]            * df["bcc_usage_norm"]
      + WEIGHTS["file_exfil"]           * df["file_exfil_norm"]
      + WEIGHTS["usb_activity"]         * df["usb_activity_norm"]
      + WEIGHTS["multi_pc"]             * df["multi_pc_norm"]
      + WEIGHTS["content_sensitivity"]  * df["content_sensitivity_norm"]
    )

    if insider_users is not None:
        df["is_insider"] = df["user"].isin(insider_users).astype(int)

    return df.sort_values("risk_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 3 -- best-first investigation queue
# ---------------------------------------------------------------------------

def build_investigation_queue(risk_df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """Return the top-N highest-risk users as a priority investigation queue.

    Equivalent to best-first search over the user population where
    the heuristic is the weighted risk score.  Investigators work through
    the queue from rank 1 downward, maximising expected insider detections
    per unit of investigation effort.
    """
    cols = [
        "user", "employee_name", "risk_score",
        "lstm_p95_norm", "after_hours_norm", "bcc_usage_norm",
        "file_exfil_norm", "usb_activity_norm", "multi_pc_norm",
        "content_sensitivity_norm",
    ]
    for opt in ("is_insider", "explanation"):
        if opt in risk_df.columns:
            cols.append(opt)
    present = [c for c in cols if c in risk_df.columns]
    queue = risk_df.head(top_n)[present].copy()
    queue.insert(0, "priority_rank", range(1, len(queue) + 1))
    return queue


# ---------------------------------------------------------------------------
# Step 4 -- per-user explainability
# ---------------------------------------------------------------------------

def explain_user(user_row: pd.Series) -> list[str]:
    """Return a list of triggered behavioural flags for one user.

    Each flag is a plain-English sentence suitable for a dashboard or report.
    """
    flags: list[str] = []
    for col, direction, threshold, text in _FLAG_RULES:
        if col not in user_row.index:
            continue
        val = float(user_row[col])
        triggered = (val >= threshold) if direction == ">=" else (val < threshold)
        if triggered:
            flags.append(text)
    if not flags:
        flags.append("No strong individual behavioural indicators -- pattern is subtle.")
    return flags


def explain_dataframe(risk_df: pd.DataFrame) -> pd.DataFrame:
    """Add an 'explanation' column to risk_df (one string per user)."""
    out = risk_df.copy()
    out["explanation"] = out.apply(
        lambda r: "; ".join(explain_user(r)), axis=1
    )
    return out
