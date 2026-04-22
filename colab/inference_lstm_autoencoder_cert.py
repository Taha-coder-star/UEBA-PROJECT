"""Reusable inference utilities for the CERT LSTM autoencoder model."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))   # for train_lstm_autoencoder_cert
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for config
from config import MODELS_DIR  # noqa: E402
from train_lstm_autoencoder_cert import (
    BEHAVIORAL_FEATURES,
    WINDOW_SIZE,
    HIDDEN_DIM,
    LATENT_DIM,
    LSTMAutoencoder,
    assign_severity,
    make_windows,
    reconstruction_errors,
)


MODEL_PATH = MODELS_DIR / "lstm_autoencoder_cert.pkl"


def load_artifacts(model_path: str | Path = MODEL_PATH) -> dict:
    with open(model_path, "rb") as f:
        return pickle.load(f)


def _rebuild_model(state: dict, device: torch.device) -> LSTMAutoencoder:
    model = LSTMAutoencoder(len(BEHAVIORAL_FEATURES), HIDDEN_DIM, LATENT_DIM).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def score_user_days(
    user: str,
    user_df: pd.DataFrame,
    artifacts: dict,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Score a DataFrame of daily feature rows for a single user.

    user_df must contain all BEHAVIORAL_FEATURES columns and be sorted by date.
    Returns user_df with lstm_raw_error, lstm_score, lstm_flag, lstm_risk_severity added.
    """
    device = device or torch.device("cpu")
    result = user_df.copy().reset_index(drop=True)

    user_artifacts = artifacts.get("users", {}).get(user)
    if user_artifacts is None or user_artifacts.get("skipped", True):
        result["lstm_raw_error"] = np.nan
        result["lstm_score"] = np.nan
        result["lstm_flag"] = 0
        result["lstm_risk_severity"] = "undetermined"
        return result

    scaler = user_artifacts["scaler"]
    suspicious_threshold = user_artifacts["suspicious_threshold"]
    high_threshold = user_artifacts["high_threshold"]
    train_err_min = user_artifacts["train_err_min"]
    train_err_max = user_artifacts["train_err_max"]

    scaled = scaler.transform(result[BEHAVIORAL_FEATURES].fillna(0).values).astype(np.float32)
    windows = make_windows(scaled, WINDOW_SIZE)

    model = _rebuild_model(user_artifacts["model_state"], device)
    errors = reconstruction_errors(model, windows, device) if len(windows) > 0 else np.array([])

    day_errors = np.full(len(result), np.nan)
    for i, err in enumerate(errors):
        day_errors[i + WINDOW_SIZE - 1] = float(err)

    day_scores = np.clip(
        (day_errors - train_err_min) / (train_err_max - train_err_min + 1e-9), 0.0, 1.0
    )

    result["lstm_raw_error"] = day_errors
    result["lstm_score"] = day_scores
    result["lstm_flag"] = (
        (~np.isnan(day_scores)) & (day_scores >= suspicious_threshold)
    ).astype(int)
    result["lstm_risk_severity"] = [
        assign_severity(s, suspicious_threshold, high_threshold) for s in day_scores
    ]
    return result


def score_dataframe(
    df: pd.DataFrame,
    artifacts: dict | None = None,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Score a full multi-user DataFrame. df must contain an email_day and user column."""
    artifacts = artifacts or load_artifacts()
    device = device or torch.device("cpu")

    scored_chunks = []
    for user, user_df in df.groupby("user", sort=False):
        user_df = user_df.sort_values("email_day").reset_index(drop=True)
        scored_chunks.append(score_user_days(str(user), user_df, artifacts, device))

    return pd.concat(scored_chunks, ignore_index=True)


def score_single_user_sequence(
    user: str,
    feature_rows: list[dict],
    artifacts: dict | None = None,
) -> list[dict]:
    """Score a chronologically-ordered list of daily feature dicts for one user.

    Each dict must contain all BEHAVIORAL_FEATURES keys.
    Returns the same list with lstm_score, lstm_flag, lstm_risk_severity added.
    """
    artifacts = artifacts or load_artifacts()
    df = pd.DataFrame(feature_rows)
    result = score_user_days(user, df, artifacts)
    return result.to_dict(orient="records")
