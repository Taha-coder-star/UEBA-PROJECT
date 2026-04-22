"""Reusable inference utilities for the CERT Isolation Forest model."""

from __future__ import annotations

import sys
from pathlib import Path
import pickle

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MODELS_DIR  # noqa: E402

MODEL_PATH = MODELS_DIR / "isolation_forest_cert.pkl"


def load_artifacts(model_path: str | Path = MODEL_PATH) -> dict:
    with open(model_path, "rb") as file_obj:
        return pickle.load(file_obj)


def assign_severity(score: float, suspicious_threshold: float, high_threshold: float) -> str:
    if score >= high_threshold:
        return "high"
    if score >= suspicious_threshold:
        return "suspicious"
    return "normal"


def score_feature_rows(feature_df: pd.DataFrame, artifacts: dict | None = None) -> pd.DataFrame:
    artifacts = artifacts or load_artifacts()
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    feature_columns = artifacts["feature_columns"]

    scoring_df = feature_df.copy()
    features = scoring_df[feature_columns].copy().fillna(0)
    raw_scores = -model.score_samples(features)
    normalized_scores = scaler.transform(raw_scores.reshape(-1, 1)).ravel().clip(0.0, 1.0)

    scoring_df["iforest_raw_score"] = raw_scores
    scoring_df["iforest_score"] = normalized_scores
    scoring_df["risk_severity"] = [
        assign_severity(float(score), artifacts["suspicious_threshold"], artifacts["high_threshold"])
        for score in normalized_scores
    ]
    scoring_df["iforest_flag"] = (scoring_df["risk_severity"] != "normal").astype(int)
    return scoring_df


def score_single_row(row: dict, artifacts: dict | None = None) -> dict:
    scored = score_feature_rows(pd.DataFrame([row]), artifacts=artifacts)
    return scored.iloc[0].to_dict()
