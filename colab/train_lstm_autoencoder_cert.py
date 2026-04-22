"""
Per-user LSTM autoencoder for insider threat anomaly detection on CERT daily email features.

Each user gets their own autoencoder trained on their historical "normal" behavior.
Reconstruction error on a sliding window of days is the anomaly score — high error
means the user's recent behavior cannot be explained by their learned baseline.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CLEANED_DIR, MODELS_DIR  # noqa: E402

INPUT_PATH   = CLEANED_DIR / "email_user_daily_with_psychometric.csv"
MODEL_PATH   = MODELS_DIR  / "lstm_autoencoder_cert.pkl"
OUTPUT_PATH  = CLEANED_DIR / "email_user_daily_lstm_scored.csv"
METRICS_PATH = MODELS_DIR  / "lstm_autoencoder_summary.json"

# Only behavioral features — psychometric scores are static per user and add no temporal signal
BEHAVIORAL_FEATURES = [
    "email_count",
    "unique_pcs",
    "total_size",
    "avg_size",
    "total_attachments",
    "emails_with_attachments",
    "after_hours_emails",
    "avg_recipients",
    "max_recipients",
    "avg_content_words",
    "max_content_words",
    "bcc_email_count",
    "cc_email_count",
    "attachment_email_ratio",
    "after_hours_ratio",
    "bcc_ratio",
    # logon features
    "logon_count",
    "logoff_count",
    "after_hours_logons",
    "unique_logon_pcs",
    # device/USB features
    "usb_connect_count",
    "usb_disconnect_count",
    # file access features
    "file_total",
    "file_to_removable",
    "file_from_removable",
    "file_write_count",
    "file_after_hours",
]

WINDOW_SIZE = 7        # days in each input sequence
HIDDEN_DIM = 32        # LSTM hidden state size
LATENT_DIM = 16        # bottleneck dimension
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MIN_TRAIN_WINDOWS = 10  # skip users with fewer training windows than this
TRAIN_FRACTION = 0.80


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder_lstm = nn.LSTM(n_features, hidden_dim, batch_first=True)
        self.enc_fc = nn.Linear(hidden_dim, latent_dim)
        self.decoder_lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.out_fc = nn.Linear(hidden_dim, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        _, (h_n, _) = self.encoder_lstm(x)
        latent = self.enc_fc(h_n.squeeze(0))                          # (batch, latent_dim)
        dec_in = latent.unsqueeze(1).repeat(1, x.size(1), 1)          # (batch, seq_len, latent_dim)
        dec_out, _ = self.decoder_lstm(dec_in)                        # (batch, seq_len, hidden_dim)
        return self.out_fc(dec_out)                                    # (batch, seq_len, n_features)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def make_windows(scaled: np.ndarray, window_size: int) -> np.ndarray:
    """Slide a window over the sequence. Returns (n_windows, window_size, n_features)."""
    n = len(scaled)
    if n < window_size:
        return np.empty((0, window_size, scaled.shape[1]), dtype=np.float32)
    windows = np.stack([scaled[i : i + window_size] for i in range(n - window_size + 1)])
    return windows.astype(np.float32)


def reconstruction_errors(model: LSTMAutoencoder, windows: np.ndarray, device: torch.device) -> np.ndarray:
    """Return per-window MSE reconstruction error."""
    model.eval()
    errors = []
    with torch.no_grad():
        for start in range(0, len(windows), 256):
            batch = torch.tensor(windows[start : start + 256], dtype=torch.float32, device=device)
            recon = model(batch)
            mse = ((batch - recon) ** 2).mean(dim=(1, 2)).cpu().numpy()
            errors.append(mse)
    return np.concatenate(errors) if errors else np.array([])


# ---------------------------------------------------------------------------
# Per-user training
# ---------------------------------------------------------------------------

def train_user_model(
    user_days: pd.DataFrame,
    device: torch.device,
) -> dict:
    """Train one LSTM autoencoder for a single user. Returns per-user artifacts dict."""
    user_days = user_days.sort_values("email_day").reset_index(drop=True)
    n_days = len(user_days)

    # Temporal train/test split
    cutoff_idx = max(1, int(n_days * TRAIN_FRACTION))
    train_rows = user_days.iloc[:cutoff_idx]
    all_rows = user_days

    # Per-user feature scaling fitted on training days only
    scaler = MinMaxScaler()
    train_features = train_rows[BEHAVIORAL_FEATURES].fillna(0).values
    scaler.fit(train_features)

    train_scaled = scaler.transform(train_features).astype(np.float32)
    all_scaled = scaler.transform(all_rows[BEHAVIORAL_FEATURES].fillna(0).values).astype(np.float32)

    train_windows = make_windows(train_scaled, WINDOW_SIZE)

    if len(train_windows) < MIN_TRAIN_WINDOWS:
        return {"skipped": True, "reason": f"only {len(train_windows)} train windows"}

    # Build and train model
    model = LSTMAutoencoder(len(BEHAVIORAL_FEATURES), HIDDEN_DIM, LATENT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    dataset = TensorDataset(torch.tensor(train_windows))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for _ in range(EPOCHS):
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()

    # Reconstruction errors on training windows → set thresholds
    train_errors = reconstruction_errors(model, train_windows, device)
    suspicious_threshold = float(np.percentile(train_errors, 95))
    high_threshold = float(np.percentile(train_errors, 99))

    # Score all windows for this user
    all_windows = make_windows(all_scaled, WINDOW_SIZE)
    all_errors = reconstruction_errors(model, all_windows, device) if len(all_windows) > 0 else np.array([])

    # Map window errors back to day indices (window i → day i + WINDOW_SIZE - 1)
    day_errors = np.full(n_days, np.nan)
    for i, err in enumerate(all_errors):
        day_errors[i + WINDOW_SIZE - 1] = float(err)

    # Normalize errors to [0, 1] using training error range
    train_err_min = float(train_errors.min())
    train_err_max = float(train_errors.max()) + 1e-9
    day_scores = np.clip((day_errors - train_err_min) / (train_err_max - train_err_min), 0.0, 1.0)

    # Normalize thresholds to the same scale
    norm_suspicious = float(np.clip((suspicious_threshold - train_err_min) / (train_err_max - train_err_min), 0.0, 1.0))
    norm_high = float(np.clip((high_threshold - train_err_min) / (train_err_max - train_err_min), 0.0, 1.0))

    return {
        "skipped": False,
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
        "scaler": scaler,
        "suspicious_threshold": norm_suspicious,
        "high_threshold": norm_high,
        "raw_suspicious_threshold": suspicious_threshold,
        "raw_high_threshold": high_threshold,
        "train_err_min": train_err_min,
        "train_err_max": train_err_max,
        "train_end_day": str(train_rows["email_day"].max().date()),
        "n_train_days": len(train_rows),
        "n_total_days": n_days,
        "day_scores": day_scores.tolist(),
        "day_raw_errors": day_errors.tolist(),
    }


# ---------------------------------------------------------------------------
# Severity assignment
# ---------------------------------------------------------------------------

def assign_severity(score: float, suspicious_threshold: float, high_threshold: float) -> str:
    if np.isnan(score):
        return "undetermined"
    if score >= high_threshold:
        return "high"
    if score >= suspicious_threshold:
        return "suspicious"
    return "normal"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_feature_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH)
    missing = [c for c in BEHAVIORAL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    df["email_day"] = pd.to_datetime(df["email_day"], errors="coerce")
    df = df.dropna(subset=["email_day"]).sort_values(["user", "email_day"]).reset_index(drop=True)
    return df


def score_all_users(df: pd.DataFrame, device: torch.device) -> tuple[dict, pd.DataFrame]:
    users = sorted(df["user"].unique())
    n_users = len(users)

    all_user_artifacts: dict[str, dict] = {}
    scored_rows = []

    # Global suspicious/high thresholds (median of per-user thresholds for users with models)
    per_user_suspicious: list[float] = []
    per_user_high: list[float] = []

    for idx, user in enumerate(users, 1):
        if idx % 100 == 0 or idx == n_users:
            print(f"  Training user {idx}/{n_users} ({user})", flush=True)

        user_df = df[df["user"] == user].copy()
        artifacts = train_user_model(user_df, device)
        all_user_artifacts[user] = artifacts

        user_df = user_df.sort_values("email_day").reset_index(drop=True)

        if artifacts["skipped"]:
            user_df["lstm_raw_error"] = np.nan
            user_df["lstm_score"] = np.nan
            user_df["lstm_flag"] = 0
            user_df["lstm_risk_severity"] = "undetermined"
        else:
            day_scores = np.array(artifacts["day_scores"])
            day_errors = np.array(artifacts["day_raw_errors"])
            user_df["lstm_raw_error"] = day_errors
            user_df["lstm_score"] = day_scores
            user_df["lstm_flag"] = (
                (~np.isnan(day_scores)) & (day_scores >= artifacts["suspicious_threshold"])
            ).astype(int)
            user_df["lstm_risk_severity"] = [
                assign_severity(s, artifacts["suspicious_threshold"], artifacts["high_threshold"])
                for s in day_scores
            ]
            per_user_suspicious.append(artifacts["suspicious_threshold"])
            per_user_high.append(artifacts["high_threshold"])

        scored_rows.append(user_df)

    scored_df = pd.concat(scored_rows, ignore_index=True)

    global_suspicious = float(np.median(per_user_suspicious)) if per_user_suspicious else 0.5
    global_high = float(np.median(per_user_high)) if per_user_high else 0.7

    summary_artifacts = {
        "window_size": WINDOW_SIZE,
        "feature_columns": BEHAVIORAL_FEATURES,
        "hidden_dim": HIDDEN_DIM,
        "latent_dim": LATENT_DIM,
        "train_fraction": TRAIN_FRACTION,
        "global_suspicious_threshold": global_suspicious,
        "global_high_threshold": global_high,
        "users": all_user_artifacts,
    }
    return summary_artifacts, scored_df


def split_label(row: pd.Series, train_cutoffs: dict[str, str]) -> str:
    cutoff = train_cutoffs.get(row["user"])
    if cutoff is None:
        return "unknown"
    return "train" if str(row["email_day"].date()) <= cutoff else "test"


def build_summary(scored_df: pd.DataFrame, artifacts: dict) -> dict:
    trained_users = sum(1 for v in artifacts["users"].values() if not v.get("skipped", True))
    skipped_users = sum(1 for v in artifacts["users"].values() if v.get("skipped", True))
    valid = scored_df.dropna(subset=["lstm_score"])
    top_anomalies = (
        valid.sort_values("lstm_score", ascending=False)
        .head(20)[["user", "email_day", "lstm_score", "lstm_risk_severity"]]
        .assign(email_day=lambda f: f["email_day"].astype(str))
        .to_dict(orient="records")
    )
    return {
        "rows": int(len(scored_df)),
        "users": int(scored_df["user"].nunique()),
        "trained_users": trained_users,
        "skipped_users": skipped_users,
        "window_size": WINDOW_SIZE,
        "hidden_dim": HIDDEN_DIM,
        "latent_dim": LATENT_DIM,
        "global_suspicious_threshold": artifacts["global_suspicious_threshold"],
        "global_high_threshold": artifacts["global_high_threshold"],
        "suspicious_rows": int((scored_df["lstm_risk_severity"] == "suspicious").sum()),
        "high_rows": int((scored_df["lstm_risk_severity"] == "high").sum()),
        "undetermined_rows": int((scored_df["lstm_risk_severity"] == "undetermined").sum()),
        "top_anomalies": top_anomalies,
    }


def save_outputs(artifacts: dict, scored_df: pd.DataFrame, summary: dict) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Strip large day_scores lists from artifacts before pickling to keep file manageable
    save_artifacts = dict(artifacts)
    save_artifacts["users"] = {
        user: {k: v for k, v in user_data.items() if k not in ("day_scores", "day_raw_errors")}
        for user, user_data in artifacts["users"].items()
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(save_artifacts, f)

    out_df = scored_df.copy()
    out_df["email_day"] = out_df["email_day"].dt.strftime("%Y-%m-%d")
    out_df.to_csv(OUTPUT_PATH, index=False)
    METRICS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading feature data...")
    df = load_feature_data()
    print(f"Loaded {len(df)} rows, {df['user'].nunique()} users")

    print("Training per-user LSTM autoencoders...")
    artifacts, scored_df = score_all_users(df, device)

    # Add dataset_split label using per-user train cutoffs
    train_cutoffs = {
        user: data["train_end_day"]
        for user, data in artifacts["users"].items()
        if not data.get("skipped", True)
    }
    scored_df["dataset_split"] = scored_df.apply(lambda r: split_label(r, train_cutoffs), axis=1)

    summary = build_summary(scored_df, artifacts)
    save_outputs(artifacts, scored_df, summary)

    print(f"\nSaved scored dataset  → {OUTPUT_PATH}")
    print(f"Saved model artifacts → {MODEL_PATH}")
    print(f"Saved summary         → {METRICS_PATH}")
    print(f"Trained users: {summary['trained_users']} | Skipped: {summary['skipped_users']}")
    print(f"High-risk rows: {summary['high_rows']} | Suspicious rows: {summary['suspicious_rows']}")

    preview = (
        scored_df.dropna(subset=["lstm_score"])
        .sort_values("lstm_score", ascending=False)
        .head(5)[["user", "email_day", "dataset_split", "lstm_score", "lstm_risk_severity"]]
    )
    print("\nTop 5 anomalies:")
    print(preview.to_string(index=False))


if __name__ == "__main__":
    main()
