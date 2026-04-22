"""Run the local CERT email pipeline on a PC without Colab."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STEPS = {
    "clean": [sys.executable, str(ROOT / "scripts" / "clean_cert_email_data.py")],
    "train": [sys.executable, str(ROOT / "colab" / "train_isolation_forest_cert.py")],
    "visualize": [sys.executable, str(ROOT / "colab" / "visualize_isolation_forest_cert.py")],
    "train_lstm": [sys.executable, str(ROOT / "colab" / "train_lstm_autoencoder_cert.py")],
    "visualize_lstm": [sys.executable, str(ROOT / "colab" / "visualize_lstm_autoencoder_cert.py")],
}


def run_step(name: str) -> None:
    print(f"\n=== Running: {name} ===")
    subprocess.run(STEPS[name], check=True, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local CERT pipeline steps.")
    parser.add_argument(
        "--skip-clean",
        action="store_true",
        help="Skip the cleaning step if cleaned files already exist.",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Run only training and visualization on existing cleaned files.",
    )
    parser.add_argument(
        "--skip-lstm",
        action="store_true",
        help="Skip LSTM autoencoder training and visualization.",
    )
    parser.add_argument(
        "--lstm-only",
        action="store_true",
        help="Run only the LSTM autoencoder steps on existing cleaned files.",
    )
    args = parser.parse_args()

    email_input = ROOT / "archive" / "email.csv"
    psychometric_input = ROOT / "archive" / "psychometric.csv"
    if not email_input.exists() or not psychometric_input.exists():
        raise FileNotFoundError("Expected archive/email.csv and archive/psychometric.csv before running the pipeline.")

    if args.lstm_only:
        run_step("train_lstm")
        run_step("visualize_lstm")
    else:
        if not args.train_only and not args.skip_clean:
            run_step("clean")

        run_step("train")
        run_step("visualize")

        if not args.skip_lstm:
            run_step("train_lstm")
            run_step("visualize_lstm")

    print("\nPipeline finished.")
    print(f"IForest scored data : {ROOT / 'cleaned' / 'email_user_daily_scored.csv'}")
    print(f"LSTM scored data    : {ROOT / 'cleaned' / 'email_user_daily_lstm_scored.csv'}")
    print(f"Models              : {ROOT / 'models'}")
    print(f"Plots               : {ROOT / 'plots'}")


if __name__ == "__main__":
    main()
