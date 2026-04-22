"""Run the full local CERT email pipeline from raw data to optional app launch."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
REQUIREMENTS = ROOT / "requirements-local.txt"

STEPS = {
    "install": [PYTHON, "-m", "pip", "install", "-r", str(REQUIREMENTS)],
    "clean": [PYTHON, str(ROOT / "scripts" / "clean_cert_email_data.py")],
    "train": [PYTHON, str(ROOT / "colab" / "train_isolation_forest_cert.py")],
    "visualize": [PYTHON, str(ROOT / "colab" / "visualize_isolation_forest_cert.py")],
    "app": [PYTHON, "-m", "streamlit", "run", str(ROOT / "app" / "monitoring_app.py")],
}


OUTPUTS = {
    "cleaned_email": ROOT / "cleaned" / "email_cleaned.csv",
    "daily_features": ROOT / "cleaned" / "email_user_daily_features.csv",
    "scored_data": ROOT / "cleaned" / "email_user_daily_scored.csv",
    "model": ROOT / "models" / "isolation_forest_cert.pkl",
    "summary": ROOT / "models" / "isolation_forest_summary.json",
    "plots": ROOT / "plots",
}


def run_step(name: str) -> None:
    print(f"\n=== Running: {name} ===")
    subprocess.run(STEPS[name], check=True, cwd=ROOT)


def validate_inputs() -> None:
    email_input = ROOT / "archive" / "email.csv"
    psychometric_input = ROOT / "archive" / "psychometric.csv"
    if not email_input.exists() or not psychometric_input.exists():
        raise FileNotFoundError(
            "Expected archive/email.csv and archive/psychometric.csv before running the full pipeline."
        )


def print_outputs() -> None:
    print("\nPipeline finished.")
    print(f"Cleaned email: {OUTPUTS['cleaned_email']}")
    print(f"Daily features: {OUTPUTS['daily_features']}")
    print(f"Scored data: {OUTPUTS['scored_data']}")
    print(f"Model: {OUTPUTS['model']}")
    print(f"Summary: {OUTPUTS['summary']}")
    print(f"Plots: {OUTPUTS['plots']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full local CERT pipeline sequentially from raw data."
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install local Python dependencies before running the pipeline.",
    )
    parser.add_argument(
        "--launch-app",
        action="store_true",
        help="Launch the Streamlit monitoring app after clean, train, and visualize finish.",
    )
    parser.add_argument(
        "--skip-clean",
        action="store_true",
        help="Skip cleaning if cleaned CSVs already exist and you only want to retrain from cleaned data.",
    )
    args = parser.parse_args()

    validate_inputs()

    if args.install_deps:
        run_step("install")

    if not args.skip_clean:
        run_step("clean")

    run_step("train")
    run_step("visualize")
    print_outputs()

    if args.launch_app:
        print("\nLaunching Streamlit app. Press Ctrl+C in the terminal to stop it.")
        run_step("app")


if __name__ == "__main__":
    main()
