"""One-command Colab runner for the UEBA + DLP pipeline.

Typical Colab usage after cloning the repo and uploading kaggle.json:

    !python /content/dlp-project/colab/run_full_pipeline.py --install-deps --download-data

For quick checks on existing artifacts:

    !python /content/dlp-project/colab/run_full_pipeline.py --smoke

Generated data is written to DLP_ROOT, which defaults to /content/dlp-data
on Colab and to the repository root elsewhere.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_COLAB_ROOT = Path("/content/dlp-data")
IS_COLAB = Path("/content").exists()
DATA_ROOT = Path(os.environ.get("DLP_ROOT", str(DEFAULT_COLAB_ROOT if IS_COLAB else REPO_DIR)))

REQUIRED_RAW_FILES = [
    "email.csv",
    "file.csv",
    "logon.csv",
    "device.csv",
    "psychometric.csv",
    # users.csv is not included in the CERT r4.2 Kaggle mirror; the cleaning
    # script skips it gracefully when absent, so it must not be required here.
]

OPTIONAL_RAW_FILES = [
    "users.csv",
    "ldap.csv",
    "decoy_file.csv",
]

KAGGLE_DATASET = "mrajaxnp/cert-insider-threat-detection-research"


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    """Run one pipeline command with repo/data env vars set."""
    merged_env = os.environ.copy()
    merged_env["DLP_ROOT"] = str(DATA_ROOT)
    merged_env["DLP_REPO"] = str(REPO_DIR)
    pythonpath = merged_env.get("PYTHONPATH", "")
    parts = [str(REPO_DIR)]
    if pythonpath:
        parts.append(pythonpath)
    merged_env["PYTHONPATH"] = os.pathsep.join(parts)
    if env:
        merged_env.update(env)

    print("\n" + "=" * 72)
    print("RUN:", " ".join(cmd))
    print("=" * 72)
    subprocess.run(cmd, cwd=REPO_DIR, env=merged_env, check=True)


def ensure_dirs() -> None:
    for name in ["archive", "cleaned", "models", "plots"]:
        (DATA_ROOT / name).mkdir(parents=True, exist_ok=True)


def copy_ground_truth() -> None:
    src = REPO_DIR / "archive" / "answers" / "answers" / "insiders.csv"
    dst = DATA_ROOT / "archive" / "answers" / "answers" / "insiders.csv"
    if not src.exists():
        print(f"[WARN] Ground-truth file not found in repo: {src}")
        return
    if src.resolve() == dst.resolve():
        print(f"Ground truth already in place -> {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"Copied ground truth -> {dst}")


def install_deps() -> None:
    run([sys.executable, "-m", "pip", "install", "-r", str(REPO_DIR / "requirements.txt"), "kaggle", "streamlit", "pyngrok", "-q"])


def download_data() -> None:
    archive_dir = DATA_ROOT / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    run(["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(archive_dir), "--unzip"])
    normalize_archive_layout()


def normalize_archive_layout() -> None:
    """Copy required raw CSVs into DATA_ROOT/archive if Kaggle nested them."""
    archive_dir = DATA_ROOT / "archive"
    missing_before = [name for name in REQUIRED_RAW_FILES if not (archive_dir / name).exists()]
    if not missing_before:
        return

    print("Normalizing Kaggle archive layout...")
    for name in missing_before:
        candidates = [p for p in archive_dir.rglob(name) if p.is_file()]
        if candidates:
            shutil.copy2(candidates[0], archive_dir / name)
            print(f"  copied {candidates[0].relative_to(archive_dir)} -> {name}")


def validate_raw_inputs() -> None:
    archive_dir = DATA_ROOT / "archive"
    normalize_archive_layout()
    missing = [name for name in REQUIRED_RAW_FILES if not (archive_dir / name).exists()]
    missing_optional = [name for name in OPTIONAL_RAW_FILES if not (archive_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing raw CERT files in "
            f"{archive_dir}: {', '.join(missing)}\n"
            "Run with --download-data after uploading kaggle.json, or place the CSVs manually."
        )
    if missing_optional:
        print(
            "[INFO] Optional raw files not found and will be skipped: "
            + ", ".join(missing_optional)
        )


def run_pipeline(args: argparse.Namespace) -> None:
    ensure_dirs()
    copy_ground_truth()

    if args.install_deps:
        install_deps()

    if args.download_data:
        download_data()

    if not args.skip_validate:
        validate_raw_inputs()

    sensitivity_limit = args.sensitivity_limit
    if args.smoke and sensitivity_limit is None:
        sensitivity_limit = 50_000

    if not args.skip_clean:
        run([sys.executable, str(REPO_DIR / "scripts" / "clean_cert_email_data.py")])

    if not args.skip_iforest:
        run([sys.executable, str(REPO_DIR / "colab" / "train_isolation_forest_cert.py")])
        run([sys.executable, str(REPO_DIR / "colab" / "visualize_isolation_forest_cert.py")])

    if not args.skip_lstm:
        run([sys.executable, str(REPO_DIR / "colab" / "train_lstm_autoencoder_cert.py")])
        run([sys.executable, str(REPO_DIR / "colab" / "visualize_lstm_autoencoder_cert.py")])

    if not args.skip_sensitivity:
        cmd = [sys.executable, str(REPO_DIR / "scripts" / "score_content_sensitivity.py")]
        if sensitivity_limit is not None:
            cmd += ["--limit", str(sensitivity_limit)]
        run(cmd)

    if not args.skip_ga:
        cmd = [sys.executable, str(REPO_DIR / "colab" / "ga_optimizer.py")]
        if args.smoke:
            cmd.append("--quick")
        run(cmd)

    if not args.skip_eval:
        run([sys.executable, str(REPO_DIR / "colab" / "evaluate_cert.py")])
        run([sys.executable, str(REPO_DIR / "colab" / "user_level_eval.py")])
        run([sys.executable, str(REPO_DIR / "colab" / "visualize_user_level.py")])

    print_outputs()

    if args.launch_dashboard:
        launch_dashboard()


def print_outputs() -> None:
    print("\n" + "=" * 72)
    print("PIPELINE COMPLETE")
    print("=" * 72)
    for label, path in [
        ("Cleaned data", DATA_ROOT / "cleaned"),
        ("Models", DATA_ROOT / "models"),
        ("Plots", DATA_ROOT / "plots"),
        ("Dashboard", REPO_DIR / "app" / "ueba_dashboard.py"),
    ]:
        print(f"{label:14}: {path}")
    print("\nLaunch later with:")
    print(f"  DLP_ROOT={DATA_ROOT} DLP_REPO={REPO_DIR} streamlit run {REPO_DIR / 'app' / 'ueba_dashboard.py'}")


def launch_dashboard() -> None:
    """Launch Streamlit and expose it with pyngrok in Colab."""
    try:
        from pyngrok import ngrok
    except ImportError as exc:
        raise RuntimeError("Install pyngrok first or rerun with --install-deps.") from exc

    env = os.environ.copy()
    env["DLP_ROOT"] = str(DATA_ROOT)
    env["DLP_REPO"] = str(REPO_DIR)
    env["PYTHONPATH"] = str(REPO_DIR)

    def run_streamlit() -> None:
        subprocess.run(
            [
                "streamlit",
                "run",
                str(REPO_DIR / "app" / "ueba_dashboard.py"),
                "--server.port",
                "8501",
                "--server.headless",
                "true",
            ],
            cwd=REPO_DIR,
            env=env,
            check=False,
        )

    ngrok.kill()
    threading.Thread(target=run_streamlit, daemon=True).start()
    time.sleep(5)
    public_url = ngrok.connect(8501)
    print("\nDashboard URL:")
    print(public_url)
    print("\nKeep this Colab cell running while you use the dashboard.")
    while True:
        time.sleep(60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the complete UEBA + DLP pipeline on Colab.")
    parser.add_argument("--install-deps", action="store_true", help="Install Python dependencies first.")
    parser.add_argument("--download-data", action="store_true", help="Download CERT data from Kaggle.")
    parser.add_argument("--launch-dashboard", action="store_true", help="Launch Streamlit through pyngrok after the pipeline.")
    parser.add_argument("--smoke", action="store_true", help="Use quick GA and 50k-row sensitivity scoring for faster validation.")
    parser.add_argument("--sensitivity-limit", type=int, default=None, help="Limit rows per source for sensitivity scoring.")
    parser.add_argument("--skip-validate", action="store_true", help="Skip raw input validation.")
    parser.add_argument("--skip-clean", action="store_true", help="Skip data cleaning.")
    parser.add_argument("--skip-iforest", action="store_true", help="Skip Isolation Forest training and plots.")
    parser.add_argument("--skip-lstm", action="store_true", help="Skip LSTM training and plots.")
    parser.add_argument("--skip-sensitivity", action="store_true", help="Skip DLP content sensitivity scoring.")
    parser.add_argument("--skip-ga", action="store_true", help="Skip GA optimisation.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation and user-level plots.")
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
