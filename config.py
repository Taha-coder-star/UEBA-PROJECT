"""Central path configuration — works locally and on Google Colab.

Local  : leave DLP_ROOT unset; all paths resolve inside the project folder.
Colab  : before running any script, set the env var to your Drive folder:
             import os
             os.environ["DLP_ROOT"] = "/content/drive/MyDrive/dlp-project"
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("DLP_ROOT", str(Path(__file__).resolve().parent)))

ARCHIVE_DIR = ROOT / "archive"
CLEANED_DIR = ROOT / "cleaned"
MODELS_DIR  = ROOT / "models"
PLOTS_DIR   = ROOT / "plots"
