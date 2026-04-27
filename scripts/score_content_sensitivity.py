"""Content sensitivity scorer for email and file events.

Applies lightweight keyword/rule-based classification to
cleaned/email_cleaned.csv and cleaned/file_cleaned.csv, then
aggregates results to one row per user per day.

Sensitivity tiers
-----------------
  RESTRICTED (3) -- credentials, PII, financial records, executables to USB
  SENSITIVE  (2) -- confidential docs, financial data, strategic info to USB
  INTERNAL   (1) -- generic internal files / activity on removable media
  PUBLIC     (0) -- no indicators

Output
------
  cleaned/content_sensitivity_daily.csv
  Columns: user, email_day, sensitivity_score (mean), max_sensitivity_score,
           sensitive_event_count, restricted_event_count, top_sensitivity_label

Usage
-----
  python scripts/score_content_sensitivity.py            # full run
  python scripts/score_content_sensitivity.py --limit 50000  # smoke test
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
REPO_DIR = Path(os.environ.get("DLP_ROOT", str(Path(__file__).resolve().parent.parent)))
sys.path.insert(0, str(REPO_DIR))
from config import CLEANED_DIR  # noqa: E402

EMAIL_CSV  = CLEANED_DIR / "email_cleaned.csv"
FILE_CSV   = CLEANED_DIR / "file_cleaned.csv"
OUTPUT_CSV = CLEANED_DIR / "content_sensitivity_daily.csv"

CHUNK_SIZE = 200_000

# ---------------------------------------------------------------------------
# Sensitivity label lookup
# ---------------------------------------------------------------------------
SCORE_TO_LABEL = {0: "PUBLIC", 1: "INTERNAL", 2: "SENSITIVE", 3: "RESTRICTED"}

# ---------------------------------------------------------------------------
# Keyword rules — ordered RESTRICTED first so higher tier wins
# Applied to the lowercased 'content' field
# ---------------------------------------------------------------------------
# (tier, regex_pattern)
_KEYWORD_RULES: list[tuple[int, str]] = [
    # RESTRICTED — credentials & auth
    (3, r"\bpassword\b|\bpasswd\b|\bcredential\b|\bpassphrase\b|\bprivate.?key\b"),
    (3, r"\bapi.?key\b|\bsecret.?key\b|\bauth.?token\b|\baccess.?token\b"),
    # RESTRICTED — PII
    (3, r"\bssn\b|\bsocial.?security\b|\bpassport\b|\btax.?id\b"),
    (3, r"\brouting.?number\b|\baccount.?number\b|\bcredit.?card\b|\bcvv\b"),
    (3, r"\bmedical.?record\b|\bpatient.?data\b|\bdiagnosis\b|\bprescription\b"),
    # SENSITIVE — financial & strategic
    (2, r"\bsalary\b|\bpayroll\b|\bcompensation\b|\bbonus\b|\bstipend\b"),
    (2, r"\bbudget\b|\bforecast\b|\brevenue\b|\bearnings\b|\bquarterly.?result\b"),
    (2, r"\bmerger\b|\bacquisition\b|\btakeover\b|\bipo\b|\bstrateg\w+\b"),
    (2, r"\blayoff\b|\brestructur\w+\b|\bdownsiz\w+\b|\btermination\b"),
    # SENSITIVE — legal & IP
    (2, r"\bconfidential\b|\bproprietary\b|\btrade.?secret\b|\bembargo\b"),
    (2, r"\blitigation\b|\blawsuit\b|\bsettlement\b|\battorney.?client\b"),
    (2, r"\bcontract\b|\bnda\b|\bnon.?disclosure\b|\bintellectual.?property\b"),
    # INTERNAL — low-sensitivity structural terms
    (1, r"\binternal.?use\b|\bdraft\b|\bmeeting.?notes\b|\bagenda\b"),
    (1, r"\bproject.?plan\b|\baction.?item\b|\bconfidential\b"),
]

# File-extension tiers (lowercase)
_EXT_TIER: dict[str, int] = {
    # Executables / scripts → highest risk on removable media
    ".exe": 3, ".dll": 3, ".bat": 3, ".cmd": 3, ".ps1": 3,
    ".sh": 3, ".vbs": 3, ".js": 2, ".py": 2,
    # Archives (may contain any content)
    ".zip": 2, ".tar": 2, ".gz": 2, ".7z": 2, ".rar": 2, ".bz2": 2,
    # Office documents
    ".doc": 2, ".docx": 2, ".xls": 2, ".xlsx": 2,
    ".ppt": 2, ".pptx": 2, ".pdf": 2, ".csv": 2,
    # Text / data
    ".txt": 1, ".log": 1, ".xml": 1, ".json": 1, ".sql": 2,
    # Images — low sensitivity by default
    ".jpg": 0, ".jpeg": 0, ".png": 0, ".gif": 0, ".bmp": 0,
}

# Magic-byte prefixes in the content field → executable signals
# The content field starts with a hex dump (e.g. "4D-5A" = MZ header)
_MAGIC_RESTRICTED  = {"4D-5A"}        # PE executable (MZ)
_MAGIC_SENSITIVE   = {
    "D0-CF-11-E0",    # OLE2 (Office 97-2003: doc/xls/ppt)
    "25-50-44-46",    # %PDF-
    "50-4B-03-04",    # ZIP / Office XML (xlsx, docx…)
    "52-61-72-21",    # RAR
    "37-7A-BC-AF",    # 7-zip
}


# ===========================================================================
# Scoring helpers (fully vectorised on a DataFrame chunk)
# ===========================================================================

def _keyword_score(text_series: pd.Series) -> pd.Series:
    """Return highest tier triggered by any keyword rule. Vectorised."""
    lower = text_series.fillna("").str.lower()
    scores = pd.Series(0, index=text_series.index, dtype=np.int8)
    for tier, pattern in _KEYWORD_RULES:
        hit = lower.str.contains(pattern, regex=True, na=False)
        scores = scores.where(~hit | (scores >= tier), other=tier)
    return scores


def _magic_score(content_series: pd.Series) -> pd.Series:
    """Return tier from magic-byte hex prefix in content. Vectorised."""
    prefix8  = content_series.fillna("").str[:8]   # "D0-CF-11-E0"
    prefix5  = content_series.fillna("").str[:5]   # "4D-5A"
    scores   = pd.Series(0, index=content_series.index, dtype=np.int8)
    for magic in _MAGIC_RESTRICTED:
        scores = scores.where(~prefix5.str.startswith(magic), other=3)
    for magic in _MAGIC_SENSITIVE:
        hit = prefix8.str.startswith(magic) | prefix5.str.startswith(magic)
        scores = scores.where(~hit | (scores >= 2), other=2)
    return scores


def _ext_score(filename_series: pd.Series) -> pd.Series:
    """Return tier from file extension. Vectorised."""
    exts = (
        filename_series.fillna("")
        .str.lower()
        .str.extract(r"(\.[a-z0-9]{1,6})$", expand=False)
        .fillna("")
    )
    return exts.map(lambda e: _EXT_TIER.get(e, 1)).astype(np.int8)


# ===========================================================================
# Per-source chunk scorers
# ===========================================================================

def score_email_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Score one chunk of email_cleaned.csv.

    Returns DataFrame with columns: user, email_day, event_score.
    """
    kw_score = _keyword_score(chunk["content"])

    # Boost for large attachments (high content_length_words) — proxy for
    # embedded sensitive doc when keywords are absent
    has_attach = chunk.get("has_attachment", pd.Series(0, index=chunk.index)).fillna(0).astype(int)
    large_attach = (chunk.get("content_length_words", pd.Series(0, index=chunk.index)).fillna(0) > 500) & (has_attach == 1)
    kw_score = (kw_score + large_attach.astype(np.int8)).clip(upper=3)

    result = pd.DataFrame({
        "user":        chunk["user"],
        "email_day":   chunk["email_day"],
        "event_score": kw_score.astype(np.int8),
    })
    return result[result["event_score"] > 0]   # drop PUBLIC rows to save memory


def score_file_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Score one chunk of file_cleaned.csv.

    Returns DataFrame with columns: user, email_day, event_score.
    """
    ext_score    = _ext_score(chunk["filename"])
    magic_score  = _magic_score(chunk["content"])
    kw_score     = _keyword_score(chunk["content"])

    # Content score = max of magic + keyword signals
    content_score = np.maximum(magic_score, kw_score)

    # Final event score: start with content, then apply activity adjustments
    base_score = np.maximum(ext_score, content_score).astype(np.int8)

    removable = chunk.get("to_removable_media", pd.Series(0, index=chunk.index)).fillna(0).astype(int)
    is_write  = chunk.get("is_write", pd.Series(0, index=chunk.index)).fillna(0).astype(int)

    # Writing an Office/archive doc to removable media → bump to SENSITIVE minimum
    written_to_usb = (removable == 1) & (is_write == 1)
    base_score = base_score.where(~written_to_usb | (base_score >= 2), other=2)

    # Writing an executable to removable media → RESTRICTED
    exe_ext = ext_score >= 3
    base_score = base_score.where(~(written_to_usb & exe_ext), other=3)

    # Even a read from removable media earns INTERNAL if score is still 0
    from_usb = chunk.get("from_removable_media", pd.Series(0, index=chunk.index)).fillna(0).astype(int)
    usb_any = (removable == 1) | (from_usb == 1)
    base_score = base_score.where(~usb_any | (base_score >= 1), other=1)

    base_score = base_score.clip(0, 3)

    result = pd.DataFrame({
        "user":        chunk["user"],
        "email_day":   chunk["day"],   # file_cleaned uses 'day', standardise here
        "event_score": base_score.astype(np.int8),
    })
    return result[result["event_score"] > 0]


# ===========================================================================
# Chunk-level aggregation
# ===========================================================================

def _agg_chunk(scored: pd.DataFrame) -> pd.DataFrame:
    """Aggregate scored events to (user, email_day) granularity."""
    if scored.empty:
        return pd.DataFrame(columns=["user", "email_day",
                                     "sum_score", "max_score",
                                     "sensitive_count", "restricted_count",
                                     "total_events"])
    g = scored.groupby(["user", "email_day"], sort=False)
    return g["event_score"].agg(
        sum_score="sum",
        max_score="max",
        sensitive_count=lambda x: (x >= 2).sum(),
        restricted_count=lambda x: (x == 3).sum(),
        total_events="count",
    ).reset_index()


# ===========================================================================
# Final aggregation across all accumulated chunk-aggs
# ===========================================================================

def _merge_aggs(agg_list: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine chunk-level aggregates into final per-user-day totals."""
    if not agg_list:
        return pd.DataFrame(columns=["user", "email_day",
                                     "sensitivity_score", "max_sensitivity_score",
                                     "sensitive_event_count", "restricted_event_count",
                                     "top_sensitivity_label"])
    combined = pd.concat(agg_list, ignore_index=True)
    final = combined.groupby(["user", "email_day"], sort=False).agg(
        total_sum_score=("sum_score", "sum"),
        max_sensitivity_score=("max_score", "max"),
        sensitive_event_count=("sensitive_count", "sum"),
        restricted_event_count=("restricted_count", "sum"),
        total_events=("total_events", "sum"),
    ).reset_index()
    final["sensitivity_score"] = (
        final["total_sum_score"] / final["total_events"]
    ).clip(0, 3).round(4)
    final["top_sensitivity_label"] = (
        final["max_sensitivity_score"]
        .clip(0, 3)
        .astype(int)
        .map(SCORE_TO_LABEL)
    )
    return final[[
        "user", "email_day",
        "sensitivity_score", "max_sensitivity_score",
        "sensitive_event_count", "restricted_event_count",
        "top_sensitivity_label",
    ]]


# ===========================================================================
# Main driver
# ===========================================================================

def _process_source(
    path: Path,
    score_fn,
    label: str,
    limit: int | None,
    chunk_size: int,
) -> list[pd.DataFrame]:
    """Process one source CSV in chunks; return list of chunk-level aggs."""
    if not path.exists():
        print(f"  [SKIP] {path.name} not found.")
        return []

    agg_list: list[pd.DataFrame] = []
    rows_read = 0
    chunks_done = 0
    t0 = time.time()

    reader = pd.read_csv(path, chunksize=chunk_size, low_memory=False)
    for chunk in reader:
        if limit and rows_read >= limit:
            break
        if limit:
            chunk = chunk.iloc[: max(0, limit - rows_read)]

        scored = score_fn(chunk)
        agg = _agg_chunk(scored)
        if not agg.empty:
            agg_list.append(agg)

        rows_read  += len(chunk)
        chunks_done += 1
        elapsed = time.time() - t0
        rate = rows_read / elapsed if elapsed > 0 else 0
        print(f"  {label} | chunk {chunks_done:>4}  "
              f"rows {rows_read:>9,}  "
              f"scored {scored['event_score'].sum() if not scored.empty else 0:>8,}  "
              f"{rate:>8,.0f} rows/s",
              end="\r", flush=True)

    print()  # newline after \r progress
    return agg_list


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Content sensitivity scorer for DLP pipeline"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only this many rows per source (smoke test)",
    )
    parser.add_argument(
        "--chunk", type=int, default=CHUNK_SIZE,
        help=f"Chunk size (default {CHUNK_SIZE:,})",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Content Sensitivity Scorer")
    print("=" * 60)
    if args.limit:
        print(f"  [smoke test] --limit {args.limit:,} rows per source\n")

    t_start = time.time()

    # --- Email ---
    print(f"\n[1/2] Scoring email events ({EMAIL_CSV.name})...")
    email_aggs = _process_source(
        EMAIL_CSV, score_email_chunk, "email",
        args.limit, args.chunk,
    )

    # --- File ---
    print(f"\n[2/2] Scoring file events ({FILE_CSV.name})...")
    file_aggs = _process_source(
        FILE_CSV, score_file_chunk, "file",
        args.limit, args.chunk,
    )

    # --- Combine ---
    print("\n  Combining sources and computing daily aggregates...")
    all_aggs = email_aggs + file_aggs
    if not all_aggs:
        print("  No data produced — check that source CSVs exist.")
        return

    daily = _merge_aggs(all_aggs)

    # --- Write ---
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUTPUT_CSV, index=False)

    elapsed = time.time() - t_start
    n_users   = daily["user"].nunique()
    n_days    = daily["email_day"].nunique()
    n_rows    = len(daily)
    n_sens    = int(daily["sensitive_event_count"].sum())
    n_restr   = int(daily["restricted_event_count"].sum())
    label_counts = daily["top_sensitivity_label"].value_counts().to_dict()

    print(f"\n  Output : {OUTPUT_CSV}")
    print(f"  Rows   : {n_rows:,}  ({n_users:,} users × {n_days:,} days)")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"\n  Sensitivity label distribution:")
    for label in ["RESTRICTED", "SENSITIVE", "INTERNAL", "PUBLIC"]:
        count = label_counts.get(label, 0)
        pct   = 100 * count / n_rows if n_rows else 0
        print(f"    {label:<12} {count:>8,}  ({pct:5.1f}%)")
    print(f"\n  Sensitive events total  : {n_sens:,}")
    print(f"  Restricted events total : {n_restr:,}")
    print()


if __name__ == "__main__":
    main()
