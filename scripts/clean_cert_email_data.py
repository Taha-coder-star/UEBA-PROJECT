"""Chunked cleaning for CERT email + psychometric + logon + device + file data."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ARCHIVE_DIR, CLEANED_DIR  # noqa: E402

EMAIL_INPUT = ARCHIVE_DIR / "email.csv"
PSY_INPUT = ARCHIVE_DIR / "psychometric.csv"
LOGON_INPUT = ARCHIVE_DIR / "logon.csv"
DEVICE_INPUT = ARCHIVE_DIR / "device.csv"
FILE_INPUT = ARCHIVE_DIR / "file.csv"
DECOY_FILE_INPUT = ARCHIVE_DIR / "decoy_file.csv"
USERS_INPUT = ARCHIVE_DIR / "users.csv"

EMAIL_OUTPUT = CLEANED_DIR / "email_cleaned.csv"
USER_DAILY_OUTPUT = CLEANED_DIR / "email_user_daily_features.csv"
PSY_OUTPUT = CLEANED_DIR / "psychometric_cleaned.csv"
MERGED_OUTPUT = CLEANED_DIR / "email_user_daily_with_psychometric.csv"
LOGON_OUTPUT = CLEANED_DIR / "logon_cleaned.csv"
DEVICE_OUTPUT = CLEANED_DIR / "device_cleaned.csv"
FILE_OUTPUT = CLEANED_DIR / "file_cleaned.csv"
DECOY_FILE_OUTPUT = CLEANED_DIR / "decoy_file_cleaned.csv"
USERS_OUTPUT = CLEANED_DIR / "users_cleaned.csv"
SUMMARY_OUTPUT = CLEANED_DIR / "cleaning_summary.txt"

CHUNK_SIZE = 200_000


def normalize_file_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize file.csv variants across CERT releases.

    Newer mirrors include explicit activity/removable-media columns.  r4.2
    commonly stores file events without those fields; in that release the file
    log is already the removable-media/file-copy signal, so we use conservative
    defaults that keep the event useful for UEBA/DLP features.
    """
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    if "filename" not in df.columns:
        for candidate in ("file", "filepath", "path"):
            if candidate in df.columns:
                df = df.rename(columns={candidate: "filename"})
                break

    if "activity" not in df.columns:
        df["activity"] = "File Write"
    if "filename" not in df.columns:
        df["filename"] = ""
    if "content" not in df.columns:
        df["content"] = ""
    if "to_removable_media" not in df.columns:
        df["to_removable_media"] = True
    if "from_removable_media" not in df.columns:
        df["from_removable_media"] = False

    return df


def boolish_to_int(series: pd.Series) -> pd.Series:
    return (
        series.map({
            True: 1, False: 0,
            "True": 1, "False": 0,
            "true": 1, "false": 0,
            "1": 1, "0": 0,
            1: 1, 0: 0,
        })
        .fillna(0)
        .astype(int)
    )


def iter_csv_chunks(path: Path, *, tolerate_bad_lines: bool = False):
    """Yield CSV chunks with an optional tolerant parser for malformed rows."""
    kwargs = {"chunksize": CHUNK_SIZE}
    if tolerate_bad_lines:
        print(
            f"Reading {path.name} with the tolerant parser; malformed CSV rows "
            "will be skipped with a warning."
        )
        kwargs.update({"engine": "python", "on_bad_lines": "warn"})
    return pd.read_csv(path, **kwargs)


def count_recipients(value: object) -> int:
    if pd.isna(value):
        return 0
    text = str(value).strip()
    if not text:
        return 0
    return len([item for item in text.split(';') if item.strip()])


def clean_email_chunk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [column.strip().lower() for column in df.columns]
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["date", "id", "user", "pc", "to", "from", "size", "attachments", "content"])
    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)

    df["user"] = df["user"].astype(str).str.strip().str.upper()
    df["pc"] = df["pc"].astype(str).str.strip().str.upper()
    df["sender"] = df["from"].astype(str).str.strip().str.lower()
    df["to"] = df["to"].astype(str).str.strip().str.lower()
    df["cc"] = df["cc"].fillna("").astype(str).str.strip().str.lower()
    df["bcc"] = df["bcc"].fillna("").astype(str).str.strip().str.lower()
    df["content"] = df["content"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip().str.lower()

    df["size"] = pd.to_numeric(df["size"], errors="coerce")
    df = df.dropna(subset=["size"]).reset_index(drop=True)
    # attachments column contains file paths (semicolon-separated), not a count
    df["attachments"] = df["attachments"].apply(
        lambda v: len([x for x in str(v).split(";") if x.strip()]) if pd.notna(v) and str(v).strip() else 0
    )

    df["email_day"] = df["date"].dt.date.astype(str)
    df["email_hour"] = df["date"].dt.hour
    df["email_month"] = df["date"].dt.month
    df["email_weekday"] = df["date"].dt.dayofweek
    df["is_after_hours"] = ((df["email_hour"] < 7) | (df["email_hour"] > 20)).astype(int)
    df["content_length_chars"] = df["content"].str.len()
    df["content_length_words"] = df["content"].str.split().str.len()
    df["to_recipient_count"] = df["to"].apply(count_recipients)
    df["cc_recipient_count"] = df["cc"].apply(count_recipients)
    df["bcc_recipient_count"] = df["bcc"].apply(count_recipients)
    df["total_recipient_count"] = df["to_recipient_count"] + df["cc_recipient_count"] + df["bcc_recipient_count"]
    df["has_cc"] = (df["cc_recipient_count"] > 0).astype(int)
    df["has_bcc"] = (df["bcc_recipient_count"] > 0).astype(int)
    df["has_attachment"] = (df["attachments"] > 0).astype(int)

    return df[[
        "id", "date", "email_day", "email_hour", "email_month", "email_weekday", "is_after_hours",
        "user", "pc", "sender", "to", "cc", "bcc", "size", "attachments", "has_attachment",
        "to_recipient_count", "cc_recipient_count", "bcc_recipient_count", "total_recipient_count",
        "has_cc", "has_bcc", "content_length_chars", "content_length_words", "content"
    ]]


def aggregate_logon_daily(logon_csv: Path) -> pd.DataFrame:
    """Aggregate logon/logoff events to per-user-day features (chunked)."""
    chunk_results = []
    for chunk in pd.read_csv(logon_csv, chunksize=CHUNK_SIZE):
        chunk.columns = [c.strip().lower() for c in chunk.columns]
        chunk["date"] = pd.to_datetime(chunk["date"], format="%m/%d/%Y %H:%M:%S", errors="coerce")
        chunk = chunk.dropna(subset=["date", "user", "activity"]).copy()
        chunk["user"] = chunk["user"].astype(str).str.strip().str.upper()
        chunk["pc"] = chunk["pc"].astype(str).str.strip().str.upper()
        chunk["day"] = chunk["date"].dt.date.astype(str)
        hour = chunk["date"].dt.hour
        is_after_hours = ((hour < 7) | (hour > 20)).astype(int)
        chunk["is_logon"] = (chunk["activity"] == "Logon").astype(int)
        chunk["is_logoff"] = (chunk["activity"] == "Logoff").astype(int)
        chunk["after_hours_logon"] = (chunk["is_logon"] & is_after_hours).astype(int)
        grouped = chunk.groupby(["user", "day"]).agg(
            logon_count=("is_logon", "sum"),
            logoff_count=("is_logoff", "sum"),
            after_hours_logons=("after_hours_logon", "sum"),
            unique_logon_pcs=("pc", "nunique"),
        ).reset_index()
        chunk_results.append(grouped)

    if not chunk_results:
        return pd.DataFrame(columns=["user", "day", "logon_count", "logoff_count", "after_hours_logons", "unique_logon_pcs"])

    combined = pd.concat(chunk_results, ignore_index=True)
    return combined.groupby(["user", "day"]).agg(
        logon_count=("logon_count", "sum"),
        logoff_count=("logoff_count", "sum"),
        after_hours_logons=("after_hours_logons", "sum"),
        unique_logon_pcs=("unique_logon_pcs", "max"),
    ).reset_index()


def aggregate_device_daily(device_csv: Path) -> pd.DataFrame:
    """Aggregate USB connect/disconnect events to per-user-day features (chunked)."""
    chunk_results = []
    for chunk in pd.read_csv(device_csv, chunksize=CHUNK_SIZE):
        chunk.columns = [c.strip().lower() for c in chunk.columns]
        chunk["date"] = pd.to_datetime(chunk["date"], format="%m/%d/%Y %H:%M:%S", errors="coerce")
        chunk = chunk.dropna(subset=["date", "user", "activity"]).copy()
        chunk["user"] = chunk["user"].astype(str).str.strip().str.upper()
        chunk["day"] = chunk["date"].dt.date.astype(str)
        chunk["is_connect"] = (chunk["activity"] == "Connect").astype(int)
        chunk["is_disconnect"] = (chunk["activity"] == "Disconnect").astype(int)
        grouped = chunk.groupby(["user", "day"]).agg(
            usb_connect_count=("is_connect", "sum"),
            usb_disconnect_count=("is_disconnect", "sum"),
        ).reset_index()
        chunk_results.append(grouped)

    if not chunk_results:
        return pd.DataFrame(columns=["user", "day", "usb_connect_count", "usb_disconnect_count"])

    combined = pd.concat(chunk_results, ignore_index=True)
    return combined.groupby(["user", "day"]).agg(
        usb_connect_count=("usb_connect_count", "sum"),
        usb_disconnect_count=("usb_disconnect_count", "sum"),
    ).reset_index()


def aggregate_file_daily(file_csv: Path) -> pd.DataFrame:
    """Aggregate file access events to per-user-day features (chunked)."""
    chunk_results = []
    for chunk in pd.read_csv(file_csv, chunksize=CHUNK_SIZE):
        chunk = normalize_file_schema(chunk)
        chunk["date"] = pd.to_datetime(chunk["date"], format="%m/%d/%Y %H:%M:%S", errors="coerce")
        chunk = chunk.dropna(subset=["date", "user", "activity"]).copy()
        chunk["user"] = chunk["user"].astype(str).str.strip().str.upper()
        chunk["day"] = chunk["date"].dt.date.astype(str)
        hour = chunk["date"].dt.hour
        chunk["is_after_hours"] = ((hour < 7) | (hour > 20)).astype(int)
        chunk["to_removable_media"] = boolish_to_int(chunk["to_removable_media"])
        chunk["from_removable_media"] = boolish_to_int(chunk["from_removable_media"])
        chunk["is_write"] = (chunk["activity"] == "File Write").astype(int)
        grouped = chunk.groupby(["user", "day"]).agg(
            file_total=("id", "count"),
            file_to_removable=("to_removable_media", "sum"),
            file_from_removable=("from_removable_media", "sum"),
            file_write_count=("is_write", "sum"),
            file_after_hours=("is_after_hours", "sum"),
        ).reset_index()
        chunk_results.append(grouped)

    if not chunk_results:
        return pd.DataFrame(columns=["user", "day", "file_total", "file_to_removable", "file_from_removable", "file_write_count", "file_after_hours"])

    combined = pd.concat(chunk_results, ignore_index=True)
    return combined.groupby(["user", "day"]).agg(
        file_total=("file_total", "sum"),
        file_to_removable=("file_to_removable", "sum"),
        file_from_removable=("file_from_removable", "sum"),
        file_write_count=("file_write_count", "sum"),
        file_after_hours=("file_after_hours", "sum"),
    ).reset_index()


def clean_psychometric_data(psychometric_df: pd.DataFrame) -> pd.DataFrame:
    df = psychometric_df.copy()
    df.columns = [column.strip().lower() for column in df.columns]
    df = df.drop_duplicates(subset=["user_id"]).reset_index(drop=True)
    df["user_id"] = df["user_id"].astype(str).str.strip().str.upper()
    df["employee_name"] = df["employee_name"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    for column in ["o", "c", "e", "a", "n"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["user_id", "o", "c", "e", "a", "n"]).reset_index(drop=True)
    return df[["employee_name", "user_id", "o", "c", "e", "a", "n"]]


def clean_logon_chunk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["date", "id", "user", "pc", "activity"]).copy()
    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)
    df["user"] = df["user"].astype(str).str.strip().str.upper()
    df["pc"] = df["pc"].astype(str).str.strip().str.upper()
    df["activity"] = df["activity"].astype(str).str.strip()
    df["day"] = df["date"].dt.date.astype(str)
    df["hour"] = df["date"].dt.hour
    df["is_after_hours"] = ((df["hour"] < 7) | (df["hour"] > 20)).astype(int)
    df["is_logon"] = (df["activity"] == "Logon").astype(int)
    return df[["id", "date", "day", "hour", "is_after_hours", "user", "pc", "activity", "is_logon"]]


def clean_device_chunk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["date", "id", "user", "pc", "activity"]).copy()
    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)
    df["user"] = df["user"].astype(str).str.strip().str.upper()
    df["pc"] = df["pc"].astype(str).str.strip().str.upper()
    df["activity"] = df["activity"].astype(str).str.strip()
    if "file_tree" in df.columns:
        df["file_tree"] = df["file_tree"].fillna("").astype(str).str.strip()
    else:
        df["file_tree"] = ""
    df["day"] = df["date"].dt.date.astype(str)
    df["hour"] = df["date"].dt.hour
    df["is_after_hours"] = ((df["hour"] < 7) | (df["hour"] > 20)).astype(int)
    df["is_connect"] = (df["activity"] == "Connect").astype(int)
    return df[["id", "date", "day", "hour", "is_after_hours", "user", "pc", "activity", "is_connect", "file_tree"]]


def clean_file_chunk(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_file_schema(df)
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["date", "id", "user", "pc", "activity", "filename"]).copy()
    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)
    df["user"] = df["user"].astype(str).str.strip().str.upper()
    df["pc"] = df["pc"].astype(str).str.strip().str.upper()
    df["activity"] = df["activity"].astype(str).str.strip()
    df["filename"] = df["filename"].astype(str).str.strip()
    df["to_removable_media"] = boolish_to_int(df["to_removable_media"])
    df["from_removable_media"] = boolish_to_int(df["from_removable_media"])
    df["day"] = df["date"].dt.date.astype(str)
    df["hour"] = df["date"].dt.hour
    df["is_after_hours"] = ((df["hour"] < 7) | (df["hour"] > 20)).astype(int)
    df["is_write"] = (df["activity"] == "File Write").astype(int)
    df["content"] = df["content"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    return df[[
        "id", "date", "day", "hour", "is_after_hours", "user", "pc",
        "filename", "activity", "is_write", "to_removable_media", "from_removable_media", "content"
    ]]


def clean_decoy_file_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"decoy_filename": "decoy_filename"})
    df["decoy_filename"] = df["decoy_filename"].astype(str).str.strip()
    df["pc"] = df["pc"].astype(str).str.strip().str.upper()
    df = df.dropna(subset=["decoy_filename", "pc"]).reset_index(drop=True)
    df = df.drop_duplicates().reset_index(drop=True)
    return df[["decoy_filename", "pc"]]


def clean_users_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    df["user_id"] = df["user_id"].astype(str).str.strip().str.upper()
    df["employee_name"] = df["employee_name"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df["email"] = df["email"].astype(str).str.strip().str.lower()
    df["role"] = df["role"].astype(str).str.strip()
    df["business_unit"] = pd.to_numeric(df["business_unit"], errors="coerce")
    for col in ["functional_unit", "department", "team", "supervisor"]:
        df[col] = df[col].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    df["projects"] = df["projects"].fillna("").astype(str).str.strip()
    df = df.dropna(subset=["user_id", "employee_name", "email"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["user_id"]).reset_index(drop=True)
    return df[[
        "user_id", "employee_name", "email", "role", "projects",
        "business_unit", "functional_unit", "department", "team",
        "supervisor", "start_date", "end_date"
    ]]


def assign_split(df: pd.DataFrame, date_col: str = "email_day",
                 train_frac: float = 0.70, val_frac: float = 0.15) -> pd.DataFrame:
    """Add dataset_split column using per-user chronological 70/15/15 split.

    Each user's own timeline is split independently so every user appears in
    train, val, and test — preventing class imbalance across splits.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    splits = []
    for user, grp in df.groupby("user", sort=False):
        grp = grp.sort_values(date_col).reset_index(drop=True)
        n = len(grp)
        train_end = int(n * train_frac)
        val_end   = int(n * (train_frac + val_frac))
        grp["dataset_split"] = "test"
        grp.loc[:train_end - 1, "dataset_split"] = "train"
        grp.loc[train_end:val_end - 1, "dataset_split"] = "val"
        splits.append(grp)

    return pd.concat(splits, ignore_index=True)


def assign_split_by_date(df: pd.DataFrame, date_col: str,
                         train_cutoff: str, val_cutoff: str) -> pd.DataFrame:
    """Add dataset_split to a row-level CSV using global date cutoffs."""
    df = df.copy()
    dates = pd.to_datetime(df[date_col], errors="coerce")
    df["dataset_split"] = "test"
    df.loc[dates < train_cutoff, "dataset_split"] = "train"
    df.loc[(dates >= train_cutoff) & (dates < val_cutoff), "dataset_split"] = "val"
    return df


def main() -> None:
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    if EMAIL_OUTPUT.exists():
        EMAIL_OUTPUT.unlink()

    summary_rows = []
    chunk_summaries = []
    header_written = False
    min_date = None
    max_date = None
    total_clean_rows = 0
    seen_ids = set()
    total_logon_rows = 0
    total_device_rows = 0
    total_file_rows = 0
    decoy_clean = pd.DataFrame()
    users_clean = pd.DataFrame()

    for chunk_index, chunk in enumerate(
        iter_csv_chunks(EMAIL_INPUT, tolerate_bad_lines=True),
        start=1,
    ):
        cleaned = clean_email_chunk(chunk)
        cleaned = cleaned[~cleaned["id"].isin(seen_ids)].copy()
        seen_ids.update(cleaned["id"].tolist())

        if not cleaned.empty:
            cleaned.to_csv(EMAIL_OUTPUT, mode="a", header=not header_written, index=False)
            header_written = True
            total_clean_rows += len(cleaned)
            chunk_min = cleaned["date"].min()
            chunk_max = cleaned["date"].max()
            min_date = chunk_min if min_date is None else min(min_date, chunk_min)
            max_date = chunk_max if max_date is None else max(max_date, chunk_max)

            grouped = cleaned.groupby(["user", "email_day"]).agg(
                email_count=("id", "count"),
                unique_pcs=("pc", "nunique"),
                total_size=("size", "sum"),
                avg_size=("size", "mean"),
                total_attachments=("attachments", "sum"),
                emails_with_attachments=("has_attachment", "sum"),
                after_hours_emails=("is_after_hours", "sum"),
                avg_recipients=("total_recipient_count", "mean"),
                max_recipients=("total_recipient_count", "max"),
                avg_content_words=("content_length_words", "mean"),
                max_content_words=("content_length_words", "max"),
                bcc_email_count=("has_bcc", "sum"),
                cc_email_count=("has_cc", "sum"),
            ).reset_index()
            summary_rows.append(grouped)
            chunk_summaries.append(f"chunk_{chunk_index}_rows={len(cleaned)}")
            print(f"Processed chunk {chunk_index}: {len(cleaned)} cleaned rows")

    if not summary_rows:
        raise RuntimeError(
            "No cleaned email rows were produced from archive/email.csv. "
            "The source file may be missing expected columns or contain too "
            "many malformed rows to parse safely."
        )

    psychometric_df = pd.read_csv(PSY_INPUT)
    psychometric_clean = clean_psychometric_data(psychometric_df)
    psychometric_clean.to_csv(PSY_OUTPUT, index=False)

    # --- Clean logon (row-level) ---
    if not LOGON_INPUT.exists():
        print("Skipping logon data (logon.csv not found in archive).")
    else:
        print("Cleaning logon data...")
        if LOGON_OUTPUT.exists():
            LOGON_OUTPUT.unlink()
        logon_header_written = False
        logon_seen_ids: set = set()
        for chunk in iter_csv_chunks(LOGON_INPUT):
            cleaned_logon = clean_logon_chunk(chunk)
            cleaned_logon = cleaned_logon[~cleaned_logon["id"].isin(logon_seen_ids)].copy()
            logon_seen_ids.update(cleaned_logon["id"].tolist())
            if not cleaned_logon.empty:
                cleaned_logon.to_csv(LOGON_OUTPUT, mode="a", header=not logon_header_written, index=False)
                logon_header_written = True
                total_logon_rows += len(cleaned_logon)
        print(f"  Saved {total_logon_rows} cleaned logon rows ->{LOGON_OUTPUT}")

    # --- Clean device (row-level) ---
    if not DEVICE_INPUT.exists():
        print("Skipping device data (device.csv not found in archive).")
    else:
        print("Cleaning device data...")
        if DEVICE_OUTPUT.exists():
            DEVICE_OUTPUT.unlink()
        device_header_written = False
        device_seen_ids: set = set()
        for chunk in iter_csv_chunks(DEVICE_INPUT):
            cleaned_device = clean_device_chunk(chunk)
            cleaned_device = cleaned_device[~cleaned_device["id"].isin(device_seen_ids)].copy()
            device_seen_ids.update(cleaned_device["id"].tolist())
            if not cleaned_device.empty:
                cleaned_device.to_csv(DEVICE_OUTPUT, mode="a", header=not device_header_written, index=False)
                device_header_written = True
                total_device_rows += len(cleaned_device)
        print(f"  Saved {total_device_rows} cleaned device rows ->{DEVICE_OUTPUT}")

    # --- Clean file (row-level) ---
    if not FILE_INPUT.exists():
        print("Skipping file access data (file.csv not found in archive).")
    else:
        print("Cleaning file access data...")
        if FILE_OUTPUT.exists():
            FILE_OUTPUT.unlink()
        file_header_written = False
        file_seen_ids: set = set()
        for chunk in iter_csv_chunks(FILE_INPUT):
            cleaned_file = clean_file_chunk(chunk)
            cleaned_file = cleaned_file[~cleaned_file["id"].isin(file_seen_ids)].copy()
            file_seen_ids.update(cleaned_file["id"].tolist())
            if not cleaned_file.empty:
                cleaned_file.to_csv(FILE_OUTPUT, mode="a", header=not file_header_written, index=False)
                file_header_written = True
                total_file_rows += len(cleaned_file)
        print(f"  Saved {total_file_rows} cleaned file rows ->{FILE_OUTPUT}")

    # --- Clean decoy_file ---
    if not DECOY_FILE_INPUT.exists():
        print("Skipping decoy file data (decoy_file.csv not found in archive).")
    else:
        print("Cleaning decoy file data...")
        decoy_df = pd.read_csv(DECOY_FILE_INPUT)
        decoy_clean = clean_decoy_file_data(decoy_df)
        decoy_clean.to_csv(DECOY_FILE_OUTPUT, index=False)
        print(f"  Saved {len(decoy_clean)} cleaned decoy file rows ->{DECOY_FILE_OUTPUT}")

    # --- Clean users ---
    if not USERS_INPUT.exists():
        print("Skipping users data (users.csv not found in archive).")
    else:
        print("Cleaning users data...")
        users_df = pd.read_csv(USERS_INPUT)
        users_clean = clean_users_data(users_df)
        users_clean.to_csv(USERS_OUTPUT, index=False)
        print(f"  Saved {len(users_clean)} cleaned user rows ->{USERS_OUTPUT}")

    daily_features = pd.concat(summary_rows, ignore_index=True)
    daily_features = daily_features.groupby(["user", "email_day"]).agg(
        email_count=("email_count", "sum"),
        unique_pcs=("unique_pcs", "max"),
        total_size=("total_size", "sum"),
        avg_size=("avg_size", "mean"),
        total_attachments=("total_attachments", "sum"),
        emails_with_attachments=("emails_with_attachments", "sum"),
        after_hours_emails=("after_hours_emails", "sum"),
        avg_recipients=("avg_recipients", "mean"),
        max_recipients=("max_recipients", "max"),
        avg_content_words=("avg_content_words", "mean"),
        max_content_words=("max_content_words", "max"),
        bcc_email_count=("bcc_email_count", "sum"),
        cc_email_count=("cc_email_count", "sum"),
    ).reset_index()
    daily_features["attachment_email_ratio"] = daily_features["emails_with_attachments"] / daily_features["email_count"]
    daily_features["after_hours_ratio"] = daily_features["after_hours_emails"] / daily_features["email_count"]
    daily_features["bcc_ratio"] = daily_features["bcc_email_count"] / daily_features["email_count"]

    # --- Merge logon features ---
    if LOGON_INPUT.exists():
        print("Aggregating logon events...")
        logon_daily = aggregate_logon_daily(LOGON_INPUT)
        logon_daily = logon_daily.rename(columns={"day": "email_day"})
        daily_features = daily_features.merge(logon_daily, on=["user", "email_day"], how="left")
        for col in ["logon_count", "logoff_count", "after_hours_logons", "unique_logon_pcs"]:
            daily_features[col] = daily_features[col].fillna(0).astype(int)
        print(f"  Logon user-days: {len(logon_daily)}")
    else:
        logon_daily = pd.DataFrame(
            columns=["user", "email_day", "logon_count", "logoff_count", "after_hours_logons", "unique_logon_pcs"]
        )
        for col in ["logon_count", "logoff_count", "after_hours_logons", "unique_logon_pcs"]:
            daily_features[col] = 0

    # --- Merge device features ---
    if DEVICE_INPUT.exists():
        print("Aggregating device events...")
        device_daily = aggregate_device_daily(DEVICE_INPUT)
        device_daily = device_daily.rename(columns={"day": "email_day"})
        daily_features = daily_features.merge(device_daily, on=["user", "email_day"], how="left")
        for col in ["usb_connect_count", "usb_disconnect_count"]:
            daily_features[col] = daily_features[col].fillna(0).astype(int)
        print(f"  Device user-days: {len(device_daily)}")
    else:
        device_daily = pd.DataFrame(columns=["user", "email_day", "usb_connect_count", "usb_disconnect_count"])
        for col in ["usb_connect_count", "usb_disconnect_count"]:
            daily_features[col] = 0

    # --- Merge file features ---
    if FILE_INPUT.exists():
        print("Aggregating file access events...")
        file_daily = aggregate_file_daily(FILE_INPUT)
        file_daily = file_daily.rename(columns={"day": "email_day"})
        daily_features = daily_features.merge(file_daily, on=["user", "email_day"], how="left")
        for col in ["file_total", "file_to_removable", "file_from_removable", "file_write_count", "file_after_hours"]:
            daily_features[col] = daily_features[col].fillna(0).astype(int)
        print(f"  File user-days: {len(file_daily)}")
    else:
        file_daily = pd.DataFrame(
            columns=["user", "email_day", "file_total", "file_to_removable", "file_from_removable", "file_write_count", "file_after_hours"]
        )
        for col in ["file_total", "file_to_removable", "file_from_removable", "file_write_count", "file_after_hours"]:
            daily_features[col] = 0

    # --- Per-user chronological 70/15/15 split ---
    daily_features = assign_split(daily_features, date_col="email_day")
    daily_features.to_csv(USER_DAILY_OUTPUT, index=False)

    merged = daily_features.merge(psychometric_clean, left_on="user", right_on="user_id", how="left")
    merged.to_csv(MERGED_OUTPUT, index=False)

    # Compute global date cutoffs from the email data for row-level CSV splits
    all_dates = pd.to_datetime(daily_features["email_day"], errors="coerce").dropna().sort_values()
    n_days = len(all_dates.unique())
    train_cutoff = str(sorted(all_dates.unique())[int(n_days * 0.70)])
    val_cutoff   = str(sorted(all_dates.unique())[int(n_days * 0.85)])
    print(f"Split cutoffs -> train:<{train_cutoff}  val:<{val_cutoff}  test:rest")

    # Apply date-based split to row-level cleaned CSVs
    for path, date_col in [
        (LOGON_OUTPUT,  "date"),
        (DEVICE_OUTPUT, "date"),
        (FILE_OUTPUT,   "date"),
    ]:
        if path.exists():
            tmp = pd.read_csv(path)
            tmp = assign_split_by_date(tmp, date_col, train_cutoff, val_cutoff)
            tmp.to_csv(path, index=False)
            print(f"  Split added to {path.name}: train={( tmp['dataset_split']=='train').sum()} val={(tmp['dataset_split']=='val').sum()} test={(tmp['dataset_split']=='test').sum()}")

    summary_lines = [
        f"clean_email_rows={total_clean_rows}",
        f"clean_email_users={daily_features['user'].nunique()}",
        f"email_date_min={min_date}",
        f"email_date_max={max_date}",
        f"clean_psychometric_rows={len(psychometric_clean)}",
        f"clean_logon_rows={total_logon_rows}",
        f"clean_device_rows={total_device_rows}",
        f"clean_file_rows={total_file_rows}",
        f"clean_decoy_file_rows={len(decoy_clean)}",
        f"clean_users_rows={len(users_clean)}",
        f"daily_feature_rows={len(daily_features)}",
        f"daily_feature_users={daily_features['user'].nunique()}",
        f"rows_with_psychometric_match={(merged['o'].notna()).sum()}",
        f"logon_user_days={len(logon_daily)}",
        f"device_user_days={len(device_daily)}",
        f"file_user_days={len(file_daily)}",
    ] + chunk_summaries
    SUMMARY_OUTPUT.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"Saved {EMAIL_OUTPUT}")
    print(f"Saved {PSY_OUTPUT}")
    print(f"Saved {LOGON_OUTPUT}")
    print(f"Saved {DEVICE_OUTPUT}")
    print(f"Saved {FILE_OUTPUT}")
    print(f"Saved {DECOY_FILE_OUTPUT}")
    print(f"Saved {USERS_OUTPUT}")
    print(f"Saved {USER_DAILY_OUTPUT}")
    print(f"Saved {MERGED_OUTPUT}")
    print(f"Saved {SUMMARY_OUTPUT}")


if __name__ == "__main__":
    main()
