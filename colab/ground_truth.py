"""CERT ground-truth helpers with automatic release matching.

The Kaggle mirror used in Colab may contain activity logs from a different
CERT release than the project README originally assumed.  These helpers select
the answer-key release whose insider users actually overlap the scored files,
so evaluation can reuse existing artifacts without redownloading or retraining.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


REPO_DIR = Path(os.environ.get("DLP_REPO", str(Path(__file__).resolve().parent.parent)))


@dataclass(frozen=True)
class GroundTruthSelection:
    dataset: str
    insiders_path: Path
    insider_users: set[str]
    matching_users: set[str]
    match_count: int
    total_release_users: int
    scored_user_count: int
    release_match_counts: dict[str, int]


def _dataset_label(value: object) -> str:
    """Return a stable label such as '4.2' instead of float artifacts."""
    try:
        number = float(value)
        return str(int(number)) if number.is_integer() else f"{number:g}"
    except (TypeError, ValueError):
        return str(value)


def candidate_insiders_paths(repo_dir: Path | None = None) -> list[Path]:
    """Search likely Colab/local locations for answers/insiders.csv."""
    repo = repo_dir or REPO_DIR
    paths: list[Path] = []

    dlp_root = os.environ.get("DLP_ROOT")
    if dlp_root:
        paths.append(Path(dlp_root) / "archive" / "answers" / "answers" / "insiders.csv")

    paths.extend([
        repo / "archive" / "answers" / "answers" / "insiders.csv",
        repo.parent / "dlp-data" / "archive" / "answers" / "answers" / "insiders.csv",
        Path("/content/dlp-data/archive/answers/answers/insiders.csv"),
        Path("/content/dlp-project/archive/answers/answers/insiders.csv"),
    ])

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.expanduser()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def find_insiders_csv(repo_dir: Path | None = None) -> Path:
    for path in candidate_insiders_paths(repo_dir):
        if path.exists():
            return path
    searched = "\n  - ".join(str(p) for p in candidate_insiders_paths(repo_dir))
    raise FileNotFoundError(
        "insiders.csv was not found. Searched:\n"
        f"  - {searched}\n"
        "Set DLP_ROOT to your artifact folder or DLP_REPO to the cloned repo."
    )


def load_scored_users(scored_paths: Iterable[Path]) -> set[str]:
    users: set[str] = set()
    for path in scored_paths:
        if not path.exists():
            continue
        try:
            for chunk in pd.read_csv(path, usecols=["user"], chunksize=500_000):
                users.update(chunk["user"].dropna().astype(str).unique())
        except ValueError:
            continue
    return users


def select_ground_truth_release(
    scored_paths: Iterable[Path],
    insiders_path: Path | None = None,
) -> GroundTruthSelection:
    """Pick the CERT answer-key release that matches existing scored users."""
    path = insiders_path or find_insiders_csv()
    scored_users = load_scored_users(scored_paths)
    if not scored_users:
        raise FileNotFoundError(
            "No scored users were found. Expected at least one existing scored CSV:\n"
            + "\n".join(f"  - {p}" for p in scored_paths)
        )

    insiders = pd.read_csv(path)
    required = {"dataset", "user"}
    missing = required - set(insiders.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    insiders = insiders.copy()
    insiders["dataset_label"] = insiders["dataset"].map(_dataset_label)
    insiders["user"] = insiders["user"].astype(str)

    release_users: dict[str, set[str]] = {
        label: set(group["user"].dropna().astype(str).unique())
        for label, group in insiders.groupby("dataset_label")
    }
    match_counts = {
        label: len(users & scored_users)
        for label, users in release_users.items()
    }
    best_label = max(
        match_counts,
        key=lambda label: (match_counts[label], len(release_users[label])),
    )
    matching_users = release_users[best_label] & scored_users
    if not matching_users:
        counts = ", ".join(f"{k}: {v}" for k, v in sorted(match_counts.items()))
        raise ValueError(
            "No CERT ground-truth release matches the scored user IDs. "
            f"Per-release matches: {counts}"
        )

    return GroundTruthSelection(
        dataset=best_label,
        insiders_path=path,
        insider_users=release_users[best_label],
        matching_users=matching_users,
        match_count=len(matching_users),
        total_release_users=len(release_users[best_label]),
        scored_user_count=len(scored_users),
        release_match_counts=match_counts,
    )


def load_day_labels(
    scored_paths: Iterable[Path],
    insiders_path: Path | None = None,
) -> tuple[pd.DataFrame, GroundTruthSelection]:
    """Expand selected-release insider windows to one row per user-day."""
    selection = select_ground_truth_release(scored_paths, insiders_path)
    insiders = pd.read_csv(selection.insiders_path)
    insiders["dataset_label"] = insiders["dataset"].map(_dataset_label)
    insiders["user"] = insiders["user"].astype(str)
    insiders = insiders[
        (insiders["dataset_label"] == selection.dataset)
        & (insiders["user"].isin(selection.matching_users))
    ].copy()

    insiders["start"] = pd.to_datetime(insiders["start"], errors="coerce").dt.normalize()
    insiders["end"] = pd.to_datetime(insiders["end"], errors="coerce").dt.normalize()

    rows: list[dict[str, object]] = []
    for _, row in insiders.iterrows():
        if pd.isna(row["start"]) or pd.isna(row["end"]):
            continue
        for day in pd.date_range(row["start"], row["end"], freq="D"):
            rows.append({"user": row["user"], "email_day": day})

    labels = pd.DataFrame(rows, columns=["user", "email_day"]).drop_duplicates()
    labels["is_insider"] = 1
    return labels, selection


def describe_selection(selection: GroundTruthSelection) -> str:
    counts = ", ".join(
        f"{release}={count}"
        for release, count in sorted(selection.release_match_counts.items())
    )
    return (
        f"CERT {selection.dataset} selected from {selection.insiders_path} "
        f"({selection.match_count}/{selection.total_release_users} insider users "
        f"match {selection.scored_user_count} scored users; matches by release: {counts})"
    )
