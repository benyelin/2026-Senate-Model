#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]

WAYBACK_ROOT = (
    PROJECT_ROOT
    / "historical"
    / "senate"
    / "warehouse"
    / "raw"
    / "polling"
    / "source_downloads"
    / "wayback"
)

BEST_SNAPSHOT_DIR = WAYBACK_ROOT / "best_snapshots"

IMMUTABLE_ROOT = (
    PROJECT_ROOT
    / "historical"
    / "senate"
    / "warehouse"
    / "raw"
    / "polling"
    / "fivethirtyeight"
)

ARCHIVE_DIR = IMMUTABLE_ROOT / "archives"
MANIFEST_CSV = IMMUTABLE_ROOT / "archive_manifest.csv"
MANIFEST_JSON = IMMUTABLE_ROOT / "archive_manifest.json"
README_PATH = IMMUTABLE_ROOT / "README.md"


ARCHIVES: dict[str, dict[str, str]] = {
    "senate_polls_historical": {
        "source_filename": "senate_polls_historical_best.csv",
        "destination_filename": "senate_polls_historical.csv",
        "metadata_filename": (
            "senate_polls_historical_best_metadata.json"
        ),
        "official_url": (
            "https://projects.fivethirtyeight.com/"
            "polls-page/data/senate_polls_historical.csv"
        ),
    },
    "house_polls_historical": {
        "source_filename": "house_polls_historical_best.csv",
        "destination_filename": "house_polls_historical.csv",
        "metadata_filename": (
            "house_polls_historical_best_metadata.json"
        ),
        "official_url": (
            "https://projects.fivethirtyeight.com/"
            "polls-page/data/house_polls_historical.csv"
        ),
    },
    "generic_ballot_polls_historical": {
        "source_filename": (
            "generic_ballot_polls_historical_best.csv"
        ),
        "destination_filename": (
            "generic_ballot_polls_historical.csv"
        ),
        "metadata_filename": (
            "generic_ballot_polls_historical_best_metadata.json"
        ),
        "official_url": (
            "https://projects.fivethirtyeight.com/"
            "polls-page/data/"
            "generic_ballot_polls_historical.csv"
        ),
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def parse_dates(
    series: pd.Series,
) -> pd.Series:
    text = series.astype("string").str.strip()

    parsed = pd.Series(
        pd.NaT,
        index=series.index,
        dtype="datetime64[ns]",
    )

    for date_format in ("%m/%d/%y", "%m/%d/%Y", "%Y-%m-%d"):
        missing = parsed.isna()

        if not missing.any():
            break

        parsed.loc[missing] = pd.to_datetime(
            text.loc[missing],
            format=date_format,
            errors="coerce",
        )

    missing = parsed.isna()

    if missing.any():
        parsed.loc[missing] = pd.to_datetime(
            text.loc[missing],
            errors="coerce",
        )

    return parsed


def inspect_csv(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path, low_memory=False)

    cycles: list[int] = []

    if "cycle" in df.columns:
        cycles = sorted(
            pd.to_numeric(
                df["cycle"],
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

    date_column_used = None
    minimum_date = None
    maximum_date = None

    for column in (
        "end_date",
        "start_date",
        "election_date",
        "created_at",
    ):
        if column not in df.columns:
            continue

        parsed = parse_dates(df[column])

        if parsed.notna().any():
            date_column_used = column
            minimum_date = parsed.min().date().isoformat()
            maximum_date = parsed.max().date().isoformat()
            break

    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": list(map(str, df.columns)),
        "cycles": cycles,
        "cycle_count": len(cycles),
        "minimum_date": minimum_date,
        "maximum_date": maximum_date,
        "date_column_used": date_column_used,
        "unique_poll_ids": (
            int(df["poll_id"].nunique(dropna=True))
            if "poll_id" in df.columns
            else None
        ),
        "unique_question_ids": (
            int(df["question_id"].nunique(dropna=True))
            if "question_id" in df.columns
            else None
        ),
        "unique_states": (
            int(df["state"].nunique(dropna=True))
            if "state" in df.columns
            else None
        ),
    }


def load_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    return json.loads(path.read_text(encoding="utf-8"))


def copy_immutably(
    source: Path,
    destination: Path,
) -> tuple[str, bool]:
    source_hash = sha256_file(source)

    if destination.exists():
        destination_hash = sha256_file(destination)

        if destination_hash != source_hash:
            raise RuntimeError(
                "Immutable archive conflict:\n"
                f"  Existing file: {destination}\n"
                f"  Existing SHA-256: {destination_hash}\n"
                f"  Source SHA-256:   {source_hash}\n"
                "The existing archive was not overwritten."
            )

        return source_hash, False

    shutil.copy2(source, destination)

    destination_hash = sha256_file(destination)

    if destination_hash != source_hash:
        raise RuntimeError(
            f"Hash verification failed after copying {source}"
        )

    return source_hash, True


def build_readme(
    records: list[dict[str, Any]],
) -> str:
    lines = [
        "# FiveThirtyEight Historical Polling Archives",
        "",
        "## Purpose",
        "",
        (
            "This directory contains immutable raw polling archives "
            "recovered from the Internet Archive's Wayback Machine."
        ),
        "",
        (
            "These files should never be edited in place. All cleaning, "
            "normalization, filtering, and feature construction must occur "
            "in separate processed warehouse files."
        ),
        "",
        "## Provenance",
        "",
        (
            "The original FiveThirtyEight polling endpoints became "
            "unavailable after FiveThirtyEight ceased operating. Archived "
            "copies were enumerated through the Wayback CDX API and "
            "downloaded using timestamped raw-payload replay URLs."
        ),
        "",
        "## Preserved archives",
        "",
        "| Dataset | Rows | Cycles | Date range | SHA-256 |",
        "|---|---:|---|---|---|",
    ]

    for record in records:
        cycles = ", ".join(
            str(cycle)
            for cycle in record.get("cycles", [])
        )

        date_range = (
            f"{record.get('minimum_date')} through "
            f"{record.get('maximum_date')}"
        )

        lines.append(
            f"| `{record['destination_filename']}` "
            f"| {record['row_count']:,} "
            f"| {cycles} "
            f"| {date_range} "
            f"| `{record['sha256']}` |"
        )

    lines.extend(
        [
            "",
            "## Source details",
            "",
        ]
    )

    for record in records:
        lines.extend(
            [
                f"### {record['dataset']}",
                "",
                f"- Official URL: `{record['official_url']}`",
                (
                    "- Wayback capture timestamp: "
                    f"`{record.get('wayback_timestamp')}`"
                ),
                (
                    "- Wayback replay URL: "
                    f"`{record.get('wayback_replay_url')}`"
                ),
                (
                    "- Archived file: "
                    f"`archives/{record['destination_filename']}`"
                ),
                f"- SHA-256: `{record['sha256']}`",
                f"- Rows: {record['row_count']:,}",
                (
                    "- Columns: "
                    f"{record['column_count']:,}"
                ),
                (
                    "- Poll IDs: "
                    f"{record.get('unique_poll_ids')}"
                ),
                (
                    "- Question IDs: "
                    f"{record.get('unique_question_ids')}"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Data-handling rules",
            "",
            "1. Do not modify files in `archives/`.",
            (
                "2. Validate source hashes before every warehouse build."
            ),
            (
                "3. Preserve poll-level and question-level identifiers."
            ),
            (
                "4. Apply election-date cutoffs during snapshot creation "
                "to prevent look-ahead leakage."
            ),
            (
                "5. Record all cleaning and exclusions in processed-data "
                "validation reports."
            ),
            "",
            "## Generated files",
            "",
            "- `archive_manifest.csv`",
            "- `archive_manifest.json`",
            "- `README.md`",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> int:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []

    print("=" * 88)
    print("FREEZE FIVETHIRTYEIGHT HISTORICAL POLLING ARCHIVES")
    print("=" * 88)

    for dataset, config in ARCHIVES.items():
        source_path = (
            BEST_SNAPSHOT_DIR
            / config["source_filename"]
        )

        metadata_path = (
            BEST_SNAPSHOT_DIR
            / config["metadata_filename"]
        )

        destination_path = (
            ARCHIVE_DIR
            / config["destination_filename"]
        )

        if not source_path.exists():
            raise FileNotFoundError(
                f"Missing recovered source file: {source_path}"
            )

        metadata = load_metadata(metadata_path)

        source_hash, newly_copied = copy_immutably(
            source_path,
            destination_path,
        )

        inspection = inspect_csv(destination_path)

        record = {
            "dataset": dataset,
            "destination_filename": (
                config["destination_filename"]
            ),
            "archive_relative_path": str(
                destination_path.relative_to(PROJECT_ROOT)
            ),
            "official_url": config["official_url"],
            "wayback_timestamp": metadata.get("timestamp"),
            "wayback_capture_date": metadata.get(
                "capture_date"
            ),
            "wayback_original_url": metadata.get("original"),
            "wayback_replay_url": metadata.get("replay_url"),
            "source_snapshot_relative_path": str(
                source_path.relative_to(PROJECT_ROOT)
            ),
            "sha256": source_hash,
            "file_size_bytes": destination_path.stat().st_size,
            "newly_copied": newly_copied,
            "frozen_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            **inspection,
        }

        records.append(record)

        action = (
            "COPIED AND VERIFIED"
            if newly_copied
            else "ALREADY PRESENT; HASH VERIFIED"
        )

        print(f"\n{dataset}")
        print(f"  Status:      {action}")
        print(f"  File:        {destination_path}")
        print(f"  SHA-256:     {source_hash}")
        print(f"  Rows:        {inspection['row_count']:,}")
        print(
            "  Cycles:      "
            + ", ".join(
                str(cycle)
                for cycle in inspection["cycles"]
            )
        )
        print(
            f"  Date range:  "
            f"{inspection['minimum_date']} through "
            f"{inspection['maximum_date']}"
        )

    manifest_df = pd.DataFrame(records)

    # Store list-valued fields in a stable serialized form for CSV.
    csv_manifest = manifest_df.copy()

    for column in ("cycles", "columns"):
        if column in csv_manifest.columns:
            csv_manifest[column] = csv_manifest[column].apply(
                lambda value: json.dumps(value)
            )

    csv_manifest.to_csv(
        MANIFEST_CSV,
        index=False,
    )

    MANIFEST_JSON.write_text(
        json.dumps(
            records,
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    README_PATH.write_text(
        build_readme(records),
        encoding="utf-8",
    )

    print("\n" + "=" * 88)
    print("VALIDATION")
    print("=" * 88)

    expected_datasets = set(ARCHIVES)
    actual_datasets = {
        record["dataset"]
        for record in records
    }

    if actual_datasets != expected_datasets:
        raise RuntimeError(
            "Dataset manifest does not match expected archive set."
        )

    for record in records:
        path = PROJECT_ROOT / record["archive_relative_path"]

        current_hash = sha256_file(path)

        if current_hash != record["sha256"]:
            raise RuntimeError(
                f"Post-write hash mismatch: {path}"
            )

        if record["row_count"] <= 0:
            raise RuntimeError(
                f"Archive contains no rows: {path}"
            )

    print("Archive count:                  3")
    print("Missing archives:               0")
    print("Empty archives:                 0")
    print("Hash verification failures:     0")
    print("Immutable archive validation:   PASSED")

    print("\nOutputs:")
    print(f"  Archives: {ARCHIVE_DIR}")
    print(f"  CSV manifest: {MANIFEST_CSV}")
    print(f"  JSON manifest: {MANIFEST_JSON}")
    print(f"  README: {README_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
