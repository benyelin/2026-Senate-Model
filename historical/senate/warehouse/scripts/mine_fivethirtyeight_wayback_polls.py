#!/usr/bin/env python3

from __future__ import annotations

import csv
import ssl
import hashlib
import io
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
import certifi


PROJECT_ROOT = Path(__file__).resolve().parents[4]

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "historical"
    / "senate"
    / "warehouse"
    / "raw"
    / "polling"
    / "source_downloads"
    / "wayback"
)

SNAPSHOT_DIR = OUTPUT_ROOT / "snapshots"
BEST_DIR = OUTPUT_ROOT / "best_snapshots"

CDX_ENDPOINT = "https://web.archive.org/cdx/search/cdx"

REQUEST_TIMEOUT_SECONDS = 60
REQUEST_DELAY_SECONDS = 0.75
MAX_RETRIES = 4


# We search both the documented projects.fivethirtyeight.com endpoints
# and plausible older aliases that may have been captured independently.
TARGETS: dict[str, list[str]] = {
    "senate_polls_historical": [
        (
            "https://projects.fivethirtyeight.com/"
            "polls-page/data/senate_polls_historical.csv"
        ),
        (
            "http://projects.fivethirtyeight.com/"
            "polls-page/data/senate_polls_historical.csv"
        ),
        (
            "https://projects.fivethirtyeight.com/"
            "polls/senate_polls_historical.csv"
        ),
        (
            "https://projects.fivethirtyeight.com/"
            "polls-page/senate_polls_historical.csv"
        ),
    ],
    "house_polls_historical": [
        (
            "https://projects.fivethirtyeight.com/"
            "polls-page/data/house_polls_historical.csv"
        ),
        (
            "http://projects.fivethirtyeight.com/"
            "polls-page/data/house_polls_historical.csv"
        ),
        (
            "https://projects.fivethirtyeight.com/"
            "polls/house_polls_historical.csv"
        ),
        (
            "https://projects.fivethirtyeight.com/"
            "polls-page/house_polls_historical.csv"
        ),
    ],
    "generic_ballot_polls_historical": [
        (
            "https://projects.fivethirtyeight.com/"
            "polls-page/data/generic_ballot_polls_historical.csv"
        ),
        (
            "http://projects.fivethirtyeight.com/"
            "polls-page/data/generic_ballot_polls_historical.csv"
        ),
        (
            "https://projects.fivethirtyeight.com/"
            "polls/generic_ballot_polls_historical.csv"
        ),
        (
            "https://projects.fivethirtyeight.com/"
            "polls-page/generic_ballot_polls_historical.csv"
        ),
    ],
}


@dataclass(frozen=True)
class Capture:
    dataset: str
    requested_url: str
    timestamp: str
    original: str
    mimetype: str
    statuscode: str
    digest: str
    length: str

    @property
    def replay_url(self) -> str:
        # "id_" requests the original archived payload without the
        # Wayback toolbar and HTML rewriting.
        return (
            f"https://web.archive.org/web/"
            f"{self.timestamp}id_/{self.original}"
        )


def request_bytes(
    url: str,
    *,
    accept: str = "*/*",
) -> tuple[bytes, dict[str, str], str]:
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        request = Request(
            url,
            headers={
                "User-Agent": (
                    "SenateElectionModelHistoricalPollingResearch/1.0"
                ),
                "Accept": accept,
            },
        )

        try:
            ssl_context = ssl.create_default_context(
                cafile=certifi.where()
            )

            with urlopen(
                request,
                timeout=REQUEST_TIMEOUT_SECONDS,
                context=ssl_context,
            ) as response:
                body = response.read()
                headers = {
                    key.lower(): value
                    for key, value in response.headers.items()
                }
                final_url = response.geturl()
                return body, headers, final_url

        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc

            if attempt < MAX_RETRIES:
                delay = 2 ** (attempt - 1)
                print(
                    f"  Request failed on attempt {attempt}: {exc}. "
                    f"Retrying in {delay}s..."
                )
                time.sleep(delay)

    raise RuntimeError(
        f"Request failed after {MAX_RETRIES} attempts: {url}"
    ) from last_error


def query_cdx(
    dataset: str,
    requested_url: str,
) -> list[Capture]:
    params = [
        ("url", requested_url),
        ("output", "json"),
        (
            "fl",
            (
                "timestamp,original,mimetype,statuscode,"
                "digest,length"
            ),
        ),
        ("filter", "statuscode:200"),
        ("filter", "mimetype:.*"),
        ("collapse", "digest"),
    ]

    url = f"{CDX_ENDPOINT}?{urlencode(params)}"

    print(f"\nCDX query: {requested_url}")

    body, _, _ = request_bytes(
        url,
        accept="application/json,text/plain,*/*",
    )

    text = body.decode("utf-8", errors="replace").strip()

    if not text:
        print("  No CDX response rows.")
        return []

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        preview = text[:500]
        raise RuntimeError(
            f"CDX response was not valid JSON:\n{preview}"
        ) from exc

    if not payload or len(payload) < 2:
        print("  No archived captures found.")
        return []

    header = payload[0]
    rows = payload[1:]

    captures: list[Capture] = []

    for row in rows:
        record = dict(zip(header, row))

        captures.append(
            Capture(
                dataset=dataset,
                requested_url=requested_url,
                timestamp=str(record.get("timestamp", "")),
                original=str(record.get("original", "")),
                mimetype=str(record.get("mimetype", "")),
                statuscode=str(record.get("statuscode", "")),
                digest=str(record.get("digest", "")),
                length=str(record.get("length", "")),
            )
        )

    print(f"  Unique captures by digest: {len(captures):,}")
    return captures


def looks_like_html(body: bytes) -> bool:
    stripped = body.lstrip().lower()

    html_prefixes = (
        b"<!doctype html",
        b"<html",
        b"<head",
        b"<body",
    )

    return any(stripped.startswith(prefix) for prefix in html_prefixes)


def looks_like_csv(body: bytes) -> bool:
    if not body or looks_like_html(body):
        return False

    text = body[:10000].decode("utf-8-sig", errors="replace")

    first_nonempty = next(
        (
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ),
        "",
    )

    if not first_nonempty:
        return False

    try:
        parsed = next(csv.reader([first_nonempty]))
    except Exception:
        return False

    normalized = {
        value.strip().lower()
        for value in parsed
    }

    expected_markers = {
        "poll_id",
        "question_id",
        "cycle",
        "state",
        "pollster",
        "start_date",
        "end_date",
        "candidate_name",
        "pct",
    }

    return (
        len(parsed) >= 4
        and len(normalized & expected_markers) >= 2
    )


def parse_date_column(
    series: pd.Series,
) -> pd.Series:
    # FiveThirtyEight commonly used M/D/YY but not every archived
    # version was necessarily identical. Try explicit formats before
    # the generic parser.
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


def inspect_csv(body: bytes) -> dict[str, Any]:
    try:
        df = pd.read_csv(
            io.BytesIO(body),
            low_memory=False,
        )
    except Exception as exc:
        return {
            "csv_parse_success": False,
            "csv_parse_error": repr(exc),
        }

    columns = [str(column) for column in df.columns]

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

    date_candidates = [
        "end_date",
        "start_date",
        "election_date",
        "created_at",
    ]

    date_column_used = None
    minimum_date = None
    maximum_date = None

    for column in date_candidates:
        if column not in df.columns:
            continue

        parsed = parse_date_column(df[column])

        if parsed.notna().any():
            date_column_used = column
            minimum_date = parsed.min().date().isoformat()
            maximum_date = parsed.max().date().isoformat()
            break

    unique_poll_ids = (
        int(df["poll_id"].nunique(dropna=True))
        if "poll_id" in df.columns
        else None
    )

    unique_question_ids = (
        int(df["question_id"].nunique(dropna=True))
        if "question_id" in df.columns
        else None
    )

    unique_states = (
        int(df["state"].nunique(dropna=True))
        if "state" in df.columns
        else None
    )

    return {
        "csv_parse_success": True,
        "csv_parse_error": None,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": "|".join(columns),
        "cycles": ",".join(str(cycle) for cycle in cycles),
        "cycle_count": len(cycles),
        "minimum_cycle": min(cycles) if cycles else None,
        "maximum_cycle": max(cycles) if cycles else None,
        "date_column_used": date_column_used,
        "minimum_date": minimum_date,
        "maximum_date": maximum_date,
        "unique_poll_ids": unique_poll_ids,
        "unique_question_ids": unique_question_ids,
        "unique_states": unique_states,
    }


def safe_component(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value.strip("_") or "unknown"


def download_and_inspect(
    capture: Capture,
) -> dict[str, Any]:
    print(
        f"  Downloading {capture.timestamp} "
        f"{capture.original}"
    )

    base_record: dict[str, Any] = {
        "dataset": capture.dataset,
        "requested_url": capture.requested_url,
        "timestamp": capture.timestamp,
        "capture_date": (
            f"{capture.timestamp[0:4]}-"
            f"{capture.timestamp[4:6]}-"
            f"{capture.timestamp[6:8]}"
            if len(capture.timestamp) >= 8
            else None
        ),
        "original": capture.original,
        "replay_url": capture.replay_url,
        "cdx_mimetype": capture.mimetype,
        "cdx_statuscode": capture.statuscode,
        "cdx_digest": capture.digest,
        "cdx_length": capture.length,
        "download_success": False,
        "download_error": None,
    }

    try:
        body, headers, final_url = request_bytes(
            capture.replay_url,
            accept="text/csv,text/plain,*/*",
        )
    except Exception as exc:
        base_record["download_error"] = repr(exc)
        return base_record

    sha256 = hashlib.sha256(body).hexdigest()
    content_type = headers.get("content-type", "")
    detected_html = looks_like_html(body)
    detected_csv = looks_like_csv(body)

    dataset_dir = SNAPSHOT_DIR / capture.dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)

    suffix = ".csv" if detected_csv else ".bin"

    filename = (
        f"{capture.timestamp}_"
        f"{sha256[:12]}"
        f"{suffix}"
    )

    output_path = dataset_dir / filename
    output_path.write_bytes(body)

    base_record.update(
        {
            "download_success": True,
            "download_error": None,
            "final_url": final_url,
            "downloaded_bytes": len(body),
            "content_type": content_type,
            "sha256": sha256,
            "looks_like_html": detected_html,
            "looks_like_csv": detected_csv,
            "saved_path": str(output_path.relative_to(PROJECT_ROOT)),
        }
    )

    if detected_csv:
        base_record.update(inspect_csv(body))
    else:
        preview = body[:300].decode(
            "utf-8",
            errors="replace",
        )
        base_record.update(
            {
                "csv_parse_success": False,
                "csv_parse_error": "Payload did not look like CSV.",
                "payload_preview": repr(preview),
            }
        )

    return base_record


def rank_inventory(inventory: pd.DataFrame) -> pd.DataFrame:
    ranked = inventory.copy()

    for column in [
        "csv_parse_success",
        "rows",
        "cycle_count",
        "unique_poll_ids",
        "unique_question_ids",
        "downloaded_bytes",
    ]:
        if column not in ranked.columns:
            ranked[column] = 0

    ranked["csv_parse_success_score"] = (
        ranked["csv_parse_success"]
        .fillna(False)
        .astype(bool)
        .astype(int)
    )

    ranked["rows_score"] = pd.to_numeric(
        ranked["rows"],
        errors="coerce",
    ).fillna(0)

    ranked["cycle_count_score"] = pd.to_numeric(
        ranked["cycle_count"],
        errors="coerce",
    ).fillna(0)

    ranked["unique_poll_ids_score"] = pd.to_numeric(
        ranked["unique_poll_ids"],
        errors="coerce",
    ).fillna(0)

    ranked["downloaded_bytes_score"] = pd.to_numeric(
        ranked["downloaded_bytes"],
        errors="coerce",
    ).fillna(0)

    ranked = ranked.sort_values(
        [
            "dataset",
            "csv_parse_success_score",
            "cycle_count_score",
            "rows_score",
            "unique_poll_ids_score",
            "downloaded_bytes_score",
            "timestamp",
        ],
        ascending=[
            True,
            False,
            False,
            False,
            False,
            False,
            False,
        ],
        na_position="last",
    ).reset_index(drop=True)

    ranked["rank_within_dataset"] = (
        ranked.groupby("dataset").cumcount() + 1
    )

    return ranked


def save_best_snapshots(
    ranked: pd.DataFrame,
) -> None:
    BEST_DIR.mkdir(parents=True, exist_ok=True)

    usable = ranked[
        ranked["csv_parse_success"].fillna(False).astype(bool)
    ].copy()

    if usable.empty:
        print("\nNo valid CSV snapshots were recovered.")
        return

    for dataset, group in usable.groupby("dataset"):
        best = group.sort_values(
            [
                "cycle_count_score",
                "rows_score",
                "unique_poll_ids_score",
                "downloaded_bytes_score",
                "timestamp",
            ],
            ascending=False,
        ).iloc[0]

        source_path = PROJECT_ROOT / str(best["saved_path"])

        destination = (
            BEST_DIR
            / f"{safe_component(dataset)}_best.csv"
        )

        destination.write_bytes(source_path.read_bytes())

        metadata_path = (
            BEST_DIR
            / f"{safe_component(dataset)}_best_metadata.json"
        )

        metadata = {
            key: (
                None
                if pd.isna(value)
                else value.item()
                if hasattr(value, "item")
                else value
            )
            for key, value in best.to_dict().items()
        }

        metadata_path.write_text(
            json.dumps(
                metadata,
                indent=2,
                sort_keys=True,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )

        print(
            f"\nBest {dataset} snapshot:\n"
            f"  Source:      {source_path}\n"
            f"  Destination: {destination}\n"
            f"  Rows:        {metadata.get('rows')}\n"
            f"  Cycles:      {metadata.get('cycles')}\n"
            f"  Date range:  "
            f"{metadata.get('minimum_date')} through "
            f"{metadata.get('maximum_date')}"
        )


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    all_captures: list[Capture] = []
    query_failures: list[dict[str, str]] = []

    print("=" * 100)
    print("FIVETHIRTYEIGHT WAYBACK POLLING ARCHIVE MINER")
    print("=" * 100)

    for dataset, urls in TARGETS.items():
        print(f"\n{'=' * 100}")
        print(dataset)
        print("=" * 100)

        for requested_url in urls:
            try:
                captures = query_cdx(
                    dataset,
                    requested_url,
                )
                all_captures.extend(captures)
            except Exception as exc:
                print(f"  CDX query failed: {exc}")

                query_failures.append(
                    {
                        "dataset": dataset,
                        "requested_url": requested_url,
                        "error": repr(exc),
                    }
                )

            time.sleep(REQUEST_DELAY_SECONDS)

    capture_rows = [
        {
            "dataset": capture.dataset,
            "requested_url": capture.requested_url,
            "timestamp": capture.timestamp,
            "original": capture.original,
            "mimetype": capture.mimetype,
            "statuscode": capture.statuscode,
            "digest": capture.digest,
            "length": capture.length,
            "replay_url": capture.replay_url,
        }
        for capture in all_captures
    ]

    captures_path = OUTPUT_ROOT / "cdx_capture_inventory.csv"

    pd.DataFrame(capture_rows).to_csv(
        captures_path,
        index=False,
    )

    print(
        f"\nTotal unique-digest captures discovered: "
        f"{len(all_captures):,}"
    )

    # Avoid downloading the same archived payload multiple times when
    # HTTP and HTTPS URL aliases point to an identical capture.
    deduplicated: dict[
        tuple[str, str, str],
        Capture,
    ] = {}

    for capture in all_captures:
        key = (
            capture.dataset,
            capture.digest,
            capture.timestamp,
        )
        deduplicated.setdefault(key, capture)

    captures_to_download = list(deduplicated.values())

    print(
        f"Captures after alias deduplication: "
        f"{len(captures_to_download):,}"
    )

    inspection_records: list[dict[str, Any]] = []

    for index, capture in enumerate(
        captures_to_download,
        start=1,
    ):
        print(
            f"\n[{index:,}/{len(captures_to_download):,}] "
            f"{capture.dataset}"
        )

        record = download_and_inspect(capture)
        inspection_records.append(record)

        time.sleep(REQUEST_DELAY_SECONDS)

    inventory = pd.DataFrame(inspection_records)

    if inventory.empty:
        inventory = pd.DataFrame(
            columns=[
                "dataset",
                "requested_url",
                "timestamp",
                "download_success",
                "csv_parse_success",
            ]
        )

    ranked = rank_inventory(inventory)

    inventory_path = OUTPUT_ROOT / "snapshot_inventory_ranked.csv"
    ranked.to_csv(inventory_path, index=False)

    failure_path = OUTPUT_ROOT / "cdx_query_failures.csv"
    pd.DataFrame(query_failures).to_csv(
        failure_path,
        index=False,
    )

    save_best_snapshots(ranked)

    print("\n" + "=" * 100)
    print("TOP RECOVERED SNAPSHOTS")
    print("=" * 100)

    display_columns = [
        column
        for column in [
            "dataset",
            "rank_within_dataset",
            "capture_date",
            "rows",
            "cycles",
            "minimum_date",
            "maximum_date",
            "unique_poll_ids",
            "unique_question_ids",
            "unique_states",
            "downloaded_bytes",
            "content_type",
            "saved_path",
        ]
        if column in ranked.columns
    ]

    valid = ranked[
        ranked["csv_parse_success"]
        .fillna(False)
        .astype(bool)
    ]

    if valid.empty:
        print("No valid CSV snapshots recovered.")
    else:
        print(
            valid[display_columns]
            .groupby("dataset", group_keys=False)
            .head(10)
            .to_string(index=False)
        )

    print("\n" + "=" * 100)
    print("OUTPUTS")
    print("=" * 100)
    print(f"CDX capture inventory: {captures_path}")
    print(f"Ranked snapshot inventory: {inventory_path}")
    print(f"Best snapshots directory: {BEST_DIR}")
    print(f"CDX failures: {failure_path}")

    if valid.empty:
        print(
            "\nRESULT: No valid historical CSV payloads were "
            "recovered from the tested URLs."
        )
        return 2

    print(
        "\nRESULT: At least one valid historical CSV snapshot "
        "was recovered."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
