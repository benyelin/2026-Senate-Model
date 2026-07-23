#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]

INPUT_PATH = (
    PROJECT_ROOT
    / "historical"
    / "senate"
    / "warehouse"
    / "raw"
    / "polling"
    / "fivethirtyeight"
    / "archives"
    / "senate_polls_historical.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "historical"
    / "senate"
    / "warehouse"
    / "validation"
    / "polling"
    / "fivethirtyeight_schema_audit"
)

SUMMARY_JSON = OUTPUT_DIR / "schema_audit_summary.json"
SUMMARY_TXT = OUTPUT_DIR / "schema_audit_summary.txt"
COLUMN_PROFILE_CSV = OUTPUT_DIR / "column_profile.csv"
KEY_PROFILE_CSV = OUTPUT_DIR / "candidate_key_profile.csv"
DUPLICATES_CSV = OUTPUT_DIR / "exact_duplicate_rows.csv"
QUESTION_PROFILE_CSV = OUTPUT_DIR / "question_structure_profile.csv"
POLL_PROFILE_CSV = OUTPUT_DIR / "poll_structure_profile.csv"
RACE_PROFILE_CSV = OUTPUT_DIR / "race_structure_profile.csv"
CATEGORY_PROFILE_CSV = OUTPUT_DIR / "categorical_value_profile.csv"
DATE_PROFILE_CSV = OUTPUT_DIR / "date_profile.csv"
SAMPLE_ROWS_CSV = OUTPUT_DIR / "sample_rows.csv"


DATE_COLUMNS = (
    "start_date",
    "end_date",
    "election_date",
    "created_at",
)

CATEGORICAL_COLUMNS = (
    "cycle",
    "state",
    "office_type",
    "seat_number",
    "seat_name",
    "stage",
    "population",
    "population_full",
    "methodology",
    "tracking",
    "internal",
    "partisan",
    "hypothetical",
    "nationwide_batch",
    "ranked_choice_reallocated",
    "ranked_choice_round",
    "party",
    "sponsor_candidate_party",
    "endorsed_candidate_party",
)

NUMERIC_COLUMNS = (
    "poll_id",
    "pollster_id",
    "pollster_rating_id",
    "question_id",
    "race_id",
    "candidate_id",
    "sample_size",
    "pct",
    "numeric_grade",
    "pollscore",
    "transparency_score",
)

POSSIBLE_ROW_KEYS = (
    ("poll_id", "question_id", "candidate_id"),
    ("poll_id", "question_id", "answer"),
    ("question_id", "candidate_id"),
    ("question_id", "answer"),
    ("poll_id", "question_id", "party", "answer"),
    (
        "poll_id",
        "question_id",
        "candidate_id",
        "ranked_choice_round",
    ),
)


def clean_string_series(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .replace(
            {
                "": pd.NA,
                "nan": pd.NA,
                "None": pd.NA,
                "NaN": pd.NA,
            }
        )
    )


def parse_date_series(series: pd.Series) -> pd.Series:
    text = clean_string_series(series)

    result = pd.Series(
        pd.NaT,
        index=series.index,
        dtype="datetime64[ns]",
    )

    for fmt in (
        "%m/%d/%y",
        "%m/%d/%Y",
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
    ):
        missing = result.isna() & text.notna()

        if not missing.any():
            break

        parsed = pd.to_datetime(
            text.loc[missing],
            format=fmt,
            errors="coerce",
            utc=False,
        )

        try:
            parsed = parsed.dt.tz_localize(None)
        except (AttributeError, TypeError):
            pass

        result.loc[missing] = parsed

    missing = result.isna() & text.notna()

    if missing.any():
        parsed = pd.to_datetime(
            text.loc[missing],
            errors="coerce",
            utc=True,
        )

        try:
            parsed = parsed.dt.tz_convert(None)
        except (AttributeError, TypeError):
            pass

        result.loc[missing] = parsed

    return result


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if pd.isna(value):
        return None

    return value


def profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for column in df.columns:
        series = df[column]
        nonnull = series.notna()
        nonnull_count = int(nonnull.sum())
        unique_count = int(series.nunique(dropna=True))

        example_values = (
            series.loc[nonnull]
            .astype(str)
            .drop_duplicates()
            .head(5)
            .tolist()
        )

        records.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "row_count": len(df),
                "nonnull_count": nonnull_count,
                "null_count": int(series.isna().sum()),
                "null_rate": float(series.isna().mean()),
                "unique_nonnull_count": unique_count,
                "uniqueness_among_nonnull": (
                    float(unique_count / nonnull_count)
                    if nonnull_count
                    else None
                ),
                "example_values": " | ".join(example_values),
            }
        )

    return pd.DataFrame(records)


def profile_candidate_keys(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for key in POSSIBLE_ROW_KEYS:
        if not all(column in df.columns for column in key):
            continue

        key_frame = df[list(key)]
        complete_mask = key_frame.notna().all(axis=1)
        complete = key_frame.loc[complete_mask]

        duplicate_mask = complete.duplicated(
            subset=list(key),
            keep=False,
        )

        records.append(
            {
                "candidate_key": " + ".join(key),
                "columns": len(key),
                "complete_rows": int(complete_mask.sum()),
                "incomplete_rows": int((~complete_mask).sum()),
                "unique_complete_keys": int(
                    complete.drop_duplicates().shape[0]
                ),
                "duplicate_complete_rows": int(
                    duplicate_mask.sum()
                ),
                "is_unique_when_complete": bool(
                    not duplicate_mask.any()
                ),
            }
        )

    return pd.DataFrame(records)


def profile_questions(df: pd.DataFrame) -> pd.DataFrame:
    if "question_id" not in df.columns:
        return pd.DataFrame()

    aggregations: dict[str, tuple[str, str]] = {
        "rows": ("question_id", "size"),
    }

    optional_nunique = (
        "poll_id",
        "race_id",
        "candidate_id",
        "candidate_name",
        "answer",
        "party",
        "state",
        "cycle",
        "ranked_choice_round",
    )

    for column in optional_nunique:
        if column in df.columns:
            aggregations[f"unique_{column}"] = (
                column,
                "nunique",
            )

    grouped = (
        df.groupby(
            "question_id",
            dropna=False,
        )
        .agg(**aggregations)
        .reset_index()
    )

    sort_columns = [
        column
        for column in (
            "rows",
            "unique_poll_id",
            "unique_candidate_id",
            "unique_answer",
        )
        if column in grouped.columns
    ]

    return grouped.sort_values(
        sort_columns,
        ascending=False,
    )


def profile_polls(df: pd.DataFrame) -> pd.DataFrame:
    if "poll_id" not in df.columns:
        return pd.DataFrame()

    aggregations: dict[str, tuple[str, str]] = {
        "rows": ("poll_id", "size"),
    }

    for column in (
        "question_id",
        "race_id",
        "state",
        "cycle",
        "population",
        "pollster_id",
        "pollster",
    ):
        if column in df.columns:
            aggregations[f"unique_{column}"] = (
                column,
                "nunique",
            )

    return (
        df.groupby(
            "poll_id",
            dropna=False,
        )
        .agg(**aggregations)
        .reset_index()
        .sort_values(
            "rows",
            ascending=False,
        )
    )


def profile_races(df: pd.DataFrame) -> pd.DataFrame:
    grouping_columns = [
        column
        for column in (
            "cycle",
            "race_id",
            "state",
            "seat_number",
            "seat_name",
            "election_date",
            "stage",
        )
        if column in df.columns
    ]

    if not grouping_columns:
        return pd.DataFrame()

    aggregations: dict[str, tuple[str, str]] = {
        "rows": (grouping_columns[0], "size"),
    }

    for column in (
        "poll_id",
        "question_id",
        "candidate_id",
        "candidate_name",
        "party",
        "end_date",
    ):
        if column in df.columns:
            aggregations[f"unique_{column}"] = (
                column,
                "nunique",
            )

    return (
        df.groupby(
            grouping_columns,
            dropna=False,
        )
        .agg(**aggregations)
        .reset_index()
        .sort_values(
            grouping_columns,
        )
    )


def profile_categories(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for column in CATEGORICAL_COLUMNS:
        if column not in df.columns:
            continue

        normalized = clean_string_series(df[column])
        counts = normalized.value_counts(
            dropna=False,
        )

        for value, count in counts.items():
            records.append(
                {
                    "column": column,
                    "value": (
                        "<NULL>"
                        if pd.isna(value)
                        else str(value)
                    ),
                    "count": int(count),
                    "share": float(count / len(df)),
                }
            )

    return pd.DataFrame(records)


def profile_dates(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for column in DATE_COLUMNS:
        if column not in df.columns:
            continue

        raw = df[column]
        parsed = parse_date_series(raw)
        nonnull_raw = raw.notna()

        records.append(
            {
                "column": column,
                "raw_nonnull_count": int(nonnull_raw.sum()),
                "parsed_nonnull_count": int(parsed.notna().sum()),
                "parse_failure_count": int(
                    (nonnull_raw & parsed.isna()).sum()
                ),
                "minimum_date": (
                    parsed.min().isoformat()
                    if parsed.notna().any()
                    else None
                ),
                "maximum_date": (
                    parsed.max().isoformat()
                    if parsed.notna().any()
                    else None
                ),
            }
        )

    return pd.DataFrame(records)


def count_question_answer_rows(
    df: pd.DataFrame,
) -> dict[str, Any]:
    result: dict[str, Any] = {}

    if "question_id" not in df.columns:
        return result

    row_counts = df.groupby(
        "question_id",
        dropna=False,
    ).size()

    result["question_count"] = int(len(row_counts))
    result["rows_per_question_min"] = int(row_counts.min())
    result["rows_per_question_median"] = float(
        row_counts.median()
    )
    result["rows_per_question_max"] = int(row_counts.max())
    result["rows_per_question_distribution"] = {
        str(int(rows)): int(count)
        for rows, count in Counter(
            row_counts.astype(int)
        ).most_common()
    }

    return result


def inspect_pct_totals(df: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}

    if not {
        "question_id",
        "pct",
    }.issubset(df.columns):
        return result

    pct = pd.to_numeric(
        df["pct"],
        errors="coerce",
    )

    work = pd.DataFrame(
        {
            "question_id": df["question_id"],
            "pct": pct,
        }
    )

    grouped = work.groupby(
        "question_id",
        dropna=False,
    )["pct"].agg(
        answer_rows="size",
        valid_pct_rows="count",
        pct_sum="sum",
        pct_min="min",
        pct_max="max",
    )

    result["question_pct_sum_min"] = float(
        grouped["pct_sum"].min()
    )
    result["question_pct_sum_median"] = float(
        grouped["pct_sum"].median()
    )
    result["question_pct_sum_max"] = float(
        grouped["pct_sum"].max()
    )

    result["questions_pct_sum_below_90"] = int(
        (grouped["pct_sum"] < 90).sum()
    )
    result["questions_pct_sum_90_to_105"] = int(
        grouped["pct_sum"].between(
            90,
            105,
            inclusive="both",
        ).sum()
    )
    result["questions_pct_sum_above_105"] = int(
        (grouped["pct_sum"] > 105).sum()
    )

    return result


def build_text_summary(
    summary: dict[str, Any],
    key_profile: pd.DataFrame,
    date_profile: pd.DataFrame,
) -> str:
    lines = [
        "=" * 92,
        "FIVETHIRTYEIGHT SENATE POLLING SCHEMA AUDIT",
        "=" * 92,
        "",
        f"Input: {summary['input_path']}",
        f"Rows: {summary['row_count']:,}",
        f"Columns: {summary['column_count']:,}",
        "",
        "ROW GRANULARITY",
        "-" * 92,
        (
            "Exact duplicate rows: "
            f"{summary['exact_duplicate_rows']:,}"
        ),
        (
            "Unique poll IDs: "
            f"{summary.get('unique_poll_ids')}"
        ),
        (
            "Unique question IDs: "
            f"{summary.get('unique_question_ids')}"
        ),
        (
            "Unique race IDs: "
            f"{summary.get('unique_race_ids')}"
        ),
        (
            "Unique candidate IDs: "
            f"{summary.get('unique_candidate_ids')}"
        ),
        "",
    ]

    for field in (
        "question_count",
        "rows_per_question_min",
        "rows_per_question_median",
        "rows_per_question_max",
    ):
        if field in summary:
            lines.append(
                f"{field}: {summary[field]}"
            )

    lines.extend(
        [
            "",
            "CANDIDATE ROW KEYS",
            "-" * 92,
        ]
    )

    if key_profile.empty:
        lines.append("No candidate keys could be tested.")
    else:
        lines.append(
            key_profile.to_string(index=False)
        )

    lines.extend(
        [
            "",
            "DATE PARSING",
            "-" * 92,
        ]
    )

    if date_profile.empty:
        lines.append("No recognized date columns found.")
    else:
        lines.append(
            date_profile.to_string(index=False)
        )

    lines.extend(
        [
            "",
            "CYCLE COVERAGE",
            "-" * 92,
            json.dumps(
                summary.get(
                    "rows_by_cycle",
                    {},
                ),
                indent=2,
                sort_keys=True,
            ),
            "",
            "STATE COVERAGE BY CYCLE",
            "-" * 92,
            json.dumps(
                summary.get(
                    "states_by_cycle",
                    {},
                ),
                indent=2,
                sort_keys=True,
            ),
            "",
            "QUESTION PERCENTAGE TOTALS",
            "-" * 92,
        ]
    )

    for field in (
        "question_pct_sum_min",
        "question_pct_sum_median",
        "question_pct_sum_max",
        "questions_pct_sum_below_90",
        "questions_pct_sum_90_to_105",
        "questions_pct_sum_above_105",
    ):
        if field in summary:
            lines.append(
                f"{field}: {summary[field]}"
            )

    lines.extend(
        [
            "",
            "OUTPUTS",
            "-" * 92,
            f"Column profile: {COLUMN_PROFILE_CSV}",
            f"Candidate keys: {KEY_PROFILE_CSV}",
            f"Question profile: {QUESTION_PROFILE_CSV}",
            f"Poll profile: {POLL_PROFILE_CSV}",
            f"Race profile: {RACE_PROFILE_CSV}",
            f"Category profile: {CATEGORY_PROFILE_CSV}",
            f"Date profile: {DATE_PROFILE_CSV}",
            f"Exact duplicates: {DUPLICATES_CSV}",
            f"Sample rows: {SAMPLE_ROWS_CSV}",
            f"JSON summary: {SUMMARY_JSON}",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> int:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Frozen Senate polling archive not found: "
            f"{INPUT_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 92)
    print("AUDIT FIVETHIRTYEIGHT SENATE POLLING SCHEMA")
    print("=" * 92)
    print(f"Reading: {INPUT_PATH}")

    df = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    print(
        f"Loaded {len(df):,} rows and "
        f"{len(df.columns):,} columns."
    )

    # Standardize obvious blank strings for auditing only.
    for column in df.select_dtypes(
        include=["object", "string"]
    ).columns:
        df[column] = clean_string_series(
            df[column]
        )

    column_profile = profile_columns(df)
    key_profile = profile_candidate_keys(df)
    question_profile = profile_questions(df)
    poll_profile = profile_polls(df)
    race_profile = profile_races(df)
    category_profile = profile_categories(df)
    date_profile = profile_dates(df)

    exact_duplicate_mask = df.duplicated(
        keep=False,
    )
    exact_duplicates = df.loc[
        exact_duplicate_mask
    ].copy()

    sample_columns = [
        column
        for column in (
            "poll_id",
            "question_id",
            "race_id",
            "cycle",
            "state",
            "seat_number",
            "seat_name",
            "election_date",
            "stage",
            "pollster",
            "start_date",
            "end_date",
            "population",
            "sample_size",
            "candidate_id",
            "candidate_name",
            "party",
            "answer",
            "pct",
            "ranked_choice_round",
            "hypothetical",
            "internal",
            "partisan",
        )
        if column in df.columns
    ]

    sample_rows = (
        df[sample_columns]
        .sort_values(
            [
                column
                for column in (
                    "cycle",
                    "state",
                    "election_date",
                    "poll_id",
                    "question_id",
                    "party",
                )
                if column in sample_columns
            ],
            na_position="last",
        )
        .head(250)
    )

    summary: dict[str, Any] = {
        "input_path": str(INPUT_PATH),
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": list(map(str, df.columns)),
        "exact_duplicate_rows": int(
            exact_duplicate_mask.sum()
        ),
        "unique_poll_ids": (
            int(df["poll_id"].nunique(dropna=True))
            if "poll_id" in df.columns
            else None
        ),
        "unique_question_ids": (
            int(
                df["question_id"].nunique(
                    dropna=True
                )
            )
            if "question_id" in df.columns
            else None
        ),
        "unique_race_ids": (
            int(df["race_id"].nunique(dropna=True))
            if "race_id" in df.columns
            else None
        ),
        "unique_candidate_ids": (
            int(
                df["candidate_id"].nunique(
                    dropna=True
                )
            )
            if "candidate_id" in df.columns
            else None
        ),
    }

    if "cycle" in df.columns:
        cycle_numeric = pd.to_numeric(
            df["cycle"],
            errors="coerce",
        )

        summary["rows_by_cycle"] = {
            str(int(cycle)): int(count)
            for cycle, count in (
                cycle_numeric.dropna()
                .astype(int)
                .value_counts()
                .sort_index()
                .items()
            )
        }

        if "state" in df.columns:
            cycle_state = pd.DataFrame(
                {
                    "cycle": cycle_numeric,
                    "state": df["state"],
                }
            ).dropna(
                subset=["cycle", "state"]
            )

            summary["states_by_cycle"] = {
                str(int(cycle)): int(
                    group["state"].nunique()
                )
                for cycle, group in cycle_state.groupby(
                    "cycle"
                )
            }

    if {
        "cycle",
        "question_id",
    }.issubset(df.columns):
        summary["questions_by_cycle"] = {
            str(int(cycle)): int(
                group["question_id"].nunique(
                    dropna=True
                )
            )
            for cycle, group in df.groupby(
                pd.to_numeric(
                    df["cycle"],
                    errors="coerce",
                ),
                dropna=True,
            )
        }

    if {
        "cycle",
        "poll_id",
    }.issubset(df.columns):
        summary["polls_by_cycle"] = {
            str(int(cycle)): int(
                group["poll_id"].nunique(
                    dropna=True
                )
            )
            for cycle, group in df.groupby(
                pd.to_numeric(
                    df["cycle"],
                    errors="coerce",
                ),
                dropna=True,
            )
        }

    summary.update(
        count_question_answer_rows(df)
    )
    summary.update(
        inspect_pct_totals(df)
    )

    column_profile.to_csv(
        COLUMN_PROFILE_CSV,
        index=False,
    )
    key_profile.to_csv(
        KEY_PROFILE_CSV,
        index=False,
    )
    question_profile.to_csv(
        QUESTION_PROFILE_CSV,
        index=False,
    )
    poll_profile.to_csv(
        POLL_PROFILE_CSV,
        index=False,
    )
    race_profile.to_csv(
        RACE_PROFILE_CSV,
        index=False,
    )
    category_profile.to_csv(
        CATEGORY_PROFILE_CSV,
        index=False,
    )
    date_profile.to_csv(
        DATE_PROFILE_CSV,
        index=False,
    )
    exact_duplicates.to_csv(
        DUPLICATES_CSV,
        index=False,
    )
    sample_rows.to_csv(
        SAMPLE_ROWS_CSV,
        index=False,
    )

    SUMMARY_JSON.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
            default=json_safe,
        )
        + "\n",
        encoding="utf-8",
    )

    text_summary = build_text_summary(
        summary,
        key_profile,
        date_profile,
    )

    SUMMARY_TXT.write_text(
        text_summary,
        encoding="utf-8",
    )

    print()
    print(text_summary)

    print("=" * 92)
    print("AUDIT COMPLETED")
    print("=" * 92)

    return 0


if __name__ == "__main__":
    sys.exit(main())
