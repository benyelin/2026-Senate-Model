#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import re
import sys
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
    / "processed"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "senate_historical_polls.csv"
)

VALIDATION_DIR = (
    PROJECT_ROOT
    / "historical"
    / "senate"
    / "warehouse"
    / "validation"
    / "polling"
    / "processed_warehouse"
)

SUMMARY_JSON = (
    VALIDATION_DIR
    / "senate_historical_polls_validation.json"
)

SUMMARY_TXT = (
    VALIDATION_DIR
    / "senate_historical_polls_validation.txt"
)

CYCLE_PROFILE_CSV = (
    VALIDATION_DIR
    / "cycle_profile.csv"
)

PARTY_PROFILE_CSV = (
    VALIDATION_DIR
    / "party_profile.csv"
)

STATE_PROFILE_CSV = (
    VALIDATION_DIR
    / "state_profile.csv"
)

QUESTION_PROFILE_CSV = (
    VALIDATION_DIR
    / "question_profile.csv"
)

FLAG_PROFILE_CSV = (
    VALIDATION_DIR
    / "flag_profile.csv"
)

EXCEPTION_ROWS_CSV = (
    VALIDATION_DIR
    / "exception_rows.csv"
)


REQUIRED_COLUMNS = {
    "poll_id",
    "question_id",
    "race_id",
    "candidate_id",
    "cycle",
    "state",
    "start_date",
    "end_date",
    "candidate_name",
    "party",
    "pct",
}


STATE_NAME_TO_ABBREVIATION = {
    "alabama": "AL",
    "alaska": "AK",
    "arizona": "AZ",
    "arkansas": "AR",
    "california": "CA",
    "colorado": "CO",
    "connecticut": "CT",
    "delaware": "DE",
    "district of columbia": "DC",
    "florida": "FL",
    "georgia": "GA",
    "hawaii": "HI",
    "idaho": "ID",
    "illinois": "IL",
    "indiana": "IN",
    "iowa": "IA",
    "kansas": "KS",
    "kentucky": "KY",
    "louisiana": "LA",
    "maine": "ME",
    "maryland": "MD",
    "massachusetts": "MA",
    "michigan": "MI",
    "minnesota": "MN",
    "mississippi": "MS",
    "missouri": "MO",
    "montana": "MT",
    "nebraska": "NE",
    "nevada": "NV",
    "new hampshire": "NH",
    "new jersey": "NJ",
    "new mexico": "NM",
    "new york": "NY",
    "north carolina": "NC",
    "north dakota": "ND",
    "ohio": "OH",
    "oklahoma": "OK",
    "oregon": "OR",
    "pennsylvania": "PA",
    "rhode island": "RI",
    "south carolina": "SC",
    "south dakota": "SD",
    "tennessee": "TN",
    "texas": "TX",
    "utah": "UT",
    "vermont": "VT",
    "virginia": "VA",
    "washington": "WA",
    "west virginia": "WV",
    "wisconsin": "WI",
    "wyoming": "WY",
}


VALID_STATE_ABBREVIATIONS = set(
    STATE_NAME_TO_ABBREVIATION.values()
)


PARTY_NORMALIZATION_MAP = {
    "dem": "D",
    "democrat": "D",
    "democratic": "D",
    "d": "D",
    "rep": "R",
    "republican": "R",
    "gop": "R",
    "r": "R",
    "ind": "I",
    "independent": "I",
    "i": "I",
    "lib": "L",
    "libertarian": "L",
    "l": "L",
    "green": "G",
    "g": "G",
    "constitution": "C",
    "c": "C",
    "other": "O",
    "nonpartisan": "O",
    "no party": "O",
    "npa": "O",
}


TRUE_VALUES = {
    "1",
    "true",
    "t",
    "yes",
    "y",
}

FALSE_VALUES = {
    "0",
    "false",
    "f",
    "no",
    "n",
}


def clean_string(series: pd.Series) -> pd.Series:
    result = (
        series.astype("string")
        .str.strip()
        .replace(
            {
                "": pd.NA,
                "nan": pd.NA,
                "NaN": pd.NA,
                "None": pd.NA,
                "null": pd.NA,
            }
        )
    )

    return result


def normalize_spaces(series: pd.Series) -> pd.Series:
    return clean_string(series).str.replace(
        r"\s+",
        " ",
        regex=True,
    )


def parse_date(series: pd.Series) -> pd.Series:
    text = clean_string(series)

    result = pd.Series(
        pd.NaT,
        index=series.index,
        dtype="datetime64[ns]",
    )

    formats = (
        "%m/%d/%y",
        "%m/%d/%Y",
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
    )

    for date_format in formats:
        missing = result.isna() & text.notna()

        if not missing.any():
            break

        parsed = pd.to_datetime(
            text.loc[missing],
            format=date_format,
            errors="coerce",
            utc=True,
        )

        result.loc[missing] = (
            parsed.dt.tz_convert(None)
        )

    missing = result.isna() & text.notna()

    if missing.any():
        parsed = pd.to_datetime(
            text.loc[missing],
            errors="coerce",
            utc=True,
        )

        result.loc[missing] = (
            parsed.dt.tz_convert(None)
        )

    return result


def normalize_state(value: Any) -> Any:
    if pd.isna(value):
        return pd.NA

    text = re.sub(
        r"\s+",
        " ",
        str(value).strip(),
    )

    uppercase = text.upper()

    if uppercase in VALID_STATE_ABBREVIATIONS:
        return uppercase

    return STATE_NAME_TO_ABBREVIATION.get(
        text.lower(),
        pd.NA,
    )


def normalize_party(value: Any) -> Any:
    if pd.isna(value):
        return pd.NA

    text = re.sub(
        r"\s+",
        " ",
        str(value).strip(),
    )

    normalized = PARTY_NORMALIZATION_MAP.get(
        text.lower()
    )

    if normalized is not None:
        return normalized

    uppercase = text.upper()

    if len(uppercase) <= 3:
        return uppercase

    return "O"


def parse_boolean(series: pd.Series) -> pd.Series:
    text = clean_string(series).str.lower()

    output = pd.Series(
        pd.NA,
        index=series.index,
        dtype="boolean",
    )

    output.loc[text.isin(TRUE_VALUES)] = True
    output.loc[text.isin(FALSE_VALUES)] = False

    return output


def safe_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series,
        errors="coerce",
    ).astype("Int64")


def safe_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series,
        errors="coerce",
    ).astype("Float64")


def hash_source_row(row: pd.Series) -> str:
    payload = "\x1f".join(
        "" if pd.isna(value) else str(value)
        for value in row
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def add_optional_normalized_string(
    output: pd.DataFrame,
    source: pd.DataFrame,
    source_column: str,
    destination_column: str,
) -> None:
    if source_column in source.columns:
        output[destination_column] = (
            normalize_spaces(source[source_column])
        )
    else:
        output[destination_column] = pd.NA


def add_optional_numeric(
    output: pd.DataFrame,
    source: pd.DataFrame,
    source_column: str,
    destination_column: str,
    integer: bool = False,
) -> None:
    if source_column in source.columns:
        if integer:
            output[destination_column] = safe_int(
                source[source_column]
            )
        else:
            output[destination_column] = safe_float(
                source[source_column]
            )
    else:
        output[destination_column] = pd.NA


def add_optional_boolean(
    output: pd.DataFrame,
    source: pd.DataFrame,
    source_column: str,
    destination_column: str,
) -> None:
    if source_column in source.columns:
        output[destination_column] = parse_boolean(
            source[source_column]
        )
    else:
        output[destination_column] = pd.Series(
            pd.NA,
            index=source.index,
            dtype="boolean",
        )


def json_safe(value: Any) -> Any:
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


def main() -> int:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input archive not found: {INPUT_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 92)
    print("BUILD SENATE HISTORICAL POLLING WAREHOUSE")
    print("=" * 92)
    print(f"Input: {INPUT_PATH}")

    raw = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    missing_required = sorted(
        REQUIRED_COLUMNS - set(raw.columns)
    )

    if missing_required:
        raise RuntimeError(
            "Missing required columns: "
            + ", ".join(missing_required)
        )

    print(
        f"Loaded {len(raw):,} rows and "
        f"{len(raw.columns):,} columns."
    )

    output = pd.DataFrame(
        index=raw.index
    )

    # ------------------------------------------------------------------
    # Source and provenance fields
    # ------------------------------------------------------------------

    output["source_name"] = "FiveThirtyEight"
    output["source_dataset"] = (
        "senate_polls_historical"
    )
    output["source_file"] = INPUT_PATH.name
    output["source_row_number"] = (
        pd.Series(
            np.arange(2, len(raw) + 2),
            index=raw.index,
            dtype="Int64",
        )
    )

    output["source_row_sha256"] = raw.apply(
        hash_source_row,
        axis=1,
    )

    # ------------------------------------------------------------------
    # Canonical identifiers
    # ------------------------------------------------------------------

    output["poll_id"] = safe_int(
        raw["poll_id"]
    )
    output["question_id"] = safe_int(
        raw["question_id"]
    )
    output["race_id"] = safe_int(
        raw["race_id"]
    )
    output["candidate_id"] = safe_int(
        raw["candidate_id"]
    )
    output["cycle"] = safe_int(
        raw["cycle"]
    )

    add_optional_numeric(
        output,
        raw,
        "pollster_id",
        "pollster_id",
        integer=True,
    )

    add_optional_numeric(
        output,
        raw,
        "pollster_rating_id",
        "pollster_rating_id",
        integer=True,
    )

    # ------------------------------------------------------------------
    # Race identity
    # ------------------------------------------------------------------

    output["state_raw"] = clean_string(
        raw["state"]
    )

    output["state"] = (
        output["state_raw"]
        .map(normalize_state)
        .astype("string")
    )

    add_optional_normalized_string(
        output,
        raw,
        "office_type",
        "office_type",
    )

    add_optional_normalized_string(
        output,
        raw,
        "seat_number",
        "seat_number",
    )

    add_optional_normalized_string(
        output,
        raw,
        "seat_name",
        "seat_name",
    )

    add_optional_normalized_string(
        output,
        raw,
        "stage",
        "stage",
    )

    # ------------------------------------------------------------------
    # Dates
    # ------------------------------------------------------------------

    output["start_date"] = parse_date(
        raw["start_date"]
    )

    output["end_date"] = parse_date(
        raw["end_date"]
    )

    if "election_date" in raw.columns:
        output["election_date"] = parse_date(
            raw["election_date"]
        )
    else:
        output["election_date"] = pd.NaT

    if "created_at" in raw.columns:
        output["created_at"] = parse_date(
            raw["created_at"]
        )
    else:
        output["created_at"] = pd.NaT

    output["as_of_date"] = (
        output["end_date"]
    )

    output["field_period_days"] = (
        output["end_date"]
        - output["start_date"]
    ).dt.days.astype("Int64")

    output["days_before_election"] = (
        output["election_date"]
        - output["end_date"]
    ).dt.days.astype("Int64")

    # ------------------------------------------------------------------
    # Poll metadata
    # ------------------------------------------------------------------

    for source_column, destination_column in (
        ("pollster", "pollster"),
        ("sponsors", "sponsors"),
        ("display_name", "display_name"),
        ("pollster_name", "pollster_name"),
        ("url", "poll_url"),
        ("source", "poll_source"),
        ("population", "population"),
        ("population_full", "population_full"),
        ("methodology", "methodology"),
        ("notes", "notes"),
    ):
        add_optional_normalized_string(
            output,
            raw,
            source_column,
            destination_column,
        )

    for source_column, destination_column in (
        ("sample_size", "sample_size"),
        ("numeric_grade", "numeric_grade"),
        ("pollscore", "pollscore"),
        (
            "transparency_score",
            "transparency_score",
        ),
    ):
        add_optional_numeric(
            output,
            raw,
            source_column,
            destination_column,
        )

    # ------------------------------------------------------------------
    # Candidate-answer information
    # ------------------------------------------------------------------

    output["candidate_name"] = (
        normalize_spaces(
            raw["candidate_name"]
        )
    )

    if "answer" in raw.columns:
        output["answer"] = normalize_spaces(
            raw["answer"]
        )
    else:
        output["answer"] = (
            output["candidate_name"]
        )

    output["party_raw"] = clean_string(
        raw["party"]
    )

    output["party"] = (
        output["party_raw"]
        .map(normalize_party)
        .astype("string")
    )

    output["pct"] = safe_float(
        raw["pct"]
    )

    output["is_democratic"] = (
        output["party"] == "D"
    ).astype("boolean")

    output["is_republican"] = (
        output["party"] == "R"
    ).astype("boolean")

    output["is_major_party"] = (
        output["party"].isin(["D", "R"])
    ).astype("boolean")

    # ------------------------------------------------------------------
    # Poll-type flags
    # ------------------------------------------------------------------

    for source_column, destination_column in (
        ("tracking", "is_tracking"),
        ("internal", "is_internal"),
        ("partisan", "is_partisan"),
        ("hypothetical", "is_hypothetical"),
        (
            "nationwide_batch",
            "is_nationwide_batch",
        ),
        (
            "ranked_choice_reallocated",
            "is_ranked_choice_reallocated",
        ),
    ):
        add_optional_boolean(
            output,
            raw,
            source_column,
            destination_column,
        )

    add_optional_numeric(
        output,
        raw,
        "ranked_choice_round",
        "ranked_choice_round",
        integer=True,
    )

    add_optional_normalized_string(
        output,
        raw,
        "sponsor_candidate",
        "sponsor_candidate",
    )

    add_optional_normalized_string(
        output,
        raw,
        "sponsor_candidate_party",
        "sponsor_candidate_party_raw",
    )

    output["sponsor_candidate_party"] = (
        output["sponsor_candidate_party_raw"]
        .map(normalize_party)
        .astype("string")
    )

    add_optional_normalized_string(
        output,
        raw,
        "endorsed_candidate",
        "endorsed_candidate",
    )

    add_optional_normalized_string(
        output,
        raw,
        "endorsed_candidate_party",
        "endorsed_candidate_party_raw",
    )

    output["endorsed_candidate_party"] = (
        output["endorsed_candidate_party_raw"]
        .map(normalize_party)
        .astype("string")
    )

    # ------------------------------------------------------------------
    # Structural and validation flags
    # ------------------------------------------------------------------

    output["has_valid_identifiers"] = (
        output[
            [
                "poll_id",
                "question_id",
                "race_id",
                "candidate_id",
            ]
        ]
        .notna()
        .all(axis=1)
    ).astype("boolean")

    output["has_valid_dates"] = (
        output[
            [
                "start_date",
                "end_date",
            ]
        ]
        .notna()
        .all(axis=1)
    ).astype("boolean")

    output["has_valid_state"] = (
        output["state"].isin(
            VALID_STATE_ABBREVIATIONS
        )
    ).astype("boolean")

    output["has_valid_percentage"] = (
        output["pct"].between(
            0,
            100,
            inclusive="both",
        )
    ).astype("boolean")

    output["field_period_is_nonnegative"] = (
        output["field_period_days"] >= 0
    ).astype("boolean")

    output["poll_ends_by_election_day"] = (
        output["end_date"]
        <= output["election_date"]
    ).astype("boolean")

    output["is_ranked_choice_row"] = (
        output["ranked_choice_round"].notna()
        | output[
            "is_ranked_choice_reallocated"
        ].fillna(False)
    ).astype("boolean")

    output["is_standard_general_election"] = (
        output["stage"]
        .fillna("")
        .str.lower()
        .eq("general")
        & ~output[
            "is_hypothetical"
        ].fillna(False)
        & ~output[
            "is_ranked_choice_row"
        ].fillna(False)
    ).astype("boolean")

    # Create stable race and question keys.
    output["race_key"] = (
        output["cycle"].astype("string")
        + "-"
        + output["state"].fillna("UNK")
        + "-"
        + output["race_id"].astype("string")
    )

    output["question_key"] = (
        output["poll_id"].astype("string")
        + "-"
        + output["question_id"].astype("string")
    )

    # Question-level candidate counts and percentage totals.
    question_stats = (
        output.groupby(
            "question_id",
            dropna=False,
        )
        .agg(
            question_candidate_rows=(
                "candidate_id",
                "size",
            ),
            question_unique_candidates=(
                "candidate_id",
                "nunique",
            ),
            question_major_party_rows=(
                "is_major_party",
                "sum",
            ),
            question_pct_sum=(
                "pct",
                "sum",
            ),
        )
        .reset_index()
    )

    output = output.merge(
        question_stats,
        on="question_id",
        how="left",
        validate="many_to_one",
    )

    output[
        "question_has_both_major_parties"
    ] = (
        output.groupby(
            "question_id",
            dropna=False,
        )["party"]
        .transform(
            lambda series: (
                {"D", "R"}
                .issubset(
                    set(
                        series.dropna()
                    )
                )
            )
        )
        .astype("boolean")
    )

    output["question_pct_sum_near_100"] = (
        output["question_pct_sum"].between(
            90,
            105,
            inclusive="both",
        )
    ).astype("boolean")

    # Preserve all remaining raw columns that are not already represented.
    represented_raw_columns = {
        "poll_id",
        "question_id",
        "race_id",
        "candidate_id",
        "cycle",
        "state",
        "start_date",
        "end_date",
        "election_date",
        "created_at",
        "pollster_id",
        "pollster_rating_id",
        "office_type",
        "seat_number",
        "seat_name",
        "stage",
        "pollster",
        "sponsors",
        "display_name",
        "pollster_name",
        "url",
        "source",
        "population",
        "population_full",
        "methodology",
        "notes",
        "sample_size",
        "numeric_grade",
        "pollscore",
        "transparency_score",
        "candidate_name",
        "answer",
        "party",
        "pct",
        "tracking",
        "internal",
        "partisan",
        "hypothetical",
        "nationwide_batch",
        "ranked_choice_reallocated",
        "ranked_choice_round",
        "sponsor_candidate",
        "sponsor_candidate_party",
        "endorsed_candidate",
        "endorsed_candidate_party",
    }

    for column in raw.columns:
        if column in represented_raw_columns:
            continue

        destination = f"source_{column}"

        if destination in output.columns:
            continue

        if raw[column].dtype == "object":
            output[destination] = clean_string(
                raw[column]
            )
        else:
            output[destination] = raw[column]

    # ------------------------------------------------------------------
    # Stable ordering
    # ------------------------------------------------------------------

    sort_columns = [
        "cycle",
        "state",
        "election_date",
        "race_id",
        "end_date",
        "poll_id",
        "question_id",
        "party",
        "candidate_id",
    ]

    output = output.sort_values(
        sort_columns,
        na_position="last",
    ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    duplicate_key_mask = output.duplicated(
        subset=[
            "question_id",
            "candidate_id",
        ],
        keep=False,
    )

    exact_duplicate_mask = output.duplicated(
        keep=False,
    )

    validation_flags = {
        "invalid_identifiers": (
            ~output[
                "has_valid_identifiers"
            ].fillna(False)
        ),
        "invalid_dates": (
            ~output[
                "has_valid_dates"
            ].fillna(False)
        ),
        "invalid_state": (
            ~output[
                "has_valid_state"
            ].fillna(False)
        ),
        "invalid_percentage": (
            ~output[
                "has_valid_percentage"
            ].fillna(False)
        ),
        "negative_field_period": (
            ~output[
                "field_period_is_nonnegative"
            ].fillna(False)
        ),
        "duplicate_question_candidate_key": (
            duplicate_key_mask
        ),
        "exact_duplicate_row": (
            exact_duplicate_mask
        ),
        "missing_election_date": (
            output["election_date"].isna()
        ),
        "poll_after_election_day": (
            output["election_date"].notna()
            & ~output[
                "poll_ends_by_election_day"
            ].fillna(False)
        ),
    }

    exception_mask = pd.Series(
        False,
        index=output.index,
    )

    for flag_name, mask in validation_flags.items():
        output[f"validation_{flag_name}"] = (
            mask.astype("boolean")
        )

        exception_mask |= mask.fillna(False)

    exception_columns = [
        "cycle",
        "state",
        "race_id",
        "poll_id",
        "question_id",
        "candidate_id",
        "candidate_name",
        "party",
        "start_date",
        "end_date",
        "election_date",
        "pct",
    ] + [
        f"validation_{name}"
        for name in validation_flags
    ]

    exception_rows = output.loc[
        exception_mask,
        exception_columns,
    ].copy()

    # Write dates in stable ISO form.
    date_columns = [
        "start_date",
        "end_date",
        "election_date",
        "created_at",
        "as_of_date",
    ]

    output_for_csv = output.copy()

    for column in date_columns:
        output_for_csv[column] = (
            output_for_csv[column]
            .dt.strftime("%Y-%m-%d")
        )

    output_for_csv.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    exception_rows_for_csv = (
        exception_rows.copy()
    )

    for column in (
        "start_date",
        "end_date",
        "election_date",
    ):
        exception_rows_for_csv[column] = (
            exception_rows_for_csv[column]
            .dt.strftime("%Y-%m-%d")
        )

    exception_rows_for_csv.to_csv(
        EXCEPTION_ROWS_CSV,
        index=False,
    )

    # ------------------------------------------------------------------
    # Profiles
    # ------------------------------------------------------------------

    cycle_profile = (
        output.groupby(
            "cycle",
            dropna=False,
        )
        .agg(
            rows=("question_id", "size"),
            polls=("poll_id", "nunique"),
            questions=(
                "question_id",
                "nunique",
            ),
            races=("race_id", "nunique"),
            states=("state", "nunique"),
            candidates=(
                "candidate_id",
                "nunique",
            ),
            earliest_end_date=(
                "end_date",
                "min",
            ),
            latest_end_date=(
                "end_date",
                "max",
            ),
            ranked_choice_rows=(
                "is_ranked_choice_row",
                "sum",
            ),
            hypothetical_rows=(
                "is_hypothetical",
                "sum",
            ),
            internal_rows=(
                "is_internal",
                "sum",
            ),
            partisan_rows=(
                "is_partisan",
                "sum",
            ),
        )
        .reset_index()
        .sort_values("cycle")
    )

    party_profile = (
        output.groupby(
            [
                "party_raw",
                "party",
            ],
            dropna=False,
        )
        .agg(
            rows=("question_id", "size"),
            candidates=(
                "candidate_id",
                "nunique",
            ),
            questions=(
                "question_id",
                "nunique",
            ),
        )
        .reset_index()
        .sort_values(
            "rows",
            ascending=False,
        )
    )

    state_profile = (
        output.groupby(
            [
                "cycle",
                "state",
            ],
            dropna=False,
        )
        .agg(
            rows=("question_id", "size"),
            polls=("poll_id", "nunique"),
            questions=(
                "question_id",
                "nunique",
            ),
            races=("race_id", "nunique"),
            earliest_end_date=(
                "end_date",
                "min",
            ),
            latest_end_date=(
                "end_date",
                "max",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "cycle",
                "state",
            ]
        )
    )

    question_profile = (
        output.groupby(
            "question_id",
            dropna=False,
        )
        .agg(
            poll_id=("poll_id", "first"),
            cycle=("cycle", "first"),
            state=("state", "first"),
            race_id=("race_id", "first"),
            end_date=("end_date", "first"),
            candidate_rows=(
                "candidate_id",
                "size",
            ),
            unique_candidates=(
                "candidate_id",
                "nunique",
            ),
            major_party_rows=(
                "is_major_party",
                "sum",
            ),
            pct_sum=("pct", "sum"),
            has_both_major_parties=(
                "question_has_both_major_parties",
                "first",
            ),
            is_ranked_choice=(
                "is_ranked_choice_row",
                "max",
            ),
            is_hypothetical=(
                "is_hypothetical",
                "max",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "cycle",
                "state",
                "end_date",
                "poll_id",
                "question_id",
            ]
        )
    )

    flag_records = []

    for flag_name, mask in validation_flags.items():
        flag_records.append(
            {
                "flag": flag_name,
                "row_count": int(
                    mask.fillna(False).sum()
                ),
                "share": float(
                    mask.fillna(False).mean()
                ),
            }
        )

    for column in (
        "is_major_party",
        "is_ranked_choice_row",
        "is_hypothetical",
        "is_internal",
        "is_partisan",
        "is_tracking",
        "is_standard_general_election",
        "question_has_both_major_parties",
        "question_pct_sum_near_100",
    ):
        flag_records.append(
            {
                "flag": column,
                "row_count": int(
                    output[column]
                    .fillna(False)
                    .sum()
                ),
                "share": float(
                    output[column]
                    .fillna(False)
                    .mean()
                ),
            }
        )

    flag_profile = pd.DataFrame(
        flag_records
    )

    cycle_profile.to_csv(
        CYCLE_PROFILE_CSV,
        index=False,
    )

    party_profile.to_csv(
        PARTY_PROFILE_CSV,
        index=False,
    )

    state_profile.to_csv(
        STATE_PROFILE_CSV,
        index=False,
    )

    question_profile.to_csv(
        QUESTION_PROFILE_CSV,
        index=False,
    )

    flag_profile.to_csv(
        FLAG_PROFILE_CSV,
        index=False,
    )

    summary: dict[str, Any] = {
        "input_path": str(INPUT_PATH),
        "output_path": str(OUTPUT_PATH),
        "input_rows": int(len(raw)),
        "output_rows": int(len(output)),
        "input_columns": int(len(raw.columns)),
        "output_columns": int(
            len(output.columns)
        ),
        "unique_poll_ids": int(
            output["poll_id"].nunique(
                dropna=True
            )
        ),
        "unique_question_ids": int(
            output["question_id"].nunique(
                dropna=True
            )
        ),
        "unique_race_ids": int(
            output["race_id"].nunique(
                dropna=True
            )
        ),
        "unique_candidate_ids": int(
            output["candidate_id"].nunique(
                dropna=True
            )
        ),
        "duplicate_question_candidate_rows": int(
            duplicate_key_mask.sum()
        ),
        "exact_duplicate_rows": int(
            exact_duplicate_mask.sum()
        ),
        "missing_election_dates": int(
            output["election_date"]
            .isna()
            .sum()
        ),
        "ranked_choice_rows": int(
            output["is_ranked_choice_row"]
            .fillna(False)
            .sum()
        ),
        "major_party_rows": int(
            output["is_major_party"]
            .fillna(False)
            .sum()
        ),
        "questions_with_both_major_parties": int(
            question_profile[
                "has_both_major_parties"
            ]
            .fillna(False)
            .sum()
        ),
        "question_count": int(
            len(question_profile)
        ),
        "validation_counts": {
            name: int(
                mask.fillna(False).sum()
            )
            for name, mask in (
                validation_flags.items()
            )
        },
        "cycles": sorted(
            output["cycle"]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        ),
    }

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

    blocking_failures = {
        "row_count_changed": (
            len(output) != len(raw)
        ),
        "duplicate_question_candidate_rows": (
            duplicate_key_mask.any()
        ),
        "exact_duplicate_rows": (
            exact_duplicate_mask.any()
        ),
        "invalid_identifiers": (
            validation_flags[
                "invalid_identifiers"
            ].any()
        ),
        "invalid_dates": (
            validation_flags[
                "invalid_dates"
            ].any()
        ),
        "invalid_percentage": (
            validation_flags[
                "invalid_percentage"
            ].any()
        ),
        "negative_field_period": (
            validation_flags[
                "negative_field_period"
            ].any()
        ),
    }

    passed = not any(
        blocking_failures.values()
    )

    lines = [
        "=" * 92,
        "SENATE HISTORICAL POLLING WAREHOUSE VALIDATION",
        "=" * 92,
        "",
        f"Input rows:                         {len(raw):,}",
        f"Output rows:                        {len(output):,}",
        (
            "Unique poll IDs:                    "
            f"{summary['unique_poll_ids']:,}"
        ),
        (
            "Unique question IDs:                "
            f"{summary['unique_question_ids']:,}"
        ),
        (
            "Unique race IDs:                    "
            f"{summary['unique_race_ids']:,}"
        ),
        (
            "Unique candidate IDs:               "
            f"{summary['unique_candidate_ids']:,}"
        ),
        (
            "Duplicate question-candidate rows:   "
            f"{summary['duplicate_question_candidate_rows']:,}"
        ),
        (
            "Exact duplicate rows:                "
            f"{summary['exact_duplicate_rows']:,}"
        ),
        (
            "Missing election dates:              "
            f"{summary['missing_election_dates']:,}"
        ),
        (
            "Ranked-choice rows retained:          "
            f"{summary['ranked_choice_rows']:,}"
        ),
        (
            "Major-party rows:                    "
            f"{summary['major_party_rows']:,}"
        ),
        "",
        "BLOCKING VALIDATION",
        "-" * 92,
    ]

    for name, failed in blocking_failures.items():
        lines.append(
            f"{name}: "
            f"{'FAILED' if failed else 'PASSED'}"
        )

    lines.extend(
        [
            "",
            (
                "Processed warehouse validation:   "
                f"{'PASSED' if passed else 'FAILED'}"
            ),
            "",
            "Outputs:",
            f"  Warehouse: {OUTPUT_PATH}",
            f"  Cycle profile: {CYCLE_PROFILE_CSV}",
            f"  Party profile: {PARTY_PROFILE_CSV}",
            f"  State profile: {STATE_PROFILE_CSV}",
            (
                "  Question profile: "
                f"{QUESTION_PROFILE_CSV}"
            ),
            f"  Flag profile: {FLAG_PROFILE_CSV}",
            (
                "  Exception rows: "
                f"{EXCEPTION_ROWS_CSV}"
            ),
            f"  JSON summary: {SUMMARY_JSON}",
            "",
        ]
    )

    text_summary = "\n".join(lines)

    SUMMARY_TXT.write_text(
        text_summary,
        encoding="utf-8",
    )

    print()
    print(text_summary)

    if not passed:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
