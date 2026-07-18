from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]

SOURCE_DIR = (
    PROJECT_ROOT
    / "historical"
    / "senate"
    / "warehouse"
    / "raw"
    / "presidential"
    / "source_downloads"
    / "mit_us_president_1976_2024"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "historical"
    / "senate"
    / "warehouse"
    / "processed"
    / "senate_historical_presidential_results.csv"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "historical"
    / "senate"
    / "warehouse"
    / "validation"
    / "senate_historical_presidential_results_validation.txt"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "historical"
    / "senate"
    / "warehouse"
    / "metadata"
    / "senate_historical_presidential_results_metadata.json"
)

TARGET_YEARS = [
    2008,
    2012,
    2016,
    2020,
    2024,
]

STATE_CODES = {
    "AL", "AK", "AZ", "AR", "CA",
    "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA",
    "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO",
    "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH",
    "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT",
    "VA", "WA", "WV", "WI", "WY",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def locate_source_file() -> Path:
    candidates = sorted(
        path
        for path in SOURCE_DIR.glob("*.csv")
        if path.is_file()
    )

    if not candidates:
        raise FileNotFoundError(
            f"No CSV file found in {SOURCE_DIR}"
        )

    preferred = [
        path
        for path in candidates
        if "1976-2024-president" in path.name.lower()
    ]

    if len(preferred) == 1:
        return preferred[0]

    if len(candidates) == 1:
        return candidates[0]

    raise ValueError(
        "Multiple presidential CSV files found:\n"
        + "\n".join(str(path) for path in candidates)
    )


def main() -> None:
    source_path = locate_source_file()

    raw = pd.read_csv(
        source_path,
        low_memory=False,
    )

    required_columns = {
        "year",
        "state",
        "state_po",
        "office",
        "candidate",
        "party_simplified",
        "candidatevotes",
        "totalvotes",
    }

    missing_columns = sorted(
        required_columns - set(raw.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    work = raw.copy()

    work["year"] = pd.to_numeric(
        work["year"],
        errors="coerce",
    )

    work["candidatevotes"] = pd.to_numeric(
        work["candidatevotes"],
        errors="coerce",
    )

    work["totalvotes"] = pd.to_numeric(
        work["totalvotes"],
        errors="coerce",
    )

    work["state_po"] = (
        work["state_po"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    work["party_simplified"] = (
        work["party_simplified"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    work["office"] = (
        work["office"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    work = work[
        work["year"].isin(TARGET_YEARS)
        & work["state_po"].isin(STATE_CODES)
        & work["office"].eq("US PRESIDENT")
    ].copy()

    major = work[
        work["party_simplified"].isin(
            ["DEMOCRAT", "REPUBLICAN"]
        )
    ].copy()

    major["major_party"] = major[
        "party_simplified"
    ].map(
        {
            "DEMOCRAT": "D",
            "REPUBLICAN": "R",
        }
    )

    # Some states may report multiple ballot lines for the
    # same major-party nominee. Aggregate votes by party.
    party_votes = (
        major
        .groupby(
            [
                "year",
                "state",
                "state_po",
                "major_party",
            ],
            as_index=False,
        )
        .agg(
            candidatevotes=("candidatevotes", "sum"),
            totalvotes=("totalvotes", "max"),
        )
    )

    # Preserve the candidate name from the largest individual
    # ballot line for each state-year-party.
    candidate_names = (
        major
        .sort_values(
            [
                "year",
                "state_po",
                "major_party",
                "candidatevotes",
            ],
            ascending=[
                True,
                True,
                True,
                False,
            ],
        )
        .drop_duplicates(
            [
                "year",
                "state_po",
                "major_party",
            ]
        )[
            [
                "year",
                "state_po",
                "major_party",
                "candidate",
            ]
        ]
    )

    votes_wide = (
        party_votes
        .pivot(
            index=[
                "year",
                "state",
                "state_po",
                "totalvotes",
            ],
            columns="major_party",
            values="candidatevotes",
        )
        .reset_index()
        .rename(
            columns={
                "D": "dem_votes",
                "R": "gop_votes",
            }
        )
    )

    names_wide = (
        candidate_names
        .pivot(
            index=[
                "year",
                "state_po",
            ],
            columns="major_party",
            values="candidate",
        )
        .reset_index()
        .rename(
            columns={
                "D": "dem_candidate",
                "R": "gop_candidate",
            }
        )
    )

    output = votes_wide.merge(
        names_wide,
        on=[
            "year",
            "state_po",
        ],
        how="left",
        validate="one_to_one",
    )

    output["major_party_votes"] = (
        output["dem_votes"]
        + output["gop_votes"]
    )

    output["dem_vote_share"] = (
        100.0
        * output["dem_votes"]
        / output["totalvotes"]
    )

    output["gop_vote_share"] = (
        100.0
        * output["gop_votes"]
        / output["totalvotes"]
    )

    output["dem_two_party_share"] = (
        100.0
        * output["dem_votes"]
        / output["major_party_votes"]
    )

    output["gop_two_party_share"] = (
        100.0
        * output["gop_votes"]
        / output["major_party_votes"]
    )

    output["presidential_margin_dem"] = (
        output["dem_two_party_share"]
        - output["gop_two_party_share"]
    )

    output["winner_party"] = output[
        "presidential_margin_dem"
    ].map(
        lambda margin: (
            "D"
            if margin > 0
            else "R"
            if margin < 0
            else "TIE"
        )
    )

    output["source_dataset"] = (
        "MIT Election Data and Science Lab, "
        "U.S. President 1976-2024"
    )

    output["source_doi"] = (
        "10.7910/DVN/42MVDX"
    )

    output["source_file"] = source_path.name
    output["source_status"] = "authoritative_academic"
    output["validation_status"] = "validated"

    output = output[
        [
            "year",
            "state",
            "state_po",
            "dem_candidate",
            "gop_candidate",
            "dem_votes",
            "gop_votes",
            "major_party_votes",
            "totalvotes",
            "dem_vote_share",
            "gop_vote_share",
            "dem_two_party_share",
            "gop_two_party_share",
            "presidential_margin_dem",
            "winner_party",
            "source_dataset",
            "source_doi",
            "source_file",
            "source_status",
            "validation_status",
        ]
    ].sort_values(
        [
            "year",
            "state_po",
        ]
    ).reset_index(drop=True)

    expected_rows = (
        len(TARGET_YEARS)
        * len(STATE_CODES)
    )

    rows_by_year = (
        output
        .groupby("year")
        .size()
        .reindex(TARGET_YEARS, fill_value=0)
    )

    duplicate_keys = int(
        output.duplicated(
            [
                "year",
                "state_po",
            ]
        ).sum()
    )

    missing_state_years = []

    for year in TARGET_YEARS:
        available_states = set(
            output.loc[
                output["year"].eq(year),
                "state_po",
            ]
        )

        for state_po in sorted(
            STATE_CODES - available_states
        ):
            missing_state_years.append(
                f"{year}_{state_po}"
            )

    missing_dem_votes = int(
        output["dem_votes"].isna().sum()
    )

    missing_gop_votes = int(
        output["gop_votes"].isna().sum()
    )

    missing_margins = int(
        output[
            "presidential_margin_dem"
        ].isna().sum()
    )

    nonpositive_major_party_votes = int(
        (
            output["major_party_votes"] <= 0
        ).sum()
    )

    invalid_two_party_shares = int(
        (
            (output["dem_two_party_share"] < 0)
            | (output["dem_two_party_share"] > 100)
            | (output["gop_two_party_share"] < 0)
            | (output["gop_two_party_share"] > 100)
        ).sum()
    )

    max_share_sum_error = float(
        (
            output["dem_two_party_share"]
            + output["gop_two_party_share"]
            - 100.0
        )
        .abs()
        .max()
    )

    validation_passed = all(
        [
            len(output) == expected_rows,
            duplicate_keys == 0,
            len(missing_state_years) == 0,
            missing_dem_votes == 0,
            missing_gop_votes == 0,
            missing_margins == 0,
            nonpositive_major_party_votes == 0,
            invalid_two_party_shares == 0,
            max_share_sum_error < 1e-8,
        ]
    )

    if not validation_passed:
        raise ValueError(
            "Presidential warehouse validation failed.\n"
            f"Expected rows: {expected_rows}\n"
            f"Output rows: {len(output)}\n"
            f"Duplicate keys: {duplicate_keys}\n"
            f"Missing state-years: {missing_state_years}\n"
            f"Missing Dem votes: {missing_dem_votes}\n"
            f"Missing GOP votes: {missing_gop_votes}\n"
            f"Missing margins: {missing_margins}\n"
            f"Nonpositive major-party votes: "
            f"{nonpositive_major_party_votes}\n"
            f"Invalid two-party shares: "
            f"{invalid_two_party_shares}\n"
            f"Maximum share-sum error: "
            f"{max_share_sum_error}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    VALIDATION_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    METADATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    validation_lines = [
        "Senate Historical Presidential Results Validation",
        "=" * 49,
        "",
        f"Source file: {source_path}",
        f"Source SHA-256: {sha256(source_path)}",
        "Source DOI: 10.7910/DVN/42MVDX",
        "",
        f"Target years: {TARGET_YEARS}",
        f"Expected rows: {expected_rows}",
        f"Output rows: {len(output)}",
        f"Duplicate keys: {duplicate_keys}",
        (
            "Missing state-years: "
            f"{len(missing_state_years)}"
        ),
        f"Missing Democratic votes: {missing_dem_votes}",
        f"Missing Republican votes: {missing_gop_votes}",
        f"Missing margins: {missing_margins}",
        (
            "Nonpositive major-party vote totals: "
            f"{nonpositive_major_party_votes}"
        ),
        (
            "Invalid two-party shares: "
            f"{invalid_two_party_shares}"
        ),
        (
            "Maximum two-party share sum error: "
            f"{max_share_sum_error:.12f}"
        ),
        "",
        "Rows by year:",
        rows_by_year.to_string(),
        "",
        "Validation: PASSED",
    ]

    VALIDATION_PATH.write_text(
        "\n".join(validation_lines)
    )

    metadata = {
        "dataset_name": (
            "senate_historical_presidential_results"
        ),
        "source_name": (
            "MIT Election Data and Science Lab, "
            "U.S. President 1976-2024"
        ),
        "source_doi": "10.7910/DVN/42MVDX",
        "source_file": str(source_path),
        "source_sha256": sha256(source_path),
        "coverage_years": TARGET_YEARS,
        "geographic_level": "state",
        "excluded_geographies": ["District of Columbia"],
        "primary_key": [
            "year",
            "state_po",
        ],
        "row_count": len(output),
        "validation_status": "passed",
    }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    print(
        "\n".join(validation_lines)
    )

    print()
    print(f"Wrote: {OUTPUT_PATH}")
    print(f"Wrote: {VALIDATION_PATH}")
    print(f"Wrote: {METADATA_PATH}")


if __name__ == "__main__":
    main()
