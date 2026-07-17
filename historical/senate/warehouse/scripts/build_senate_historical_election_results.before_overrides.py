#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]

RAW_PATH = (
    PROJECT_ROOT
    / "historical"
    / "senate"
    / "warehouse"
    / "raw"
    / "election_results"
    / "1976-2024-senate-state.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "historical"
    / "senate"
    / "warehouse"
    / "processed"
    / "senate_historical_election_results_2012_2024.csv"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "historical"
    / "senate"
    / "warehouse"
    / "validation"
    / "senate_historical_election_results_2012_2024_validation.csv"
)

NONSTANDARD_PATH = (
    PROJECT_ROOT
    / "historical"
    / "senate"
    / "warehouse"
    / "validation"
    / "senate_historical_nonstandard_races_2012_2024.csv"
)

CYCLES = [
    2012,
    2014,
    2016,
    2018,
    2020,
    2022,
    2024,
]

ELECTION_DATES = {
    2012: "2012-11-06",
    2014: "2014-11-04",
    2016: "2016-11-08",
    2018: "2018-11-06",
    2020: "2020-11-03",
    2022: "2022-11-08",
    2024: "2024-11-05",
}

VALID_STATE_ABBREVIATIONS = {
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


def clean_text(series: pd.Series) -> pd.Series:
    return (
        series
        .astype("string")
        .str.strip()
        .replace(
            {
                "": pd.NA,
                "nan": pd.NA,
                "None": pd.NA,
                "NA": pd.NA,
            }
        )
    )


def normalize_party(value: object) -> str:
    if pd.isna(value):
        return "Other"

    party = str(value).strip().upper()

    if party in {
        "DEMOCRAT",
        "DEMOCRATIC",
        "DEM",
        "D",
        "DFL",
    }:
        return "D"

    if party in {
        "REPUBLICAN",
        "GOP",
        "REP",
        "R",
    }:
        return "R"

    if party in {
        "INDEPENDENT",
        "IND",
        "I",
    }:
        return "I"

    return "Other"


def candidate_totals(
    race: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate all ballot lines belonging to the same named candidate.

    This matters in fusion-voting states such as New York, where one candidate
    can appear under multiple ballot-party labels.
    """
    grouped = (
        race.groupby(
            "candidate_clean",
            dropna=False,
            as_index=False,
        )
        .agg(
            candidatevotes=("candidatevotes", "sum"),
            has_dem_line=(
                "party_standard",
                lambda values: (values == "D").any(),
            ),
            has_gop_line=(
                "party_standard",
                lambda values: (values == "R").any(),
            ),
            has_ind_line=(
                "party_standard",
                lambda values: (values == "I").any(),
            ),
        )
    )

    return grouped.sort_values(
        "candidatevotes",
        ascending=False,
    ).reset_index(drop=True)


def select_major_party_candidate(
    totals: pd.DataFrame,
    party: str,
) -> tuple[object, float]:
    flag = {
        "D": "has_dem_line",
        "R": "has_gop_line",
    }[party]

    candidates = totals.loc[
        totals[flag]
    ].copy()

    if candidates.empty:
        return pd.NA, 0.0

    winner = candidates.iloc[0]

    return (
        winner["candidate_clean"],
        float(winner["candidatevotes"]),
    )


def build_race_id(
    cycle: int,
    state: str,
    special: bool,
    races_in_state_cycle: int,
) -> str:
    if races_in_state_cycle == 1:
        return f"{cycle}_{state}"

    suffix = "SPECIAL" if special else "REGULAR"
    return f"{cycle}_{state}_{suffix}"


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            "Raw MIT Senate file not found:\n"
            f"{RAW_PATH}"
        )

    raw = pd.read_csv(
        RAW_PATH,
        low_memory=False,
    )

    required_columns = {
        "year",
        "state",
        "state_po",
        "office",
        "district",
        "stage",
        "special",
        "candidate",
        "party_simplified",
        "writein",
        "mode",
        "candidatevotes",
        "totalvotes",
        "unofficial",
    }

    missing_columns = sorted(
        required_columns - set(raw.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required MIT columns: "
            + ", ".join(missing_columns)
        )

    print("Raw rows:", len(raw))
    print("Raw years:", raw["year"].min(), "-", raw["year"].max())

    data = raw.copy()

    data["year"] = pd.to_numeric(
        data["year"],
        errors="coerce",
    ).astype("Int64")

    data = data.loc[
        data["year"].isin(CYCLES)
    ].copy()

    data = data.loc[
        clean_text(data["office"])
        .str.upper()
        .eq("US SENATE")
    ].copy()

    data = data.loc[
        clean_text(data["stage"])
        .str.lower()
        .eq("gen")
    ].copy()

    data = data.loc[
        clean_text(data["mode"])
        .str.lower()
        .eq("total")
    ].copy()

    if data.empty:
        raise ValueError(
            "No general-election Senate rows remained "
            "after filtering."
        )

    data["state_po"] = (
        clean_text(data["state_po"])
        .str.upper()
    )

    invalid_states = sorted(
        set(data["state_po"].dropna())
        - VALID_STATE_ABBREVIATIONS
    )

    if invalid_states:
        raise ValueError(
            "Unexpected state abbreviations: "
            + ", ".join(invalid_states)
        )

    data["state_name"] = (
        clean_text(data["state"])
        .str.title()
    )

    data["candidate_clean"] = (
        clean_text(data["candidate"])
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
    )

    data["party_standard"] = (
        data["party_simplified"]
        .map(normalize_party)
    )

    data["candidatevotes"] = pd.to_numeric(
        data["candidatevotes"],
        errors="coerce",
    ).fillna(0.0)

    data["totalvotes"] = pd.to_numeric(
        data["totalvotes"],
        errors="coerce",
    )

    data["special"] = (
        data["special"]
        .fillna(False)
        .astype(bool)
    )

    data["unofficial"] = (
        data["unofficial"]
        .fillna(False)
        .astype(bool)
    )

    race_key = [
        "year",
        "state_po",
        "special",
    ]

    race_counts = (
        data[race_key]
        .drop_duplicates()
        .groupby(
            ["year", "state_po"]
        )
        .size()
        .to_dict()
    )

    output_rows: list[dict[str, object]] = []

    for key, race in data.groupby(
        race_key,
        sort=True,
        dropna=False,
    ):
        cycle, state, special = key

        cycle = int(cycle)
        state = str(state)
        special = bool(special)

        races_in_state_cycle = int(
            race_counts[(cycle, state)]
        )

        totals = candidate_totals(race)

        dem_candidate, dem_votes = (
            select_major_party_candidate(
                totals,
                "D",
            )
        )

        gop_candidate, gop_votes = (
            select_major_party_candidate(
                totals,
                "R",
            )
        )

        candidate_vote_sum = float(
            totals["candidatevotes"].sum()
        )

        reported_totals = (
            race["totalvotes"]
            .dropna()
            .unique()
        )

        if len(reported_totals):
            total_votes = float(
                np.max(reported_totals)
            )
        else:
            total_votes = candidate_vote_sum

        # Protect against malformed totals.
        total_votes = max(
            total_votes,
            candidate_vote_sum,
        )

        other_votes = max(
            total_votes
            - dem_votes
            - gop_votes,
            0.0,
        )

        two_party_votes = (
            dem_votes
            + gop_votes
        )

        if total_votes > 0:
            dem_vote_share = (
                100.0
                * dem_votes
                / total_votes
            )
            gop_vote_share = (
                100.0
                * gop_votes
                / total_votes
            )
        else:
            dem_vote_share = np.nan
            gop_vote_share = np.nan

        if two_party_votes > 0:
            dem_two_party_share = (
                100.0
                * dem_votes
                / two_party_votes
            )
            gop_two_party_share = (
                100.0
                * gop_votes
                / two_party_votes
            )
            actual_margin_dem = (
                dem_two_party_share
                - gop_two_party_share
            )
        else:
            dem_two_party_share = np.nan
            gop_two_party_share = np.nan
            actual_margin_dem = np.nan

        major_party_contested = bool(
            dem_votes > 0
            and gop_votes > 0
        )

        if dem_votes > gop_votes:
            winner_party = "D"
        elif gop_votes > dem_votes:
            winner_party = "R"
        else:
            winner_party = pd.NA

        unofficial = bool(
            race["unofficial"].any()
        )

        race_id = build_race_id(
            cycle=cycle,
            state=state,
            special=special,
            races_in_state_cycle=(
                races_in_state_cycle
            ),
        )

        output_rows.append(
            {
                "race_id": race_id,
                "cycle": cycle,
                "election_date": (
                    ELECTION_DATES[cycle]
                ),
                "state": state,
                "state_name": (
                    race["state_name"]
                    .dropna()
                    .iloc[0]
                ),
                "senate_class": pd.NA,
                "election_type": (
                    "special"
                    if special
                    else "regular"
                ),
                "special_election": special,
                "dem_candidate": dem_candidate,
                "gop_candidate": gop_candidate,
                "incumbent_name": pd.NA,
                "incumbent_party": pd.NA,
                "incumbent_running": pd.NA,
                "open_seat": pd.NA,
                "dem_votes": int(
                    round(dem_votes)
                ),
                "gop_votes": int(
                    round(gop_votes)
                ),
                "other_votes": int(
                    round(other_votes)
                ),
                "total_votes": int(
                    round(total_votes)
                ),
                "dem_vote_share": (
                    dem_vote_share
                ),
                "gop_vote_share": (
                    gop_vote_share
                ),
                "dem_two_party_share": (
                    dem_two_party_share
                ),
                "gop_two_party_share": (
                    gop_two_party_share
                ),
                "actual_margin_dem": (
                    actual_margin_dem
                ),
                "winner_party": (
                    winner_party
                ),
                "major_party_contested": (
                    major_party_contested
                ),
                "source": (
                    "MIT Election Data and "
                    "Science Lab"
                ),
                "source_url": (
                    "1976-2024-senate-state.csv"
                ),
                "source_status": (
                    "unofficial"
                    if unofficial
                    else "official"
                ),
                "notes": (
                    "MIT row marked unofficial."
                    if unofficial
                    else pd.NA
                ),
            }
        )

    output = pd.DataFrame(
        output_rows
    ).sort_values(
        [
            "cycle",
            "state",
            "special_election",
        ]
    ).reset_index(drop=True)

    if output["race_id"].duplicated().any():
        duplicate_rows = output.loc[
            output["race_id"].duplicated(
                keep=False
            )
        ]

        raise ValueError(
            "Duplicate race IDs detected:\n"
            + duplicate_rows[
                [
                    "race_id",
                    "cycle",
                    "state",
                    "election_type",
                ]
            ].to_string(index=False)
        )

    output["two_party_sum"] = (
        output["dem_two_party_share"]
        + output["gop_two_party_share"]
    )

    output["calculated_margin"] = (
        output["dem_two_party_share"]
        - output["gop_two_party_share"]
    )

    output["two_party_sum_error"] = (
        output["two_party_sum"]
        - 100.0
    ).abs()

    output["margin_error"] = (
        output["calculated_margin"]
        - output["actual_margin_dem"]
    ).abs()

    validation_rows = []

    for cycle in CYCLES:
        cycle_data = output.loc[
            output["cycle"] == cycle
        ]

        validation_rows.append(
            {
                "cycle": cycle,
                "race_rows": len(
                    cycle_data
                ),
                "unique_states": (
                    cycle_data[
                        "state"
                    ].nunique()
                ),
                "regular_elections": int(
                    (
                        ~cycle_data[
                            "special_election"
                        ]
                    ).sum()
                ),
                "special_elections": int(
                    cycle_data[
                        "special_election"
                    ].sum()
                ),
                "major_party_contested": int(
                    cycle_data[
                        "major_party_contested"
                    ].sum()
                ),
                "nonstandard_races": int(
                    (
                        ~cycle_data[
                            "major_party_contested"
                        ]
                    ).sum()
                ),
                "missing_dem_candidate": int(
                    cycle_data[
                        "dem_candidate"
                    ].isna().sum()
                ),
                "missing_gop_candidate": int(
                    cycle_data[
                        "gop_candidate"
                    ].isna().sum()
                ),
                "missing_margin": int(
                    cycle_data[
                        "actual_margin_dem"
                    ].isna().sum()
                ),
                "duplicate_race_ids": int(
                    cycle_data[
                        "race_id"
                    ].duplicated().sum()
                ),
                "two_party_sum_failures": int(
                    (
                        cycle_data[
                            "two_party_sum_error"
                        ]
                        > 1e-8
                    ).sum()
                ),
                "margin_failures": int(
                    (
                        cycle_data[
                            "margin_error"
                        ]
                        > 1e-8
                    ).sum()
                ),
                "unofficial_races": int(
                    (
                        cycle_data[
                            "source_status"
                        ]
                        == "unofficial"
                    ).sum()
                ),
            }
        )

    validation = pd.DataFrame(
        validation_rows
    )

    final_columns = [
        "race_id",
        "cycle",
        "election_date",
        "state",
        "state_name",
        "senate_class",
        "election_type",
        "special_election",
        "dem_candidate",
        "gop_candidate",
        "incumbent_name",
        "incumbent_party",
        "incumbent_running",
        "open_seat",
        "dem_votes",
        "gop_votes",
        "other_votes",
        "total_votes",
        "dem_vote_share",
        "gop_vote_share",
        "dem_two_party_share",
        "gop_two_party_share",
        "actual_margin_dem",
        "winner_party",
        "major_party_contested",
        "source",
        "source_url",
        "source_status",
        "notes",
    ]

    nonstandard = output.loc[
        ~output["major_party_contested"],
        [
            "race_id",
            "cycle",
            "state",
            "election_type",
            "dem_candidate",
            "gop_candidate",
            "dem_votes",
            "gop_votes",
            "winner_party",
            "source_status",
        ],
    ].copy()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    VALIDATION_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output[
        final_columns
    ].to_csv(
        OUTPUT_PATH,
        index=False,
    )

    validation.to_csv(
        VALIDATION_PATH,
        index=False,
    )

    nonstandard.to_csv(
        NONSTANDARD_PATH,
        index=False,
    )

    print()
    print(
        "Processed Senate races:",
        len(output),
    )

    print()
    print("Rows by cycle:")
    print(
        output.groupby("cycle")
        .size()
        .to_string()
    )

    print()
    print("Validation:")
    print(
        validation.to_string(
            index=False
        )
    )

    print()
    print("Nonstandard races:")
    print(
        nonstandard.to_string(
            index=False
        )
        if not nonstandard.empty
        else "None"
    )

    print()
    print("Wrote:", OUTPUT_PATH)
    print("Wrote:", VALIDATION_PATH)
    print("Wrote:", NONSTANDARD_PATH)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        raise
