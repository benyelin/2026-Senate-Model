#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]

WAREHOUSE_ROOT = (
    PROJECT_ROOT
    / "historical"
    / "senate"
    / "warehouse"
)

RAW_PATH = (
    WAREHOUSE_ROOT
    / "raw"
    / "election_results"
    / "1976-2024-senate-state.csv"
)

OVERRIDE_ROOT = (
    WAREHOUSE_ROOT
    / "raw"
    / "overrides"
)

ALIGNMENT_PATH = (
    OVERRIDE_ROOT
    / "senate_candidate_alignment_overrides.csv"
)

SCORABLE_PATH = (
    OVERRIDE_ROOT
    / "senate_backtest_scorable_overrides.csv"
)

METADATA_PATH = (
    OVERRIDE_ROOT
    / "senate_race_metadata_overrides.csv"
)

OUTPUT_PATH = (
    WAREHOUSE_ROOT
    / "processed"
    / "senate_historical_election_results_2012_2024.csv"
)

VALIDATION_PATH = (
    WAREHOUSE_ROOT
    / "validation"
    / "senate_historical_election_results_2012_2024_validation.csv"
)

NONSTANDARD_PATH = (
    WAREHOUSE_ROOT
    / "validation"
    / "senate_historical_nonstandard_races_2012_2024.csv"
)

OVERRIDE_AUDIT_PATH = (
    WAREHOUSE_ROOT
    / "validation"
    / "senate_historical_override_audit_2012_2024.csv"
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
        series.astype("string")
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


def parse_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    text = str(value).strip().lower()

    if text in {"true", "t", "1", "yes", "y"}:
        return True

    if text in {"false", "f", "0", "no", "n"}:
        return False

    raise ValueError(
        f"Could not parse Boolean value: {value!r}"
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


def load_csv(
    path: Path,
    required_columns: set[str],
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    frame = pd.read_csv(
        path,
        dtype="string",
        keep_default_na=False,
    )

    missing = sorted(
        required_columns - set(frame.columns)
    )

    if missing:
        raise ValueError(
            f"{path.name} is missing columns: "
            + ", ".join(missing)
        )

    return frame


def build_race_id(
    cycle: int,
    state: str,
    special: bool,
    races_in_state_cycle: int,
) -> str:
    if races_in_state_cycle == 1:
        return f"{cycle}_{state}"

    suffix = (
        "SPECIAL"
        if special
        else "REGULAR"
    )

    return f"{cycle}_{state}_{suffix}"


def candidate_totals(
    race: pd.DataFrame,
) -> pd.DataFrame:
    return (
        race.groupby(
            "candidate_clean",
            dropna=False,
            as_index=False,
        )
        .agg(
            candidatevotes=(
                "candidatevotes",
                "sum",
            ),
            has_dem_line=(
                "party_standard",
                lambda values: (
                    values == "D"
                ).any(),
            ),
            has_gop_line=(
                "party_standard",
                lambda values: (
                    values == "R"
                ).any(),
            ),
            has_ind_line=(
                "party_standard",
                lambda values: (
                    values == "I"
                ).any(),
            ),
        )
        .sort_values(
            "candidatevotes",
            ascending=False,
        )
        .reset_index(drop=True)
    )


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
    ]

    if candidates.empty:
        return pd.NA, 0.0

    winner = candidates.iloc[0]

    return (
        winner["candidate_clean"],
        float(winner["candidatevotes"]),
    )


def assign_race_ids(
    data: pd.DataFrame,
) -> pd.DataFrame:
    data = data.copy()

    race_keys = (
        data[
            ["year", "state_po", "special"]
        ]
        .drop_duplicates()
    )

    race_counts = (
        race_keys.groupby(
            ["year", "state_po"]
        )
        .size()
        .to_dict()
    )

    race_keys["race_id"] = race_keys.apply(
        lambda row: build_race_id(
            cycle=int(row["year"]),
            state=str(row["state_po"]),
            special=bool(row["special"]),
            races_in_state_cycle=int(
                race_counts[
                    (
                        int(row["year"]),
                        str(row["state_po"]),
                    )
                ]
            ),
        ),
        axis=1,
    )

    return data.merge(
        race_keys,
        on=["year", "state_po", "special"],
        how="left",
        validate="many_to_one",
    )


def apply_alignment_overrides(
    data: pd.DataFrame,
    alignment: pd.DataFrame,
    audit_rows: list[dict[str, object]],
) -> pd.DataFrame:
    data = data.copy()

    duplicate_keys = alignment.duplicated(
        subset=["race_id", "candidate"],
        keep=False,
    )

    if duplicate_keys.any():
        raise ValueError(
            "Duplicate candidate-alignment overrides:\n"
            + alignment.loc[
                duplicate_keys
            ].to_string(index=False)
        )

    valid_parties = {"D", "R", "I", "Other"}

    for row in alignment.itertuples(
        index=False
    ):
        race_id = str(row.race_id).strip()
        candidate = str(row.candidate).strip()
        aligned_party = (
            str(row.aligned_party)
            .strip()
        )

        if aligned_party not in valid_parties:
            raise ValueError(
                f"Invalid aligned party "
                f"{aligned_party!r} for "
                f"{race_id} / {candidate}"
            )

        race_exists = (
            data["race_id"] == race_id
        ).any()

        if not race_exists:
            raise ValueError(
                "Alignment override references "
                f"unknown race: {race_id}"
            )

        mask = (
            (data["race_id"] == race_id)
            & (
                data["candidate_clean"]
                == candidate
            )
        )

        matched_rows = int(mask.sum())

        if matched_rows == 0:
            available = (
                data.loc[
                    data["race_id"] == race_id,
                    "candidate_clean",
                ]
                .dropna()
                .drop_duplicates()
                .tolist()
            )

            raise ValueError(
                "Alignment override candidate "
                f"not found: {race_id} / "
                f"{candidate}. Available: "
                f"{available}"
            )

        old_parties = sorted(
            data.loc[
                mask,
                "party_standard",
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        data.loc[
            mask,
            "party_standard",
        ] = aligned_party

        audit_rows.append(
            {
                "override_file": (
                    ALIGNMENT_PATH.name
                ),
                "race_id": race_id,
                "candidate": candidate,
                "field": "party_standard",
                "old_value": "|".join(
                    old_parties
                ),
                "new_value": aligned_party,
                "matched_rows": matched_rows,
                "notes": str(row.notes),
            }
        )

    return data


def apply_metadata_overrides(
    output: pd.DataFrame,
    metadata: pd.DataFrame,
    audit_rows: list[dict[str, object]],
) -> pd.DataFrame:
    output = output.copy()

    duplicate_keys = metadata.duplicated(
        subset=["race_id", "field"],
        keep=False,
    )

    if duplicate_keys.any():
        raise ValueError(
            "Duplicate metadata overrides:\n"
            + metadata.loc[
                duplicate_keys
            ].to_string(index=False)
        )

    for row in metadata.itertuples(
        index=False
    ):
        race_id = str(row.race_id).strip()
        field = str(row.field).strip()
        new_value = str(row.new_value).strip()

        mask = (
            output["race_id"] == race_id
        )

        if not mask.any():
            raise ValueError(
                "Metadata override references "
                f"unknown race: {race_id}"
            )

        if field not in output.columns:
            output[field] = pd.NA

        old_value = output.loc[
            mask,
            field,
        ].iloc[0]

        if field in {
            "manual_review",
            "backtest_scorable",
        }:
            parsed_value: object = (
                parse_bool(new_value)
            )
        else:
            parsed_value = new_value

        output.loc[
            mask,
            field,
        ] = parsed_value

        audit_rows.append(
            {
                "override_file": (
                    METADATA_PATH.name
                ),
                "race_id": race_id,
                "candidate": "",
                "field": field,
                "old_value": old_value,
                "new_value": parsed_value,
                "matched_rows": 1,
                "notes": str(row.reason),
            }
        )

    return output


def apply_scorable_overrides(
    output: pd.DataFrame,
    scorable: pd.DataFrame,
    audit_rows: list[dict[str, object]],
) -> pd.DataFrame:
    output = output.copy()

    duplicate_ids = scorable.duplicated(
        subset=["race_id"],
        keep=False,
    )

    if duplicate_ids.any():
        raise ValueError(
            "Duplicate backtest-scorable "
            "overrides:\n"
            + scorable.loc[
                duplicate_ids
            ].to_string(index=False)
        )

    for row in scorable.itertuples(
        index=False
    ):
        race_id = str(row.race_id).strip()

        mask = (
            output["race_id"] == race_id
        )

        if not mask.any():
            raise ValueError(
                "Scorable override references "
                f"unknown race: {race_id}"
            )

        old_value = bool(
            output.loc[
                mask,
                "backtest_scorable",
            ].iloc[0]
        )

        new_value = parse_bool(
            row.backtest_scorable
        )

        output.loc[
            mask,
            "backtest_scorable",
        ] = new_value

        output.loc[
            mask,
            "backtest_exclusion_reason",
        ] = (
            ""
            if new_value
            else str(row.reason)
        )

        audit_rows.append(
            {
                "override_file": (
                    SCORABLE_PATH.name
                ),
                "race_id": race_id,
                "candidate": "",
                "field": "backtest_scorable",
                "old_value": old_value,
                "new_value": new_value,
                "matched_rows": 1,
                "notes": str(row.reason),
            }
        )

    return output


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            "Raw MIT Senate file not found:\n"
            f"{RAW_PATH}"
        )

    alignment = load_csv(
        ALIGNMENT_PATH,
        {
            "race_id",
            "candidate",
            "aligned_party",
            "notes",
        },
    )

    scorable = load_csv(
        SCORABLE_PATH,
        {
            "race_id",
            "backtest_scorable",
            "reason",
        },
    )

    metadata = load_csv(
        METADATA_PATH,
        {
            "race_id",
            "field",
            "new_value",
            "reason",
        },
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
        "party_detailed",
        "party_simplified",
        "writein",
        "mode",
        "candidatevotes",
        "totalvotes",
        "unofficial",
    }

    missing_columns = sorted(
        required_columns
        - set(raw.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required MIT columns: "
            + ", ".join(missing_columns)
        )

    print("Raw rows:", len(raw))
    print(
        "Raw years:",
        raw["year"].min(),
        "-",
        raw["year"].max(),
    )

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
            "No general-election Senate rows "
            "remained after filtering."
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

    simplified_party = (
        data["party_simplified"]
        .map(normalize_party)
    )

    detailed_party = (
        data["party_detailed"]
        .map(normalize_party)
    )

    data["party_standard"] = (
        simplified_party.where(
            simplified_party != "Other",
            detailed_party,
        )
    )

    data["candidatevotes"] = (
        pd.to_numeric(
            data["candidatevotes"],
            errors="coerce",
        )
        .fillna(0.0)
    )

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

    data = assign_race_ids(data)

    audit_rows: list[
        dict[str, object]
    ] = []

    data = apply_alignment_overrides(
        data,
        alignment,
        audit_rows,
    )

    output_rows: list[
        dict[str, object]
    ] = []

    for race_id, race in data.groupby(
        "race_id",
        sort=True,
        dropna=False,
    ):
        cycle = int(
            race["year"].iloc[0]
        )

        state = str(
            race["state_po"].iloc[0]
        )

        special = bool(
            race["special"].iloc[0]
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

        total_votes = (
            float(np.max(reported_totals))
            if len(reported_totals)
            else candidate_vote_sum
        )

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

        dem_vote_share = (
            100.0 * dem_votes / total_votes
            if total_votes > 0
            else np.nan
        )

        gop_vote_share = (
            100.0 * gop_votes / total_votes
            if total_votes > 0
            else np.nan
        )

        dem_two_party_share = (
            100.0
            * dem_votes
            / two_party_votes
            if two_party_votes > 0
            else np.nan
        )

        gop_two_party_share = (
            100.0
            * gop_votes
            / two_party_votes
            if two_party_votes > 0
            else np.nan
        )

        actual_margin_dem = (
            dem_two_party_share
            - gop_two_party_share
            if two_party_votes > 0
            else np.nan
        )

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
                "backtest_scorable": (
                    major_party_contested
                ),
                "backtest_exclusion_reason": (
                    ""
                    if major_party_contested
                    else (
                        "No aligned Democratic-"
                        "versus-Republican contest"
                    )
                ),
                "manual_review": False,
                "source": (
                    "MIT Election Data and "
                    "Science Lab"
                ),
                "source_url": (
                    "1976-2024-senate-state.csv"
                ),
                "source_status": (
                    "unofficial"
                    if race[
                        "unofficial"
                    ].any()
                    else "official"
                ),
                "notes": pd.NA,
            }
        )

    output = (
        pd.DataFrame(output_rows)
        .sort_values(
            [
                "cycle",
                "state",
                "special_election",
            ]
        )
        .reset_index(drop=True)
    )

    if output["race_id"].duplicated().any():
        raise ValueError(
            "Duplicate race IDs detected."
        )

    output = apply_metadata_overrides(
        output,
        metadata,
        audit_rows,
    )

    output = apply_scorable_overrides(
        output,
        scorable,
        audit_rows,
    )

    invalid_scorable = output.loc[
        output["backtest_scorable"]
        & (
            output[
                "actual_margin_dem"
            ].isna()
        )
    ]

    if not invalid_scorable.empty:
        raise ValueError(
            "Scorable races have missing margins:\n"
            + invalid_scorable[
                [
                    "race_id",
                    "dem_candidate",
                    "gop_candidate",
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
                "backtest_scorable": int(
                    cycle_data[
                        "backtest_scorable"
                    ].sum()
                ),
                "excluded_from_backtest": int(
                    (
                        ~cycle_data[
                            "backtest_scorable"
                        ]
                    ).sum()
                ),
                "manual_review": int(
                    cycle_data[
                        "manual_review"
                    ].sum()
                ),
                "missing_margin": int(
                    cycle_data[
                        "actual_margin_dem"
                    ].isna().sum()
                ),
                "scorable_missing_margin": int(
                    (
                        cycle_data[
                            "backtest_scorable"
                        ]
                        & cycle_data[
                            "actual_margin_dem"
                        ].isna()
                    ).sum()
                ),
                "duplicate_race_ids": int(
                    cycle_data[
                        "race_id"
                    ].duplicated().sum()
                ),
                "two_party_sum_failures": int(
                    (
                        cycle_data[
                            "backtest_scorable"
                        ]
                        & (
                            cycle_data[
                                "two_party_sum_error"
                            ]
                            > 1e-8
                        )
                    ).sum()
                ),
                "margin_failures": int(
                    (
                        cycle_data[
                            "backtest_scorable"
                        ]
                        & (
                            cycle_data[
                                "margin_error"
                            ]
                            > 1e-8
                        )
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

    nonstandard = output.loc[
        ~output["backtest_scorable"],
        [
            "race_id",
            "cycle",
            "state",
            "election_type",
            "dem_candidate",
            "gop_candidate",
            "dem_votes",
            "gop_votes",
            "actual_margin_dem",
            "winner_party",
            "major_party_contested",
            "manual_review",
            "backtest_exclusion_reason",
            "source_status",
        ],
    ].copy()

    audit = pd.DataFrame(
        audit_rows
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
        "backtest_scorable",
        "backtest_exclusion_reason",
        "manual_review",
        "source",
        "source_url",
        "source_status",
        "notes",
    ]

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

    audit.to_csv(
        OVERRIDE_AUDIT_PATH,
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
    print(
        "Excluded from backtesting:"
    )
    print(
        nonstandard.to_string(
            index=False
        )
        if not nonstandard.empty
        else "None"
    )

    print()
    print("Override audit:")
    print(
        audit.to_string(
            index=False
        )
        if not audit.empty
        else "None"
    )

    print()
    print("Wrote:", OUTPUT_PATH)
    print("Wrote:", VALIDATION_PATH)
    print("Wrote:", NONSTANDARD_PATH)
    print("Wrote:", OVERRIDE_AUDIT_PATH)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        raise
