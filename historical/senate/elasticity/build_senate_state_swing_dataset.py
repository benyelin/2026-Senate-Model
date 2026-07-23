#!/usr/bin/env python3
"""
Build leakage-free historical Senate seat swing observations.

This script adapts the proven House district-swing architecture to the
institutional structure of the U.S. Senate.

A Senate observation compares the same physical Senate seat across its
regular six-year election interval:

    state + Senate class, cycle t -> cycle t + 6

Examples:

    AZ_CLASS_1: 2012 -> 2018 -> 2024
    PA_CLASS_3: 2016 -> 2022

The builder deliberately excludes special elections from Version 1.
Special elections may involve appointments, shortened terms, irregular
timing, and electorates that are not directly comparable to regular
six-year Senate elections.

The output is a reusable observation warehouse. It does not estimate
elasticity and does not use future information when constructing any
individual transition.

Usage
-----
From the repository root:

    python3 historical/senate/elasticity/build_senate_state_swing_dataset.py

Optional paths:

    python3 historical/senate/elasticity/build_senate_state_swing_dataset.py \
        --input historical/senate/warehouse/processed/senate_historical_fundamentals_2012_2024.csv \
        --output historical/senate/warehouse/processed/senate_state_swing_observations.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "historical/senate/warehouse/processed"
    / "senate_historical_fundamentals_2012_2024.csv"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "historical/senate/warehouse/processed"
    / "senate_state_swing_observations.csv"
)

DEFAULT_VALIDATION = (
    PROJECT_ROOT
    / "historical/senate/warehouse/validation"
    / "senate_state_swing_observations_validation.txt"
)

DEFAULT_REJECTED_AUDIT = (
    PROJECT_ROOT
    / "historical/senate/warehouse/validation"
    / "senate_state_swing_rejected_transitions.csv"
)

EXPECTED_CYCLE_GAP = 6

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "race_id": (
        "race_id",
    ),
    "cycle": (
        "cycle",
        "forecast_cycle",
        "election_cycle",
    ),
    "state": (
        "state",
        "state_po",
        "state_abbreviation",
    ),
    "senate_class": (
        "senate_class",
        "seat_class",
        "class",
    ),
    "seat_id": (
        "seat_id",
        "senate_seat_id",
    ),
    "actual_margin_dem": (
        "actual_margin_dem",
        "dem_margin",
        "dem_margin_actual",
        "actual_dem_margin",
    ),
    "generic_ballot_margin_dem": (
        "generic_ballot_margin_dem",
        "national_environment_margin_dem",
        "election_day_generic_ballot_margin_dem",
    ),
    "backtest_scorable": (
        "backtest_scorable",
        "scorable",
        "is_scorable",
    ),
    "special_election": (
        "special_election",
        "is_special_election",
        "special",
    ),
    "election_type": (
        "election_type",
        "race_type",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build leakage-free regular-election Senate seat swing "
            "observations across six-year same-seat transitions."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Canonical historical Senate fundamentals warehouse.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV for reusable Senate swing observations.",
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=DEFAULT_VALIDATION,
        help="Human-readable validation report.",
    )
    parser.add_argument(
        "--rejected-audit-output",
        type=Path,
        default=DEFAULT_REJECTED_AUDIT,
        help=(
            "Audit CSV for adjacent same-seat transitions rejected "
            "because they are not exactly six years apart."
        ),
    )
    return parser.parse_args()


def find_column(
    frame: pd.DataFrame,
    logical_name: str,
    *,
    required: bool = True,
) -> str | None:
    """Return the first matching source column for a logical field."""
    aliases = COLUMN_ALIASES.get(logical_name, (logical_name,))
    lower_to_original = {
        str(column).strip().lower(): str(column)
        for column in frame.columns
    }

    for alias in aliases:
        match = lower_to_original.get(alias.lower())
        if match is not None:
            return match

    if required:
        raise ValueError(
            f"Could not find required logical column {logical_name!r}. "
            f"Accepted aliases: {list(aliases)}. "
            f"Available columns: {list(frame.columns)}"
        )

    return None


def normalize_boolean(series: pd.Series) -> pd.Series:
    """Normalize common CSV boolean encodings without silently accepting junk."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    normalized = series.astype("string").str.strip().str.lower()

    true_values = {
        "true",
        "t",
        "yes",
        "y",
        "1",
        "1.0",
    }
    false_values = {
        "false",
        "f",
        "no",
        "n",
        "0",
        "0.0",
        "",
        "nan",
        "none",
        "<na>",
    }

    unknown = (
        normalized.notna()
        & ~normalized.isin(true_values)
        & ~normalized.isin(false_values)
    )
    if unknown.any():
        values = sorted(normalized.loc[unknown].dropna().unique().tolist())
        raise ValueError(
            "Unrecognized boolean values: "
            + json.dumps(values[:20])
        )

    return normalized.isin(true_values)


def canonicalize_state(series: pd.Series) -> pd.Series:
    states = series.astype("string").str.strip().str.upper()

    missing = states.isna() | states.eq("")
    if missing.any():
        raise ValueError(
            f"Missing state values in {int(missing.sum())} input rows."
        )

    return states


def derive_seat_id(
    state: pd.Series,
    senate_class: pd.Series,
) -> pd.Series:
    """Create a stable physical-seat identifier when one is not supplied."""
    class_as_int = senate_class.astype("Int64").astype("string")
    return state + "_CLASS_" + class_as_int


def load_canonical_input(
    input_path: Path,
) -> tuple[pd.DataFrame, dict[str, str]]:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Canonical Senate warehouse not found: {input_path}"
        )

    raw = pd.read_csv(input_path)

    if raw.empty:
        raise ValueError(f"Canonical Senate warehouse is empty: {input_path}")

    required_logical_columns = (
        "race_id",
        "cycle",
        "state",
        "senate_class",
        "actual_margin_dem",
        "generic_ballot_margin_dem",
    )

    column_map: dict[str, str] = {
        logical_name: find_column(raw, logical_name, required=True)
        for logical_name in required_logical_columns
    }

    optional_logical_columns = (
        "seat_id",
        "backtest_scorable",
        "special_election",
        "election_type",
    )

    for logical_name in optional_logical_columns:
        source = find_column(raw, logical_name, required=False)
        if source is not None:
            column_map[logical_name] = source

    data = pd.DataFrame(
        {
            logical_name: raw[source_column]
            for logical_name, source_column in column_map.items()
        }
    )

    data["race_id"] = data["race_id"].astype("string").str.strip()
    data["state"] = canonicalize_state(data["state"])

    numeric_columns = (
        "cycle",
        "senate_class",
        "actual_margin_dem",
        "generic_ballot_margin_dem",
    )
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if "backtest_scorable" in data.columns:
        data["backtest_scorable"] = normalize_boolean(
            data["backtest_scorable"]
        )
    else:
        data["backtest_scorable"] = True

    if "special_election" in data.columns:
        data["special_election"] = normalize_boolean(
            data["special_election"]
        )
    elif "election_type" in data.columns:
        election_type = (
            data["election_type"]
            .astype("string")
            .str.strip()
            .str.lower()
        )
        data["special_election"] = election_type.eq("special")
    else:
        raise ValueError(
            "The input must contain either special_election or election_type "
            "so regular and special elections can be separated safely."
        )

    # Race outcomes may legitimately be missing on rows already excluded
    # from historical backtesting. Require structural identifiers and the
    # cycle-level national environment here; require actual margins only
    # after selecting scorable regular elections.
    required_nonmissing = (
        "race_id",
        "cycle",
        "state",
        "senate_class",
        "generic_ballot_margin_dem",
    )
    missing_counts = {
        column: int(data[column].isna().sum())
        for column in required_nonmissing
        if data[column].isna().any()
    }
    if missing_counts:
        raise ValueError(
            "Missing required values in canonical warehouse: "
            + json.dumps(missing_counts, sort_keys=True)
        )

    noninteger_cycle = ~np.isclose(
        data["cycle"],
        np.round(data["cycle"]),
    )
    if noninteger_cycle.any():
        raise ValueError("Non-integer election cycles found.")

    noninteger_class = ~np.isclose(
        data["senate_class"],
        np.round(data["senate_class"]),
    )
    if noninteger_class.any():
        raise ValueError("Non-integer Senate classes found.")

    data["cycle"] = data["cycle"].astype(int)
    data["senate_class"] = data["senate_class"].astype(int)

    invalid_class = ~data["senate_class"].isin([1, 2, 3])
    if invalid_class.any():
        invalid_values = sorted(
            data.loc[invalid_class, "senate_class"].unique().tolist()
        )
        raise ValueError(
            f"Invalid Senate classes found: {invalid_values}"
        )

    derived_seat_id = derive_seat_id(
        data["state"],
        data["senate_class"],
    )

    if "seat_id" in data.columns:
        supplied = data["seat_id"].astype("string").str.strip().str.upper()
        missing_supplied = supplied.isna() | supplied.eq("")
        supplied = supplied.mask(missing_supplied, derived_seat_id)

        disagreement = supplied.ne(derived_seat_id)
        if disagreement.any():
            examples = (
                pd.DataFrame(
                    {
                        "race_id": data.loc[disagreement, "race_id"],
                        "supplied_seat_id": supplied.loc[disagreement],
                        "derived_seat_id": derived_seat_id.loc[disagreement],
                    }
                )
                .head(20)
                .to_dict("records")
            )
            raise ValueError(
                "Supplied seat_id disagrees with state and Senate class: "
                + json.dumps(examples)
            )

        data["seat_id"] = supplied
    else:
        data["seat_id"] = derived_seat_id

    duplicate_races = data.duplicated(
        subset=["race_id", "cycle"],
        keep=False,
    )
    if duplicate_races.any():
        examples = (
            data.loc[
                duplicate_races,
                ["race_id", "cycle", "state", "senate_class"],
            ]
            .sort_values(["cycle", "race_id"])
            .head(20)
            .to_dict("records")
        )
        raise ValueError(
            "Duplicate race-cycle rows found: "
            + json.dumps(examples)
        )

    # The generic ballot is a required cycle-level input for every row.
    # Actual margins are validated after eligibility filtering because
    # excluded/nonstandard races may intentionally lack a usable D-R margin.
    generic_values = data[
        "generic_ballot_margin_dem"
    ].to_numpy(dtype=float)
    if not np.isfinite(generic_values).all():
        raise ValueError(
            "Non-finite values found in generic_ballot_margin_dem."
        )

    invalid_margin = (
        data["actual_margin_dem"].notna()
        & data["actual_margin_dem"].abs().gt(100.0 + 1e-9)
    )
    if invalid_margin.any():
        examples = (
            data.loc[
                invalid_margin,
                ["race_id", "actual_margin_dem"],
            ]
            .head(20)
            .to_dict("records")
        )
        raise ValueError(
            "Actual Democratic margins outside [-100, 100]: "
            + json.dumps(examples)
        )

    return data, column_map


def validate_cycle_environment(
    races: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ensure every election cycle has exactly one national-environment value.

    The canonical race-level warehouse repeats the generic-ballot value for
    every race in a cycle. This validation prevents silently using internally
    inconsistent cycle values.
    """
    cycle_environment = (
        races.groupby("cycle", as_index=False)
        .agg(
            generic_ballot_min=(
                "generic_ballot_margin_dem",
                "min",
            ),
            generic_ballot_max=(
                "generic_ballot_margin_dem",
                "max",
            ),
            generic_ballot_margin_dem=(
                "generic_ballot_margin_dem",
                "first",
            ),
        )
        .sort_values("cycle")
        .reset_index(drop=True)
    )

    spread = (
        cycle_environment["generic_ballot_max"]
        - cycle_environment["generic_ballot_min"]
    )
    inconsistent = spread.abs().gt(1e-9)

    if inconsistent.any():
        examples = (
            cycle_environment.loc[
                inconsistent,
                [
                    "cycle",
                    "generic_ballot_min",
                    "generic_ballot_max",
                ],
            ]
            .to_dict("records")
        )
        raise ValueError(
            "Generic-ballot margin is not constant within cycle: "
            + json.dumps(examples)
        )

    return cycle_environment[
        ["cycle", "generic_ballot_margin_dem"]
    ].copy()


def select_eligible_regular_races(
    canonical: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    exclusion_counts = {
        "input_rows": int(len(canonical)),
        "not_backtest_scorable": int(
            (~canonical["backtest_scorable"]).sum()
        ),
        "special_elections": int(
            canonical["special_election"].sum()
        ),
    }

    eligible = canonical.loc[
        canonical["backtest_scorable"]
        & ~canonical["special_election"]
    ].copy()

    if eligible.empty:
        raise ValueError(
            "No scorable regular Senate elections remain after filtering."
        )

    # Every race that can enter a swing observation must have a complete,
    # finite election result and national-environment value.
    eligible_required = (
        "actual_margin_dem",
        "generic_ballot_margin_dem",
    )
    eligible_missing = {
        column: int(eligible[column].isna().sum())
        for column in eligible_required
        if eligible[column].isna().any()
    }
    if eligible_missing:
        examples = (
            eligible.loc[
                eligible[list(eligible_required)].isna().any(axis=1),
                [
                    "race_id",
                    "cycle",
                    "state",
                    "senate_class",
                    "actual_margin_dem",
                    "generic_ballot_margin_dem",
                ],
            ]
            .head(20)
            .to_dict("records")
        )
        raise ValueError(
            "Eligible regular races have missing required values: "
            + json.dumps(
                {
                    "counts": eligible_missing,
                    "examples": examples,
                },
                sort_keys=True,
            )
        )

    for column in eligible_required:
        values = eligible[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(
                f"Eligible regular races contain non-finite {column} values."
            )

    duplicate_seat_cycles = eligible.duplicated(
        subset=["seat_id", "cycle"],
        keep=False,
    )
    if duplicate_seat_cycles.any():
        examples = (
            eligible.loc[
                duplicate_seat_cycles,
                [
                    "seat_id",
                    "cycle",
                    "race_id",
                    "special_election",
                ],
            ]
            .sort_values(["seat_id", "cycle", "race_id"])
            .head(20)
            .to_dict("records")
        )
        raise ValueError(
            "More than one eligible regular race exists for a seat-cycle: "
            + json.dumps(examples)
        )

    eligible = eligible.sort_values(
        ["seat_id", "cycle", "race_id"]
    ).reset_index(drop=True)

    exclusion_counts["eligible_regular_races"] = int(len(eligible))

    return eligible, exclusion_counts


def build_swing_observations(
    eligible: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construct same-seat observations using only six-year regular transitions.

    Adjacent observed elections for the same seat are inspected. A pair is
    accepted only when current_cycle - previous_cycle == 6. Larger gaps are
    preserved in a rejected-transition audit rather than treated as a single
    elasticity observation.
    """
    observation_rows: list[dict[str, object]] = []
    rejected_rows: list[dict[str, object]] = []

    for seat_id, group in eligible.groupby("seat_id", sort=True):
        group = group.sort_values("cycle").reset_index(drop=True)

        if len(group) < 2:
            continue

        for previous_index in range(len(group) - 1):
            previous = group.iloc[previous_index]
            current = group.iloc[previous_index + 1]

            previous_cycle = int(previous["cycle"])
            current_cycle = int(current["cycle"])
            cycle_gap = current_cycle - previous_cycle

            audit_payload = {
                "seat_id": str(seat_id),
                "state": str(current["state"]),
                "senate_class": int(current["senate_class"]),
                "previous_race_id": str(previous["race_id"]),
                "current_race_id": str(current["race_id"]),
                "previous_cycle": previous_cycle,
                "current_cycle": current_cycle,
                "cycle_gap": cycle_gap,
            }

            if cycle_gap != EXPECTED_CYCLE_GAP:
                rejected_rows.append(
                    {
                        **audit_payload,
                        "rejection_reason": (
                            "cycle_gap_not_equal_to_six"
                        ),
                    }
                )
                continue

            previous_margin = float(previous["actual_margin_dem"])
            current_margin = float(current["actual_margin_dem"])

            previous_environment = float(
                previous["generic_ballot_margin_dem"]
            )
            current_environment = float(
                current["generic_ballot_margin_dem"]
            )

            observation_rows.append(
                {
                    **audit_payload,
                    "previous_margin_dem": previous_margin,
                    "current_margin_dem": current_margin,
                    "seat_swing_dem": (
                        current_margin - previous_margin
                    ),
                    "previous_generic_ballot_margin_dem": (
                        previous_environment
                    ),
                    "current_generic_ballot_margin_dem": (
                        current_environment
                    ),
                    "national_swing_dem": (
                        current_environment - previous_environment
                    ),
                }
            )

    observations = pd.DataFrame(observation_rows)
    rejected = pd.DataFrame(rejected_rows)

    if observations.empty:
        raise ValueError(
            "No valid six-year same-seat Senate transitions were produced."
        )

    observations = observations.sort_values(
        ["previous_cycle", "current_cycle", "state", "senate_class"]
    ).reset_index(drop=True)

    if not rejected.empty:
        rejected = rejected.sort_values(
            ["seat_id", "previous_cycle", "current_cycle"]
        ).reset_index(drop=True)

    return observations, rejected


def validate_observations(
    observations: pd.DataFrame,
    eligible: pd.DataFrame,
) -> list[str]:
    required_columns = (
        "seat_id",
        "state",
        "senate_class",
        "previous_race_id",
        "current_race_id",
        "previous_cycle",
        "current_cycle",
        "cycle_gap",
        "previous_margin_dem",
        "current_margin_dem",
        "seat_swing_dem",
        "previous_generic_ballot_margin_dem",
        "current_generic_ballot_margin_dem",
        "national_swing_dem",
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in observations.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Output is missing required columns: {missing_columns}"
        )

    if observations[list(required_columns)].isna().any().any():
        missing = (
            observations[list(required_columns)]
            .isna()
            .sum()
        )
        missing = missing.loc[missing.gt(0)].to_dict()
        raise ValueError(
            "Missing values in swing observations: "
            + json.dumps(missing, sort_keys=True)
        )

    duplicate_transitions = observations.duplicated(
        subset=["seat_id", "previous_cycle", "current_cycle"],
        keep=False,
    )
    if duplicate_transitions.any():
        examples = (
            observations.loc[
                duplicate_transitions,
                ["seat_id", "previous_cycle", "current_cycle"],
            ]
            .head(20)
            .to_dict("records")
        )
        raise ValueError(
            "Duplicate seat transitions found: "
            + json.dumps(examples)
        )

    invalid_gap = observations["cycle_gap"].ne(EXPECTED_CYCLE_GAP)
    if invalid_gap.any():
        raise ValueError(
            "Output contains transitions that are not exactly six years."
        )

    class_changed = (
        observations["previous_race_id"].isna()
        | observations["current_race_id"].isna()
    )
    if class_changed.any():
        raise ValueError("Transition race identifiers may not be missing.")

    seat_swing_recalculated = (
        observations["current_margin_dem"]
        - observations["previous_margin_dem"]
    )
    seat_swing_failure = ~np.isclose(
        observations["seat_swing_dem"],
        seat_swing_recalculated,
        atol=1e-10,
        rtol=0.0,
    )
    if seat_swing_failure.any():
        raise ValueError("Seat-swing arithmetic validation failed.")

    national_swing_recalculated = (
        observations["current_generic_ballot_margin_dem"]
        - observations["previous_generic_ballot_margin_dem"]
    )
    national_swing_failure = ~np.isclose(
        observations["national_swing_dem"],
        national_swing_recalculated,
        atol=1e-10,
        rtol=0.0,
    )
    if national_swing_failure.any():
        raise ValueError("National-swing arithmetic validation failed.")

    numeric_columns = (
        "previous_margin_dem",
        "current_margin_dem",
        "seat_swing_dem",
        "previous_generic_ballot_margin_dem",
        "current_generic_ballot_margin_dem",
        "national_swing_dem",
    )
    for column in numeric_columns:
        if not np.isfinite(
            observations[column].to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"Non-finite values found in output column {column}."
            )

    eligible_lookup = eligible.set_index(["seat_id", "cycle"])

    for row in observations.itertuples(index=False):
        previous_key = (row.seat_id, row.previous_cycle)
        current_key = (row.seat_id, row.current_cycle)

        if previous_key not in eligible_lookup.index:
            raise ValueError(
                f"Previous race missing from eligible input: {previous_key}"
            )
        if current_key not in eligible_lookup.index:
            raise ValueError(
                f"Current race missing from eligible input: {current_key}"
            )

    transition_counts = (
        observations.groupby(
            ["previous_cycle", "current_cycle"],
            as_index=False,
        )
        .size()
        .sort_values(["previous_cycle", "current_cycle"])
    )

    validation_lines = [
        f"Required output columns present: {len(required_columns)}",
        "Missing output values: 0",
        "Duplicate seat transitions: 0",
        f"All cycle gaps equal {EXPECTED_CYCLE_GAP}: yes",
        "Seat-swing arithmetic failures: 0",
        "National-swing arithmetic failures: 0",
        "Non-finite numeric values: 0",
        "",
        "Transition counts:",
    ]

    for row in transition_counts.itertuples(index=False):
        validation_lines.append(
            f"  {int(row.previous_cycle)} -> "
            f"{int(row.current_cycle)}: {int(row.size)}"
        )

    return validation_lines


def describe_series(
    series: pd.Series,
    *,
    decimals: int = 6,
) -> list[str]:
    stats = series.describe()
    return [
        f"  count:  {int(stats['count'])}",
        f"  mean:   {stats['mean']:.{decimals}f}",
        f"  median: {series.median():.{decimals}f}",
        f"  std:    {stats['std']:.{decimals}f}",
        f"  min:    {stats['min']:.{decimals}f}",
        f"  max:    {stats['max']:.{decimals}f}",
    ]


def write_validation_report(
    *,
    input_path: Path,
    output_path: Path,
    validation_path: Path,
    observations: pd.DataFrame,
    eligible: pd.DataFrame,
    rejected: pd.DataFrame,
    exclusion_counts: dict[str, int],
    column_map: dict[str, str],
    validation_lines: Iterable[str],
) -> None:
    transition_counts = (
        observations.groupby(
            ["previous_cycle", "current_cycle"],
            as_index=False,
        )
        .size()
        .sort_values(["previous_cycle", "current_cycle"])
    )

    report = [
        "Senate State Swing Observation Warehouse Validation",
        "=" * 51,
        "",
        f"Input: {input_path}",
        f"Output: {output_path}",
        "",
        "Source column mapping:",
    ]

    report.extend(
        f"  {logical_name}: {source_column}"
        for logical_name, source_column in sorted(column_map.items())
    )

    report.extend(
        [
            "",
            "Input and filtering:",
            f"  Input rows: {exclusion_counts['input_rows']}",
            (
                "  Excluded as not backtest-scorable: "
                f"{exclusion_counts['not_backtest_scorable']}"
            ),
            (
                "  Excluded special elections: "
                f"{exclusion_counts['special_elections']}"
            ),
            (
                "  Eligible regular races: "
                f"{exclusion_counts['eligible_regular_races']}"
            ),
            "",
            "Output:",
            f"  Swing observations: {len(observations)}",
            (
                "  Unique physical Senate seats represented: "
                f"{observations['seat_id'].nunique()}"
            ),
            (
                "  Unique states represented: "
                f"{observations['state'].nunique()}"
            ),
            (
                "  Non-six-year adjacent transitions rejected: "
                f"{len(rejected)}"
            ),
            "",
            "Transitions:",
        ]
    )

    for row in transition_counts.itertuples(index=False):
        report.append(
            f"  {int(row.previous_cycle)} -> "
            f"{int(row.current_cycle)}: {int(row.size)}"
        )

    report.extend(
        [
            "",
            "Seat swing statistics:",
            *describe_series(observations["seat_swing_dem"]),
            "",
            "National swing statistics:",
            *describe_series(observations["national_swing_dem"]),
            "",
            "Validation checks:",
            *validation_lines,
            "",
            "VALIDATION STATUS: PASSED",
            "",
        ]
    )

    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.write_text(
        "\n".join(report),
        encoding="utf-8",
    )


def print_summary(
    *,
    observations: pd.DataFrame,
    eligible: pd.DataFrame,
    rejected: pd.DataFrame,
    output_path: Path,
    validation_path: Path,
) -> None:
    transition_counts = (
        observations.groupby(
            ["previous_cycle", "current_cycle"],
            as_index=False,
        )
        .size()
        .sort_values(["previous_cycle", "current_cycle"])
    )

    print("Senate State Swing Dataset")
    print("=" * 40)
    print(f"Eligible regular races: {len(eligible)}")
    print(f"Swing observations: {len(observations)}")
    print(
        "Unique physical Senate seats: "
        f"{observations['seat_id'].nunique()}"
    )
    print(f"Unique states: {observations['state'].nunique()}")
    print(
        "Rejected non-six-year adjacent transitions: "
        f"{len(rejected)}"
    )
    print("")
    print("Transition counts")
    print("-" * 17)

    for row in transition_counts.itertuples(index=False):
        print(
            f"{int(row.previous_cycle)} -> "
            f"{int(row.current_cycle)}: {int(row.size)}"
        )

    print("")
    print("Seat swing")
    print("-" * 10)
    for line in describe_series(observations["seat_swing_dem"]):
        print(line)

    print("")
    print("National swing")
    print("-" * 14)
    for line in describe_series(observations["national_swing_dem"]):
        print(line)

    print("")
    print("Validation status: PASSED")
    print("")
    print(f"Wrote: {output_path}")
    print(f"Wrote: {validation_path}")


def main() -> None:
    args = parse_args()

    canonical, column_map = load_canonical_input(args.input)

    # Validate the race-level repetition of national environment before
    # filtering or constructing transitions.
    validate_cycle_environment(canonical)

    eligible, exclusion_counts = select_eligible_regular_races(
        canonical
    )

    observations, rejected = build_swing_observations(eligible)

    validation_lines = validate_observations(
        observations,
        eligible,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    observations.to_csv(args.output, index=False)

    args.rejected_audit_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rejected_columns = [
        "seat_id",
        "state",
        "senate_class",
        "previous_race_id",
        "current_race_id",
        "previous_cycle",
        "current_cycle",
        "cycle_gap",
        "rejection_reason",
    ]

    if rejected.empty:
        rejected = pd.DataFrame(columns=rejected_columns)
    else:
        rejected = rejected.reindex(columns=rejected_columns)

    rejected.to_csv(
        args.rejected_audit_output,
        index=False,
    )

    write_validation_report(
        input_path=args.input,
        output_path=args.output,
        validation_path=args.validation_output,
        observations=observations,
        eligible=eligible,
        rejected=rejected,
        exclusion_counts=exclusion_counts,
        column_map=column_map,
        validation_lines=validation_lines,
    )

    print_summary(
        observations=observations,
        eligible=eligible,
        rejected=rejected,
        output_path=args.output,
        validation_path=args.validation_output,
    )

    print(f"Wrote: {args.rejected_audit_output}")


if __name__ == "__main__":
    main()
