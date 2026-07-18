#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]

WAREHOUSE_ROOT = (
    PROJECT_ROOT
    / "historical"
    / "senate"
    / "warehouse"
)

ELECTION_RESULTS_PATH = (
    WAREHOUSE_ROOT
    / "processed"
    / "senate_historical_election_results_2012_2024.csv"
)

SPECIAL_OVERRIDE_PATH = (
    WAREHOUSE_ROOT
    / "raw"
    / "overrides"
    / "senate_special_election_class_overrides.csv"
)

OUTPUT_PATH = (
    WAREHOUSE_ROOT
    / "processed"
    / "senate_historical_classes_2012_2024.csv"
)

VALIDATION_PATH = (
    WAREHOUSE_ROOT
    / "validation"
    / "senate_historical_classes_2012_2024_validation.csv"
)

AUDIT_PATH = (
    WAREHOUSE_ROOT
    / "validation"
    / "senate_historical_class_override_audit.csv"
)


REGULAR_CLASS_BY_CYCLE = {
    2012: 1,
    2014: 2,
    2016: 3,
    2018: 1,
    2020: 2,
    2022: 3,
    2024: 1,
}


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not ELECTION_RESULTS_PATH.exists():
        raise FileNotFoundError(
            "Historical election-results warehouse not found:\n"
            f"{ELECTION_RESULTS_PATH}"
        )

    if not SPECIAL_OVERRIDE_PATH.exists():
        raise FileNotFoundError(
            "Special-election class override file not found:\n"
            f"{SPECIAL_OVERRIDE_PATH}"
        )

    results = pd.read_csv(ELECTION_RESULTS_PATH)

    overrides = pd.read_csv(
        SPECIAL_OVERRIDE_PATH,
        dtype={
            "race_id": "string",
            "senate_class": "Int64",
            "notes": "string",
        },
    )

    required_result_columns = {
        "race_id",
        "cycle",
        "state",
        "election_type",
        "special_election",
    }

    required_override_columns = {
        "race_id",
        "senate_class",
        "notes",
    }

    missing_results = sorted(
        required_result_columns - set(results.columns)
    )

    missing_overrides = sorted(
        required_override_columns - set(overrides.columns)
    )

    if missing_results:
        raise ValueError(
            "Election-results file is missing columns: "
            + ", ".join(missing_results)
        )

    if missing_overrides:
        raise ValueError(
            "Class override file is missing columns: "
            + ", ".join(missing_overrides)
        )

    return results, overrides


def validate_overrides(
    results: pd.DataFrame,
    overrides: pd.DataFrame,
) -> None:
    duplicate_ids = overrides.loc[
        overrides["race_id"].duplicated(keep=False)
    ]

    if not duplicate_ids.empty:
        raise ValueError(
            "Duplicate special-election class overrides:\n"
            + duplicate_ids.to_string(index=False)
        )

    invalid_classes = overrides.loc[
        ~overrides["senate_class"].isin([1, 2, 3])
    ]

    if not invalid_classes.empty:
        raise ValueError(
            "Invalid Senate classes in override file:\n"
            + invalid_classes.to_string(index=False)
        )

    special_ids = set(
        results.loc[
            results["special_election"].astype(bool),
            "race_id",
        ].astype(str)
    )

    override_ids = set(
        overrides["race_id"].astype(str)
    )

    missing_overrides = sorted(
        special_ids - override_ids
    )

    extra_overrides = sorted(
        override_ids - special_ids
    )

    if missing_overrides:
        raise ValueError(
            "Special elections missing class overrides: "
            + ", ".join(missing_overrides)
        )

    if extra_overrides:
        raise ValueError(
            "Class overrides reference races that are not "
            "special elections: "
            + ", ".join(extra_overrides)
        )


def build_classes(
    results: pd.DataFrame,
    overrides: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = results[
        [
            "race_id",
            "cycle",
            "state",
            "election_type",
            "special_election",
        ]
    ].copy()

    output["cycle"] = pd.to_numeric(
        output["cycle"],
        errors="raise",
    ).astype(int)

    output["special_election"] = (
        output["special_election"]
        .fillna(False)
        .astype(bool)
    )

    output["senate_class"] = pd.NA
    output["class_assignment_method"] = pd.NA
    output["class_assignment_notes"] = pd.NA

    regular_mask = ~output["special_election"]

    output.loc[
        regular_mask,
        "senate_class",
    ] = output.loc[
        regular_mask,
        "cycle",
    ].map(REGULAR_CLASS_BY_CYCLE)

    output.loc[
        regular_mask,
        "class_assignment_method",
    ] = "regular_cycle"

    output.loc[
        regular_mask,
        "class_assignment_notes",
    ] = (
        "Assigned from the regular six-year "
        "Senate election cycle"
    )

    override_lookup = overrides.set_index("race_id")

    audit_rows: list[dict[str, object]] = []

    for index, row in output.loc[
        output["special_election"]
    ].iterrows():
        race_id = str(row["race_id"])

        override = override_lookup.loc[race_id]

        senate_class = int(
            override["senate_class"]
        )

        output.at[
            index,
            "senate_class",
        ] = senate_class

        output.at[
            index,
            "class_assignment_method",
        ] = "special_election_override"

        output.at[
            index,
            "class_assignment_notes",
        ] = str(override["notes"])

        audit_rows.append(
            {
                "race_id": race_id,
                "cycle": int(row["cycle"]),
                "state": row["state"],
                "senate_class": senate_class,
                "assignment_method": (
                    "special_election_override"
                ),
                "notes": override["notes"],
            }
        )

    output["senate_class"] = pd.to_numeric(
        output["senate_class"],
        errors="raise",
    ).astype(int)

    output["seat_id"] = (
        output["state"].astype(str)
        + "_CLASS_"
        + output["senate_class"].astype(str)
    )

    output = output.sort_values(
        [
            "cycle",
            "state",
            "special_election",
        ]
    ).reset_index(drop=True)

    audit = pd.DataFrame(audit_rows)

    return output, audit


def build_validation(
    output: pd.DataFrame,
) -> pd.DataFrame:
    validation_rows = []

    for cycle, cycle_data in output.groupby(
        "cycle",
        sort=True,
    ):
        expected_regular_class = (
            REGULAR_CLASS_BY_CYCLE[int(cycle)]
        )

        regular = cycle_data.loc[
            ~cycle_data["special_election"]
        ]

        special = cycle_data.loc[
            cycle_data["special_election"]
        ]

        duplicate_seat_groups = (
            cycle_data.loc[
                cycle_data[
                    "seat_id"
                ].duplicated(keep=False)
            ]
            .groupby(
                "seat_id",
                dropna=False,
            )
        )

        valid_concurrent_pairs = 0
        invalid_duplicate_seats = 0

        for _, seat_group in duplicate_seat_groups:
            is_valid_pair = (
                len(seat_group) == 2
                and int(
                    seat_group[
                        "special_election"
                    ].sum()
                ) == 1
                and int(
                    (
                        ~seat_group[
                            "special_election"
                        ]
                    ).sum()
                ) == 1
            )

            if is_valid_pair:
                valid_concurrent_pairs += 1
            else:
                invalid_duplicate_seats += 1

        validation_rows.append(
            {
                "cycle": int(cycle),
                "race_rows": len(cycle_data),
                "regular_races": len(regular),
                "special_races": len(special),
                "expected_regular_class": (
                    expected_regular_class
                ),
                "regular_class_mismatches": int(
                    (
                        regular["senate_class"]
                        != expected_regular_class
                    ).sum()
                ),
                "missing_class": int(
                    cycle_data[
                        "senate_class"
                    ].isna().sum()
                ),
                "invalid_class": int(
                    (
                        ~cycle_data[
                            "senate_class"
                        ].isin([1, 2, 3])
                    ).sum()
                ),
                "duplicate_race_ids": int(
                    cycle_data[
                        "race_id"
                    ].duplicated().sum()
                ),
                "valid_concurrent_special_regular_pairs": (
                    valid_concurrent_pairs
                ),
                "invalid_duplicate_seats_same_cycle": (
                    invalid_duplicate_seats
                ),
            }
        )

    return pd.DataFrame(validation_rows)


def main() -> None:
    results, overrides = load_inputs()

    validate_overrides(
        results,
        overrides,
    )

    output, audit = build_classes(
        results,
        overrides,
    )

    validation = build_validation(output)

    fatal_columns = [
        "regular_class_mismatches",
        "missing_class",
        "invalid_class",
        "duplicate_race_ids",
        "invalid_duplicate_seats_same_cycle",
    ]

    failures = validation.loc[
        validation[fatal_columns]
        .sum(axis=1)
        .gt(0)
    ]

    if not failures.empty:
        raise ValueError(
            "Senate-class validation failed:\n"
            + failures.to_string(index=False)
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    VALIDATION_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    validation.to_csv(
        VALIDATION_PATH,
        index=False,
    )

    audit.to_csv(
        AUDIT_PATH,
        index=False,
    )

    print(
        "Historical races assigned:",
        len(output),
    )

    print()
    print("Class counts:")
    print(
        output["senate_class"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Assignment methods:")
    print(
        output[
            "class_assignment_method"
        ]
        .value_counts()
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
    print("Special-election audit:")
    print(
        audit.to_string(index=False)
    )

    print()
    print("Wrote:", OUTPUT_PATH)
    print("Wrote:", VALIDATION_PATH)
    print("Wrote:", AUDIT_PATH)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        raise
