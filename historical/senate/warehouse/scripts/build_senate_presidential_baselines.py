from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]

PRESIDENTIAL_RESULTS_PATH = (
    PROJECT_ROOT
    / "historical"
    / "senate"
    / "warehouse"
    / "processed"
    / "senate_historical_presidential_results.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "historical"
    / "senate"
    / "warehouse"
    / "processed"
    / "senate_presidential_baselines_by_cycle.csv"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "historical"
    / "senate"
    / "warehouse"
    / "validation"
    / "senate_presidential_baselines_by_cycle_validation.txt"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "historical"
    / "senate"
    / "warehouse"
    / "metadata"
    / "senate_presidential_baselines_by_cycle_metadata.json"
)

CYCLE_TO_PRESIDENTIAL_YEAR = {
    2012: 2008,
    2014: 2012,
    2016: 2012,
    2018: 2016,
    2020: 2016,
    2022: 2020,
    2024: 2020,
}

EXPECTED_STATES = 50


def main() -> None:
    if not PRESIDENTIAL_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing presidential warehouse: "
            f"{PRESIDENTIAL_RESULTS_PATH}"
        )

    presidential = pd.read_csv(
        PRESIDENTIAL_RESULTS_PATH
    )

    required_columns = {
        "year",
        "state",
        "state_po",
        "dem_candidate",
        "gop_candidate",
        "dem_votes",
        "gop_votes",
        "totalvotes",
        "dem_two_party_share",
        "gop_two_party_share",
        "presidential_margin_dem",
        "winner_party",
        "source_dataset",
        "source_doi",
        "source_file",
        "source_status",
        "validation_status",
    }

    missing_columns = sorted(
        required_columns - set(presidential.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing presidential columns: "
            f"{missing_columns}"
        )

    outputs = []

    for (
        forecast_cycle,
        presidential_year,
    ) in CYCLE_TO_PRESIDENTIAL_YEAR.items():
        cycle_rows = presidential[
            presidential["year"].eq(
                presidential_year
            )
        ].copy()

        if len(cycle_rows) != EXPECTED_STATES:
            raise ValueError(
                f"Expected {EXPECTED_STATES} presidential "
                f"rows for {presidential_year}; "
                f"found {len(cycle_rows)}."
            )

        cycle_rows.insert(
            0,
            "forecast_cycle",
            forecast_cycle,
        )

        cycle_rows = cycle_rows.rename(
            columns={
                "year": (
                    "presidential_result_year"
                ),
                "dem_candidate": (
                    "presidential_dem_candidate"
                ),
                "gop_candidate": (
                    "presidential_gop_candidate"
                ),
                "dem_votes": (
                    "presidential_dem_votes"
                ),
                "gop_votes": (
                    "presidential_gop_votes"
                ),
                "totalvotes": (
                    "presidential_total_votes"
                ),
                "dem_two_party_share": (
                    "presidential_dem_two_party_share"
                ),
                "gop_two_party_share": (
                    "presidential_gop_two_party_share"
                ),
                "winner_party": (
                    "presidential_winner_party"
                ),
            }
        )

        cycle_rows["baseline_method"] = (
            "most_recent_completed_presidential_election"
        )

        cycle_rows["lookahead_protection"] = True

        outputs.append(cycle_rows)

    output = pd.concat(
        outputs,
        ignore_index=True,
    )

    output = output[
        [
            "forecast_cycle",
            "presidential_result_year",
            "state",
            "state_po",
            "presidential_dem_candidate",
            "presidential_gop_candidate",
            "presidential_dem_votes",
            "presidential_gop_votes",
            "presidential_total_votes",
            "presidential_dem_two_party_share",
            "presidential_gop_two_party_share",
            "presidential_margin_dem",
            "presidential_winner_party",
            "baseline_method",
            "lookahead_protection",
            "source_dataset",
            "source_doi",
            "source_file",
            "source_status",
            "validation_status",
        ]
    ].sort_values(
        [
            "forecast_cycle",
            "state_po",
        ]
    ).reset_index(drop=True)

    expected_rows = (
        len(CYCLE_TO_PRESIDENTIAL_YEAR)
        * EXPECTED_STATES
    )

    duplicate_keys = int(
        output.duplicated(
            [
                "forecast_cycle",
                "state_po",
            ]
        ).sum()
    )

    rows_by_cycle = (
        output
        .groupby("forecast_cycle")
        .size()
        .reindex(
            sorted(
                CYCLE_TO_PRESIDENTIAL_YEAR
            ),
            fill_value=0,
        )
    )

    incorrect_year_assignments = int(
        (
            output.apply(
                lambda row: (
                    CYCLE_TO_PRESIDENTIAL_YEAR[
                        int(row["forecast_cycle"])
                    ]
                    != int(
                        row[
                            "presidential_result_year"
                        ]
                    )
                ),
                axis=1,
            )
        ).sum()
    )

    future_information_rows = int(
        (
            output["presidential_result_year"]
            >= output["forecast_cycle"]
        ).sum()
    )

    missing_margins = int(
        output[
            "presidential_margin_dem"
        ].isna().sum()
    )

    invalid_lookahead_flags = int(
        (
            output["lookahead_protection"]
            != True
        ).sum()
    )

    validation_passed = all(
        [
            len(output) == expected_rows,
            duplicate_keys == 0,
            incorrect_year_assignments == 0,
            future_information_rows == 0,
            missing_margins == 0,
            invalid_lookahead_flags == 0,
            rows_by_cycle.eq(
                EXPECTED_STATES
            ).all(),
        ]
    )

    if not validation_passed:
        raise ValueError(
            "Presidential baseline validation failed.\n"
            f"Expected rows: {expected_rows}\n"
            f"Output rows: {len(output)}\n"
            f"Duplicate keys: {duplicate_keys}\n"
            f"Incorrect year assignments: "
            f"{incorrect_year_assignments}\n"
            f"Future-information rows: "
            f"{future_information_rows}\n"
            f"Missing margins: {missing_margins}\n"
            f"Invalid lookahead flags: "
            f"{invalid_lookahead_flags}\n"
            f"Rows by cycle:\n{rows_by_cycle}"
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
        "Senate Presidential Baselines by Cycle Validation",
        "=" * 49,
        "",
        f"Source: {PRESIDENTIAL_RESULTS_PATH}",
        "",
        (
            "Cycle mapping: "
            f"{CYCLE_TO_PRESIDENTIAL_YEAR}"
        ),
        f"Expected rows: {expected_rows}",
        f"Output rows: {len(output)}",
        f"Duplicate cycle-state keys: {duplicate_keys}",
        (
            "Incorrect year assignments: "
            f"{incorrect_year_assignments}"
        ),
        (
            "Rows using same-cycle or future results: "
            f"{future_information_rows}"
        ),
        f"Missing presidential margins: {missing_margins}",
        (
            "Invalid lookahead-protection flags: "
            f"{invalid_lookahead_flags}"
        ),
        "",
        "Rows by cycle:",
        rows_by_cycle.to_string(),
        "",
        "Validation: PASSED",
    ]

    VALIDATION_PATH.write_text(
        "\n".join(validation_lines)
    )

    metadata = {
        "dataset_name": (
            "senate_presidential_baselines_by_cycle"
        ),
        "source_dataset": (
            "senate_historical_presidential_results"
        ),
        "primary_key": [
            "forecast_cycle",
            "state_po",
        ],
        "cycle_to_presidential_year": (
            CYCLE_TO_PRESIDENTIAL_YEAR
        ),
        "baseline_method": (
            "most_recent_completed_presidential_election"
        ),
        "lookahead_protection": True,
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
