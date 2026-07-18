#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
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
    / "senate_historical_election_results_enriched_2012_2024.csv"
)

PRESIDENTIAL_BASELINES_PATH = (
    WAREHOUSE_ROOT
    / "processed"
    / "senate_presidential_baselines_by_cycle.csv"
)

OUTPUT_PATH = (
    WAREHOUSE_ROOT
    / "processed"
    / "senate_historical_baselines_2012_2024.csv"
)

VALIDATION_PATH = (
    WAREHOUSE_ROOT
    / "validation"
    / "senate_historical_baselines_2012_2024_validation.txt"
)

METADATA_PATH = (
    WAREHOUSE_ROOT
    / "metadata"
    / "senate_historical_baselines_2012_2024_metadata.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def normalize_bool(
    series: pd.Series,
) -> pd.Series:
    mapping = {
        True: True,
        False: False,
        "True": True,
        "False": False,
        "TRUE": True,
        "FALSE": False,
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        1: True,
        0: False,
    }

    return series.map(mapping).astype("boolean")


def main() -> None:
    for path in [
        ELECTION_RESULTS_PATH,
        PRESIDENTIAL_BASELINES_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required input not found:\n{path}"
            )

    elections = pd.read_csv(
        ELECTION_RESULTS_PATH,
        low_memory=False,
    )

    baselines = pd.read_csv(
        PRESIDENTIAL_BASELINES_PATH,
        low_memory=False,
    )

    election_required = {
        "race_id",
        "cycle",
        "state",
        "state_name",
        "actual_margin_dem",
        "backtest_scorable",
        "lineage_incumbent_name",
        "lineage_incumbent_party",
        "lineage_incumbent_running",
        "lineage_open_seat",
        "incumbency_audit_flag",
    }

    baseline_required = {
        "forecast_cycle",
        "state_po",
        "presidential_result_year",
        "presidential_margin_dem",
        "presidential_dem_two_party_share",
        "presidential_gop_two_party_share",
        "presidential_winner_party",
        "baseline_method",
        "lookahead_protection",
    }

    missing_election_columns = sorted(
        election_required - set(elections.columns)
    )

    missing_baseline_columns = sorted(
        baseline_required - set(baselines.columns)
    )

    if missing_election_columns:
        raise ValueError(
            "Election results are missing columns: "
            + ", ".join(missing_election_columns)
        )

    if missing_baseline_columns:
        raise ValueError(
            "Presidential baselines are missing columns: "
            + ", ".join(missing_baseline_columns)
        )

    elections["cycle"] = pd.to_numeric(
        elections["cycle"],
        errors="raise",
    ).astype(int)

    elections["state"] = (
        elections["state"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    elections["actual_margin_dem"] = pd.to_numeric(
        elections["actual_margin_dem"],
        errors="coerce",
    )

    baselines["forecast_cycle"] = pd.to_numeric(
        baselines["forecast_cycle"],
        errors="raise",
    ).astype(int)

    baselines["state_po"] = (
        baselines["state_po"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    baselines["presidential_margin_dem"] = pd.to_numeric(
        baselines["presidential_margin_dem"],
        errors="coerce",
    )

    baseline_subset = baselines[
        [
            "forecast_cycle",
            "state_po",
            "presidential_result_year",
            "presidential_dem_candidate",
            "presidential_gop_candidate",
            "presidential_dem_two_party_share",
            "presidential_gop_two_party_share",
            "presidential_margin_dem",
            "presidential_winner_party",
            "baseline_method",
            "lookahead_protection",
            "source_dataset",
            "source_doi",
            "source_file",
        ]
    ].copy()

    baseline_duplicate_keys = int(
        baseline_subset.duplicated(
            [
                "forecast_cycle",
                "state_po",
            ]
        ).sum()
    )

    if baseline_duplicate_keys:
        raise ValueError(
            "Presidential baseline table contains "
            f"{baseline_duplicate_keys} duplicate "
            "cycle-state keys."
        )

    output = elections.merge(
        baseline_subset,
        left_on=[
            "cycle",
            "state",
        ],
        right_on=[
            "forecast_cycle",
            "state_po",
        ],
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    unmatched_races = output.loc[
        output["_merge"].ne("both"),
        [
            "race_id",
            "cycle",
            "state",
        ],
    ].copy()

    output = output.drop(
        columns=[
            "_merge",
            "forecast_cycle",
            "state_po",
        ]
    )

    # Canonical modeling incumbency fields come from
    # the audited Senate seat-lineage enrichment.
    output["model_incumbent_name"] = (
        output["lineage_incumbent_name"]
    )

    output["model_incumbent_party"] = (
        output["lineage_incumbent_party"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    output["model_incumbent_running"] = normalize_bool(
        output["lineage_incumbent_running"]
    )

    output["model_open_seat"] = normalize_bool(
        output["lineage_open_seat"]
    )

    output["incumbency_source"] = (
        "audited_senate_seat_lineage"
    )

    output["senate_overperformance_dem"] = (
        output["actual_margin_dem"]
        - output["presidential_margin_dem"]
    )

    output["senate_overperformance_abs"] = (
        output["senate_overperformance_dem"].abs()
    )

    output["senate_outperformed_presidential_baseline"] = (
        output["senate_overperformance_dem"] > 0
    ).astype("boolean")

    output["presidential_baseline_available"] = (
        output["presidential_margin_dem"].notna()
    )

    output["historical_baseline_scorable"] = (
        output["backtest_scorable"].fillna(False).astype(bool)
        & output["actual_margin_dem"].notna()
        & output["presidential_margin_dem"].notna()
    )

    output["baseline_validation_status"] = np.where(
        output["historical_baseline_scorable"],
        "validated_scorable",
        "not_scorable",
    )

    preferred_columns = [
        "race_id",
        "cycle",
        "election_date",
        "state",
        "state_name",
        "senate_class",
        "seat_id",
        "election_type",
        "special_election",
        "dem_candidate",
        "gop_candidate",
        "dem_votes",
        "gop_votes",
        "other_votes",
        "total_votes",
        "dem_two_party_share",
        "gop_two_party_share",
        "actual_margin_dem",
        "winner_party",
        "presidential_result_year",
        "presidential_dem_candidate",
        "presidential_gop_candidate",
        "presidential_dem_two_party_share",
        "presidential_gop_two_party_share",
        "presidential_margin_dem",
        "presidential_winner_party",
        "senate_overperformance_dem",
        "senate_overperformance_abs",
        "senate_outperformed_presidential_baseline",
        "model_incumbent_name",
        "model_incumbent_party",
        "model_incumbent_running",
        "model_open_seat",
        "incumbency_source",
        "lineage_appointed_incumbent",
        "lineage_candidate_match_method",
        "incumbency_override_applied",
        "incumbency_audit_flag",
        "major_party_contested",
        "backtest_scorable",
        "backtest_exclusion_reason",
        "presidential_baseline_available",
        "historical_baseline_scorable",
        "baseline_method",
        "lookahead_protection",
        "baseline_validation_status",
        "source",
        "source_url",
        "source_status",
        "source_dataset",
        "source_doi",
        "source_file",
        "notes",
    ]

    remaining_columns = [
        column
        for column in output.columns
        if column not in preferred_columns
    ]

    output = output[
        preferred_columns + remaining_columns
    ].sort_values(
        [
            "cycle",
            "state",
            "special_election",
            "race_id",
        ]
    ).reset_index(drop=True)

    expected_rows = len(elections)
    output_rows = len(output)

    duplicate_race_ids = int(
        output["race_id"].duplicated().sum()
    )

    missing_presidential_baselines = int(
        output[
            "presidential_margin_dem"
        ].isna().sum()
    )

    missing_actual_margins = int(
        output["actual_margin_dem"].isna().sum()
    )

    missing_overperformance = int(
        output[
            "senate_overperformance_dem"
        ].isna().sum()
    )

    future_information_rows = int(
        (
            output["presidential_result_year"]
            >= output["cycle"]
        ).fillna(False).sum()
    )

    inconsistent_overperformance = int(
        (
            (
                output["actual_margin_dem"]
                - output["presidential_margin_dem"]
                - output["senate_overperformance_dem"]
            ).abs()
            > 1e-10
        ).fillna(False).sum()
    )

    invalid_open_seat_running_rows = int(
        (
            output["model_open_seat"].eq(True)
            & output["model_incumbent_running"].eq(True)
        ).fillna(False).sum()
    )

    scorable_missing_inputs = int(
        (
            output["historical_baseline_scorable"]
            & (
                output["actual_margin_dem"].isna()
                | output[
                    "presidential_margin_dem"
                ].isna()
            )
        ).sum()
    )

    rows_by_cycle = (
        output
        .groupby("cycle")
        .size()
        .sort_index()
    )

    scorable_by_cycle = (
        output
        .groupby("cycle")[
            "historical_baseline_scorable"
        ]
        .sum()
        .astype(int)
        .sort_index()
    )

    overperformance_summary = (
        output.loc[
            output["historical_baseline_scorable"]
        ]
        .groupby("cycle")[
            "senate_overperformance_dem"
        ]
        .agg(
            races="count",
            mean="mean",
            median="median",
            minimum="min",
            maximum="max",
        )
    )

    validation_passed = all(
        [
            output_rows == expected_rows,
            len(unmatched_races) == 0,
            duplicate_race_ids == 0,
            missing_presidential_baselines == 0,
            missing_overperformance
            == missing_actual_margins,
            future_information_rows == 0,
            inconsistent_overperformance == 0,
            invalid_open_seat_running_rows == 0,
            scorable_missing_inputs == 0,
        ]
    )

    if not validation_passed:
        raise ValueError(
            "Historical baseline validation failed.\n"
            f"Expected rows: {expected_rows}\n"
            f"Output rows: {output_rows}\n"
            f"Unmatched races: {len(unmatched_races)}\n"
            f"Duplicate race IDs: {duplicate_race_ids}\n"
            f"Missing presidential baselines: "
            f"{missing_presidential_baselines}\n"
            f"Missing actual margins: "
            f"{missing_actual_margins}\n"
            f"Missing overperformance: "
            f"{missing_overperformance}\n"
            f"Future-information rows: "
            f"{future_information_rows}\n"
            f"Inconsistent overperformance rows: "
            f"{inconsistent_overperformance}\n"
            f"Open-seat/incumbent-running conflicts: "
            f"{invalid_open_seat_running_rows}\n"
            f"Scorable rows missing inputs: "
            f"{scorable_missing_inputs}\n"
            f"Unmatched race details:\n"
            f"{unmatched_races.to_string(index=False)}"
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
        "Senate Historical Baselines 2012-2024 Validation",
        "=" * 50,
        "",
        f"Election source: {ELECTION_RESULTS_PATH}",
        (
            "Election source SHA-256: "
            f"{sha256(ELECTION_RESULTS_PATH)}"
        ),
        f"Baseline source: {PRESIDENTIAL_BASELINES_PATH}",
        (
            "Baseline source SHA-256: "
            f"{sha256(PRESIDENTIAL_BASELINES_PATH)}"
        ),
        "",
        f"Expected rows: {expected_rows}",
        f"Output rows: {output_rows}",
        f"Unmatched races: {len(unmatched_races)}",
        f"Duplicate race IDs: {duplicate_race_ids}",
        (
            "Missing presidential baselines: "
            f"{missing_presidential_baselines}"
        ),
        f"Missing actual margins: {missing_actual_margins}",
        (
            "Missing overperformance values: "
            f"{missing_overperformance}"
        ),
        (
            "Rows using same-cycle or future "
            f"presidential results: {future_information_rows}"
        ),
        (
            "Inconsistent overperformance values: "
            f"{inconsistent_overperformance}"
        ),
        (
            "Open-seat/incumbent-running conflicts: "
            f"{invalid_open_seat_running_rows}"
        ),
        (
            "Scorable rows missing required inputs: "
            f"{scorable_missing_inputs}"
        ),
        "",
        "Rows by cycle:",
        rows_by_cycle.to_string(),
        "",
        "Scorable rows by cycle:",
        scorable_by_cycle.to_string(),
        "",
        "Democratic Senate overperformance by cycle:",
        overperformance_summary.to_string(
            float_format=lambda value: f"{value:.4f}"
        ),
        "",
        "Validation: PASSED",
    ]

    VALIDATION_PATH.write_text(
        "\n".join(validation_lines)
    )

    metadata = {
        "dataset_name": (
            "senate_historical_baselines_2012_2024"
        ),
        "description": (
            "Historical Senate election results merged "
            "with the most recent completed presidential "
            "election baseline available before each race."
        ),
        "primary_key": ["race_id"],
        "join_keys": {
            "election_results": [
                "cycle",
                "state",
            ],
            "presidential_baselines": [
                "forecast_cycle",
                "state_po",
            ],
        },
        "row_count": int(output_rows),
        "cycles": sorted(
            int(value)
            for value in output["cycle"].unique()
        ),
        "overperformance_formula": (
            "actual_margin_dem - "
            "presidential_margin_dem"
        ),
        "incumbency_source": (
            "audited_senate_seat_lineage"
        ),
        "lookahead_protection": True,
        "input_files": {
            "election_results": str(
                ELECTION_RESULTS_PATH
            ),
            "presidential_baselines": str(
                PRESIDENTIAL_BASELINES_PATH
            ),
        },
        "input_sha256": {
            "election_results": sha256(
                ELECTION_RESULTS_PATH
            ),
            "presidential_baselines": sha256(
                PRESIDENTIAL_BASELINES_PATH
            ),
        },
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
