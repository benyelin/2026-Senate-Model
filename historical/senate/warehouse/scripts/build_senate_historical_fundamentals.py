from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]

BASELINES_PATH = (
    PROJECT_ROOT
    / "historical/senate/warehouse/processed/"
    "senate_historical_baselines_2012_2024.csv"
)

NATIONAL_ENVIRONMENT_PATH = (
    PROJECT_ROOT
    / "historical/senate/warehouse/processed/"
    "senate_historical_national_environment_2012_2024.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "historical/senate/warehouse/processed/"
    "senate_historical_fundamentals_2012_2024.csv"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "historical/senate/warehouse/validation/"
    "senate_historical_fundamentals_2012_2024_validation.txt"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "historical/senate/warehouse/metadata/"
    "senate_historical_fundamentals_2012_2024_metadata.json"
)

EXPECTED_CYCLES = [
    2012,
    2014,
    2016,
    2018,
    2020,
    2022,
    2024,
]

REQUIRED_BASELINE_COLUMNS = [
    "race_id",
    "cycle",
    "election_date",
    "state",
    "state_name",
    "senate_class",
    "seat_id",
    "election_type",
    "special_election",
    "actual_margin_dem",
    "winner_party",
    "presidential_result_year",
    "presidential_margin_dem",
    "senate_overperformance_dem",
    "model_incumbent_name",
    "model_incumbent_party",
    "model_incumbent_running",
    "model_open_seat",
    "major_party_contested",
    "backtest_scorable",
    "backtest_exclusion_reason",
    "presidential_baseline_available",
    "historical_baseline_scorable",
    "baseline_method",
    "lookahead_protection",
    "baseline_validation_status",
]

REQUIRED_ENVIRONMENT_COLUMNS = [
    "cycle",
    "election_date",
    "election_type",
    "president",
    "president_party",
    "generic_ballot_dem",
    "generic_ballot_gop",
    "generic_ballot_margin_dem",
    "generic_ballot_components_available",
    "presidential_approval",
    "presidential_disapproval",
    "approval_net",
    "approval_poll_start",
    "approval_poll_end",
    "approval_observation_pre_election",
    "generic_ballot_source",
    "approval_source",
    "source_status",
    "notes",
    "national_environment_scorable",
    "national_environment_validation_status",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def require_columns(
    df: pd.DataFrame,
    required: list[str],
    label: str,
) -> None:
    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{label} is missing required columns: "
            + ", ".join(missing)
        )


def normalize_boolean(
    series: pd.Series,
    column_name: str,
) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    normalized = (
        series
        .astype("string")
        .str.strip()
        .str.lower()
    )

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }

    unexpected = sorted(
        set(
            normalized.dropna().unique()
        )
        - set(mapping)
    )

    if unexpected:
        raise ValueError(
            f"Unexpected boolean values in {column_name}: "
            + ", ".join(map(str, unexpected))
        )

    return normalized.map(mapping).astype("boolean")


def main() -> None:
    for path in [
        BASELINES_PATH,
        NATIONAL_ENVIRONMENT_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    baseline_hash = sha256_file(
        BASELINES_PATH
    )

    environment_hash = sha256_file(
        NATIONAL_ENVIRONMENT_PATH
    )

    baselines = pd.read_csv(
        BASELINES_PATH
    )

    environment = pd.read_csv(
        NATIONAL_ENVIRONMENT_PATH
    )

    require_columns(
        baselines,
        REQUIRED_BASELINE_COLUMNS,
        "Historical baselines dataset",
    )

    require_columns(
        environment,
        REQUIRED_ENVIRONMENT_COLUMNS,
        "National-environment dataset",
    )

    baseline_rows = len(baselines)

    duplicate_race_ids_before = int(
        baselines["race_id"]
        .duplicated(keep=False)
        .sum()
    )

    duplicate_environment_cycles = int(
        environment["cycle"]
        .duplicated(keep=False)
        .sum()
    )

    if duplicate_race_ids_before:
        raise ValueError(
            "Historical baselines contain duplicate race IDs."
        )

    if duplicate_environment_cycles:
        raise ValueError(
            "National environment contains duplicate cycles."
        )

    boolean_columns = [
        "special_election",
        "model_incumbent_running",
        "model_open_seat",
        "major_party_contested",
        "backtest_scorable",
        "presidential_baseline_available",
        "historical_baseline_scorable",
    ]

    for column in boolean_columns:
        baselines[column] = normalize_boolean(
            baselines[column],
            column,
        )

    environment_boolean_columns = [
        "generic_ballot_components_available",
        "approval_observation_pre_election",
        "national_environment_scorable",
    ]

    for column in environment_boolean_columns:
        environment[column] = normalize_boolean(
            environment[column],
            column,
        )

    baselines["cycle"] = pd.to_numeric(
        baselines["cycle"],
        errors="raise",
    ).astype(int)

    environment["cycle"] = pd.to_numeric(
        environment["cycle"],
        errors="raise",
    ).astype(int)

    numeric_baseline_columns = [
        "actual_margin_dem",
        "presidential_margin_dem",
        "senate_overperformance_dem",
    ]

    for column in numeric_baseline_columns:
        baselines[column] = pd.to_numeric(
            baselines[column],
            errors="coerce",
        )

    numeric_environment_columns = [
        "generic_ballot_dem",
        "generic_ballot_gop",
        "generic_ballot_margin_dem",
        "presidential_approval",
        "presidential_disapproval",
        "approval_net",
    ]

    for column in numeric_environment_columns:
        environment[column] = pd.to_numeric(
            environment[column],
            errors="coerce",
        )

    environment_for_merge = environment[
        [
            "cycle",
            "election_date",
            "election_type",
            "president",
            "president_party",
            "generic_ballot_dem",
            "generic_ballot_gop",
            "generic_ballot_margin_dem",
            "generic_ballot_components_available",
            "presidential_approval",
            "presidential_disapproval",
            "approval_net",
            "approval_poll_start",
            "approval_poll_end",
            "approval_observation_pre_election",
            "generic_ballot_source",
            "approval_source",
            "source_status",
            "notes",
            "national_environment_scorable",
            "national_environment_validation_status",
        ]
    ].rename(
        columns={
            "election_date": (
                "national_environment_election_date"
            ),
            "election_type": (
                "national_environment_election_type"
            ),
            "source_status": (
                "national_environment_source_status"
            ),
            "notes": (
                "national_environment_notes"
            ),
        }
    )

    merged = baselines.merge(
        environment_for_merge,
        on="cycle",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    merged["national_environment_matched"] = (
        merged["_merge"] == "both"
    )

    unmatched_rows = int(
        (~merged["national_environment_matched"])
        .sum()
    )

    merged = merged.drop(
        columns=["_merge"]
    )

    merged["incumbent_party_dem"] = np.select(
        [
            (
                merged["model_incumbent_running"].fillna(False)
                & (
                    merged["model_incumbent_party"]
                    .astype("string")
                    .str.upper()
                    == "D"
                )
            ),
            (
                merged["model_incumbent_running"].fillna(False)
                & (
                    merged["model_incumbent_party"]
                    .astype("string")
                    .str.upper()
                    == "R"
                )
            ),
        ],
        [
            1.0,
            -1.0,
        ],
        default=0.0,
    )

    merged["incumbent_running_indicator"] = (
        merged["model_incumbent_running"]
        .fillna(False)
        .astype(int)
    )

    merged["open_seat_indicator"] = (
        merged["model_open_seat"]
        .fillna(False)
        .astype(int)
    )

    # Derive midterm status directly from the federal
    # election cycle rather than relying on election_type
    # labels, which are not consistently encoded as
    # "midterm" in the source data.
    #
    # Modern presidential election years are divisible
    # by four; federal elections two years later are
    # midterms.
    cycle_numeric = pd.to_numeric(
        merged["cycle"],
        errors="coerce",
    )

    merged["midterm_indicator"] = (
        cycle_numeric.mod(4).eq(2).astype(int)
    )

    merged["president_party_dem"] = np.select(
        [
            (
                merged["president_party"]
                .astype("string")
                .str.upper()
                == "D"
            ),
            (
                merged["president_party"]
                .astype("string")
                .str.upper()
                == "R"
            ),
        ],
        [
            1.0,
            -1.0,
        ],
        default=np.nan,
    )

    merged["presidential_approval_centered"] = (
        merged["presidential_approval"]
        - 50.0
    )

    merged["approval_net_dem_oriented"] = np.where(
        merged["president_party_dem"] == 1.0,
        merged["approval_net"],
        -merged["approval_net"],
    )

    merged["approval_centered_dem_oriented"] = np.where(
        merged["president_party_dem"] == 1.0,
        merged["presidential_approval_centered"],
        -merged["presidential_approval_centered"],
    )

    merged["historical_fundamentals_scorable"] = (
        merged["backtest_scorable"].fillna(False)
        & merged["historical_baseline_scorable"].fillna(False)
        & merged["national_environment_scorable"].fillna(False)
        & merged["major_party_contested"].fillna(False)
        & merged["actual_margin_dem"].notna()
        & merged["presidential_margin_dem"].notna()
        & merged["generic_ballot_margin_dem"].notna()
        & merged["presidential_approval"].notna()
        & merged["presidential_disapproval"].notna()
        & merged["approval_net"].notna()
        & merged["president_party_dem"].notna()
    )

    merged["fundamentals_exclusion_reason"] = ""

    exclusion_rules = [
        (
            ~merged["backtest_scorable"].fillna(False),
            "backtest_not_scorable",
        ),
        (
            ~merged["major_party_contested"].fillna(False),
            "not_major_party_contested",
        ),
        (
            ~merged["historical_baseline_scorable"].fillna(False),
            "historical_baseline_not_scorable",
        ),
        (
            ~merged["national_environment_scorable"].fillna(False),
            "national_environment_not_scorable",
        ),
        (
            merged["actual_margin_dem"].isna(),
            "missing_actual_margin",
        ),
        (
            merged["presidential_margin_dem"].isna(),
            "missing_presidential_margin",
        ),
        (
            merged["generic_ballot_margin_dem"].isna(),
            "missing_generic_ballot",
        ),
        (
            merged["presidential_approval"].isna(),
            "missing_presidential_approval",
        ),
        (
            merged["approval_net"].isna(),
            "missing_approval_net",
        ),
        (
            merged["president_party_dem"].isna(),
            "invalid_president_party",
        ),
    ]

    for mask, reason in exclusion_rules:
        add_reason = (
            mask
            & (
                merged["fundamentals_exclusion_reason"]
                == ""
            )
        )

        merged.loc[
            add_reason,
            "fundamentals_exclusion_reason",
        ] = reason

    merged.loc[
        merged["historical_fundamentals_scorable"],
        "fundamentals_exclusion_reason",
    ] = ""

    output_columns = [
        "race_id",
        "cycle",
        "election_date",
        "state",
        "state_name",
        "senate_class",
        "seat_id",
        "election_type",
        "special_election",
        "actual_margin_dem",
        "winner_party",
        "presidential_result_year",
        "presidential_margin_dem",
        "senate_overperformance_dem",
        "model_incumbent_name",
        "model_incumbent_party",
        "model_incumbent_running",
        "model_open_seat",
        "incumbent_party_dem",
        "incumbent_running_indicator",
        "open_seat_indicator",
        "major_party_contested",
        "backtest_scorable",
        "backtest_exclusion_reason",
        "presidential_baseline_available",
        "historical_baseline_scorable",
        "baseline_method",
        "lookahead_protection",
        "baseline_validation_status",
        "president",
        "president_party",
        "president_party_dem",
        "midterm_indicator",
        "generic_ballot_dem",
        "generic_ballot_gop",
        "generic_ballot_margin_dem",
        "generic_ballot_components_available",
        "presidential_approval",
        "presidential_disapproval",
        "approval_net",
        "presidential_approval_centered",
        "approval_net_dem_oriented",
        "approval_centered_dem_oriented",
        "approval_poll_start",
        "approval_poll_end",
        "approval_observation_pre_election",
        "generic_ballot_source",
        "approval_source",
        "national_environment_source_status",
        "national_environment_notes",
        "national_environment_scorable",
        "national_environment_validation_status",
        "national_environment_matched",
        "historical_fundamentals_scorable",
        "fundamentals_exclusion_reason",
    ]

    output = (
        merged[output_columns]
        .sort_values(
            [
                "cycle",
                "state",
                "special_election",
                "race_id",
            ]
        )
        .reset_index(drop=True)
    )

    output_rows = len(output)

    duplicate_race_ids_after = int(
        output["race_id"]
        .duplicated(keep=False)
        .sum()
    )

    output_cycles = sorted(
        output["cycle"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    missing_cycles = sorted(
        set(EXPECTED_CYCLES)
        - set(output_cycles)
    )

    unexpected_cycles = sorted(
        set(output_cycles)
        - set(EXPECTED_CYCLES)
    )

    scorable_rows = int(
        output["historical_fundamentals_scorable"]
        .sum()
    )

    unscorable_rows = int(
        (
            ~output["historical_fundamentals_scorable"]
        ).sum()
    )

    missing_environment_matches = int(
        (
            ~output["national_environment_matched"]
        ).sum()
    )

    baseline_identity_error = (
        output["actual_margin_dem"]
        - output["presidential_margin_dem"]
        - output["senate_overperformance_dem"]
    )

    baseline_identity_failures = int(
        (
            baseline_identity_error
            .abs()
            .fillna(0.0)
            > 1e-8
        ).sum()
    )

    invalid_incumbent_encoding = int(
        (
            ~output["incumbent_party_dem"]
            .isin([-1.0, 0.0, 1.0])
        ).sum()
    )

    invalid_open_seat_encoding = int(
        (
            ~output["open_seat_indicator"]
            .isin([0, 1])
        ).sum()
    )

    invalid_midterm_encoding = int(
        (
            ~output["midterm_indicator"]
            .isin([0, 1])
        ).sum()
    )

    expected_midterm_indicator = (
        pd.to_numeric(
            output["cycle"],
            errors="coerce",
        )
        .mod(4)
        .eq(2)
        .astype(int)
    )

    invalid_midterm_cycle_alignment = int(
        (
            output["midterm_indicator"]
            .ne(expected_midterm_indicator)
        ).sum()
    )

    cycle_summary = (
        output.groupby(
            "cycle",
            as_index=False,
        )
        .agg(
            races=("race_id", "size"),
            scorable=(
                "historical_fundamentals_scorable",
                "sum",
            ),
            generic_ballot_margin_dem=(
                "generic_ballot_margin_dem",
                "first",
            ),
            presidential_approval=(
                "presidential_approval",
                "first",
            ),
            approval_net=(
                "approval_net",
                "first",
            ),
        )
    )

    validation_passed = (
        baseline_rows == output_rows
        and duplicate_race_ids_before == 0
        and duplicate_race_ids_after == 0
        and duplicate_environment_cycles == 0
        and missing_environment_matches == 0
        and not missing_cycles
        and not unexpected_cycles
        and baseline_identity_failures == 0
        and invalid_incumbent_encoding == 0
        and invalid_open_seat_encoding == 0
        and invalid_midterm_encoding == 0
        and invalid_midterm_cycle_alignment == 0
        and scorable_rows > 0
    )

    validation_status = (
        "PASSED"
        if validation_passed
        else "FAILED"
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
        (
            "Senate Historical Fundamentals "
            "2012-2024 Validation"
        ),
        "=" * 55,
        "",
        f"Baselines source: {BASELINES_PATH}",
        f"Baselines SHA-256: {baseline_hash}",
        "",
        (
            "National-environment source: "
            f"{NATIONAL_ENVIRONMENT_PATH}"
        ),
        (
            "National-environment SHA-256: "
            f"{environment_hash}"
        ),
        "",
        f"Input baseline rows: {baseline_rows}",
        f"Output rows: {output_rows}",
        (
            "Duplicate baseline race IDs: "
            f"{duplicate_race_ids_before}"
        ),
        (
            "Duplicate output race IDs: "
            f"{duplicate_race_ids_after}"
        ),
        (
            "Duplicate environment cycles: "
            f"{duplicate_environment_cycles}"
        ),
        (
            "Missing environment matches: "
            f"{missing_environment_matches}"
        ),
        (
            "Missing expected cycles: "
            + (
                ", ".join(map(str, missing_cycles))
                if missing_cycles
                else "none"
            )
        ),
        (
            "Unexpected cycles: "
            + (
                ", ".join(map(str, unexpected_cycles))
                if unexpected_cycles
                else "none"
            )
        ),
        (
            "Baseline identity failures: "
            f"{baseline_identity_failures}"
        ),
        (
            "Invalid incumbent encodings: "
            f"{invalid_incumbent_encoding}"
        ),
        (
            "Invalid open-seat encodings: "
            f"{invalid_open_seat_encoding}"
        ),
        (
            "Invalid midterm encodings: "
            f"{invalid_midterm_encoding}"
        ),
        (
            "Invalid midterm/cycle alignments: "
            f"{invalid_midterm_cycle_alignment}"
        ),
        f"Scorable rows: {scorable_rows}",
        f"Unscorable rows: {unscorable_rows}",
        "",
        "Cycle summary:",
        cycle_summary.to_string(index=False),
        "",
        "No forecasting coefficients were applied.",
        (
            "This dataset contains historical observations "
            "and model-ready feature encodings only."
        ),
        "",
        f"Validation: {validation_status}",
    ]

    validation_text = "\n".join(
        validation_lines
    )

    VALIDATION_PATH.write_text(
        validation_text
    )

    metadata = {
        "dataset": (
            "senate_historical_fundamentals_"
            "2012_2024"
        ),
        "baseline_source_path": str(
            BASELINES_PATH
        ),
        "baseline_source_sha256": baseline_hash,
        "national_environment_source_path": str(
            NATIONAL_ENVIRONMENT_PATH
        ),
        "national_environment_source_sha256": (
            environment_hash
        ),
        "output_path": str(OUTPUT_PATH),
        "validation_path": str(
            VALIDATION_PATH
        ),
        "expected_cycles": EXPECTED_CYCLES,
        "row_count": int(output_rows),
        "scorable_row_count": int(
            scorable_rows
        ),
        "formula_coefficients_applied": False,
        "validation_status": validation_status,
        "description": (
            "Race-level Senate historical fundamentals "
            "dataset joining presidential baselines, "
            "national environment, incumbency, open-seat "
            "status, and observed Senate results."
        ),
    }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
        + "\n"
    )

    print(validation_text)
    print()
    print(f"Wrote: {OUTPUT_PATH}")
    print(f"Wrote: {VALIDATION_PATH}")
    print(f"Wrote: {METADATA_PATH}")

    if not validation_passed:
        raise SystemExit(
            "Historical fundamentals validation failed."
        )


if __name__ == "__main__":
    main()
