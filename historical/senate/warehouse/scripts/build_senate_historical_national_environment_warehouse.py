from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]

SOURCE_PATH = (
    PROJECT_ROOT
    / "historical/senate/warehouse/raw/national_environment/"
    "senate_historical_election_day_environment_sources.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "historical/senate/warehouse/processed/"
    "senate_historical_national_environment_2012_2024.csv"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "historical/senate/warehouse/validation/"
    "senate_historical_national_environment_2012_2024_validation.txt"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "historical/senate/warehouse/metadata/"
    "senate_historical_national_environment_2012_2024_metadata.json"
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

REQUIRED_COLUMNS = [
    "cycle",
    "election_date",
    "president",
    "president_party",
    "generic_ballot_dem",
    "generic_ballot_gop",
    "generic_ballot_margin_dem",
    "presidential_approval",
    "presidential_disapproval",
    "approval_net",
    "approval_poll_start",
    "approval_poll_end",
    "generic_ballot_source",
    "approval_source",
    "source_status",
    "notes",
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


def main() -> None:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(SOURCE_PATH)

    source_hash = sha256_file(SOURCE_PATH)

    df = pd.read_csv(
        SOURCE_PATH,
        skipinitialspace=True,
    )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    df = df[REQUIRED_COLUMNS].copy()

    numeric_columns = [
        "cycle",
        "generic_ballot_dem",
        "generic_ballot_gop",
        "generic_ballot_margin_dem",
        "presidential_approval",
        "presidential_disapproval",
        "approval_net",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    date_columns = [
        "election_date",
        "approval_poll_start",
        "approval_poll_end",
    ]

    for column in date_columns:
        df[column] = pd.to_datetime(
            df[column],
            errors="coerce",
        )

    df["cycle"] = df["cycle"].astype("Int64")

    df["election_type"] = np.where(
        df["cycle"] % 4 == 0,
        "presidential",
        "midterm",
    )

    df["president_party"] = (
        df["president_party"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    df["generic_ballot_components_available"] = (
        df["generic_ballot_dem"].notna()
        & df["generic_ballot_gop"].notna()
    )

    calculated_generic_margin = (
        df["generic_ballot_dem"]
        - df["generic_ballot_gop"]
    )

    df["generic_ballot_margin_difference"] = np.where(
        df["generic_ballot_components_available"],
        (
            df["generic_ballot_margin_dem"]
            - calculated_generic_margin
        ),
        np.nan,
    )

    calculated_approval_net = (
        df["presidential_approval"]
        - df["presidential_disapproval"]
    )

    df["approval_net_difference"] = (
        df["approval_net"]
        - calculated_approval_net
    )

    df["approval_observation_pre_election"] = (
        df["approval_poll_end"].isna()
        | (
            df["approval_poll_end"]
            < df["election_date"]
        )
    )

    df["national_environment_scorable"] = (
        df["generic_ballot_margin_dem"].notna()
        & df["presidential_approval"].notna()
        & df["presidential_disapproval"].notna()
        & df["president_party"].isin(["D", "R"])
    )

    actual_cycles = sorted(
        df["cycle"]
        .dropna()
        .astype(int)
        .tolist()
    )

    missing_cycles = sorted(
        set(EXPECTED_CYCLES)
        - set(actual_cycles)
    )

    unexpected_cycles = sorted(
        set(actual_cycles)
        - set(EXPECTED_CYCLES)
    )

    duplicate_cycles = int(
        df["cycle"].duplicated(
            keep=False
        ).sum()
    )

    missing_generic_margins = int(
        df["generic_ballot_margin_dem"]
        .isna()
        .sum()
    )

    missing_approval = int(
        df["presidential_approval"]
        .isna()
        .sum()
    )

    missing_disapproval = int(
        df["presidential_disapproval"]
        .isna()
        .sum()
    )

    invalid_party_rows = int(
        (~df["president_party"].isin(["D", "R"]))
        .sum()
    )

    generic_arithmetic_failures = int(
        (
            df["generic_ballot_margin_difference"]
            .abs()
            .fillna(0.0)
            > 1e-9
        ).sum()
    )

    approval_arithmetic_failures = int(
        (
            df["approval_net_difference"]
            .abs()
            .fillna(0.0)
            > 1e-9
        ).sum()
    )

    post_election_approval_rows = int(
        (~df["approval_observation_pre_election"])
        .sum()
    )

    unscorable_rows = int(
        (~df["national_environment_scorable"])
        .sum()
    )

    validation_passed = (
        len(df) == len(EXPECTED_CYCLES)
        and not missing_cycles
        and not unexpected_cycles
        and duplicate_cycles == 0
        and missing_generic_margins == 0
        and missing_approval == 0
        and missing_disapproval == 0
        and invalid_party_rows == 0
        and generic_arithmetic_failures == 0
        and approval_arithmetic_failures == 0
        and post_election_approval_rows == 0
        and unscorable_rows == 0
    )

    if not validation_passed:
        validation_status = "FAILED"
    else:
        validation_status = "PASSED"

    df["national_environment_validation_status"] = (
        validation_status
    )

    output_columns = [
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

    output = (
        df[output_columns]
        .sort_values("cycle")
        .reset_index(drop=True)
    )

    for column in date_columns:
        output[column] = (
            output[column]
            .dt.strftime("%Y-%m-%d")
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
            "Senate Historical National Environment "
            "2012-2024 Validation"
        ),
        "=" * 62,
        "",
        f"Source: {SOURCE_PATH}",
        f"Source SHA-256: {source_hash}",
        "",
        f"Expected cycles: {len(EXPECTED_CYCLES)}",
        f"Output rows: {len(output)}",
        (
            "Missing cycles: "
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
        f"Duplicate cycle rows: {duplicate_cycles}",
        (
            "Missing generic-ballot margins: "
            f"{missing_generic_margins}"
        ),
        (
            "Missing presidential approval: "
            f"{missing_approval}"
        ),
        (
            "Missing presidential disapproval: "
            f"{missing_disapproval}"
        ),
        f"Invalid president-party rows: {invalid_party_rows}",
        (
            "Generic-ballot arithmetic failures: "
            f"{generic_arithmetic_failures}"
        ),
        (
            "Approval-net arithmetic failures: "
            f"{approval_arithmetic_failures}"
        ),
        (
            "Post-election approval observations: "
            f"{post_election_approval_rows}"
        ),
        (
            "National-environment unscorable rows: "
            f"{unscorable_rows}"
        ),
        "",
        "Cycle summary:",
        output[
            [
                "cycle",
                "election_type",
                "president_party",
                "generic_ballot_margin_dem",
                "presidential_approval",
                "presidential_disapproval",
                "approval_net",
                "source_status",
            ]
        ].to_string(index=False),
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
            "senate_historical_national_environment_"
            "2012_2024"
        ),
        "source_path": str(SOURCE_PATH),
        "source_sha256": source_hash,
        "output_path": str(OUTPUT_PATH),
        "validation_path": str(VALIDATION_PATH),
        "expected_cycles": EXPECTED_CYCLES,
        "row_count": int(len(output)),
        "formula_coefficients_applied": False,
        "description": (
            "Raw election-day national environment "
            "observations. No forecasting coefficients "
            "or modeled contributions are applied."
        ),
        "validation_status": validation_status,
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
            "Historical national-environment "
            "validation failed."
        )


if __name__ == "__main__":
    main()
