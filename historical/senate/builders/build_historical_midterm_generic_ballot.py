from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]

RAW_OUTPUT = (
    ROOT
    / "historical/senate/warehouse/raw/generic_ballot/"
      "historical_midterm_generic_ballot_manual.csv"
)

PROCESSED_OUTPUT = (
    ROOT
    / "historical/senate/warehouse/processed/generic_ballot/"
      "historical_midterm_generic_ballot.csv"
)

VALIDATION_OUTPUT = (
    ROOT
    / "historical/senate/diagnostics/gasoline/"
      "historical_midterm_generic_ballot_validation.csv"
)


# IMPORTANT:
# These values must represent PRE-ELECTION generic-ballot polling,
# not actual House popular vote.
#
# margin_dem convention:
#   positive = Democratic advantage
#   negative = Republican advantage
#
# Older Gallup observations use the final pre-election likely-voter
# estimate where available.
#
# Modern observations may use final polling averages.
#
# We deliberately leave years missing rather than impute from actual
# election results.

MIDTERMS = [
    1978,
    1982,
    1986,
    1990,
    1994,
    1998,
    2002,
    2006,
    2010,
    2014,
    2018,
    2022,
]


def main():
    print("=" * 110)
    print("HISTORICAL MIDTERM GENERIC BALLOT")
    print("=" * 110)

    RAW_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    PROCESSED_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    VALIDATION_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not RAW_OUTPUT.exists():
        template = pd.DataFrame(
            {
                "year": MIDTERMS,
                "generic_ballot_margin_dem": pd.NA,
                "source": "",
                "source_type": "",
                "population": "",
                "source_quality": "",
                "notes": "",
            }
        )

        template.to_csv(
            RAW_OUTPUT,
            index=False,
        )

        print()
        print("Created manual source template:")
        print(RAW_OUTPUT)

    raw = pd.read_csv(
        RAW_OUTPUT,
        low_memory=False,
    )

    raw["year"] = pd.to_numeric(
        raw["year"],
        errors="coerce",
    ).astype("Int64")

    raw["generic_ballot_margin_dem"] = pd.to_numeric(
        raw["generic_ballot_margin_dem"],
        errors="coerce",
    )

    raw = (
        raw.loc[
            raw["year"].isin(
                MIDTERMS
            )
        ]
        .sort_values("year")
        .reset_index(drop=True)
    )

    raw.to_csv(
        PROCESSED_OUTPUT,
        index=False,
    )

    observed = raw.loc[
        raw["generic_ballot_margin_dem"].notna()
    ].copy()

    missing = raw.loc[
        raw["generic_ballot_margin_dem"].isna(),
        "year",
    ].tolist()

    checks = pd.DataFrame(
        [
            {
                "check": "all_target_midterms_present",
                "passed": set(raw["year"].dropna())
                == set(MIDTERMS),
            },
            {
                "check": "unique_years",
                "passed": not raw["year"].duplicated().any(),
            },
            {
                "check": "observed_values_plausible",
                "passed": (
                    observed["generic_ballot_margin_dem"]
                    .between(-40, 40)
                    .all()
                    if len(observed)
                    else True
                ),
            },
        ]
    )

    checks.to_csv(
        VALIDATION_OUTPUT,
        index=False,
    )

    print()
    print("TARGET MIDTERMS")
    print("-" * 110)

    print(
        raw.to_string(
            index=False
        )
    )

    print()
    print("COVERAGE")
    print("-" * 110)

    print(
        "Observed:",
        len(observed),
        "/",
        len(MIDTERMS),
    )

    print(
        "Missing years:",
        missing,
    )

    print()
    print("VALIDATION")
    print("-" * 110)

    print(
        checks.to_string(
            index=False
        )
    )

    if not checks["passed"].all():
        raise RuntimeError(
            "Historical generic-ballot validation failed."
        )

    print()
    print(
        "Historical generic-ballot structure validation PASSED."
    )

    print()
    print(
        "Wrote:",
        PROCESSED_OUTPUT,
    )

    print(
        "Wrote:",
        VALIDATION_OUTPUT,
    )


if __name__ == "__main__":
    main()
