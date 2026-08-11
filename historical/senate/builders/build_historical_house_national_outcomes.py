from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


HOUSE_SOURCE = Path(
    "/Users/benyelin/Developer/house_model_python/"
    "historical/house/raw/2022/source_downloads/"
    "1976-2024-house.tab"
)

ROOT = Path(__file__).resolve().parents[3]

OUTPUT = (
    ROOT
    / "historical/senate/warehouse/processed/economic/"
    / "historical_house_national_outcomes_1976_2024.csv"
)

VALIDATION_OUTPUT = (
    ROOT
    / "historical/senate/diagnostics/gasoline/"
    / "historical_house_national_outcomes_validation.csv"
)


PRESIDENT_PARTY = {
    1976: "R",
    1978: "D",
    1980: "D",
    1982: "R",
    1984: "R",
    1986: "R",
    1988: "R",
    1990: "R",
    1992: "R",
    1994: "D",
    1996: "D",
    1998: "D",
    2000: "D",
    2002: "R",
    2004: "R",
    2006: "R",
    2008: "R",
    2010: "D",
    2012: "D",
    2014: "D",
    2016: "D",
    2018: "R",
    2020: "R",
    2022: "D",
    2024: "D",
}


# Party labels that clearly represent the
# Democratic or Republican major-party line.
#
# We intentionally keep this conservative.
# Fusion-ticket handling below assigns fusion
# votes according to the candidate's major-party
# affiliation where one is identifiable.
DEM_LABELS = {
    "DEMOCRAT",
    "DEMOCRATIC",
    "DEMOCRATIC-FARMER-LABOR",
}

GOP_LABELS = {
    "REPUBLICAN",
}


def normalize_text(value):
    if pd.isna(value):
        return ""

    return str(value).strip().upper()


def main():
    print("=" * 120)
    print("HISTORICAL NATIONAL HOUSE OUTCOMES — 1976–2024")
    print("=" * 120)

    df = pd.read_csv(
        HOUSE_SOURCE,
        low_memory=False,
    )

    df["year"] = pd.to_numeric(
        df["year"],
        errors="coerce",
    )

    df["candidatevotes"] = pd.to_numeric(
        df["candidatevotes"],
        errors="coerce",
    )

    df["party_norm"] = (
        df["party"]
        .map(normalize_text)
    )

    df["candidate_norm"] = (
        df["candidate"]
        .map(normalize_text)
    )

    # Keep regular general-election House returns.
    df = df.loc[
        df["stage"]
        .astype("string")
        .str.upper()
        .eq("GEN")
    ].copy()

    df = df.loc[
        ~df["special"].fillna(False).astype(bool)
    ].copy()

    df = df.loc[
        df["mode"]
        .astype("string")
        .str.upper()
        .eq("TOTAL")
    ].copy()

    # --------------------------------------------------
    # Assign votes by the ballot-line party actually
    # reported in the source.
    #
    # This is important in fusion-voting states such as
    # New York, where the same candidate can appear on
    # Democratic, Republican, Liberal, Conservative, or
    # other ballot lines in the same election.
    #
    # We therefore do NOT assign all of a candidate's
    # votes to a single inferred major party. Only votes
    # explicitly cast on Democratic-family or Republican
    # ballot lines enter the two-party national margin.
    # --------------------------------------------------

    df["major_party"] = np.select(
        [
            df["party_norm"].isin(
                DEM_LABELS
            ),
            df["party_norm"].isin(
                GOP_LABELS
            ),
        ],
        [
            "D",
            "R",
        ],
        default="OTHER",
    )

    candidate_totals = df[
        [
            "year",
            "state_po",
            "district",
            "candidate_norm",
            "major_party",
            "candidatevotes",
        ]
    ].copy()

    # National D/R totals.
    major = candidate_totals.loc[
        candidate_totals[
            "major_party"
        ].isin(
            ["D", "R"]
        )
    ].copy()

    national = (
        major.groupby(
            [
                "year",
                "major_party",
            ],
            as_index=False,
        )["candidatevotes"]
        .sum()
        .pivot(
            index="year",
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

    national.columns.name = None

    national["cycle"] = (
        national["year"]
        .astype(int)
    )

    national["two_party_votes"] = (
        national["dem_votes"]
        + national["gop_votes"]
    )

    national["dem_two_party_share"] = (
        national["dem_votes"]
        / national["two_party_votes"]
    )

    national["gop_two_party_share"] = (
        national["gop_votes"]
        / national["two_party_votes"]
    )

    national["house_margin_dem"] = (
        (
            national[
                "dem_two_party_share"
            ]
            - national[
                "gop_two_party_share"
            ]
        )
        * 100.0
    )

    national = national.sort_values(
        "cycle"
    ).reset_index(
        drop=True
    )

    national[
        "previous_house_margin_dem"
    ] = (
        national[
            "house_margin_dem"
        ]
        .shift(1)
    )

    national[
        "house_margin_change_dem"
    ] = (
        national[
            "house_margin_dem"
        ]
        - national[
            "previous_house_margin_dem"
        ]
    )

    national["president_party"] = (
        national["cycle"]
        .map(
            PRESIDENT_PARTY
        )
    )

    national[
        "president_party_house_margin"
    ] = np.where(
        national[
            "president_party"
        ].eq("D"),
        national[
            "house_margin_dem"
        ],
        -national[
            "house_margin_dem"
        ],
    )

    # Orient the change toward the party holding
    # the presidency in the CURRENT election cycle.
    national[
        "president_party_house_margin_change"
    ] = np.where(
        national[
            "president_party"
        ].eq("D"),
        national[
            "house_margin_change_dem"
        ],
        -national[
            "house_margin_change_dem"
        ],
    )

    national["election_type"] = np.where(
        national["cycle"] % 4 == 0,
        "presidential",
        "midterm",
    )

    output_columns = [
        "cycle",
        "election_type",
        "president_party",
        "dem_votes",
        "gop_votes",
        "two_party_votes",
        "dem_two_party_share",
        "gop_two_party_share",
        "house_margin_dem",
        "previous_house_margin_dem",
        "house_margin_change_dem",
        "president_party_house_margin",
        "president_party_house_margin_change",
    ]

    national = national[
        output_columns
    ]

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    VALIDATION_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    national.to_csv(
        OUTPUT,
        index=False,
    )

    checks = []

    def check(
        name,
        passed,
    ):
        checks.append(
            {
                "check": name,
                "passed": bool(
                    passed
                ),
            }
        )

    check(
        "expected_25_cycles",
        len(national) == 25,
    )

    check(
        "cycles_unique",
        national[
            "cycle"
        ].is_unique,
    )

    check(
        "starts_1976",
        national[
            "cycle"
        ].min() == 1976,
    )

    check(
        "ends_2024",
        national[
            "cycle"
        ].max() == 2024,
    )

    check(
        "all_president_parties_present",
        national[
            "president_party"
        ].notna().all(),
    )

    check(
        "all_dem_votes_positive",
        (
            national[
                "dem_votes"
            ] > 0
        ).all(),
    )

    check(
        "all_gop_votes_positive",
        (
            national[
                "gop_votes"
            ] > 0
        ).all(),
    )

    check(
        "shares_sum_to_one",
        np.allclose(
            (
                national[
                    "dem_two_party_share"
                ]
                + national[
                    "gop_two_party_share"
                ]
            ),
            1.0,
            atol=1e-10,
        ),
    )

    check(
        "twelve_midterms",
        national[
            "election_type"
        ].eq(
            "midterm"
        ).sum() == 12,
    )

    check(
        "all_midterm_changes_present",
        national.loc[
            national[
                "election_type"
            ].eq(
                "midterm"
            ),
            "president_party_house_margin_change",
        ].notna().all(),
    )

    validation = pd.DataFrame(
        checks
    )

    validation.to_csv(
        VALIDATION_OUTPUT,
        index=False,
    )

    print()
    print("ALL ELECTIONS")
    print("-" * 120)

    print(
        national.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("MIDTERMS")
    print("-" * 120)

    midterms = national.loc[
        national[
            "election_type"
        ].eq(
            "midterm"
        )
    ]

    print(
        midterms[
            [
                "cycle",
                "president_party",
                "house_margin_dem",
                "house_margin_change_dem",
                "president_party_house_margin",
                "president_party_house_margin_change",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print()
    print("VALIDATION")
    print("-" * 120)

    print(
        validation.to_string(
            index=False
        )
    )

    if not validation[
        "passed"
    ].all():
        raise RuntimeError(
            "Historical House national outcome "
            "validation failed."
        )

    print()
    print(
        "Historical House national outcome "
        "validation PASSED."
    )

    print()
    print(
        "Wrote:",
        OUTPUT,
    )

    print(
        "Wrote:",
        VALIDATION_OUTPUT,
    )


if __name__ == "__main__":
    main()
