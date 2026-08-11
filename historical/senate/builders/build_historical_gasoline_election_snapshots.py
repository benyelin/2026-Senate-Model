from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]

INPUT = (
    ROOT
    / "historical/senate/warehouse/processed/economic/"
    / "historical_us_gasoline_monthly_1976_present.csv"
)

OUTPUT = (
    ROOT
    / "historical/senate/warehouse/processed/economic/"
    / "historical_us_gasoline_election_snapshots_1976_2024.csv"
)

VALIDATION_OUTPUT = (
    ROOT
    / "historical/senate/diagnostics/gasoline/"
    / "historical_us_gasoline_election_snapshots_validation.csv"
)


def federal_election_date(
    year: int,
) -> date:
    """
    Federal general election:
    Tuesday after the first Monday in November.
    """

    current = date(
        year,
        11,
        1,
    )

    while current.weekday() != 0:
        current += timedelta(
            days=1
        )

    return current + timedelta(
        days=1
    )


def safe_pct_change(
    new_value,
    old_value,
):
    if (
        pd.isna(new_value)
        or pd.isna(old_value)
        or float(old_value) == 0.0
    ):
        return np.nan

    return (
        (
            float(new_value)
            / float(old_value)
        )
        - 1.0
    ) * 100.0


def main():
    print("=" * 120)
    print("HISTORICAL GASOLINE — FEDERAL ELECTION SNAPSHOTS")
    print("=" * 120)

    df = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    numeric_columns = [
        "gas_price_nominal",
        "cpi",
        "gas_price_real_latest_dollars",
        "gas_change_3m_pct",
        "gas_change_6m_pct",
        "gas_change_12m_pct",
        "real_gas_change_12m_pct",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df["year"] = (
        df["date"]
        .dt.year
    )

    df["month"] = (
        df["date"]
        .dt.month
    )

    rows = []

    for year in range(
        1976,
        2025,
        2,
    ):
        election_date = (
            federal_election_date(
                year
            )
        )

        election_type = (
            "presidential"
            if year % 4 == 0
            else "midterm"
        )

        october_date = pd.Timestamp(
            year=year,
            month=10,
            day=1,
        )

        october = df.loc[
            df["date"].eq(
                october_date
            )
        ]

        if len(october) != 1:
            raise RuntimeError(
                f"{year}: expected exactly one "
                "October gasoline observation; "
                f"found {len(october)}."
            )

        october = october.iloc[0]

        january = df.loc[
            df["date"].eq(
                pd.Timestamp(
                    year=year,
                    month=1,
                    day=1,
                )
            )
        ]

        if len(january) != 1:
            raise RuntimeError(
                f"{year}: January observation missing."
            )

        january = january.iloc[0]

        current_jan_oct = df.loc[
            df["year"].eq(year)
            & df["month"].between(
                1,
                10,
            )
        ].copy()

        prior_jan_oct = df.loc[
            df["year"].eq(
                year - 1
            )
            & df["month"].between(
                1,
                10,
            )
        ].copy()

        current_avg = (
            current_jan_oct[
                "gas_price_nominal"
            ].mean()
        )

        prior_avg = (
            prior_jan_oct[
                "gas_price_nominal"
            ].mean()
            if not prior_jan_oct.empty
            else np.nan
        )

        current_real_avg = (
            current_jan_oct[
                "gas_price_real_latest_dollars"
            ].mean()
        )

        prior_real_avg = (
            prior_jan_oct[
                "gas_price_real_latest_dollars"
            ].mean()
            if not prior_jan_oct.empty
            else np.nan
        )

        current_peak = (
            current_jan_oct[
                "gas_price_nominal"
            ].max()
        )

        current_peak_month = (
            current_jan_oct.loc[
                current_jan_oct[
                    "gas_price_nominal"
                ].idxmax(),
                "date",
            ]
            if not current_jan_oct.empty
            else pd.NaT
        )

        row = {
            "cycle": year,
            "election_date": (
                election_date.isoformat()
            ),
            "election_type": (
                election_type
            ),

            # Price level as of the final
            # complete pre-election month.
            "october_gas_price_nominal": (
                october[
                    "gas_price_nominal"
                ]
            ),
            "october_gas_price_real": (
                october[
                    "gas_price_real_latest_dollars"
                ]
            ),

            # Prespecified recent trajectory.
            "gas_change_3m_pct": (
                october[
                    "gas_change_3m_pct"
                ]
            ),
            "gas_change_6m_pct": (
                october[
                    "gas_change_6m_pct"
                ]
            ),
            "gas_change_12m_pct": (
                october[
                    "gas_change_12m_pct"
                ]
            ),
            "real_gas_change_12m_pct": (
                october[
                    "real_gas_change_12m_pct"
                ]
            ),

            # Explicit January-to-October
            # election-year movement.
            "gas_change_jan_to_oct_pct": (
                safe_pct_change(
                    october[
                        "gas_price_nominal"
                    ],
                    january[
                        "gas_price_nominal"
                    ],
                )
            ),

            # Compare equivalent Jan-Oct periods,
            # never full election-year averages.
            "jan_oct_avg_gas": (
                current_avg
            ),
            "prior_year_jan_oct_avg_gas": (
                prior_avg
            ),
            "jan_oct_vs_prior_year_pct": (
                safe_pct_change(
                    current_avg,
                    prior_avg,
                )
            ),

            "jan_oct_avg_real_gas": (
                current_real_avg
            ),
            "prior_year_jan_oct_avg_real_gas": (
                prior_real_avg
            ),
            "real_jan_oct_vs_prior_year_pct": (
                safe_pct_change(
                    current_real_avg,
                    prior_real_avg,
                )
            ),

            # Salience / shock candidates.
            "jan_oct_peak_gas": (
                current_peak
            ),
            "jan_oct_peak_month": (
                current_peak_month.date().isoformat()
                if pd.notna(
                    current_peak_month
                )
                else ""
            ),
            "peak_vs_prior_year_jan_oct_avg_pct": (
                safe_pct_change(
                    current_peak,
                    prior_avg,
                )
            ),
            "october_change_from_ytd_peak_pct": (
                safe_pct_change(
                    october[
                        "gas_price_nominal"
                    ],
                    current_peak,
                )
            ),
        }

        rows.append(
            row
        )

    snapshots = pd.DataFrame(
        rows
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    VALIDATION_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    snapshots.to_csv(
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
        "expected_25_election_cycles",
        len(
            snapshots
        ) == 25,
    )

    check(
        "unique_cycles",
        snapshots[
            "cycle"
        ].is_unique,
    )

    check(
        "starts_1976",
        snapshots[
            "cycle"
        ].min()
        == 1976,
    )

    check(
        "ends_2024",
        snapshots[
            "cycle"
        ].max()
        == 2024,
    )

    check(
        "twelve_midterms",
        (
            snapshots[
                "election_type"
            ]
            .eq(
                "midterm"
            )
            .sum()
            == 12
        ),
    )

    check(
        "thirteen_presidential_cycles",
        (
            snapshots[
                "election_type"
            ]
            .eq(
                "presidential"
            )
            .sum()
            == 13
        ),
    )

    check(
        "all_october_prices_present",
        snapshots[
            "october_gas_price_nominal"
        ].notna().all(),
    )

    check(
        "all_october_real_prices_present",
        snapshots[
            "october_gas_price_real"
        ].notna().all(),
    )

    midterms = snapshots.loc[
        snapshots[
            "election_type"
        ].eq(
            "midterm"
        )
    ]

    gas_metric_columns = [
        "gas_change_3m_pct",
        "gas_change_6m_pct",
        "gas_change_12m_pct",
        "real_gas_change_12m_pct",
        "gas_change_jan_to_oct_pct",
        "jan_oct_vs_prior_year_pct",
        "real_jan_oct_vs_prior_year_pct",
        "peak_vs_prior_year_jan_oct_avg_pct",
        "october_change_from_ytd_peak_pct",
    ]

    check(
        "all_midterm_change_metrics_present",
        midterms[
            gas_metric_columns
        ].notna().all().all(),
    )

    validation = pd.DataFrame(
        checks
    )

    validation.to_csv(
        VALIDATION_OUTPUT,
        index=False,
    )

    print()
    print("ELECTION SNAPSHOTS")
    print("-" * 120)

    display_columns = [
        "cycle",
        "election_type",
        "october_gas_price_nominal",
        "october_gas_price_real",
        "gas_change_3m_pct",
        "gas_change_6m_pct",
        "gas_change_12m_pct",
        "gas_change_jan_to_oct_pct",
        "jan_oct_vs_prior_year_pct",
        "october_change_from_ytd_peak_pct",
    ]

    print(
        snapshots[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print("MIDTERMS ONLY")
    print("-" * 120)

    print(
        snapshots.loc[
            snapshots[
                "election_type"
            ].eq(
                "midterm"
            ),
            display_columns,
        ].to_string(
            index=False
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
            "Gasoline election snapshot "
            "validation failed."
        )

    print()
    print(
        "Gasoline election snapshot "
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
