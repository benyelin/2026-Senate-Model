from __future__ import annotations

from io import StringIO
from pathlib import Path
from urllib.request import urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]

RAW_DIR = (
    ROOT
    / "historical/senate/warehouse/raw/economic/gasoline"
)

PROCESSED_DIR = (
    ROOT
    / "historical/senate/warehouse/processed/economic"
)

VALIDATION_DIR = (
    ROOT
    / "historical/senate/diagnostics/gasoline"
)

RAW_GAS = (
    RAW_DIR
    / "fred_APU000074714_monthly_gasoline.csv"
)

RAW_CPI = (
    RAW_DIR
    / "fred_CPIAUCNS_monthly_cpi.csv"
)

OUTPUT = (
    PROCESSED_DIR
    / "historical_us_gasoline_monthly_1976_present.csv"
)

VALIDATION_OUTPUT = (
    VALIDATION_DIR
    / "historical_us_gasoline_monthly_validation.csv"
)

GAS_URL = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv"
    "?id=APU000074714"
)

CPI_URL = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv"
    "?id=CPIAUCNS"
)


def download_csv(url: str, output: Path) -> pd.DataFrame:
    import subprocess

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    subprocess.run(
        [
            "curl",
            "-L",
            "--fail",
            "--silent",
            "--show-error",
            url,
            "-o",
            str(output),
        ],
        check=True,
    )

    return pd.read_csv(
        output,
        low_memory=False,
    )


def main():
    print("=" * 110)
    print("HISTORICAL U.S. GASOLINE DATASET")
    print("=" * 110)

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    gas = download_csv(
        GAS_URL,
        RAW_GAS,
    )

    cpi = download_csv(
        CPI_URL,
        RAW_CPI,
    )

    gas = gas.rename(
        columns={
            "DATE": "date",
            "observation_date": "date",
            "APU000074714": "gas_price_nominal",
        }
    )

    cpi = cpi.rename(
        columns={
            "DATE": "date",
            "observation_date": "date",
            "CPIAUCNS": "cpi",
        }
    )

    gas["date"] = pd.to_datetime(
        gas["date"],
        errors="coerce",
    )

    cpi["date"] = pd.to_datetime(
        cpi["date"],
        errors="coerce",
    )

    gas["gas_price_nominal"] = pd.to_numeric(
        gas["gas_price_nominal"],
        errors="coerce",
    )

    cpi["cpi"] = pd.to_numeric(
        cpi["cpi"],
        errors="coerce",
    )

    df = gas.merge(
        cpi,
        on="date",
        how="left",
        validate="one_to_one",
    )

    df = (
        df.loc[
            df["date"]
            >= pd.Timestamp("1976-01-01")
        ]
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Normalize real gasoline prices to the latest
    # available CPI observation. The particular
    # reference month affects the scale, not the
    # underlying relationship we will test.
    base_cpi = float(
        df["cpi"]
        .dropna()
        .iloc[-1]
    )

    df["gas_price_real_latest_dollars"] = (
        df["gas_price_nominal"]
        * base_cpi
        / df["cpi"]
    )

    # Prespecified gasoline-change measures.
    df["gas_change_3m_pct"] = (
        df["gas_price_nominal"]
        .pct_change(3)
        * 100.0
    )

    df["gas_change_6m_pct"] = (
        df["gas_price_nominal"]
        .pct_change(6)
        * 100.0
    )

    df["gas_change_12m_pct"] = (
        df["gas_price_nominal"]
        .pct_change(12)
        * 100.0
    )

    df["real_gas_change_12m_pct"] = (
        df["gas_price_real_latest_dollars"]
        .pct_change(12)
        * 100.0
    )

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    annual = (
        df.groupby(
            "year",
            as_index=False,
        )
        .agg(
            annual_avg_gas=(
                "gas_price_nominal",
                "mean",
            ),
            annual_avg_real_gas=(
                "gas_price_real_latest_dollars",
                "mean",
            ),
        )
    )

    annual["prior_year_avg_gas"] = (
        annual["annual_avg_gas"]
        .shift(1)
    )

    annual["annual_vs_prior_pct"] = (
        (
            annual["annual_avg_gas"]
            / annual["prior_year_avg_gas"]
        )
        - 1.0
    ) * 100.0

    df = df.merge(
        annual,
        on="year",
        how="left",
        validate="many_to_one",
    )

    df.to_csv(
        OUTPUT,
        index=False,
    )

    checks = [
        {
            "check": "rows_present",
            "passed": len(df) > 500,
        },
        {
            "check": "starts_in_1976",
            "passed": (
                df["date"].min()
                == pd.Timestamp("1976-01-01")
            ),
        },
        {
            "check": "gas_prices_positive",
            "passed": bool(
                (
                    df["gas_price_nominal"]
                    .dropna()
                    > 0
                ).all()
            ),
        },
        {
            "check": "cpi_positive",
            "passed": bool(
                (
                    df["cpi"]
                    .dropna()
                    > 0
                ).all()
            ),
        },
        {
            "check": "real_prices_positive",
            "passed": bool(
                (
                    df["gas_price_real_latest_dollars"]
                    .dropna()
                    > 0
                ).all()
            ),
        },
    ]

    validation = pd.DataFrame(checks)

    validation.to_csv(
        VALIDATION_OUTPUT,
        index=False,
    )

    print()
    print("Rows:", len(df))
    print(
        "Date range:",
        df["date"].min().date(),
        "to",
        df["date"].max().date(),
    )

    print()
    print("FIRST FIVE")
    print("-" * 110)
    print(
        df.head(5).to_string(
            index=False
        )
    )

    print()
    print("LATEST FIVE")
    print("-" * 110)
    print(
        df.tail(5).to_string(
            index=False
        )
    )

    print()
    print("VALIDATION")
    print("-" * 110)
    print(
        validation.to_string(
            index=False
        )
    )

    if not validation["passed"].all():
        raise RuntimeError(
            "Gasoline dataset validation failed."
        )

    print()
    print(
        "Historical gasoline dataset validation PASSED."
    )
    print()
    print("Wrote:", OUTPUT)
    print("Wrote:", VALIDATION_OUTPUT)


if __name__ == "__main__":
    main()
