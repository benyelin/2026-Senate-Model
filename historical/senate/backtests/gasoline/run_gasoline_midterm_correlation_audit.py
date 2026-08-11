from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]

GAS_INPUT = (
    ROOT
    / "historical/senate/warehouse/processed/economic/"
    / "historical_us_gasoline_election_snapshots_1976_2024.csv"
)

HOUSE_INPUT = (
    ROOT
    / "historical/senate/warehouse/processed/economic/"
    / "historical_house_national_outcomes_1976_2024.csv"
)

MERGED_OUTPUT = (
    ROOT
    / "historical/senate/backtests/outputs/gasoline_environment/"
    / "gasoline_midterm_merged_dataset.csv"
)

CORRELATION_OUTPUT = (
    ROOT
    / "historical/senate/backtests/outputs/gasoline_environment/"
    / "gasoline_midterm_correlations.csv"
)

VALIDATION_OUTPUT = (
    ROOT
    / "historical/senate/diagnostics/gasoline/"
    / "gasoline_midterm_correlation_validation.csv"
)


# These metrics are intentionally declared here
# before examining their electoral correlations.
GAS_METRICS = [
    (
        "october_gas_price_real",
        "October real gasoline price",
    ),
    (
        "gas_change_3m_pct",
        "3-month gasoline change",
    ),
    (
        "gas_change_6m_pct",
        "6-month gasoline change",
    ),
    (
        "gas_change_12m_pct",
        "12-month gasoline change",
    ),
    (
        "real_gas_change_12m_pct",
        "12-month real gasoline change",
    ),
    (
        "gas_change_jan_to_oct_pct",
        "January-to-October gasoline change",
    ),
    (
        "jan_oct_vs_prior_year_pct",
        "Jan-Oct average vs prior year",
    ),
    (
        "real_jan_oct_vs_prior_year_pct",
        "Real Jan-Oct average vs prior year",
    ),
    (
        "peak_vs_prior_year_jan_oct_avg_pct",
        "YTD peak vs prior-year Jan-Oct average",
    ),
    (
        "october_change_from_ytd_peak_pct",
        "October change from YTD peak",
    ),
]


OUTCOMES = [
    (
        "president_party_house_margin_change",
        "President-party House swing",
        "PRIMARY",
    ),
    (
        "president_party_house_margin",
        "President-party House margin",
        "SECONDARY",
    ),
]


def pearson_correlation(
    x: pd.Series,
    y: pd.Series,
) -> float:
    frame = pd.concat(
        [
            x,
            y,
        ],
        axis=1,
    ).dropna()

    if len(frame) < 3:
        return np.nan

    return float(
        frame.iloc[:, 0].corr(
            frame.iloc[:, 1],
            method="pearson",
        )
    )


def spearman_correlation(
    x: pd.Series,
    y: pd.Series,
) -> float:
    frame = pd.concat(
        [
            x,
            y,
        ],
        axis=1,
    ).dropna()

    if len(frame) < 3:
        return np.nan

    x_rank = (
        frame.iloc[:, 0]
        .rank(
            method="average"
        )
    )

    y_rank = (
        frame.iloc[:, 1]
        .rank(
            method="average"
        )
    )

    return float(
        x_rank.corr(
            y_rank,
            method="pearson",
        )
    )


def qualitative_strength(
    correlation: float,
) -> str:
    if pd.isna(
        correlation
    ):
        return "NA"

    magnitude = abs(
        float(
            correlation
        )
    )

    if magnitude >= 0.70:
        return "very strong"

    if magnitude >= 0.50:
        return "strong"

    if magnitude >= 0.30:
        return "moderate"

    if magnitude >= 0.10:
        return "weak"

    return "very weak"


def expected_direction_label(
    metric: str,
    correlation: float,
) -> str:
    """
    Most gas metrics are coded so larger values
    mean higher prices or larger increases.

    For October change from YTD peak, more-negative
    values mean more relief from the peak. Therefore
    its intuitive expected sign is reversed.
    """

    if pd.isna(
        correlation
    ):
        return "NA"

    if metric == (
        "october_change_from_ytd_peak_pct"
    ):
        # More positive = less decline / more persistent
        # high prices, so negative electoral correlation
        # is still the intuitive direction.
        pass

    if correlation < 0:
        return (
            "higher/rising gas associated "
            "with worse president-party performance"
        )

    if correlation > 0:
        return (
            "higher/rising gas associated "
            "with better president-party performance"
        )

    return "no linear relationship"


def main():
    print("=" * 126)
    print("GASOLINE AND MIDTERM ELECTIONS — FIRST CORRELATION AUDIT")
    print("=" * 126)

    gas = pd.read_csv(
        GAS_INPUT,
        low_memory=False,
    )

    house = pd.read_csv(
        HOUSE_INPUT,
        low_memory=False,
    )

    for frame in [
        gas,
        house,
    ]:
        frame[
            "cycle"
        ] = pd.to_numeric(
            frame[
                "cycle"
            ],
            errors="coerce",
        )

    merged = gas.merge(
        house,
        on=[
            "cycle",
            "election_type",
        ],
        how="inner",
        validate="one_to_one",
        suffixes=(
            "_gas",
            "_house",
        ),
    )

    midterms = merged.loc[
        merged[
            "election_type"
        ].eq(
            "midterm"
        )
    ].copy()

    midterms = midterms.sort_values(
        "cycle"
    ).reset_index(
        drop=True
    )

    numeric_columns = (
        [
            metric
            for metric, _ in GAS_METRICS
        ]
        + [
            outcome
            for outcome, _, _ in OUTCOMES
        ]
    )

    for column in numeric_columns:
        midterms[
            column
        ] = pd.to_numeric(
            midterms[
                column
            ],
            errors="coerce",
        )

    display_columns = [
        "cycle",
        "president_party",
        "president_party_house_margin",
        "president_party_house_margin_change",
        "october_gas_price_real",
        "gas_change_3m_pct",
        "gas_change_6m_pct",
        "gas_change_12m_pct",
        "gas_change_jan_to_oct_pct",
        "jan_oct_vs_prior_year_pct",
        "october_change_from_ytd_peak_pct",
    ]

    print()
    print("MIDTERM MASTER TABLE")
    print("-" * 126)

    print(
        midterms[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.3f}"
            ),
        )
    )

    correlation_rows = []

    for (
        outcome,
        outcome_label,
        outcome_priority,
    ) in OUTCOMES:

        for (
            metric,
            metric_label,
        ) in GAS_METRICS:

            pair = midterms[
                [
                    metric,
                    outcome,
                ]
            ].dropna()

            pearson = (
                pearson_correlation(
                    pair[
                        metric
                    ],
                    pair[
                        outcome
                    ],
                )
            )

            spearman = (
                spearman_correlation(
                    pair[
                        metric
                    ],
                    pair[
                        outcome
                    ],
                )
            )

            correlation_rows.append(
                {
                    "outcome": outcome,
                    "outcome_label": (
                        outcome_label
                    ),
                    "outcome_priority": (
                        outcome_priority
                    ),
                    "gas_metric": metric,
                    "gas_metric_label": (
                        metric_label
                    ),
                    "n_midterms": len(
                        pair
                    ),
                    "pearson_r": pearson,
                    "spearman_rho": (
                        spearman
                    ),
                    "pearson_abs": abs(
                        pearson
                    ),
                    "spearman_abs": abs(
                        spearman
                    ),
                    "pearson_strength": (
                        qualitative_strength(
                            pearson
                        )
                    ),
                    "spearman_strength": (
                        qualitative_strength(
                            spearman
                        )
                    ),
                    "pearson_direction": (
                        expected_direction_label(
                            metric,
                            pearson,
                        )
                    ),
                }
            )

    correlations = pd.DataFrame(
        correlation_rows
    )

    correlations[
        "mean_abs_correlation"
    ] = (
        correlations[
            [
                "pearson_abs",
                "spearman_abs",
            ]
        ]
        .mean(
            axis=1
        )
    )

    correlations[
        "sign_agreement"
    ] = (
        np.sign(
            correlations[
                "pearson_r"
            ]
        )
        == np.sign(
            correlations[
                "spearman_rho"
            ]
        )
    )

    correlations = correlations.sort_values(
        [
            "outcome_priority",
            "mean_abs_correlation",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(
        drop=True
    )

    print()
    print("PRIMARY OUTCOME: PRESIDENT-PARTY HOUSE SWING")
    print("-" * 126)

    primary = correlations.loc[
        correlations[
            "outcome_priority"
        ].eq(
            "PRIMARY"
        )
    ].copy()

    print(
        primary[
            [
                "gas_metric_label",
                "n_midterms",
                "pearson_r",
                "spearman_rho",
                "sign_agreement",
                "mean_abs_correlation",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.3f}"
            ),
        )
    )

    print()
    print("SECONDARY OUTCOME: PRESIDENT-PARTY HOUSE MARGIN")
    print("-" * 126)

    secondary = correlations.loc[
        correlations[
            "outcome_priority"
        ].eq(
            "SECONDARY"
        )
    ].copy()

    print(
        secondary[
            [
                "gas_metric_label",
                "n_midterms",
                "pearson_r",
                "spearman_rho",
                "sign_agreement",
                "mean_abs_correlation",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.3f}"
            ),
        )
    )

    # --------------------------------------------------
    # Basic robustness diagnostics:
    # leave-one-midterm-out correlation stability.
    # No coefficients are being fitted yet.
    # --------------------------------------------------

    loo_rows = []

    for (
        metric,
        metric_label,
    ) in GAS_METRICS:

        full = midterms[
            [
                "cycle",
                metric,
                (
                    "president_party_"
                    "house_margin_change"
                ),
            ]
        ].dropna()

        full_r = pearson_correlation(
            full[
                metric
            ],
            full[
                "president_party_house_margin_change"
            ],
        )

        fold_values = []

        for held_out_cycle in (
            full[
                "cycle"
            ].tolist()
        ):
            train = full.loc[
                ~full[
                    "cycle"
                ].eq(
                    held_out_cycle
                )
            ]

            fold_r = pearson_correlation(
                train[
                    metric
                ],
                train[
                    "president_party_house_margin_change"
                ],
            )

            fold_values.append(
                fold_r
            )

        fold_values = np.array(
            fold_values,
            dtype=float,
        )

        loo_rows.append(
            {
                "gas_metric": metric,
                "gas_metric_label": (
                    metric_label
                ),
                "full_pearson_r": (
                    full_r
                ),
                "loo_min_r": float(
                    np.nanmin(
                        fold_values
                    )
                ),
                "loo_max_r": float(
                    np.nanmax(
                        fold_values
                    )
                ),
                "loo_mean_r": float(
                    np.nanmean(
                        fold_values
                    )
                ),
                "loo_sign_stable": bool(
                    np.all(
                        np.sign(
                            fold_values
                        )
                        == np.sign(
                            full_r
                        )
                    )
                ),
            }
        )

    loo = pd.DataFrame(
        loo_rows
    ).sort_values(
        "full_pearson_r"
    )

    print()
    print("LEAVE-ONE-MIDTERM-OUT SIGN STABILITY — PRIMARY OUTCOME")
    print("-" * 126)

    print(
        loo.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.3f}"
            ),
        )
    )

    MERGED_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    VALIDATION_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    midterms.to_csv(
        MERGED_OUTPUT,
        index=False,
    )

    correlations.to_csv(
        CORRELATION_OUTPUT,
        index=False,
    )

    validation_rows = [
        {
            "check": (
                "exactly_twelve_midterms"
            ),
            "passed": (
                len(
                    midterms
                )
                == 12
            ),
        },
        {
            "check": (
                "midterm_cycles_unique"
            ),
            "passed": (
                midterms[
                    "cycle"
                ].is_unique
            ),
        },
        {
            "check": (
                "all_primary_outcomes_present"
            ),
            "passed": (
                midterms[
                    "president_party_house_margin_change"
                ]
                .notna()
                .all()
            ),
        },
        {
            "check": (
                "all_prespecified_gas_metrics_present"
            ),
            "passed": (
                midterms[
                    [
                        metric
                        for metric, _
                        in GAS_METRICS
                    ]
                ]
                .notna()
                .all()
                .all()
            ),
        },
        {
            "check": (
                "twenty_correlation_rows"
            ),
            "passed": (
                len(
                    correlations
                )
                == 20
            ),
        },
    ]

    validation = pd.DataFrame(
        validation_rows
    )

    validation.to_csv(
        VALIDATION_OUTPUT,
        index=False,
    )

    print()
    print("VALIDATION")
    print("-" * 126)

    print(
        validation.to_string(
            index=False
        )
    )

    if not validation[
        "passed"
    ].all():
        raise RuntimeError(
            "Gasoline correlation audit "
            "validation failed."
        )

    print()
    print(
        "Gasoline correlation audit validation PASSED."
    )

    print()
    print("Wrote:", MERGED_OUTPUT)
    print("Wrote:", CORRELATION_OUTPUT)
    print("Wrote:", VALIDATION_OUTPUT)


if __name__ == "__main__":
    main()
