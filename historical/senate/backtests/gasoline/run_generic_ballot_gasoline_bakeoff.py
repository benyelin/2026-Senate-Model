from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]

GB_INPUT = (
    ROOT
    / "historical/senate/warehouse/processed/generic_ballot/"
    / "historical_midterm_generic_ballot.csv"
)

GAS_HOUSE_INPUT = (
    ROOT
    / "historical/senate/backtests/outputs/gasoline_environment/"
    / "gasoline_midterm_merged_dataset.csv"
)

SUMMARY_OUTPUT = (
    ROOT
    / "historical/senate/backtests/outputs/gasoline_environment/"
    / "generic_ballot_gasoline_bakeoff_summary.csv"
)

PREDICTIONS_OUTPUT = (
    ROOT
    / "historical/senate/backtests/outputs/gasoline_environment/"
    / "generic_ballot_gasoline_bakeoff_predictions.csv"
)

COEFFICIENT_OUTPUT = (
    ROOT
    / "historical/senate/backtests/outputs/gasoline_environment/"
    / "generic_ballot_gasoline_bakeoff_coefficients.csv"
)

VALIDATION_OUTPUT = (
    ROOT
    / "historical/senate/diagnostics/gasoline/"
    / "generic_ballot_gasoline_bakeoff_validation.csv"
)


OUTCOME = (
    "president_party_house_margin_change"
)

MODEL_SPECS = {
    "mean_only": [],
    "generic_ballot_only": [
        "president_party_generic_ballot",
    ],
    "generic_ballot_plus_real_gas_burden": [
        "president_party_generic_ballot",
        "real_jan_oct_vs_prior_year_pct",
    ],
    "generic_ballot_plus_october_real_gas": [
        "president_party_generic_ballot",
        "october_gas_price_real",
    ],
    "generic_ballot_plus_nominal_gas_burden": [
        "president_party_generic_ballot",
        "jan_oct_vs_prior_year_pct",
    ],
}


def fit_ols(
    train: pd.DataFrame,
    predictors: list[str],
    outcome: str,
):
    y = train[
        outcome
    ].to_numpy(
        dtype=float
    )

    if not predictors:
        beta = np.array(
            [
                np.mean(
                    y
                )
            ],
            dtype=float,
        )

        return {
            "intercept": beta[0],
            "coefficients": {},
        }

    x = train[
        predictors
    ].to_numpy(
        dtype=float
    )

    design = np.column_stack(
        [
            np.ones(
                len(
                    train
                )
            ),
            x,
        ]
    )

    beta, _, _, _ = np.linalg.lstsq(
        design,
        y,
        rcond=None,
    )

    return {
        "intercept": float(
            beta[0]
        ),
        "coefficients": {
            predictor: float(
                coefficient
            )
            for predictor, coefficient
            in zip(
                predictors,
                beta[1:],
            )
        },
    }


def predict_row(
    model,
    row: pd.Series,
    predictors: list[str],
):
    prediction = float(
        model[
            "intercept"
        ]
    )

    for predictor in predictors:
        prediction += (
            model[
                "coefficients"
            ][predictor]
            * float(
                row[
                    predictor
                ]
            )
        )

    return prediction


def mae(
    actual,
    predicted,
):
    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    return float(
        np.mean(
            np.abs(
                actual
                - predicted
            )
        )
    )


def rmse(
    actual,
    predicted,
):
    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    return float(
        np.sqrt(
            np.mean(
                (
                    actual
                    - predicted
                )
                ** 2
            )
        )
    )


def main():
    print("=" * 132)
    print("MIDTERM GENERIC BALLOT + GASOLINE INCREMENTAL-VALUE BAKEOFF")
    print("=" * 132)

    gb = pd.read_csv(
        GB_INPUT,
        low_memory=False,
    )

    data = pd.read_csv(
        GAS_HOUSE_INPUT,
        low_memory=False,
    )

    gb[
        "year"
    ] = pd.to_numeric(
        gb[
            "year"
        ],
        errors="coerce",
    )

    gb[
        "generic_ballot_margin_dem"
    ] = pd.to_numeric(
        gb[
            "generic_ballot_margin_dem"
        ],
        errors="coerce",
    )

    data[
        "cycle"
    ] = pd.to_numeric(
        data[
            "cycle"
        ],
        errors="coerce",
    )

    merged = data.merge(
        gb[
            [
                "year",
                "generic_ballot_margin_dem",
                "source",
                "source_type",
                "source_quality",
            ]
        ],
        left_on="cycle",
        right_on="year",
        how="left",
        validate="one_to_one",
    )

    merged = merged.loc[
        merged[
            "generic_ballot_margin_dem"
        ].notna()
    ].copy()

    merged = merged.sort_values(
        "cycle"
    ).reset_index(
        drop=True
    )

    # Orient the generic ballot toward the
    # sitting president's party.
    #
    # Democratic president:
    #     D+5 -> +5
    #
    # Republican president:
    #     D+5 -> -5
    #
    merged[
        "president_party_generic_ballot"
    ] = np.where(
        merged[
            "president_party"
        ].eq(
            "D"
        ),
        merged[
            "generic_ballot_margin_dem"
        ],
        -merged[
            "generic_ballot_margin_dem"
        ],
    )

    required_numeric = [
        OUTCOME,
        "president_party_generic_ballot",
        "real_jan_oct_vs_prior_year_pct",
        "october_gas_price_real",
        "jan_oct_vs_prior_year_pct",
    ]

    for column in required_numeric:
        merged[
            column
        ] = pd.to_numeric(
            merged[
                column
            ],
            errors="coerce",
        )

    if merged[
        required_numeric
    ].isna().any().any():
        bad = merged.loc[
            merged[
                required_numeric
            ].isna().any(
                axis=1
            )
        ]

        print()
        print("ROWS WITH MISSING REQUIRED DATA")
        print("-" * 132)

        print(
            bad[
                [
                    "cycle",
                    "president_party",
                ]
                + required_numeric
            ].to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Missing data remain in bakeoff sample."
        )

    print()
    print("ANALYSIS SAMPLE")
    print("-" * 132)

    print(
        merged[
            [
                "cycle",
                "president_party",
                "generic_ballot_margin_dem",
                "president_party_generic_ballot",
                "real_jan_oct_vs_prior_year_pct",
                "october_gas_price_real",
                OUTCOME,
                "source",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.3f}"
            ),
        )
    )

    prediction_rows = []
    coefficient_rows = []

    # --------------------------------------------------
    # Leave-one-midterm-out prediction.
    #
    # Every held-out election is predicted using
    # coefficients estimated only from the other
    # elections.
    # --------------------------------------------------

    for held_out_cycle in merged[
        "cycle"
    ].tolist():

        train = merged.loc[
            ~merged[
                "cycle"
            ].eq(
                held_out_cycle
            )
        ].copy()

        test = merged.loc[
            merged[
                "cycle"
            ].eq(
                held_out_cycle
            )
        ].iloc[
            0
        ]

        for (
            model_name,
            predictors,
        ) in MODEL_SPECS.items():

            model = fit_ols(
                train,
                predictors,
                OUTCOME,
            )

            prediction = predict_row(
                model,
                test,
                predictors,
            )

            actual = float(
                test[
                    OUTCOME
                ]
            )

            prediction_rows.append(
                {
                    "held_out_cycle": int(
                        held_out_cycle
                    ),
                    "model": model_name,
                    "actual_president_party_house_swing": (
                        actual
                    ),
                    "predicted_president_party_house_swing": (
                        prediction
                    ),
                    "error": (
                        prediction
                        - actual
                    ),
                    "absolute_error": abs(
                        prediction
                        - actual
                    ),
                    "squared_error": (
                        prediction
                        - actual
                    )
                    ** 2,
                    "president_party_generic_ballot": float(
                        test[
                            "president_party_generic_ballot"
                        ]
                    ),
                    "real_gas_burden": float(
                        test[
                            "real_jan_oct_vs_prior_year_pct"
                        ]
                    ),
                    "october_real_gas": float(
                        test[
                            "october_gas_price_real"
                        ]
                    ),
                    "nominal_gas_burden": float(
                        test[
                            "jan_oct_vs_prior_year_pct"
                        ]
                    ),
                }
            )

            coefficient_row = {
                "held_out_cycle": int(
                    held_out_cycle
                ),
                "model": model_name,
                "intercept": (
                    model[
                        "intercept"
                    ]
                ),
            }

            for predictor in [
                "president_party_generic_ballot",
                "real_jan_oct_vs_prior_year_pct",
                "october_gas_price_real",
                "jan_oct_vs_prior_year_pct",
            ]:
                coefficient_row[
                    f"coef_{predictor}"
                ] = (
                    model[
                        "coefficients"
                    ].get(
                        predictor,
                        np.nan,
                    )
                )

            coefficient_rows.append(
                coefficient_row
            )

    predictions = pd.DataFrame(
        prediction_rows
    )

    coefficients = pd.DataFrame(
        coefficient_rows
    )

    summary_rows = []

    for model_name in (
        MODEL_SPECS.keys()
    ):
        model_predictions = (
            predictions.loc[
                predictions[
                    "model"
                ].eq(
                    model_name
                )
            ]
            .sort_values(
                "held_out_cycle"
            )
        )

        actual = model_predictions[
            "actual_president_party_house_swing"
        ]

        predicted = model_predictions[
            "predicted_president_party_house_swing"
        ]

        summary_rows.append(
            {
                "model": model_name,
                "n_midterms": len(
                    model_predictions
                ),
                "loo_mae": mae(
                    actual,
                    predicted,
                ),
                "loo_rmse": rmse(
                    actual,
                    predicted,
                ),
                "mean_error": float(
                    (
                        predicted
                        - actual
                    ).mean()
                ),
                "median_absolute_error": float(
                    model_predictions[
                        "absolute_error"
                    ].median()
                ),
                "max_absolute_error": float(
                    model_predictions[
                        "absolute_error"
                    ].max()
                ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    gb_row = summary.loc[
        summary[
            "model"
        ].eq(
            "generic_ballot_only"
        )
    ].iloc[
        0
    ]

    summary[
        "mae_change_vs_gb_only"
    ] = (
        summary[
            "loo_mae"
        ]
        - float(
            gb_row[
                "loo_mae"
            ]
        )
    )

    summary[
        "rmse_change_vs_gb_only"
    ] = (
        summary[
            "loo_rmse"
        ]
        - float(
            gb_row[
                "loo_rmse"
            ]
        )
    )

    summary[
        "mae_improvement_pct_vs_gb_only"
    ] = (
        -summary[
            "mae_change_vs_gb_only"
        ]
        / float(
            gb_row[
                "loo_mae"
            ]
        )
        * 100.0
    )

    summary[
        "rmse_improvement_pct_vs_gb_only"
    ] = (
        -summary[
            "rmse_change_vs_gb_only"
        ]
        / float(
            gb_row[
                "loo_rmse"
            ]
        )
        * 100.0
    )

    summary = summary.sort_values(
        [
            "loo_rmse",
            "loo_mae",
        ]
    ).reset_index(
        drop=True
    )

    print()
    print("LOO MODEL COMPARISON")
    print("-" * 132)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )

    # --------------------------------------------------
    # Direct held-out comparison:
    # on how many elections does each gas model
    # beat generic-ballot-only?
    # --------------------------------------------------

    gb_predictions = (
        predictions.loc[
            predictions[
                "model"
            ].eq(
                "generic_ballot_only"
            ),
            [
                "held_out_cycle",
                "absolute_error",
            ],
        ]
        .rename(
            columns={
                "absolute_error": (
                    "gb_absolute_error"
                )
            }
        )
    )

    comparison_rows = []

    for model_name in [
        "generic_ballot_plus_real_gas_burden",
        "generic_ballot_plus_october_real_gas",
        "generic_ballot_plus_nominal_gas_burden",
    ]:

        candidate = (
            predictions.loc[
                predictions[
                    "model"
                ].eq(
                    model_name
                ),
                [
                    "held_out_cycle",
                    "absolute_error",
                ],
            ]
            .merge(
                gb_predictions,
                on="held_out_cycle",
                how="left",
                validate="one_to_one",
            )
        )

        candidate[
            "gas_model_better"
        ] = (
            candidate[
                "absolute_error"
            ]
            < candidate[
                "gb_absolute_error"
            ]
        )

        comparison_rows.append(
            {
                "model": model_name,
                "midterms_beating_gb_only": int(
                    candidate[
                        "gas_model_better"
                    ].sum()
                ),
                "midterms_total": len(
                    candidate
                ),
                "share_beating_gb_only": float(
                    candidate[
                        "gas_model_better"
                    ].mean()
                ),
                "mean_absolute_error_difference": float(
                    (
                        candidate[
                            "absolute_error"
                        ]
                        - candidate[
                            "gb_absolute_error"
                        ]
                    ).mean()
                ),
            }
        )

    comparison = pd.DataFrame(
        comparison_rows
    )

    print()
    print("HEAD-TO-HEAD AGAINST GENERIC BALLOT ONLY")
    print("-" * 132)

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )

    # --------------------------------------------------
    # Full-sample coefficients are descriptive only.
    # They are NOT used for assessing predictive value.
    # --------------------------------------------------

    print()
    print("FULL-SAMPLE DESCRIPTIVE COEFFICIENTS")
    print("-" * 132)

    for (
        model_name,
        predictors,
    ) in MODEL_SPECS.items():

        if not predictors:
            continue

        model = fit_ols(
            merged,
            predictors,
            OUTCOME,
        )

        pieces = [
            (
                f"intercept="
                f"{model['intercept']:.4f}"
            )
        ]

        for predictor in predictors:
            pieces.append(
                (
                    f"{predictor}="
                    f"{model['coefficients'][predictor]:.4f}"
                )
            )

        print(
            model_name,
            " | ",
            " | ".join(
                pieces
            ),
        )

    # --------------------------------------------------
    # Detailed held-out errors.
    # --------------------------------------------------

    pivot = predictions.pivot(
        index="held_out_cycle",
        columns="model",
        values="absolute_error",
    ).reset_index()

    print()
    print("HELD-OUT ABSOLUTE ERRORS BY MIDTERM")
    print("-" * 132)

    print(
        pivot.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.3f}"
            ),
        )
    )

    SUMMARY_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    VALIDATION_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    predictions.to_csv(
        PREDICTIONS_OUTPUT,
        index=False,
    )

    coefficients.to_csv(
        COEFFICIENT_OUTPUT,
        index=False,
    )

    validation_rows = [
        {
            "check": "exactly_eleven_scorable_midterms",
            "passed": (
                len(
                    merged
                )
                == 11
            ),
        },
        {
            "check": "1986_excluded",
            "passed": (
                1986
                not in merged[
                    "cycle"
                ].tolist()
            ),
        },
        {
            "check": "cycles_unique",
            "passed": (
                merged[
                    "cycle"
                ].is_unique
            ),
        },
        {
            "check": "all_required_values_present",
            "passed": (
                merged[
                    required_numeric
                ].notna().all().all()
            ),
        },
        {
            "check": "all_models_have_eleven_predictions",
            "passed": all(
                predictions.loc[
                    predictions[
                        "model"
                    ].eq(
                        model_name
                    )
                ].shape[0]
                == 11
                for model_name
                in MODEL_SPECS
            ),
        },
        {
            "check": "all_predictions_finite",
            "passed": bool(
                np.isfinite(
                    predictions[
                        "predicted_president_party_house_swing"
                    ]
                ).all()
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
    print("-" * 132)

    print(
        validation.to_string(
            index=False
        )
    )

    if not validation[
        "passed"
    ].all():
        raise RuntimeError(
            "Generic ballot + gasoline bakeoff "
            "validation failed."
        )

    print()
    print(
        "Generic ballot + gasoline bakeoff validation PASSED."
    )

    print()
    print("Wrote:", SUMMARY_OUTPUT)
    print("Wrote:", PREDICTIONS_OUTPUT)
    print("Wrote:", COEFFICIENT_OUTPUT)
    print("Wrote:", VALIDATION_OUTPUT)


if __name__ == "__main__":
    main()
