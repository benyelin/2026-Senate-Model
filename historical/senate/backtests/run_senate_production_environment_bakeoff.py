from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_PATH = (
    PROJECT_ROOT
    / "historical/senate/warehouse/processed"
    / "senate_historical_fundamentals_2012_2024.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "historical/senate/backtests/outputs"
    / "production_environment_bakeoff"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "senate_production_environment_bakeoff_summary.csv"
)

CYCLE_PATH = (
    OUTPUT_DIR
    / "senate_production_environment_bakeoff_by_cycle.csv"
)

PREDICTIONS_PATH = (
    OUTPUT_DIR
    / "senate_production_environment_bakeoff_predictions.csv"
)

COEFFICIENTS_PATH = (
    OUTPUT_DIR
    / "senate_production_environment_bakeoff_fold_coefficients.csv"
)

VALIDATION_PATH = (
    OUTPUT_DIR
    / "senate_production_environment_bakeoff_validation.txt"
)

CONFIG_PATH = (
    OUTPUT_DIR
    / "senate_production_environment_bakeoff_config.json"
)


COLUMN_ALIASES = {
    "race_id": [
        "race_id",
    ],
    "cycle": [
        "cycle",
        "year",
        "election_year",
    ],
    "state": [
        "state",
        "state_abbreviation",
        "state_po",
    ],
    "actual_margin_dem": [
        "actual_margin_dem",
        "senate_margin_dem",
        "dem_margin",
        "result_margin_dem",
    ],
    "baseline_margin_dem": [
        "fundamentals_baseline_margin_dem",
        "baseline_margin_dem",
        "presidential_margin_dem",
        "pres_margin_dem",
    ],
    "generic_ballot_margin_dem": [
        "generic_ballot_margin_dem",
        "generic_ballot_dem_margin",
        "generic_margin_dem",
    ],
    "presidential_approval": [
        "presidential_approval",
        "president_approval",
        "approval",
    ],
    "president_party": [
        "president_party",
        "presidential_party",
        "party_of_president",
    ],
    "midterm_adjustment_dem": [
        "midterm_adjustment_dem",
        "midterm_effect_dem",
        "midterm_margin_dem",
        "midterm_indicator",
    ],
    "scorable": [
        "historical_fundamentals_scorable",
        "scorable",
        "is_scorable",
    ],
}


FORMULA_ORDER = [
    "current_production",
    "generic_0_85",
    "generic_1_00",
    "generic_1_190420_reference",
    "generic_loco_calibrated",
    "generic_plus_midterm_loco_calibrated",
]


FORMULA_LABELS = {
    "current_production": (
        "Current: 0.85 GB + 0.50 approval "
        "+ 0.50 midterm"
    ),
    "generic_0_85": (
        "0.85 × generic ballot"
    ),
    "generic_1_00": (
        "1.00 × generic ballot"
    ),
    "generic_1_190420_reference": (
        "1.190420 × generic ballot "
        "(full-sample reference)"
    ),
    "generic_loco_calibrated": (
        "Generic ballot with LOCO-calibrated slope"
    ),
    "generic_plus_midterm_loco_calibrated": (
        "Generic ballot + midterm with "
        "LOCO-calibrated slopes"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def find_column(
    frame: pd.DataFrame,
    logical_name: str,
    required: bool = True,
) -> str | None:
    aliases = COLUMN_ALIASES[logical_name]

    for column in aliases:
        if column in frame.columns:
            return column

    if required:
        raise ValueError(
            f"Could not locate {logical_name!r}.\n"
            f"Tried: {aliases}\n"
            "Available columns:\n"
            + ", ".join(frame.columns)
        )

    return None


def normalize_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    normalized = (
        series.astype("string")
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
        "y": True,
        "n": False,
    }

    unexpected = sorted(
        set(normalized.dropna().unique())
        - set(mapping)
    )

    if unexpected:
        raise ValueError(
            "Unexpected scorable values: "
            + ", ".join(map(str, unexpected))
        )

    return (
        normalized.map(mapping)
        .fillna(False)
        .astype(bool)
    )


def normalize_party(value: object) -> str:
    text = str(value).strip().upper()

    if text in {
        "D",
        "DEM",
        "DEMOCRAT",
        "DEMOCRATIC",
    }:
        return "D"

    if text in {
        "R",
        "REP",
        "REPUBLICAN",
        "GOP",
    }:
        return "R"

    raise ValueError(
        f"Unrecognized president party: {value!r}"
    )


def approval_adjustment_for_republican_president(
    approval: float,
) -> float:
    """
    Mirror the active production helper when importable.

    This fallback reflects the current July 2026 input:
    39.7% approval -> approximately D+1.7667.

    Formula:
        (45 - approval) / 3

    Positive values favor Democrats under a Republican
    president. The sign reverses under a Democratic
    president.
    """
    return (45.0 - float(approval)) / 3.0


def load_production_approval_function():
    try:
        from update_national_environment import (
            approval_adjustment_for_republican_president
            as production_function,
        )

        return (
            production_function,
            "imported from update_national_environment.py",
        )
    except Exception:
        return (
            approval_adjustment_for_republican_president,
            "local fallback: (45 - approval) / 3",
        )


def dem_approval_adjustment(
    approval: float,
    president_party: str,
    adjustment_function,
) -> float:
    republican_president_adjustment = float(
        adjustment_function(float(approval))
    )

    party = normalize_party(
        president_party
    )

    if party == "R":
        return republican_president_adjustment

    return -republican_president_adjustment


def sigmoid_probability(
    margin: np.ndarray,
    probability_scale: float,
) -> np.ndarray:
    scaled = np.clip(
        margin / probability_scale,
        -50.0,
        50.0,
    )

    return 1.0 / (
        1.0 + np.exp(-scaled)
    )


def calculate_metrics(
    frame: pd.DataFrame,
    probability_scale: float,
) -> dict:
    actual = frame[
        "actual_margin_dem"
    ].to_numpy(dtype=float)

    predicted = frame[
        "predicted_margin_dem"
    ].to_numpy(dtype=float)

    errors = predicted - actual

    probabilities = sigmoid_probability(
        predicted,
        probability_scale,
    )

    outcomes = (
        actual > 0.0
    ).astype(float)

    called_dem = predicted > 0.0
    actual_dem = actual > 0.0

    return {
        "races": len(frame),
        "mae": float(
            np.mean(
                np.abs(errors)
            )
        ),
        "rmse": float(
            np.sqrt(
                np.mean(
                    errors ** 2
                )
            )
        ),
        "mean_error_dem": float(
            np.mean(errors)
        ),
        "brier": float(
            np.mean(
                (
                    probabilities
                    - outcomes
                ) ** 2
            )
        ),
        "winner_accuracy": float(
            np.mean(
                called_dem
                == actual_dem
            )
        ),
    }


def fit_through_origin(
    training: pd.DataFrame,
    feature_columns: list[str],
) -> np.ndarray:
    design = training[
        feature_columns
    ].to_numpy(dtype=float)

    target = (
        training[
            "actual_margin_dem"
        ].to_numpy(dtype=float)
        - training[
            "baseline_margin_dem"
        ].to_numpy(dtype=float)
    )

    coefficients, *_ = np.linalg.lstsq(
        design,
        target,
        rcond=None,
    )

    return coefficients


def validate_finite(
    frame: pd.DataFrame,
    columns: list[str],
) -> None:
    for column in columns:
        values = frame[
            column
        ].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite values remain in {column}."
            )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--probability-scale",
        type=float,
        default=6.0,
        help=(
            "Margin-to-probability logistic scale "
            "used for Brier scoring."
        ),
    )

    args = parser.parse_args()

    if args.probability_scale <= 0.0:
        raise ValueError(
            "Probability scale must be positive."
        )

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            INPUT_PATH
        )

    input_hash = sha256_file(
        INPUT_PATH
    )

    raw = pd.read_csv(
        INPUT_PATH
    )

    race_id_column = find_column(
        raw,
        "race_id",
    )

    cycle_column = find_column(
        raw,
        "cycle",
    )

    state_column = find_column(
        raw,
        "state",
    )

    actual_column = find_column(
        raw,
        "actual_margin_dem",
    )

    baseline_column = find_column(
        raw,
        "baseline_margin_dem",
    )

    generic_column = find_column(
        raw,
        "generic_ballot_margin_dem",
    )

    approval_column = find_column(
        raw,
        "presidential_approval",
    )

    president_party_column = find_column(
        raw,
        "president_party",
    )

    midterm_column = find_column(
        raw,
        "midterm_adjustment_dem",
    )

    scorable_column = find_column(
        raw,
        "scorable",
        required=False,
    )

    selected_columns = {
        "race_id": race_id_column,
        "cycle": cycle_column,
        "state": state_column,
        "actual_margin_dem": actual_column,
        "baseline_margin_dem": baseline_column,
        "generic_ballot_margin_dem": (
            generic_column
        ),
        "presidential_approval": (
            approval_column
        ),
        "president_party": (
            president_party_column
        ),
        "midterm_adjustment_dem": (
            midterm_column
        ),
    }

    data = pd.DataFrame(
        {
            logical_name: raw[
                source_column
            ]
            for logical_name, source_column
            in selected_columns.items()
        }
    )

    if scorable_column is not None:
        data["scorable"] = normalize_boolean(
            raw[scorable_column]
        )
    else:
        data["scorable"] = True

    numeric_columns = [
        "cycle",
        "actual_margin_dem",
        "baseline_margin_dem",
        "generic_ballot_margin_dem",
        "presidential_approval",
        "midterm_adjustment_dem",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    required_model_columns = [
        "cycle",
        "actual_margin_dem",
        "baseline_margin_dem",
        "generic_ballot_margin_dem",
        "presidential_approval",
        "midterm_adjustment_dem",
        "president_party",
    ]

    model_data = data.loc[
        data["scorable"]
    ].dropna(
        subset=required_model_columns
    ).copy()

    model_data["cycle"] = (
        model_data["cycle"]
        .astype(int)
    )

    model_data["president_party"] = (
        model_data[
            "president_party"
        ].map(normalize_party)
    )

    # Orient the warehouse's neutral midterm indicator
    # toward the Democratic margin:
    #
    #   Republican president midterm -> +1
    #   Democratic president midterm -> -1
    #   Presidential election year   ->  0
    if midterm_column == "midterm_indicator":
        president_party_sign_dem = np.where(
            model_data["president_party"] == "R",
            1.0,
            -1.0,
        )

        model_data["midterm_adjustment_dem"] = (
            model_data["midterm_adjustment_dem"]
            * president_party_sign_dem
        )

    adjustment_function, adjustment_source = (
        load_production_approval_function()
    )

    model_data[
        "approval_adjustment_dem"
    ] = [
        dem_approval_adjustment(
            approval=approval,
            president_party=party,
            adjustment_function=(
                adjustment_function
            ),
        )
        for approval, party in zip(
            model_data[
                "presidential_approval"
            ],
            model_data[
                "president_party"
            ],
        )
    ]

    validate_finite(
        model_data,
        [
            "actual_margin_dem",
            "baseline_margin_dem",
            "generic_ballot_margin_dem",
            "approval_adjustment_dem",
            "midterm_adjustment_dem",
        ],
    )

    model_data = model_data.sort_values(
        [
            "cycle",
            "state",
            "race_id",
        ]
    ).reset_index(drop=True)

    cycles = sorted(
        model_data[
            "cycle"
        ].unique().tolist()
    )

    if len(cycles) < 3:
        raise ValueError(
            "Too few cycles for LOCO testing."
        )

    prediction_frames = []
    coefficient_rows = []

    for holdout_cycle in cycles:
        training = model_data.loc[
            model_data["cycle"]
            != holdout_cycle
        ].copy()

        holdout = model_data.loc[
            model_data["cycle"]
            == holdout_cycle
        ].copy()

        generic_coefficient = (
            fit_through_origin(
                training,
                [
                    "generic_ballot_margin_dem",
                ],
            )[0]
        )

        generic_midterm_coefficients = (
            fit_through_origin(
                training,
                [
                    "generic_ballot_margin_dem",
                    "midterm_adjustment_dem",
                ],
            )
        )

        generic_plus_midterm_coefficient = float(
            generic_midterm_coefficients[0]
        )

        midterm_coefficient = float(
            generic_midterm_coefficients[1]
        )

        coefficient_rows.extend(
            [
                {
                    "holdout_cycle": (
                        holdout_cycle
                    ),
                    "model": (
                        "generic_loco_calibrated"
                    ),
                    "generic_ballot_coefficient": (
                        float(
                            generic_coefficient
                        )
                    ),
                    "midterm_coefficient": (
                        0.0
                    ),
                    "training_races": len(
                        training
                    ),
                },
                {
                    "holdout_cycle": (
                        holdout_cycle
                    ),
                    "model": (
                        "generic_plus_midterm_"
                        "loco_calibrated"
                    ),
                    "generic_ballot_coefficient": (
                        generic_plus_midterm_coefficient
                    ),
                    "midterm_coefficient": (
                        midterm_coefficient
                    ),
                    "training_races": len(
                        training
                    ),
                },
            ]
        )

        environments = {
            "current_production": (
                0.85
                * holdout[
                    "generic_ballot_margin_dem"
                ]
                + 0.50
                * holdout[
                    "approval_adjustment_dem"
                ]
                + 0.50
                * holdout[
                    "midterm_adjustment_dem"
                ]
            ),
            "generic_0_85": (
                0.85
                * holdout[
                    "generic_ballot_margin_dem"
                ]
            ),
            "generic_1_00": (
                holdout[
                    "generic_ballot_margin_dem"
                ]
            ),
            "generic_1_190420_reference": (
                1.190420
                * holdout[
                    "generic_ballot_margin_dem"
                ]
            ),
            "generic_loco_calibrated": (
                generic_coefficient
                * holdout[
                    "generic_ballot_margin_dem"
                ]
            ),
            (
                "generic_plus_midterm_"
                "loco_calibrated"
            ): (
                generic_plus_midterm_coefficient
                * holdout[
                    "generic_ballot_margin_dem"
                ]
                + midterm_coefficient
                * holdout[
                    "midterm_adjustment_dem"
                ]
            ),
        }

        for formula_name, environment in (
            environments.items()
        ):
            predictions = holdout[
                [
                    "race_id",
                    "cycle",
                    "state",
                    "actual_margin_dem",
                    "baseline_margin_dem",
                    "generic_ballot_margin_dem",
                    "presidential_approval",
                    "president_party",
                    "approval_adjustment_dem",
                    "midterm_adjustment_dem",
                ]
            ].copy()

            predictions["formula"] = (
                formula_name
            )

            predictions[
                "formula_label"
            ] = FORMULA_LABELS[
                formula_name
            ]

            predictions[
                "national_environment_margin_dem"
            ] = environment.to_numpy(
                dtype=float
            )

            predictions[
                "predicted_margin_dem"
            ] = (
                predictions[
                    "baseline_margin_dem"
                ]
                + predictions[
                    "national_environment_margin_dem"
                ]
            )

            predictions[
                "prediction_error_dem"
            ] = (
                predictions[
                    "predicted_margin_dem"
                ]
                - predictions[
                    "actual_margin_dem"
                ]
            )

            prediction_frames.append(
                predictions
            )

    all_predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    fold_coefficients = pd.DataFrame(
        coefficient_rows
    )

    summary_rows = []

    for formula_name in FORMULA_ORDER:
        subset = all_predictions.loc[
            all_predictions["formula"]
            == formula_name
        ]

        metrics = calculate_metrics(
            subset,
            args.probability_scale,
        )

        summary_rows.append(
            {
                "formula": formula_name,
                "formula_label": (
                    FORMULA_LABELS[
                        formula_name
                    ]
                ),
                **metrics,
            }
        )

    summary = pd.DataFrame(
        summary_rows
    ).sort_values(
        [
            "mae",
            "rmse",
            "brier",
        ]
    ).reset_index(drop=True)

    summary[
        "mae_rank"
    ] = (
        summary["mae"]
        .rank(method="min")
        .astype(int)
    )

    summary[
        "rmse_rank"
    ] = (
        summary["rmse"]
        .rank(method="min")
        .astype(int)
    )

    summary[
        "brier_rank"
    ] = (
        summary["brier"]
        .rank(method="min")
        .astype(int)
    )

    cycle_rows = []

    for (
        formula_name,
        cycle,
    ), subset in all_predictions.groupby(
        [
            "formula",
            "cycle",
        ]
    ):
        metrics = calculate_metrics(
            subset,
            args.probability_scale,
        )

        cycle_rows.append(
            {
                "formula": formula_name,
                "formula_label": (
                    FORMULA_LABELS[
                        formula_name
                    ]
                ),
                "cycle": int(cycle),
                **metrics,
            }
        )

    by_cycle = pd.DataFrame(
        cycle_rows
    ).sort_values(
        [
            "formula",
            "cycle",
        ]
    )

    duplicate_predictions = int(
        all_predictions[
            [
                "race_id",
                "cycle",
                "formula",
            ]
        ].duplicated().sum()
    )

    expected_predictions = (
        len(model_data)
        * len(FORMULA_ORDER)
    )

    expected_coefficient_rows = (
        len(cycles) * 2
    )

    validation_passed = (
        len(model_data) == 234
        and len(cycles) == 7
        and len(all_predictions)
        == expected_predictions
        and len(fold_coefficients)
        == expected_coefficient_rows
        and duplicate_predictions == 0
        and summary[
            [
                "mae",
                "rmse",
                "brier",
            ]
        ].notna().all().all()
    )

    validation_status = (
        "PASSED"
        if validation_passed
        else "FAILED"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    by_cycle.to_csv(
        CYCLE_PATH,
        index=False,
    )

    all_predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    fold_coefficients.to_csv(
        COEFFICIENTS_PATH,
        index=False,
    )

    config = {
        "input_path": str(INPUT_PATH),
        "input_sha256": input_hash,
        "cycles": cycles,
        "scorable_races": len(model_data),
        "probability_scale": (
            args.probability_scale
        ),
        "baseline_source_column": (
            baseline_column
        ),
        "approval_adjustment_source": (
            adjustment_source
        ),
        "column_mapping": selected_columns,
        "method": (
            "Production-style prediction equals "
            "historical baseline plus candidate "
            "national-environment formula. "
            "Calibrated coefficients are estimated "
            "without an intercept on six cycles and "
            "applied to the held-out cycle."
        ),
        "full_sample_1_190420_warning": (
            "The 1.190420 formula is reported only "
            "as a production reference because its "
            "coefficient was estimated from all "
            "seven cycles and is not strictly "
            "out-of-sample."
        ),
        "validation_status": (
            validation_status
        ),
    }

    CONFIG_PATH.write_text(
        json.dumps(
            config,
            indent=2,
        )
        + "\n"
    )

    coefficient_summary = (
        fold_coefficients.groupby(
            "model",
            as_index=False,
        )
        .agg(
            generic_mean=(
                "generic_ballot_coefficient",
                "mean",
            ),
            generic_sd=(
                "generic_ballot_coefficient",
                "std",
            ),
            generic_min=(
                "generic_ballot_coefficient",
                "min",
            ),
            generic_max=(
                "generic_ballot_coefficient",
                "max",
            ),
            midterm_mean=(
                "midterm_coefficient",
                "mean",
            ),
            midterm_sd=(
                "midterm_coefficient",
                "std",
            ),
            midterm_min=(
                "midterm_coefficient",
                "min",
            ),
            midterm_max=(
                "midterm_coefficient",
                "max",
            ),
        )
    )

    validation_lines = [
        (
            "Senate Production National-Environment "
            "Formula Bake-Off"
        ),
        "=" * 88,
        "",
        f"Input: {INPUT_PATH}",
        f"Input SHA-256: {input_hash}",
        (
            "Historical baseline column: "
            f"{baseline_column}"
        ),
        (
            "Approval adjustment implementation: "
            f"{adjustment_source}"
        ),
        (
            "Cycles: "
            + ", ".join(
                map(str, cycles)
            )
        ),
        f"Scorable races: {len(model_data)}",
        (
            "Probability scale: "
            f"{args.probability_scale:.3f}"
        ),
        "",
        "Overall formula comparison:",
        summary[
            [
                "formula_label",
                "races",
                "mae",
                "rmse",
                "brier",
                "winner_accuracy",
                "mean_error_dem",
                "mae_rank",
                "rmse_rank",
                "brier_rank",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        ),
        "",
        "LOCO coefficient stability:",
        coefficient_summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        ),
        "",
        (
            "Interpretation rule: choose using the "
            "strictly out-of-sample formulas. Treat "
            "the fixed 1.190420 result as a "
            "production reference, not independent "
            "validation."
        ),
        "",
        (
            "Duplicate race-cycle-formula rows: "
            f"{duplicate_predictions}"
        ),
        f"Validation: {validation_status}",
    ]

    validation_text = "\n".join(
        validation_lines
    )

    VALIDATION_PATH.write_text(
        validation_text
        + "\n"
    )

    print(validation_text)
    print()
    print(f"Wrote: {SUMMARY_PATH}")
    print(f"Wrote: {CYCLE_PATH}")
    print(f"Wrote: {PREDICTIONS_PATH}")
    print(f"Wrote: {COEFFICIENTS_PATH}")
    print(f"Wrote: {CONFIG_PATH}")
    print(f"Wrote: {VALIDATION_PATH}")

    if not validation_passed:
        raise SystemExit(
            "Production environment bake-off "
            "validation failed."
        )


if __name__ == "__main__":
    main()
