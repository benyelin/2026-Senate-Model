from __future__ import annotations

import hashlib
import json
from math import sqrt
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
    / "layer1_production_calibration"
)

CALIBRATION_CSV_PATH = (
    OUTPUT_DIR
    / "senate_layer1_production_coefficients.csv"
)

FOLD_CSV_PATH = (
    OUTPUT_DIR
    / "senate_layer1_coefficient_stability.csv"
)

CONFIG_JSON_PATH = (
    OUTPUT_DIR
    / "senate_layer1_production_config.json"
)

VALIDATION_PATH = (
    OUTPUT_DIR
    / "senate_layer1_production_calibration_validation.txt"
)

REQUIRED_COLUMNS = [
    "race_id",
    "cycle",
    "state",
    "actual_margin_dem",
    "presidential_margin_dem",
    "generic_ballot_margin_dem",
    "historical_fundamentals_scorable",
]

FEATURE_COLUMNS = [
    "presidential_margin_dem",
    "generic_ballot_margin_dem",
]

TERM_NAMES = [
    "intercept",
    *FEATURE_COLUMNS,
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


def normalize_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

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
    }

    unexpected = sorted(
        set(normalized.dropna().unique())
        - set(mapping)
    )

    if unexpected:
        raise ValueError(
            "Unexpected boolean values: "
            + ", ".join(map(str, unexpected))
        )

    return (
        normalized.map(mapping)
        .fillna(False)
        .astype(bool)
    )


def build_design_matrix(
    frame: pd.DataFrame,
) -> np.ndarray:
    features = frame[
        FEATURE_COLUMNS
    ].to_numpy(dtype=float)

    return np.column_stack(
        [
            np.ones(len(frame)),
            features,
        ]
    )


def fit_ols(
    frame: pd.DataFrame,
) -> dict:
    design = build_design_matrix(frame)

    target = frame[
        "actual_margin_dem"
    ].to_numpy(dtype=float)

    coefficients, *_ = np.linalg.lstsq(
        design,
        target,
        rcond=None,
    )

    predictions = design @ coefficients
    residuals = target - predictions

    observation_count = len(frame)
    parameter_count = design.shape[1]
    degrees_of_freedom = max(
        observation_count - parameter_count,
        1,
    )

    residual_sum_of_squares = float(
        residuals @ residuals
    )

    residual_variance = (
        residual_sum_of_squares
        / degrees_of_freedom
    )

    residual_sd = sqrt(
        max(residual_variance, 0.0)
    )

    xtx_inverse = np.linalg.pinv(
        design.T @ design
    )

    covariance_matrix = (
        residual_variance
        * xtx_inverse
    )

    standard_errors = np.sqrt(
        np.maximum(
            np.diag(covariance_matrix),
            0.0,
        )
    )

    critical_value = 1.96

    lower_95 = (
        coefficients
        - critical_value * standard_errors
    )

    upper_95 = (
        coefficients
        + critical_value * standard_errors
    )

    total_variation = float(
        np.sum(
            (
                target
                - np.mean(target)
            ) ** 2
        )
    )

    r_squared = (
        1.0
        - residual_sum_of_squares
        / total_variation
        if total_variation > 0.0
        else float("nan")
    )

    adjusted_r_squared = (
        1.0
        - (
            1.0 - r_squared
        )
        * (
            observation_count - 1
        )
        / degrees_of_freedom
        if observation_count > parameter_count
        else float("nan")
    )

    mae = float(
        np.mean(
            np.abs(residuals)
        )
    )

    rmse = float(
        sqrt(
            np.mean(
                residuals ** 2
            )
        )
    )

    return {
        "coefficients": coefficients,
        "standard_errors": standard_errors,
        "lower_95": lower_95,
        "upper_95": upper_95,
        "predictions": predictions,
        "residuals": residuals,
        "residual_sd": residual_sd,
        "mae": mae,
        "rmse": rmse,
        "r_squared": float(r_squared),
        "adjusted_r_squared": float(
            adjusted_r_squared
        ),
        "observation_count": observation_count,
        "parameter_count": parameter_count,
        "degrees_of_freedom": degrees_of_freedom,
    }


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            INPUT_PATH
        )

    input_hash = sha256_file(
        INPUT_PATH
    )

    data = pd.read_csv(
        INPUT_PATH
    )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    data[
        "historical_fundamentals_scorable"
    ] = normalize_boolean(
        data[
            "historical_fundamentals_scorable"
        ]
    )

    numeric_columns = [
        "cycle",
        "actual_margin_dem",
        "presidential_margin_dem",
        "generic_ballot_margin_dem",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    model_data = data.loc[
        data[
            "historical_fundamentals_scorable"
        ]
        & data["cycle"].notna()
        & data["actual_margin_dem"].notna()
        & data[
            "presidential_margin_dem"
        ].notna()
        & data[
            "generic_ballot_margin_dem"
        ].notna()
    ].copy()

    model_data["cycle"] = (
        model_data["cycle"].astype(int)
    )

    model_data = model_data.sort_values(
        [
            "cycle",
            "state",
            "race_id",
        ]
    ).reset_index(drop=True)

    cycles = sorted(
        model_data["cycle"]
        .unique()
        .tolist()
    )

    if len(cycles) < 3:
        raise ValueError(
            "Too few cycles for stability analysis."
        )

    full_fit = fit_ols(
        model_data
    )

    fold_rows = []

    for holdout_cycle in cycles:
        training = model_data.loc[
            model_data["cycle"]
            != holdout_cycle
        ].copy()

        fold_fit = fit_ols(
            training
        )

        for term_name, coefficient in zip(
            TERM_NAMES,
            fold_fit["coefficients"],
        ):
            fold_rows.append(
                {
                    "holdout_cycle": (
                        holdout_cycle
                    ),
                    "term": term_name,
                    "coefficient": float(
                        coefficient
                    ),
                    "training_rows": len(
                        training
                    ),
                    "training_cycles": (
                        len(cycles) - 1
                    ),
                    "training_residual_sd": (
                        fold_fit[
                            "residual_sd"
                        ]
                    ),
                }
            )

    fold_coefficients = pd.DataFrame(
        fold_rows
    )

    stability = (
        fold_coefficients.groupby(
            "term",
            as_index=False,
        )
        .agg(
            loco_mean=(
                "coefficient",
                "mean",
            ),
            loco_sd=(
                "coefficient",
                "std",
            ),
            loco_min=(
                "coefficient",
                "min",
            ),
            loco_max=(
                "coefficient",
                "max",
            ),
        )
    )

    calibration_rows = []

    for index, term_name in enumerate(
        TERM_NAMES
    ):
        stability_row = stability.loc[
            stability["term"]
            == term_name
        ].iloc[0]

        estimate = float(
            full_fit["coefficients"][index]
        )

        calibration_rows.append(
            {
                "term": term_name,
                "estimate": estimate,
                "standard_error": float(
                    full_fit[
                        "standard_errors"
                    ][index]
                ),
                "analytic_ci_95_lower": float(
                    full_fit[
                        "lower_95"
                    ][index]
                ),
                "analytic_ci_95_upper": float(
                    full_fit[
                        "upper_95"
                    ][index]
                ),
                "loco_mean": float(
                    stability_row[
                        "loco_mean"
                    ]
                ),
                "loco_sd": float(
                    stability_row[
                        "loco_sd"
                    ]
                ),
                "loco_min": float(
                    stability_row[
                        "loco_min"
                    ]
                ),
                "loco_max": float(
                    stability_row[
                        "loco_max"
                    ]
                ),
                "max_abs_loco_deviation": float(
                    fold_coefficients.loc[
                        fold_coefficients[
                            "term"
                        ]
                        == term_name,
                        "coefficient",
                    ]
                    .sub(estimate)
                    .abs()
                    .max()
                ),
            }
        )

    calibration = pd.DataFrame(
        calibration_rows
    )

    coefficient_lookup = dict(
        zip(
            calibration["term"],
            calibration["estimate"],
        )
    )

    config = {
        "model_name": (
            "senate_layer1_presidential_"
            "baseline_plus_generic_ballot"
        ),
        "calibration_status": (
            "candidate_for_production_review"
        ),
        "input_path": str(INPUT_PATH),
        "input_sha256": input_hash,
        "calibration_cycles": cycles,
        "scorable_races": len(model_data),
        "formula": (
            "predicted_margin_dem = intercept "
            "+ presidential_margin_coefficient "
            "* presidential_margin_dem "
            "+ generic_ballot_coefficient "
            "* generic_ballot_margin_dem"
        ),
        "coefficients": {
            "intercept": coefficient_lookup[
                "intercept"
            ],
            "presidential_margin_dem": (
                coefficient_lookup[
                    "presidential_margin_dem"
                ]
            ),
            "generic_ballot_margin_dem": (
                coefficient_lookup[
                    "generic_ballot_margin_dem"
                ]
            ),
        },
        "residual_sd": full_fit[
            "residual_sd"
        ],
        "in_sample_metrics": {
            "mae": full_fit["mae"],
            "rmse": full_fit["rmse"],
            "r_squared": (
                full_fit["r_squared"]
            ),
            "adjusted_r_squared": (
                full_fit[
                    "adjusted_r_squared"
                ]
            ),
        },
        "production_integration_note": (
            "Do not automatically add the fitted "
            "intercept to the existing production "
            "national-environment field. Determine "
            "whether the district or state baseline "
            "already incorporates the structural "
            "intercept before integration."
        ),
        "approval_component": {
            "recommended_status": (
                "exclude_from_retained_layer1"
            ),
            "reason": (
                "Approval-based specifications did "
                "not improve leave-one-cycle-out "
                "performance over generic ballot."
            ),
        },
        "midterm_component": {
            "recommended_status": (
                "unresolved_pending_identifiable_test"
            ),
            "reason": (
                "The prior fixed-shift test was "
                "aliased with the fold intercept."
            ),
        },
    }

    duplicate_race_ids = int(
        model_data["race_id"]
        .duplicated()
        .sum()
    )

    missing_coefficient_values = int(
        calibration["estimate"]
        .isna()
        .sum()
    )

    expected_fold_rows = (
        len(cycles)
        * len(TERM_NAMES)
    )

    validation_passed = (
        len(model_data) == 234
        and len(cycles) == 7
        and len(calibration)
        == len(TERM_NAMES)
        and len(fold_coefficients)
        == expected_fold_rows
        and duplicate_race_ids == 0
        and missing_coefficient_values == 0
        and full_fit["residual_sd"] > 0.0
    )

    validation_status = (
        "PASSED"
        if validation_passed
        else "FAILED"
    )

    config[
        "validation_status"
    ] = validation_status

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    calibration.to_csv(
        CALIBRATION_CSV_PATH,
        index=False,
    )

    fold_coefficients.sort_values(
        [
            "term",
            "holdout_cycle",
        ]
    ).to_csv(
        FOLD_CSV_PATH,
        index=False,
    )

    CONFIG_JSON_PATH.write_text(
        json.dumps(
            config,
            indent=2,
        )
        + "\n"
    )

    display_table = calibration[
        [
            "term",
            "estimate",
            "standard_error",
            "analytic_ci_95_lower",
            "analytic_ci_95_upper",
            "loco_mean",
            "loco_sd",
            "loco_min",
            "loco_max",
        ]
    ]

    validation_lines = [
        (
            "Senate Layer 1 Production "
            "Calibration"
        ),
        "=" * 76,
        "",
        f"Input: {INPUT_PATH}",
        f"Input SHA-256: {input_hash}",
        (
            "Cycles: "
            + ", ".join(map(str, cycles))
        ),
        f"Scorable races: {len(model_data)}",
        (
            "Duplicate race IDs: "
            f"{duplicate_race_ids}"
        ),
        "",
        "Full-sample coefficients and stability:",
        display_table.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        ),
        "",
        "Full-sample fit:",
        (
            f"MAE: {full_fit['mae']:.6f}"
        ),
        (
            f"RMSE: {full_fit['rmse']:.6f}"
        ),
        (
            "Residual SD: "
            f"{full_fit['residual_sd']:.6f}"
        ),
        (
            "R-squared: "
            f"{full_fit['r_squared']:.6f}"
        ),
        (
            "Adjusted R-squared: "
            f"{full_fit['adjusted_r_squared']:.6f}"
        ),
        "",
        (
            "Important: these are full-sample "
            "production-candidate estimates. "
            "The previously reported LOCO metrics "
            "remain the proper out-of-sample "
            "performance estimates."
        ),
        "",
        (
            "Important: do not insert the fitted "
            "intercept into production until baseline "
            "double-counting has been evaluated."
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

    print(validation_text)
    print()
    print(f"Wrote: {CALIBRATION_CSV_PATH}")
    print(f"Wrote: {FOLD_CSV_PATH}")
    print(f"Wrote: {CONFIG_JSON_PATH}")
    print(f"Wrote: {VALIDATION_PATH}")

    if not validation_passed:
        raise SystemExit(
            "Layer 1 production calibration "
            "validation failed."
        )


if __name__ == "__main__":
    main()
