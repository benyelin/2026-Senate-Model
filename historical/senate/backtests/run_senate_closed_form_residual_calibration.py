#!/usr/bin/env python3
"""
Closed-form Senate bias and uncertainty calibration.

The canonical forecast error convention is:

    forecast_error_dem
        = actual_margin_dem - production_predicted_margin_dem

Therefore:

    bias_correction_dem
        = mean(training forecast errors)

    adjusted_prediction
        = original_prediction + bias_correction_dem

    global_sd
        = sample standard deviation of training forecast errors

The script performs nested leave-one-cycle-out validation and compares:

1. Existing probability-oriented candidate:
       bias = 0.00
       SD   = 8.75

2. Closed-form residual calibration:
       bias and SD estimated only from the training cycles

It also reports a full-history production estimate after the nested
validation is complete.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]

INPUT_PATH = (
    ROOT
    / "historical/senate/backtests/outputs/canonical/"
    "senate_canonical_backtest_dataset.csv"
)

OUTPUT_DIR = (
    ROOT
    / "historical/senate/backtests/outputs/residual_calibration"
)

KEY_COLUMNS = ["race_id", "cycle", "state"]

REQUIRED_COLUMNS = [
    *KEY_COLUMNS,
    "actual_margin_dem",
    "production_predicted_margin_dem",
    "forecast_error_dem",
]

EPSILON = 1e-12

INTERVAL_Z = {
    "50": 0.6744897501960817,
    "80": 1.2815515655446004,
    "95": 1.959963984540054,
}

TARGET_COVERAGE = {
    "50": 0.50,
    "80": 0.80,
    "95": 0.95,
}

COMPARISON_MODELS = {
    "fixed_sd_8_75": {
        "bias_correction_dem": 0.0,
        "global_sd": 8.75,
    },
}


def normal_cdf(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)

    transformed = np.fromiter(
        (
            math.erf(float(value) / math.sqrt(2.0))
            for value in values
        ),
        dtype=float,
        count=len(values),
    )

    return 0.5 * (1.0 + transformed)


def validate_input(frame: pd.DataFrame) -> None:
    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"Canonical dataset is missing required columns: {missing}"
        )

    duplicate_count = int(
        frame.duplicated(KEY_COLUMNS).sum()
    )

    if duplicate_count:
        raise ValueError(
            "Canonical dataset contains duplicate race keys."
        )

    for column in [
        "cycle",
        "actual_margin_dem",
        "production_predicted_margin_dem",
        "forecast_error_dem",
    ]:
        converted = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

        if converted.isna().any():
            raise ValueError(
                f"Canonical dataset contains invalid values in {column}."
            )

    calculated_error = (
        pd.to_numeric(
            frame["actual_margin_dem"],
            errors="raise",
        )
        - pd.to_numeric(
            frame["production_predicted_margin_dem"],
            errors="raise",
        )
    )

    stored_error = pd.to_numeric(
        frame["forecast_error_dem"],
        errors="raise",
    )

    maximum_difference = float(
        (calculated_error - stored_error).abs().max()
    )

    if maximum_difference > 1e-8:
        raise ValueError(
            "Canonical forecast errors do not reconcile. "
            f"Maximum difference: {maximum_difference:.12f}"
        )


def estimate_closed_form_parameters(
    training: pd.DataFrame,
) -> tuple[float, float]:
    residuals = training[
        "forecast_error_dem"
    ].to_numpy(dtype=float)

    bias_correction_dem = float(
        np.mean(residuals)
    )

    centered_residuals = (
        residuals - bias_correction_dem
    )

    global_sd = float(
        np.std(centered_residuals, ddof=1)
    )

    if not np.isfinite(global_sd) or global_sd <= 0:
        raise ValueError(
            "Estimated training residual SD is invalid."
        )

    return bias_correction_dem, global_sd


def build_predictions(
    frame: pd.DataFrame,
    model_name: str,
    bias_correction_dem: float,
    global_sd: float,
    parameter_source: str,
) -> pd.DataFrame:
    output = frame.copy()

    output["model_name"] = model_name
    output["parameter_source"] = parameter_source
    output["bias_correction_dem"] = float(
        bias_correction_dem
    )
    output["global_sd"] = float(global_sd)

    output["adjusted_predicted_margin_dem"] = (
        output["production_predicted_margin_dem"]
        + bias_correction_dem
    )

    output["adjusted_forecast_error_dem"] = (
        output["actual_margin_dem"]
        - output["adjusted_predicted_margin_dem"]
    )

    output["predicted_dem_win_probability"] = normal_cdf(
        output[
            "adjusted_predicted_margin_dem"
        ].to_numpy(dtype=float)
        / global_sd
    )

    output["actual_dem_win"] = (
        output["actual_margin_dem"] > 0
    ).astype(int)

    output["brier_contribution"] = (
        output["predicted_dem_win_probability"]
        - output["actual_dem_win"]
    ) ** 2

    clipped = np.clip(
        output[
            "predicted_dem_win_probability"
        ].to_numpy(dtype=float),
        EPSILON,
        1.0 - EPSILON,
    )

    actual_winner = output[
        "actual_dem_win"
    ].to_numpy(dtype=float)

    output["binary_log_loss_contribution"] = -(
        actual_winner * np.log(clipped)
        + (1.0 - actual_winner)
        * np.log(1.0 - clipped)
    )

    residual = output[
        "adjusted_forecast_error_dem"
    ].to_numpy(dtype=float)

    output["margin_nll_contribution"] = (
        0.5 * np.log(
            2.0 * math.pi * global_sd ** 2
        )
        + 0.5 * (
            residual / global_sd
        ) ** 2
    )

    for interval_name, z_value in INTERVAL_Z.items():
        lower_column = (
            f"predicted_margin_lower_{interval_name}"
        )
        upper_column = (
            f"predicted_margin_upper_{interval_name}"
        )
        covered_column = (
            f"actual_margin_covered_{interval_name}"
        )

        output[lower_column] = (
            output["adjusted_predicted_margin_dem"]
            - z_value * global_sd
        )

        output[upper_column] = (
            output["adjusted_predicted_margin_dem"]
            + z_value * global_sd
        )

        output[covered_column] = (
            (
                output["actual_margin_dem"]
                >= output[lower_column]
            )
            & (
                output["actual_margin_dem"]
                <= output[upper_column]
            )
        )

    return output


def calibration_metrics(
    predictions: pd.DataFrame,
    bins: int = 10,
) -> dict[str, float]:
    bucket_frame = predictions[
        [
            "predicted_dem_win_probability",
            "actual_dem_win",
        ]
    ].copy()

    bucket_frame["bucket"] = pd.cut(
        bucket_frame[
            "predicted_dem_win_probability"
        ],
        bins=np.linspace(0.0, 1.0, bins + 1),
        include_lowest=True,
        right=True,
        duplicates="drop",
    )

    grouped = (
        bucket_frame.groupby(
            "bucket",
            observed=False,
        )
        .agg(
            races=("actual_dem_win", "size"),
            mean_probability=(
                "predicted_dem_win_probability",
                "mean",
            ),
            observed_rate=(
                "actual_dem_win",
                "mean",
            ),
        )
        .reset_index()
    )

    grouped = grouped[
        grouped["races"] > 0
    ].copy()

    grouped["absolute_error"] = (
        grouped["mean_probability"]
        - grouped["observed_rate"]
    ).abs()

    weights = grouped["races"] / grouped["races"].sum()

    return {
        "expected_calibration_error": float(
            np.sum(
                weights
                * grouped["absolute_error"]
            )
        ),
        "maximum_calibration_error": float(
            grouped["absolute_error"].max()
        ),
    }


def summarize_predictions(
    predictions: pd.DataFrame,
) -> dict[str, float | int | str]:
    residual = predictions[
        "adjusted_forecast_error_dem"
    ].to_numpy(dtype=float)

    summary: dict[str, float | int | str] = {
        "model_name": str(
            predictions["model_name"].iloc[0]
        ),
        "races": int(len(predictions)),
        "residual_mean": float(
            np.mean(residual)
        ),
        "residual_sd": float(
            np.std(residual, ddof=1)
        ),
        "margin_mae": float(
            np.mean(np.abs(residual))
        ),
        "margin_rmse": float(
            np.sqrt(np.mean(residual ** 2))
        ),
        "brier": float(
            predictions[
                "brier_contribution"
            ].mean()
        ),
        "binary_log_loss": float(
            predictions[
                "binary_log_loss_contribution"
            ].mean()
        ),
        "margin_negative_log_likelihood": float(
            predictions[
                "margin_nll_contribution"
            ].mean()
        ),
        **calibration_metrics(predictions),
    }

    coverage_errors = []

    for interval_name in INTERVAL_Z:
        coverage = float(
            predictions[
                f"actual_margin_covered_{interval_name}"
            ].mean()
        )

        target = TARGET_COVERAGE[interval_name]

        summary[
            f"coverage_{interval_name}"
        ] = coverage

        summary[
            f"coverage_error_{interval_name}"
        ] = coverage - target

        summary[
            f"absolute_coverage_error_{interval_name}"
        ] = abs(coverage - target)

        coverage_errors.append(
            abs(coverage - target)
        )

    summary["mean_absolute_coverage_error"] = float(
        np.mean(coverage_errors)
    )

    return summary


def run_nested_validation(
    frame: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    cycles = sorted(
        int(cycle)
        for cycle in frame["cycle"].unique()
    )

    parameter_rows = []
    prediction_frames = []

    for holdout_cycle in cycles:
        training = frame[
            frame["cycle"] != holdout_cycle
        ].copy()

        holdout = frame[
            frame["cycle"] == holdout_cycle
        ].copy()

        closed_form_bias, closed_form_sd = (
            estimate_closed_form_parameters(training)
        )

        parameter_rows.append(
            {
                "holdout_cycle": holdout_cycle,
                "training_cycles": ",".join(
                    str(cycle)
                    for cycle in cycles
                    if cycle != holdout_cycle
                ),
                "training_rows": int(len(training)),
                "holdout_rows": int(len(holdout)),
                "training_original_residual_mean": float(
                    training[
                        "forecast_error_dem"
                    ].mean()
                ),
                "selected_bias_correction_dem":
                    closed_form_bias,
                "selected_global_sd":
                    closed_form_sd,
            }
        )

        closed_form_predictions = build_predictions(
            frame=holdout,
            model_name="closed_form_residual",
            bias_correction_dem=closed_form_bias,
            global_sd=closed_form_sd,
            parameter_source=(
                "training_cycles_only"
            ),
        )

        closed_form_predictions[
            "holdout_cycle"
        ] = holdout_cycle

        prediction_frames.append(
            closed_form_predictions
        )

        for model_name, parameters in (
            COMPARISON_MODELS.items()
        ):
            comparison_predictions = build_predictions(
                frame=holdout,
                model_name=model_name,
                bias_correction_dem=float(
                    parameters[
                        "bias_correction_dem"
                    ]
                ),
                global_sd=float(
                    parameters["global_sd"]
                ),
                parameter_source=(
                    "fixed_comparison_parameters"
                ),
            )

            comparison_predictions[
                "holdout_cycle"
            ] = holdout_cycle

            prediction_frames.append(
                comparison_predictions
            )

    parameters = pd.DataFrame(
        parameter_rows
    ).sort_values("holdout_cycle")

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    ).sort_values(
        [
            "model_name",
            "cycle",
            "state",
            "race_id",
        ]
    )

    summary_rows = []

    for model_name, model_predictions in (
        predictions.groupby(
            "model_name",
            sort=True,
        )
    ):
        summary_rows.append(
            summarize_predictions(
                model_predictions
            )
        )

    summaries = pd.DataFrame(
        summary_rows
    ).sort_values("model_name")

    return parameters, predictions, summaries


def build_cycle_summary(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        model_name,
        cycle,
    ), cycle_predictions in predictions.groupby(
        ["model_name", "cycle"],
        sort=True,
    ):
        summary = summarize_predictions(
            cycle_predictions
        )

        summary["cycle"] = int(cycle)
        rows.append(summary)

    return pd.DataFrame(rows).sort_values(
        ["model_name", "cycle"]
    )


def validate_outputs(
    frame: pd.DataFrame,
    parameters: pd.DataFrame,
    predictions: pd.DataFrame,
    summaries: pd.DataFrame,
) -> dict[str, object]:
    expected_models = (
        1 + len(COMPARISON_MODELS)
    )

    expected_prediction_rows = (
        len(frame) * expected_models
    )

    validation = {
        "input_rows": int(len(frame)),
        "input_unique_keys": int(
            frame[KEY_COLUMNS]
            .drop_duplicates()
            .shape[0]
        ),
        "cycles": int(
            frame["cycle"].nunique()
        ),
        "parameter_rows": int(
            len(parameters)
        ),
        "models": int(
            summaries["model_name"].nunique()
        ),
        "prediction_rows": int(
            len(predictions)
        ),
        "expected_prediction_rows":
            expected_prediction_rows,
        "missing_probabilities": int(
            predictions[
                "predicted_dem_win_probability"
            ].isna().sum()
        ),
        "nonpositive_sds": int(
            (
                predictions["global_sd"] <= 0
            ).sum()
        ),
    }

    failures = []

    if (
        validation["input_rows"]
        != validation["input_unique_keys"]
    ):
        failures.append(
            "Canonical input keys are not unique."
        )

    if (
        validation["parameter_rows"]
        != validation["cycles"]
    ):
        failures.append(
            "Parameter row count does not match cycle count."
        )

    if validation["models"] != expected_models:
        failures.append(
            "Unexpected number of comparison models."
        )

    if (
        validation["prediction_rows"]
        != expected_prediction_rows
    ):
        failures.append(
            "Nested prediction count does not reconcile."
        )

    if validation["missing_probabilities"] != 0:
        failures.append(
            "Nested predictions contain missing probabilities."
        )

    if validation["nonpositive_sds"] != 0:
        failures.append(
            "Nested predictions contain invalid SD values."
        )

    if failures:
        formatted = "\n".join(
            f"  - {failure}"
            for failure in failures
        )

        raise RuntimeError(
            "Closed-form residual calibration validation FAILED:\n"
            f"{formatted}"
        )

    validation["validation_status"] = "PASSED"

    return validation


def print_report(
    frame: pd.DataFrame,
    production_bias: float,
    production_sd: float,
    parameters: pd.DataFrame,
    summaries: pd.DataFrame,
    validation: dict[str, object],
) -> None:
    print("=" * 80)
    print("SENATE CLOSED-FORM RESIDUAL CALIBRATION")
    print("=" * 80)
    print(
        f"Historical races:        {len(frame)}"
    )
    print(
        "Cycles:                  "
        + ",".join(
            str(cycle)
            for cycle in sorted(
                frame["cycle"].unique()
            )
        )
    )
    print(
        f"Original residual mean:  "
        f"{frame['forecast_error_dem'].mean():.4f}"
    )
    print(
        f"Original residual SD:    "
        f"{frame['forecast_error_dem'].std(ddof=1):.4f}"
    )

    print("\nFULL-HISTORY PRODUCTION ESTIMATE")
    print(
        f"Bias correction:         "
        f"{production_bias:+.4f}"
    )
    print(
        f"Global SD:               "
        f"{production_sd:.4f}"
    )

    print("\nTRAINING-ONLY PARAMETERS BY HOLDOUT CYCLE")
    print(
        parameters[
            [
                "holdout_cycle",
                "training_rows",
                "holdout_rows",
                "selected_bias_correction_dem",
                "selected_global_sd",
            ]
        ].to_string(index=False)
    )

    print("\nAGGREGATE NESTED OUT-OF-SAMPLE COMPARISON")
    print(
        summaries[
            [
                "model_name",
                "races",
                "residual_mean",
                "margin_mae",
                "margin_rmse",
                "brier",
                "binary_log_loss",
                "margin_negative_log_likelihood",
                "expected_calibration_error",
                "coverage_50",
                "coverage_80",
                "coverage_95",
                "mean_absolute_coverage_error",
            ]
        ].to_string(index=False)
    )

    print("\nVALIDATION")
    print(
        f"Input rows:              "
        f"{validation['input_rows']}"
    )
    print(
        f"Cycles/folds:            "
        f"{validation['cycles']}"
    )
    print(
        f"Models compared:         "
        f"{validation['models']}"
    )
    print(
        f"Prediction rows:         "
        f"{validation['prediction_rows']}"
    )
    print(
        f"Missing probabilities:   "
        f"{validation['missing_probabilities']}"
    )
    print("\nValidation PASSED.")
    print(f"\nOutputs written to:\n{OUTPUT_DIR}")


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Canonical dataset does not exist: {INPUT_PATH}"
        )

    frame = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    validate_input(frame)

    for column in [
        "cycle",
        "actual_margin_dem",
        "production_predicted_margin_dem",
        "forecast_error_dem",
    ]:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="raise",
        )

    frame["cycle"] = frame["cycle"].astype(int)

    frame = frame.sort_values(
        ["cycle", "state", "race_id"]
    ).reset_index(drop=True)

    production_bias, production_sd = (
        estimate_closed_form_parameters(frame)
    )

    parameters, predictions, summaries = (
        run_nested_validation(frame)
    )

    cycle_summary = build_cycle_summary(
        predictions
    )

    production_predictions = build_predictions(
        frame=frame,
        model_name="closed_form_full_history",
        bias_correction_dem=production_bias,
        global_sd=production_sd,
        parameter_source=(
            "full_history_production_estimate"
        ),
    )

    validation = validate_outputs(
        frame=frame,
        parameters=parameters,
        predictions=predictions,
        summaries=summaries,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        [
            {
                "bias_correction_dem":
                    production_bias,
                "global_sd":
                    production_sd,
                "source":
                    "full_history_closed_form",
                "historical_rows":
                    int(len(frame)),
                "cycles":
                    ",".join(
                        str(cycle)
                        for cycle in sorted(
                            frame["cycle"].unique()
                        )
                    ),
            }
        ]
    ).to_csv(
        OUTPUT_DIR
        / "senate_closed_form_production_parameters.csv",
        index=False,
    )

    parameters.to_csv(
        OUTPUT_DIR
        / "senate_closed_form_nested_parameters.csv",
        index=False,
    )

    predictions.to_csv(
        OUTPUT_DIR
        / "senate_closed_form_nested_predictions.csv",
        index=False,
    )

    summaries.to_csv(
        OUTPUT_DIR
        / "senate_closed_form_nested_summary.csv",
        index=False,
    )

    cycle_summary.to_csv(
        OUTPUT_DIR
        / "senate_closed_form_nested_by_cycle.csv",
        index=False,
    )

    production_predictions.to_csv(
        OUTPUT_DIR
        / "senate_closed_form_full_history_predictions.csv",
        index=False,
    )

    pd.DataFrame([validation]).to_csv(
        OUTPUT_DIR
        / "senate_closed_form_validation.csv",
        index=False,
    )

    print_report(
        frame=frame,
        production_bias=production_bias,
        production_sd=production_sd,
        parameters=parameters,
        summaries=summaries,
        validation=validation,
    )


if __name__ == "__main__":
    main()
