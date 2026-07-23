#!/usr/bin/env python3
"""
Jointly calibrate Senate forecast bias and global margin uncertainty.

Forecast distribution
---------------------

    adjusted_margin = production_predicted_margin_dem + bias_correction_dem

    actual_margin ~ Normal(
        mean=adjusted_margin,
        sd=global_sd
    )

The script produces:

1. A full-history grid recommendation for the eventual production parameters.
2. A nested leave-one-cycle-out backtest:
       - tune bias and SD using six cycles
       - evaluate the chosen pair on the held-out seventh cycle
3. Continuous margin likelihood, probability metrics, calibration, and
   predictive-interval coverage.
4. Validation outputs suitable for comparing later race-specific uncertainty
   additions against this simple global model.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]

DEFAULT_INPUT = (
    ROOT
    / "historical/senate/backtests/outputs/canonical/"
    "senate_canonical_backtest_dataset.csv"
)

DEFAULT_OUTPUT_DIR = (
    ROOT
    / "historical/senate/backtests/outputs/bias_uncertainty"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Jointly calibrate Senate forecast bias and global uncertainty."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--bias-min",
        type=float,
        default=-3.0,
    )
    parser.add_argument(
        "--bias-max",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--bias-step",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--sd-min",
        type=float,
        default=7.0,
    )
    parser.add_argument(
        "--sd-max",
        type=float,
        default=14.0,
    )
    parser.add_argument(
        "--sd-step",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=10,
    )

    return parser.parse_args()


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


def make_grid(
    minimum: float,
    maximum: float,
    step: float,
) -> np.ndarray:
    if step <= 0:
        raise ValueError("Grid step must be greater than zero.")

    if maximum < minimum:
        raise ValueError("Grid maximum must be at least the minimum.")

    count = int(round((maximum - minimum) / step))

    grid = minimum + np.arange(count + 1) * step
    grid = grid[grid <= maximum + 1e-10]

    return np.round(grid.astype(float), 10)


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
                f"Canonical dataset contains invalid {column} values."
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
            "Canonical errors do not reconcile. "
            f"Maximum difference: {maximum_difference:.12f}"
        )


def calibration_metrics(
    actual_winner_dem: np.ndarray,
    probabilities: np.ndarray,
    bins: int,
) -> dict[str, float]:
    bucket_frame = pd.DataFrame(
        {
            "actual": actual_winner_dem,
            "probability": probabilities,
        }
    )

    bucket_frame["bucket"] = pd.cut(
        bucket_frame["probability"],
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
            races=("actual", "size"),
            mean_probability=("probability", "mean"),
            observed_rate=("actual", "mean"),
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

    grouped["squared_error"] = (
        grouped["mean_probability"]
        - grouped["observed_rate"]
    ) ** 2

    weights = grouped["races"] / grouped["races"].sum()

    return {
        "expected_calibration_error": float(
            np.sum(weights * grouped["absolute_error"])
        ),
        "root_mean_squared_calibration_error": float(
            np.sqrt(
                np.sum(weights * grouped["squared_error"])
            )
        ),
        "maximum_calibration_error": float(
            grouped["absolute_error"].max()
        ),
    }


def evaluate_candidate(
    frame: pd.DataFrame,
    bias_correction_dem: float,
    global_sd: float,
    calibration_bins: int,
) -> dict[str, float]:
    actual_margin = frame[
        "actual_margin_dem"
    ].to_numpy(dtype=float)

    original_prediction = frame[
        "production_predicted_margin_dem"
    ].to_numpy(dtype=float)

    adjusted_prediction = (
        original_prediction + bias_correction_dem
    )

    adjusted_residual = (
        actual_margin - adjusted_prediction
    )

    actual_winner_dem = (
        actual_margin > 0
    ).astype(float)

    probability_dem = normal_cdf(
        adjusted_prediction / global_sd
    )

    clipped_probability = np.clip(
        probability_dem,
        EPSILON,
        1.0 - EPSILON,
    )

    brier = float(
        np.mean(
            (probability_dem - actual_winner_dem) ** 2
        )
    )

    binary_log_loss = float(
        -np.mean(
            actual_winner_dem * np.log(clipped_probability)
            + (1.0 - actual_winner_dem)
            * np.log(1.0 - clipped_probability)
        )
    )

    margin_negative_log_likelihood = float(
        np.mean(
            0.5 * np.log(2.0 * math.pi * global_sd ** 2)
            + 0.5 * (
                adjusted_residual / global_sd
            ) ** 2
        )
    )

    metrics = {
        "bias_correction_dem": float(bias_correction_dem),
        "global_sd": float(global_sd),
        "races": int(len(frame)),
        "adjusted_residual_mean": float(
            np.mean(adjusted_residual)
        ),
        "adjusted_residual_sd": float(
            np.std(adjusted_residual, ddof=1)
        ),
        "margin_mae": float(
            np.mean(np.abs(adjusted_residual))
        ),
        "margin_rmse": float(
            np.sqrt(np.mean(adjusted_residual ** 2))
        ),
        "brier": brier,
        "binary_log_loss": binary_log_loss,
        "margin_negative_log_likelihood":
            margin_negative_log_likelihood,
        **calibration_metrics(
            actual_winner_dem=actual_winner_dem,
            probabilities=probability_dem,
            bins=calibration_bins,
        ),
    }

    coverage_errors = []

    for interval_name, z_value in INTERVAL_Z.items():
        lower = (
            adjusted_prediction
            - z_value * global_sd
        )

        upper = (
            adjusted_prediction
            + z_value * global_sd
        )

        coverage = float(
            np.mean(
                (actual_margin >= lower)
                & (actual_margin <= upper)
            )
        )

        target = TARGET_COVERAGE[interval_name]
        coverage_error = coverage - target

        metrics[f"coverage_{interval_name}"] = coverage
        metrics[
            f"coverage_error_{interval_name}"
        ] = coverage_error
        metrics[
            f"absolute_coverage_error_{interval_name}"
        ] = abs(coverage_error)

        coverage_errors.append(abs(coverage_error))

    metrics["mean_absolute_coverage_error"] = float(
        np.mean(coverage_errors)
    )

    metrics["maximum_absolute_coverage_error"] = float(
        np.max(coverage_errors)
    )

    return metrics


def evaluate_grid(
    frame: pd.DataFrame,
    bias_grid: np.ndarray,
    sd_grid: np.ndarray,
    calibration_bins: int,
) -> pd.DataFrame:
    rows = []

    for bias_correction in bias_grid:
        for global_sd in sd_grid:
            rows.append(
                evaluate_candidate(
                    frame=frame,
                    bias_correction_dem=float(
                        bias_correction
                    ),
                    global_sd=float(global_sd),
                    calibration_bins=calibration_bins,
                )
            )

    return pd.DataFrame(rows)


def add_selection_fields(
    grid: pd.DataFrame,
) -> pd.DataFrame:
    grid = grid.copy()

    ranking_metrics = [
        "brier",
        "binary_log_loss",
        "margin_negative_log_likelihood",
        "expected_calibration_error",
        "mean_absolute_coverage_error",
    ]

    for metric in ranking_metrics:
        grid[f"rank_{metric}"] = grid[
            metric
        ].rank(
            method="min",
            ascending=True,
        )

    grid["average_rank"] = grid[
        [f"rank_{metric}" for metric in ranking_metrics]
    ].mean(axis=1)

    # The continuous margin likelihood receives extra weight because the
    # selected parameters drive full race-margin simulations, not only
    # binary win probabilities.
    grid["weighted_selection_score"] = (
        grid["rank_brier"]
        + grid["rank_binary_log_loss"]
        + 2.0
        * grid["rank_margin_negative_log_likelihood"]
        + grid["rank_expected_calibration_error"]
        + grid["rank_mean_absolute_coverage_error"]
    ) / 6.0

    pareto_columns = [
        "brier",
        "binary_log_loss",
        "margin_negative_log_likelihood",
        "expected_calibration_error",
        "mean_absolute_coverage_error",
    ]

    values = grid[
        pareto_columns
    ].to_numpy(dtype=float)

    pareto_flags = []

    for index, candidate in enumerate(values):
        dominated = False

        for other_index, other in enumerate(values):
            if index == other_index:
                continue

            if (
                np.all(other <= candidate)
                and np.any(other < candidate)
            ):
                dominated = True
                break

        pareto_flags.append(not dominated)

    grid["pareto_optimal"] = pareto_flags

    eligible = grid[
        grid["pareto_optimal"]
    ].copy()

    eligible["absolute_remaining_bias"] = (
        eligible["adjusted_residual_mean"].abs()
    )

    eligible = eligible.sort_values(
        [
            "weighted_selection_score",
            "margin_negative_log_likelihood",
            "binary_log_loss",
            "brier",
            "absolute_remaining_bias",
            "global_sd",
            "bias_correction_dem",
        ]
    )

    selected_index = eligible.index[0]

    grid["recommended"] = False
    grid.loc[selected_index, "recommended"] = True

    return grid.sort_values(
        [
            "bias_correction_dem",
            "global_sd",
        ]
    ).reset_index(drop=True)


def build_predictions(
    frame: pd.DataFrame,
    bias_correction_dem: float,
    global_sd: float,
    parameter_source: str,
) -> pd.DataFrame:
    output = frame.copy()

    output["bias_correction_dem"] = float(
        bias_correction_dem
    )
    output["global_sd"] = float(global_sd)
    output["parameter_source"] = parameter_source

    output["adjusted_predicted_margin_dem"] = (
        output["production_predicted_margin_dem"]
        + bias_correction_dem
    )

    output["adjusted_forecast_error_dem"] = (
        output["actual_margin_dem"]
        - output["adjusted_predicted_margin_dem"]
    )

    output["predicted_dem_win_probability"] = normal_cdf(
        output["adjusted_predicted_margin_dem"].to_numpy(
            dtype=float
        )
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

    output["binary_log_loss_contribution"] = -(
        output["actual_dem_win"] * np.log(clipped)
        + (1 - output["actual_dem_win"])
        * np.log(1.0 - clipped)
    )

    residual = output[
        "adjusted_forecast_error_dem"
    ].to_numpy(dtype=float)

    output["margin_nll_contribution"] = (
        0.5 * np.log(2.0 * math.pi * global_sd ** 2)
        + 0.5 * (residual / global_sd) ** 2
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


def nested_leave_one_cycle_out(
    frame: pd.DataFrame,
    bias_grid: np.ndarray,
    sd_grid: np.ndarray,
    calibration_bins: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selection_rows = []
    prediction_frames = []

    cycles = sorted(
        int(cycle)
        for cycle in frame["cycle"].unique()
    )

    for holdout_cycle in cycles:
        training = frame[
            frame["cycle"] != holdout_cycle
        ].copy()

        holdout = frame[
            frame["cycle"] == holdout_cycle
        ].copy()

        training_grid = evaluate_grid(
            frame=training,
            bias_grid=bias_grid,
            sd_grid=sd_grid,
            calibration_bins=calibration_bins,
        )

        training_grid = add_selection_fields(
            training_grid
        )

        selected = training_grid[
            training_grid["recommended"]
        ].iloc[0]

        selected_bias = float(
            selected["bias_correction_dem"]
        )

        selected_sd = float(
            selected["global_sd"]
        )

        holdout_metrics = evaluate_candidate(
            frame=holdout,
            bias_correction_dem=selected_bias,
            global_sd=selected_sd,
            calibration_bins=min(
                calibration_bins,
                max(2, len(holdout) // 4),
            ),
        )

        selection_rows.append(
            {
                "holdout_cycle": holdout_cycle,
                "training_cycles": ",".join(
                    str(cycle)
                    for cycle in cycles
                    if cycle != holdout_cycle
                ),
                "training_rows": int(len(training)),
                "holdout_rows": int(len(holdout)),
                "selected_bias_correction_dem":
                    selected_bias,
                "selected_global_sd": selected_sd,
                "training_weighted_selection_score":
                    float(
                        selected[
                            "weighted_selection_score"
                        ]
                    ),
                "training_brier": float(
                    selected["brier"]
                ),
                "training_binary_log_loss": float(
                    selected["binary_log_loss"]
                ),
                "training_margin_nll": float(
                    selected[
                        "margin_negative_log_likelihood"
                    ]
                ),
                "training_expected_calibration_error":
                    float(
                        selected[
                            "expected_calibration_error"
                        ]
                    ),
                "training_mean_absolute_coverage_error":
                    float(
                        selected[
                            "mean_absolute_coverage_error"
                        ]
                    ),
                **{
                    f"holdout_{key}": value
                    for key, value in holdout_metrics.items()
                    if key not in {
                        "bias_correction_dem",
                        "global_sd",
                        "races",
                    }
                },
            }
        )

        holdout_predictions = build_predictions(
            frame=holdout,
            bias_correction_dem=selected_bias,
            global_sd=selected_sd,
            parameter_source=(
                "nested_leave_one_cycle_out"
            ),
        )

        holdout_predictions[
            "holdout_cycle"
        ] = holdout_cycle

        prediction_frames.append(
            holdout_predictions
        )

    selections = pd.DataFrame(
        selection_rows
    ).sort_values("holdout_cycle")

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    ).sort_values(
        ["cycle", "state", "race_id"]
    )

    return selections, predictions


def aggregate_nested_metrics(
    predictions: pd.DataFrame,
    calibration_bins: int,
) -> dict[str, float]:
    actual_margin = predictions[
        "actual_margin_dem"
    ].to_numpy(dtype=float)

    adjusted_prediction = predictions[
        "adjusted_predicted_margin_dem"
    ].to_numpy(dtype=float)

    probabilities = predictions[
        "predicted_dem_win_probability"
    ].to_numpy(dtype=float)

    actual_winner = predictions[
        "actual_dem_win"
    ].to_numpy(dtype=float)

    residual = actual_margin - adjusted_prediction

    metrics = {
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
            np.mean(
                (probabilities - actual_winner) ** 2
            )
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
        **calibration_metrics(
            actual_winner_dem=actual_winner,
            probabilities=probabilities,
            bins=calibration_bins,
        ),
    }

    coverage_errors = []

    for interval_name in INTERVAL_Z:
        coverage = float(
            predictions[
                f"actual_margin_covered_{interval_name}"
            ].mean()
        )

        target = TARGET_COVERAGE[interval_name]

        metrics[
            f"coverage_{interval_name}"
        ] = coverage

        metrics[
            f"coverage_error_{interval_name}"
        ] = coverage - target

        metrics[
            f"absolute_coverage_error_{interval_name}"
        ] = abs(coverage - target)

        coverage_errors.append(
            abs(coverage - target)
        )

    metrics["mean_absolute_coverage_error"] = float(
        np.mean(coverage_errors)
    )

    return metrics


def build_calibration_buckets(
    predictions: pd.DataFrame,
    bins: int,
) -> pd.DataFrame:
    output = predictions.copy()

    output["probability_bucket"] = pd.cut(
        output["predicted_dem_win_probability"],
        bins=np.linspace(0.0, 1.0, bins + 1),
        include_lowest=True,
        right=True,
        duplicates="drop",
    )

    grouped = (
        output.groupby(
            "probability_bucket",
            observed=False,
        )
        .agg(
            races=("actual_dem_win", "size"),
            mean_predicted_dem_probability=(
                "predicted_dem_win_probability",
                "mean",
            ),
            observed_dem_win_rate=(
                "actual_dem_win",
                "mean",
            ),
            mean_adjusted_predicted_margin_dem=(
                "adjusted_predicted_margin_dem",
                "mean",
            ),
            mean_actual_margin_dem=(
                "actual_margin_dem",
                "mean",
            ),
            mean_brier_contribution=(
                "brier_contribution",
                "mean",
            ),
            mean_binary_log_loss_contribution=(
                "binary_log_loss_contribution",
                "mean",
            ),
            mean_margin_nll_contribution=(
                "margin_nll_contribution",
                "mean",
            ),
        )
        .reset_index()
    )

    grouped["probability_bucket"] = (
        grouped["probability_bucket"].astype(str)
    )

    grouped["calibration_error"] = (
        grouped["observed_dem_win_rate"]
        - grouped[
            "mean_predicted_dem_probability"
        ]
    )

    grouped["absolute_calibration_error"] = (
        grouped["calibration_error"].abs()
    )

    return grouped


def validate_outputs(
    frame: pd.DataFrame,
    full_grid: pd.DataFrame,
    full_predictions: pd.DataFrame,
    nested_selections: pd.DataFrame,
    nested_predictions: pd.DataFrame,
) -> dict[str, object]:
    validation = {
        "input_rows": int(len(frame)),
        "input_unique_keys": int(
            frame[KEY_COLUMNS]
            .drop_duplicates()
            .shape[0]
        ),
        "full_grid_rows": int(len(full_grid)),
        "full_recommended_rows": int(
            full_grid["recommended"].sum()
        ),
        "full_prediction_rows": int(
            len(full_predictions)
        ),
        "nested_folds": int(
            len(nested_selections)
        ),
        "nested_prediction_rows": int(
            len(nested_predictions)
        ),
        "nested_prediction_unique_keys": int(
            nested_predictions[KEY_COLUMNS]
            .drop_duplicates()
            .shape[0]
        ),
        "missing_nested_probabilities": int(
            nested_predictions[
                "predicted_dem_win_probability"
            ].isna().sum()
        ),
    }

    failures = []

    if (
        validation["input_rows"]
        != validation["input_unique_keys"]
    ):
        failures.append(
            "Input keys are not unique."
        )

    if validation["full_recommended_rows"] != 1:
        failures.append(
            "Full-history grid must recommend exactly one pair."
        )

    if (
        validation["full_prediction_rows"]
        != validation["input_rows"]
    ):
        failures.append(
            "Full-history predictions do not cover all races."
        )

    if (
        validation["nested_folds"]
        != frame["cycle"].nunique()
    ):
        failures.append(
            "Nested fold count does not equal cycle count."
        )

    if (
        validation["nested_prediction_rows"]
        != validation["input_rows"]
    ):
        failures.append(
            "Nested predictions do not cover all races."
        )

    if (
        validation["nested_prediction_unique_keys"]
        != validation["input_unique_keys"]
    ):
        failures.append(
            "Nested prediction keys do not reconcile."
        )

    if validation["missing_nested_probabilities"] != 0:
        failures.append(
            "Nested predictions contain missing probabilities."
        )

    if failures:
        formatted = "\n".join(
            f"  - {failure}"
            for failure in failures
        )

        raise RuntimeError(
            "Bias-and-uncertainty validation FAILED:\n"
            f"{formatted}"
        )

    validation["validation_status"] = "PASSED"

    return validation


def print_report(
    frame: pd.DataFrame,
    full_grid: pd.DataFrame,
    nested_selections: pd.DataFrame,
    nested_metrics: dict[str, float],
    validation: dict[str, object],
    output_dir: Path,
) -> None:
    recommended = full_grid[
        full_grid["recommended"]
    ].iloc[0]

    print("=" * 80)
    print("SENATE JOINT BIAS AND UNCERTAINTY CALIBRATION")
    print("=" * 80)
    print(f"Historical races:        {len(frame)}")
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

    print("\nFULL-HISTORY PRODUCTION RECOMMENDATION")
    print(
        f"Bias correction:         "
        f"{recommended['bias_correction_dem']:+.2f}"
    )
    print(
        f"Global SD:               "
        f"{recommended['global_sd']:.2f}"
    )
    print(
        f"Remaining residual mean: "
        f"{recommended['adjusted_residual_mean']:.4f}"
    )
    print(
        f"Brier:                   "
        f"{recommended['brier']:.5f}"
    )
    print(
        f"Binary log loss:         "
        f"{recommended['binary_log_loss']:.5f}"
    )
    print(
        f"Margin NLL:              "
        f"{recommended['margin_negative_log_likelihood']:.5f}"
    )
    print(
        f"Expected calibration error: "
        f"{recommended['expected_calibration_error']:.5f}"
    )
    print(
        f"50% interval coverage:   "
        f"{recommended['coverage_50']:.1%}"
    )
    print(
        f"80% interval coverage:   "
        f"{recommended['coverage_80']:.1%}"
    )
    print(
        f"95% interval coverage:   "
        f"{recommended['coverage_95']:.1%}"
    )
    print(
        f"Pareto optimal:          "
        f"{bool(recommended['pareto_optimal'])}"
    )

    print("\nNESTED LEAVE-ONE-CYCLE-OUT SELECTIONS")
    print(
        nested_selections[
            [
                "holdout_cycle",
                "training_rows",
                "holdout_rows",
                "selected_bias_correction_dem",
                "selected_global_sd",
                "holdout_brier",
                "holdout_binary_log_loss",
                "holdout_margin_negative_log_likelihood",
                "holdout_coverage_50",
                "holdout_coverage_80",
                "holdout_coverage_95",
            ]
        ].to_string(index=False)
    )

    print("\nAGGREGATE NESTED OUT-OF-SAMPLE METRICS")
    print(
        f"Residual mean:           "
        f"{nested_metrics['residual_mean']:.4f}"
    )
    print(
        f"Residual SD:             "
        f"{nested_metrics['residual_sd']:.4f}"
    )
    print(
        f"Margin MAE:              "
        f"{nested_metrics['margin_mae']:.4f}"
    )
    print(
        f"Margin RMSE:             "
        f"{nested_metrics['margin_rmse']:.4f}"
    )
    print(
        f"Brier:                   "
        f"{nested_metrics['brier']:.5f}"
    )
    print(
        f"Binary log loss:         "
        f"{nested_metrics['binary_log_loss']:.5f}"
    )
    print(
        f"Margin NLL:              "
        f"{nested_metrics['margin_negative_log_likelihood']:.5f}"
    )
    print(
        f"Expected calibration error: "
        f"{nested_metrics['expected_calibration_error']:.5f}"
    )
    print(
        f"50% interval coverage:   "
        f"{nested_metrics['coverage_50']:.1%}"
    )
    print(
        f"80% interval coverage:   "
        f"{nested_metrics['coverage_80']:.1%}"
    )
    print(
        f"95% interval coverage:   "
        f"{nested_metrics['coverage_95']:.1%}"
    )

    print("\nTOP 10 FULL-HISTORY CANDIDATES")
    print(
        full_grid.sort_values(
            [
                "weighted_selection_score",
                "margin_negative_log_likelihood",
                "binary_log_loss",
            ]
        )
        .head(10)
        [
            [
                "bias_correction_dem",
                "global_sd",
                "brier",
                "binary_log_loss",
                "margin_negative_log_likelihood",
                "expected_calibration_error",
                "coverage_50",
                "coverage_80",
                "coverage_95",
                "weighted_selection_score",
                "pareto_optimal",
                "recommended",
            ]
        ]
        .to_string(index=False)
    )

    print("\nVALIDATION")
    print(
        f"Input rows:              "
        f"{validation['input_rows']}"
    )
    print(
        f"Full grid rows:          "
        f"{validation['full_grid_rows']}"
    )
    print(
        f"Full recommended rows:   "
        f"{validation['full_recommended_rows']}"
    )
    print(
        f"Nested folds:            "
        f"{validation['nested_folds']}"
    )
    print(
        f"Nested prediction rows:  "
        f"{validation['nested_prediction_rows']}"
    )
    print("\nValidation PASSED.")
    print(f"\nOutputs written to:\n{output_dir}")


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Canonical dataset does not exist: {args.input}"
        )

    frame = pd.read_csv(
        args.input,
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

    bias_grid = make_grid(
        minimum=args.bias_min,
        maximum=args.bias_max,
        step=args.bias_step,
    )

    sd_grid = make_grid(
        minimum=args.sd_min,
        maximum=args.sd_max,
        step=args.sd_step,
    )

    full_grid = evaluate_grid(
        frame=frame,
        bias_grid=bias_grid,
        sd_grid=sd_grid,
        calibration_bins=args.calibration_bins,
    )

    full_grid = add_selection_fields(
        full_grid
    )

    full_recommended = full_grid[
        full_grid["recommended"]
    ].iloc[0]

    full_bias = float(
        full_recommended["bias_correction_dem"]
    )

    full_sd = float(
        full_recommended["global_sd"]
    )

    full_predictions = build_predictions(
        frame=frame,
        bias_correction_dem=full_bias,
        global_sd=full_sd,
        parameter_source=(
            "full_history_production_recommendation"
        ),
    )

    nested_selections, nested_predictions = (
        nested_leave_one_cycle_out(
            frame=frame,
            bias_grid=bias_grid,
            sd_grid=sd_grid,
            calibration_bins=args.calibration_bins,
        )
    )

    nested_metrics = aggregate_nested_metrics(
        predictions=nested_predictions,
        calibration_bins=args.calibration_bins,
    )

    full_calibration_buckets = (
        build_calibration_buckets(
            predictions=full_predictions,
            bins=args.calibration_bins,
        )
    )

    nested_calibration_buckets = (
        build_calibration_buckets(
            predictions=nested_predictions,
            bins=args.calibration_bins,
        )
    )

    validation = validate_outputs(
        frame=frame,
        full_grid=full_grid,
        full_predictions=full_predictions,
        nested_selections=nested_selections,
        nested_predictions=nested_predictions,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    full_grid.to_csv(
        args.output_dir
        / "senate_bias_uncertainty_full_grid.csv",
        index=False,
    )

    full_predictions.to_csv(
        args.output_dir
        / "senate_bias_uncertainty_full_recommended_predictions.csv",
        index=False,
    )

    nested_selections.to_csv(
        args.output_dir
        / "senate_bias_uncertainty_nested_selections.csv",
        index=False,
    )

    nested_predictions.to_csv(
        args.output_dir
        / "senate_bias_uncertainty_nested_predictions.csv",
        index=False,
    )

    pd.DataFrame([nested_metrics]).to_csv(
        args.output_dir
        / "senate_bias_uncertainty_nested_summary.csv",
        index=False,
    )

    full_calibration_buckets.to_csv(
        args.output_dir
        / "senate_bias_uncertainty_full_calibration_buckets.csv",
        index=False,
    )

    nested_calibration_buckets.to_csv(
        args.output_dir
        / "senate_bias_uncertainty_nested_calibration_buckets.csv",
        index=False,
    )

    pd.DataFrame([validation]).to_csv(
        args.output_dir
        / "senate_bias_uncertainty_validation.csv",
        index=False,
    )

    print_report(
        frame=frame,
        full_grid=full_grid,
        nested_selections=nested_selections,
        nested_metrics=nested_metrics,
        validation=validation,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
