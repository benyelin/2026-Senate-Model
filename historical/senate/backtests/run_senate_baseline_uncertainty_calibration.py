#!/usr/bin/env python3
"""
Calibrate a global baseline uncertainty value for the Senate model.

This script uses the canonical historical Senate backtest dataset and evaluates
a grid of global forecast standard deviations. Point forecasts remain fixed.
Only the conversion from predicted margin to win probability and predictive
intervals changes.

Core probability model
----------------------

    predicted_margin ~ Normal(mean=forecast_margin, sd=global_race_sd)

Therefore:

    P(Dem win) = NormalCDF(predicted_margin / global_race_sd)

Validation design
-----------------

1. Every historical race is treated as held out by cycle.
2. The point forecast is the canonical production forecast.
3. A training-only residual SD is calculated for every held-out cycle as a
   diagnostic benchmark.
4. A common grid of candidate global SD values is evaluated exclusively on
   held-out race outcomes.
5. Recommendation considers:
       - Brier score
       - log loss
       - calibration error
       - 50%, 80%, and 95% interval coverage

The output does not yet add race-specific uncertainty. It establishes the
simple global baseline against which later adjustments must improve.
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
    / "historical/senate/backtests/outputs/uncertainty_baseline"
)

DEFAULT_GRID_SUMMARY = (
    DEFAULT_OUTPUT_DIR
    / "senate_baseline_uncertainty_grid_summary.csv"
)

DEFAULT_GRID_BY_CYCLE = (
    DEFAULT_OUTPUT_DIR
    / "senate_baseline_uncertainty_by_cycle.csv"
)

DEFAULT_RECOMMENDED_PREDICTIONS = (
    DEFAULT_OUTPUT_DIR
    / "senate_baseline_uncertainty_recommended_predictions.csv"
)

DEFAULT_CALIBRATION_BUCKETS = (
    DEFAULT_OUTPUT_DIR
    / "senate_baseline_uncertainty_calibration_buckets.csv"
)

DEFAULT_FOLD_BENCHMARKS = (
    DEFAULT_OUTPUT_DIR
    / "senate_baseline_uncertainty_fold_benchmarks.csv"
)

DEFAULT_VALIDATION = (
    DEFAULT_OUTPUT_DIR
    / "senate_baseline_uncertainty_validation.csv"
)


KEY_COLUMNS = ["race_id", "cycle", "state"]

REQUIRED_COLUMNS = [
    *KEY_COLUMNS,
    "actual_margin_dem",
    "production_predicted_margin_dem",
    "forecast_error_dem",
]

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

EPSILON = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate global baseline uncertainty for historical "
            "Senate forecasts."
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
        "--sd-min",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--sd-max",
        type=float,
        default=20.0,
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
    """Standard normal CDF without requiring scipy."""

    values = np.asarray(values, dtype=float)

    erf_values = np.fromiter(
        (
            math.erf(float(value) / math.sqrt(2.0))
            for value in values
        ),
        dtype=float,
        count=len(values),
    )

    return 0.5 * (1.0 + erf_values)


def require_columns(
    df: pd.DataFrame,
    required: list[str],
) -> None:
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(
            f"Canonical dataset is missing required columns: {missing}"
        )


def validate_input(df: pd.DataFrame) -> None:
    require_columns(df, REQUIRED_COLUMNS)

    duplicate_count = int(df.duplicated(KEY_COLUMNS).sum())

    if duplicate_count:
        duplicate_rows = (
            df.loc[
                df.duplicated(KEY_COLUMNS, keep=False),
                KEY_COLUMNS,
            ]
            .sort_values(KEY_COLUMNS)
            .head(20)
        )

        raise ValueError(
            "Canonical dataset contains duplicate race keys:\n"
            f"{duplicate_rows.to_string(index=False)}"
        )

    numeric_columns = [
        "cycle",
        "actual_margin_dem",
        "production_predicted_margin_dem",
        "forecast_error_dem",
    ]

    for column in numeric_columns:
        if pd.to_numeric(df[column], errors="coerce").isna().any():
            raise ValueError(
                f"Canonical dataset contains invalid values in {column}."
            )

    calculated_error = (
        pd.to_numeric(df["actual_margin_dem"], errors="coerce")
        - pd.to_numeric(
            df["production_predicted_margin_dem"],
            errors="coerce",
        )
    )

    stored_error = pd.to_numeric(
        df["forecast_error_dem"],
        errors="coerce",
    )

    maximum_difference = float(
        (calculated_error - stored_error).abs().max()
    )

    if maximum_difference > 1e-8:
        raise ValueError(
            "Canonical forecast errors do not reconcile. "
            f"Maximum difference: {maximum_difference:.12f}"
        )


def make_sd_grid(
    minimum: float,
    maximum: float,
    step: float,
) -> np.ndarray:
    if minimum <= 0:
        raise ValueError("--sd-min must be greater than zero.")

    if maximum < minimum:
        raise ValueError("--sd-max must be at least --sd-min.")

    if step <= 0:
        raise ValueError("--sd-step must be greater than zero.")

    count = int(round((maximum - minimum) / step))

    grid = minimum + np.arange(count + 1, dtype=float) * step
    grid = grid[grid <= maximum + 1e-10]

    return np.round(grid, 10)


def interval_coverage(
    actual_margin: np.ndarray,
    predicted_margin: np.ndarray,
    sd: float,
    z_value: float,
) -> float:
    lower = predicted_margin - z_value * sd
    upper = predicted_margin + z_value * sd

    covered = (
        (actual_margin >= lower)
        & (actual_margin <= upper)
    )

    return float(np.mean(covered))


def probability_metrics(
    actual_winner_dem: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    probabilities = np.clip(
        probabilities,
        EPSILON,
        1.0 - EPSILON,
    )

    brier = float(
        np.mean(
            (probabilities - actual_winner_dem) ** 2
        )
    )

    log_loss = float(
        -np.mean(
            actual_winner_dem * np.log(probabilities)
            + (1.0 - actual_winner_dem)
            * np.log(1.0 - probabilities)
        )
    )

    return {
        "brier": brier,
        "log_loss": log_loss,
    }


def calibration_metrics(
    actual_winner_dem: np.ndarray,
    probabilities: np.ndarray,
    bins: int,
) -> dict[str, float]:
    frame = pd.DataFrame(
        {
            "actual_winner_dem": actual_winner_dem,
            "probability": probabilities,
        }
    )

    edges = np.linspace(0.0, 1.0, bins + 1)

    frame["bucket"] = pd.cut(
        frame["probability"],
        bins=edges,
        include_lowest=True,
        right=True,
        duplicates="drop",
    )

    grouped = (
        frame.groupby(
            "bucket",
            observed=False,
        )
        .agg(
            races=("actual_winner_dem", "size"),
            mean_probability=("probability", "mean"),
            observed_dem_win_rate=("actual_winner_dem", "mean"),
        )
        .reset_index()
    )

    grouped = grouped[grouped["races"] > 0].copy()

    grouped["absolute_calibration_error"] = (
        grouped["mean_probability"]
        - grouped["observed_dem_win_rate"]
    ).abs()

    grouped["squared_calibration_error"] = (
        grouped["mean_probability"]
        - grouped["observed_dem_win_rate"]
    ) ** 2

    weights = grouped["races"] / grouped["races"].sum()

    expected_calibration_error = float(
        np.sum(
            weights
            * grouped["absolute_calibration_error"]
        )
    )

    root_mean_squared_calibration_error = float(
        np.sqrt(
            np.sum(
                weights
                * grouped["squared_calibration_error"]
            )
        )
    )

    maximum_calibration_error = float(
        grouped["absolute_calibration_error"].max()
    )

    return {
        "expected_calibration_error":
            expected_calibration_error,
        "root_mean_squared_calibration_error":
            root_mean_squared_calibration_error,
        "maximum_calibration_error":
            maximum_calibration_error,
    }


def evaluate_sd(
    frame: pd.DataFrame,
    sd: float,
    calibration_bins: int,
) -> dict[str, float]:
    predicted_margin = frame[
        "production_predicted_margin_dem"
    ].to_numpy(dtype=float)

    actual_margin = frame[
        "actual_margin_dem"
    ].to_numpy(dtype=float)

    actual_winner_dem = (
        actual_margin > 0.0
    ).astype(float)

    probabilities = normal_cdf(
        predicted_margin / sd
    )

    metrics = {
        "global_sd": float(sd),
        "races": int(len(frame)),
        **probability_metrics(
            actual_winner_dem,
            probabilities,
        ),
        **calibration_metrics(
            actual_winner_dem,
            probabilities,
            bins=calibration_bins,
        ),
    }

    for interval_name, z_value in INTERVAL_Z.items():
        coverage = interval_coverage(
            actual_margin=actual_margin,
            predicted_margin=predicted_margin,
            sd=sd,
            z_value=z_value,
        )

        target = TARGET_COVERAGE[interval_name]

        metrics[f"coverage_{interval_name}"] = coverage
        metrics[
            f"coverage_error_{interval_name}"
        ] = coverage - target
        metrics[
            f"absolute_coverage_error_{interval_name}"
        ] = abs(coverage - target)

    metrics["mean_absolute_coverage_error"] = float(
        np.mean(
            [
                metrics[
                    f"absolute_coverage_error_{interval_name}"
                ]
                for interval_name in INTERVAL_Z
            ]
        )
    )

    metrics["maximum_absolute_coverage_error"] = float(
        np.max(
            [
                metrics[
                    f"absolute_coverage_error_{interval_name}"
                ]
                for interval_name in INTERVAL_Z
            ]
        )
    )

    return metrics


def build_grid_results(
    frame: pd.DataFrame,
    sd_grid: np.ndarray,
    calibration_bins: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_rows: list[dict[str, float]] = []
    cycle_rows: list[dict[str, float]] = []

    for sd in sd_grid:
        overall_rows.append(
            evaluate_sd(
                frame=frame,
                sd=float(sd),
                calibration_bins=calibration_bins,
            )
        )

        for cycle, cycle_frame in frame.groupby(
            "cycle",
            sort=True,
        ):
            row = evaluate_sd(
                frame=cycle_frame,
                sd=float(sd),
                calibration_bins=min(
                    calibration_bins,
                    max(2, len(cycle_frame) // 4),
                ),
            )

            row["cycle"] = int(cycle)
            cycle_rows.append(row)

    overall = pd.DataFrame(overall_rows)
    by_cycle = pd.DataFrame(cycle_rows)

    return overall, by_cycle


def add_selection_ranks(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    summary = summary.copy()

    ranking_metrics = [
        "brier",
        "log_loss",
        "expected_calibration_error",
        "mean_absolute_coverage_error",
    ]

    for metric in ranking_metrics:
        summary[f"rank_{metric}"] = summary[
            metric
        ].rank(
            method="min",
            ascending=True,
        )

    summary["average_rank"] = summary[
        [f"rank_{metric}" for metric in ranking_metrics]
    ].mean(axis=1)

    # A candidate is Pareto-optimal when no other candidate performs at least
    # as well on every selection metric and strictly better on one.
    pareto_flags: list[bool] = []

    values = summary[ranking_metrics].to_numpy(dtype=float)

    for index, row in enumerate(values):
        dominated = False

        for other_index, other in enumerate(values):
            if index == other_index:
                continue

            weakly_better = np.all(other <= row)
            strictly_better = np.any(other < row)

            if weakly_better and strictly_better:
                dominated = True
                break

        pareto_flags.append(not dominated)

    summary["pareto_optimal"] = pareto_flags

    # Recommendation:
    #   1. Restrict to Pareto-optimal values.
    #   2. Minimize average rank across probability and coverage metrics.
    #   3. Break ties using log loss, then Brier, then closeness to the
    #      empirical full-sample residual SD.
    empirical_sd = float(
        summary.attrs["empirical_full_sample_residual_sd"]
    )

    eligible = summary[
        summary["pareto_optimal"]
    ].copy()

    eligible["distance_from_empirical_sd"] = (
        eligible["global_sd"] - empirical_sd
    ).abs()

    eligible = eligible.sort_values(
        [
            "average_rank",
            "log_loss",
            "brier",
            "distance_from_empirical_sd",
            "global_sd",
        ]
    )

    recommended_sd = float(
        eligible.iloc[0]["global_sd"]
    )

    summary["recommended"] = np.isclose(
        summary["global_sd"],
        recommended_sd,
        atol=1e-12,
    )

    return summary.sort_values(
        "global_sd"
    ).reset_index(drop=True)


def build_fold_benchmarks(
    frame: pd.DataFrame,
    calibration_bins: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, float]] = []
    prediction_frames: list[pd.DataFrame] = []

    cycles = sorted(
        int(value)
        for value in frame["cycle"].unique()
    )

    for holdout_cycle in cycles:
        training = frame[
            frame["cycle"] != holdout_cycle
        ].copy()

        test = frame[
            frame["cycle"] == holdout_cycle
        ].copy()

        training_residual_sd = float(
            training["forecast_error_dem"].std(ddof=1)
        )

        metrics = evaluate_sd(
            frame=test,
            sd=training_residual_sd,
            calibration_bins=min(
                calibration_bins,
                max(2, len(test) // 4),
            ),
        )

        metrics.update(
            {
                "holdout_cycle": holdout_cycle,
                "training_cycles": ",".join(
                    str(cycle)
                    for cycle in cycles
                    if cycle != holdout_cycle
                ),
                "training_rows": int(len(training)),
                "test_rows": int(len(test)),
                "training_residual_mean": float(
                    training["forecast_error_dem"].mean()
                ),
                "training_residual_sd":
                    training_residual_sd,
            }
        )

        fold_rows.append(metrics)

        test_predictions = make_prediction_output(
            frame=test,
            sd=training_residual_sd,
        )

        test_predictions["sd_source"] = (
            "leave_one_cycle_out_training_residual_sd"
        )
        test_predictions["holdout_cycle"] = holdout_cycle
        test_predictions["training_residual_sd"] = (
            training_residual_sd
        )

        prediction_frames.append(test_predictions)

    return (
        pd.DataFrame(fold_rows).sort_values(
            "holdout_cycle"
        ),
        pd.concat(
            prediction_frames,
            ignore_index=True,
        ).sort_values(
            ["cycle", "state", "race_id"]
        ),
    )


def make_prediction_output(
    frame: pd.DataFrame,
    sd: float,
) -> pd.DataFrame:
    output = frame.copy()

    predicted_margin = output[
        "production_predicted_margin_dem"
    ].to_numpy(dtype=float)

    actual_margin = output[
        "actual_margin_dem"
    ].to_numpy(dtype=float)

    probabilities = normal_cdf(
        predicted_margin / sd
    )

    output["global_sd"] = float(sd)
    output["predicted_dem_win_probability"] = probabilities
    output["actual_dem_win"] = (
        actual_margin > 0.0
    ).astype(int)

    output["brier_contribution"] = (
        output["predicted_dem_win_probability"]
        - output["actual_dem_win"]
    ) ** 2

    clipped = np.clip(
        probabilities,
        EPSILON,
        1.0 - EPSILON,
    )

    output["log_loss_contribution"] = -(
        output["actual_dem_win"] * np.log(clipped)
        + (1 - output["actual_dem_win"])
        * np.log(1.0 - clipped)
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
            predicted_margin - z_value * sd
        )
        output[upper_column] = (
            predicted_margin + z_value * sd
        )
        output[covered_column] = (
            (actual_margin >= output[lower_column])
            & (actual_margin <= output[upper_column])
        )

    preferred_columns = [
        *KEY_COLUMNS,
        "election_date",
        "actual_margin_dem",
        "production_predicted_margin_dem",
        "forecast_error_dem",
        "global_sd",
        "predicted_dem_win_probability",
        "actual_dem_win",
        "brier_contribution",
        "log_loss_contribution",
        "predicted_margin_lower_50",
        "predicted_margin_upper_50",
        "actual_margin_covered_50",
        "predicted_margin_lower_80",
        "predicted_margin_upper_80",
        "actual_margin_covered_80",
        "predicted_margin_lower_95",
        "predicted_margin_upper_95",
        "actual_margin_covered_95",
        "incumbency_category",
        "model_incumbent_party",
        "model_open_seat",
        "lineage_appointed_incumbent",
    ]

    existing_columns = [
        column
        for column in preferred_columns
        if column in output.columns
    ]

    return output[existing_columns].copy()


def build_calibration_buckets(
    predictions: pd.DataFrame,
    bins: int,
) -> pd.DataFrame:
    output = predictions.copy()

    edges = np.linspace(0.0, 1.0, bins + 1)

    output["probability_bucket"] = pd.cut(
        output["predicted_dem_win_probability"],
        bins=edges,
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
            mean_predicted_margin_dem=(
                "production_predicted_margin_dem",
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
            mean_log_loss_contribution=(
                "log_loss_contribution",
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
        - grouped["mean_predicted_dem_probability"]
    )

    grouped["absolute_calibration_error"] = (
        grouped["calibration_error"].abs()
    )

    return grouped


def validate_outputs(
    input_frame: pd.DataFrame,
    summary: pd.DataFrame,
    predictions: pd.DataFrame,
    calibration: pd.DataFrame,
    fold_benchmarks: pd.DataFrame,
) -> dict[str, object]:
    recommended_count = int(
        summary["recommended"].sum()
    )

    recommended_sd = float(
        summary.loc[
            summary["recommended"],
            "global_sd",
        ].iloc[0]
    ) if recommended_count == 1 else float("nan")

    validation = {
        "input_rows": int(len(input_frame)),
        "input_unique_keys": int(
            input_frame[KEY_COLUMNS]
            .drop_duplicates()
            .shape[0]
        ),
        "grid_candidates": int(len(summary)),
        "recommended_rows": recommended_count,
        "recommended_sd": recommended_sd,
        "recommended_prediction_rows": int(
            len(predictions)
        ),
        "recommended_prediction_unique_keys": int(
            predictions[KEY_COLUMNS]
            .drop_duplicates()
            .shape[0]
        ),
        "missing_recommended_probabilities": int(
            predictions[
                "predicted_dem_win_probability"
            ].isna().sum()
        ),
        "probabilities_below_zero": int(
            (
                predictions[
                    "predicted_dem_win_probability"
                ] < 0
            ).sum()
        ),
        "probabilities_above_one": int(
            (
                predictions[
                    "predicted_dem_win_probability"
                ] > 1
            ).sum()
        ),
        "calibration_bucket_races": int(
            calibration["races"].sum()
        ),
        "folds": int(len(fold_benchmarks)),
        "cycles": ",".join(
            str(value)
            for value in sorted(
                input_frame["cycle"].unique()
            )
        ),
    }

    failures: list[str] = []

    if (
        validation["input_rows"]
        != validation["input_unique_keys"]
    ):
        failures.append(
            "Input does not contain one row per canonical key."
        )

    if recommended_count != 1:
        failures.append(
            "Exactly one global SD must be recommended."
        )

    if (
        validation["recommended_prediction_rows"]
        != validation["input_rows"]
    ):
        failures.append(
            "Recommended predictions do not cover all races."
        )

    if (
        validation["recommended_prediction_unique_keys"]
        != validation["input_unique_keys"]
    ):
        failures.append(
            "Recommended prediction keys do not reconcile."
        )

    if validation["missing_recommended_probabilities"] != 0:
        failures.append(
            "Recommended predictions contain missing probabilities."
        )

    if validation["probabilities_below_zero"] != 0:
        failures.append(
            "Recommended probabilities fall below zero."
        )

    if validation["probabilities_above_one"] != 0:
        failures.append(
            "Recommended probabilities exceed one."
        )

    if (
        validation["calibration_bucket_races"]
        != validation["input_rows"]
    ):
        failures.append(
            "Calibration buckets do not cover all races."
        )

    expected_folds = int(
        input_frame["cycle"].nunique()
    )

    if validation["folds"] != expected_folds:
        failures.append(
            "Fold benchmark count does not equal cycle count."
        )

    if failures:
        formatted = "\n".join(
            f"  - {failure}"
            for failure in failures
        )

        raise RuntimeError(
            "Baseline uncertainty validation FAILED:\n"
            f"{formatted}"
        )

    validation["validation_status"] = "PASSED"

    return validation


def print_report(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    by_cycle: pd.DataFrame,
    fold_benchmarks: pd.DataFrame,
    validation: dict[str, object],
    output_dir: Path,
) -> None:
    recommended = summary[
        summary["recommended"]
    ].iloc[0]

    empirical_sd = float(
        frame["forecast_error_dem"].std(ddof=1)
    )

    residual_mean = float(
        frame["forecast_error_dem"].mean()
    )

    print("=" * 80)
    print("SENATE BASELINE UNCERTAINTY CALIBRATION")
    print("=" * 80)
    print(f"Historical races:        {len(frame)}")
    print(
        f"Cycles:                  "
        f"{validation['cycles']}"
    )
    print(
        f"Residual mean:           "
        f"{residual_mean:.4f}"
    )
    print(
        f"Full-sample residual SD: "
        f"{empirical_sd:.4f}"
    )
    print(
        f"Recommended global SD:   "
        f"{recommended['global_sd']:.2f}"
    )

    print("\nRECOMMENDED OUT-OF-SAMPLE METRICS")
    print(
        f"Brier:                   "
        f"{recommended['brier']:.5f}"
    )
    print(
        f"Log loss:                "
        f"{recommended['log_loss']:.5f}"
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

    print("\nTOP 10 CANDIDATE SD VALUES")
    top = (
        summary.sort_values(
            [
                "average_rank",
                "log_loss",
                "brier",
            ]
        )
        .head(10)
        [
            [
                "global_sd",
                "brier",
                "log_loss",
                "expected_calibration_error",
                "coverage_50",
                "coverage_80",
                "coverage_95",
                "average_rank",
                "pareto_optimal",
                "recommended",
            ]
        ]
    )

    print(top.to_string(index=False))

    recommended_sd = float(
        recommended["global_sd"]
    )

    recommended_by_cycle = by_cycle[
        np.isclose(
            by_cycle["global_sd"],
            recommended_sd,
            atol=1e-12,
        )
    ].copy()

    print("\nRECOMMENDED SD BY CYCLE")
    print(
        recommended_by_cycle[
            [
                "cycle",
                "races",
                "brier",
                "log_loss",
                "coverage_50",
                "coverage_80",
                "coverage_95",
            ]
        ].to_string(index=False)
    )

    print("\nTRAINING-ONLY RESIDUAL SD BY HOLDOUT CYCLE")
    print(
        fold_benchmarks[
            [
                "holdout_cycle",
                "training_rows",
                "test_rows",
                "training_residual_mean",
                "training_residual_sd",
                "brier",
                "log_loss",
            ]
        ].to_string(index=False)
    )

    print("\nVALIDATION")
    print(
        f"Input rows:              "
        f"{validation['input_rows']}"
    )
    print(
        f"Unique keys:             "
        f"{validation['input_unique_keys']}"
    )
    print(
        f"Grid candidates:         "
        f"{validation['grid_candidates']}"
    )
    print(
        f"Recommended rows:        "
        f"{validation['recommended_rows']}"
    )
    print(
        f"Prediction rows:         "
        f"{validation['recommended_prediction_rows']}"
    )
    print(
        f"Calibration bucket rows: "
        f"{validation['calibration_bucket_races']}"
    )
    print(
        f"LOCO folds:              "
        f"{validation['folds']}"
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

    sd_grid = make_sd_grid(
        minimum=args.sd_min,
        maximum=args.sd_max,
        step=args.sd_step,
    )

    summary, by_cycle = build_grid_results(
        frame=frame,
        sd_grid=sd_grid,
        calibration_bins=args.calibration_bins,
    )

    summary.attrs[
        "empirical_full_sample_residual_sd"
    ] = float(
        frame["forecast_error_dem"].std(ddof=1)
    )

    summary = add_selection_ranks(summary)

    recommended_sd = float(
        summary.loc[
            summary["recommended"],
            "global_sd",
        ].iloc[0]
    )

    predictions = make_prediction_output(
        frame=frame,
        sd=recommended_sd,
    )

    predictions["sd_source"] = (
        "recommended_global_baseline"
    )

    calibration = build_calibration_buckets(
        predictions=predictions,
        bins=args.calibration_bins,
    )

    fold_benchmarks, fold_predictions = (
        build_fold_benchmarks(
            frame=frame,
            calibration_bins=args.calibration_bins,
        )
    )

    validation = validate_outputs(
        input_frame=frame,
        summary=summary,
        predictions=predictions,
        calibration=calibration,
        fold_benchmarks=fold_benchmarks,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    grid_summary_path = (
        args.output_dir
        / DEFAULT_GRID_SUMMARY.name
    )
    grid_by_cycle_path = (
        args.output_dir
        / DEFAULT_GRID_BY_CYCLE.name
    )
    recommended_predictions_path = (
        args.output_dir
        / DEFAULT_RECOMMENDED_PREDICTIONS.name
    )
    calibration_path = (
        args.output_dir
        / DEFAULT_CALIBRATION_BUCKETS.name
    )
    fold_benchmarks_path = (
        args.output_dir
        / DEFAULT_FOLD_BENCHMARKS.name
    )
    fold_predictions_path = (
        args.output_dir
        / "senate_baseline_uncertainty_fold_predictions.csv"
    )
    validation_path = (
        args.output_dir
        / DEFAULT_VALIDATION.name
    )

    summary.to_csv(
        grid_summary_path,
        index=False,
    )

    by_cycle.to_csv(
        grid_by_cycle_path,
        index=False,
    )

    predictions.to_csv(
        recommended_predictions_path,
        index=False,
    )

    calibration.to_csv(
        calibration_path,
        index=False,
    )

    fold_benchmarks.to_csv(
        fold_benchmarks_path,
        index=False,
    )

    fold_predictions.to_csv(
        fold_predictions_path,
        index=False,
    )

    pd.DataFrame([validation]).to_csv(
        validation_path,
        index=False,
    )

    print_report(
        frame=frame,
        summary=summary,
        by_cycle=by_cycle,
        fold_benchmarks=fold_benchmarks,
        validation=validation,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
