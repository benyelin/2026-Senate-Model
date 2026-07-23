#!/usr/bin/env python3
"""
Nested Senate heavy-tail uncertainty comparison.

Models
------

1. normal_baseline
       Normal distribution
       SD = 8.75

2. student_t
       Student's t distribution
       Scale and degrees of freedom selected on training cycles only

3. normal_mixture
       Two-component zero-mean normal mixture:
           ordinary regime: SD = 8.75
           tail regime: wider SD
       Tail probability and tail SD selected on training cycles only

The predicted margin center is unchanged. No bias correction is applied.

Selection is based primarily on binary log loss and Brier score, with
continuous margin likelihood used to distinguish distributions that have
similar winner-probability performance.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import norm, t
except ImportError as exc:
    raise ImportError(
        "This script requires scipy. Install it with:\n"
        "python3 -m pip install scipy"
    ) from exc


ROOT = Path(__file__).resolve().parents[3]

INPUT_PATH = (
    ROOT
    / "historical/senate/backtests/outputs/canonical/"
    "senate_canonical_backtest_dataset.csv"
)

OUTPUT_DIR = (
    ROOT
    / "historical/senate/backtests/outputs/"
    "heavy_tail_uncertainty"
)

KEY_COLUMNS = ["race_id", "cycle", "state"]

REQUIRED_COLUMNS = [
    *KEY_COLUMNS,
    "actual_margin_dem",
    "production_predicted_margin_dem",
    "forecast_error_dem",
]

BASELINE_SD = 8.75
EPSILON = 1e-12

STUDENT_T_DF_GRID = [
    3,
    4,
    5,
    6,
    8,
    10,
    15,
    20,
    30,
]

STUDENT_T_SCALE_GRID = np.arange(
    6.0,
    11.01,
    0.25,
)

MIXTURE_TAIL_PROBABILITY_GRID = [
    0.025,
    0.05,
    0.075,
    0.10,
    0.15,
    0.20,
]

MIXTURE_TAIL_SD_GRID = np.arange(
    12.0,
    25.01,
    1.0,
)

INTERVAL_LEVELS = {
    "50": 0.50,
    "80": 0.80,
    "95": 0.95,
}


def validate_input(frame: pd.DataFrame) -> None:
    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"Canonical dataset is missing columns: {missing}"
        )

    if frame.duplicated(KEY_COLUMNS).any():
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
                f"Invalid numeric values in {column}."
            )

    calculated_error = (
        frame["actual_margin_dem"].astype(float)
        - frame[
            "production_predicted_margin_dem"
        ].astype(float)
    )

    stored_error = frame[
        "forecast_error_dem"
    ].astype(float)

    maximum_difference = float(
        (calculated_error - stored_error).abs().max()
    )

    if maximum_difference > 1e-8:
        raise ValueError(
            "Forecast-error column does not reconcile. "
            f"Maximum difference: {maximum_difference:.12f}"
        )


def binary_metrics(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
) -> tuple[float, float]:
    probabilities = np.clip(
        probabilities,
        EPSILON,
        1.0 - EPSILON,
    )

    brier = float(
        np.mean(
            (probabilities - outcomes) ** 2
        )
    )

    log_loss = float(
        -np.mean(
            outcomes * np.log(probabilities)
            + (1.0 - outcomes)
            * np.log(1.0 - probabilities)
        )
    )

    return brier, log_loss


def expected_calibration_error(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    bins: int = 10,
) -> float:
    bucket_ids = np.minimum(
        (probabilities * bins).astype(int),
        bins - 1,
    )

    total = len(probabilities)
    error = 0.0

    for bucket in range(bins):
        mask = bucket_ids == bucket

        if not np.any(mask):
            continue

        weight = float(np.sum(mask)) / total

        error += weight * abs(
            float(np.mean(probabilities[mask]))
            - float(np.mean(outcomes[mask]))
        )

    return float(error)


def normal_probabilities(
    predicted_margins: np.ndarray,
    sd: float,
) -> np.ndarray:
    return norm.cdf(
        predicted_margins / sd
    )


def student_t_probabilities(
    predicted_margins: np.ndarray,
    degrees_of_freedom: float,
    scale: float,
) -> np.ndarray:
    return t.cdf(
        predicted_margins / scale,
        df=degrees_of_freedom,
    )


def mixture_probabilities(
    predicted_margins: np.ndarray,
    ordinary_sd: float,
    tail_probability: float,
    tail_sd: float,
) -> np.ndarray:
    ordinary_component = norm.cdf(
        predicted_margins / ordinary_sd
    )

    tail_component = norm.cdf(
        predicted_margins / tail_sd
    )

    return (
        (1.0 - tail_probability)
        * ordinary_component
        + tail_probability
        * tail_component
    )


def normal_margin_nll(
    residuals: np.ndarray,
    sd: float,
) -> float:
    densities = norm.pdf(
        residuals,
        loc=0.0,
        scale=sd,
    )

    return float(
        -np.mean(
            np.log(
                np.clip(
                    densities,
                    EPSILON,
                    None,
                )
            )
        )
    )


def student_t_margin_nll(
    residuals: np.ndarray,
    degrees_of_freedom: float,
    scale: float,
) -> float:
    densities = t.pdf(
        residuals / scale,
        df=degrees_of_freedom,
    ) / scale

    return float(
        -np.mean(
            np.log(
                np.clip(
                    densities,
                    EPSILON,
                    None,
                )
            )
        )
    )


def mixture_margin_nll(
    residuals: np.ndarray,
    ordinary_sd: float,
    tail_probability: float,
    tail_sd: float,
) -> float:
    densities = (
        (1.0 - tail_probability)
        * norm.pdf(
            residuals,
            loc=0.0,
            scale=ordinary_sd,
        )
        + tail_probability
        * norm.pdf(
            residuals,
            loc=0.0,
            scale=tail_sd,
        )
    )

    return float(
        -np.mean(
            np.log(
                np.clip(
                    densities,
                    EPSILON,
                    None,
                )
            )
        )
    )


def selection_score(
    brier: float,
    binary_log_loss: float,
    margin_nll: float,
) -> float:
    """
    Preserve winner-probability quality as the main objective.

    Binary log loss receives the greatest weight because it strongly
    penalizes overconfident misses. Brier remains important, and margin
    likelihood rewards realistic tails.
    """
    return (
        2.0 * binary_log_loss
        + 1.0 * brier
        + 0.25 * margin_nll
    )


def tune_student_t(
    training: pd.DataFrame,
) -> dict[str, float]:
    predicted_margins = training[
        "production_predicted_margin_dem"
    ].to_numpy(dtype=float)

    residuals = training[
        "forecast_error_dem"
    ].to_numpy(dtype=float)

    outcomes = (
        training["actual_margin_dem"]
        .to_numpy(dtype=float)
        > 0
    ).astype(float)

    rows = []

    for degrees_of_freedom in STUDENT_T_DF_GRID:
        for scale in STUDENT_T_SCALE_GRID:
            probabilities = student_t_probabilities(
                predicted_margins,
                degrees_of_freedom,
                float(scale),
            )

            brier, log_loss = binary_metrics(
                probabilities,
                outcomes,
            )

            margin_nll = student_t_margin_nll(
                residuals,
                degrees_of_freedom,
                float(scale),
            )

            rows.append(
                {
                    "degrees_of_freedom":
                        float(degrees_of_freedom),
                    "scale": float(scale),
                    "training_brier": brier,
                    "training_binary_log_loss":
                        log_loss,
                    "training_margin_nll":
                        margin_nll,
                    "selection_score":
                        selection_score(
                            brier,
                            log_loss,
                            margin_nll,
                        ),
                }
            )

    results = pd.DataFrame(rows).sort_values(
        [
            "selection_score",
            "training_binary_log_loss",
            "training_brier",
        ]
    )

    return results.iloc[0].to_dict()


def tune_mixture(
    training: pd.DataFrame,
) -> dict[str, float]:
    predicted_margins = training[
        "production_predicted_margin_dem"
    ].to_numpy(dtype=float)

    residuals = training[
        "forecast_error_dem"
    ].to_numpy(dtype=float)

    outcomes = (
        training["actual_margin_dem"]
        .to_numpy(dtype=float)
        > 0
    ).astype(float)

    rows = []

    for tail_probability in (
        MIXTURE_TAIL_PROBABILITY_GRID
    ):
        for tail_sd in MIXTURE_TAIL_SD_GRID:
            probabilities = mixture_probabilities(
                predicted_margins,
                BASELINE_SD,
                float(tail_probability),
                float(tail_sd),
            )

            brier, log_loss = binary_metrics(
                probabilities,
                outcomes,
            )

            margin_nll = mixture_margin_nll(
                residuals,
                BASELINE_SD,
                float(tail_probability),
                float(tail_sd),
            )

            rows.append(
                {
                    "ordinary_sd": BASELINE_SD,
                    "tail_probability":
                        float(tail_probability),
                    "tail_sd": float(tail_sd),
                    "training_brier": brier,
                    "training_binary_log_loss":
                        log_loss,
                    "training_margin_nll":
                        margin_nll,
                    "selection_score":
                        selection_score(
                            brier,
                            log_loss,
                            margin_nll,
                        ),
                }
            )

    results = pd.DataFrame(rows).sort_values(
        [
            "selection_score",
            "training_binary_log_loss",
            "training_brier",
        ]
    )

    return results.iloc[0].to_dict()


def distribution_cdf_error(
    model_name: str,
    values: np.ndarray,
    parameters: dict[str, float],
) -> np.ndarray:
    if model_name == "normal_baseline":
        return norm.cdf(
            values / parameters["sd"]
        )

    if model_name == "student_t":
        return t.cdf(
            values / parameters["scale"],
            df=parameters["degrees_of_freedom"],
        )

    if model_name == "normal_mixture":
        return (
            (1.0 - parameters["tail_probability"])
            * norm.cdf(
                values / parameters["ordinary_sd"]
            )
            + parameters["tail_probability"]
            * norm.cdf(
                values / parameters["tail_sd"]
            )
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )


def central_interval_half_width(
    model_name: str,
    level: float,
    parameters: dict[str, float],
) -> float:
    target_cdf = 0.5 + level / 2.0

    if model_name == "normal_baseline":
        return float(
            norm.ppf(target_cdf)
            * parameters["sd"]
        )

    if model_name == "student_t":
        return float(
            t.ppf(
                target_cdf,
                df=parameters[
                    "degrees_of_freedom"
                ],
            )
            * parameters["scale"]
        )

    if model_name == "normal_mixture":
        low = 0.0
        high = 100.0

        for _ in range(100):
            midpoint = (low + high) / 2.0

            cdf_value = float(
                distribution_cdf_error(
                    model_name,
                    np.array([midpoint]),
                    parameters,
                )[0]
            )

            if cdf_value < target_cdf:
                low = midpoint
            else:
                high = midpoint

        return float(
            (low + high) / 2.0
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )


def build_model_predictions(
    holdout: pd.DataFrame,
    model_name: str,
    parameters: dict[str, float],
    holdout_cycle: int,
) -> pd.DataFrame:
    output = holdout.copy()

    predicted_margins = output[
        "production_predicted_margin_dem"
    ].to_numpy(dtype=float)

    residuals = output[
        "forecast_error_dem"
    ].to_numpy(dtype=float)

    outcomes = (
        output["actual_margin_dem"]
        .to_numpy(dtype=float)
        > 0
    ).astype(float)

    if model_name == "normal_baseline":
        probabilities = normal_probabilities(
            predicted_margins,
            parameters["sd"],
        )

        margin_density = norm.pdf(
            residuals,
            loc=0.0,
            scale=parameters["sd"],
        )

    elif model_name == "student_t":
        probabilities = student_t_probabilities(
            predicted_margins,
            parameters["degrees_of_freedom"],
            parameters["scale"],
        )

        margin_density = t.pdf(
            residuals / parameters["scale"],
            df=parameters["degrees_of_freedom"],
        ) / parameters["scale"]

    elif model_name == "normal_mixture":
        probabilities = mixture_probabilities(
            predicted_margins,
            parameters["ordinary_sd"],
            parameters["tail_probability"],
            parameters["tail_sd"],
        )

        margin_density = (
            (
                1.0
                - parameters["tail_probability"]
            )
            * norm.pdf(
                residuals,
                loc=0.0,
                scale=parameters["ordinary_sd"],
            )
            + parameters["tail_probability"]
            * norm.pdf(
                residuals,
                loc=0.0,
                scale=parameters["tail_sd"],
            )
        )

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    clipped_probabilities = np.clip(
        probabilities,
        EPSILON,
        1.0 - EPSILON,
    )

    output["model_name"] = model_name
    output["holdout_cycle"] = holdout_cycle
    output["predicted_dem_win_probability"] = (
        probabilities
    )
    output["actual_dem_win"] = outcomes.astype(int)

    output["brier_contribution"] = (
        probabilities - outcomes
    ) ** 2

    output["binary_log_loss_contribution"] = -(
        outcomes * np.log(clipped_probabilities)
        + (1.0 - outcomes)
        * np.log(
            1.0 - clipped_probabilities
        )
    )

    output["margin_nll_contribution"] = -np.log(
        np.clip(
            margin_density,
            EPSILON,
            None,
        )
    )

    for parameter_name, parameter_value in (
        parameters.items()
    ):
        output[parameter_name] = (
            float(parameter_value)
        )

    for interval_name, level in (
        INTERVAL_LEVELS.items()
    ):
        half_width = central_interval_half_width(
            model_name,
            level,
            parameters,
        )

        output[
            f"interval_half_width_{interval_name}"
        ] = half_width

        output[
            f"actual_margin_covered_{interval_name}"
        ] = (
            np.abs(residuals) <= half_width
        )

    return output


def summarize_model(
    predictions: pd.DataFrame,
) -> dict[str, float | int | str]:
    probabilities = predictions[
        "predicted_dem_win_probability"
    ].to_numpy(dtype=float)

    outcomes = predictions[
        "actual_dem_win"
    ].to_numpy(dtype=float)

    residuals = predictions[
        "forecast_error_dem"
    ].to_numpy(dtype=float)

    summary = {
        "model_name": str(
            predictions["model_name"].iloc[0]
        ),
        "races": int(len(predictions)),
        "residual_mean": float(
            np.mean(residuals)
        ),
        "residual_sd": float(
            np.std(residuals, ddof=1)
        ),
        "margin_mae": float(
            np.mean(np.abs(residuals))
        ),
        "margin_rmse": float(
            np.sqrt(
                np.mean(residuals ** 2)
            )
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
        "margin_negative_log_likelihood":
            float(
                predictions[
                    "margin_nll_contribution"
                ].mean()
            ),
        "expected_calibration_error":
            expected_calibration_error(
                probabilities,
                outcomes,
            ),
    }

    coverage_errors = []

    for interval_name, target in (
        INTERVAL_LEVELS.items()
    ):
        coverage = float(
            predictions[
                f"actual_margin_covered_{interval_name}"
            ].mean()
        )

        summary[
            f"coverage_{interval_name}"
        ] = coverage

        summary[
            f"coverage_error_{interval_name}"
        ] = coverage - target

        coverage_errors.append(
            abs(coverage - target)
        )

        summary[
            f"mean_half_width_{interval_name}"
        ] = float(
            predictions[
                f"interval_half_width_{interval_name}"
            ].mean()
        )

    summary[
        "mean_absolute_coverage_error"
    ] = float(
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

        student_parameters = tune_student_t(
            training
        )

        mixture_parameters = tune_mixture(
            training
        )

        model_parameters = {
            "normal_baseline": {
                "sd": BASELINE_SD,
            },
            "student_t": {
                "degrees_of_freedom":
                    student_parameters[
                        "degrees_of_freedom"
                    ],
                "scale":
                    student_parameters["scale"],
            },
            "normal_mixture": {
                "ordinary_sd": BASELINE_SD,
                "tail_probability":
                    mixture_parameters[
                        "tail_probability"
                    ],
                "tail_sd":
                    mixture_parameters["tail_sd"],
            },
        }

        for model_name, parameters in (
            model_parameters.items()
        ):
            parameter_row = {
                "holdout_cycle":
                    holdout_cycle,
                "model_name":
                    model_name,
                "training_rows":
                    int(len(training)),
                "holdout_rows":
                    int(len(holdout)),
            }

            parameter_row.update(parameters)

            parameter_rows.append(
                parameter_row
            )

            prediction_frames.append(
                build_model_predictions(
                    holdout=holdout,
                    model_name=model_name,
                    parameters=parameters,
                    holdout_cycle=holdout_cycle,
                )
            )

    parameters = pd.DataFrame(
        parameter_rows
    ).sort_values(
        ["holdout_cycle", "model_name"]
    )

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
            summarize_model(
                model_predictions
            )
        )

    summaries = pd.DataFrame(
        summary_rows
    )

    model_order = {
        "normal_baseline": 1,
        "student_t": 2,
        "normal_mixture": 3,
    }

    summaries["model_order"] = summaries[
        "model_name"
    ].map(model_order)

    summaries = summaries.sort_values(
        "model_order"
    ).drop(
        columns=["model_order"]
    )

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
        summary = summarize_model(
            cycle_predictions
        )

        summary["cycle"] = int(cycle)

        rows.append(summary)

    return pd.DataFrame(rows).sort_values(
        ["model_name", "cycle"]
    )


def build_deltas(
    summaries: pd.DataFrame,
) -> pd.DataFrame:
    baseline = summaries[
        summaries["model_name"]
        == "normal_baseline"
    ]

    if len(baseline) != 1:
        raise RuntimeError(
            "Expected one normal-baseline row."
        )

    baseline = baseline.iloc[0]

    metrics = [
        "brier",
        "binary_log_loss",
        "margin_negative_log_likelihood",
        "expected_calibration_error",
        "coverage_50",
        "coverage_80",
        "coverage_95",
        "mean_absolute_coverage_error",
        "mean_half_width_50",
        "mean_half_width_80",
        "mean_half_width_95",
    ]

    rows = []

    for _, candidate in summaries.iterrows():
        row = {
            "model_name":
                candidate["model_name"],
        }

        for metric in metrics:
            row[f"{metric}_value"] = float(
                candidate[metric]
            )

            row[
                f"{metric}_delta_vs_baseline"
            ] = (
                float(candidate[metric])
                - float(baseline[metric])
            )

        rows.append(row)

    return pd.DataFrame(rows)


def validate_outputs(
    frame: pd.DataFrame,
    parameters: pd.DataFrame,
    predictions: pd.DataFrame,
    summaries: pd.DataFrame,
) -> dict[str, object]:
    expected_models = 3
    cycles = int(
        frame["cycle"].nunique()
    )

    expected_parameter_rows = (
        expected_models * cycles
    )

    expected_prediction_rows = (
        expected_models * len(frame)
    )

    validation = {
        "input_rows": int(len(frame)),
        "cycles": cycles,
        "models": int(
            summaries["model_name"].nunique()
        ),
        "parameter_rows": int(
            len(parameters)
        ),
        "expected_parameter_rows":
            expected_parameter_rows,
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
        "invalid_probabilities": int(
            (
                (
                    predictions[
                        "predicted_dem_win_probability"
                    ] < 0
                )
                | (
                    predictions[
                        "predicted_dem_win_probability"
                    ] > 1
                )
            ).sum()
        ),
    }

    failures = []

    if validation["models"] != expected_models:
        failures.append(
            "Expected three models."
        )

    if (
        validation["parameter_rows"]
        != expected_parameter_rows
    ):
        failures.append(
            "Parameter rows do not reconcile."
        )

    if (
        validation["prediction_rows"]
        != expected_prediction_rows
    ):
        failures.append(
            "Prediction rows do not reconcile."
        )

    if validation["missing_probabilities"] != 0:
        failures.append(
            "Missing probabilities detected."
        )

    if validation["invalid_probabilities"] != 0:
        failures.append(
            "Invalid probabilities detected."
        )

    for model_name, model_frame in (
        predictions.groupby("model_name")
    ):
        unique_keys = model_frame[
            KEY_COLUMNS
        ].drop_duplicates().shape[0]

        if unique_keys != len(frame):
            failures.append(
                f"{model_name} does not contain "
                "one prediction per race."
            )

    if failures:
        formatted = "\n".join(
            f"  - {failure}"
            for failure in failures
        )

        raise RuntimeError(
            "Heavy-tail validation FAILED:\n"
            f"{formatted}"
        )

    validation["validation_status"] = "PASSED"

    return validation


def print_report(
    frame: pd.DataFrame,
    parameters: pd.DataFrame,
    summaries: pd.DataFrame,
    validation: dict[str, object],
) -> None:
    print("=" * 80)
    print("SENATE HEAVY-TAIL UNCERTAINTY COMPARISON")
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
        f"Residual mean:           "
        f"{frame['forecast_error_dem'].mean():.4f}"
    )
    print(
        f"Residual SD:             "
        f"{frame['forecast_error_dem'].std(ddof=1):.4f}"
    )

    print("\nSELECTED STUDENT-T PARAMETERS BY HOLDOUT")
    print(
        parameters[
            parameters["model_name"]
            == "student_t"
        ][
            [
                "holdout_cycle",
                "training_rows",
                "holdout_rows",
                "degrees_of_freedom",
                "scale",
            ]
        ].to_string(index=False)
    )

    print("\nSELECTED MIXTURE PARAMETERS BY HOLDOUT")
    print(
        parameters[
            parameters["model_name"]
            == "normal_mixture"
        ][
            [
                "holdout_cycle",
                "training_rows",
                "holdout_rows",
                "ordinary_sd",
                "tail_probability",
                "tail_sd",
            ]
        ].to_string(index=False)
    )

    print("\nAGGREGATE NESTED OUT-OF-SAMPLE COMPARISON")
    print(
        summaries[
            [
                "model_name",
                "races",
                "brier",
                "binary_log_loss",
                "margin_negative_log_likelihood",
                "expected_calibration_error",
                "coverage_50",
                "coverage_80",
                "coverage_95",
                "mean_absolute_coverage_error",
                "mean_half_width_50",
                "mean_half_width_80",
                "mean_half_width_95",
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
        f"Parameter rows:          "
        f"{validation['parameter_rows']}"
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

    parameters, predictions, summaries = (
        run_nested_validation(frame)
    )

    cycle_summary = build_cycle_summary(
        predictions
    )

    deltas = build_deltas(
        summaries
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

    parameters.to_csv(
        OUTPUT_DIR
        / "senate_heavy_tail_nested_parameters.csv",
        index=False,
    )

    predictions.to_csv(
        OUTPUT_DIR
        / "senate_heavy_tail_nested_predictions.csv",
        index=False,
    )

    summaries.to_csv(
        OUTPUT_DIR
        / "senate_heavy_tail_nested_summary.csv",
        index=False,
    )

    cycle_summary.to_csv(
        OUTPUT_DIR
        / "senate_heavy_tail_nested_by_cycle.csv",
        index=False,
    )

    deltas.to_csv(
        OUTPUT_DIR
        / "senate_heavy_tail_deltas_vs_baseline.csv",
        index=False,
    )

    pd.DataFrame([validation]).to_csv(
        OUTPUT_DIR
        / "senate_heavy_tail_validation.csv",
        index=False,
    )

    print_report(
        frame=frame,
        parameters=parameters,
        summaries=summaries,
        validation=validation,
    )


if __name__ == "__main__":
    main()
