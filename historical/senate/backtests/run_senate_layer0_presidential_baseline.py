from __future__ import annotations

import hashlib
import json
from math import sqrt
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_PATH = (
    PROJECT_ROOT
    / "historical/senate/warehouse/processed/"
    "senate_historical_fundamentals_2012_2024.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "historical/senate/backtests/outputs/"
    "layer0_presidential_baseline"
)

PREDICTIONS_PATH = (
    OUTPUT_DIR
    / "senate_layer0_loco_predictions.csv"
)

FOLDS_PATH = (
    OUTPUT_DIR
    / "senate_layer0_loco_fold_metrics.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "senate_layer0_loco_summary.json"
)

VALIDATION_PATH = (
    OUTPUT_DIR
    / "senate_layer0_loco_validation.txt"
)

REQUIRED_COLUMNS = [
    "race_id",
    "cycle",
    "state",
    "election_type",
    "special_election",
    "actual_margin_dem",
    "presidential_margin_dem",
    "historical_fundamentals_scorable",
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


def normalize_boolean(
    series: pd.Series,
) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    normalized = (
        series
        .astype("string")
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

    return normalized.map(mapping).fillna(False).astype(bool)


def fit_ols(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float]:
    design = np.column_stack(
        [
            np.ones(len(x)),
            x,
        ]
    )

    coefficients, *_ = np.linalg.lstsq(
        design,
        y,
        rcond=None,
    )

    intercept = float(coefficients[0])
    slope = float(coefficients[1])

    return intercept, slope


def predict_margin(
    x: np.ndarray,
    intercept: float,
    slope: float,
) -> np.ndarray:
    return intercept + slope * x


def residual_standard_deviation(
    actual: np.ndarray,
    predicted: np.ndarray,
    parameter_count: int = 2,
) -> float:
    residuals = actual - predicted

    degrees_of_freedom = max(
        len(residuals) - parameter_count,
        1,
    )

    variance = float(
        np.sum(residuals ** 2)
        / degrees_of_freedom
    )

    return max(
        sqrt(variance),
        1e-6,
    )


def margin_to_probability(
    margin: float,
    residual_sd: float,
) -> float:
    z_score = margin / residual_sd

    return float(
        NormalDist().cdf(z_score)
    )


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    errors = predicted - actual

    actual_dem_win = (
        actual > 0.0
    ).astype(float)

    predicted_dem_win = (
        predicted > 0.0
    ).astype(float)

    mae = float(
        np.mean(np.abs(errors))
    )

    rmse = float(
        sqrt(np.mean(errors ** 2))
    )

    brier = float(
        np.mean(
            (
                probabilities
                - actual_dem_win
            ) ** 2
        )
    )

    directional_accuracy = float(
        np.mean(
            predicted_dem_win
            == actual_dem_win
        )
    )

    actual_variance = float(
        np.sum(
            (
                actual
                - np.mean(actual)
            ) ** 2
        )
    )

    if actual_variance > 0.0:
        r_squared = float(
            1.0
            - (
                np.sum(errors ** 2)
                / actual_variance
            )
        )
    else:
        r_squared = float("nan")

    return {
        "mae": mae,
        "rmse": rmse,
        "brier": brier,
        "directional_accuracy": (
            directional_accuracy
        ),
        "r_squared": r_squared,
    }


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    source_hash = sha256_file(
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

    data["cycle"] = pd.to_numeric(
        data["cycle"],
        errors="raise",
    ).astype(int)

    data["actual_margin_dem"] = pd.to_numeric(
        data["actual_margin_dem"],
        errors="coerce",
    )

    data[
        "presidential_margin_dem"
    ] = pd.to_numeric(
        data["presidential_margin_dem"],
        errors="coerce",
    )

    model_data = data.loc[
        data[
            "historical_fundamentals_scorable"
        ]
        & data["actual_margin_dem"].notna()
        & data[
            "presidential_margin_dem"
        ].notna()
    ].copy()

    model_data = model_data.sort_values(
        [
            "cycle",
            "state",
            "race_id",
        ]
    ).reset_index(drop=True)

    if model_data.empty:
        raise ValueError(
            "No scorable historical races found."
        )

    cycles = sorted(
        model_data["cycle"]
        .unique()
        .tolist()
    )

    prediction_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, float | int]] = []

    for holdout_cycle in cycles:
        train = model_data.loc[
            model_data["cycle"]
            != holdout_cycle
        ].copy()

        test = model_data.loc[
            model_data["cycle"]
            == holdout_cycle
        ].copy()

        x_train = train[
            "presidential_margin_dem"
        ].to_numpy(dtype=float)

        y_train = train[
            "actual_margin_dem"
        ].to_numpy(dtype=float)

        x_test = test[
            "presidential_margin_dem"
        ].to_numpy(dtype=float)

        y_test = test[
            "actual_margin_dem"
        ].to_numpy(dtype=float)

        intercept, slope = fit_ols(
            x=x_train,
            y=y_train,
        )

        train_predictions = predict_margin(
            x=x_train,
            intercept=intercept,
            slope=slope,
        )

        residual_sd = residual_standard_deviation(
            actual=y_train,
            predicted=train_predictions,
        )

        test_predictions = predict_margin(
            x=x_test,
            intercept=intercept,
            slope=slope,
        )

        test_probabilities = np.array(
            [
                margin_to_probability(
                    margin=float(margin),
                    residual_sd=residual_sd,
                )
                for margin in test_predictions
            ],
            dtype=float,
        )

        metrics = calculate_metrics(
            actual=y_test,
            predicted=test_predictions,
            probabilities=test_probabilities,
        )

        test[
            "holdout_cycle"
        ] = holdout_cycle

        test[
            "layer0_intercept"
        ] = intercept

        test[
            "layer0_presidential_slope"
        ] = slope

        test[
            "training_residual_sd"
        ] = residual_sd

        test[
            "predicted_margin_dem"
        ] = test_predictions

        test[
            "predicted_dem_win_probability"
        ] = test_probabilities

        test[
            "prediction_error_dem"
        ] = (
            test["predicted_margin_dem"]
            - test["actual_margin_dem"]
        )

        test[
            "absolute_error"
        ] = (
            test["prediction_error_dem"]
            .abs()
        )

        test[
            "predicted_winner_party"
        ] = np.where(
            test[
                "predicted_margin_dem"
            ] > 0.0,
            "D",
            "R",
        )

        test[
            "actual_winner_party"
        ] = np.where(
            test[
                "actual_margin_dem"
            ] > 0.0,
            "D",
            "R",
        )

        test[
            "winner_correct"
        ] = (
            test[
                "predicted_winner_party"
            ]
            == test[
                "actual_winner_party"
            ]
        )

        prediction_frames.append(
            test
        )

        fold_rows.append(
            {
                "holdout_cycle": (
                    holdout_cycle
                ),
                "training_rows": len(train),
                "test_rows": len(test),
                "intercept": intercept,
                "presidential_slope": slope,
                "training_residual_sd": (
                    residual_sd
                ),
                **metrics,
            }
        )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    fold_metrics = pd.DataFrame(
        fold_rows
    ).sort_values(
        "holdout_cycle"
    ).reset_index(drop=True)

    overall_metrics = calculate_metrics(
        actual=predictions[
            "actual_margin_dem"
        ].to_numpy(dtype=float),
        predicted=predictions[
            "predicted_margin_dem"
        ].to_numpy(dtype=float),
        probabilities=predictions[
            "predicted_dem_win_probability"
        ].to_numpy(dtype=float),
    )

    prediction_count_by_race = (
        predictions.groupby("race_id")
        .size()
    )

    missing_predictions = int(
        len(model_data)
        - len(predictions)
    )

    duplicate_predictions = int(
        (
            prediction_count_by_race
            > 1
        ).sum()
    )

    incorrect_holdout_assignments = int(
        (
            predictions["cycle"]
            != predictions["holdout_cycle"]
        ).sum()
    )

    in_sample_leakage_rows = int(
        predictions[
            "holdout_cycle"
        ].isna().sum()
    )

    probability_out_of_bounds = int(
        (
            (
                predictions[
                    "predicted_dem_win_probability"
                ] < 0.0
            )
            | (
                predictions[
                    "predicted_dem_win_probability"
                ] > 1.0
            )
        ).sum()
    )

    validation_passed = (
        len(predictions) == len(model_data)
        and missing_predictions == 0
        and duplicate_predictions == 0
        and incorrect_holdout_assignments == 0
        and in_sample_leakage_rows == 0
        and probability_out_of_bounds == 0
        and len(fold_metrics) == len(cycles)
    )

    validation_status = (
        "PASSED"
        if validation_passed
        else "FAILED"
    )

    output_columns = [
        "race_id",
        "cycle",
        "holdout_cycle",
        "state",
        "election_type",
        "special_election",
        "presidential_margin_dem",
        "actual_margin_dem",
        "layer0_intercept",
        "layer0_presidential_slope",
        "training_residual_sd",
        "predicted_margin_dem",
        "prediction_error_dem",
        "absolute_error",
        "predicted_dem_win_probability",
        "predicted_winner_party",
        "actual_winner_party",
        "winner_correct",
    ]

    predictions = (
        predictions[output_columns]
        .sort_values(
            [
                "cycle",
                "state",
                "race_id",
            ]
        )
        .reset_index(drop=True)
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    fold_metrics.to_csv(
        FOLDS_PATH,
        index=False,
    )

    summary = {
        "model": (
            "layer0_presidential_baseline"
        ),
        "validation_method": (
            "leave_one_cycle_out"
        ),
        "input_path": str(INPUT_PATH),
        "input_sha256": source_hash,
        "cycles": [
            int(cycle)
            for cycle in cycles
        ],
        "scorable_races": int(
            len(model_data)
        ),
        "fold_count": int(
            len(fold_metrics)
        ),
        "overall_metrics": (
            overall_metrics
        ),
        "mean_fold_coefficients": {
            "intercept": float(
                fold_metrics[
                    "intercept"
                ].mean()
            ),
            "presidential_slope": float(
                fold_metrics[
                    "presidential_slope"
                ].mean()
            ),
            "training_residual_sd": float(
                fold_metrics[
                    "training_residual_sd"
                ].mean()
            ),
        },
        "validation_status": (
            validation_status
        ),
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )

    validation_lines = [
        (
            "Senate Layer 0 Presidential-Baseline "
            "LOCO Backtest"
        ),
        "=" * 60,
        "",
        f"Input: {INPUT_PATH}",
        f"Input SHA-256: {source_hash}",
        "",
        (
            "Cycles: "
            + ", ".join(
                map(str, cycles)
            )
        ),
        (
            "Scorable historical races: "
            f"{len(model_data)}"
        ),
        f"Completed folds: {len(fold_metrics)}",
        (
            "Out-of-sample predictions: "
            f"{len(predictions)}"
        ),
        (
            "Missing predictions: "
            f"{missing_predictions}"
        ),
        (
            "Duplicate race predictions: "
            f"{duplicate_predictions}"
        ),
        (
            "Incorrect holdout assignments: "
            f"{incorrect_holdout_assignments}"
        ),
        (
            "Probability bounds failures: "
            f"{probability_out_of_bounds}"
        ),
        "",
        "Overall out-of-sample metrics:",
        (
            f"MAE: "
            f"{overall_metrics['mae']:.4f}"
        ),
        (
            f"RMSE: "
            f"{overall_metrics['rmse']:.4f}"
        ),
        (
            f"Brier: "
            f"{overall_metrics['brier']:.5f}"
        ),
        (
            "Directional accuracy: "
            f"{overall_metrics['directional_accuracy']:.4%}"
        ),
        (
            f"Out-of-sample R-squared: "
            f"{overall_metrics['r_squared']:.4f}"
        ),
        "",
        "Fold metrics:",
        fold_metrics.to_string(
            index=False
        ),
        "",
        (
            "All predictions are generated using "
            "coefficients estimated without the held-out "
            "election cycle."
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
    print(f"Wrote: {PREDICTIONS_PATH}")
    print(f"Wrote: {FOLDS_PATH}")
    print(f"Wrote: {SUMMARY_PATH}")
    print(f"Wrote: {VALIDATION_PATH}")

    if not validation_passed:
        raise SystemExit(
            "Layer 0 backtest validation failed."
        )


if __name__ == "__main__":
    main()
