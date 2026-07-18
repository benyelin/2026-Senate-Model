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

LAYER1_SUMMARY_PATH = (
    PROJECT_ROOT
    / "historical/senate/backtests/outputs/"
    "layer1_generic_ballot/"
    "senate_layer1_loco_summary.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "historical/senate/backtests/outputs/"
    "layer2_approval"
)

PREDICTIONS_PATH = (
    OUTPUT_DIR
    / "senate_layer2_loco_predictions.csv"
)

FOLDS_PATH = (
    OUTPUT_DIR
    / "senate_layer2_loco_fold_metrics.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "senate_layer2_loco_summary.json"
)

VALIDATION_PATH = (
    OUTPUT_DIR
    / "senate_layer2_loco_validation.txt"
)

REQUIRED_COLUMNS = [
    "race_id",
    "cycle",
    "state",
    "election_type",
    "special_election",
    "actual_margin_dem",
    "presidential_margin_dem",
    "generic_ballot_margin_dem",
    "approval_net_dem_oriented",
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

    unexpected = (
        set(normalized.dropna().unique())
        - set(mapping)
    )

    if unexpected:
        raise ValueError(
            "Unexpected boolean values: "
            + ", ".join(sorted(map(str, unexpected)))
        )

    return (
        normalized.map(mapping)
        .fillna(False)
        .astype(bool)
    )


def fit_ols(
    features: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    design = np.column_stack(
        [
            np.ones(len(features)),
            features,
        ]
    )

    coefficients, *_ = np.linalg.lstsq(
        design,
        target,
        rcond=None,
    )

    return coefficients.astype(float)


def predict(
    features: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    design = np.column_stack(
        [
            np.ones(len(features)),
            features,
        ]
    )

    return design @ coefficients


def residual_standard_deviation(
    actual: np.ndarray,
    predicted: np.ndarray,
    parameter_count: int,
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

    return max(sqrt(variance), 1e-6)


def margin_to_probability(
    margin: float,
    residual_sd: float,
) -> float:
    return float(
        NormalDist().cdf(
            margin / residual_sd
        )
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

    total_variation = float(
        np.sum(
            (
                actual
                - np.mean(actual)
            ) ** 2
        )
    )

    residual_variation = float(
        np.sum(errors ** 2)
    )

    return {
        "mae": float(
            np.mean(np.abs(errors))
        ),
        "rmse": float(
            sqrt(np.mean(errors ** 2))
        ),
        "brier": float(
            np.mean(
                (
                    probabilities
                    - actual_dem_win
                ) ** 2
            )
        ),
        "directional_accuracy": float(
            np.mean(
                predicted_dem_win
                == actual_dem_win
            )
        ),
        "r_squared": (
            float(
                1.0
                - residual_variation
                / total_variation
            )
            if total_variation > 0.0
            else float("nan")
        ),
    }


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    if not LAYER1_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            LAYER1_SUMMARY_PATH
        )

    source_hash = sha256_file(INPUT_PATH)

    data = pd.read_csv(INPUT_PATH)

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
        "approval_net_dem_oriented",
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
        & data["presidential_margin_dem"].notna()
        & data["generic_ballot_margin_dem"].notna()
        & data["approval_net_dem_oriented"].notna()
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

    feature_columns = [
        "presidential_margin_dem",
        "generic_ballot_margin_dem",
        "approval_net_dem_oriented",
    ]

    prediction_frames = []
    fold_rows = []

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
            feature_columns
        ].to_numpy(dtype=float)

        y_train = train[
            "actual_margin_dem"
        ].to_numpy(dtype=float)

        x_test = test[
            feature_columns
        ].to_numpy(dtype=float)

        y_test = test[
            "actual_margin_dem"
        ].to_numpy(dtype=float)

        coefficients = fit_ols(
            x_train,
            y_train,
        )

        train_predictions = predict(
            x_train,
            coefficients,
        )

        residual_sd = residual_standard_deviation(
            actual=y_train,
            predicted=train_predictions,
            parameter_count=4,
        )

        test_predictions = predict(
            x_test,
            coefficients,
        )

        probabilities = np.array(
            [
                margin_to_probability(
                    float(margin),
                    residual_sd,
                )
                for margin in test_predictions
            ],
            dtype=float,
        )

        metrics = calculate_metrics(
            actual=y_test,
            predicted=test_predictions,
            probabilities=probabilities,
        )

        test["holdout_cycle"] = holdout_cycle
        test["layer2_intercept"] = coefficients[0]
        test[
            "layer2_presidential_slope"
        ] = coefficients[1]
        test[
            "layer2_generic_ballot_slope"
        ] = coefficients[2]
        test[
            "layer2_approval_slope"
        ] = coefficients[3]
        test[
            "training_residual_sd"
        ] = residual_sd
        test[
            "predicted_margin_dem"
        ] = test_predictions
        test[
            "predicted_dem_win_probability"
        ] = probabilities
        test[
            "prediction_error_dem"
        ] = (
            test["predicted_margin_dem"]
            - test["actual_margin_dem"]
        )
        test["absolute_error"] = (
            test["prediction_error_dem"].abs()
        )
        test[
            "predicted_winner_party"
        ] = np.where(
            test["predicted_margin_dem"] > 0.0,
            "D",
            "R",
        )
        test[
            "actual_winner_party"
        ] = np.where(
            test["actual_margin_dem"] > 0.0,
            "D",
            "R",
        )
        test["winner_correct"] = (
            test["predicted_winner_party"]
            == test["actual_winner_party"]
        )

        prediction_frames.append(test)

        fold_rows.append(
            {
                "holdout_cycle": holdout_cycle,
                "training_rows": len(train),
                "test_rows": len(test),
                "intercept": coefficients[0],
                "presidential_slope": coefficients[1],
                "generic_ballot_slope": coefficients[2],
                "approval_slope": coefficients[3],
                "training_residual_sd": residual_sd,
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

    layer1_summary = json.loads(
        LAYER1_SUMMARY_PATH.read_text()
    )

    layer1_metrics = layer1_summary[
        "overall_metrics"
    ]

    metric_changes = {
        "mae_change": (
            overall_metrics["mae"]
            - layer1_metrics["mae"]
        ),
        "rmse_change": (
            overall_metrics["rmse"]
            - layer1_metrics["rmse"]
        ),
        "brier_change": (
            overall_metrics["brier"]
            - layer1_metrics["brier"]
        ),
        "directional_accuracy_change": (
            overall_metrics[
                "directional_accuracy"
            ]
            - layer1_metrics[
                "directional_accuracy"
            ]
        ),
        "r_squared_change": (
            overall_metrics["r_squared"]
            - layer1_metrics["r_squared"]
        ),
    }

    prediction_count_by_race = (
        predictions.groupby("race_id").size()
    )

    missing_predictions = (
        len(model_data)
        - len(predictions)
    )

    duplicate_predictions = int(
        (
            prediction_count_by_race > 1
        ).sum()
    )

    incorrect_holdout_assignments = int(
        (
            predictions["cycle"]
            != predictions["holdout_cycle"]
        ).sum()
    )

    probability_failures = int(
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
        and probability_failures == 0
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
        "generic_ballot_margin_dem",
        "approval_net_dem_oriented",
        "actual_margin_dem",
        "layer2_intercept",
        "layer2_presidential_slope",
        "layer2_generic_ballot_slope",
        "layer2_approval_slope",
        "training_residual_sd",
        "predicted_margin_dem",
        "prediction_error_dem",
        "absolute_error",
        "predicted_dem_win_probability",
        "predicted_winner_party",
        "actual_winner_party",
        "winner_correct",
    ]

    predictions = predictions[
        output_columns
    ].sort_values(
        [
            "cycle",
            "state",
            "race_id",
        ]
    ).reset_index(drop=True)

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
        "model": "layer2_approval",
        "validation_method": (
            "leave_one_cycle_out"
        ),
        "input_path": str(INPUT_PATH),
        "input_sha256": source_hash,
        "cycles": cycles,
        "scorable_races": len(model_data),
        "overall_metrics": overall_metrics,
        "layer1_metrics": layer1_metrics,
        "changes_from_layer1": metric_changes,
        "mean_fold_coefficients": {
            "intercept": float(
                fold_metrics["intercept"].mean()
            ),
            "presidential_slope": float(
                fold_metrics[
                    "presidential_slope"
                ].mean()
            ),
            "generic_ballot_slope": float(
                fold_metrics[
                    "generic_ballot_slope"
                ].mean()
            ),
            "approval_slope": float(
                fold_metrics[
                    "approval_slope"
                ].mean()
            ),
            "training_residual_sd": float(
                fold_metrics[
                    "training_residual_sd"
                ].mean()
            ),
        },
        "validation_status": validation_status,
    }

    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    lines = [
        (
            "Senate Layer 2 Approval "
            "LOCO Backtest"
        ),
        "=" * 60,
        "",
        f"Input: {INPUT_PATH}",
        f"Input SHA-256: {source_hash}",
        "",
        (
            "Cycles: "
            + ", ".join(map(str, cycles))
        ),
        (
            "Scorable historical races: "
            f"{len(model_data)}"
        ),
        (
            "Out-of-sample predictions: "
            f"{len(predictions)}"
        ),
        f"Missing predictions: {missing_predictions}",
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
            f"{probability_failures}"
        ),
        "",
        "Overall out-of-sample metrics:",
        f"MAE: {overall_metrics['mae']:.4f}",
        f"RMSE: {overall_metrics['rmse']:.4f}",
        f"Brier: {overall_metrics['brier']:.5f}",
        (
            "Directional accuracy: "
            f"{overall_metrics['directional_accuracy']:.4%}"
        ),
        (
            "Out-of-sample R-squared: "
            f"{overall_metrics['r_squared']:.4f}"
        ),
        "",
        "Changes from Layer 1:",
        (
            "MAE change: "
            f"{metric_changes['mae_change']:+.4f}"
        ),
        (
            "RMSE change: "
            f"{metric_changes['rmse_change']:+.4f}"
        ),
        (
            "Brier change: "
            f"{metric_changes['brier_change']:+.5f}"
        ),
        (
            "Directional-accuracy change: "
            f"{metric_changes['directional_accuracy_change']:+.4%}"
        ),
        (
            "R-squared change: "
            f"{metric_changes['r_squared_change']:+.4f}"
        ),
        "",
        "Fold metrics:",
        fold_metrics.to_string(index=False),
        "",
        (
            "Approval is Democratic-oriented: "
            "positive values favor Democrats and "
            "negative values favor Republicans."
        ),
        (
            "All coefficients were estimated without "
            "the held-out election cycle."
        ),
        "",
        f"Validation: {validation_status}",
    ]

    validation_text = "\n".join(lines)

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
            "Layer 2 backtest validation failed."
        )


if __name__ == "__main__":
    main()
