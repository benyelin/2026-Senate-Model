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
    "national_environment_sensitivity"
)

SUMMARY_CSV_PATH = (
    OUTPUT_DIR
    / "senate_national_environment_model_summary.csv"
)

FOLD_CSV_PATH = (
    OUTPUT_DIR
    / "senate_national_environment_fold_metrics.csv"
)

PREDICTIONS_CSV_PATH = (
    OUTPUT_DIR
    / "senate_national_environment_predictions.csv"
)

COEFFICIENTS_CSV_PATH = (
    OUTPUT_DIR
    / "senate_national_environment_coefficients.csv"
)

SUMMARY_JSON_PATH = (
    OUTPUT_DIR
    / "senate_national_environment_summary.json"
)

VALIDATION_PATH = (
    OUTPUT_DIR
    / "senate_national_environment_validation.txt"
)

REQUIRED_COLUMNS = [
    "race_id",
    "cycle",
    "state",
    "actual_margin_dem",
    "presidential_margin_dem",
    "generic_ballot_margin_dem",
    "presidential_approval",
    "presidential_disapproval",
    "approval_net_dem_oriented",
    "president_party_dem",
    "midterm_indicator",
    "historical_fundamentals_scorable",
]


MODEL_SPECS = {
    "baseline_only": {
        "features": [
            "presidential_margin_dem",
        ],
        "fixed_midterm_bonus": False,
    },
    "generic_ballot": {
        "features": [
            "presidential_margin_dem",
            "generic_ballot_margin_dem",
        ],
        "fixed_midterm_bonus": False,
    },
    "approval_net_only": {
        "features": [
            "presidential_margin_dem",
            "approval_net_dem_oriented",
        ],
        "fixed_midterm_bonus": False,
    },
    "generic_ballot_plus_approval_net": {
        "features": [
            "presidential_margin_dem",
            "generic_ballot_margin_dem",
            "approval_net_dem_oriented",
        ],
        "fixed_midterm_bonus": False,
    },
    "generic_ballot_plus_fixed_midterm_1pt": {
        "features": [
            "presidential_margin_dem",
            "generic_ballot_margin_dem",
        ],
        "fixed_midterm_bonus": True,
    },
    "generic_ballot_plus_estimated_midterm": {
        "features": [
            "presidential_margin_dem",
            "generic_ballot_margin_dem",
            "midterm_outparty_dem",
        ],
        "fixed_midterm_bonus": False,
    },
    "generic_ballot_plus_approval_plus_fixed_midterm": {
        "features": [
            "presidential_margin_dem",
            "generic_ballot_margin_dem",
            "approval_net_dem_oriented",
        ],
        "fixed_midterm_bonus": True,
    },
    "approval_and_disapproval_separate": {
        "features": [
            "presidential_margin_dem",
            "approval_centered_dem_oriented",
            "disapproval_centered_dem_oriented",
        ],
        "fixed_midterm_bonus": False,
    },
    "composite_environment_index": {
        "features": [
            "presidential_margin_dem",
            "national_environment_composite",
        ],
        "fixed_midterm_bonus": False,
    },
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


def predict_ols(
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
        len(actual) - parameter_count,
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

    actual_wins = (
        actual > 0.0
    ).astype(float)

    predicted_wins = (
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

    squared_error = float(
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
                    - actual_wins
                ) ** 2
            )
        ),
        "directional_accuracy": float(
            np.mean(
                predicted_wins
                == actual_wins
            )
        ),
        "r_squared": (
            float(
                1.0
                - squared_error
                / total_variation
            )
            if total_variation > 0.0
            else float("nan")
        ),
    }


def add_derived_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    result = data.copy()

    democratic_president = (
        result["president_party_dem"] > 0
    )

    republican_president = (
        result["president_party_dem"] < 0
    )

    result[
        "midterm_outparty_dem"
    ] = np.select(
        [
            (
                result["midterm_indicator"] == 1
            )
            & democratic_president,
            (
                result["midterm_indicator"] == 1
            )
            & republican_president,
        ],
        [
            -1.0,
            1.0,
        ],
        default=0.0,
    )

    approval_centered = (
        result["presidential_approval"]
        - 50.0
    )

    disapproval_centered = (
        result["presidential_disapproval"]
        - 50.0
    )

    president_orientation = np.where(
        democratic_president,
        1.0,
        -1.0,
    )

    result[
        "approval_centered_dem_oriented"
    ] = (
        approval_centered
        * president_orientation
    )

    result[
        "disapproval_centered_dem_oriented"
    ] = (
        disapproval_centered
        * -president_orientation
    )

    return result


def add_fold_composite(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()

    cycle_train = (
        train[
            [
                "cycle",
                "generic_ballot_margin_dem",
                "approval_net_dem_oriented",
            ]
        ]
        .drop_duplicates("cycle")
        .copy()
    )

    gb_mean = float(
        cycle_train[
            "generic_ballot_margin_dem"
        ].mean()
    )

    gb_sd = float(
        cycle_train[
            "generic_ballot_margin_dem"
        ].std(ddof=0)
    )

    approval_mean = float(
        cycle_train[
            "approval_net_dem_oriented"
        ].mean()
    )

    approval_sd = float(
        cycle_train[
            "approval_net_dem_oriented"
        ].std(ddof=0)
    )

    gb_sd = max(gb_sd, 1e-6)
    approval_sd = max(approval_sd, 1e-6)

    for frame in [train, test]:
        frame[
            "generic_ballot_z"
        ] = (
            frame[
                "generic_ballot_margin_dem"
            ]
            - gb_mean
        ) / gb_sd

        frame[
            "approval_net_z"
        ] = (
            frame[
                "approval_net_dem_oriented"
            ]
            - approval_mean
        ) / approval_sd

        frame[
            "national_environment_composite"
        ] = (
            frame["generic_ballot_z"]
            + frame["approval_net_z"]
        ) / 2.0

    return train, test


def prepare_model_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    numeric_columns = [
        "cycle",
        "actual_margin_dem",
        "presidential_margin_dem",
        "generic_ballot_margin_dem",
        "presidential_approval",
        "presidential_disapproval",
        "approval_net_dem_oriented",
        "president_party_dem",
        "midterm_indicator",
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
        & data[
            "presidential_approval"
        ].notna()
        & data[
            "presidential_disapproval"
        ].notna()
        & data[
            "approval_net_dem_oriented"
        ].notna()
        & data[
            "president_party_dem"
        ].notna()
        & data[
            "midterm_indicator"
        ].notna()
    ].copy()

    model_data["cycle"] = (
        model_data["cycle"].astype(int)
    )

    model_data = add_derived_features(
        model_data
    )

    return model_data.sort_values(
        [
            "cycle",
            "state",
            "race_id",
        ]
    ).reset_index(drop=True)


def run_model(
    model_name: str,
    spec: dict,
    model_data: pd.DataFrame,
    cycles: list[int],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, float],
]:
    prediction_frames = []
    fold_rows = []
    coefficient_rows = []

    for holdout_cycle in cycles:
        train = model_data.loc[
            model_data["cycle"]
            != holdout_cycle
        ].copy()

        test = model_data.loc[
            model_data["cycle"]
            == holdout_cycle
        ].copy()

        train, test = add_fold_composite(
            train,
            test,
        )

        feature_columns = spec[
            "features"
        ]

        fixed_midterm_bonus = bool(
            spec["fixed_midterm_bonus"]
        )

        x_train = train[
            feature_columns
        ].to_numpy(dtype=float)

        x_test = test[
            feature_columns
        ].to_numpy(dtype=float)

        y_train_actual = train[
            "actual_margin_dem"
        ].to_numpy(dtype=float)

        y_test = test[
            "actual_margin_dem"
        ].to_numpy(dtype=float)

        train_offset = np.zeros(
            len(train),
            dtype=float,
        )

        test_offset = np.zeros(
            len(test),
            dtype=float,
        )

        if fixed_midterm_bonus:
            train_offset = train[
                "midterm_outparty_dem"
            ].to_numpy(dtype=float)

            test_offset = test[
                "midterm_outparty_dem"
            ].to_numpy(dtype=float)

        y_train_for_fit = (
            y_train_actual
            - train_offset
        )

        coefficients = fit_ols(
            x_train,
            y_train_for_fit,
        )

        train_predictions = (
            predict_ols(
                x_train,
                coefficients,
            )
            + train_offset
        )

        test_predictions = (
            predict_ols(
                x_test,
                coefficients,
            )
            + test_offset
        )

        parameter_count = (
            len(feature_columns)
            + 1
        )

        residual_sd = residual_standard_deviation(
            actual=y_train_actual,
            predicted=train_predictions,
            parameter_count=parameter_count,
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

        fold_metrics = calculate_metrics(
            actual=y_test,
            predicted=test_predictions,
            probabilities=probabilities,
        )

        test_output = test[
            [
                "race_id",
                "cycle",
                "state",
                "actual_margin_dem",
                "presidential_margin_dem",
                "generic_ballot_margin_dem",
                "presidential_approval",
                "presidential_disapproval",
                "approval_net_dem_oriented",
                "midterm_outparty_dem",
            ]
        ].copy()

        test_output[
            "model_name"
        ] = model_name

        test_output[
            "holdout_cycle"
        ] = holdout_cycle

        test_output[
            "predicted_margin_dem"
        ] = test_predictions

        test_output[
            "predicted_dem_win_probability"
        ] = probabilities

        test_output[
            "prediction_error_dem"
        ] = (
            test_predictions
            - y_test
        )

        test_output[
            "absolute_error"
        ] = np.abs(
            test_predictions
            - y_test
        )

        test_output[
            "winner_correct"
        ] = (
            (
                test_predictions > 0.0
            )
            == (
                y_test > 0.0
            )
        )

        prediction_frames.append(
            test_output
        )

        fold_rows.append(
            {
                "model_name": model_name,
                "holdout_cycle": holdout_cycle,
                "training_rows": len(train),
                "test_rows": len(test),
                "training_residual_sd": residual_sd,
                **fold_metrics,
            }
        )

        coefficient_rows.append(
            {
                "model_name": model_name,
                "holdout_cycle": holdout_cycle,
                "term": "intercept",
                "coefficient": float(
                    coefficients[0]
                ),
            }
        )

        for feature, coefficient in zip(
            feature_columns,
            coefficients[1:],
        ):
            coefficient_rows.append(
                {
                    "model_name": model_name,
                    "holdout_cycle": holdout_cycle,
                    "term": feature,
                    "coefficient": float(
                        coefficient
                    ),
                }
            )

        if fixed_midterm_bonus:
            coefficient_rows.append(
                {
                    "model_name": model_name,
                    "holdout_cycle": holdout_cycle,
                    "term": (
                        "fixed_midterm_outparty_bonus"
                    ),
                    "coefficient": 1.0,
                }
            )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    folds = pd.DataFrame(
        fold_rows
    )

    coefficients = pd.DataFrame(
        coefficient_rows
    )

    overall = calculate_metrics(
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

    return (
        predictions,
        folds,
        coefficients,
        overall,
    )


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            INPUT_PATH
        )

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

    model_data = prepare_model_data(
        data
    )

    cycles = sorted(
        model_data[
            "cycle"
        ].unique().tolist()
    )

    if len(cycles) < 3:
        raise ValueError(
            "Too few election cycles for LOCO validation."
        )

    all_predictions = []
    all_folds = []
    all_coefficients = []
    summary_rows = []

    for model_name, spec in MODEL_SPECS.items():
        (
            predictions,
            folds,
            coefficients,
            overall,
        ) = run_model(
            model_name=model_name,
            spec=spec,
            model_data=model_data,
            cycles=cycles,
        )

        all_predictions.append(
            predictions
        )

        all_folds.append(
            folds
        )

        all_coefficients.append(
            coefficients
        )

        summary_rows.append(
            {
                "model_name": model_name,
                "mae": overall["mae"],
                "rmse": overall["rmse"],
                "brier": overall["brier"],
                "directional_accuracy": (
                    overall[
                        "directional_accuracy"
                    ]
                ),
                "r_squared": (
                    overall["r_squared"]
                ),
            }
        )

    predictions_all = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    folds_all = pd.concat(
        all_folds,
        ignore_index=True,
    )

    coefficients_all = pd.concat(
        all_coefficients,
        ignore_index=True,
    )

    summary = pd.DataFrame(
        summary_rows
    )

    baseline_row = summary.loc[
        summary["model_name"]
        == "baseline_only"
    ].iloc[0]

    generic_row = summary.loc[
        summary["model_name"]
        == "generic_ballot"
    ].iloc[0]

    for metric in [
        "mae",
        "rmse",
        "brier",
        "directional_accuracy",
        "r_squared",
    ]:
        summary[
            f"{metric}_change_vs_baseline"
        ] = (
            summary[metric]
            - baseline_row[metric]
        )

        summary[
            f"{metric}_change_vs_generic"
        ] = (
            summary[metric]
            - generic_row[metric]
        )

    summary[
        "rank_mae"
    ] = summary[
        "mae"
    ].rank(
        method="min",
        ascending=True,
    ).astype(int)

    summary[
        "rank_rmse"
    ] = summary[
        "rmse"
    ].rank(
        method="min",
        ascending=True,
    ).astype(int)

    summary[
        "rank_brier"
    ] = summary[
        "brier"
    ].rank(
        method="min",
        ascending=True,
    ).astype(int)

    summary[
        "rank_directional_accuracy"
    ] = summary[
        "directional_accuracy"
    ].rank(
        method="min",
        ascending=False,
    ).astype(int)

    summary[
        "rank_r_squared"
    ] = summary[
        "r_squared"
    ].rank(
        method="min",
        ascending=False,
    ).astype(int)

    summary[
        "average_rank"
    ] = summary[
        [
            "rank_mae",
            "rank_rmse",
            "rank_brier",
            "rank_directional_accuracy",
            "rank_r_squared",
        ]
    ].mean(axis=1)

    summary = summary.sort_values(
        [
            "average_rank",
            "brier",
            "mae",
        ]
    ).reset_index(drop=True)

    expected_predictions = (
        len(model_data)
        * len(MODEL_SPECS)
    )

    duplicate_model_races = int(
        predictions_all.groupby(
            [
                "model_name",
                "race_id",
            ]
        ).size().gt(1).sum()
    )

    incorrect_holdouts = int(
        (
            predictions_all[
                "cycle"
            ]
            != predictions_all[
                "holdout_cycle"
            ]
        ).sum()
    )

    probability_failures = int(
        (
            (
                predictions_all[
                    "predicted_dem_win_probability"
                ] < 0.0
            )
            | (
                predictions_all[
                    "predicted_dem_win_probability"
                ] > 1.0
            )
        ).sum()
    )

    validation_passed = (
        len(predictions_all)
        == expected_predictions
        and duplicate_model_races == 0
        and incorrect_holdouts == 0
        and probability_failures == 0
        and len(summary)
        == len(MODEL_SPECS)
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
        SUMMARY_CSV_PATH,
        index=False,
    )

    folds_all.sort_values(
        [
            "model_name",
            "holdout_cycle",
        ]
    ).to_csv(
        FOLD_CSV_PATH,
        index=False,
    )

    predictions_all.sort_values(
        [
            "model_name",
            "cycle",
            "state",
            "race_id",
        ]
    ).to_csv(
        PREDICTIONS_CSV_PATH,
        index=False,
    )

    coefficients_all.sort_values(
        [
            "model_name",
            "holdout_cycle",
            "term",
        ]
    ).to_csv(
        COEFFICIENTS_CSV_PATH,
        index=False,
    )

    best_mae = (
        summary.sort_values(
            "mae"
        ).iloc[0]["model_name"]
    )

    best_rmse = (
        summary.sort_values(
            "rmse"
        ).iloc[0]["model_name"]
    )

    best_brier = (
        summary.sort_values(
            "brier"
        ).iloc[0]["model_name"]
    )

    best_average_rank = (
        summary.iloc[0][
            "model_name"
        ]
    )

    summary_json = {
        "input_path": str(INPUT_PATH),
        "input_sha256": source_hash,
        "cycles": cycles,
        "scorable_races": len(model_data),
        "models_tested": list(
            MODEL_SPECS.keys()
        ),
        "best_mae_model": best_mae,
        "best_rmse_model": best_rmse,
        "best_brier_model": best_brier,
        "best_average_rank_model": (
            best_average_rank
        ),
        "validation_status": (
            validation_status
        ),
    }

    SUMMARY_JSON_PATH.write_text(
        json.dumps(
            summary_json,
            indent=2,
        )
        + "\n"
    )

    display_columns = [
        "model_name",
        "mae",
        "rmse",
        "brier",
        "directional_accuracy",
        "r_squared",
        "mae_change_vs_generic",
        "rmse_change_vs_generic",
        "brier_change_vs_generic",
        "average_rank",
    ]

    validation_lines = [
        (
            "Senate National-Environment "
            "Sensitivity Backtest"
        ),
        "=" * 76,
        "",
        f"Input: {INPUT_PATH}",
        f"Input SHA-256: {source_hash}",
        (
            "Cycles: "
            + ", ".join(map(str, cycles))
        ),
        (
            "Scorable races per model: "
            f"{len(model_data)}"
        ),
        (
            "Models tested: "
            f"{len(MODEL_SPECS)}"
        ),
        (
            "Expected predictions: "
            f"{expected_predictions}"
        ),
        (
            "Actual predictions: "
            f"{len(predictions_all)}"
        ),
        (
            "Duplicate model/race predictions: "
            f"{duplicate_model_races}"
        ),
        (
            "Incorrect holdout assignments: "
            f"{incorrect_holdouts}"
        ),
        (
            "Probability bounds failures: "
            f"{probability_failures}"
        ),
        "",
        "Overall model comparison:",
        summary[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        ),
        "",
        f"Best MAE: {best_mae}",
        f"Best RMSE: {best_rmse}",
        f"Best Brier: {best_brier}",
        (
            "Best average rank: "
            f"{best_average_rank}"
        ),
        "",
        (
            "Negative MAE/RMSE/Brier changes versus "
            "generic_ballot indicate improvement."
        ),
        (
            "The fixed midterm model applies exactly "
            "one Democratic-margin point toward the "
            "president's out-party in midterm years."
        ),
        (
            "Composite standardization is estimated "
            "using training cycles only in each fold."
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
    print(f"Wrote: {SUMMARY_CSV_PATH}")
    print(f"Wrote: {FOLD_CSV_PATH}")
    print(f"Wrote: {PREDICTIONS_CSV_PATH}")
    print(f"Wrote: {COEFFICIENTS_CSV_PATH}")
    print(f"Wrote: {SUMMARY_JSON_PATH}")
    print(f"Wrote: {VALIDATION_PATH}")

    if not validation_passed:
        raise SystemExit(
            "National-environment sensitivity "
            "validation failed."
        )


if __name__ == "__main__":
    main()
