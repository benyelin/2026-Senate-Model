#!/usr/bin/env python3
"""
Decompose the effects of historical bias correction and residual SD.

Nested leave-one-cycle-out models
---------------------------------

1. current_baseline
       bias = 0.00
       SD   = 8.75

2. bias_only
       bias = mean training residual
       SD   = 8.75

3. sd_only
       bias = 0.00
       SD   = training residual SD

4. closed_form_residual
       bias = mean training residual
       SD   = training residual SD

All estimated parameters use training cycles only.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]

SOURCE_SCRIPT = (
    ROOT
    / "historical/senate/backtests/"
    "run_senate_closed_form_residual_calibration.py"
)

INPUT_PATH = (
    ROOT
    / "historical/senate/backtests/outputs/canonical/"
    "senate_canonical_backtest_dataset.csv"
)

OUTPUT_DIR = (
    ROOT
    / "historical/senate/backtests/outputs/"
    "bias_sd_decomposition"
)

FIXED_PROBABILITY_SD = 8.75


def load_shared_module():
    if not SOURCE_SCRIPT.exists():
        raise FileNotFoundError(
            f"Shared calibration script does not exist: {SOURCE_SCRIPT}"
        )

    specification = importlib.util.spec_from_file_location(
        "senate_closed_form_shared",
        SOURCE_SCRIPT,
    )

    if (
        specification is None
        or specification.loader is None
    ):
        raise ImportError(
            f"Could not load shared module from {SOURCE_SCRIPT}"
        )

    module = importlib.util.module_from_spec(
        specification
    )

    specification.loader.exec_module(module)

    return module


def build_nested_predictions(
    frame: pd.DataFrame,
    shared,
) -> tuple[
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

        training_bias, training_sd = (
            shared.estimate_closed_form_parameters(
                training
            )
        )

        model_parameters = {
            "current_baseline": {
                "bias_correction_dem": 0.0,
                "global_sd": FIXED_PROBABILITY_SD,
                "bias_source": "fixed_zero",
                "sd_source": "fixed_8_75",
            },
            "bias_only": {
                "bias_correction_dem": training_bias,
                "global_sd": FIXED_PROBABILITY_SD,
                "bias_source": "training_cycles",
                "sd_source": "fixed_8_75",
            },
            "sd_only": {
                "bias_correction_dem": 0.0,
                "global_sd": training_sd,
                "bias_source": "fixed_zero",
                "sd_source": "training_cycles",
            },
            "closed_form_residual": {
                "bias_correction_dem": training_bias,
                "global_sd": training_sd,
                "bias_source": "training_cycles",
                "sd_source": "training_cycles",
            },
        }

        for model_name, parameters in (
            model_parameters.items()
        ):
            parameter_rows.append(
                {
                    "holdout_cycle": holdout_cycle,
                    "model_name": model_name,
                    "training_cycles": ",".join(
                        str(cycle)
                        for cycle in cycles
                        if cycle != holdout_cycle
                    ),
                    "training_rows": int(
                        len(training)
                    ),
                    "holdout_rows": int(
                        len(holdout)
                    ),
                    "bias_correction_dem": float(
                        parameters[
                            "bias_correction_dem"
                        ]
                    ),
                    "global_sd": float(
                        parameters["global_sd"]
                    ),
                    "bias_source": parameters[
                        "bias_source"
                    ],
                    "sd_source": parameters[
                        "sd_source"
                    ],
                    "training_residual_mean": float(
                        training[
                            "forecast_error_dem"
                        ].mean()
                    ),
                    "training_residual_sd": float(
                        training[
                            "forecast_error_dem"
                        ].std(ddof=1)
                    ),
                }
            )

            predictions = shared.build_predictions(
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
                    "nested_training_only"
                    if (
                        parameters["bias_source"]
                        == "training_cycles"
                        or parameters["sd_source"]
                        == "training_cycles"
                    )
                    else "fixed_baseline"
                ),
            )

            predictions[
                "holdout_cycle"
            ] = holdout_cycle

            predictions[
                "bias_source"
            ] = parameters["bias_source"]

            predictions[
                "sd_source"
            ] = parameters["sd_source"]

            prediction_frames.append(
                predictions
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

    return parameters, predictions


def build_aggregate_summary(
    predictions: pd.DataFrame,
    shared,
) -> pd.DataFrame:
    rows = []

    for model_name, model_frame in (
        predictions.groupby(
            "model_name",
            sort=True,
        )
    ):
        summary = shared.summarize_predictions(
            model_frame
        )

        summary["model_name"] = model_name

        summary["mean_applied_bias"] = float(
            model_frame[
                "bias_correction_dem"
            ].mean()
        )

        summary["mean_applied_sd"] = float(
            model_frame["global_sd"].mean()
        )

        rows.append(summary)

    output = pd.DataFrame(rows)

    model_order = {
        "current_baseline": 1,
        "bias_only": 2,
        "sd_only": 3,
        "closed_form_residual": 4,
    }

    output["model_order"] = output[
        "model_name"
    ].map(model_order)

    return output.sort_values(
        "model_order"
    ).drop(
        columns=["model_order"]
    )


def build_cycle_summary(
    predictions: pd.DataFrame,
    shared,
) -> pd.DataFrame:
    rows = []

    for (
        model_name,
        cycle,
    ), cycle_frame in predictions.groupby(
        ["model_name", "cycle"],
        sort=True,
    ):
        summary = shared.summarize_predictions(
            cycle_frame
        )

        summary["model_name"] = model_name
        summary["cycle"] = int(cycle)

        summary["applied_bias"] = float(
            cycle_frame[
                "bias_correction_dem"
            ].iloc[0]
        )

        summary["applied_sd"] = float(
            cycle_frame[
                "global_sd"
            ].iloc[0]
        )

        rows.append(summary)

    return pd.DataFrame(rows).sort_values(
        ["model_name", "cycle"]
    )


def build_deltas(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    baseline = summary[
        summary["model_name"]
        == "current_baseline"
    ]

    if len(baseline) != 1:
        raise RuntimeError(
            "Expected exactly one current-baseline summary row."
        )

    baseline = baseline.iloc[0]

    comparison_metrics = [
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

    rows = []

    for _, candidate in summary.iterrows():
        row = {
            "model_name": candidate[
                "model_name"
            ],
            "mean_applied_bias": candidate[
                "mean_applied_bias"
            ],
            "mean_applied_sd": candidate[
                "mean_applied_sd"
            ],
        }

        for metric in comparison_metrics:
            row[f"{metric}_value"] = float(
                candidate[metric]
            )

            row[f"{metric}_delta_vs_baseline"] = (
                float(candidate[metric])
                - float(baseline[metric])
            )

        rows.append(row)

    return pd.DataFrame(rows)


def validate_outputs(
    frame: pd.DataFrame,
    parameters: pd.DataFrame,
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
) -> dict[str, object]:
    expected_models = 4
    expected_cycles = int(
        frame["cycle"].nunique()
    )

    expected_parameter_rows = (
        expected_models * expected_cycles
    )

    expected_prediction_rows = (
        expected_models * len(frame)
    )

    validation = {
        "input_rows": int(len(frame)),
        "cycles": expected_cycles,
        "models": int(
            summary["model_name"].nunique()
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
        "nonpositive_sds": int(
            (
                predictions["global_sd"] <= 0
            ).sum()
        ),
    }

    failures = []

    if validation["models"] != expected_models:
        failures.append(
            "Expected four comparison models."
        )

    if (
        validation["parameter_rows"]
        != expected_parameter_rows
    ):
        failures.append(
            "Parameter row count does not reconcile."
        )

    if (
        validation["prediction_rows"]
        != expected_prediction_rows
    ):
        failures.append(
            "Prediction row count does not reconcile."
        )

    if validation["missing_probabilities"] != 0:
        failures.append(
            "Predictions contain missing probabilities."
        )

    if validation["nonpositive_sds"] != 0:
        failures.append(
            "Predictions contain invalid SD values."
        )

    for model_name, model_frame in (
        predictions.groupby("model_name")
    ):
        unique_keys = model_frame[
            shared_key_columns()
        ].drop_duplicates().shape[0]

        if unique_keys != len(frame):
            failures.append(
                f"{model_name} does not contain one "
                "prediction per canonical race."
            )

    if failures:
        formatted = "\n".join(
            f"  - {failure}"
            for failure in failures
        )

        raise RuntimeError(
            "Bias/SD decomposition validation FAILED:\n"
            f"{formatted}"
        )

    validation["validation_status"] = "PASSED"

    return validation


def shared_key_columns() -> list[str]:
    return ["race_id", "cycle", "state"]


def print_report(
    frame: pd.DataFrame,
    parameters: pd.DataFrame,
    summary: pd.DataFrame,
    validation: dict[str, object],
) -> None:
    print("=" * 80)
    print("SENATE BIAS AND SD DECOMPOSITION")
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

    print("\nTRAINING-ONLY ESTIMATES BY HOLDOUT CYCLE")
    closed_form_parameters = parameters[
        parameters["model_name"]
        == "closed_form_residual"
    ]

    print(
        closed_form_parameters[
            [
                "holdout_cycle",
                "training_rows",
                "holdout_rows",
                "bias_correction_dem",
                "global_sd",
            ]
        ].to_string(index=False)
    )

    print("\nAGGREGATE NESTED OUT-OF-SAMPLE COMPARISON")
    print(
        summary[
            [
                "model_name",
                "mean_applied_bias",
                "mean_applied_sd",
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
    shared = load_shared_module()

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Canonical dataset does not exist: {INPUT_PATH}"
        )

    frame = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    shared.validate_input(frame)

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

    parameters, predictions = (
        build_nested_predictions(
            frame=frame,
            shared=shared,
        )
    )

    summary = build_aggregate_summary(
        predictions=predictions,
        shared=shared,
    )

    cycle_summary = build_cycle_summary(
        predictions=predictions,
        shared=shared,
    )

    deltas = build_deltas(summary)

    validation = validate_outputs(
        frame=frame,
        parameters=parameters,
        predictions=predictions,
        summary=summary,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    parameters.to_csv(
        OUTPUT_DIR
        / "senate_bias_sd_nested_parameters.csv",
        index=False,
    )

    predictions.to_csv(
        OUTPUT_DIR
        / "senate_bias_sd_nested_predictions.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR
        / "senate_bias_sd_nested_summary.csv",
        index=False,
    )

    cycle_summary.to_csv(
        OUTPUT_DIR
        / "senate_bias_sd_nested_by_cycle.csv",
        index=False,
    )

    deltas.to_csv(
        OUTPUT_DIR
        / "senate_bias_sd_deltas_vs_baseline.csv",
        index=False,
    )

    pd.DataFrame([validation]).to_csv(
        OUTPUT_DIR
        / "senate_bias_sd_validation.csv",
        index=False,
    )

    print_report(
        frame=frame,
        parameters=parameters,
        summary=summary,
        validation=validation,
    )


if __name__ == "__main__":
    main()
