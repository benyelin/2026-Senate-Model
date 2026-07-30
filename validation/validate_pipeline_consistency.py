from __future__ import annotations

from pathlib import Path

import pandas as pd

from .models import ValidationResult
from .utils import (
    DEFAULT_TOLERANCE,
    maximum_absolute_error,
    read_csv,
)


POLLING_PATH = Path("inputs/polling_averages_generated.csv")
BAYESIAN_PATH = Path("inputs/bayesian_update_generated.csv")
RACE_INPUTS_PATH = Path("inputs/race_inputs.csv")
RACE_STATS_PATH = Path("outputs/race_stats.csv")


def _compare_state_fields(
    result: ValidationResult,
    *,
    name: str,
    left: pd.DataFrame,
    left_column: str,
    right: pd.DataFrame,
    right_column: str,
    tolerance: float,
) -> None:
    if left_column not in left.columns:
        result.add(
            name,
            False,
            severity="warning",
            details=f"Missing left column: {left_column}",
        )
        return

    if right_column not in right.columns:
        result.add(
            name,
            False,
            severity="warning",
            details=f"Missing right column: {right_column}",
        )
        return

    comparison = (
        left[["state", left_column]]
        .merge(
            right[["state", right_column]],
            on="state",
            how="inner",
            suffixes=("_left", "_right"),
            validate="one_to_one",
        )
    )

    left_name = (
        f"{left_column}_left"
        if left_column == right_column
        else left_column
    )
    right_name = (
        f"{right_column}_right"
        if left_column == right_column
        else right_column
    )

    max_error, rows = maximum_absolute_error(
        comparison[left_name],
        comparison[right_name],
    )

    result.add(
        name,
        rows > 0 and max_error <= tolerance,
        max_error=max_error,
        rows_checked=rows,
    )


def validate_pipeline_consistency(
    tolerance: float = DEFAULT_TOLERANCE,
) -> ValidationResult:
    result = ValidationResult("Cross-File Pipeline Consistency")

    polling = read_csv(POLLING_PATH, required=False)
    bayes = read_csv(BAYESIAN_PATH, required=False)
    races = read_csv(RACE_INPUTS_PATH, required=False)
    stats = read_csv(RACE_STATS_PATH, required=False)

    for label, path, frame in [
        ("Polling averages", POLLING_PATH, polling),
        ("Bayesian output", BAYESIAN_PATH, bayes),
        ("Race inputs", RACE_INPUTS_PATH, races),
        ("Race stats", RACE_STATS_PATH, stats),
    ]:
        result.add(
            f"{label} file exists and is nonempty",
            not frame.empty,
            details=str(path),
        )

    if not polling.empty and not bayes.empty:
        _compare_state_fields(
            result,
            name=(
                "Polling margin propagates from generated averages "
                "to Bayesian input"
            ),
            left=polling,
            left_column="polling_margin_dem",
            right=bayes,
            right_column="polling_margin_used",
            tolerance=tolerance,
        )

        for column in [
            "poll_count",
            "total_poll_weight",
            "avg_poll_age_days",
        ]:
            _compare_state_fields(
                result,
                name=(
                    f"{column} propagates from generated averages "
                    "to Bayesian output"
                ),
                left=polling,
                left_column=column,
                right=bayes,
                right_column=column,
                tolerance=tolerance,
            )

    if not bayes.empty and not races.empty:
        final_weight_column = next(
            (
                column
                for column in [
                    (
                        "bayesian_polling_weight_capped_after_"
                        "polling_confidence_accelerator"
                    ),
                    "bayesian_polling_weight_capped",
                    "bayesian_polling_weight",
                ]
                if column in bayes.columns
            ),
            None,
        )

        final_margin_column = next(
            (
                column
                for column in [
                    "bayesian_model_margin_dem_capped",
                    "bayesian_model_margin_dem",
                    "posterior_margin_dem",
                ]
                if column in bayes.columns
            ),
            None,
        )

        if final_weight_column:
            _compare_state_fields(
                result,
                name="Final polling weight propagates into race inputs",
                left=bayes,
                left_column=final_weight_column,
                right=races,
                right_column="bayesian_polling_weight",
                tolerance=tolerance,
            )

        if final_margin_column:
            _compare_state_fields(
                result,
                name="Final Bayesian margin propagates into race inputs",
                left=bayes,
                left_column=final_margin_column,
                right=races,
                right_column="bayesian_model_margin_dem",
                tolerance=tolerance,
            )

        for column in [
            "poll_count",
            "polling_margin_used",
            "original_bayesian_polling_weight",
            "poll_count_weight_multiplier",
            "cycle_max_polling_weight",
        ]:
            if column in bayes.columns and column in races.columns:
                _compare_state_fields(
                    result,
                    name=f"{column} propagates into race inputs",
                    left=bayes,
                    left_column=column,
                    right=races,
                    right_column=column,
                    tolerance=tolerance,
                )

    if not races.empty and not stats.empty:
        race_margin_column = next(
            (
                column
                for column in [
                    "bayesian_model_margin_dem",
                    "model_margin_dem",
                    "fundamentals_margin_dem",
                ]
                if column in races.columns
            ),
            None,
        )

        stats_source_column = next(
            (
                column
                for column in [
                    "bayesian_model_margin_dem",
                    "model_margin_dem",
                ]
                if column in stats.columns
            ),
            None,
        )

        if race_margin_column and stats_source_column:
            _compare_state_fields(
                result,
                name="Authoritative model margin propagates to race stats",
                left=races,
                left_column=race_margin_column,
                right=stats,
                right_column=stats_source_column,
                tolerance=tolerance,
            )
        else:
            result.add(
                "Authoritative model margin propagates to race stats",
                False,
                severity="warning",
                details=(
                    "Could not identify compatible model-margin columns "
                    "in race inputs and race stats."
                ),
            )

        if (
            "bayesian_polling_weight" in races.columns
            and "bayesian_polling_weight" in stats.columns
        ):
            _compare_state_fields(
                result,
                name="Final polling weight propagates to race stats",
                left=races,
                left_column="bayesian_polling_weight",
                right=stats,
                right_column="bayesian_polling_weight",
                tolerance=tolerance,
            )

    return result
