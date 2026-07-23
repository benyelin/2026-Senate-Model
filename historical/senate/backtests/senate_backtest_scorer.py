#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SenateBacktestScore:
    scored_predictions: pd.DataFrame
    overall_metrics: pd.DataFrame
    grouped_metrics: dict[str, pd.DataFrame]


def _require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
) -> None:
    missing = [
        column
        for column in columns
        if column not in frame.columns
    ]

    if missing:
        raise KeyError(
            "Missing required columns: "
            + ", ".join(missing)
        )


def _safe_float(value: object) -> float:
    numeric = pd.to_numeric(
        pd.Series([value]),
        errors="coerce",
    ).iloc[0]

    if pd.isna(numeric):
        return np.nan

    return float(numeric)


def _metric_record(
    frame: pd.DataFrame,
    *,
    predicted_margin_column: str,
    actual_margin_column: str,
) -> dict[str, float | int]:
    predicted = pd.to_numeric(
        frame[predicted_margin_column],
        errors="coerce",
    )

    actual = pd.to_numeric(
        frame[actual_margin_column],
        errors="coerce",
    )

    valid = predicted.notna() & actual.notna()

    predicted = predicted.loc[valid]
    actual = actual.loc[valid]

    if predicted.empty:
        return {
            "observations": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "mean_error": np.nan,
            "median_absolute_error": np.nan,
            "winner_accuracy": np.nan,
            "dem_win_accuracy": np.nan,
            "gop_win_accuracy": np.nan,
            "predicted_dem_win_rate": np.nan,
            "actual_dem_win_rate": np.nan,
        }

    error = predicted - actual
    absolute_error = error.abs()

    predicted_dem_win = predicted > 0
    actual_dem_win = actual > 0

    winner_correct = (
        predicted_dem_win
        == actual_dem_win
    )

    dem_rows = actual_dem_win
    gop_rows = ~actual_dem_win

    return {
        "observations": int(
            len(predicted)
        ),
        "mae": float(
            absolute_error.mean()
        ),
        "rmse": float(
            np.sqrt(
                np.square(error).mean()
            )
        ),
        "mean_error": float(
            error.mean()
        ),
        "median_absolute_error": float(
            absolute_error.median()
        ),
        "winner_accuracy": float(
            winner_correct.mean()
        ),
        "dem_win_accuracy": (
            float(
                winner_correct.loc[
                    dem_rows
                ].mean()
            )
            if dem_rows.any()
            else np.nan
        ),
        "gop_win_accuracy": (
            float(
                winner_correct.loc[
                    gop_rows
                ].mean()
            )
            if gop_rows.any()
            else np.nan
        ),
        "predicted_dem_win_rate": float(
            predicted_dem_win.mean()
        ),
        "actual_dem_win_rate": float(
            actual_dem_win.mean()
        ),
    }


def prepare_scored_predictions(
    predictions: pd.DataFrame,
    *,
    predicted_margin_column: str,
    actual_margin_column: str = "actual_margin_dem",
) -> pd.DataFrame:
    _require_columns(
        predictions,
        [
            predicted_margin_column,
            actual_margin_column,
        ],
    )

    scored = predictions.copy()

    scored[
        "scored_predicted_margin_dem"
    ] = pd.to_numeric(
        scored[predicted_margin_column],
        errors="coerce",
    )

    scored[
        "scored_actual_margin_dem"
    ] = pd.to_numeric(
        scored[actual_margin_column],
        errors="coerce",
    )

    scored[
        "scored_error_dem"
    ] = (
        scored[
            "scored_predicted_margin_dem"
        ]
        - scored[
            "scored_actual_margin_dem"
        ]
    )

    scored[
        "scored_absolute_error"
    ] = scored[
        "scored_error_dem"
    ].abs()

    scored[
        "scored_squared_error"
    ] = np.square(
        scored[
            "scored_error_dem"
        ]
    )

    scored[
        "scored_predicted_winner_party"
    ] = np.where(
        scored[
            "scored_predicted_margin_dem"
        ]
        > 0,
        "D",
        "R",
    )

    scored[
        "scored_actual_winner_party"
    ] = np.where(
        scored[
            "scored_actual_margin_dem"
        ]
        > 0,
        "D",
        "R",
    )

    valid = (
        scored[
            "scored_predicted_margin_dem"
        ].notna()
        & scored[
            "scored_actual_margin_dem"
        ].notna()
    )

    scored[
        "scored_winner_correct"
    ] = pd.NA

    scored.loc[
        valid,
        "scored_winner_correct",
    ] = (
        scored.loc[
            valid,
            "scored_predicted_winner_party",
        ]
        == scored.loc[
            valid,
            "scored_actual_winner_party",
        ]
    )

    scored[
        "included_in_score"
    ] = valid

    return scored


def score_overall(
    scored_predictions: pd.DataFrame,
    *,
    model_name: str,
) -> pd.DataFrame:
    metrics = _metric_record(
        scored_predictions,
        predicted_margin_column=(
            "scored_predicted_margin_dem"
        ),
        actual_margin_column=(
            "scored_actual_margin_dem"
        ),
    )

    return pd.DataFrame(
        [
            {
                "model_name": model_name,
                **metrics,
            }
        ]
    )


def score_by_group(
    scored_predictions: pd.DataFrame,
    *,
    group_columns: list[str],
    model_name: str,
) -> pd.DataFrame:
    _require_columns(
        scored_predictions,
        group_columns,
    )

    records: list[dict[str, object]] = []

    grouper: str | list[str]

    if len(group_columns) == 1:
        grouper = group_columns[0]
    else:
        grouper = group_columns

    for group_values, group in (
        scored_predictions.groupby(
            grouper,
            dropna=False,
            sort=True,
        )
    ):
        if len(group_columns) == 1:
            # Depending on the pandas version and grouper form,
            # a single grouping key may be returned either as a
            # scalar or as a one-element tuple. Normalize it to a
            # one-element tuple without double-wrapping.
            if isinstance(group_values, tuple):
                if len(group_values) != 1:
                    raise ValueError(
                        "Expected exactly one group value for "
                        f"{group_columns}, got {group_values!r}."
                    )
            else:
                group_values = (
                    group_values,
                )

        record = {
            "model_name": model_name,
        }

        for column, value in zip(
            group_columns,
            group_values,
        ):
            record[column] = value

        record.update(
            _metric_record(
                group,
                predicted_margin_column=(
                    "scored_predicted_margin_dem"
                ),
                actual_margin_column=(
                    "scored_actual_margin_dem"
                ),
            )
        )

        records.append(record)

    return pd.DataFrame(records)


def run_senate_backtest_scoring(
    predictions: pd.DataFrame,
    *,
    predicted_margin_column: str,
    actual_margin_column: str = (
        "actual_margin_dem"
    ),
    model_name: str,
    groupings: dict[str, list[str]]
    | None = None,
) -> SenateBacktestScore:
    scored = prepare_scored_predictions(
        predictions,
        predicted_margin_column=(
            predicted_margin_column
        ),
        actual_margin_column=(
            actual_margin_column
        ),
    )

    included = scored.loc[
        scored[
            "included_in_score"
        ]
    ].copy()

    overall = score_overall(
        included,
        model_name=model_name,
    )

    grouped: dict[str, pd.DataFrame] = {}

    if groupings:
        for label, columns in (
            groupings.items()
        ):
            grouped[label] = score_by_group(
                included,
                group_columns=columns,
                model_name=model_name,
            )

    return SenateBacktestScore(
        scored_predictions=scored,
        overall_metrics=overall,
        grouped_metrics=grouped,
    )
