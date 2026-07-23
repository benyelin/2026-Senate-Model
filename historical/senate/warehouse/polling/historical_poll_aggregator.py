#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]

DEFAULT_SNAPSHOT_QUESTIONS_PATH = (
    PROJECT_ROOT
    / "historical"
    / "senate"
    / "warehouse"
    / "processed"
    / "polling_snapshots"
    / "senate_historical_poll_snapshot_questions.csv"
)

DEFAULT_SNAPSHOT_SUMMARY_PATH = (
    PROJECT_ROOT
    / "historical"
    / "senate"
    / "warehouse"
    / "processed"
    / "polling_snapshots"
    / "senate_historical_poll_snapshot_summary.csv"
)


DEFAULT_RECENCY_HALF_LIFE_DAYS = 14.0
DEFAULT_REFERENCE_SAMPLE_SIZE = 600.0
DEFAULT_MIN_SAMPLE_FACTOR = 0.50
DEFAULT_MAX_SAMPLE_FACTOR = 1.75


@dataclass(frozen=True)
class HistoricalPollAggregation:
    snapshot_aggregates: pd.DataFrame
    weighted_questions: pd.DataFrame
    configuration: dict[str, float]


def _resolve_column(
    frame: pd.DataFrame,
    candidates: Iterable[str],
    *,
    required: bool,
) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

    if required:
        raise KeyError(
            "Could not find any of these required columns: "
            + ", ".join(candidates)
        )

    return None


def load_snapshot_questions(
    path: Path | str = DEFAULT_SNAPSHOT_QUESTIONS_PATH,
) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        low_memory=False,
    )

    date_columns = [
        "snapshot_date",
        "snapshot_election_date",
        "election_date",
        "start_date",
        "end_date",
        "created_at",
    ]

    for column in date_columns:
        if column in frame.columns:
            frame[column] = pd.to_datetime(
                frame[column],
                errors="coerce",
            ).dt.normalize()

    return frame


def load_snapshot_summary(
    path: Path | str = DEFAULT_SNAPSHOT_SUMMARY_PATH,
) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        low_memory=False,
    )

    for column in [
        "snapshot_date",
        "election_date",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(
                frame[column],
                errors="coerce",
            ).dt.normalize()

    return frame


def calculate_recency_weight(
    days_old: pd.Series,
    *,
    half_life_days: float,
) -> pd.Series:
    if half_life_days <= 0:
        raise ValueError(
            "half_life_days must be greater than zero."
        )

    numeric_days = pd.to_numeric(
        days_old,
        errors="coerce",
    )

    numeric_days = numeric_days.clip(
        lower=0,
    )

    return np.exp(
        -np.log(2.0)
        * numeric_days
        / float(half_life_days)
    )


def calculate_sample_size_weight(
    sample_size: pd.Series,
    *,
    reference_sample_size: float,
    minimum_factor: float,
    maximum_factor: float,
) -> tuple[pd.Series, pd.Series]:
    if reference_sample_size <= 0:
        raise ValueError(
            "reference_sample_size must be greater than zero."
        )

    if minimum_factor <= 0:
        raise ValueError(
            "minimum_factor must be greater than zero."
        )

    if maximum_factor < minimum_factor:
        raise ValueError(
            "maximum_factor must be at least minimum_factor."
        )

    numeric_sample = pd.to_numeric(
        sample_size,
        errors="coerce",
    )

    valid_sample = numeric_sample.where(
        numeric_sample > 0
    )

    imputed_sample = valid_sample.fillna(
        float(reference_sample_size)
    )

    factor = np.sqrt(
        imputed_sample
        / float(reference_sample_size)
    )

    factor = factor.clip(
        lower=float(minimum_factor),
        upper=float(maximum_factor),
    )

    return factor, valid_sample


def _weighted_mean(
    values: pd.Series,
    weights: pd.Series,
) -> float:
    numeric_values = pd.to_numeric(
        values,
        errors="coerce",
    )

    numeric_weights = pd.to_numeric(
        weights,
        errors="coerce",
    )

    valid = (
        numeric_values.notna()
        & numeric_weights.notna()
        & (numeric_weights > 0)
    )

    if not valid.any():
        return np.nan

    return float(
        np.average(
            numeric_values.loc[valid],
            weights=numeric_weights.loc[valid],
        )
    )


def _weighted_standard_deviation(
    values: pd.Series,
    weights: pd.Series,
) -> float:
    numeric_values = pd.to_numeric(
        values,
        errors="coerce",
    )

    numeric_weights = pd.to_numeric(
        weights,
        errors="coerce",
    )

    valid = (
        numeric_values.notna()
        & numeric_weights.notna()
        & (numeric_weights > 0)
    )

    if valid.sum() <= 1:
        return np.nan

    values_array = numeric_values.loc[
        valid
    ].to_numpy(dtype=float)

    weights_array = numeric_weights.loc[
        valid
    ].to_numpy(dtype=float)

    mean = np.average(
        values_array,
        weights=weights_array,
    )

    variance = np.average(
        (values_array - mean) ** 2,
        weights=weights_array,
    )

    return float(
        np.sqrt(variance)
    )


def _effective_sample_size(
    weights: pd.Series,
) -> float:
    numeric_weights = pd.to_numeric(
        weights,
        errors="coerce",
    )

    numeric_weights = numeric_weights.loc[
        numeric_weights.notna()
        & (numeric_weights > 0)
    ]

    if numeric_weights.empty:
        return 0.0

    numerator = float(
        numeric_weights.sum() ** 2
    )

    denominator = float(
        np.square(
            numeric_weights
        ).sum()
    )

    if denominator <= 0:
        return 0.0

    return numerator / denominator


def prepare_weighted_questions(
    snapshot_questions: pd.DataFrame,
    *,
    recency_half_life_days: float = (
        DEFAULT_RECENCY_HALF_LIFE_DAYS
    ),
    reference_sample_size: float = (
        DEFAULT_REFERENCE_SAMPLE_SIZE
    ),
    minimum_sample_factor: float = (
        DEFAULT_MIN_SAMPLE_FACTOR
    ),
    maximum_sample_factor: float = (
        DEFAULT_MAX_SAMPLE_FACTOR
    ),
) -> pd.DataFrame:
    questions = snapshot_questions.copy()

    required_columns = [
        "snapshot_id",
        "question_id",
        "dem_margin",
        "days_old",
    ]

    missing = [
        column
        for column in required_columns
        if column not in questions.columns
    ]

    if missing:
        raise KeyError(
            "Snapshot questions are missing columns: "
            + ", ".join(missing)
        )

    sample_size_column = _resolve_column(
        questions,
        [
            "sample_size",
            "sample_size_numeric",
            "samplesize",
            "sample",
        ],
        required=False,
    )

    if sample_size_column is None:
        questions[
            "aggregation_sample_size_raw"
        ] = np.nan
    else:
        questions[
            "aggregation_sample_size_raw"
        ] = pd.to_numeric(
            questions[
                sample_size_column
            ],
            errors="coerce",
        )

    questions[
        "aggregation_dem_margin"
    ] = pd.to_numeric(
        questions["dem_margin"],
        errors="coerce",
    )

    questions[
        "aggregation_days_old"
    ] = pd.to_numeric(
        questions["days_old"],
        errors="coerce",
    )

    questions[
        "recency_weight"
    ] = calculate_recency_weight(
        questions[
            "aggregation_days_old"
        ],
        half_life_days=(
            recency_half_life_days
        ),
    )

    (
        questions["sample_size_weight"],
        valid_sample,
    ) = calculate_sample_size_weight(
        questions[
            "aggregation_sample_size_raw"
        ],
        reference_sample_size=(
            reference_sample_size
        ),
        minimum_factor=(
            minimum_sample_factor
        ),
        maximum_factor=(
            maximum_sample_factor
        ),
    )

    questions[
        "aggregation_sample_size"
    ] = valid_sample.fillna(
        float(reference_sample_size)
    )

    questions[
        "sample_size_imputed"
    ] = valid_sample.isna()

    questions[
        "raw_poll_weight"
    ] = (
        questions[
            "recency_weight"
        ]
        * questions[
            "sample_size_weight"
        ]
    )

    valid_margin = questions[
        "aggregation_dem_margin"
    ].notna()

    valid_weight = (
        questions[
            "raw_poll_weight"
        ].notna()
        & (
            questions[
                "raw_poll_weight"
            ]
            > 0
        )
    )

    questions[
        "included_in_aggregate"
    ] = (
        valid_margin
        & valid_weight
    )

    questions[
        "normalized_poll_weight"
    ] = 0.0

    included = questions.loc[
        questions[
            "included_in_aggregate"
        ]
    ]

    if not included.empty:
        weight_totals = (
            included.groupby(
                "snapshot_id"
            )[
                "raw_poll_weight"
            ]
            .transform("sum")
        )

        questions.loc[
            included.index,
            "normalized_poll_weight",
        ] = (
            included[
                "raw_poll_weight"
            ]
            / weight_totals
        )

    return questions


def aggregate_weighted_questions(
    weighted_questions: pd.DataFrame,
    snapshot_summary: pd.DataFrame,
) -> pd.DataFrame:
    included = weighted_questions.loc[
        weighted_questions[
            "included_in_aggregate"
        ]
    ].copy()

    aggregate_records: list[dict[str, object]] = []

    poll_id_column = _resolve_column(
        included,
        [
            "poll_id",
            "pollid",
        ],
        required=False,
    )

    pollster_column = _resolve_column(
        included,
        [
            "pollster",
            "pollster_name",
            "display_name",
        ],
        required=False,
    )

    for snapshot_id, group in included.groupby(
        "snapshot_id",
        sort=False,
    ):
        aggregate_records.append(
            {
                "snapshot_id": snapshot_id,
                "baseline_poll_margin_dem": (
                    _weighted_mean(
                        group[
                            "aggregation_dem_margin"
                        ],
                        group[
                            "raw_poll_weight"
                        ],
                    )
                ),
                "baseline_poll_margin_dem_unweighted": (
                    float(
                        group[
                            "aggregation_dem_margin"
                        ].mean()
                    )
                ),
                "baseline_poll_margin_median": (
                    float(
                        group[
                            "aggregation_dem_margin"
                        ].median()
                    )
                ),
                "baseline_poll_margin_weighted_sd": (
                    _weighted_standard_deviation(
                        group[
                            "aggregation_dem_margin"
                        ],
                        group[
                            "raw_poll_weight"
                        ],
                    )
                ),
                "baseline_poll_question_count": int(
                    len(group)
                ),
                "baseline_poll_count": (
                    int(
                        group[
                            poll_id_column
                        ].nunique()
                    )
                    if poll_id_column
                    else int(
                        len(group)
                    )
                ),
                "baseline_pollster_count": (
                    int(
                        group[
                            pollster_column
                        ]
                        .dropna()
                        .nunique()
                    )
                    if pollster_column
                    else np.nan
                ),
                "baseline_effective_question_count": (
                    _effective_sample_size(
                        group[
                            "raw_poll_weight"
                        ]
                    )
                ),
                "baseline_total_raw_weight": float(
                    group[
                        "raw_poll_weight"
                    ].sum()
                ),
                "baseline_mean_poll_age_days": (
                    _weighted_mean(
                        group[
                            "aggregation_days_old"
                        ],
                        group[
                            "raw_poll_weight"
                        ],
                    )
                ),
                "baseline_oldest_poll_age_days": (
                    float(
                        group[
                            "aggregation_days_old"
                        ].max()
                    )
                ),
                "baseline_newest_poll_age_days": (
                    float(
                        group[
                            "aggregation_days_old"
                        ].min()
                    )
                ),
                "baseline_mean_sample_size": (
                    _weighted_mean(
                        group[
                            "aggregation_sample_size"
                        ],
                        group[
                            "raw_poll_weight"
                        ],
                    )
                ),
                "baseline_imputed_sample_count": int(
                    group[
                        "sample_size_imputed"
                    ].sum()
                ),
            }
        )

    aggregate_frame = pd.DataFrame(
        aggregate_records
    )

    output = snapshot_summary.copy()

    if not aggregate_frame.empty:
        output = output.merge(
            aggregate_frame,
            how="left",
            on="snapshot_id",
            validate="one_to_one",
        )

    output[
        "baseline_has_polling"
    ] = output[
        "baseline_poll_margin_dem"
    ].notna()

    count_columns = [
        "baseline_poll_question_count",
        "baseline_poll_count",
        "baseline_effective_question_count",
        "baseline_total_raw_weight",
        "baseline_imputed_sample_count",
    ]

    for column in count_columns:
        if column in output.columns:
            output[column] = (
                pd.to_numeric(
                    output[column],
                    errors="coerce",
                )
                .fillna(0)
            )

    return output


def aggregate_historical_poll_snapshots(
    snapshot_questions: pd.DataFrame,
    snapshot_summary: pd.DataFrame,
    *,
    recency_half_life_days: float = (
        DEFAULT_RECENCY_HALF_LIFE_DAYS
    ),
    reference_sample_size: float = (
        DEFAULT_REFERENCE_SAMPLE_SIZE
    ),
    minimum_sample_factor: float = (
        DEFAULT_MIN_SAMPLE_FACTOR
    ),
    maximum_sample_factor: float = (
        DEFAULT_MAX_SAMPLE_FACTOR
    ),
) -> HistoricalPollAggregation:
    weighted_questions = (
        prepare_weighted_questions(
            snapshot_questions,
            recency_half_life_days=(
                recency_half_life_days
            ),
            reference_sample_size=(
                reference_sample_size
            ),
            minimum_sample_factor=(
                minimum_sample_factor
            ),
            maximum_sample_factor=(
                maximum_sample_factor
            ),
        )
    )

    snapshot_aggregates = (
        aggregate_weighted_questions(
            weighted_questions,
            snapshot_summary,
        )
    )

    configuration = {
        "recency_half_life_days": float(
            recency_half_life_days
        ),
        "reference_sample_size": float(
            reference_sample_size
        ),
        "minimum_sample_factor": float(
            minimum_sample_factor
        ),
        "maximum_sample_factor": float(
            maximum_sample_factor
        ),
    }

    return HistoricalPollAggregation(
        snapshot_aggregates=(
            snapshot_aggregates
        ),
        weighted_questions=(
            weighted_questions
        ),
        configuration=configuration,
    )
