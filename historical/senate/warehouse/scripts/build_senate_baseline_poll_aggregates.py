#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from historical.senate.warehouse.polling.historical_poll_aggregator import (  # noqa: E402
    DEFAULT_RECENCY_HALF_LIFE_DAYS,
    DEFAULT_REFERENCE_SAMPLE_SIZE,
    DEFAULT_MIN_SAMPLE_FACTOR,
    DEFAULT_MAX_SAMPLE_FACTOR,
    aggregate_historical_poll_snapshots,
    load_snapshot_questions,
    load_snapshot_summary,
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "historical"
    / "senate"
    / "warehouse"
    / "processed"
    / "polling_aggregates"
)

VALIDATION_DIR = (
    PROJECT_ROOT
    / "historical"
    / "senate"
    / "warehouse"
    / "validation"
    / "polling"
    / "baseline_poll_aggregates"
)

AGGREGATE_PATH = (
    OUTPUT_DIR
    / "senate_historical_baseline_poll_aggregates.csv"
)

WEIGHTED_QUESTIONS_PATH = (
    OUTPUT_DIR
    / "senate_historical_baseline_weighted_questions.csv"
)

CYCLE_PROFILE_PATH = (
    VALIDATION_DIR
    / "baseline_aggregate_cycle_profile.csv"
)

DAYS_OUT_PROFILE_PATH = (
    VALIDATION_DIR
    / "baseline_aggregate_days_out_profile.csv"
)

VALIDATION_JSON_PATH = (
    VALIDATION_DIR
    / "baseline_poll_aggregate_validation.json"
)

VALIDATION_TEXT_PATH = (
    VALIDATION_DIR
    / "baseline_poll_aggregate_validation.txt"
)


def json_safe(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if value is pd.NA or pd.isna(value):
        return None

    return value


def validate_aggregates(
    aggregates: pd.DataFrame,
    weighted_questions: pd.DataFrame,
    source_summary: pd.DataFrame,
) -> tuple[dict[str, Any], list[str]]:
    checks: dict[str, bool] = {}
    failures: list[str] = []

    checks["row_count_preserved"] = (
        len(aggregates)
        == len(source_summary)
    )

    checks["unique_snapshot_ids"] = (
        not aggregates[
            "snapshot_id"
        ].duplicated().any()
    )

    checks["snapshot_ids_preserved"] = (
        set(
            aggregates[
                "snapshot_id"
            ]
        )
        == set(
            source_summary[
                "snapshot_id"
            ]
        )
    )

    expected_has_polling = (
        source_summary[
            "eligible_questions"
        ]
        > 0
    )

    actual_has_polling = (
        aggregates[
            "baseline_has_polling"
        ]
        .fillna(False)
        .astype(bool)
    )

    checks[
        "polling_availability_matches"
    ] = bool(
        (
            expected_has_polling.to_numpy()
            == actual_has_polling.to_numpy()
        ).all()
    )

    no_poll_rows = aggregates.loc[
        ~aggregates[
            "baseline_has_polling"
        ]
    ]

    checks[
        "no_poll_snapshots_have_null_margin"
    ] = bool(
        no_poll_rows[
            "baseline_poll_margin_dem"
        ].isna().all()
    )

    poll_rows = aggregates.loc[
        aggregates[
            "baseline_has_polling"
        ]
    ]

    checks[
        "poll_snapshots_have_finite_margin"
    ] = bool(
        np.isfinite(
            poll_rows[
                "baseline_poll_margin_dem"
            ]
        ).all()
    )

    checks[
        "poll_margins_within_bounds"
    ] = bool(
        poll_rows[
            "baseline_poll_margin_dem"
        ]
        .between(
            -100,
            100,
            inclusive="both",
        )
        .all()
    )

    included = weighted_questions.loc[
        weighted_questions[
            "included_in_aggregate"
        ]
    ].copy()

    checks[
        "included_weights_positive"
    ] = bool(
        (
            included[
                "raw_poll_weight"
            ]
            > 0
        ).all()
    )

    normalized_totals = (
        included.groupby(
            "snapshot_id"
        )[
            "normalized_poll_weight"
        ]
        .sum()
    )

    checks[
        "normalized_weights_sum_to_one"
    ] = bool(
        np.allclose(
            normalized_totals.to_numpy(),
            1.0,
            atol=1e-10,
            rtol=0,
        )
    )

    recalculated = (
        included.assign(
            weighted_component=(
                included[
                    "aggregation_dem_margin"
                ]
                * included[
                    "normalized_poll_weight"
                ]
            )
        )
        .groupby(
            "snapshot_id"
        )[
            "weighted_component"
        ]
        .sum()
        .rename(
            "recalculated_margin"
        )
    )

    comparison = (
        poll_rows[
            [
                "snapshot_id",
                "baseline_poll_margin_dem",
            ]
        ]
        .merge(
            recalculated,
            how="left",
            left_on="snapshot_id",
            right_index=True,
            validate="one_to_one",
        )
    )

    checks[
        "aggregate_margin_recalculates"
    ] = bool(
        np.allclose(
            comparison[
                "baseline_poll_margin_dem"
            ],
            comparison[
                "recalculated_margin"
            ],
            atol=1e-10,
            rtol=0,
        )
    )

    question_counts = (
        included.groupby(
            "snapshot_id"
        )
        .size()
        .rename(
            "recalculated_question_count"
        )
    )

    count_comparison = (
        aggregates[
            [
                "snapshot_id",
                "baseline_poll_question_count",
            ]
        ]
        .merge(
            question_counts,
            how="left",
            left_on="snapshot_id",
            right_index=True,
        )
    )

    count_comparison[
        "recalculated_question_count"
    ] = (
        count_comparison[
            "recalculated_question_count"
        ]
        .fillna(0)
    )

    checks[
        "aggregate_question_counts_match"
    ] = bool(
        (
            count_comparison[
                "baseline_poll_question_count"
            ]
            == count_comparison[
                "recalculated_question_count"
            ]
        ).all()
    )

    snapshot_date_map = (
        aggregates.set_index(
            "snapshot_id"
        )[
            "snapshot_date"
        ]
    )

    weighted_check = (
        included[
            [
                "snapshot_id",
                "end_date",
            ]
        ]
        .copy()
    )

    weighted_check[
        "snapshot_date"
    ] = weighted_check[
        "snapshot_id"
    ].map(
        snapshot_date_map
    )

    weighted_check[
        "end_date"
    ] = pd.to_datetime(
        weighted_check[
            "end_date"
        ],
        errors="coerce",
    )

    weighted_check[
        "snapshot_date"
    ] = pd.to_datetime(
        weighted_check[
            "snapshot_date"
        ],
        errors="coerce",
    )

    checks[
        "no_aggregation_date_leakage"
    ] = bool(
        (
            weighted_check[
                "end_date"
            ]
            <= weighted_check[
                "snapshot_date"
            ]
        ).all()
    )

    for name, passed in checks.items():
        if not passed:
            failures.append(
                f"Validation check failed: {name}"
            )

    validation = {
        "snapshot_rows": int(
            len(aggregates)
        ),
        "weighted_question_rows": int(
            len(weighted_questions)
        ),
        "included_question_rows": int(
            included.shape[0]
        ),
        "snapshots_with_polling": int(
            aggregates[
                "baseline_has_polling"
            ].sum()
        ),
        "snapshots_without_polling": int(
            (
                ~aggregates[
                    "baseline_has_polling"
                ]
            ).sum()
        ),
        "checks": checks,
        "failure_count": int(
            len(failures)
        ),
        "failures": failures,
        "passed": not failures,
    }

    return validation, failures


def build_profiles(
    aggregates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cycle_profile = (
        aggregates.groupby(
            "cycle",
            dropna=False,
        )
        .agg(
            races=(
                "race_key",
                "nunique",
            ),
            snapshots=(
                "snapshot_id",
                "nunique",
            ),
            snapshots_with_polling=(
                "baseline_has_polling",
                "sum",
            ),
            mean_poll_questions=(
                "baseline_poll_question_count",
                "mean",
            ),
            median_poll_questions=(
                "baseline_poll_question_count",
                "median",
            ),
            mean_effective_questions=(
                "baseline_effective_question_count",
                "mean",
            ),
            mean_weighted_poll_age=(
                "baseline_mean_poll_age_days",
                "mean",
            ),
        )
        .reset_index()
    )

    days_out_profile = (
        aggregates.groupby(
            "days_before_election",
            dropna=False,
        )
        .agg(
            snapshots=(
                "snapshot_id",
                "nunique",
            ),
            snapshots_with_polling=(
                "baseline_has_polling",
                "sum",
            ),
            mean_poll_questions=(
                "baseline_poll_question_count",
                "mean",
            ),
            median_poll_questions=(
                "baseline_poll_question_count",
                "median",
            ),
            mean_effective_questions=(
                "baseline_effective_question_count",
                "mean",
            ),
            mean_weighted_poll_age=(
                "baseline_mean_poll_age_days",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            "days_before_election",
            ascending=False,
        )
    )

    for frame in [
        cycle_profile,
        days_out_profile,
    ]:
        frame[
            "share_snapshots_with_polling"
        ] = (
            frame[
                "snapshots_with_polling"
            ]
            / frame[
                "snapshots"
            ]
        )

    return cycle_profile, days_out_profile


def main() -> int:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 92)
    print(
        "BUILD SENATE BASELINE HISTORICAL POLL AGGREGATES"
    )
    print("=" * 92)

    questions = load_snapshot_questions()
    summary = load_snapshot_summary()

    print(
        f"Snapshot-question rows: "
        f"{len(questions):,}"
    )

    print(
        f"Snapshot-summary rows:  "
        f"{len(summary):,}"
    )

    result = aggregate_historical_poll_snapshots(
        questions,
        summary,
        recency_half_life_days=(
            DEFAULT_RECENCY_HALF_LIFE_DAYS
        ),
        reference_sample_size=(
            DEFAULT_REFERENCE_SAMPLE_SIZE
        ),
        minimum_sample_factor=(
            DEFAULT_MIN_SAMPLE_FACTOR
        ),
        maximum_sample_factor=(
            DEFAULT_MAX_SAMPLE_FACTOR
        ),
    )

    validation, failures = validate_aggregates(
        result.snapshot_aggregates,
        result.weighted_questions,
        summary,
    )

    cycle_profile, days_out_profile = (
        build_profiles(
            result.snapshot_aggregates
        )
    )

    result.snapshot_aggregates.to_csv(
        AGGREGATE_PATH,
        index=False,
    )

    result.weighted_questions.to_csv(
        WEIGHTED_QUESTIONS_PATH,
        index=False,
    )

    cycle_profile.to_csv(
        CYCLE_PROFILE_PATH,
        index=False,
    )

    days_out_profile.to_csv(
        DAYS_OUT_PROFILE_PATH,
        index=False,
    )

    validation[
        "configuration"
    ] = result.configuration

    VALIDATION_JSON_PATH.write_text(
        json.dumps(
            validation,
            indent=2,
            sort_keys=True,
            default=json_safe,
        )
        + "\n",
        encoding="utf-8",
    )

    checks = validation[
        "checks"
    ]

    labels = {
        "row_count_preserved": (
            "Snapshot row count preserved"
        ),
        "unique_snapshot_ids": (
            "Unique snapshot IDs"
        ),
        "snapshot_ids_preserved": (
            "Snapshot IDs preserved"
        ),
        "polling_availability_matches": (
            "Polling availability matches"
        ),
        "no_poll_snapshots_have_null_margin": (
            "No-poll snapshots have null margin"
        ),
        "poll_snapshots_have_finite_margin": (
            "Polling snapshots have finite margin"
        ),
        "poll_margins_within_bounds": (
            "Polling margins within bounds"
        ),
        "included_weights_positive": (
            "Included weights positive"
        ),
        "normalized_weights_sum_to_one": (
            "Normalized weights sum to one"
        ),
        "aggregate_margin_recalculates": (
            "Aggregate margin recalculates"
        ),
        "aggregate_question_counts_match": (
            "Aggregate question counts match"
        ),
        "no_aggregation_date_leakage": (
            "No aggregation date leakage"
        ),
    }

    lines = [
        "",
        "=" * 92,
        (
            "SENATE BASELINE HISTORICAL "
            "POLL AGGREGATE VALIDATION"
        ),
        "=" * 92,
        "",
        (
            "Snapshots:                         "
            f"{validation['snapshot_rows']:,}"
        ),
        (
            "Weighted question rows:            "
            f"{validation['weighted_question_rows']:,}"
        ),
        (
            "Included question rows:            "
            f"{validation['included_question_rows']:,}"
        ),
        (
            "Snapshots with polling:            "
            f"{validation['snapshots_with_polling']:,}"
        ),
        (
            "Snapshots without polling:         "
            f"{validation['snapshots_without_polling']:,}"
        ),
        "",
        "CONFIGURATION",
        "-" * 92,
        (
            "Recency half-life:                 "
            f"{result.configuration['recency_half_life_days']:.1f} days"
        ),
        (
            "Reference sample size:             "
            f"{result.configuration['reference_sample_size']:.0f}"
        ),
        (
            "Minimum sample-size factor:        "
            f"{result.configuration['minimum_sample_factor']:.2f}"
        ),
        (
            "Maximum sample-size factor:        "
            f"{result.configuration['maximum_sample_factor']:.2f}"
        ),
        "",
        "BLOCKING CHECKS",
        "-" * 92,
    ]

    for key, label in labels.items():
        lines.append(
            f"{label + ':':<44}"
            f"{'PASSED' if checks[key] else 'FAILED'}"
        )

    lines.extend(
        [
            "",
            (
                "Baseline aggregation validation: "
                f"{'PASSED' if validation['passed'] else 'FAILED'}"
            ),
            "",
            "Outputs:",
            (
                "  Snapshot aggregates: "
                f"{AGGREGATE_PATH}"
            ),
            (
                "  Weighted questions: "
                f"{WEIGHTED_QUESTIONS_PATH}"
            ),
            (
                "  Cycle profile: "
                f"{CYCLE_PROFILE_PATH}"
            ),
            (
                "  Days-out profile: "
                f"{DAYS_OUT_PROFILE_PATH}"
            ),
            (
                "  JSON validation: "
                f"{VALIDATION_JSON_PATH}"
            ),
            "",
        ]
    )

    if failures:
        lines.extend(
            [
                "FAILURES",
                "-" * 92,
            ]
        )

        lines.extend(
            f"- {failure}"
            for failure in failures
        )

        lines.append("")

    text = "\n".join(
        lines
    )

    VALIDATION_TEXT_PATH.write_text(
        text,
        encoding="utf-8",
    )

    print(text)

    return 0 if validation[
        "passed"
    ] else 1


if __name__ == "__main__":
    sys.exit(main())
