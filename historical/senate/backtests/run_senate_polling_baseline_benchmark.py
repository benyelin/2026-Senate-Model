#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from historical.senate.backtests.senate_backtest_scorer import (  # noqa: E402
    run_senate_backtest_scoring,
)


POLL_AGGREGATES_PATH = (
    PROJECT_ROOT
    / "historical"
    / "senate"
    / "warehouse"
    / "processed"
    / "polling_aggregates"
    / "senate_historical_baseline_poll_aggregates.csv"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "historical"
    / "senate"
    / "warehouse"
    / "processed"
    / "senate_historical_baselines_2012_2024.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "historical"
    / "senate"
    / "backtests"
    / "outputs"
    / "polling_baseline"
)

SCORED_PREDICTIONS_PATH = (
    OUTPUT_DIR
    / "senate_polling_baseline_scored_predictions.csv"
)

OVERALL_METRICS_PATH = (
    OUTPUT_DIR
    / "senate_polling_baseline_overall_metrics.csv"
)

BY_CYCLE_PATH = (
    OUTPUT_DIR
    / "senate_polling_baseline_metrics_by_cycle.csv"
)

BY_DAYS_OUT_PATH = (
    OUTPUT_DIR
    / "senate_polling_baseline_metrics_by_days_out.csv"
)

BY_CYCLE_DAYS_OUT_PATH = (
    OUTPUT_DIR
    / "senate_polling_baseline_metrics_by_cycle_days_out.csv"
)

BY_POLL_COUNT_PATH = (
    OUTPUT_DIR
    / "senate_polling_baseline_metrics_by_poll_count_bucket.csv"
)

VALIDATION_JSON_PATH = (
    OUTPUT_DIR
    / "senate_polling_baseline_validation.json"
)

VALIDATION_TEXT_PATH = (
    OUTPUT_DIR
    / "senate_polling_baseline_validation.txt"
)


def normalize_boolean(
    series: pd.Series,
) -> pd.Series:
    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    return normalized.isin(
        {
            "true",
            "1",
            "yes",
            "y",
            "t",
        }
    )


def load_poll_aggregates() -> pd.DataFrame:
    frame = pd.read_csv(
        POLL_AGGREGATES_PATH,
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


def load_results() -> pd.DataFrame:
    frame = pd.read_csv(
        RESULTS_PATH,
        low_memory=False,
    )

    frame[
        "cycle"
    ] = pd.to_numeric(
        frame["cycle"],
        errors="coerce",
    ).astype("Int64")

    frame[
        "actual_margin_dem"
    ] = pd.to_numeric(
        frame[
            "actual_margin_dem"
        ],
        errors="coerce",
    )

    if "backtest_scorable" in frame.columns:
        frame[
            "backtest_scorable_normalized"
        ] = normalize_boolean(
            frame[
                "backtest_scorable"
            ]
        )
    else:
        frame[
            "backtest_scorable_normalized"
        ] = frame[
            "actual_margin_dem"
        ].notna()

    return frame


def prepare_results_for_merge(
    results: pd.DataFrame,
) -> pd.DataFrame:
    desired_columns = [
        "race_id",
        "cycle",
        "state",
        "election_date",
        "actual_margin_dem",
        "winner_party",
        "major_party_contested",
        "backtest_scorable",
        "backtest_scorable_normalized",
        "backtest_exclusion_reason",
        "special_election",
        "election_type",
        "senate_class",
        "seat_id",
        "dem_candidate",
        "gop_candidate",
    ]

    available_columns = [
        column
        for column in desired_columns
        if column in results.columns
    ]

    merge_results = results[
        available_columns
    ].copy()

    duplicate_keys = (
        merge_results.duplicated(
            subset=[
                "race_id",
                "cycle",
            ],
            keep=False,
        )
    )

    if duplicate_keys.any():
        duplicates = (
            merge_results.loc[
                duplicate_keys,
                [
                    "race_id",
                    "cycle",
                    "state",
                ],
            ]
            .sort_values(
                [
                    "cycle",
                    "state",
                    "race_id",
                ]
            )
        )

        raise ValueError(
            "Historical results contain duplicate "
            "race_id/cycle keys:\n"
            + duplicates.to_string(
                index=False
            )
        )

    rename_map = {
        column: (
            f"result_{column}"
            if column
            not in {
                "race_id",
                "cycle",
                "actual_margin_dem",
                "winner_party",
            }
            else column
        )
        for column in merge_results.columns
    }

    return merge_results.rename(
        columns=rename_map
    )


def add_poll_count_bucket(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    output = frame.copy()

    poll_count = pd.to_numeric(
        output[
            "baseline_poll_count"
        ],
        errors="coerce",
    ).fillna(0)

    output[
        "poll_count_bucket"
    ] = pd.cut(
        poll_count,
        bins=[
            -np.inf,
            1,
            2,
            4,
            7,
            12,
            np.inf,
        ],
        labels=[
            "1 poll",
            "2 polls",
            "3-4 polls",
            "5-7 polls",
            "8-12 polls",
            "13+ polls",
        ],
        ordered=True,
    )

    return output


def build_validation(
    *,
    aggregates: pd.DataFrame,
    merged: pd.DataFrame,
    eligible: pd.DataFrame,
    scored: pd.DataFrame,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}

    checks[
        "aggregate_row_count_preserved_after_merge"
    ] = (
        len(merged)
        == len(aggregates)
    )

    checks[
        "unique_snapshot_ids"
    ] = not merged[
        "snapshot_id"
    ].duplicated().any()

    polling_rows = merged.loc[
        merged[
            "baseline_has_polling"
        ].fillna(False)
    ]

    checks[
        "all_polling_snapshots_match_result"
    ] = bool(
        polling_rows[
            "actual_margin_dem"
        ].notna().all()
    )

    checks[
        "all_polling_snapshots_match_race"
    ] = bool(
        polling_rows[
            "winner_party"
        ].notna().all()
    )

    checks[
        "eligible_rows_have_poll_margin"
    ] = bool(
        eligible[
            "baseline_poll_margin_dem"
        ].notna().all()
    )

    checks[
        "eligible_rows_have_actual_margin"
    ] = bool(
        eligible[
            "actual_margin_dem"
        ].notna().all()
    )

    checks[
        "all_eligible_rows_scored"
    ] = bool(
        scored[
            "included_in_score"
        ].all()
    )

    checks[
        "prediction_errors_finite"
    ] = bool(
        np.isfinite(
            scored[
                "scored_error_dem"
            ]
        ).all()
    )

    checks[
        "winner_correct_nonmissing"
    ] = bool(
        scored[
            "scored_winner_correct"
        ].notna().all()
    )

    cycle_values = sorted(
        int(value)
        for value in eligible[
            "cycle"
        ].dropna().unique()
    )

    expected_cycles = [
        2018,
        2020,
        2022,
        2024,
    ]

    checks[
        "expected_polling_cycles_present"
    ] = (
        cycle_values
        == expected_cycles
    )

    failures = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    return {
        "aggregate_snapshot_rows": int(
            len(aggregates)
        ),
        "merged_snapshot_rows": int(
            len(merged)
        ),
        "snapshots_with_polling": int(
            merged[
                "baseline_has_polling"
            ].fillna(False).sum()
        ),
        "eligible_scorable_snapshots": int(
            len(eligible)
        ),
        "scored_snapshots": int(
            scored[
                "included_in_score"
            ].sum()
        ),
        "eligible_unique_races": int(
            eligible[
                "race_id"
            ].nunique()
        ),
        "cycles": cycle_values,
        "checks": checks,
        "failure_count": int(
            len(failures)
        ),
        "failures": failures,
        "passed": not failures,
    }


def main() -> int:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 96)
    print(
        "SENATE HISTORICAL POLLING BASELINE BENCHMARK"
    )
    print("=" * 96)

    aggregates = load_poll_aggregates()
    results = load_results()

    print(
        f"Polling snapshot rows:       "
        f"{len(aggregates):,}"
    )

    print(
        f"Historical result rows:      "
        f"{len(results):,}"
    )

    merge_results = (
        prepare_results_for_merge(
            results
        )
    )

    # Normalize the composite merge keys across both warehouses.
    # CSV inference may load race_id as an integer in one source and
    # as text in the other even though the underlying identifiers match.
    for frame_name, frame in [
        ("polling aggregates", aggregates),
        ("historical results", merge_results),
    ]:
        frame["race_id"] = (
            frame["race_id"]
            .astype("string")
            .str.strip()
        )

        frame["cycle"] = pd.to_numeric(
            frame["cycle"],
            errors="raise",
        ).astype("Int64")

        if (
            frame["race_id"].isna().any()
            or frame["race_id"].eq("").any()
        ):
            raise ValueError(
                f"{frame_name} contains missing or blank race_id values."
            )

        if frame["cycle"].isna().any():
            raise ValueError(
                f"{frame_name} contains missing cycle values."
            )

    merged = aggregates.merge(
        merge_results,
        how="left",
        on=[
            "race_id",
            "cycle",
        ],
        validate="many_to_one",
        indicator=True,
    )

    merged[
        "result_match_status"
    ] = merged[
        "_merge"
    ].astype(str)

    merged = merged.drop(
        columns=[
            "_merge",
        ]
    )

    polling_mask = merged[
        "baseline_has_polling"
    ].fillna(False).astype(bool)

    scorable_mask = merged[
        "result_backtest_scorable_normalized"
    ].fillna(False).astype(bool)

    actual_mask = merged[
        "actual_margin_dem"
    ].notna()

    eligible = merged.loc[
        polling_mask
        & scorable_mask
        & actual_mask
    ].copy()

    eligible = add_poll_count_bucket(
        eligible
    )

    groupings = {
        "by_cycle": [
            "cycle",
        ],
        "by_days_out": [
            "days_before_election",
        ],
        "by_cycle_days_out": [
            "cycle",
            "days_before_election",
        ],
        "by_poll_count_bucket": [
            "poll_count_bucket",
        ],
    }

    result = run_senate_backtest_scoring(
        eligible,
        predicted_margin_column=(
            "baseline_poll_margin_dem"
        ),
        actual_margin_column=(
            "actual_margin_dem"
        ),
        model_name=(
            "Baseline polling: "
            "14-day recency + sqrt sample size"
        ),
        groupings=groupings,
    )

    validation = build_validation(
        aggregates=aggregates,
        merged=merged,
        eligible=eligible,
        scored=(
            result.scored_predictions
        ),
    )

    result.scored_predictions.to_csv(
        SCORED_PREDICTIONS_PATH,
        index=False,
    )

    result.overall_metrics.to_csv(
        OVERALL_METRICS_PATH,
        index=False,
    )

    result.grouped_metrics[
        "by_cycle"
    ].to_csv(
        BY_CYCLE_PATH,
        index=False,
    )

    result.grouped_metrics[
        "by_days_out"
    ].to_csv(
        BY_DAYS_OUT_PATH,
        index=False,
    )

    result.grouped_metrics[
        "by_cycle_days_out"
    ].to_csv(
        BY_CYCLE_DAYS_OUT_PATH,
        index=False,
    )

    result.grouped_metrics[
        "by_poll_count_bucket"
    ].to_csv(
        BY_POLL_COUNT_PATH,
        index=False,
    )

    VALIDATION_JSON_PATH.write_text(
        json.dumps(
            validation,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    overall = (
        result.overall_metrics.iloc[0]
    )

    by_cycle = (
        result.grouped_metrics[
            "by_cycle"
        ]
        .sort_values(
            "cycle"
        )
    )

    by_days = (
        result.grouped_metrics[
            "by_days_out"
        ]
        .sort_values(
            "days_before_election",
            ascending=False,
        )
    )

    lines = [
        "",
        "=" * 96,
        "BASELINE POLLING PERFORMANCE",
        "=" * 96,
        "",
        (
            "Scored snapshots:           "
            f"{int(overall['observations']):,}"
        ),
        (
            "Mean absolute error:        "
            f"{overall['mae']:.4f}"
        ),
        (
            "Root mean squared error:    "
            f"{overall['rmse']:.4f}"
        ),
        (
            "Mean error / bias:          "
            f"{overall['mean_error']:+.4f}"
        ),
        (
            "Median absolute error:      "
            f"{overall['median_absolute_error']:.4f}"
        ),
        (
            "Winner accuracy:            "
            f"{overall['winner_accuracy']:.2%}"
        ),
        (
            "Democratic-win accuracy:    "
            f"{overall['dem_win_accuracy']:.2%}"
        ),
        (
            "Republican-win accuracy:    "
            f"{overall['gop_win_accuracy']:.2%}"
        ),
        "",
        "PERFORMANCE BY CYCLE",
        "-" * 96,
        by_cycle[
            [
                "cycle",
                "observations",
                "mae",
                "rmse",
                "mean_error",
                "winner_accuracy",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        ),
        "",
        "PERFORMANCE BY DAYS BEFORE ELECTION",
        "-" * 96,
        by_days[
            [
                "days_before_election",
                "observations",
                "mae",
                "rmse",
                "mean_error",
                "winner_accuracy",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        ),
        "",
        "BLOCKING CHECKS",
        "-" * 96,
    ]

    for name, passed in (
        validation[
            "checks"
        ].items()
    ):
        lines.append(
            f"{name + ':':<58}"
            f"{'PASSED' if passed else 'FAILED'}"
        )

    lines.extend(
        [
            "",
            (
                "Polling baseline benchmark: "
                f"{'PASSED' if validation['passed'] else 'FAILED'}"
            ),
            "",
            "Outputs:",
            (
                "  Scored predictions: "
                f"{SCORED_PREDICTIONS_PATH}"
            ),
            (
                "  Overall metrics: "
                f"{OVERALL_METRICS_PATH}"
            ),
            (
                "  Metrics by cycle: "
                f"{BY_CYCLE_PATH}"
            ),
            (
                "  Metrics by days out: "
                f"{BY_DAYS_OUT_PATH}"
            ),
            (
                "  Cycle/days-out metrics: "
                f"{BY_CYCLE_DAYS_OUT_PATH}"
            ),
            (
                "  Poll-count metrics: "
                f"{BY_POLL_COUNT_PATH}"
            ),
            (
                "  Validation JSON: "
                f"{VALIDATION_JSON_PATH}"
            ),
            "",
        ]
    )

    if validation[
        "failures"
    ]:
        lines.extend(
            [
                "FAILURES",
                "-" * 96,
            ]
        )

        lines.extend(
            f"- {failure}"
            for failure in validation[
                "failures"
            ]
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

    return (
        0
        if validation[
            "passed"
        ]
        else 1
    )


if __name__ == "__main__":
    sys.exit(main())
