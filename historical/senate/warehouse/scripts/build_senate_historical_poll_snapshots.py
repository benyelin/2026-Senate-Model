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

from historical.senate.warehouse.polling.historical_poll_selector import (  # noqa: E402
    DEFAULT_WAREHOUSE_PATH,
    load_historical_polling_warehouse,
    select_historical_senate_polls,
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "historical"
    / "senate"
    / "warehouse"
    / "processed"
    / "polling_snapshots"
)

VALIDATION_DIR = (
    PROJECT_ROOT
    / "historical"
    / "senate"
    / "warehouse"
    / "validation"
    / "polling"
    / "historical_poll_snapshots"
)

SNAPSHOT_QUESTIONS_PATH = (
    OUTPUT_DIR
    / "senate_historical_poll_snapshot_questions.csv"
)

SNAPSHOT_SUMMARY_PATH = (
    OUTPUT_DIR
    / "senate_historical_poll_snapshot_summary.csv"
)

RACE_PROFILE_PATH = (
    OUTPUT_DIR
    / "senate_historical_poll_race_profile.csv"
)

CANONICAL_RESULTS_PATH = (
    PROJECT_ROOT
    / "historical"
    / "senate"
    / "warehouse"
    / "processed"
    / "senate_historical_baselines_2012_2024.csv"
)

# Polling-provider race IDs are not stable warehouse identifiers.
# Most polling races map automatically by cycle and state. These
# overrides resolve state-cycles containing both regular and special
# elections and collapse runoff-only provider IDs into the underlying
# canonical general-election race.
POLL_RACE_ID_TO_CANONICAL_RACE_ID = {
    # 2018 Minnesota
    (2018, "MN", 107): "2018_MN_REGULAR",
    (2018, "MN", 129): "2018_MN_SPECIAL",

    # 2018 Mississippi: 6209 is the special-election runoff.
    (2018, "MS", 108): "2018_MS_REGULAR",
    (2018, "MS", 130): "2018_MS_SPECIAL",
    (2018, "MS", 6209): "2018_MS_SPECIAL",

    # 2020 Georgia: 8737 and 7781 are runoff identifiers.
    (2020, "GA", 6271): "2020_GA_REGULAR",
    (2020, "GA", 8737): "2020_GA_REGULAR",
    (2020, "GA", 7780): "2020_GA_SPECIAL",
    (2020, "GA", 7781): "2020_GA_SPECIAL",

    # 2022 California full-term and remainder elections.
    (2022, "CA", 8921): "2022_CA_REGULAR",
    (2022, "CA", 9480): "2022_CA_SPECIAL",

    # 2022 Oklahoma regular and special elections.
    (2022, "OK", 8943): "2022_OK_REGULAR",
    (2022, "OK", 9482): "2022_OK_SPECIAL",

    # 2024 California full-term and remainder elections.
    (2024, "CA", 9506): "2024_CA_REGULAR",
    (2024, "CA", 10013): "2024_CA_SPECIAL",

    # 2024 Nebraska regular and special elections.
    (2024, "NE", 9520): "2024_NE_REGULAR",
    (2024, "NE", 9553): "2024_NE_SPECIAL",
}

EXCLUSION_SUMMARY_PATH = (
    VALIDATION_DIR
    / "snapshot_exclusion_summary.csv"
)

CYCLE_PROFILE_PATH = (
    VALIDATION_DIR
    / "snapshot_cycle_profile.csv"
)

DAYS_OUT_PROFILE_PATH = (
    VALIDATION_DIR
    / "snapshot_days_out_profile.csv"
)

VALIDATION_JSON_PATH = (
    VALIDATION_DIR
    / "snapshot_warehouse_validation.json"
)

VALIDATION_TEXT_PATH = (
    VALIDATION_DIR
    / "snapshot_warehouse_validation.txt"
)


SNAPSHOT_DAYS_BEFORE_ELECTION = (
    120,
    90,
    60,
    30,
    14,
    7,
    0,
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


def assign_canonical_race_ids(
    profile: pd.DataFrame,
) -> pd.DataFrame:
    """
    Map polling-provider race IDs to the canonical historical-results
    warehouse IDs.

    Provider runoff IDs are deliberately collapsed into the underlying
    general-election race. State-cycles containing only one canonical
    race are mapped automatically. Genuine regular/special pairs use
    the documented override table above.
    """
    if not CANONICAL_RESULTS_PATH.exists():
        raise FileNotFoundError(
            "Canonical Senate results warehouse not found: "
            f"{CANONICAL_RESULTS_PATH}"
        )

    canonical = pd.read_csv(
        CANONICAL_RESULTS_PATH,
        low_memory=False,
    )

    required = {
        "race_id",
        "cycle",
        "state",
    }

    missing = sorted(
        required - set(canonical.columns)
    )

    if missing:
        raise RuntimeError(
            "Canonical results warehouse is missing columns: "
            + ", ".join(missing)
        )

    canonical = canonical[
        [
            "race_id",
            "cycle",
            "state",
        ]
    ].drop_duplicates()

    canonical["race_id"] = (
        canonical["race_id"]
        .astype("string")
        .str.strip()
    )

    canonical["cycle"] = pd.to_numeric(
        canonical["cycle"],
        errors="raise",
    ).astype(int)

    canonical["state"] = (
        canonical["state"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    duplicate_canonical_ids = int(
        canonical["race_id"]
        .duplicated()
        .sum()
    )

    if duplicate_canonical_ids:
        raise RuntimeError(
            "Canonical results warehouse contains duplicate "
            f"race IDs: {duplicate_canonical_ids}"
        )

    canonical_by_state_cycle = {
        key: group["race_id"].tolist()
        for key, group in canonical.groupby(
            [
                "cycle",
                "state",
            ],
            dropna=False,
        )
    }

    profile = profile.copy()

    # Preserve the source provider's numeric identifier permanently.
    profile = profile.rename(
        columns={
            "race_id": "poll_race_id",
        }
    )

    canonical_ids: list[str] = []
    mapping_methods: list[str] = []

    for row in profile.itertuples(index=False):
        key = (
            int(row.cycle),
            str(row.state),
            int(row.poll_race_id),
        )

        override = (
            POLL_RACE_ID_TO_CANONICAL_RACE_ID
            .get(key)
        )

        if override is not None:
            canonical_ids.append(override)
            mapping_methods.append(
                "explicit_regular_special_or_runoff_override"
            )
            continue

        state_cycle_key = (
            int(row.cycle),
            str(row.state),
        )

        candidates = canonical_by_state_cycle.get(
            state_cycle_key,
            [],
        )

        if len(candidates) == 1:
            canonical_ids.append(
                str(candidates[0])
            )
            mapping_methods.append(
                "unique_cycle_state"
            )
            continue

        if not candidates:
            canonical_ids.append(pd.NA)
            mapping_methods.append(
                "excluded_no_canonical_results_race"
            )
            continue

        raise RuntimeError(
            "Ambiguous canonical results mapping for polling race "
            f"{key}. Candidate canonical IDs: {candidates}. "
            "Add a documented entry to "
            "POLL_RACE_ID_TO_CANONICAL_RACE_ID."
        )

    profile["race_id"] = pd.Series(
        canonical_ids,
        index=profile.index,
        dtype="string",
    )

    profile["race_id_mapping_method"] = pd.Series(
        mapping_methods,
        index=profile.index,
        dtype="string",
    )

    valid_canonical_ids = set(
        canonical["race_id"].dropna().astype(str)
    )

    invalid_ids = sorted(
        set(
            profile["race_id"]
            .dropna()
            .astype(str)
        )
        - valid_canonical_ids
    )

    if invalid_ids:
        raise RuntimeError(
            "Polling race mapping produced IDs absent from the "
            "canonical results warehouse: "
            + ", ".join(invalid_ids)
        )

    missing_canonical_mask = (
        profile["race_id"].isna()
    )

    if missing_canonical_mask.any():
        missing_rows = profile.loc[
            missing_canonical_mask,
            [
                "cycle",
                "state",
                "poll_race_id",
                "election_date",
                "race_id_mapping_method",
            ],
        ].copy()

        for row in missing_rows.itertuples(
            index=False
        ):
            print(
                "Skipping polling race with no canonical "
                "results match: "
                f"cycle={int(row.cycle)}, "
                f"state={row.state}, "
                f"poll_race_id={int(row.poll_race_id)}, "
                f"election_date={row.election_date}"
            )

        profile = profile.loc[
            ~missing_canonical_mask
        ].copy()

    # More than one provider ID may deliberately map to one canonical
    # race when the provider created a separate runoff identifier.
    provider_key_duplicates = int(
        profile[
            [
                "cycle",
                "state",
                "poll_race_id",
                "election_date",
            ]
        ]
        .duplicated()
        .sum()
    )

    if provider_key_duplicates:
        raise RuntimeError(
            "Race profile contains duplicate polling-provider race "
            f"keys after canonical mapping: {provider_key_duplicates}"
        )

    return profile


def build_race_profile(
    warehouse: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "cycle",
        "state",
        "race_id",
        "election_date",
        "question_id",
    }

    missing = sorted(
        required - set(warehouse.columns)
    )

    if missing:
        raise RuntimeError(
            "Warehouse is missing race-profile columns: "
            + ", ".join(missing)
        )

    profile = (
        warehouse.loc[
            warehouse["cycle"].notna()
            & warehouse["state"].notna()
            & warehouse["race_id"].notna()
            & warehouse["election_date"].notna()
        ]
        .groupby(
            [
                "cycle",
                "state",
                "race_id",
                "election_date",
            ],
            dropna=False,
        )
        .agg(
            source_candidate_rows=(
                "question_id",
                "size",
            ),
            source_questions=(
                "question_id",
                "nunique",
            ),
            earliest_poll_end_date=(
                "end_date",
                "min",
            ),
            latest_poll_end_date=(
                "end_date",
                "max",
            ),
        )
        .reset_index()
    )

    profile["cycle"] = pd.to_numeric(
        profile["cycle"],
        errors="raise",
    ).astype(int)

    profile["race_id"] = pd.to_numeric(
        profile["race_id"],
        errors="raise",
    ).astype(int)

    profile["state"] = (
        profile["state"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    profile["election_date"] = pd.to_datetime(
        profile["election_date"],
        errors="coerce",
    ).dt.normalize()

    profile = assign_canonical_race_ids(
        profile
    )

    profile["race_key"] = (
        profile["cycle"].astype(str)
        + "_"
        + profile["state"].astype(str)
        + "_"
        + profile["poll_race_id"].astype(str)
        + "_"
        + profile["election_date"]
        .dt.strftime("%Y%m%d")
        .fillna("UNKNOWN")
    )

    duplicate_race_keys = int(
        profile["race_key"]
        .duplicated()
        .sum()
    )

    if duplicate_race_keys:
        raise RuntimeError(
            "Race profile contains duplicate race keys: "
            f"{duplicate_race_keys}"
        )

    return profile.sort_values(
        [
            "cycle",
            "state",
            "race_id",
            "poll_race_id",
            "election_date",
        ]
    ).reset_index(drop=True)


def build_snapshot_request_table(
    race_profile: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for race in race_profile.itertuples(
        index=False
    ):
        election_date = pd.Timestamp(
            race.election_date
        ).normalize()

        for days_before in (
            SNAPSHOT_DAYS_BEFORE_ELECTION
        ):
            snapshot_date = (
                election_date
                - pd.Timedelta(
                    days=days_before
                )
            )

            snapshot_id = (
                f"{race.race_key}"
                f"_D{days_before:03d}"
            )

            records.append(
                {
                    "snapshot_id": snapshot_id,
                    "race_key": race.race_key,
                    "cycle": int(
                        race.cycle
                    ),
                    "state": str(
                        race.state
                    ),
                    "race_id": str(
                        race.race_id
                    ),
                    "poll_race_id": int(
                        race.poll_race_id
                    ),
                    "race_id_mapping_method": str(
                        race.race_id_mapping_method
                    ),
                    "election_date": (
                        election_date
                    ),
                    "snapshot_date": (
                        snapshot_date
                    ),
                    "days_before_election": int(
                        days_before
                    ),
                    "source_candidate_rows": int(
                        race.source_candidate_rows
                    ),
                    "source_questions": int(
                        race.source_questions
                    ),
                }
            )

    requests = pd.DataFrame(
        records
    )

    duplicate_snapshot_ids = int(
        requests["snapshot_id"]
        .duplicated()
        .sum()
    )

    if duplicate_snapshot_ids:
        raise RuntimeError(
            "Snapshot request table contains "
            "duplicate snapshot IDs: "
            f"{duplicate_snapshot_ids}"
        )

    expected = (
        len(race_profile)
        * len(
            SNAPSHOT_DAYS_BEFORE_ELECTION
        )
    )

    if len(requests) != expected:
        raise RuntimeError(
            "Snapshot request count mismatch: "
            f"expected {expected:,}, "
            f"found {len(requests):,}."
        )

    return requests


def build_snapshot_warehouse(
    warehouse: pd.DataFrame,
    requests: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    question_frames: list[pd.DataFrame] = []
    summary_records: list[dict[str, Any]] = []
    exclusion_frames: list[pd.DataFrame] = []

    total_requests = len(requests)

    for position, request in enumerate(
        requests.itertuples(index=False),
        start=1,
    ):
        selection = (
            select_historical_senate_polls(
                warehouse,
                cycle=int(
                    request.cycle
                ),
                state=str(
                    request.state
                ),
                race_id=int(
                    request.poll_race_id
                ),
                snapshot_date=(
                    request.snapshot_date
                ),
            )
        )

        eligible = (
            selection
            .eligible_questions
            .copy()
        )

        if not eligible.empty:
            eligible.insert(
                0,
                "snapshot_id",
                request.snapshot_id,
            )

            eligible.insert(
                1,
                "race_key",
                request.race_key,
            )

            eligible[
                "snapshot_date"
            ] = pd.Timestamp(
                request.snapshot_date
            )

            eligible[
                "days_before_election"
            ] = int(
                request.days_before_election
            )

            eligible[
                "snapshot_election_date"
            ] = pd.Timestamp(
                request.election_date
            )

            question_frames.append(
                eligible
            )

        exclusion = (
            selection
            .exclusion_summary
            .copy()
        )

        exclusion.insert(
            0,
            "snapshot_id",
            request.snapshot_id,
        )

        exclusion.insert(
            1,
            "race_key",
            request.race_key,
        )

        exclusion["cycle"] = int(
            request.cycle
        )

        exclusion["state"] = str(
            request.state
        )

        exclusion["race_id"] = str(
            request.race_id
        )
        exclusion["poll_race_id"] = int(
            request.poll_race_id
        )

        exclusion[
            "snapshot_date"
        ] = pd.Timestamp(
            request.snapshot_date
        )

        exclusion[
            "election_date"
        ] = pd.Timestamp(
            request.election_date
        )

        exclusion[
            "days_before_election"
        ] = int(
            request.days_before_election
        )

        exclusion_frames.append(
            exclusion
        )

        eligible_count = int(
            len(eligible)
        )

        summary_records.append(
            {
                "snapshot_id": (
                    request.snapshot_id
                ),
                "race_key": (
                    request.race_key
                ),
                "cycle": int(
                    request.cycle
                ),
                "state": str(
                    request.state
                ),
                "race_id": str(
                    request.race_id
                ),
                "poll_race_id": int(
                    request.poll_race_id
                ),
                "election_date": (
                    pd.Timestamp(
                        request.election_date
                    )
                ),
                "snapshot_date": (
                    pd.Timestamp(
                        request.snapshot_date
                    )
                ),
                "days_before_election": int(
                    request.days_before_election
                ),
                "source_candidate_rows": int(
                    request.source_candidate_rows
                ),
                "source_questions": int(
                    request.source_questions
                ),
                "candidate_questions": int(
                    selection.request_summary.get(
                        "candidate_questions",
                        0,
                    )
                ),
                "eligible_questions": (
                    eligible_count
                ),
                "excluded_questions": int(
                    selection.request_summary.get(
                        "excluded_questions",
                        0,
                    )
                ),
                "has_eligible_polling": bool(
                    eligible_count > 0
                ),
                "earliest_eligible_end_date": (
                    eligible["end_date"].min()
                    if eligible_count
                    else pd.NaT
                ),
                "latest_eligible_end_date": (
                    eligible["end_date"].max()
                    if eligible_count
                    else pd.NaT
                ),
                "mean_dem_margin_unweighted": (
                    float(
                        eligible[
                            "dem_margin"
                        ].mean()
                    )
                    if eligible_count
                    else np.nan
                ),
                "median_dem_margin_unweighted": (
                    float(
                        eligible[
                            "dem_margin"
                        ].median()
                    )
                    if eligible_count
                    else np.nan
                ),
                "mean_poll_age_days": (
                    float(
                        eligible[
                            "days_old"
                        ].mean()
                    )
                    if eligible_count
                    else np.nan
                ),
                "median_poll_age_days": (
                    float(
                        eligible[
                            "days_old"
                        ].median()
                    )
                    if eligible_count
                    else np.nan
                ),
                "internal_questions": (
                    int(
                        eligible[
                            "is_internal"
                        ]
                        .fillna(False)
                        .sum()
                    )
                    if eligible_count
                    else 0
                ),
                "partisan_questions": (
                    int(
                        eligible[
                            "is_partisan"
                        ]
                        .fillna(False)
                        .sum()
                    )
                    if eligible_count
                    else 0
                ),
                "tracking_questions": (
                    int(
                        eligible[
                            "is_tracking"
                        ]
                        .fillna(False)
                        .sum()
                    )
                    if eligible_count
                    else 0
                ),
            }
        )

        if (
            position == 1
            or position % 100 == 0
            or position == total_requests
        ):
            print(
                f"Processed "
                f"{position:,}/"
                f"{total_requests:,} "
                "race snapshots."
            )

    question_table = (
        pd.concat(
            question_frames,
            ignore_index=True,
        )
        if question_frames
        else pd.DataFrame()
    )

    summary_table = pd.DataFrame(
        summary_records
    )

    exclusion_table = (
        pd.concat(
            exclusion_frames,
            ignore_index=True,
        )
        if exclusion_frames
        else pd.DataFrame()
    )

    return (
        question_table,
        summary_table,
        exclusion_table,
    )


def validate_snapshot_warehouse(
    race_profile: pd.DataFrame,
    requests: pd.DataFrame,
    questions: pd.DataFrame,
    summary: pd.DataFrame,
) -> tuple[
    dict[str, Any],
    list[str],
]:
    failures: list[str] = []

    expected_snapshot_rows = (
        len(race_profile)
        * len(
            SNAPSHOT_DAYS_BEFORE_ELECTION
        )
    )

    checks: dict[str, bool] = {}

    checks[
        "snapshot_request_count"
    ] = (
        len(requests)
        == expected_snapshot_rows
    )

    checks[
        "snapshot_summary_count"
    ] = (
        len(summary)
        == expected_snapshot_rows
    )

    checks[
        "unique_snapshot_ids"
    ] = (
        not summary[
            "snapshot_id"
        ].duplicated().any()
    )

    checks[
        "request_summary_alignment"
    ] = (
        set(
            requests[
                "snapshot_id"
            ]
        )
        == set(
            summary[
                "snapshot_id"
            ]
        )
    )

    if questions.empty:
        checks[
            "question_rows_present"
        ] = False
    else:
        checks[
            "question_rows_present"
        ] = True

        checks[
            "unique_snapshot_question"
        ] = (
            not questions[
                [
                    "snapshot_id",
                    "question_id",
                ]
            ].duplicated().any()
        )

        checks[
            "no_snapshot_date_leakage"
        ] = bool(
            (
                questions["end_date"]
                <= questions[
                    "snapshot_date"
                ]
            ).all()
        )

        checks[
            "no_election_date_leakage"
        ] = bool(
            (
                questions["end_date"]
                <= questions[
                    "snapshot_election_date"
                ]
            ).all()
        )

        checks[
            "nonnegative_poll_age"
        ] = bool(
            (
                questions["days_old"]
                >= 0
            ).all()
        )

        checks[
            "eligible_only"
        ] = bool(
            questions[
                "is_eligible"
            ].fillna(False).all()
        )

        checks[
            "exactly_one_democrat"
        ] = bool(
            (
                questions[
                    "democratic_rows"
                ]
                == 1
            ).all()
        )

        checks[
            "exactly_one_republican"
        ] = bool(
            (
                questions[
                    "republican_rows"
                ]
                == 1
            ).all()
        )

        recalculated_margin = (
            questions["dem_pct"]
            - questions["rep_pct"]
        )

        checks[
            "dem_margin_recalculates"
        ] = bool(
            (
                recalculated_margin
                - questions[
                    "dem_margin"
                ]
            )
            .abs()
            .le(1e-10)
            .all()
        )

        question_counts = (
            questions.groupby(
                "snapshot_id"
            )
            .size()
            .rename(
                "recalculated_eligible_questions"
            )
        )

        summary_check = (
            summary[
                [
                    "snapshot_id",
                    "eligible_questions",
                ]
            ]
            .merge(
                question_counts,
                how="left",
                left_on="snapshot_id",
                right_index=True,
            )
        )

        summary_check[
            "recalculated_eligible_questions"
        ] = (
            summary_check[
                "recalculated_eligible_questions"
            ]
            .fillna(0)
            .astype(int)
        )

        checks[
            "summary_question_counts_match"
        ] = bool(
            (
                summary_check[
                    "eligible_questions"
                ]
                == summary_check[
                    "recalculated_eligible_questions"
                ]
            ).all()
        )

    monotonic_failures = []

    for race_key, group in summary.groupby(
        "race_key",
        sort=False,
    ):
        ordered = group.sort_values(
            "snapshot_date"
        )

        differences = (
            ordered[
                "eligible_questions"
            ]
            .diff()
            .dropna()
        )

        if (
            differences < 0
        ).any():
            monotonic_failures.append(
                race_key
            )

    checks[
        "eligible_counts_monotonic"
    ] = (
        len(
            monotonic_failures
        )
        == 0
    )

    expected_days = set(
        SNAPSHOT_DAYS_BEFORE_ELECTION
    )

    days_failures = []

    for race_key, group in summary.groupby(
        "race_key",
        sort=False,
    ):
        actual_days = set(
            group[
                "days_before_election"
            ].astype(int)
        )

        if actual_days != expected_days:
            days_failures.append(
                race_key
            )

    checks[
        "complete_days_out_grid"
    ] = (
        len(days_failures)
        == 0
    )

    for name, passed in checks.items():
        if not passed:
            failures.append(
                f"Validation check failed: {name}"
            )

    validation = {
        "race_count": int(
            len(race_profile)
        ),
        "cycle_count": int(
            race_profile[
                "cycle"
            ].nunique()
        ),
        "cycles": sorted(
            int(value)
            for value in race_profile[
                "cycle"
            ].unique()
        ),
        "snapshot_days_before_election": list(
            SNAPSHOT_DAYS_BEFORE_ELECTION
        ),
        "expected_snapshot_count": int(
            expected_snapshot_rows
        ),
        "actual_snapshot_count": int(
            len(summary)
        ),
        "question_snapshot_rows": int(
            len(questions)
        ),
        "snapshots_with_polling": int(
            summary[
                "has_eligible_polling"
            ].sum()
        ),
        "snapshots_without_polling": int(
            (
                ~summary[
                    "has_eligible_polling"
                ]
            ).sum()
        ),
        "monotonic_failure_count": int(
            len(monotonic_failures)
        ),
        "monotonic_failure_races": (
            monotonic_failures
        ),
        "incomplete_days_grid_count": int(
            len(days_failures)
        ),
        "incomplete_days_grid_races": (
            days_failures
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
    summary: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    cycle_profile = (
        summary.groupby(
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
                "has_eligible_polling",
                "sum",
            ),
            total_question_snapshot_rows=(
                "eligible_questions",
                "sum",
            ),
            mean_questions_per_snapshot=(
                "eligible_questions",
                "mean",
            ),
            median_questions_per_snapshot=(
                "eligible_questions",
                "median",
            ),
            max_questions_in_snapshot=(
                "eligible_questions",
                "max",
            ),
        )
        .reset_index()
    )

    days_out_profile = (
        summary.groupby(
            "days_before_election",
            dropna=False,
        )
        .agg(
            snapshots=(
                "snapshot_id",
                "nunique",
            ),
            snapshots_with_polling=(
                "has_eligible_polling",
                "sum",
            ),
            total_question_snapshot_rows=(
                "eligible_questions",
                "sum",
            ),
            mean_questions_per_snapshot=(
                "eligible_questions",
                "mean",
            ),
            median_questions_per_snapshot=(
                "eligible_questions",
                "median",
            ),
            max_questions_in_snapshot=(
                "eligible_questions",
                "max",
            ),
        )
        .reset_index()
        .sort_values(
            "days_before_election",
            ascending=False,
        )
    )

    cycle_profile[
        "share_snapshots_with_polling"
    ] = (
        cycle_profile[
            "snapshots_with_polling"
        ]
        / cycle_profile[
            "snapshots"
        ]
    )

    days_out_profile[
        "share_snapshots_with_polling"
    ] = (
        days_out_profile[
            "snapshots_with_polling"
        ]
        / days_out_profile[
            "snapshots"
        ]
    )

    return (
        cycle_profile,
        days_out_profile,
    )


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
        "BUILD SENATE HISTORICAL POLL SNAPSHOT WAREHOUSE"
    )
    print("=" * 92)
    print(
        f"Input warehouse: "
        f"{DEFAULT_WAREHOUSE_PATH}"
    )
    print(
        "Snapshot days before election: "
        + ", ".join(
            str(value)
            for value in (
                SNAPSHOT_DAYS_BEFORE_ELECTION
            )
        )
    )

    warehouse = (
        load_historical_polling_warehouse()
    )

    print(
        f"Loaded {len(warehouse):,} "
        "candidate-answer rows."
    )

    race_profile = build_race_profile(
        warehouse
    )

    requests = build_snapshot_request_table(
        race_profile
    )

    print(
        f"Race-election combinations: "
        f"{len(race_profile):,}"
    )

    print(
        f"Requested snapshots: "
        f"{len(requests):,}"
    )

    (
        snapshot_questions,
        snapshot_summary,
        exclusion_summary,
    ) = build_snapshot_warehouse(
        warehouse,
        requests,
    )

    validation, failures = (
        validate_snapshot_warehouse(
            race_profile,
            requests,
            snapshot_questions,
            snapshot_summary,
        )
    )

    (
        cycle_profile,
        days_out_profile,
    ) = build_profiles(
        snapshot_summary
    )

    race_profile.to_csv(
        RACE_PROFILE_PATH,
        index=False,
    )

    snapshot_questions.to_csv(
        SNAPSHOT_QUESTIONS_PATH,
        index=False,
    )

    snapshot_summary.to_csv(
        SNAPSHOT_SUMMARY_PATH,
        index=False,
    )

    exclusion_summary.to_csv(
        EXCLUSION_SUMMARY_PATH,
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

    lines = [
        "",
        "=" * 92,
        (
            "SENATE HISTORICAL POLL "
            "SNAPSHOT WAREHOUSE VALIDATION"
        ),
        "=" * 92,
        "",
        (
            "Race-election combinations:       "
            f"{len(race_profile):,}"
        ),
        (
            "Expected snapshots:               "
            f"{validation['expected_snapshot_count']:,}"
        ),
        (
            "Actual snapshots:                 "
            f"{validation['actual_snapshot_count']:,}"
        ),
        (
            "Question-snapshot rows:            "
            f"{validation['question_snapshot_rows']:,}"
        ),
        (
            "Snapshots with eligible polling:  "
            f"{validation['snapshots_with_polling']:,}"
        ),
        (
            "Snapshots without polling:        "
            f"{validation['snapshots_without_polling']:,}"
        ),
        "",
        "BLOCKING CHECKS",
        "-" * 92,
    ]

    label_map = {
        "snapshot_request_count": (
            "Snapshot request count"
        ),
        "snapshot_summary_count": (
            "Snapshot summary count"
        ),
        "unique_snapshot_ids": (
            "Unique snapshot IDs"
        ),
        "request_summary_alignment": (
            "Request-summary alignment"
        ),
        "question_rows_present": (
            "Question rows present"
        ),
        "unique_snapshot_question": (
            "Unique snapshot-question rows"
        ),
        "no_snapshot_date_leakage": (
            "No snapshot-date leakage"
        ),
        "no_election_date_leakage": (
            "No Election Day leakage"
        ),
        "nonnegative_poll_age": (
            "Nonnegative poll age"
        ),
        "eligible_only": (
            "Eligible questions only"
        ),
        "exactly_one_democrat": (
            "Exactly one Democrat"
        ),
        "exactly_one_republican": (
            "Exactly one Republican"
        ),
        "dem_margin_recalculates": (
            "Democratic margin recalculates"
        ),
        "summary_question_counts_match": (
            "Summary question counts match"
        ),
        "eligible_counts_monotonic": (
            "Eligible counts monotonic"
        ),
        "complete_days_out_grid": (
            "Complete days-out grid"
        ),
    }

    for key, label in label_map.items():
        if key not in checks:
            continue

        lines.append(
            f"{label + ':':<38}"
            f"{'PASSED' if checks[key] else 'FAILED'}"
        )

    lines.extend(
        [
            "",
            (
                "Snapshot warehouse validation: "
                f"{'PASSED' if validation['passed'] else 'FAILED'}"
            ),
            "",
            "Outputs:",
            (
                "  Race profile: "
                f"{RACE_PROFILE_PATH}"
            ),
            (
                "  Snapshot questions: "
                f"{SNAPSHOT_QUESTIONS_PATH}"
            ),
            (
                "  Snapshot summary: "
                f"{SNAPSHOT_SUMMARY_PATH}"
            ),
            (
                "  Exclusion summary: "
                f"{EXCLUSION_SUMMARY_PATH}"
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
