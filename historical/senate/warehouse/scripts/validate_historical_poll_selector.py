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
    / "validation"
    / "polling"
    / "historical_poll_selector"
)

VALIDATION_SUMMARY_JSON = (
    OUTPUT_DIR
    / "selector_validation_summary.json"
)

VALIDATION_SUMMARY_TXT = (
    OUTPUT_DIR
    / "selector_validation_summary.txt"
)

SNAPSHOT_PROFILE_CSV = (
    OUTPUT_DIR
    / "snapshot_profile.csv"
)

EXCLUSION_PROFILE_CSV = (
    OUTPUT_DIR
    / "exclusion_profile.csv"
)

ELIGIBLE_QUESTIONS_CSV = (
    OUTPUT_DIR
    / "sample_eligible_questions.csv"
)

QUESTION_DIAGNOSTICS_CSV = (
    OUTPUT_DIR
    / "sample_question_diagnostics.csv"
)


def json_safe(
    value: Any,
) -> Any:
    if isinstance(
        value,
        np.integer,
    ):
        return int(value)

    if isinstance(
        value,
        np.floating,
    ):
        if np.isnan(value):
            return None

        return float(value)

    if isinstance(
        value,
        pd.Timestamp,
    ):
        return value.isoformat()

    if pd.isna(value):
        return None

    return value


def choose_validation_races(
    warehouse: pd.DataFrame,
) -> pd.DataFrame:
    profile = (
        warehouse.groupby(
            [
                "cycle",
                "state",
                "race_id",
                "election_date",
            ],
            dropna=False,
        )
        .agg(
            candidate_rows=(
                "question_id",
                "size",
            ),
            questions=(
                "question_id",
                "nunique",
            ),
        )
        .reset_index()
    )

    profile = profile.loc[
        profile["state"].notna()
        & profile["race_id"].notna()
        & profile["election_date"].notna()
    ].copy()

    profile["race_id"] = (
        pd.to_numeric(
            profile["race_id"],
            errors="coerce",
        )
        .astype("Int64")
    )

    chosen = (
        profile.sort_values(
            [
                "cycle",
                "questions",
                "state",
                "race_id",
                "election_date",
            ],
            ascending=[
                True,
                False,
                True,
                True,
                True,
            ],
        )
        .groupby(
            "cycle",
            as_index=False,
        )
        .head(1)
        .reset_index(drop=True)
    )

    return chosen


def build_snapshot_requests(
    chosen_races: pd.DataFrame,
) -> list[dict[str, Any]]:
    requests: list[
        dict[str, Any]
    ] = []

    for row in chosen_races.itertuples(
        index=False
    ):
        election_date = pd.Timestamp(
            row.election_date
        ).normalize()

        for days_before in (
            60,
            30,
            7,
            0,
        ):
            requests.append(
                {
                    "cycle": int(
                        row.cycle
                    ),
                    "state": str(
                        row.state
                    ),
                    "race_id": int(
                        row.race_id
                    ),
                    "election_date": (
                        election_date
                    ),
                    "days_before_election": (
                        days_before
                    ),
                    "snapshot_date": (
                        election_date
                        - pd.Timedelta(
                            days=days_before
                        )
                    ),
                }
            )

    return requests


def main() -> int:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 92)
    print("VALIDATE HISTORICAL SENATE POLL SELECTOR")
    print("=" * 92)
    print(
        f"Warehouse: {DEFAULT_WAREHOUSE_PATH}"
    )

    warehouse = (
        load_historical_polling_warehouse()
    )

    print(
        f"Loaded {len(warehouse):,} "
        "candidate-answer rows."
    )

    chosen_races = (
        choose_validation_races(
            warehouse
        )
    )

    requests = (
        build_snapshot_requests(
            chosen_races
        )
    )

    snapshot_records: list[
        dict[str, Any]
    ] = []

    exclusion_records: list[
        dict[str, Any]
    ] = []

    sample_eligible_frames = []
    sample_diagnostic_frames = []

    failures: list[str] = []

    previous_eligible: dict[
        tuple[int, str],
        int,
    ] = {}

    for request in requests:
        result = (
            select_historical_senate_polls(
                warehouse,
                cycle=request["cycle"],
                state=request["state"],
                race_id=request["race_id"],
                snapshot_date=(
                    request[
                        "snapshot_date"
                    ]
                ),
            )
        )

        eligible = (
            result.eligible_questions
        )

        diagnostics = (
            result.question_diagnostics
        )

        summary = dict(
            result.request_summary
        )

        summary["race_id"] = (
            request["race_id"]
        )

        summary[
            "days_before_election"
        ] = request[
            "days_before_election"
        ]

        summary["election_date"] = (
            request["election_date"]
            .date()
            .isoformat()
        )

        summary[
            "eligible_dem_margin_mean"
        ] = (
            float(
                eligible[
                    "dem_margin"
                ].mean()
            )
            if not eligible.empty
            else None
        )

        summary[
            "eligible_dem_margin_median"
        ] = (
            float(
                eligible[
                    "dem_margin"
                ].median()
            )
            if not eligible.empty
            else None
        )

        summary[
            "eligible_internal_questions"
        ] = (
            int(
                eligible[
                    "is_internal"
                ]
                .fillna(False)
                .sum()
            )
            if not eligible.empty
            else 0
        )

        summary[
            "eligible_partisan_questions"
        ] = (
            int(
                eligible[
                    "is_partisan"
                ]
                .fillna(False)
                .sum()
            )
            if not eligible.empty
            else 0
        )

        snapshot_records.append(
            summary
        )

        exclusions = (
            result.exclusion_summary
            .copy()
        )

        exclusions["cycle"] = (
            request["cycle"]
        )
        exclusions["state"] = (
            request["state"]
        )
        exclusions["race_id"] = (
            request["race_id"]
        )
        exclusions[
            "snapshot_date"
        ] = request[
            "snapshot_date"
        ].date().isoformat()

        exclusions[
            "days_before_election"
        ] = request[
            "days_before_election"
        ]

        exclusion_records.extend(
            exclusions.to_dict(
                orient="records"
            )
        )

        sample_eligible = (
            eligible.copy()
        )

        sample_eligible[
            "validation_cycle"
        ] = request["cycle"]

        sample_eligible[
            "validation_state"
        ] = request["state"]

        sample_eligible[
            "validation_snapshot_date"
        ] = request[
            "snapshot_date"
        ]

        sample_eligible[
            "validation_days_before"
        ] = request[
            "days_before_election"
        ]

        sample_eligible_frames.append(
            sample_eligible
        )

        sample_diagnostics = (
            diagnostics.copy()
        )

        sample_diagnostics[
            "validation_cycle"
        ] = request["cycle"]

        sample_diagnostics[
            "validation_state"
        ] = request["state"]

        sample_diagnostics[
            "validation_snapshot_date"
        ] = request[
            "snapshot_date"
        ]

        sample_diagnostics[
            "validation_days_before"
        ] = request[
            "days_before_election"
        ]

        sample_diagnostic_frames.append(
            sample_diagnostics
        )

        # --------------------------------------------------------------
        # Leakage checks
        # --------------------------------------------------------------

        if not eligible.empty:
            after_snapshot = (
                eligible["end_date"]
                > request[
                    "snapshot_date"
                ]
            )

            if after_snapshot.any():
                failures.append(
                    "Eligible questions after "
                    "snapshot date: "
                    f"{request}"
                )

            after_election = (
                eligible[
                    "election_date"
                ].notna()
                & (
                    eligible["end_date"]
                    > eligible[
                        "election_date"
                    ]
                )
            )

            if after_election.any():
                failures.append(
                    "Eligible questions after "
                    "Election Day: "
                    f"{request}"
                )

            if (
                ~eligible[
                    "is_general_stage"
                ]
            ).any():
                failures.append(
                    "Non-general question "
                    "marked eligible: "
                    f"{request}"
                )

            if eligible[
                "is_hypothetical"
            ].any():
                failures.append(
                    "Hypothetical question "
                    "marked eligible: "
                    f"{request}"
                )

            if eligible[
                "is_ranked_choice"
            ].any():
                failures.append(
                    "Ranked-choice question "
                    "marked eligible: "
                    f"{request}"
                )

            if (
                eligible[
                    "democratic_rows"
                ]
                != 1
            ).any():
                failures.append(
                    "Eligible question without "
                    "exactly one Democrat: "
                    f"{request}"
                )

            if (
                eligible[
                    "republican_rows"
                ]
                != 1
            ).any():
                failures.append(
                    "Eligible question without "
                    "exactly one Republican: "
                    f"{request}"
                )

            recalculated_margin = (
                eligible["dem_pct"]
                - eligible["rep_pct"]
            )

            margin_difference = (
                recalculated_margin
                - eligible["dem_margin"]
            ).abs()

            if (
                margin_difference
                > 1e-10
            ).any():
                failures.append(
                    "Democratic margin "
                    "calculation mismatch: "
                    f"{request}"
                )

            if eligible[
                "question_id"
            ].duplicated().any():
                failures.append(
                    "Duplicate eligible "
                    "question IDs: "
                    f"{request}"
                )

        # --------------------------------------------------------------
        # Snapshot monotonicity check
        # --------------------------------------------------------------

        key = (
            request["cycle"],
            request["state"],
            request["race_id"],
        )

        current_count = len(
            eligible
        )

        if key in previous_eligible:
            previous_count = (
                previous_eligible[key]
            )

            if current_count < previous_count:
                failures.append(
                    "Eligible question count "
                    "decreased as snapshot "
                    "advanced: "
                    f"{request}; "
                    f"previous={previous_count}, "
                    f"current={current_count}"
                )

        previous_eligible[key] = (
            current_count
        )

        print(
            f"Cycle {request['cycle']} "
            f"{request['state']} "
            f"race {request['race_id']} "
            f"{request['days_before_election']:>2} "
            "days before election: "
            f"{len(eligible):>3} eligible "
            "questions"
        )

    snapshot_profile = pd.DataFrame(
        snapshot_records
    ).sort_values(
        [
            "cycle",
            "state",
            "snapshot_date",
        ]
    )

    exclusion_profile = pd.DataFrame(
        exclusion_records
    ).sort_values(
        [
            "cycle",
            "state",
            "snapshot_date",
            "question_count",
            "reason",
        ],
        ascending=[
            True,
            True,
            True,
            False,
            True,
        ],
    )

    eligible_questions = (
        pd.concat(
            sample_eligible_frames,
            ignore_index=True,
        )
        if sample_eligible_frames
        else pd.DataFrame()
    )

    diagnostics = (
        pd.concat(
            sample_diagnostic_frames,
            ignore_index=True,
        )
        if sample_diagnostic_frames
        else pd.DataFrame()
    )

    for frame in (
        snapshot_profile,
        exclusion_profile,
        eligible_questions,
        diagnostics,
    ):
        for column in frame.columns:
            if "date" in column:
                try:
                    frame[column] = (
                        pd.to_datetime(
                            frame[column],
                            errors="ignore",
                        )
                    )
                except Exception:
                    pass

    snapshot_profile.to_csv(
        SNAPSHOT_PROFILE_CSV,
        index=False,
    )

    exclusion_profile.to_csv(
        EXCLUSION_PROFILE_CSV,
        index=False,
    )

    eligible_questions.to_csv(
        ELIGIBLE_QUESTIONS_CSV,
        index=False,
    )

    diagnostics.to_csv(
        QUESTION_DIAGNOSTICS_CSV,
        index=False,
    )

    cycles_found = sorted(
        snapshot_profile[
            "cycle"
        ].unique().tolist()
    )

    expected_cycles = [
        2018,
        2020,
        2022,
        2024,
    ]

    if cycles_found != expected_cycles:
        failures.append(
            "Validation cycles do not match "
            f"expected cycles: {cycles_found}"
        )

    if len(snapshot_profile) != 16:
        failures.append(
            "Expected 16 validation snapshots; "
            f"found {len(snapshot_profile)}."
        )

    if (
        snapshot_profile[
            "candidate_questions"
        ]
        <= 0
    ).any():
        failures.append(
            "At least one validation state-cycle "
            "contains no candidate questions."
        )

    if (
        snapshot_profile.loc[
            snapshot_profile[
                "days_before_election"
            ].eq(0),
            "eligible_questions",
        ]
        <= 0
    ).any():
        failures.append(
            "At least one Election Day snapshot "
            "contains no eligible questions."
        )

    passed = not failures

    summary = {
        "warehouse_path": str(
            DEFAULT_WAREHOUSE_PATH
        ),
        "warehouse_candidate_rows": int(
            len(warehouse)
        ),
        "validation_cycles": cycles_found,
        "validation_snapshots": int(
            len(snapshot_profile)
        ),
        "selected_state_by_cycle": {
            str(int(row.cycle)): str(
                row.state
            )
            for row in chosen_races.itertuples(
                index=False
            )
        },
        "total_sample_eligible_rows": int(
            len(eligible_questions)
        ),
        "validation_failure_count": int(
            len(failures)
        ),
        "validation_failures": failures,
        "passed": passed,
    }

    VALIDATION_SUMMARY_JSON.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
            default=json_safe,
        )
        + "\n",
        encoding="utf-8",
    )

    lines = [
        "",
        "=" * 92,
        "HISTORICAL POLL SELECTOR VALIDATION",
        "=" * 92,
        "",
        (
            "Warehouse candidate rows:          "
            f"{len(warehouse):,}"
        ),
        (
            "Validation cycles:                "
            + ", ".join(
                str(cycle)
                for cycle in cycles_found
            )
        ),
        (
            "Validation snapshots:             "
            f"{len(snapshot_profile):,}"
        ),
        (
            "Validation failures:              "
            f"{len(failures):,}"
        ),
        "",
        "SELECTED TEST RACE BY CYCLE",
        "-" * 92,
    ]

    for row in chosen_races.itertuples(
        index=False
    ):
        lines.append(
            f"{int(row.cycle)}: "
            f"{row.state}, "
            f"race_id={int(row.race_id)}, "
            f"election_date="
            f"{pd.Timestamp(row.election_date).date()}, "
            f"({int(row.questions):,} "
            "candidate questions)"
        )

    lines.extend(
        [
            "",
            "BLOCKING CHECKS",
            "-" * 92,
            (
                "Snapshot-date leakage:          "
                + (
                    "PASSED"
                    if not any(
                        "snapshot date"
                        in failure
                        for failure in failures
                    )
                    else "FAILED"
                )
            ),
            (
                "Election-date leakage:          "
                + (
                    "PASSED"
                    if not any(
                        "Election Day"
                        in failure
                        for failure in failures
                    )
                    else "FAILED"
                )
            ),
            (
                "General-election filtering:     "
                + (
                    "PASSED"
                    if not any(
                        "Non-general"
                        in failure
                        for failure in failures
                    )
                    else "FAILED"
                )
            ),
            (
                "Hypothetical filtering:         "
                + (
                    "PASSED"
                    if not any(
                        "Hypothetical"
                        in failure
                        for failure in failures
                    )
                    else "FAILED"
                )
            ),
            (
                "Ranked-choice filtering:        "
                + (
                    "PASSED"
                    if not any(
                        "Ranked-choice"
                        in failure
                        for failure in failures
                    )
                    else "FAILED"
                )
            ),
            (
                "Major-party matchup validation: "
                + (
                    "PASSED"
                    if not any(
                        (
                            "Democrat"
                            in failure
                            or "Republican"
                            in failure
                        )
                        for failure in failures
                    )
                    else "FAILED"
                )
            ),
            (
                "Democratic-margin calculation:  "
                + (
                    "PASSED"
                    if not any(
                        "margin calculation"
                        in failure
                        for failure in failures
                    )
                    else "FAILED"
                )
            ),
            (
                "Question uniqueness:            "
                + (
                    "PASSED"
                    if not any(
                        "Duplicate eligible"
                        in failure
                        for failure in failures
                    )
                    else "FAILED"
                )
            ),
            (
                "Snapshot monotonicity:          "
                + (
                    "PASSED"
                    if not any(
                        "decreased"
                        in failure
                        for failure in failures
                    )
                    else "FAILED"
                )
            ),
            "",
            (
                "Historical poll selector:       "
                f"{'PASSED' if passed else 'FAILED'}"
            ),
            "",
            "Outputs:",
            (
                "  Snapshot profile: "
                f"{SNAPSHOT_PROFILE_CSV}"
            ),
            (
                "  Exclusion profile: "
                f"{EXCLUSION_PROFILE_CSV}"
            ),
            (
                "  Eligible questions: "
                f"{ELIGIBLE_QUESTIONS_CSV}"
            ),
            (
                "  Question diagnostics: "
                f"{QUESTION_DIAGNOSTICS_CSV}"
            ),
            (
                "  JSON summary: "
                f"{VALIDATION_SUMMARY_JSON}"
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

    VALIDATION_SUMMARY_TXT.write_text(
        text,
        encoding="utf-8",
    )

    print(text)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
