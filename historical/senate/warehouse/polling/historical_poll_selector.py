#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]

DEFAULT_WAREHOUSE_PATH = (
    PROJECT_ROOT
    / "historical"
    / "senate"
    / "warehouse"
    / "processed"
    / "senate_historical_polls.csv"
)


@dataclass(frozen=True)
class HistoricalPollSelection:
    eligible_questions: pd.DataFrame
    question_diagnostics: pd.DataFrame
    exclusion_summary: pd.DataFrame
    request_summary: dict[str, Any]


def normalize_state(value: str) -> str:
    text = str(value).strip().upper()

    if len(text) != 2:
        raise ValueError(
            f"State must be a two-letter abbreviation: {value!r}"
        )

    return text


def parse_snapshot_date(value: Any) -> pd.Timestamp:
    parsed = pd.to_datetime(
        value,
        errors="raise",
    )

    if isinstance(parsed, pd.DatetimeIndex):
        raise TypeError(
            "Expected one snapshot date, not a sequence."
        )

    if parsed.tzinfo is not None:
        parsed = parsed.tz_convert(None)

    return pd.Timestamp(parsed).normalize()


def parse_boolean_series(
    series: pd.Series,
) -> pd.Series:
    if str(series.dtype) == "boolean":
        return series

    if series.dtype == bool:
        return series.astype("boolean")

    text = (
        series.astype("string")
        .str.strip()
        .str.lower()
    )

    result = pd.Series(
        pd.NA,
        index=series.index,
        dtype="boolean",
    )

    result.loc[
        text.isin(
            {
                "true",
                "t",
                "1",
                "yes",
                "y",
            }
        )
    ] = True

    result.loc[
        text.isin(
            {
                "false",
                "f",
                "0",
                "no",
                "n",
            }
        )
    ] = False

    return result


def load_historical_polling_warehouse(
    path: Path | str = DEFAULT_WAREHOUSE_PATH,
) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Historical polling warehouse not found: {path}"
        )

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    required_columns = {
        "poll_id",
        "question_id",
        "race_id",
        "candidate_id",
        "cycle",
        "state",
        "start_date",
        "end_date",
        "election_date",
        "stage",
        "candidate_name",
        "party",
        "pct",
    }

    missing = sorted(
        required_columns - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "Historical polling warehouse is missing "
            "required columns: "
            + ", ".join(missing)
        )

    for column in (
        "start_date",
        "end_date",
        "election_date",
        "created_at",
        "as_of_date",
    ):
        if column in df.columns:
            df[column] = pd.to_datetime(
                df[column],
                errors="coerce",
            )

    for column in (
        "poll_id",
        "question_id",
        "race_id",
        "candidate_id",
        "cycle",
        "sample_size",
        "pollster_id",
        "pollster_rating_id",
        "ranked_choice_round",
    ):
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            ).astype("Int64")

    for column in (
        "pct",
        "numeric_grade",
        "pollscore",
        "transparency_score",
    ):
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            ).astype("Float64")

    for column in (
        "is_hypothetical",
        "is_ranked_choice_row",
        "is_ranked_choice_reallocated",
        "is_internal",
        "is_partisan",
        "is_tracking",
        "is_nationwide_batch",
        "is_standard_general_election",
        "question_has_both_major_parties",
    ):
        if column in df.columns:
            df[column] = parse_boolean_series(
                df[column]
            )

    df["state"] = (
        df["state"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    df["party"] = (
        df["party"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    return df


def first_nonnull(
    series: pd.Series,
) -> Any:
    nonnull = series.dropna()

    if nonnull.empty:
        return pd.NA

    return nonnull.iloc[0]


def unique_nonnull_count(
    series: pd.Series,
) -> int:
    return int(
        series.dropna().nunique()
    )


def boolean_any(
    group: pd.DataFrame,
    column: str,
) -> bool:
    if column not in group.columns:
        return False

    return bool(
        group[column]
        .fillna(False)
        .astype(bool)
        .any()
    )


def build_question_record(
    question_id: Any,
    group: pd.DataFrame,
) -> dict[str, Any]:
    parties = (
        group["party"]
        .astype("string")
        .str.upper()
    )

    pct = pd.to_numeric(
        group["pct"],
        errors="coerce",
    )

    dem_mask = parties.eq("D")
    rep_mask = parties.eq("R")

    dem_rows = group.loc[dem_mask]
    rep_rows = group.loc[rep_mask]

    dem_pct_values = pd.to_numeric(
        dem_rows["pct"],
        errors="coerce",
    ).dropna()

    rep_pct_values = pd.to_numeric(
        rep_rows["pct"],
        errors="coerce",
    ).dropna()

    dem_pct = (
        float(dem_pct_values.iloc[0])
        if len(dem_pct_values) == 1
        else np.nan
    )

    rep_pct = (
        float(rep_pct_values.iloc[0])
        if len(rep_pct_values) == 1
        else np.nan
    )

    stage = first_nonnull(
        group["stage"]
    )

    stage_normalized = (
        str(stage).strip().lower()
        if not pd.isna(stage)
        else ""
    )

    election_dates = (
        group["election_date"]
        .dropna()
        .drop_duplicates()
    )

    election_date = (
        election_dates.iloc[0]
        if len(election_dates) == 1
        else pd.NaT
    )

    start_dates = (
        group["start_date"]
        .dropna()
        .drop_duplicates()
    )

    end_dates = (
        group["end_date"]
        .dropna()
        .drop_duplicates()
    )

    start_date = (
        start_dates.iloc[0]
        if len(start_dates) == 1
        else pd.NaT
    )

    end_date = (
        end_dates.iloc[0]
        if len(end_dates) == 1
        else pd.NaT
    )

    record = {
        "question_id": question_id,
        "poll_id": first_nonnull(
            group["poll_id"]
        ),
        "race_id": first_nonnull(
            group["race_id"]
        ),
        "cycle": first_nonnull(
            group["cycle"]
        ),
        "state": first_nonnull(
            group["state"]
        ),
        "stage": stage,
        "start_date": start_date,
        "end_date": end_date,
        "election_date": election_date,
        "candidate_rows": int(
            len(group)
        ),
        "unique_candidates": int(
            group["candidate_id"].nunique(
                dropna=True
            )
        ),
        "democratic_rows": int(
            dem_mask.sum()
        ),
        "republican_rows": int(
            rep_mask.sum()
        ),
        "major_party_rows": int(
            (dem_mask | rep_mask).sum()
        ),
        "dem_candidate_id": (
            first_nonnull(
                dem_rows["candidate_id"]
            )
            if not dem_rows.empty
            else pd.NA
        ),
        "rep_candidate_id": (
            first_nonnull(
                rep_rows["candidate_id"]
            )
            if not rep_rows.empty
            else pd.NA
        ),
        "dem_candidate_name": (
            first_nonnull(
                dem_rows["candidate_name"]
            )
            if not dem_rows.empty
            else pd.NA
        ),
        "rep_candidate_name": (
            first_nonnull(
                rep_rows["candidate_name"]
            )
            if not rep_rows.empty
            else pd.NA
        ),
        "dem_pct": dem_pct,
        "rep_pct": rep_pct,
        "major_party_pct_sum": (
            dem_pct + rep_pct
            if not np.isnan(dem_pct)
            and not np.isnan(rep_pct)
            else np.nan
        ),
        "dem_margin": (
            dem_pct - rep_pct
            if not np.isnan(dem_pct)
            and not np.isnan(rep_pct)
            else np.nan
        ),
        "question_pct_sum": float(
            pct.sum()
        ),
        "has_one_democrat": bool(
            dem_mask.sum() == 1
        ),
        "has_one_republican": bool(
            rep_mask.sum() == 1
        ),
        "has_both_major_parties": bool(
            dem_mask.any()
            and rep_mask.any()
        ),
        "has_valid_major_party_pct": bool(
            len(dem_pct_values) == 1
            and len(rep_pct_values) == 1
            and dem_pct_values.iloc[0] >= 0
            and dem_pct_values.iloc[0] <= 100
            and rep_pct_values.iloc[0] >= 0
            and rep_pct_values.iloc[0] <= 100
        ),
        "is_general_stage": bool(
            stage_normalized == "general"
        ),
        "is_hypothetical": boolean_any(
            group,
            "is_hypothetical",
        ),
        "is_ranked_choice": (
            boolean_any(
                group,
                "is_ranked_choice_row",
            )
            or boolean_any(
                group,
                "is_ranked_choice_reallocated",
            )
            or (
                "ranked_choice_round"
                in group.columns
                and group[
                    "ranked_choice_round"
                ].notna().any()
            )
        ),
        "is_internal": boolean_any(
            group,
            "is_internal",
        ),
        "is_partisan": boolean_any(
            group,
            "is_partisan",
        ),
        "is_tracking": boolean_any(
            group,
            "is_tracking",
        ),
        "metadata_consistent": bool(
            unique_nonnull_count(
                group["poll_id"]
            ) <= 1
            and unique_nonnull_count(
                group["race_id"]
            ) <= 1
            and unique_nonnull_count(
                group["cycle"]
            ) <= 1
            and unique_nonnull_count(
                group["state"]
            ) <= 1
            and unique_nonnull_count(
                group["start_date"]
            ) <= 1
            and unique_nonnull_count(
                group["end_date"]
            ) <= 1
        ),
    }

    optional_first_columns = (
        "pollster",
        "pollster_id",
        "pollster_rating_id",
        "display_name",
        "pollster_name",
        "sponsors",
        "poll_url",
        "poll_source",
        "population",
        "population_full",
        "sample_size",
        "methodology",
        "numeric_grade",
        "pollscore",
        "transparency_score",
        "notes",
    )

    for column in optional_first_columns:
        record[column] = (
            first_nonnull(
                group[column]
            )
            if column in group.columns
            else pd.NA
        )

    return record


def build_question_table(
    rows: pd.DataFrame,
) -> pd.DataFrame:
    records = [
        build_question_record(
            question_id,
            group,
        )
        for question_id, group in rows.groupby(
            "question_id",
            dropna=False,
            sort=False,
        )
    ]

    questions = pd.DataFrame(
        records
    )

    if questions.empty:
        return questions

    for column in (
        "start_date",
        "end_date",
        "election_date",
    ):
        questions[column] = pd.to_datetime(
            questions[column],
            errors="coerce",
        )

    return questions


def assign_exclusion_reasons(
    questions: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    *,
    include_hypothetical: bool,
    include_ranked_choice: bool,
    require_general_stage: bool,
    require_exactly_one_dem_and_rep: bool,
) -> pd.DataFrame:
    result = questions.copy()

    result["excluded_after_snapshot"] = (
        result["end_date"].isna()
        | (
            result["end_date"]
            > snapshot_date
        )
    )

    result["excluded_after_election"] = (
        result["election_date"].notna()
        & result["end_date"].notna()
        & (
            result["end_date"]
            > result["election_date"]
        )
    )

    result["excluded_non_general_stage"] = (
        require_general_stage
        & ~result["is_general_stage"]
    )

    result["excluded_hypothetical"] = (
        not include_hypothetical
    ) & result["is_hypothetical"]

    result["excluded_ranked_choice"] = (
        not include_ranked_choice
    ) & result["is_ranked_choice"]

    result[
        "excluded_missing_major_party_matchup"
    ] = ~result[
        "has_both_major_parties"
    ]

    result[
        "excluded_multiple_democrats"
    ] = (
        require_exactly_one_dem_and_rep
        & (
            result["democratic_rows"] != 1
        )
    )

    result[
        "excluded_multiple_republicans"
    ] = (
        require_exactly_one_dem_and_rep
        & (
            result["republican_rows"] != 1
        )
    )

    result[
        "excluded_invalid_major_party_pct"
    ] = ~result[
        "has_valid_major_party_pct"
    ]

    result[
        "excluded_inconsistent_metadata"
    ] = ~result[
        "metadata_consistent"
    ]

    exclusion_columns = [
        column
        for column in result.columns
        if column.startswith("excluded_")
    ]

    result["is_eligible"] = ~result[
        exclusion_columns
    ].any(axis=1)

    def combine_reasons(
        row: pd.Series,
    ) -> str:
        reasons = [
            column.removeprefix(
                "excluded_"
            )
            for column in exclusion_columns
            if bool(row[column])
        ]

        return "|".join(reasons)

    result["exclusion_reasons"] = (
        result.apply(
            combine_reasons,
            axis=1,
        )
    )

    result["snapshot_date"] = (
        snapshot_date
    )

    return result


def build_exclusion_summary(
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    exclusion_columns = [
        column
        for column in diagnostics.columns
        if column.startswith("excluded_")
    ]

    records = []

    for column in exclusion_columns:
        count = int(
            diagnostics[column]
            .fillna(False)
            .sum()
        )

        records.append(
            {
                "reason": column.removeprefix(
                    "excluded_"
                ),
                "question_count": count,
                "share_of_candidate_questions": (
                    float(
                        count
                        / len(diagnostics)
                    )
                    if len(diagnostics)
                    else 0.0
                ),
            }
        )

    records.append(
        {
            "reason": "eligible",
            "question_count": int(
                diagnostics[
                    "is_eligible"
                ].sum()
            ),
            "share_of_candidate_questions": (
                float(
                    diagnostics[
                        "is_eligible"
                    ].mean()
                )
                if len(diagnostics)
                else 0.0
            ),
        }
    )

    return pd.DataFrame(
        records
    ).sort_values(
        [
            "question_count",
            "reason",
        ],
        ascending=[
            False,
            True,
        ],
    )


def select_historical_senate_polls(
    warehouse: pd.DataFrame,
    *,
    cycle: int,
    state: str,
    snapshot_date: Any,
    race_id: int | None = None,
    include_hypothetical: bool = False,
    include_ranked_choice: bool = False,
    require_general_stage: bool = True,
    require_exactly_one_dem_and_rep: bool = True,
) -> HistoricalPollSelection:
    state = normalize_state(
        state
    )

    snapshot_date = parse_snapshot_date(
        snapshot_date
    )

    cycle_rows = warehouse.loc[
        warehouse["cycle"].eq(
            int(cycle)
        )
        & warehouse["state"].eq(
            state
        )
    ].copy()

    if race_id is not None:
        cycle_rows = cycle_rows.loc[
            cycle_rows["race_id"].eq(
                int(race_id)
            )
        ].copy()

    questions = build_question_table(
        cycle_rows
    )

    if questions.empty:
        empty_summary = pd.DataFrame(
            columns=[
                "reason",
                "question_count",
                "share_of_candidate_questions",
            ]
        )

        return HistoricalPollSelection(
            eligible_questions=questions,
            question_diagnostics=questions,
            exclusion_summary=empty_summary,
            request_summary={
                "cycle": int(cycle),
                "state": state,
                "race_id": race_id,
                "snapshot_date": (
                    snapshot_date.date()
                    .isoformat()
                ),
                "candidate_rows": int(
                    len(cycle_rows)
                ),
                "candidate_questions": 0,
                "eligible_questions": 0,
            },
        )

    diagnostics = assign_exclusion_reasons(
        questions,
        snapshot_date,
        include_hypothetical=(
            include_hypothetical
        ),
        include_ranked_choice=(
            include_ranked_choice
        ),
        require_general_stage=(
            require_general_stage
        ),
        require_exactly_one_dem_and_rep=(
            require_exactly_one_dem_and_rep
        ),
    )

    eligible = diagnostics.loc[
        diagnostics["is_eligible"]
    ].copy()

    eligible["days_old"] = (
        snapshot_date
        - eligible["end_date"]
    ).dt.days.astype("Int64")

    eligible[
        "days_before_election_at_snapshot"
    ] = (
        eligible["election_date"]
        - snapshot_date
    ).dt.days.astype("Int64")

    eligible = eligible.sort_values(
        [
            "end_date",
            "poll_id",
            "question_id",
        ],
        ascending=[
            False,
            True,
            True,
        ],
    ).reset_index(drop=True)

    diagnostics = diagnostics.sort_values(
        [
            "end_date",
            "poll_id",
            "question_id",
        ],
        ascending=[
            False,
            True,
            True,
        ],
        na_position="last",
    ).reset_index(drop=True)

    exclusion_summary = (
        build_exclusion_summary(
            diagnostics
        )
    )

    request_summary = {
        "cycle": int(cycle),
        "state": state,
        "race_id": race_id,
        "snapshot_date": (
            snapshot_date.date()
            .isoformat()
        ),
        "candidate_rows": int(
            len(cycle_rows)
        ),
        "candidate_questions": int(
            len(diagnostics)
        ),
        "eligible_questions": int(
            len(eligible)
        ),
        "excluded_questions": int(
            len(diagnostics)
            - len(eligible)
        ),
        "latest_eligible_end_date": (
            eligible["end_date"]
            .max()
            .date()
            .isoformat()
            if not eligible.empty
            else None
        ),
        "earliest_eligible_end_date": (
            eligible["end_date"]
            .min()
            .date()
            .isoformat()
            if not eligible.empty
            else None
        ),
    }

    return HistoricalPollSelection(
        eligible_questions=eligible,
        question_diagnostics=diagnostics,
        exclusion_summary=exclusion_summary,
        request_summary=request_summary,
    )
