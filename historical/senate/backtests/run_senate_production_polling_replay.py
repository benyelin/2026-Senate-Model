from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

CANONICAL_PATH = (
    PROJECT_ROOT
    / "historical/senate/backtests/outputs/canonical/"
    / "senate_canonical_backtest_dataset.csv"
)

POLL_AGGREGATES_PATH = (
    PROJECT_ROOT
    / "historical/senate/warehouse/processed/polling_aggregates/"
    / "senate_historical_baseline_poll_aggregates.csv"
)

WEIGHTED_QUESTIONS_PATH = (
    PROJECT_ROOT
    / "historical/senate/warehouse/processed/polling_aggregates/"
    / "senate_historical_baseline_weighted_questions.csv"
)

CALIBRATION_PATH = (
    PROJECT_ROOT / "inputs/calibration_parameters.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "historical/senate/backtests/outputs/"
    / "production_polling_replay"
)

RACE_OUTPUT_PATH = OUTPUT_DIR / "senate_production_polling_replay_races.csv"
VALIDATION_PATH = OUTPUT_DIR / "senate_production_polling_replay_validation.json"

EXPECTED_DAYS = [120, 90, 60, 30, 14, 7, 0]
EXPECTED_CYCLES = [2018, 2020, 2022, 2024]


def normalize_bool(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "t"})
    )


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def run_command(command: list[str], cwd: Path) -> None:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
    )

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(
            "Historical production replay subprocess failed:\n"
            + " ".join(command)
        )


def load_inputs():
    require(CANONICAL_PATH)
    require(POLL_AGGREGATES_PATH)
    require(WEIGHTED_QUESTIONS_PATH)
    require(CALIBRATION_PATH)

    canonical = pd.read_csv(CANONICAL_PATH, low_memory=False)
    polls = pd.read_csv(POLL_AGGREGATES_PATH, low_memory=False)

    # The weighted-question warehouse is larger and may live on a
    # Desktop/iCloud-backed filesystem. Copy it once to local /tmp so a
    # long replay cannot fail because of a transient filesystem stall.
    cached_questions_path = Path(
        tempfile.gettempdir()
    ) / "senate_historical_baseline_weighted_questions.csv"

    shutil.copy2(
        WEIGHTED_QUESTIONS_PATH,
        cached_questions_path,
    )

    questions = pd.read_csv(
        cached_questions_path,
        low_memory=False,
    )

    canonical["race_id"] = canonical["race_id"].astype("string").str.strip()
    polls["race_id"] = polls["race_id"].astype("string").str.strip()

    canonical["cycle"] = pd.to_numeric(
        canonical["cycle"], errors="raise"
    ).astype(int)

    polls["cycle"] = pd.to_numeric(
        polls["cycle"], errors="raise"
    ).astype(int)

    polls["days_before_election"] = pd.to_numeric(
        polls["days_before_election"], errors="raise"
    ).astype(int)

    for column in ["snapshot_date", "election_date"]:
        if column in polls.columns:
            polls[column] = pd.to_datetime(
                polls[column], errors="raise"
            ).dt.normalize()

    if "snapshot_id" in questions.columns:
        questions["snapshot_id"] = (
            questions["snapshot_id"].astype("string").str.strip()
        )

    return canonical, polls, questions


def build_replay_universe(
    canonical: pd.DataFrame,
    polls: pd.DataFrame,
) -> pd.DataFrame:
    c = canonical.copy()

    if "backtest_scorable" in c.columns:
        c = c.loc[normalize_bool(c["backtest_scorable"])].copy()

    if "historical_fundamentals_scorable" in c.columns:
        c = c.loc[
            normalize_bool(c["historical_fundamentals_scorable"])
        ].copy()

    c = c.loc[c["cycle"].isin(EXPECTED_CYCLES)].copy()

    desired = [
        "race_id",
        "cycle",
        "state",
        "election_date",
        "actual_margin_dem",
        "winner_party",
        "production_predicted_margin_dem",
        "production_baseline_margin_dem",
        "production_national_environment_margin_dem",
        "generic_ballot_margin_dem",
        "model_incumbent_party",
        "senate_class",
        "seat_id",
        "election_type",
        "special_election",
    ]

    desired = [x for x in desired if x in c.columns]

    merged = polls.merge(
        c[desired],
        on=["race_id", "cycle"],
        how="inner",
        suffixes=("", "_canonical"),
        validate="many_to_one",
    )

    # A polling provider may create separate race IDs for an initial
    # special-election round and a later runoff. Both provider streams
    # are intentionally retained in the polling warehouse, but the
    # production replay must score only the stream corresponding to the
    # canonical historical election outcome.
    #
    # Example:
    #   2018_MS_SPECIAL provider 130  -> 2018-11-06
    #   2018_MS_SPECIAL provider 6209 -> 2018-11-27 runoff
    #
    # The canonical results warehouse scores the Nov. 27 runoff, so only
    # the matching provider stream belongs in the replay universe.

    if "election_date_canonical" in merged.columns:
        provider_date = pd.to_datetime(
            merged["election_date"],
            errors="coerce",
        ).dt.normalize()

        canonical_date = pd.to_datetime(
            merged["election_date_canonical"],
            errors="coerce",
        ).dt.normalize()

        date_match = provider_date.eq(canonical_date)

        # Apply this preference only where a canonical race has multiple
        # provider election streams. Ordinary races remain untouched.
        provider_stream_counts = (
            merged[
                [
                    "cycle",
                    "race_id",
                    "poll_race_id",
                    "election_date",
                ]
            ]
            .drop_duplicates()
            .groupby(["cycle", "race_id"])
            .size()
        )

        duplicated_races = set(
            provider_stream_counts[
                provider_stream_counts > 1
            ].index
        )

        duplicated_mask = pd.Series(
            [
                (int(cycle), str(race_id))
                in duplicated_races
                for cycle, race_id in zip(
                    merged["cycle"],
                    merged["race_id"],
                )
            ],
            index=merged.index,
        )

        ambiguous_without_match = (
            merged.loc[duplicated_mask]
            .assign(_date_match=date_match.loc[duplicated_mask])
            .groupby(["cycle", "race_id"])["_date_match"]
            .sum()
        )

        bad = ambiguous_without_match[
            ambiguous_without_match.eq(0)
        ]

        if not bad.empty:
            raise RuntimeError(
                "Canonical races with multiple polling-provider streams "
                "have no provider election-date match: "
                + ", ".join(
                    f"{cycle}:{race_id}"
                    for cycle, race_id in bad.index
                )
            )

        merged = merged.loc[
            ~duplicated_mask | date_match
        ].copy()

    duplicate_keys = merged.duplicated(
        ["cycle", "race_id", "days_before_election"],
        keep=False,
    )

    if duplicate_keys.any():
        raise RuntimeError(
            "Replay universe still contains duplicate canonical "
            "race/day snapshots after provider-stream filtering:\n"
            + merged.loc[
                duplicate_keys,
                [
                    "snapshot_id",
                    "cycle",
                    "state",
                    "race_id",
                    "poll_race_id",
                    "election_date",
                    "days_before_election",
                ],
            ]
            .sort_values(
                ["cycle", "race_id", "days_before_election"]
            )
            .to_string(index=False)
        )

    return merged.sort_values(
        ["cycle", "days_before_election", "race_id"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def write_race_inputs(
    row: pd.Series,
    input_dir: Path,
) -> None:
    fundamentals = float(row["production_predicted_margin_dem"])

    race = pd.DataFrame(
        [
            {
                "state": str(row["state"]).strip().upper(),
                "race_id": str(row["race_id"]),
                "cycle": int(row["cycle"]),
                "dem_candidate": "Historical Democratic nominee",
                "gop_candidate": "Historical Republican nominee",
                "current_holder": str(
                    row.get("model_incumbent_party", "Open")
                ),
                "race_tier": "historical_replay",
                "polling_margin_dem": (
                    float(row["baseline_poll_margin_dem"])
                    if bool(row["baseline_has_polling"])
                    and pd.notna(row["baseline_poll_margin_dem"])
                    else np.nan
                ),
                "fundamentals_margin_dem": fundamentals,
                "polling_active": bool(row["baseline_has_polling"]),
                "poll_count": float(
                    row.get("baseline_poll_count", 0.0)
                ),
                "latest_poll_end_date": row.get(
                    "latest_eligible_end_date", np.nan
                ),
                "avg_poll_age_days": float(
                    row.get("baseline_mean_poll_age_days", 999.0)
                )
                if pd.notna(row.get("baseline_mean_poll_age_days", np.nan))
                else 999.0,
                "elasticity": 1.0,
                "tier_error_multiplier": 1.0,
                "candidate_uncertainty_penalty": 0.0,
                "dem_win_counts_for_seat_change": 1.0,
                "actual_margin_dem": float(row["actual_margin_dem"]),
            }
        ]
    )

    race.to_csv(input_dir / "race_inputs.csv", index=False)


def write_polling_average(
    row: pd.Series,
    input_dir: Path,
) -> None:
    if (
        bool(row["baseline_has_polling"])
        and pd.notna(row["baseline_poll_margin_dem"])
        and float(row.get("baseline_poll_count", 0.0)) > 0
    ):
        frame = pd.DataFrame(
            [
                {
                    "state": str(row["state"]).strip().upper(),
                    "polling_margin_dem": float(
                        row["baseline_poll_margin_dem"]
                    ),
                    "poll_count": float(
                        row["baseline_poll_count"]
                    ),
                    "effective_poll_count": float(
                        row["baseline_effective_question_count"]
                    ),
                    "latest_poll_end_date": row.get(
                        "latest_eligible_end_date", np.nan
                    ),
                    "avg_poll_age_days": float(
                        row["baseline_mean_poll_age_days"]
                    ),
                    "total_poll_weight": float(
                        row["baseline_total_raw_weight"]
                    ),
                }
            ]
        )
    else:
        frame = pd.DataFrame(
            columns=[
                "state",
                "polling_margin_dem",
                "poll_count",
                "effective_poll_count",
                "latest_poll_end_date",
                "avg_poll_age_days",
                "total_poll_weight",
            ]
        )

    frame.to_csv(
        input_dir / "polling_averages_generated.csv",
        index=False,
    )


def choose_column(frame: pd.DataFrame, candidates: list[str]):
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def write_confidence_audit(
    row: pd.Series,
    questions: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Reconstruct the input used by cap_bayesian_poll_weight.py's
    polling-confidence finalizer from historical poll-question rows.

    We retain only rows from this snapshot and give the finalizer the
    state, poll end date, pollster and sample size it expects.
    """
    snapshot_id = str(row["snapshot_id"])

    q = questions.loc[
        questions["snapshot_id"].astype(str).eq(snapshot_id)
    ].copy()

    if q.empty:
        pd.DataFrame(
            columns=["state", "end_date", "pollster", "sample_size"]
        ).to_csv(
            output_dir / "senate_poll_weighting_live_audit.csv",
            index=False,
        )
        return

    pollster_col = choose_column(
        q,
        [
            "pollster",
            "pollster_name",
            "display_name",
            "sponsor",
        ],
    )

    end_col = choose_column(
        q,
        [
            "end_date",
            "poll_end_date",
            "field_end",
            "date",
            "poll_date",
        ],
    )

    sample_col = choose_column(
        q,
        [
            "sample_size",
            "sample",
            "n",
        ],
    )

    poll_id_col = choose_column(
        q,
        [
            "poll_id",
            "poll_id_numeric",
            "pollster_poll_id",
        ],
    )

    if end_col is None:
        raise RuntimeError(
            "Historical weighted-question warehouse has no "
            "poll end-date column."
        )

    audit = pd.DataFrame(index=q.index)
    audit["state"] = str(row["state"]).strip().upper()
    audit["end_date"] = q[end_col]

    if pollster_col is None:
        audit["pollster"] = "Historical pollster"
    else:
        audit["pollster"] = (
            q[pollster_col]
            .fillna("Historical pollster")
            .astype(str)
        )

    if sample_col is None:
        audit["sample_size"] = 600.0
    else:
        audit["sample_size"] = pd.to_numeric(
            q[sample_col], errors="coerce"
        ).fillna(600.0)

    # The live confidence layer conceptually counts polls, not candidate
    # answer rows. Deduplicate to one row per provider poll when possible.
    if poll_id_col is not None:
        audit["_poll_id"] = q[poll_id_col].astype(str)
        audit = audit.drop_duplicates("_poll_id")
        audit = audit.drop(columns="_poll_id")
    else:
        # Fallback: pollster/end-date/sample combination.
        audit = audit.drop_duplicates(
            ["pollster", "end_date", "sample_size"]
        )

    audit.to_csv(
        output_dir / "senate_poll_weighting_live_audit.csv",
        index=False,
    )


def run_one_snapshot(
    row: pd.Series,
    questions: pd.DataFrame,
) -> dict:
    with tempfile.TemporaryDirectory(
        prefix="senate_polling_replay_"
    ) as tmp:
        work = Path(tmp)
        input_dir = work / "inputs"
        output_dir = work / "outputs"

        input_dir.mkdir()
        output_dir.mkdir()

        write_race_inputs(row, input_dir)
        write_polling_average(row, input_dir)
        write_confidence_audit(row, questions, output_dir)

        shutil.copy2(
            CALIBRATION_PATH,
            input_dir / "calibration_parameters.csv",
        )

        days_out = int(row["days_before_election"])
        as_of = pd.Timestamp(row["snapshot_date"]).date().isoformat()

        run_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "bayesian_update.py"),
                "--input-dir",
                "inputs",
                "--days-out",
                str(days_out),
            ],
            work,
        )

        run_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "sync_bayesian_poll_metadata.py"),
            ],
            work,
        )

        run_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "cap_bayesian_poll_weight.py"),
                "--days-out",
                str(days_out),
                "--as-of",
                as_of,
            ],
            work,
        )

        run_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "sync_senate_model_fields.py"),
            ],
            work,
        )

        bayes = pd.read_csv(
            input_dir / "bayesian_update_generated.csv",
            low_memory=False,
        )

        races = pd.read_csv(
            input_dir / "race_inputs.csv",
            low_memory=False,
        )

        if len(bayes) != 1 or len(races) != 1:
            raise RuntimeError(
                "One-race replay workspace produced unexpected row count."
            )

        b = bayes.iloc[0]
        r = races.iloc[0]

        fundamentals = float(row["production_predicted_margin_dem"])

        polling_weight = float(
            pd.to_numeric(
                pd.Series([b.get("bayesian_polling_weight", 0.0)]),
                errors="coerce",
            ).fillna(0.0).iloc[0]
        )

        bayesian_margin = float(
            pd.to_numeric(
                pd.Series(
                    [
                        r.get(
                            "bayesian_model_margin_dem",
                            fundamentals,
                        )
                    ]
                ),
                errors="coerce",
            ).fillna(fundamentals).iloc[0]
        )

        posterior_sd = float(
            pd.to_numeric(
                pd.Series(
                    [
                        r.get(
                            "bayesian_posterior_sd",
                            np.nan,
                        )
                    ]
                ),
                errors="coerce",
            ).iloc[0]
        )

        return {
            "snapshot_id": row["snapshot_id"],
            "cycle": int(row["cycle"]),
            "state": row["state"],
            "race_id": row["race_id"],
            "snapshot_date": as_of,
            "days_out": days_out,
            "actual_margin_dem": float(row["actual_margin_dem"]),
            "fundamentals_margin_dem": fundamentals,
            "polling_margin_dem": (
                float(row["baseline_poll_margin_dem"])
                if pd.notna(row["baseline_poll_margin_dem"])
                else np.nan
            ),
            "has_polling": bool(row["baseline_has_polling"]),
            "poll_count": float(
                row.get("baseline_poll_count", 0.0)
            ),
            "pollster_count": float(
                row.get("baseline_pollster_count", 0.0)
            )
            if pd.notna(row.get("baseline_pollster_count", np.nan))
            else np.nan,
            "effective_poll_count": float(
                row.get("baseline_effective_question_count", 0.0)
            ),
            "mean_poll_age_days": float(
                row.get("baseline_mean_poll_age_days", np.nan)
            )
            if pd.notna(row.get("baseline_mean_poll_age_days", np.nan))
            else np.nan,
            "production_polling_weight": polling_weight,
            "production_bayesian_margin_dem": bayesian_margin,
            "production_posterior_sd": posterior_sd,
            "margin_movement_dem": bayesian_margin - fundamentals,
        }


def validate(
    universe: pd.DataFrame,
    replay: pd.DataFrame,
    require_full_grid: bool = True,
) -> dict:
    checks = {}

    checks["row_count_preserved"] = len(replay) == len(universe)

    checks["unique_cycle_race_days"] = not replay[
        ["cycle", "race_id", "days_out"]
    ].duplicated().any()

    checks["polling_weights_bounded"] = bool(
        replay["production_polling_weight"]
        .between(0.0, 1.0)
        .all()
    )

    checks["bayesian_margins_finite"] = bool(
        np.isfinite(
            replay["production_bayesian_margin_dem"]
        ).all()
    )

    no_poll = replay.loc[~replay["has_polling"]].copy()

    checks["no_poll_snapshots_unchanged"] = bool(
        np.allclose(
            no_poll["fundamentals_margin_dem"],
            no_poll["production_bayesian_margin_dem"],
            atol=1e-10,
            rtol=0.0,
        )
    )

    checks["no_poll_snapshots_zero_weight"] = bool(
        no_poll["production_polling_weight"]
        .abs()
        .le(1e-12)
        .all()
    )

    cycles = sorted(replay["cycle"].unique().tolist())

    if require_full_grid:
        checks["expected_days_present"] = (
            sorted(
                replay["days_out"].unique().tolist(),
                reverse=True,
            )
            == EXPECTED_DAYS
        )

        checks["expected_cycles_present"] = (
            cycles == EXPECTED_CYCLES
        )

    failures = [
        name for name, passed in checks.items() if not passed
    ]

    return {
        "input_rows": int(len(universe)),
        "replay_rows": int(len(replay)),
        "cycles": cycles,
        "days_out": sorted(
            replay["days_out"].unique().tolist(),
            reverse=True,
        ),
        "snapshots_with_polling": int(
            replay["has_polling"].sum()
        ),
        "snapshots_without_polling": int(
            (~replay["has_polling"]).sum()
        ),
        "checks": checks,
        "failure_count": len(failures),
        "failures": failures,
        "passed": not failures,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N race-snapshots for validation.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    canonical, polls, questions = load_inputs()
    universe = build_replay_universe(canonical, polls)

    print("=" * 96)
    print("SENATE PRODUCTION POLLING REPLAY")
    print("=" * 96)
    print(f"Replay universe: {len(universe):,} race-snapshots")

    if args.limit is not None:
        universe = universe.head(args.limit).copy()
        print(f"Validation limit: {len(universe):,}")

    records = []

    for i, row in universe.iterrows():
        records.append(
            run_one_snapshot(
                row=row,
                questions=questions,
            )
        )

        count = len(records)

        if count == 1 or count % 25 == 0 or count == len(universe):
            print(
                f"Processed {count:,}/{len(universe):,} "
                "race-snapshots."
            )

    replay = pd.DataFrame(records)

    validation = validate(
        universe,
        replay,
        require_full_grid=(args.limit is None),
    )

    replay.to_csv(RACE_OUTPUT_PATH, index=False)

    VALIDATION_PATH.write_text(
        json.dumps(validation, indent=2, sort_keys=True)
        + "\n"
    )

    print()
    print("=" * 96)
    print("VALIDATION")
    print("=" * 96)

    for name, passed in validation["checks"].items():
        print(
            f"{name:45s} "
            f"{'PASSED' if passed else 'FAILED'}"
        )

    print()
    print(
        "Replay validation:",
        "PASSED" if validation["passed"] else "FAILED",
    )

    print()
    print("Outputs:")
    print(" ", RACE_OUTPUT_PATH)
    print(" ", VALIDATION_PATH)

    if not validation["passed"]:
        raise SystemExit(
            "Production polling replay validation FAILED."
        )


if __name__ == "__main__":
    main()
