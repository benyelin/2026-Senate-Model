#!/usr/bin/env python3
"""
Version 1 Senate historical production replay.

Scope
-----
This runner evaluates the current production simulation engine using the
validated historical central-margin formula:

    presidential baseline + 0.90 * generic-ballot margin

It produces:

- race-level historical predictions and simulated win probabilities;
- overall and by-cycle margin/probability metrics;
- probability-bucket calibration;
- Democratic wins among the Senate races contested in each cycle.

Important limitation
--------------------
This version does NOT estimate historical full-chamber control probability.
The canonical dataset contains the seats contested in each cycle, but this
runner does not yet contain each cycle's Democratic holdover-seat count.
Accordingly, seat outputs are explicitly labeled as "seats up" results.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]

# Allow this runner to be executed directly by file path while still
# importing the repository's top-level senate_model package.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from senate_model.engine import ModelConfig, run_forecast

DEFAULT_CANONICAL = (
    ROOT
    / "historical/senate/backtests/outputs/canonical/"
    "senate_canonical_backtest_dataset.csv"
)

DEFAULT_CALIBRATION = ROOT / "inputs/calibration_parameters.csv"

DEFAULT_OUTPUT_DIR = (
    ROOT
    / "historical/senate/backtests/outputs/production_replay_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the validated Senate national-environment model through "
            "the current production simulation engine."
        )
    )
    parser.add_argument(
        "--canonical",
        type=Path,
        default=DEFAULT_CANONICAL,
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--sims",
        type=int,
        default=50000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260719,
    )
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    return normalized.isin(
        {"true", "1", "yes", "y", "t"}
    )


def logistic_probability(
    margin: pd.Series | np.ndarray,
    scale: float,
) -> np.ndarray:
    values = np.asarray(margin, dtype=float)
    return 1.0 / (1.0 + np.exp(-values / scale))


def assign_race_tier(
    predicted_margin: pd.Series,
) -> pd.Series:
    """
    Create stable diagnostic tiers from the historical central margin.

    These labels are used only to assign uncertainty multipliers for this
    first replay. They do not alter the historical central prediction.
    """
    absolute_margin = predicted_margin.abs()

    return pd.Series(
        np.select(
            [
                absolute_margin <= 3.0,
                absolute_margin <= 7.0,
                absolute_margin <= 12.0,
            ],
            [
                "Toss-up",
                "Lean",
                "Likely",
            ],
            default="Safe",
        ),
        index=predicted_margin.index,
    )


def tier_multiplier(
    tier: pd.Series,
) -> pd.Series:
    """
    Conservative Version 1 uncertainty multipliers.

    These are intentionally simple and should later be replaced or validated
    against a historical tier-calibration study.
    """
    mapping = {
        "Toss-up": 1.10,
        "Lean": 1.05,
        "Likely": 1.00,
        "Safe": 0.95,
    }

    return tier.map(mapping).astype(float)


def election_date_for_cycle(
    frame: pd.DataFrame,
    cycle: int,
) -> str:
    if "election_date" in frame.columns:
        dates = pd.to_datetime(
            frame["election_date"],
            errors="coerce",
        ).dropna()

        if not dates.empty:
            return dates.iloc[0].date().isoformat()

    # Federal general-election fallback: first Tuesday after first Monday.
    november_first = pd.Timestamp(year=cycle, month=11, day=1)
    first_monday_offset = (7 - november_first.weekday()) % 7
    first_monday = november_first + pd.Timedelta(
        days=first_monday_offset
    )
    election_day = first_monday + pd.Timedelta(days=1)

    return election_day.date().isoformat()


def build_cycle_inputs(
    cycle_data: pd.DataFrame,
    calibration: pd.DataFrame,
    input_dir: Path,
) -> None:
    predicted = pd.to_numeric(
        cycle_data["production_predicted_margin_dem"],
        errors="raise",
    )

    baseline = pd.to_numeric(
        cycle_data["production_baseline_margin_dem"],
        errors="raise",
    )

    environment = pd.to_numeric(
        cycle_data[
            "production_national_environment_margin_dem"
        ],
        errors="raise",
    )

    if environment.nunique(dropna=True) != 1:
        raise ValueError(
            "Expected one national-environment value per cycle; "
            f"found {environment.nunique(dropna=True)}."
        )

    tiers = assign_race_tier(predicted)

    race_inputs = pd.DataFrame(
        {
            "state": cycle_data["state"].astype(str),
            "dem_candidate": "Historical Democratic nominee",
            "gop_candidate": "Historical Republican nominee",
            "current_holder": (
                cycle_data["model_incumbent_party"]
                .fillna("Open")
                .astype(str)
            ),
            "race_tier": tiers,
            # These fallback fields satisfy the production schema.
            # bayesian_model_margin_dem below preserves the exact canonical
            # central prediction.
            "polling_margin_dem": predicted,
            "fundamentals_margin_dem": baseline,
            "elasticity": 1.0,
            "tier_error_multiplier": tier_multiplier(tiers),
            "dem_win_counts_for_seat_change": 1.0,
            "bayesian_model_margin_dem": predicted,
            "candidate_uncertainty_penalty": 0.0,
            "race_id": cycle_data["race_id"].astype(str),
            "cycle": cycle_data["cycle"].astype(int),
            "actual_margin_dem": pd.to_numeric(
                cycle_data["actual_margin_dem"],
                errors="raise",
            ),
        }
    )

    national_environment = pd.DataFrame(
        {
            "parameter": [
                "national_environment_margin_dem",
                "generic_ballot_margin_dem",
            ],
            "value": [
                float(environment.iloc[0]),
                float(
                    pd.to_numeric(
                        cycle_data["generic_ballot_margin_dem"],
                        errors="raise",
                    ).iloc[0]
                ),
            ],
        }
    )

    input_dir.mkdir(parents=True, exist_ok=True)

    race_inputs.to_csv(
        input_dir / "race_inputs.csv",
        index=False,
    )

    national_environment.to_csv(
        input_dir / "national_environment.csv",
        index=False,
    )

    calibration.to_csv(
        input_dir / "calibration_parameters.csv",
        index=False,
    )


def calculate_race_metrics(
    predictions: pd.DataFrame,
) -> dict[str, float | int]:
    actual_margin = predictions[
        "actual_margin_dem"
    ].to_numpy(dtype=float)

    predicted_margin = predictions[
        "model_margin_dem"
    ].to_numpy(dtype=float)

    probability = predictions[
        "simulated_dem_win_prob"
    ].to_numpy(dtype=float)

    actual_win = (
        actual_margin > 0
    ).astype(float)

    error = predicted_margin - actual_margin

    predicted_winner = predicted_margin > 0
    actual_winner = actual_margin > 0

    return {
        "races": int(len(predictions)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mean_error_dem": float(np.mean(error)),
        "winner_accuracy": float(
            np.mean(predicted_winner == actual_winner)
        ),
        "brier": float(
            np.mean((probability - actual_win) ** 2)
        ),
        "mean_predicted_dem_probability": float(
            np.mean(probability)
        ),
        "actual_dem_win_rate": float(
            np.mean(actual_win)
        ),
    }


def build_calibration_table(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    frame = predictions.copy()

    frame["actual_dem_win"] = (
        frame["actual_margin_dem"] > 0
    ).astype(int)

    edges = np.linspace(0.0, 1.0, 11)

    frame["probability_bucket"] = pd.cut(
        frame["simulated_dem_win_prob"],
        bins=edges,
        include_lowest=True,
        right=True,
    )

    calibration = (
        frame.groupby(
            "probability_bucket",
            observed=False,
        )
        .agg(
            races=("race_id", "size"),
            mean_predicted_probability=(
                "simulated_dem_win_prob",
                "mean",
            ),
            actual_dem_win_rate=(
                "actual_dem_win",
                "mean",
            ),
        )
        .reset_index()
    )

    calibration["probability_bucket"] = (
        calibration["probability_bucket"]
        .astype(str)
    )

    calibration["calibration_error"] = (
        calibration["mean_predicted_probability"]
        - calibration["actual_dem_win_rate"]
    )

    return calibration


def main() -> None:
    args = parse_args()

    require_file(args.canonical)
    require_file(args.calibration)

    canonical = pd.read_csv(
        args.canonical,
        low_memory=False,
    )

    calibration = pd.read_csv(
        args.calibration,
        low_memory=False,
    )

    required_columns = {
        "race_id",
        "cycle",
        "state",
        "actual_margin_dem",
        "backtest_scorable",
        "generic_ballot_margin_dem",
        "production_baseline_margin_dem",
        "production_national_environment_margin_dem",
        "production_predicted_margin_dem",
        "model_incumbent_party",
    }

    missing = required_columns - set(canonical.columns)

    if missing:
        raise ValueError(
            "Canonical dataset missing columns: "
            f"{sorted(missing)}"
        )

    scorable = as_bool(canonical["backtest_scorable"])

    model_data = canonical.loc[scorable].copy()

    model_data["cycle"] = pd.to_numeric(
        model_data["cycle"],
        errors="raise",
    ).astype(int)

    duplicate_keys = int(
        model_data[
            ["race_id", "cycle"]
        ].duplicated().sum()
    )

    if duplicate_keys:
        raise ValueError(
            f"Found {duplicate_keys} duplicate race-cycle keys."
        )

    output_dir = args.output_dir
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cycle_output_root = output_dir / "cycles"

    if cycle_output_root.exists():
        shutil.rmtree(cycle_output_root)

    cycle_output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_frames: list[pd.DataFrame] = []
    cycle_metric_rows: list[dict] = []
    seats_up_rows: list[dict] = []

    cycles = sorted(
        model_data["cycle"].unique()
    )

    for cycle in cycles:
        cycle_data = (
            model_data.loc[
                model_data["cycle"] == cycle
            ]
            .copy()
            .sort_values(
                ["state", "race_id"]
            )
            .reset_index(drop=True)
        )

        election_date = election_date_for_cycle(
            cycle_data,
            int(cycle),
        )

        cycle_dir = (
            cycle_output_root / str(int(cycle))
        )

        input_dir = cycle_dir / "inputs"
        engine_output_dir = cycle_dir / "engine_outputs"

        build_cycle_inputs(
            cycle_data=cycle_data,
            calibration=calibration,
            input_dir=input_dir,
        )

        # A majority of seats contested in the cycle—not chamber control.
        seats_up_majority_threshold = (
            math.floor(len(cycle_data) / 2) + 1
        )

        config = ModelConfig(
            election_date=election_date,
            today=election_date,
            n_sims=args.sims,
            dem_baseline_seats=0,
            control_threshold=seats_up_majority_threshold,
            seed=args.seed + int(cycle),
        )

        result = run_forecast(
            input_dir=input_dir,
            output_dir=engine_output_dir,
            config=config,
        )

        race_stats = result["race_stats"].copy()

        input_payload = pd.read_csv(
            input_dir / "race_inputs.csv"
        )

        race_stats.insert(
            0,
            "race_id",
            input_payload["race_id"].to_numpy(),
        )

        race_stats.insert(
            1,
            "cycle",
            int(cycle),
        )

        race_stats["actual_margin_dem"] = (
            input_payload[
                "actual_margin_dem"
            ].to_numpy(dtype=float)
        )

        race_stats["prediction_error_dem"] = (
            race_stats["model_margin_dem"]
            - race_stats["actual_margin_dem"]
        )

        race_stats["actual_dem_win"] = (
            race_stats["actual_margin_dem"] > 0
        ).astype(int)

        race_stats["predicted_dem_win"] = (
            race_stats["model_margin_dem"] > 0
        ).astype(int)

        race_stats["winner_correct"] = (
            race_stats["actual_dem_win"]
            == race_stats["predicted_dem_win"]
        ).astype(int)

        prediction_frames.append(race_stats)

        cycle_metrics = calculate_race_metrics(
            race_stats
        )

        cycle_metric_rows.append(
            {
                "cycle": int(cycle),
                **cycle_metrics,
            }
        )

        simulation_draws = result[
            "simulation_draws"
        ].copy()

        actual_dem_wins = int(
            race_stats["actual_dem_win"].sum()
        )

        simulated_dem_wins = simulation_draws[
            "dem_seats"
        ].to_numpy(dtype=float)

        seats_up_rows.append(
            {
                "cycle": int(cycle),
                "races_up": int(len(cycle_data)),
                "actual_dem_wins_among_seats_up": (
                    actual_dem_wins
                ),
                "expected_dem_wins_among_seats_up": (
                    float(np.mean(simulated_dem_wins))
                ),
                "median_dem_wins_among_seats_up": (
                    float(np.median(simulated_dem_wins))
                ),
                "dem_wins_p25_among_seats_up": (
                    float(
                        np.percentile(
                            simulated_dem_wins,
                            25,
                        )
                    )
                ),
                "dem_wins_p75_among_seats_up": (
                    float(
                        np.percentile(
                            simulated_dem_wins,
                            75,
                        )
                    )
                ),
                "absolute_expected_wins_error": (
                    float(
                        abs(
                            np.mean(simulated_dem_wins)
                            - actual_dem_wins
                        )
                    )
                ),
                "seats_up_majority_threshold": (
                    seats_up_majority_threshold
                ),
                "probability_dem_won_majority_of_seats_up": (
                    float(
                        np.mean(
                            simulated_dem_wins
                            >= seats_up_majority_threshold
                        )
                    )
                ),
            }
        )

        print(
            f"{cycle}: "
            f"races={len(race_stats)}, "
            f"MAE={cycle_metrics['mae']:.3f}, "
            f"Brier={cycle_metrics['brier']:.4f}, "
            f"winner accuracy="
            f"{cycle_metrics['winner_accuracy']:.1%}"
        )

    all_predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    by_cycle = pd.DataFrame(
        cycle_metric_rows
    ).sort_values("cycle")

    seats_up_summary = pd.DataFrame(
        seats_up_rows
    ).sort_values("cycle")

    overall_metrics = calculate_race_metrics(
        all_predictions
    )

    overall = pd.DataFrame(
        [
            {
                "model_version": (
                    "production_replay_v1_"
                    "presidential_baseline_plus_"
                    "generic_0_90"
                ),
                **overall_metrics,
            }
        ]
    )

    probability_calibration = (
        build_calibration_table(
            all_predictions
        )
    )

    all_predictions.to_csv(
        output_dir / "race_predictions.csv",
        index=False,
    )

    by_cycle.to_csv(
        output_dir / "metrics_by_cycle.csv",
        index=False,
    )

    overall.to_csv(
        output_dir / "metrics_overall.csv",
        index=False,
    )

    probability_calibration.to_csv(
        output_dir / "probability_calibration.csv",
        index=False,
    )

    seats_up_summary.to_csv(
        output_dir / "seats_up_summary.csv",
        index=False,
    )

    config_payload = {
        "model_version": (
            "production_replay_v1_"
            "presidential_baseline_plus_generic_0_90"
        ),
        "canonical_input": str(args.canonical),
        "calibration_input": str(args.calibration),
        "output_dir": str(output_dir),
        "cycles": [int(cycle) for cycle in cycles],
        "scorable_races": int(len(model_data)),
        "simulations_per_cycle": int(args.sims),
        "seed": int(args.seed),
        "central_margin_source": (
            "production_predicted_margin_dem"
        ),
        "scope": (
            "Race-level and seats-up replay only. "
            "No historical full-chamber control probability."
        ),
        "limitations": [
            (
                "Historical candidate-quality effects are not yet "
                "replayed as a separate layer."
            ),
            (
                "Historical polling and Bayesian updating are not yet "
                "replayed."
            ),
            (
                "Tier uncertainty multipliers are provisional and "
                "must be separately validated."
            ),
            (
                "Full Senate control requires historical holdover-seat "
                "composition for every cycle."
            ),
        ],
    }

    (
        output_dir / "replay_config.json"
    ).write_text(
        json.dumps(
            config_payload,
            indent=2,
        )
        + "\n"
    )

    expected_races = 234

    validation_passed = (
        len(model_data) == expected_races
        and len(all_predictions) == expected_races
        and len(cycles) == 7
        and duplicate_keys == 0
        and all_predictions[
            [
                "model_margin_dem",
                "simulated_dem_win_prob",
                "actual_margin_dem",
            ]
        ].notna().all().all()
        and all_predictions[
            "simulated_dem_win_prob"
        ].between(0.0, 1.0).all()
    )

    validation_status = (
        "PASSED"
        if validation_passed
        else "FAILED"
    )

    report_lines = [
        "Senate Historical Production Replay — Version 1",
        "=" * 72,
        "",
        (
            "Central model: presidential baseline + "
            "0.90 × generic ballot"
        ),
        (
            "Scope: race-level and seats-up simulation; "
            "not full-chamber control"
        ),
        f"Cycles: {', '.join(map(str, cycles))}",
        f"Scorable races: {len(model_data)}",
        f"Simulations per cycle: {args.sims:,}",
        "",
        "Overall metrics",
        "-" * 72,
        f"MAE:             {overall_metrics['mae']:.4f}",
        f"RMSE:            {overall_metrics['rmse']:.4f}",
        (
            "Mean error Dem:  "
            f"{overall_metrics['mean_error_dem']:.4f}"
        ),
        (
            "Winner accuracy: "
            f"{overall_metrics['winner_accuracy']:.2%}"
        ),
        f"Brier score:     {overall_metrics['brier']:.5f}",
        "",
        f"Validation: {validation_status}",
        "",
        "Important:",
        (
            "The probability reported for each cycle is the probability "
            "that Democrats won a majority of the seats contested in that "
            "cycle. It is not Senate-control probability."
        ),
    ]

    report_text = "\n".join(report_lines) + "\n"

    (
        output_dir / "validation_report.txt"
    ).write_text(report_text)

    print()
    print(report_text)

    print("Outputs:")
    for filename in [
        "metrics_overall.csv",
        "metrics_by_cycle.csv",
        "probability_calibration.csv",
        "seats_up_summary.csv",
        "race_predictions.csv",
        "replay_config.json",
        "validation_report.txt",
    ]:
        print(f"  - {output_dir / filename}")

    if not validation_passed:
        raise SystemExit(
            "Replay validation FAILED."
        )


if __name__ == "__main__":
    main()
