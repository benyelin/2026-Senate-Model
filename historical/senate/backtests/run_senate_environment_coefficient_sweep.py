from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from historical.senate.backtests import (
        run_senate_production_environment_bakeoff as bakeoff,
    )
except ImportError:
    # Supports direct execution from the repository root:
    # python3 historical/senate/backtests/run_senate_environment_coefficient_sweep.py
    import run_senate_production_environment_bakeoff as bakeoff


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_INPUT_PATH = (
    PROJECT_ROOT
    / "historical/senate/warehouse/processed"
    / "senate_historical_fundamentals_2012_2024.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "historical/senate/backtests/outputs"
    / "environment_coefficient_sweep"
)

GENERIC_BALLOT_COEFFICIENTS = tuple(
    round(value, 2)
    for value in np.arange(0.00, 1.20 + 0.001, 0.05)
)

APPROVAL_COEFFICIENTS = tuple(
    round(value, 2)
    for value in np.arange(0.00, 1.00 + 0.001, 0.20)
)

MIDTERM_COEFFICIENTS = tuple(
    round(value, 2)
    for value in np.arange(-1.00, 1.00 + 0.001, 0.20)
)

MODEL_FAMILIES = {
    "generic_only": {
        "label": "Generic ballot only",
        "approval_coefficients": (0.0,),
        "midterm_coefficients": (0.0,),
        "component_count": 1,
    },
    "generic_plus_approval": {
        "label": "Generic ballot + approval",
        "approval_coefficients": APPROVAL_COEFFICIENTS,
        "midterm_coefficients": (0.0,),
        "component_count": 2,
    },
    "generic_plus_approval_plus_midterm": {
        "label": "Generic ballot + approval + midterm",
        "approval_coefficients": APPROVAL_COEFFICIENTS,
        "midterm_coefficients": MIDTERM_COEFFICIENTS,
        "component_count": 3,
    },
}


def expected_specification_count() -> int:
    return sum(
        len(GENERIC_BALLOT_COEFFICIENTS)
        * len(family["approval_coefficients"])
        * len(family["midterm_coefficients"])
        for family in MODEL_FAMILIES.values()
    )

SUMMARY_FILENAME = "senate_environment_coefficient_sweep_summary.csv"
CYCLE_FILENAME = "senate_environment_coefficient_sweep_by_cycle.csv"
PREDICTIONS_FILENAME = "senate_environment_coefficient_sweep_predictions.csv"
PARETO_FILENAME = "senate_environment_coefficient_sweep_pareto.csv"
RECOMMENDATION_FILENAME = "senate_environment_coefficient_sweep_recommendation.json"
VALIDATION_FILENAME = "senate_environment_coefficient_sweep_validation.txt"
CONFIG_FILENAME = "senate_environment_coefficient_sweep_config.json"


@dataclass(frozen=True)
class SweepPaths:
    output_dir: Path
    summary: Path
    by_cycle: Path
    predictions: Path
    pareto: Path
    recommendation: Path
    validation: Path
    config: Path


def build_paths(output_dir: Path) -> SweepPaths:
    return SweepPaths(
        output_dir=output_dir,
        summary=output_dir / SUMMARY_FILENAME,
        by_cycle=output_dir / CYCLE_FILENAME,
        predictions=output_dir / PREDICTIONS_FILENAME,
        pareto=output_dir / PARETO_FILENAME,
        recommendation=output_dir / RECOMMENDATION_FILENAME,
        validation=output_dir / VALIDATION_FILENAME,
        config=output_dir / CONFIG_FILENAME,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate nested Senate national-environment model families across "
            "independently calibrated generic-ballot, approval, and midterm grids."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Historical Senate fundamentals warehouse CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for sweep outputs.",
    )
    parser.add_argument(
        "--probability-scale",
        type=float,
        default=5.5,
        help="Margin-to-probability logistic scale passed to bake-off metrics.",
    )
    parser.add_argument(
        "--simplicity-penalty",
        type=float,
        default=0.03,
        help=(
            "Penalty added per extra model component when making the "
            "simplicity-aware recommendation."
        ),
    )
    parser.add_argument(
        "--near-best-tolerance",
        type=float,
        default=0.01,
        help=(
            "Maximum relative increase from the best composite performance "
            "score eligible for a simpler recommendation."
        ),
    )
    return parser.parse_args()


def require_positive_finite(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite; received {value!r}.")


def require_nonnegative_finite(value: float, name: str) -> None:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(
            f"{name} must be nonnegative and finite; received {value!r}."
        )


def load_model_data(input_path: Path) -> tuple[pd.DataFrame, dict[str, str], str]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input warehouse not found: {input_path}")

    raw = pd.read_csv(input_path)

    if raw.empty:
        raise ValueError(f"Input warehouse is empty: {input_path}")

    column_map = {
        logical_name: bakeoff.find_column(
            raw,
            logical_name,
            required=(logical_name != "scorable"),
        )
        for logical_name in (
            "race_id",
            "cycle",
            "state",
            "actual_margin_dem",
            "baseline_margin_dem",
            "generic_ballot_margin_dem",
            "presidential_approval",
            "president_party",
            "midterm_adjustment_dem",
            "scorable",
        )
    }

    selected = {
        logical_name: source_column
        for logical_name, source_column in column_map.items()
        if source_column is not None and logical_name != "scorable"
    }

    data = pd.DataFrame(
        {
            logical_name: raw[source_column]
            for logical_name, source_column in selected.items()
        }
    )

    scorable_column = column_map["scorable"]
    if scorable_column is None:
        data["scorable"] = True
    else:
        data["scorable"] = bakeoff.normalize_boolean(raw[scorable_column])

    numeric_columns = (
        "cycle",
        "actual_margin_dem",
        "baseline_margin_dem",
        "generic_ballot_margin_dem",
        "presidential_approval",
        "midterm_adjustment_dem",
    )
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    required_columns = (
        "race_id",
        "cycle",
        "state",
        "actual_margin_dem",
        "baseline_margin_dem",
        "generic_ballot_margin_dem",
        "presidential_approval",
        "president_party",
        "midterm_adjustment_dem",
    )

    model_data = (
        data.loc[data["scorable"]]
        .dropna(subset=required_columns)
        .copy()
    )

    if model_data.empty:
        raise ValueError("No complete scorable observations remain after validation.")

    model_data["cycle"] = model_data["cycle"].astype(int)
    model_data["president_party"] = model_data["president_party"].map(
        bakeoff.normalize_party
    )

    invalid_parties = ~model_data["president_party"].isin(["D", "R"])
    if invalid_parties.any():
        examples = (
            model_data.loc[invalid_parties, "president_party"]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        raise ValueError(f"Unsupported president-party values: {examples}")

    # The warehouse now owns the validated midterm indicator.  The sweep does
    # not re-derive or overwrite it.
    midterm_source = str(column_map["midterm_adjustment_dem"])

    adjustment_function, adjustment_source = (
        bakeoff.load_production_approval_function()
    )
    model_data["approval_adjustment_dem"] = [
        bakeoff.dem_approval_adjustment(
            approval=approval,
            president_party=party,
            adjustment_function=adjustment_function,
        )
        for approval, party in zip(
            model_data["presidential_approval"],
            model_data["president_party"],
        )
    ]

    bakeoff.validate_finite(
        model_data,
        [
            "actual_margin_dem",
            "baseline_margin_dem",
            "generic_ballot_margin_dem",
            "approval_adjustment_dem",
            "midterm_adjustment_dem",
        ],
    )

    duplicate_mask = model_data.duplicated(
        subset=["race_id", "cycle"],
        keep=False,
    )
    if duplicate_mask.any():
        duplicates = (
            model_data.loc[duplicate_mask, ["race_id", "cycle"]]
            .sort_values(["cycle", "race_id"])
            .to_dict("records")
        )
        raise ValueError(
            "Duplicate race-cycle observations found: "
            + json.dumps(duplicates[:20])
        )

    model_data = model_data.sort_values(
        ["cycle", "state", "race_id"]
    ).reset_index(drop=True)

    cycles = sorted(model_data["cycle"].unique().tolist())
    if len(cycles) < 3:
        raise ValueError(
            f"At least three cycles are required; found {len(cycles)}: {cycles}"
        )

    return model_data, {
        key: str(value) for key, value in column_map.items() if value is not None
    }, adjustment_source


def build_predictions(model_data: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    base_columns = [
        "race_id",
        "cycle",
        "state",
        "actual_margin_dem",
        "baseline_margin_dem",
        "generic_ballot_margin_dem",
        "approval_adjustment_dem",
        "midterm_adjustment_dem",
    ]

    for family_name, family in MODEL_FAMILIES.items():
        for generic_coefficient in GENERIC_BALLOT_COEFFICIENTS:
            for approval_coefficient in family["approval_coefficients"]:
                for midterm_coefficient in family["midterm_coefficients"]:
                    frame = model_data[base_columns].copy()

                    frame["model_family"] = family_name
                    frame["model_family_label"] = family["label"]
                    frame["generic_ballot_coefficient"] = float(
                        generic_coefficient
                    )
                    frame["approval_coefficient"] = float(
                        approval_coefficient
                    )
                    frame["midterm_coefficient"] = float(
                        midterm_coefficient
                    )
                    frame["component_count"] = int(
                        family["component_count"]
                    )

                    frame["generic_ballot_contribution_dem"] = (
                        float(generic_coefficient)
                        * frame["generic_ballot_margin_dem"]
                    )
                    frame["approval_contribution_dem"] = (
                        float(approval_coefficient)
                        * frame["approval_adjustment_dem"]
                    )
                    frame["midterm_contribution_dem"] = (
                        float(midterm_coefficient)
                        * frame["midterm_adjustment_dem"]
                    )
                    frame["environment_adjustment_dem"] = (
                        frame["generic_ballot_contribution_dem"]
                        + frame["approval_contribution_dem"]
                        + frame["midterm_contribution_dem"]
                    )
                    frame["predicted_margin_dem"] = (
                        frame["baseline_margin_dem"]
                        + frame["environment_adjustment_dem"]
                    )
                    frame["error_dem"] = (
                        frame["predicted_margin_dem"]
                        - frame["actual_margin_dem"]
                    )
                    frame["absolute_error"] = frame["error_dem"].abs()
                    frame["squared_error"] = frame["error_dem"] ** 2
                    frame["actual_dem_win"] = (
                        frame["actual_margin_dem"] > 0.0
                    ).astype(int)
                    frame["predicted_dem_win"] = (
                        frame["predicted_margin_dem"] > 0.0
                    ).astype(int)

                    gb_id = f"{generic_coefficient:+.2f}".replace(
                        "+", "p"
                    ).replace("-", "m").replace(".", "_")
                    approval_id = f"{approval_coefficient:+.2f}".replace(
                        "+", "p"
                    ).replace("-", "m").replace(".", "_")
                    midterm_id = f"{midterm_coefficient:+.2f}".replace(
                        "+", "p"
                    ).replace("-", "m").replace(".", "_")

                    frame["model_id"] = (
                        f"{family_name}"
                        f"__gb_{gb_id}"
                        f"__approval_{approval_id}"
                        f"__midterm_{midterm_id}"
                    )

                    bakeoff.validate_finite(
                        frame,
                        [
                            "predicted_margin_dem",
                            "error_dem",
                            "absolute_error",
                            "squared_error",
                        ],
                    )
                    frames.append(frame)

    predictions = pd.concat(frames, ignore_index=True)

    expected_models = expected_specification_count()
    observed_models = predictions["model_id"].nunique()
    if observed_models != expected_models:
        raise AssertionError(
            f"Expected {expected_models} model specifications; "
            f"found {observed_models}."
        )

    return predictions

def metric_row(
    group: pd.DataFrame,
    probability_scale: float,
) -> dict[str, float | int]:
    metrics = bakeoff.calculate_metrics(group, probability_scale)
    metrics["mean_error"] = float(group["error_dem"].mean())
    metrics["absolute_mean_error"] = abs(float(metrics["mean_error"]))
    return metrics


def summarize_predictions(
    predictions: pd.DataFrame,
    probability_scale: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    cycle_rows: list[dict[str, object]] = []

    model_columns = [
        "model_id",
        "model_family",
        "model_family_label",
        "generic_ballot_coefficient",
        "approval_coefficient",
        "midterm_coefficient",
        "component_count",
    ]

    for model_id, group in predictions.groupby("model_id", sort=True):
        metadata = group.iloc[0][model_columns].to_dict()
        summary_rows.append(
            {
                **metadata,
                "n_observations": int(len(group)),
                "n_cycles": int(group["cycle"].nunique()),
                **metric_row(group, probability_scale),
            }
        )

        for cycle, cycle_group in group.groupby("cycle", sort=True):
            cycle_rows.append(
                {
                    **metadata,
                    "cycle": int(cycle),
                    "n_observations": int(len(cycle_group)),
                    **metric_row(cycle_group, probability_scale),
                }
            )

    summary = pd.DataFrame(summary_rows)
    by_cycle = pd.DataFrame(cycle_rows)

    metric_directions = {
        "mae": True,
        "rmse": True,
        "brier": True,
        "winner_accuracy": False,
        "absolute_mean_error": True,
    }
    for metric, ascending in metric_directions.items():
        summary[f"{metric}_rank"] = summary[metric].rank(
            method="min",
            ascending=ascending,
        )

    summary["mean_primary_rank"] = summary[
        [
            "mae_rank",
            "rmse_rank",
            "brier_rank",
            "winner_accuracy_rank",
            "absolute_mean_error_rank",
        ]
    ].mean(axis=1)

    summary = summary.sort_values(
        [
            "mean_primary_rank",
            "mae",
            "rmse",
            "brier",
            "component_count",
            "generic_ballot_coefficient",
        ],
        ascending=[True, True, True, True, True, True],
    ).reset_index(drop=True)
    summary["overall_rank"] = np.arange(1, len(summary) + 1)

    by_cycle = by_cycle.sort_values(
        [
            "cycle",
            "model_family",
            "generic_ballot_coefficient",
            "approval_coefficient",
            "midterm_coefficient",
        ]
    ).reset_index(drop=True)

    return summary, by_cycle


def dominates(left: pd.Series, right: pd.Series, tolerance: float = 1e-12) -> bool:
    minimize = ("mae", "rmse", "brier", "absolute_mean_error")
    maximize = ("winner_accuracy",)

    no_worse = all(
        float(left[column]) <= float(right[column]) + tolerance
        for column in minimize
    ) and all(
        float(left[column]) + tolerance >= float(right[column])
        for column in maximize
    )

    strictly_better = any(
        float(left[column]) < float(right[column]) - tolerance
        for column in minimize
    ) or any(
        float(left[column]) > float(right[column]) + tolerance
        for column in maximize
    )

    return no_worse and strictly_better


def identify_pareto_models(summary: pd.DataFrame) -> pd.DataFrame:
    pareto_flags: list[bool] = []

    for index, candidate in summary.iterrows():
        is_dominated = any(
            dominates(other, candidate)
            for other_index, other in summary.iterrows()
            if other_index != index
        )
        pareto_flags.append(not is_dominated)

    marked = summary.copy()
    marked["pareto_optimal"] = pareto_flags

    pareto = marked.loc[marked["pareto_optimal"]].copy()
    pareto = pareto.sort_values(
        [
            "component_count",
            "mean_primary_rank",
            "mae",
            "generic_ballot_coefficient",
        ]
    ).reset_index(drop=True)
    pareto["pareto_order"] = np.arange(1, len(pareto) + 1)

    return marked, pareto


def make_recommendation(
    summary: pd.DataFrame,
    simplicity_penalty: float,
    near_best_tolerance: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    scored = summary.copy()

    rank_columns = [
        "mae",
        "rmse",
        "brier",
        "absolute_mean_error",
    ]
    for column in rank_columns:
        scored[f"{column}_percentile"] = scored[column].rank(
            method="average",
            pct=True,
            ascending=True,
        )

    scored["winner_accuracy_percentile"] = scored[
        "winner_accuracy"
    ].rank(
        method="average",
        pct=True,
        ascending=False,
    )

    scored["performance_score"] = scored[
        [
            "mae_percentile",
            "rmse_percentile",
            "brier_percentile",
            "absolute_mean_error_percentile",
            "winner_accuracy_percentile",
        ]
    ].mean(axis=1)

    best_performance_score = float(scored["performance_score"].min())
    eligibility_limit = best_performance_score * (1.0 + near_best_tolerance)

    scored["near_best_performance"] = (
        scored["performance_score"] <= eligibility_limit + 1e-12
    )
    scored["complexity_penalty"] = (
        simplicity_penalty
        * (scored["component_count"].astype(float) - 1.0)
    )
    scored["simplicity_aware_score"] = (
        scored["performance_score"]
        + scored["complexity_penalty"]
    )

    eligible = scored.loc[scored["near_best_performance"]].copy()
    recommendation_row = eligible.sort_values(
        [
            "simplicity_aware_score",
            "component_count",
            "performance_score",
            "mae",
            "rmse",
            "brier",
            "generic_ballot_coefficient",
        ]
    ).iloc[0]

    scored["recommended"] = (
        scored["model_id"] == recommendation_row["model_id"]
    )

    recommendation = {
        "model_id": str(recommendation_row["model_id"]),
        "model_family": str(recommendation_row["model_family"]),
        "model_family_label": str(recommendation_row["model_family_label"]),
        "generic_ballot_coefficient": float(
            recommendation_row["generic_ballot_coefficient"]
        ),
        "approval_coefficient": float(
            recommendation_row["approval_coefficient"]
        ),
        "midterm_coefficient": float(
            recommendation_row["midterm_coefficient"]
        ),
        "component_count": int(recommendation_row["component_count"]),
        "mae": float(recommendation_row["mae"]),
        "rmse": float(recommendation_row["rmse"]),
        "brier": float(recommendation_row["brier"]),
        "winner_accuracy": float(recommendation_row["winner_accuracy"]),
        "mean_error": float(recommendation_row["mean_error"]),
        "absolute_mean_error": float(
            recommendation_row["absolute_mean_error"]
        ),
        "performance_score": float(
            recommendation_row["performance_score"]
        ),
        "complexity_penalty": float(
            recommendation_row["complexity_penalty"]
        ),
        "simplicity_aware_score": float(
            recommendation_row["simplicity_aware_score"]
        ),
        "near_best_tolerance": near_best_tolerance,
        "simplicity_penalty_per_extra_component": simplicity_penalty,
        "selection_rule": (
            "Among models within the configured relative tolerance of the best "
            "five-metric percentile score, minimize performance score plus a "
            "fixed penalty for each component beyond generic ballot."
        ),
    }

    scored = scored.sort_values(
        [
            "recommended",
            "simplicity_aware_score",
            "performance_score",
            "component_count",
        ],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)

    return scored, recommendation


def validate_outputs(
    model_data: pd.DataFrame,
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
    by_cycle: pd.DataFrame,
    pareto: pd.DataFrame,
    recommendation: dict[str, object],
) -> list[str]:
    checks: list[tuple[str, bool, str]] = []

    expected_model_count = expected_specification_count()
    expected_prediction_count = expected_model_count * len(model_data)
    expected_cycle_rows = expected_model_count * model_data["cycle"].nunique()

    checks.extend(
        [
            (
                "model specification count",
                summary["model_id"].nunique() == expected_model_count,
                f"expected={expected_model_count}, observed="
                f"{summary['model_id'].nunique()}",
            ),
            (
                "prediction row count",
                len(predictions) == expected_prediction_count,
                f"expected={expected_prediction_count}, observed={len(predictions)}",
            ),
            (
                "cycle result row count",
                len(by_cycle) == expected_cycle_rows,
                f"expected={expected_cycle_rows}, observed={len(by_cycle)}",
            ),
            (
                "all generic-ballot values represented",
                set(summary["generic_ballot_coefficient"].round(2))
                == set(GENERIC_BALLOT_COEFFICIENTS),
                "generic-ballot coefficient grid mismatch",
            ),
            (
                "all approval values represented",
                set(summary["approval_coefficient"].round(2))
                == set(APPROVAL_COEFFICIENTS),
                "approval coefficient grid mismatch",
            ),
            (
                "all midterm values represented",
                set(summary["midterm_coefficient"].round(2))
                == set(MIDTERM_COEFFICIENTS),
                "midterm coefficient grid mismatch",
            ),
            (
                "all model families represented",
                set(summary["model_family"]) == set(MODEL_FAMILIES),
                "model-family mismatch",
            ),
            (
                "metrics finite",
                np.isfinite(
                    summary[
                        [
                            "mae",
                            "rmse",
                            "brier",
                            "winner_accuracy",
                            "mean_error",
                        ]
                    ].to_numpy(dtype=float)
                ).all(),
                "one or more non-finite summary metrics",
            ),
            (
                "winner accuracy bounded",
                summary["winner_accuracy"].between(0.0, 1.0).all(),
                "winner accuracy outside [0, 1]",
            ),
            (
                "Brier score bounded",
                summary["brier"].between(0.0, 1.0).all(),
                "Brier score outside [0, 1]",
            ),
            (
                "Pareto set nonempty",
                not pareto.empty,
                f"pareto_models={len(pareto)}",
            ),
            (
                "exactly one recommendation",
                int(summary["recommended"].sum()) == 1,
                f"recommended_count={int(summary['recommended'].sum())}",
            ),
            (
                "recommendation exists in summary",
                recommendation["model_id"] in set(summary["model_id"]),
                f"model_id={recommendation['model_id']}",
            ),
        ]
    )

    failed = [
        f"{name}: {detail}"
        for name, passed, detail in checks
        if not passed
    ]
    if failed:
        raise AssertionError(
            "Sweep validation failed:\n- " + "\n- ".join(failed)
        )

    return [
        f"PASS — {name}: {detail}"
        for name, passed, detail in checks
        if passed
    ]


def write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    args = parse_args()

    require_positive_finite(args.probability_scale, "probability scale")
    require_nonnegative_finite(args.simplicity_penalty, "simplicity penalty")
    require_nonnegative_finite(args.near_best_tolerance, "near-best tolerance")

    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    paths = build_paths(output_dir)
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    input_hash_before = bakeoff.sha256_file(input_path)

    model_data, column_map, approval_source = load_model_data(input_path)
    predictions = build_predictions(model_data)
    summary, by_cycle = summarize_predictions(
        predictions,
        args.probability_scale,
    )
    summary, pareto = identify_pareto_models(summary)
    summary, recommendation = make_recommendation(
        summary,
        args.simplicity_penalty,
        args.near_best_tolerance,
    )

    input_hash_after = bakeoff.sha256_file(input_path)
    if input_hash_before != input_hash_after:
        raise RuntimeError(
            "Input warehouse changed during execution; outputs were not written."
        )

    validation_lines = validate_outputs(
        model_data=model_data,
        predictions=predictions,
        summary=summary,
        by_cycle=by_cycle,
        pareto=pareto,
        recommendation=recommendation,
    )

    summary.to_csv(paths.summary, index=False)
    by_cycle.to_csv(paths.by_cycle, index=False)
    predictions.to_csv(paths.predictions, index=False)
    pareto.to_csv(paths.pareto, index=False)
    write_json(paths.recommendation, recommendation)

    config = {
        "script": str(Path(__file__).resolve()),
        "input_path": str(input_path),
        "input_sha256": input_hash_before,
        "output_dir": str(output_dir),
        "probability_scale": args.probability_scale,
        "generic_ballot_coefficients": list(
            GENERIC_BALLOT_COEFFICIENTS
        ),
        "approval_coefficients": list(APPROVAL_COEFFICIENTS),
        "midterm_coefficients": list(MIDTERM_COEFFICIENTS),
        "model_families": {
            family_name: {
                "label": family["label"],
                "approval_coefficients": list(
                    family["approval_coefficients"]
                ),
                "midterm_coefficients": list(
                    family["midterm_coefficients"]
                ),
                "component_count": family["component_count"],
            }
            for family_name, family in MODEL_FAMILIES.items()
        },
        "specification_count": int(len(summary)),
        "observation_count": int(len(model_data)),
        "cycles": sorted(model_data["cycle"].unique().astype(int).tolist()),
        "column_map": column_map,
        "approval_adjustment_source": approval_source,
        "midterm_source": column_map["midterm_adjustment_dem"],
        "simplicity_penalty": args.simplicity_penalty,
        "near_best_tolerance": args.near_best_tolerance,
        "reusable_helper_module": (
            "historical.senate.backtests."
            "run_senate_production_environment_bakeoff"
        ),
    }
    write_json(paths.config, config)

    validation_report = [
        "Senate Environment Coefficient Sweep Validation",
        "=" * 47,
        "",
        f"Input: {input_path}",
        f"Input SHA-256: {input_hash_before}",
        f"Scorable observations: {len(model_data)}",
        f"Cycles: {', '.join(map(str, config['cycles']))}",
        f"Specifications: {len(summary)}",
        f"Pareto-optimal specifications: {len(pareto)}",
        f"Approval adjustment source: {approval_source}",
        f"Midterm source: {column_map['midterm_adjustment_dem']}",
        "",
        *validation_lines,
        "",
        "VALIDATION STATUS: PASSED",
        "",
    ]
    paths.validation.write_text(
        "\n".join(validation_report),
        encoding="utf-8",
    )

    recommended = recommendation

    print("Senate Environment Coefficient Sweep")
    print("=" * 40)
    print(f"Input observations: {len(model_data)}")
    print(
        "Cycles: "
        + ", ".join(str(cycle) for cycle in config["cycles"])
    )
    print(f"Specifications evaluated: {len(summary)}")
    print(f"Pareto-optimal specifications: {len(pareto)}")
    print("")
    print("Simplicity-aware recommendation")
    print("-" * 31)
    print(f"Model: {recommended['model_family_label']}")
    print(
        "Coefficients: "
        f"GB={recommended['generic_ballot_coefficient']:.2f}, "
        f"approval={recommended['approval_coefficient']:.2f}, "
        f"midterm={recommended['midterm_coefficient']:.2f}"
    )
    print(
        "Metrics: "
        f"MAE={recommended['mae']:.6f}, "
        f"RMSE={recommended['rmse']:.6f}, "
        f"Brier={recommended['brier']:.6f}, "
        f"winner accuracy={recommended['winner_accuracy']:.4%}, "
        f"mean error={recommended['mean_error']:.6f}"
    )
    print("")
    print("Validation status: PASSED")
    print("")
    print(f"Wrote: {paths.summary}")
    print(f"Wrote: {paths.by_cycle}")
    print(f"Wrote: {paths.predictions}")
    print(f"Wrote: {paths.pareto}")
    print(f"Wrote: {paths.recommendation}")
    print(f"Wrote: {paths.config}")
    print(f"Wrote: {paths.validation}")


if __name__ == "__main__":
    main()
