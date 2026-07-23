from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from historical.senate.backtests import (
        run_senate_environment_coefficient_sweep as sweep,
    )
except ImportError:
    import run_senate_environment_coefficient_sweep as sweep


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_INPUT_PATH = (
    PROJECT_ROOT
    / "historical/senate/warehouse/processed"
    / "senate_historical_fundamentals_2012_2024.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "historical/senate/backtests/outputs"
    / "environment_leave_one_cycle_out"
)

FOLD_RESULTS_FILENAME = (
    "senate_environment_leave_one_cycle_out_folds.csv"
)
COMPARISON_FILENAME = (
    "senate_environment_leave_one_cycle_out_comparisons.csv"
)
SELECTION_STABILITY_FILENAME = (
    "senate_environment_leave_one_cycle_out_selection_stability.csv"
)
POOLED_PREDICTIONS_FILENAME = (
    "senate_environment_leave_one_cycle_out_predictions.csv"
)
SUMMARY_FILENAME = (
    "senate_environment_leave_one_cycle_out_summary.json"
)
VALIDATION_FILENAME = (
    "senate_environment_leave_one_cycle_out_validation.txt"
)
CONFIG_FILENAME = (
    "senate_environment_leave_one_cycle_out_config.json"
)


BENCHMARKS = {
    "fixed_generic_0_90": {
        "label": "Fixed generic ballot only, GB=0.90",
        "model_family": "generic_only",
        "generic_ballot_coefficient": 0.90,
        "approval_coefficient": 0.00,
        "midterm_coefficient": 0.00,
    },
    "legacy_full_environment": {
        "label": "Legacy environment, GB=1.00 / approval=0.50 / midterm=0.50",
        "model_family": "generic_plus_approval_plus_midterm",
        "generic_ballot_coefficient": 1.00,
        "approval_coefficient": 0.50,
        "midterm_coefficient": 0.50,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run leave-one-cycle-out validation for the Senate "
            "national-environment coefficient sweep."
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
        help="Directory for LOCO validation outputs.",
    )
    parser.add_argument(
        "--probability-scale",
        type=float,
        default=5.5,
        help="Margin-to-probability logistic scale.",
    )
    parser.add_argument(
        "--simplicity-penalty",
        type=float,
        default=0.03,
        help="Penalty per component beyond generic ballot.",
    )
    parser.add_argument(
        "--near-best-tolerance",
        type=float,
        default=0.01,
        help="Relative tolerance used for the simpler recommendation.",
    )
    return parser.parse_args()


def select_specification(
    predictions: pd.DataFrame,
    specification: dict[str, object],
) -> pd.DataFrame:
    mask = (
        predictions["model_family"].eq(
            specification["model_family"]
        )
        & np.isclose(
            predictions["generic_ballot_coefficient"],
            float(specification["generic_ballot_coefficient"]),
        )
        & np.isclose(
            predictions["approval_coefficient"],
            float(specification["approval_coefficient"]),
        )
        & np.isclose(
            predictions["midterm_coefficient"],
            float(specification["midterm_coefficient"]),
        )
    )

    selected = predictions.loc[mask].copy()

    if selected.empty:
        raise ValueError(
            "Could not find requested specification: "
            + json.dumps(specification, sort_keys=True)
        )

    model_count = selected["model_id"].nunique()
    if model_count != 1:
        raise AssertionError(
            "Expected exactly one model ID for specification; "
            f"found {model_count}."
        )

    return selected


def build_custom_predictions(
    model_data: pd.DataFrame,
    specification: dict[str, object],
) -> pd.DataFrame:
    """Build predictions for coefficients that need not lie on the sweep grid."""
    frame = model_data[
        [
            "race_id",
            "cycle",
            "state",
            "actual_margin_dem",
            "baseline_margin_dem",
            "generic_ballot_margin_dem",
            "approval_adjustment_dem",
            "midterm_adjustment_dem",
        ]
    ].copy()

    generic_coefficient = float(
        specification["generic_ballot_coefficient"]
    )
    approval_coefficient = float(
        specification["approval_coefficient"]
    )
    midterm_coefficient = float(
        specification["midterm_coefficient"]
    )

    frame["model_family"] = str(
        specification["model_family"]
    )
    frame["model_family_label"] = str(
        specification.get(
            "label",
            specification["model_family"],
        )
    )
    frame["generic_ballot_coefficient"] = generic_coefficient
    frame["approval_coefficient"] = approval_coefficient
    frame["midterm_coefficient"] = midterm_coefficient

    frame["generic_ballot_contribution_dem"] = (
        generic_coefficient
        * frame["generic_ballot_margin_dem"]
    )
    frame["approval_contribution_dem"] = (
        approval_coefficient
        * frame["approval_adjustment_dem"]
    )
    frame["midterm_contribution_dem"] = (
        midterm_coefficient
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

    frame["model_id"] = (
        f"custom__{specification.get('model_family', 'benchmark')}"
        f"__gb_{generic_coefficient:.2f}"
        f"__approval_{approval_coefficient:.2f}"
        f"__midterm_{midterm_coefficient:.2f}"
    )

    sweep.bakeoff.validate_finite(
        frame,
        [
            "predicted_margin_dem",
            "error_dem",
            "absolute_error",
            "squared_error",
        ],
    )

    return frame



def metric_payload(
    predictions: pd.DataFrame,
    probability_scale: float,
) -> dict[str, float | int]:
    return sweep.metric_row(predictions, probability_scale)


def run_fold(
    model_data: pd.DataFrame,
    held_out_cycle: int,
    probability_scale: float,
    simplicity_penalty: float,
    near_best_tolerance: float,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    pd.DataFrame,
]:
    train_data = model_data.loc[
        model_data["cycle"] != held_out_cycle
    ].copy()
    test_data = model_data.loc[
        model_data["cycle"] == held_out_cycle
    ].copy()

    if train_data.empty or test_data.empty:
        raise ValueError(
            f"Invalid fold for held-out cycle {held_out_cycle}."
        )

    train_predictions = sweep.build_predictions(train_data)
    train_summary, _ = sweep.summarize_predictions(
        train_predictions,
        probability_scale,
    )

    scored_train_summary, recommendation = (
        sweep.make_recommendation(
            train_summary,
            simplicity_penalty=simplicity_penalty,
            near_best_tolerance=near_best_tolerance,
        )
    )

    recommended_rows = scored_train_summary.loc[
        scored_train_summary["recommended"]
    ]
    if len(recommended_rows) != 1:
        raise AssertionError(
            f"Fold {held_out_cycle}: expected one recommendation; "
            f"found {len(recommended_rows)}."
        )

    selected_train_row = recommended_rows.iloc[0]
    selected_model_id = str(selected_train_row["model_id"])

    # Generate every candidate on the held-out cycle, then freeze the
    # specification selected using training data only.
    test_predictions = sweep.build_predictions(test_data)
    selected_test = test_predictions.loc[
        test_predictions["model_id"] == selected_model_id
    ].copy()

    if len(selected_test) != len(test_data):
        raise AssertionError(
            f"Fold {held_out_cycle}: selected held-out row count "
            f"{len(selected_test)} does not match expected "
            f"{len(test_data)}."
        )

    selected_metrics = metric_payload(
        selected_test,
        probability_scale,
    )

    fold_row = {
        "held_out_cycle": int(held_out_cycle),
        "train_cycles": ",".join(
            str(value)
            for value in sorted(train_data["cycle"].unique())
        ),
        "train_observations": int(len(train_data)),
        "held_out_observations": int(len(test_data)),
        "selected_model_id": selected_model_id,
        "selected_model_family": str(
            selected_train_row["model_family"]
        ),
        "selected_model_family_label": str(
            selected_train_row["model_family_label"]
        ),
        "selected_generic_ballot_coefficient": float(
            selected_train_row["generic_ballot_coefficient"]
        ),
        "selected_approval_coefficient": float(
            selected_train_row["approval_coefficient"]
        ),
        "selected_midterm_coefficient": float(
            selected_train_row["midterm_coefficient"]
        ),
        "selected_component_count": int(
            selected_train_row["component_count"]
        ),
        "train_mae": float(selected_train_row["mae"]),
        "train_rmse": float(selected_train_row["rmse"]),
        "train_brier": float(selected_train_row["brier"]),
        "train_winner_accuracy": float(
            selected_train_row["winner_accuracy"]
        ),
        "train_mean_error": float(
            selected_train_row["mean_error"]
        ),
        "held_out_mae": float(selected_metrics["mae"]),
        "held_out_rmse": float(selected_metrics["rmse"]),
        "held_out_brier": float(selected_metrics["brier"]),
        "held_out_winner_accuracy": float(
            selected_metrics["winner_accuracy"]
        ),
        "held_out_mean_error": float(
            selected_metrics["mean_error"]
        ),
        "held_out_absolute_mean_error": float(
            selected_metrics["absolute_mean_error"]
        ),
    }

    comparison_rows: list[dict[str, object]] = []

    selected_comparison = {
        "held_out_cycle": int(held_out_cycle),
        "comparison_id": "fold_selected",
        "comparison_label": "Training-fold selected model",
        "model_id": selected_model_id,
        "model_family": str(
            selected_train_row["model_family"]
        ),
        "generic_ballot_coefficient": float(
            selected_train_row["generic_ballot_coefficient"]
        ),
        "approval_coefficient": float(
            selected_train_row["approval_coefficient"]
        ),
        "midterm_coefficient": float(
            selected_train_row["midterm_coefficient"]
        ),
        "held_out_observations": int(len(selected_test)),
        **selected_metrics,
    }
    comparison_rows.append(selected_comparison)

    for benchmark_id, benchmark in BENCHMARKS.items():
        benchmark_predictions = build_custom_predictions(
            test_data,
            benchmark,
        )
        benchmark_metrics = metric_payload(
            benchmark_predictions,
            probability_scale,
        )

        comparison_rows.append(
            {
                "held_out_cycle": int(held_out_cycle),
                "comparison_id": benchmark_id,
                "comparison_label": benchmark["label"],
                "model_id": str(
                    benchmark_predictions["model_id"].iloc[0]
                ),
                "model_family": benchmark["model_family"],
                "generic_ballot_coefficient": float(
                    benchmark["generic_ballot_coefficient"]
                ),
                "approval_coefficient": float(
                    benchmark["approval_coefficient"]
                ),
                "midterm_coefficient": float(
                    benchmark["midterm_coefficient"]
                ),
                "held_out_observations": int(
                    len(benchmark_predictions)
                ),
                **benchmark_metrics,
            }
        )

    selected_test["held_out_cycle"] = int(held_out_cycle)
    selected_test["selection_source"] = (
        "training_cycles_only"
    )

    return fold_row, comparison_rows, selected_test


def build_selection_stability(
    folds: pd.DataFrame,
) -> pd.DataFrame:
    grouping_columns = [
        "selected_model_family",
        "selected_model_family_label",
        "selected_generic_ballot_coefficient",
        "selected_approval_coefficient",
        "selected_midterm_coefficient",
        "selected_component_count",
    ]

    stability = (
        folds.groupby(grouping_columns, dropna=False)
        .agg(
            folds_selected=("held_out_cycle", "count"),
            held_out_cycles=(
                "held_out_cycle",
                lambda values: ",".join(
                    str(int(value))
                    for value in sorted(values)
                ),
            ),
            mean_held_out_mae=("held_out_mae", "mean"),
            mean_held_out_rmse=("held_out_rmse", "mean"),
            mean_held_out_brier=("held_out_brier", "mean"),
            mean_held_out_winner_accuracy=(
                "held_out_winner_accuracy",
                "mean",
            ),
            mean_held_out_error=("held_out_mean_error", "mean"),
        )
        .reset_index()
    )

    stability["selection_share"] = (
        stability["folds_selected"] / len(folds)
    )

    return stability.sort_values(
        [
            "folds_selected",
            "mean_held_out_mae",
            "mean_held_out_rmse",
        ],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def pooled_metrics_by_comparison(
    comparisons: pd.DataFrame,
    pooled_selected_predictions: pd.DataFrame,
    probability_scale: float,
    model_data: pd.DataFrame,
) -> dict[str, dict[str, float | int]]:
    payload: dict[str, dict[str, float | int]] = {}

    payload["fold_selected"] = metric_payload(
        pooled_selected_predictions,
        probability_scale,
    )

    for benchmark_id, benchmark in BENCHMARKS.items():
        benchmark_predictions = build_custom_predictions(
            model_data,
            benchmark,
        )
        payload[benchmark_id] = metric_payload(
            benchmark_predictions,
            probability_scale,
        )

    return payload


def validate_results(
    model_data: pd.DataFrame,
    folds: pd.DataFrame,
    comparisons: pd.DataFrame,
    pooled_predictions: pd.DataFrame,
) -> list[str]:
    cycles = sorted(model_data["cycle"].unique())
    expected_cycle_count = len(cycles)
    expected_comparison_rows = expected_cycle_count * (
        1 + len(BENCHMARKS)
    )

    checks = [
        (
            "one fold per cycle",
            len(folds) == expected_cycle_count,
            f"expected={expected_cycle_count}, observed={len(folds)}",
        ),
        (
            "all cycles held out exactly once",
            sorted(folds["held_out_cycle"].tolist()) == cycles,
            "held-out cycle mismatch",
        ),
        (
            "comparison row count",
            len(comparisons) == expected_comparison_rows,
            (
                f"expected={expected_comparison_rows}, "
                f"observed={len(comparisons)}"
            ),
        ),
        (
            "all observations receive one out-of-sample prediction",
            len(pooled_predictions) == len(model_data),
            (
                f"expected={len(model_data)}, "
                f"observed={len(pooled_predictions)}"
            ),
        ),
        (
            "no duplicate pooled race-cycle predictions",
            not pooled_predictions.duplicated(
                ["race_id", "cycle"]
            ).any(),
            "duplicate race-cycle predictions found",
        ),
        (
            "selected coefficients finite",
            np.isfinite(
                folds[
                    [
                        "selected_generic_ballot_coefficient",
                        "selected_approval_coefficient",
                        "selected_midterm_coefficient",
                    ]
                ].to_numpy(dtype=float)
            ).all(),
            "non-finite selected coefficient",
        ),
        (
            "held-out metrics finite",
            np.isfinite(
                folds[
                    [
                        "held_out_mae",
                        "held_out_rmse",
                        "held_out_brier",
                        "held_out_winner_accuracy",
                        "held_out_mean_error",
                    ]
                ].to_numpy(dtype=float)
            ).all(),
            "non-finite held-out metric",
        ),
        (
            "held-out accuracy bounded",
            folds["held_out_winner_accuracy"].between(
                0.0, 1.0
            ).all(),
            "winner accuracy outside [0, 1]",
        ),
        (
            "held-out Brier bounded",
            folds["held_out_brier"].between(
                0.0, 1.0
            ).all(),
            "Brier score outside [0, 1]",
        ),
    ]

    lines = [
        "Senate Environment Leave-One-Cycle-Out Validation",
        "=================================================",
        "",
    ]

    passed = True
    for label, condition, detail in checks:
        status = "PASS" if condition else "FAIL"
        lines.append(f"{status} — {label}: {detail}")
        passed = passed and bool(condition)

    lines.extend(
        [
            "",
            (
                "VALIDATION STATUS: PASSED"
                if passed
                else "VALIDATION STATUS: FAILED"
            ),
        ]
    )

    if not passed:
        raise AssertionError("\n".join(lines))

    return lines


def main() -> None:
    args = parse_args()

    sweep.require_positive_finite(
        args.probability_scale,
        "probability_scale",
    )
    sweep.require_nonnegative_finite(
        args.simplicity_penalty,
        "simplicity_penalty",
    )
    sweep.require_nonnegative_finite(
        args.near_best_tolerance,
        "near_best_tolerance",
    )

    model_data, column_map, approval_source = (
        sweep.load_model_data(args.input)
    )
    cycles = sorted(model_data["cycle"].unique())

    fold_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    pooled_frames: list[pd.DataFrame] = []

    print("Senate Environment Leave-One-Cycle-Out")
    print("=" * 44)
    print(f"Input observations: {len(model_data)}")
    print(
        "Cycles: " + ", ".join(str(cycle) for cycle in cycles)
    )
    print(
        "Specifications evaluated per training fold: "
        f"{sweep.expected_specification_count()}"
    )
    print()

    for held_out_cycle in cycles:
        print(f"Running fold: hold out {held_out_cycle}")

        fold_row, fold_comparisons, fold_predictions = (
            run_fold(
                model_data=model_data,
                held_out_cycle=int(held_out_cycle),
                probability_scale=args.probability_scale,
                simplicity_penalty=args.simplicity_penalty,
                near_best_tolerance=args.near_best_tolerance,
            )
        )

        fold_rows.append(fold_row)
        comparison_rows.extend(fold_comparisons)
        pooled_frames.append(fold_predictions)

        print(
            "  Selected: "
            f"{fold_row['selected_model_family_label']} | "
            f"GB={fold_row['selected_generic_ballot_coefficient']:.2f}, "
            f"approval={fold_row['selected_approval_coefficient']:.2f}, "
            f"midterm={fold_row['selected_midterm_coefficient']:.2f}"
        )
        print(
            "  Held-out metrics: "
            f"MAE={fold_row['held_out_mae']:.6f}, "
            f"RMSE={fold_row['held_out_rmse']:.6f}, "
            f"Brier={fold_row['held_out_brier']:.6f}, "
            "accuracy="
            f"{fold_row['held_out_winner_accuracy']:.4%}"
        )

    folds = pd.DataFrame(fold_rows).sort_values(
        "held_out_cycle"
    ).reset_index(drop=True)

    comparisons = pd.DataFrame(comparison_rows).sort_values(
        ["held_out_cycle", "comparison_id"]
    ).reset_index(drop=True)

    pooled_predictions = pd.concat(
        pooled_frames,
        ignore_index=True,
    ).sort_values(
        ["cycle", "state", "race_id"]
    ).reset_index(drop=True)

    stability = build_selection_stability(folds)

    pooled_metrics = pooled_metrics_by_comparison(
        comparisons=comparisons,
        pooled_selected_predictions=pooled_predictions,
        probability_scale=args.probability_scale,
        model_data=model_data,
    )

    selected_approval_zero_share = float(
        np.isclose(
            folds["selected_approval_coefficient"],
            0.0,
        ).mean()
    )
    selected_midterm_zero_share = float(
        np.isclose(
            folds["selected_midterm_coefficient"],
            0.0,
        ).mean()
    )
    selected_generic_only_share = float(
        folds["selected_model_family"]
        .eq("generic_only")
        .mean()
    )

    summary = {
        "input": str(args.input.resolve()),
        "observations": int(len(model_data)),
        "cycles": [int(value) for value in cycles],
        "folds": int(len(folds)),
        "specifications_per_fold": int(
            sweep.expected_specification_count()
        ),
        "approval_adjustment_source": approval_source,
        "selection_stability": {
            "generic_only_share": selected_generic_only_share,
            "approval_zero_share": selected_approval_zero_share,
            "midterm_zero_share": selected_midterm_zero_share,
            "mean_selected_generic_ballot_coefficient": float(
                folds[
                    "selected_generic_ballot_coefficient"
                ].mean()
            ),
            "median_selected_generic_ballot_coefficient": float(
                folds[
                    "selected_generic_ballot_coefficient"
                ].median()
            ),
        },
        "pooled_out_of_sample_metrics": pooled_metrics,
        "interpretation_rule": {
            "strong_simplification_support": (
                "Generic-only is selected in most folds, approval and "
                "midterm are usually zero, and pooled selected-model "
                "performance is competitive with or superior to the "
                "legacy full-environment benchmark."
            )
        },
        "column_map": column_map,
    }

    validation_lines = validate_results(
        model_data=model_data,
        folds=folds,
        comparisons=comparisons,
        pooled_predictions=pooled_predictions,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    folds.to_csv(
        args.output_dir / FOLD_RESULTS_FILENAME,
        index=False,
    )
    comparisons.to_csv(
        args.output_dir / COMPARISON_FILENAME,
        index=False,
    )
    stability.to_csv(
        args.output_dir / SELECTION_STABILITY_FILENAME,
        index=False,
    )
    pooled_predictions.to_csv(
        args.output_dir / POOLED_PREDICTIONS_FILENAME,
        index=False,
    )

    with (
        args.output_dir / SUMMARY_FILENAME
    ).open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    config = {
        "input": str(args.input.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "probability_scale": args.probability_scale,
        "simplicity_penalty": args.simplicity_penalty,
        "near_best_tolerance": args.near_best_tolerance,
        "generic_ballot_coefficients": list(
            sweep.GENERIC_BALLOT_COEFFICIENTS
        ),
        "approval_coefficients": list(
            sweep.APPROVAL_COEFFICIENTS
        ),
        "midterm_coefficients": list(
            sweep.MIDTERM_COEFFICIENTS
        ),
        "benchmarks": BENCHMARKS,
    }

    with (
        args.output_dir / CONFIG_FILENAME
    ).open("w") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")

    (
        args.output_dir / VALIDATION_FILENAME
    ).write_text("\n".join(validation_lines) + "\n")

    print()
    print("Selection stability")
    print("-------------------")
    print(
        "Generic-only selected: "
        f"{selected_generic_only_share:.1%}"
    )
    print(
        "Approval coefficient zero: "
        f"{selected_approval_zero_share:.1%}"
    )
    print(
        "Midterm coefficient zero: "
        f"{selected_midterm_zero_share:.1%}"
    )
    print(
        "Mean selected GB coefficient: "
        f"{summary['selection_stability']['mean_selected_generic_ballot_coefficient']:.3f}"
    )

    selected_pooled = pooled_metrics["fold_selected"]
    legacy_pooled = pooled_metrics["legacy_full_environment"]
    fixed_pooled = pooled_metrics["fixed_generic_0_90"]

    print()
    print("Pooled out-of-sample performance")
    print("--------------------------------")
    print(
        "Fold-selected: "
        f"MAE={selected_pooled['mae']:.6f}, "
        f"RMSE={selected_pooled['rmse']:.6f}, "
        f"Brier={selected_pooled['brier']:.6f}, "
        f"accuracy={selected_pooled['winner_accuracy']:.4%}"
    )
    print(
        "Fixed GB=0.90: "
        f"MAE={fixed_pooled['mae']:.6f}, "
        f"RMSE={fixed_pooled['rmse']:.6f}, "
        f"Brier={fixed_pooled['brier']:.6f}, "
        f"accuracy={fixed_pooled['winner_accuracy']:.4%}"
    )
    print(
        "Legacy full environment: "
        f"MAE={legacy_pooled['mae']:.6f}, "
        f"RMSE={legacy_pooled['rmse']:.6f}, "
        f"Brier={legacy_pooled['brier']:.6f}, "
        f"accuracy={legacy_pooled['winner_accuracy']:.4%}"
    )

    print()
    print("Validation status: PASSED")
    print()
    print(f"Wrote outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
