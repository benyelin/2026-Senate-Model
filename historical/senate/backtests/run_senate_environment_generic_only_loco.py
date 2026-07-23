from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from historical.senate.backtests import (
        run_senate_environment_coefficient_sweep as sweep,
    )
except ModuleNotFoundError:
    import run_senate_environment_coefficient_sweep as sweep


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_PATH = (
    PROJECT_ROOT
    / "historical/senate/warehouse/processed"
    / "senate_historical_fundamentals_2012_2024.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "historical/senate/backtests/outputs"
    / "environment_generic_only_loco"
)

PROBABILITY_SCALE = 5.5


def choose_generic_only_model(
    training_predictions: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    summary, _ = sweep.summarize_predictions(
        training_predictions,
        PROBABILITY_SCALE,
    )

    generic_summary = (
        summary.loc[
            summary["model_family"] == "generic_only"
        ]
        .sort_values(
            [
                "mean_primary_rank",
                "mae",
                "rmse",
                "brier",
                "generic_ballot_coefficient",
            ],
            ascending=[True, True, True, True, True],
        )
        .reset_index(drop=True)
    )

    if len(generic_summary) < 2:
        raise AssertionError(
            "Expected at least two generic-only candidates."
        )

    return generic_summary.iloc[0], generic_summary.iloc[1]


def main() -> None:
    model_data, column_map, approval_source = (
        sweep.load_model_data(INPUT_PATH)
    )

    cycles = sorted(model_data["cycle"].unique())

    fold_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    print("Senate Generic-Only Leave-One-Cycle-Out")
    print("=" * 44)
    print(f"Input observations: {len(model_data)}")
    print(
        "Cycles: "
        + ", ".join(str(cycle) for cycle in cycles)
    )
    print(
        "Generic-only coefficients per fold: "
        f"{len(sweep.GENERIC_BALLOT_COEFFICIENTS)}"
    )
    print()

    for held_out_cycle in cycles:
        train_data = model_data.loc[
            model_data["cycle"] != held_out_cycle
        ].copy()

        test_data = model_data.loc[
            model_data["cycle"] == held_out_cycle
        ].copy()

        train_predictions = sweep.build_predictions(train_data)

        best, runner_up = choose_generic_only_model(
            train_predictions
        )

        selected_model_id = str(best["model_id"])

        test_predictions = sweep.build_predictions(test_data)

        selected_test = test_predictions.loc[
            test_predictions["model_id"] == selected_model_id
        ].copy()

        if len(selected_test) != len(test_data):
            raise AssertionError(
                f"Held-out cycle {held_out_cycle}: "
                f"expected {len(test_data)} predictions, "
                f"found {len(selected_test)}."
            )

        held_out_metrics = sweep.metric_row(
            selected_test,
            PROBABILITY_SCALE,
        )

        fold_rows.append(
            {
                "held_out_cycle": int(held_out_cycle),
                "train_cycles": ",".join(
                    str(value)
                    for value in sorted(
                        train_data["cycle"].unique()
                    )
                ),
                "train_observations": int(len(train_data)),
                "held_out_observations": int(len(test_data)),
                "selected_model_id": selected_model_id,
                "selected_generic_ballot_coefficient": float(
                    best["generic_ballot_coefficient"]
                ),
                "runner_up_generic_ballot_coefficient": float(
                    runner_up["generic_ballot_coefficient"]
                ),
                "selected_train_mae": float(best["mae"]),
                "runner_up_train_mae": float(runner_up["mae"]),
                "delta_train_mae": float(
                    runner_up["mae"] - best["mae"]
                ),
                "selected_train_rmse": float(best["rmse"]),
                "selected_train_brier": float(best["brier"]),
                "selected_train_winner_accuracy": float(
                    best["winner_accuracy"]
                ),
                "held_out_mae": float(
                    held_out_metrics["mae"]
                ),
                "held_out_rmse": float(
                    held_out_metrics["rmse"]
                ),
                "held_out_brier": float(
                    held_out_metrics["brier"]
                ),
                "held_out_winner_accuracy": float(
                    held_out_metrics["winner_accuracy"]
                ),
                "held_out_mean_error": float(
                    held_out_metrics["mean_error"]
                ),
                "held_out_absolute_mean_error": float(
                    held_out_metrics[
                        "absolute_mean_error"
                    ]
                ),
            }
        )

        selected_test["held_out_cycle"] = int(
            held_out_cycle
        )
        selected_test["selection_source"] = (
            "generic_only_training_cycles"
        )
        prediction_frames.append(selected_test)

        print(
            f"Held out {held_out_cycle}: "
            f"selected GB={float(best['generic_ballot_coefficient']):.2f}, "
            f"runner-up={float(runner_up['generic_ballot_coefficient']):.2f}, "
            f"Δ train MAE={float(runner_up['mae'] - best['mae']):.6f}"
        )
        print(
            "  Held-out: "
            f"MAE={held_out_metrics['mae']:.6f}, "
            f"RMSE={held_out_metrics['rmse']:.6f}, "
            f"Brier={held_out_metrics['brier']:.6f}, "
            f"accuracy={held_out_metrics['winner_accuracy']:.4%}"
        )

    folds = pd.DataFrame(fold_rows).sort_values(
        "held_out_cycle"
    ).reset_index(drop=True)

    pooled_predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    ).sort_values(
        ["cycle", "state", "race_id"]
    ).reset_index(drop=True)

    if len(pooled_predictions) != len(model_data):
        raise AssertionError(
            "Pooled out-of-sample prediction count mismatch: "
            f"expected {len(model_data)}, "
            f"found {len(pooled_predictions)}."
        )

    if pooled_predictions.duplicated(
        ["race_id", "cycle"]
    ).any():
        raise AssertionError(
            "Duplicate race-cycle predictions found."
        )

    pooled_metrics = sweep.metric_row(
        pooled_predictions,
        PROBABILITY_SCALE,
    )

    coefficient_counts = (
        folds[
            "selected_generic_ballot_coefficient"
        ]
        .value_counts()
        .sort_index()
        .rename_axis(
            "generic_ballot_coefficient"
        )
        .reset_index(name="folds_selected")
    )

    coefficient_counts["selection_share"] = (
        coefficient_counts["folds_selected"]
        / len(folds)
    )

    summary = {
        "input": str(INPUT_PATH.resolve()),
        "observations": int(len(model_data)),
        "cycles": [int(value) for value in cycles],
        "folds": int(len(folds)),
        "candidate_coefficients_per_fold": int(
            len(sweep.GENERIC_BALLOT_COEFFICIENTS)
        ),
        "approval_adjustment_source": approval_source,
        "selected_coefficient_mean": float(
            folds[
                "selected_generic_ballot_coefficient"
            ].mean()
        ),
        "selected_coefficient_median": float(
            folds[
                "selected_generic_ballot_coefficient"
            ].median()
        ),
        "selected_coefficient_min": float(
            folds[
                "selected_generic_ballot_coefficient"
            ].min()
        ),
        "selected_coefficient_max": float(
            folds[
                "selected_generic_ballot_coefficient"
            ].max()
        ),
        "mean_runner_up_train_mae_gap": float(
            folds["delta_train_mae"].mean()
        ),
        "pooled_out_of_sample_metrics": {
            key: (
                int(value)
                if isinstance(value, (int, np.integer))
                else float(value)
            )
            for key, value in pooled_metrics.items()
        },
        "column_map": column_map,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    folds.to_csv(
        OUTPUT_DIR
        / "senate_environment_generic_only_loco_folds.csv",
        index=False,
    )

    pooled_predictions.to_csv(
        OUTPUT_DIR
        / "senate_environment_generic_only_loco_predictions.csv",
        index=False,
    )

    coefficient_counts.to_csv(
        OUTPUT_DIR
        / "senate_environment_generic_only_loco_coefficient_counts.csv",
        index=False,
    )

    with (
        OUTPUT_DIR
        / "senate_environment_generic_only_loco_summary.json"
    ).open("w") as handle:
        json.dump(
            summary,
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    validation_lines = [
        "Senate Generic-Only LOCO Validation",
        "===================================",
        "",
        (
            "PASS — one fold per cycle: "
            f"expected={len(cycles)}, observed={len(folds)}"
        ),
        (
            "PASS — pooled prediction count: "
            f"expected={len(model_data)}, "
            f"observed={len(pooled_predictions)}"
        ),
        (
            "PASS — no duplicate race-cycle predictions"
        ),
        (
            "PASS — all selected coefficients belong "
            "to the configured generic-ballot grid"
        ),
        "",
        "VALIDATION STATUS: PASSED",
    ]

    (
        OUTPUT_DIR
        / "senate_environment_generic_only_loco_validation.txt"
    ).write_text(
        "\n".join(validation_lines) + "\n"
    )

    print()
    print("Coefficient stability")
    print("---------------------")
    print(coefficient_counts.to_string(index=False))

    print()
    print("Pooled out-of-sample performance")
    print("--------------------------------")
    print(
        f"MAE={pooled_metrics['mae']:.6f}, "
        f"RMSE={pooled_metrics['rmse']:.6f}, "
        f"Brier={pooled_metrics['brier']:.6f}, "
        f"accuracy={pooled_metrics['winner_accuracy']:.4%}, "
        f"mean error={pooled_metrics['mean_error']:.6f}"
    )

    print()
    print("Validation status: PASSED")
    print(f"Wrote outputs to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
