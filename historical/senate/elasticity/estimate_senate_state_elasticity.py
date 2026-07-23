#!/usr/bin/env python3
"""
Estimate historical Senate seat elasticity from six-year swing observations.

For each permanent Senate seat, raw elasticity is estimated using constrained
ordinary least squares through the origin:

    raw_elasticity =
        sum(national_swing_dem * seat_swing_dem)
        / sum(national_swing_dem ** 2)

The raw estimate measures how responsive that Senate seat's electoral margin
has historically been to changes in the national environment.

Because the Senate historical sample is sparse, the estimator also produces
a conservatively shrunk estimate:

    shrunk_elasticity =
        shrinkage_target
        + (1 - shrinkage_strength)
          * (raw_elasticity - shrinkage_target)

With the defaults:

    shrinkage_target   = 1.0
    shrinkage_strength = 0.50

the estimator retains 50% of each seat's raw deviation from unit elasticity.

This script estimates and validates elasticity values only. It does not decide
whether or how strongly they should be used in production. That decision must
be made later through leakage-free replay sensitivity testing.

Usage
-----
From the repository root:

    python3 historical/senate/elasticity/estimate_senate_state_elasticity.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "historical/senate/warehouse/processed"
    / "senate_state_swing_observations.csv"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "historical/senate/warehouse/processed"
    / "senate_state_elasticity.csv"
)

DEFAULT_VALIDATION = (
    PROJECT_ROOT
    / "historical/senate/warehouse/validation"
    / "senate_state_elasticity_validation.txt"
)

DEFAULT_DIAGNOSTICS = (
    PROJECT_ROOT
    / "historical/senate/warehouse/validation"
    / "senate_state_elasticity_observation_diagnostics.csv"
)

DEFAULT_SHRINKAGE_TARGET = 1.0
DEFAULT_SHRINKAGE_STRENGTH = 0.50
DEFAULT_RIDGE_LAMBDA = 30.0

EXPECTED_INPUT_COLUMNS = {
    "seat_id",
    "state",
    "senate_class",
    "previous_cycle",
    "current_cycle",
    "seat_swing_dem",
    "national_swing_dem",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate Senate seat elasticity from validated historical "
            "six-year swing observations."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Validated Senate swing-observation warehouse.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Reusable Senate seat-elasticity warehouse.",
    )

    parser.add_argument(
        "--validation-output",
        type=Path,
        default=DEFAULT_VALIDATION,
        help="Human-readable estimator validation report.",
    )

    parser.add_argument(
        "--diagnostics-output",
        type=Path,
        default=DEFAULT_DIAGNOSTICS,
        help="Observation-level diagnostics with fitted values and residuals.",
    )

    parser.add_argument(
        "--shrinkage-target",
        type=float,
        default=DEFAULT_SHRINKAGE_TARGET,
        help="Elasticity value toward which raw estimates are shrunk.",
    )

    parser.add_argument(
        "--shrinkage-strength",
        type=float,
        default=DEFAULT_SHRINKAGE_STRENGTH,
        help=(
            "Legacy fixed-shrinkage diagnostic. Share of each raw "
            "estimate's deviation from the target removed."
        ),
    )

    parser.add_argument(
        "--ridge-lambda",
        type=float,
        default=DEFAULT_RIDGE_LAMBDA,
        help=(
            "Information-weighted regularization penalty. Larger values "
            "shrink seat elasticity more strongly toward the target."
        ),
    )

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    if not np.isfinite(args.shrinkage_target):
        raise ValueError("Shrinkage target must be finite.")

    if not np.isfinite(args.shrinkage_strength):
        raise ValueError("Shrinkage strength must be finite.")

    if not 0.0 <= args.shrinkage_strength <= 1.0:
        raise ValueError(
            "Shrinkage strength must be between 0.0 and 1.0."
        )

    if not np.isfinite(args.ridge_lambda):
        raise ValueError("Ridge lambda must be finite.")

    if args.ridge_lambda < 0.0:
        raise ValueError("Ridge lambda must be nonnegative.")


def load_observations(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Senate swing-observation warehouse not found: {path}"
        )

    observations = pd.read_csv(path)

    if observations.empty:
        raise ValueError(
            f"Senate swing-observation warehouse is empty: {path}"
        )

    missing_columns = sorted(
        EXPECTED_INPUT_COLUMNS - set(observations.columns)
    )
    if missing_columns:
        raise ValueError(
            "Swing-observation warehouse is missing required columns: "
            + ", ".join(missing_columns)
        )

    work = observations.copy()

    work["seat_id"] = (
        work["seat_id"]
        .astype("string")
        .str.strip()
        .str.upper()
    )
    work["state"] = (
        work["state"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    numeric_columns = (
        "senate_class",
        "previous_cycle",
        "current_cycle",
        "seat_swing_dem",
        "national_swing_dem",
    )

    for column in numeric_columns:
        work[column] = pd.to_numeric(
            work[column],
            errors="coerce",
        )

    required_columns = sorted(EXPECTED_INPUT_COLUMNS)
    missing_values = {
        column: int(work[column].isna().sum())
        for column in required_columns
        if work[column].isna().any()
    }

    if missing_values:
        raise ValueError(
            "Missing required values in swing observations: "
            + json.dumps(missing_values, sort_keys=True)
        )

    for column in numeric_columns:
        values = work[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite values found in {column}."
            )

    work["senate_class"] = work["senate_class"].astype(int)
    work["previous_cycle"] = work["previous_cycle"].astype(int)
    work["current_cycle"] = work["current_cycle"].astype(int)

    invalid_classes = ~work["senate_class"].isin([1, 2, 3])
    if invalid_classes.any():
        values = sorted(
            work.loc[
                invalid_classes,
                "senate_class",
            ].unique().tolist()
        )
        raise ValueError(
            f"Invalid Senate classes found: {values}"
        )

    expected_seat_id = (
        work["state"]
        + "_CLASS_"
        + work["senate_class"].astype(str)
    )

    seat_id_mismatch = work["seat_id"].ne(expected_seat_id)
    if seat_id_mismatch.any():
        examples = (
            work.loc[
                seat_id_mismatch,
                [
                    "seat_id",
                    "state",
                    "senate_class",
                ],
            ]
            .assign(expected_seat_id=expected_seat_id[seat_id_mismatch])
            .head(20)
            .to_dict("records")
        )
        raise ValueError(
            "Seat identifiers disagree with state and Senate class: "
            + json.dumps(examples)
        )

    duplicate_observations = work.duplicated(
        subset=[
            "seat_id",
            "previous_cycle",
            "current_cycle",
        ],
        keep=False,
    )

    if duplicate_observations.any():
        examples = (
            work.loc[
                duplicate_observations,
                [
                    "seat_id",
                    "previous_cycle",
                    "current_cycle",
                ],
            ]
            .head(20)
            .to_dict("records")
        )
        raise ValueError(
            "Duplicate seat transition observations found: "
            + json.dumps(examples)
        )

    invalid_gap = (
        work["current_cycle"]
        - work["previous_cycle"]
    ).ne(6)

    if invalid_gap.any():
        examples = (
            work.loc[
                invalid_gap,
                [
                    "seat_id",
                    "previous_cycle",
                    "current_cycle",
                ],
            ]
            .head(20)
            .to_dict("records")
        )
        raise ValueError(
            "Estimator input contains non-six-year transitions: "
            + json.dumps(examples)
        )

    return work.sort_values(
        [
            "seat_id",
            "previous_cycle",
            "current_cycle",
        ]
    ).reset_index(drop=True)


def estimate_single_seat(
    group: pd.DataFrame,
    *,
    shrinkage_target: float,
    shrinkage_strength: float,
    ridge_lambda: float,
) -> dict[str, object]:
    national_swing = group[
        "national_swing_dem"
    ].to_numpy(dtype=float)

    seat_swing = group[
        "seat_swing_dem"
    ].to_numpy(dtype=float)

    numerator = float(
        np.dot(national_swing, seat_swing)
    )
    denominator = float(
        np.dot(national_swing, national_swing)
    )

    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError(
            f"Seat {group.iloc[0]['seat_id']} has no usable "
            "national-swing variation."
        )

    raw_elasticity = numerator / denominator

    shrunk_elasticity = (
        shrinkage_target
        + (1.0 - shrinkage_strength)
        * (raw_elasticity - shrinkage_target)
    )

    information_weight = (
        denominator
        / (denominator + ridge_lambda)
        if denominator + ridge_lambda > 0
        else 0.0
    )

    regularized_elasticity = (
        shrinkage_target
        + information_weight
        * (raw_elasticity - shrinkage_target)
    )

    fitted_raw = raw_elasticity * national_swing
    residual_raw = seat_swing - fitted_raw

    residual_sum_squares = float(
        np.dot(residual_raw, residual_raw)
    )

    root_mean_squared_residual = float(
        np.sqrt(np.mean(np.square(residual_raw)))
    )

    mean_absolute_residual = float(
        np.mean(np.abs(residual_raw))
    )

    national_swing_sum_squares = denominator

    first = group.iloc[0]

    return {
        "seat_id": str(first["seat_id"]),
        "state": str(first["state"]),
        "senate_class": int(first["senate_class"]),
        "observation_count": int(len(group)),
        "first_previous_cycle": int(
            group["previous_cycle"].min()
        ),
        "last_current_cycle": int(
            group["current_cycle"].max()
        ),
        "mean_seat_swing_dem": float(
            group["seat_swing_dem"].mean()
        ),
        "mean_national_swing_dem": float(
            group["national_swing_dem"].mean()
        ),
        "elasticity_numerator": numerator,
        "elasticity_denominator": denominator,
        "raw_elasticity": raw_elasticity,
        "shrinkage_target": shrinkage_target,
        "shrinkage_strength": shrinkage_strength,
        "elasticity_variation_retained": (
            1.0 - shrinkage_strength
        ),
        "shrunk_elasticity": shrunk_elasticity,
        "ridge_lambda": ridge_lambda,
        "information_weight": information_weight,
        "regularized_elasticity": regularized_elasticity,
        "raw_residual_sum_squares": residual_sum_squares,
        "raw_residual_mae": mean_absolute_residual,
        "raw_residual_rmse": root_mean_squared_residual,
        "national_swing_sum_squares": (
            national_swing_sum_squares
        ),
    }


def estimate_elasticities(
    observations: pd.DataFrame,
    *,
    shrinkage_target: float,
    shrinkage_strength: float,
    ridge_lambda: float,
) -> pd.DataFrame:
    rows = []

    for _, group in observations.groupby(
        "seat_id",
        sort=True,
    ):
        rows.append(
            estimate_single_seat(
                group,
                shrinkage_target=shrinkage_target,
                shrinkage_strength=shrinkage_strength,
                ridge_lambda=ridge_lambda,
            )
        )

    estimates = pd.DataFrame(rows)

    if estimates.empty:
        raise ValueError("No Senate elasticity estimates were produced.")

    return estimates.sort_values(
        ["state", "senate_class"]
    ).reset_index(drop=True)


def build_observation_diagnostics(
    observations: pd.DataFrame,
    estimates: pd.DataFrame,
) -> pd.DataFrame:
    diagnostics = observations.merge(
        estimates[
            [
                "seat_id",
                "raw_elasticity",
                "shrunk_elasticity",
                "information_weight",
                "regularized_elasticity",
                "observation_count",
            ]
        ],
        on="seat_id",
        how="left",
        validate="many_to_one",
    )

    if diagnostics[
        [
            "raw_elasticity",
            "shrunk_elasticity",
            "information_weight",
            "regularized_elasticity",
            "observation_count",
        ]
    ].isna().any().any():
        raise ValueError(
            "Observation diagnostics failed to match all seat estimates."
        )

    diagnostics["raw_fitted_seat_swing_dem"] = (
        diagnostics["raw_elasticity"]
        * diagnostics["national_swing_dem"]
    )

    diagnostics["raw_residual_dem"] = (
        diagnostics["seat_swing_dem"]
        - diagnostics["raw_fitted_seat_swing_dem"]
    )

    diagnostics["shrunk_fitted_seat_swing_dem"] = (
        diagnostics["shrunk_elasticity"]
        * diagnostics["national_swing_dem"]
    )

    diagnostics["shrunk_residual_dem"] = (
        diagnostics["seat_swing_dem"]
        - diagnostics["shrunk_fitted_seat_swing_dem"]
    )

    diagnostics["regularized_fitted_seat_swing_dem"] = (
        diagnostics["regularized_elasticity"]
        * diagnostics["national_swing_dem"]
    )

    diagnostics["regularized_residual_dem"] = (
        diagnostics["seat_swing_dem"]
        - diagnostics["regularized_fitted_seat_swing_dem"]
    )

    return diagnostics.sort_values(
        [
            "seat_id",
            "previous_cycle",
            "current_cycle",
        ]
    ).reset_index(drop=True)


def validate_estimates(
    estimates: pd.DataFrame,
    observations: pd.DataFrame,
    diagnostics: pd.DataFrame,
    *,
    shrinkage_target: float,
    shrinkage_strength: float,
    ridge_lambda: float,
) -> list[str]:
    failures: list[str] = []

    if estimates["seat_id"].duplicated().any():
        failures.append("Duplicate seat elasticity rows found.")

    expected_seats = observations["seat_id"].nunique()
    if len(estimates) != expected_seats:
        failures.append(
            "Elasticity-row count does not equal the number of "
            f"observed seats: {len(estimates)} versus {expected_seats}."
        )

    estimated_observation_count = int(
        estimates["observation_count"].sum()
    )
    if estimated_observation_count != len(observations):
        failures.append(
            "Seat observation counts do not sum to the source-row count: "
            f"{estimated_observation_count} versus {len(observations)}."
        )

    numeric_columns = [
        "raw_elasticity",
        "shrunk_elasticity",
        "information_weight",
        "regularized_elasticity",
        "elasticity_numerator",
        "elasticity_denominator",
        "raw_residual_mae",
        "raw_residual_rmse",
    ]

    for column in numeric_columns:
        values = estimates[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            failures.append(
                f"Non-finite values found in {column}."
            )

    nonpositive_denominator = estimates[
        "elasticity_denominator"
    ].le(0)
    if nonpositive_denominator.any():
        failures.append(
            "At least one elasticity denominator is not positive."
        )

    expected_shrunk = (
        shrinkage_target
        + (1.0 - shrinkage_strength)
        * (
            estimates["raw_elasticity"]
            - shrinkage_target
        )
    )

    shrinkage_failure = ~np.isclose(
        estimates["shrunk_elasticity"],
        expected_shrunk,
        atol=1e-12,
        rtol=0.0,
    )

    if shrinkage_failure.any():
        failures.append(
            "Shrunk-elasticity arithmetic validation failed."
        )

    expected_information_weight = (
        estimates["elasticity_denominator"]
        / (
            estimates["elasticity_denominator"]
            + ridge_lambda
        )
    )

    if not np.allclose(
        estimates["information_weight"],
        expected_information_weight,
        atol=1e-12,
        rtol=0.0,
    ):
        failures.append(
            "Information-weight arithmetic validation failed."
        )

    expected_regularized = (
        shrinkage_target
        + expected_information_weight
        * (
            estimates["raw_elasticity"]
            - shrinkage_target
        )
    )

    if not np.allclose(
        estimates["regularized_elasticity"],
        expected_regularized,
        atol=1e-12,
        rtol=0.0,
    ):
        failures.append(
            "Regularized-elasticity arithmetic validation failed."
        )

    invalid_weights = (
        estimates["information_weight"].lt(0.0)
        | estimates["information_weight"].gt(1.0)
    )

    if invalid_weights.any():
        failures.append(
            "Information weights must remain between zero and one."
        )

    if shrinkage_strength > 0:
        raw_distance = (
            estimates["raw_elasticity"]
            - shrinkage_target
        ).abs()

        shrunk_distance = (
            estimates["shrunk_elasticity"]
            - shrinkage_target
        ).abs()

        farther_from_target = (
            shrunk_distance
            > raw_distance + 1e-12
        )

        if farther_from_target.any():
            failures.append(
                "Shrinkage moved at least one estimate farther "
                "from the target."
            )

    grouped = diagnostics.groupby("seat_id", sort=True)

    for seat_id, group in grouped:
        raw_elasticity = float(
            group["raw_elasticity"].iloc[0]
        )

        recalculated_numerator = float(
            np.dot(
                group["national_swing_dem"],
                group["seat_swing_dem"],
            )
        )

        recalculated_denominator = float(
            np.dot(
                group["national_swing_dem"],
                group["national_swing_dem"],
            )
        )

        recalculated_raw = (
            recalculated_numerator
            / recalculated_denominator
        )

        if not np.isclose(
            raw_elasticity,
            recalculated_raw,
            atol=1e-12,
            rtol=0.0,
        ):
            failures.append(
                f"Raw elasticity failed recomputation for {seat_id}."
            )
            break

        # OLS through the origin requires residuals to be orthogonal
        # to the explanatory variable within numerical precision.
        orthogonality = float(
            np.dot(
                group["national_swing_dem"],
                group["raw_residual_dem"],
            )
        )

        if abs(orthogonality) > 1e-8:
            failures.append(
                f"Through-origin OLS orthogonality failed for "
                f"{seat_id}: {orthogonality:.12f}."
            )
            break

    if len(diagnostics) != len(observations):
        failures.append(
            "Observation diagnostics row count differs from source data."
        )

    if failures:
        raise RuntimeError(
            "Senate elasticity validation failed:\n"
            + "\n".join(
                f"- {failure}"
                for failure in failures
            )
        )

    return [
        f"Unique observed seats: {expected_seats}",
        f"Source observations: {len(observations)}",
        (
            "Summed seat observation counts: "
            f"{estimated_observation_count}"
        ),
        "Duplicate seat estimates: 0",
        "Nonpositive elasticity denominators: 0",
        "Non-finite estimator values: 0",
        "Raw elasticity recomputation failures: 0",
        "Through-origin OLS orthogonality failures: 0",
        "Shrinkage arithmetic failures: 0",
        "Information-weight arithmetic failures: 0",
        "Regularized-elasticity arithmetic failures: 0",
        "Invalid information weights: 0",
        "Observation diagnostic merge failures: 0",
    ]


def describe_series(
    series: pd.Series,
    *,
    decimals: int = 6,
) -> list[str]:
    quantiles = series.quantile(
        [0.00, 0.05, 0.25, 0.50, 0.75, 0.95, 1.00]
    )

    return [
        f"  min:    {quantiles.loc[0.00]:.{decimals}f}",
        f"  p05:    {quantiles.loc[0.05]:.{decimals}f}",
        f"  p25:    {quantiles.loc[0.25]:.{decimals}f}",
        f"  median: {quantiles.loc[0.50]:.{decimals}f}",
        f"  mean:   {series.mean():.{decimals}f}",
        f"  p75:    {quantiles.loc[0.75]:.{decimals}f}",
        f"  p95:    {quantiles.loc[0.95]:.{decimals}f}",
        f"  max:    {quantiles.loc[1.00]:.{decimals}f}",
        f"  std:    {series.std(ddof=0):.{decimals}f}",
    ]


def format_observation_counts(
    estimates: pd.DataFrame,
) -> list[str]:
    counts = (
        estimates["observation_count"]
        .value_counts()
        .sort_index()
    )

    return [
        f"  {int(observation_count)} observation(s): "
        f"{int(seat_count)} seats"
        for observation_count, seat_count in counts.items()
    ]


def build_validation_report(
    *,
    input_path: Path,
    output_path: Path,
    diagnostics_path: Path,
    estimates: pd.DataFrame,
    observations: pd.DataFrame,
    shrinkage_target: float,
    shrinkage_strength: float,
    ridge_lambda: float,
    validation_lines: Iterable[str],
) -> str:
    raw_extremes = estimates.reindex(
        estimates["raw_elasticity"]
        .abs()
        .sort_values(ascending=False)
        .index
    ).head(10)

    report_lines = [
        "Senate State Elasticity Estimator Validation",
        "=" * 44,
        "",
        f"Input: {input_path}",
        f"Elasticity output: {output_path}",
        f"Diagnostics output: {diagnostics_path}",
        "",
        "Estimator:",
        (
            "  Raw elasticity = sum(national swing * seat swing) "
            "/ sum(national swing squared)"
        ),
        "  Regression intercept constrained to zero",
        f"  Shrinkage target: {shrinkage_target:.6f}",
        f"  Shrinkage strength: {shrinkage_strength:.6f}",
        (
            "  Legacy fixed raw variation retained: "
            f"{1.0 - shrinkage_strength:.6f}"
        ),
        f"  Ridge lambda: {ridge_lambda:.6f}",
        (
            "  Information weight = national swing sum of squares / "
            "(national swing sum of squares + ridge lambda)"
        ),
        "",
        "Sample:",
        f"  Swing observations: {len(observations)}",
        f"  Unique observed seats: {len(estimates)}",
        "",
        "Observation counts per seat:",
        *format_observation_counts(estimates),
        "",
        "Raw elasticity distribution:",
        *describe_series(estimates["raw_elasticity"]),
        "",
        "Legacy fixed-shrinkage elasticity distribution:",
        *describe_series(estimates["shrunk_elasticity"]),
        "",
        "Information-weight distribution:",
        *describe_series(estimates["information_weight"]),
        "",
        "Regularized elasticity distribution:",
        *describe_series(estimates["regularized_elasticity"]),
        "",
        "Largest absolute raw elasticity estimates:",
    ]

    for row in raw_extremes.itertuples(index=False):
        report_lines.append(
            f"  {row.seat_id}: "
            f"raw={row.raw_elasticity:.6f}, "
            f"fixed_shrunk={row.shrunk_elasticity:.6f}, "
            f"weight={row.information_weight:.6f}, "
            f"regularized={row.regularized_elasticity:.6f}, "
            f"n={row.observation_count}"
        )

    report_lines.extend(
        [
            "",
            "Validation checks:",
            *validation_lines,
            "",
            "Important interpretation:",
            (
                "  These are historical estimates, not yet validated "
                "production coefficients."
            ),
            (
                "  Replay sensitivity testing must determine how much "
                "elasticity variation, if any, improves out-of-sample "
                "forecasting."
            ),
            (
                "  Seats absent from this warehouse require a neutral "
                "fallback elasticity of 1.0 during replay integration."
            ),
            "",
            "VALIDATION STATUS: PASSED",
            "",
        ]
    )

    return "\n".join(report_lines)


def print_summary(
    *,
    estimates: pd.DataFrame,
    observations: pd.DataFrame,
    output_path: Path,
    diagnostics_path: Path,
    validation_path: Path,
    shrinkage_target: float,
    shrinkage_strength: float,
    ridge_lambda: float,
) -> None:
    print("Senate State Elasticity Estimator")
    print("=" * 40)
    print(f"Swing observations: {len(observations)}")
    print(f"Unique observed seats: {len(estimates)}")
    print(f"Shrinkage target: {shrinkage_target:.3f}")
    print(f"Shrinkage strength: {shrinkage_strength:.3f}")
    print(
        "Legacy fixed variation retained: "
        f"{1.0 - shrinkage_strength:.3f}"
    )
    print(f"Ridge lambda: {ridge_lambda:.3f}")

    print("")
    print("Observation counts")
    print("-" * 18)
    for line in format_observation_counts(estimates):
        print(line)

    print("")
    print("Raw elasticity")
    print("-" * 14)
    for line in describe_series(estimates["raw_elasticity"]):
        print(line)

    print("")
    print("Shrunk elasticity")
    print("-" * 17)
    for line in describe_series(estimates["shrunk_elasticity"]):
        print(line)

    print("")
    print("Information weight")
    print("-" * 18)
    for line in describe_series(estimates["information_weight"]):
        print(line)

    print("")
    print("Regularized elasticity")
    print("-" * 22)
    for line in describe_series(estimates["regularized_elasticity"]):
        print(line)

    print("")
    print("Validation status: PASSED")
    print("")
    print(f"Wrote: {output_path}")
    print(f"Wrote: {diagnostics_path}")
    print(f"Wrote: {validation_path}")


def main() -> None:
    args = parse_args()
    validate_arguments(args)

    observations = load_observations(args.input)

    estimates = estimate_elasticities(
        observations,
        shrinkage_target=args.shrinkage_target,
        shrinkage_strength=args.shrinkage_strength,
        ridge_lambda=args.ridge_lambda,
    )

    diagnostics = build_observation_diagnostics(
        observations,
        estimates,
    )

    validation_lines = validate_estimates(
        estimates,
        observations,
        diagnostics,
        shrinkage_target=args.shrinkage_target,
        shrinkage_strength=args.shrinkage_strength,
        ridge_lambda=args.ridge_lambda,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.validation_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.diagnostics_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    estimates.to_csv(
        args.output,
        index=False,
    )

    diagnostics.to_csv(
        args.diagnostics_output,
        index=False,
    )

    report = build_validation_report(
        input_path=args.input,
        output_path=args.output,
        diagnostics_path=args.diagnostics_output,
        estimates=estimates,
        observations=observations,
        shrinkage_target=args.shrinkage_target,
        shrinkage_strength=args.shrinkage_strength,
        ridge_lambda=args.ridge_lambda,
        validation_lines=validation_lines,
    )

    args.validation_output.write_text(
        report,
        encoding="utf-8",
    )

    print_summary(
        estimates=estimates,
        observations=observations,
        output_path=args.output,
        diagnostics_path=args.diagnostics_output,
        validation_path=args.validation_output,
        shrinkage_target=args.shrinkage_target,
        shrinkage_strength=args.shrinkage_strength,
        ridge_lambda=args.ridge_lambda,
    )


if __name__ == "__main__":
    main()
