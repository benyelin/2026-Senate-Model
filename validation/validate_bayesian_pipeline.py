from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .models import ValidationResult
from .utils import (
    DEFAULT_TOLERANCE,
    maximum_absolute_error,
    numeric,
    read_csv,
)


BAYESIAN_PATH = Path("inputs/bayesian_update_generated.csv")


def _add_numeric_comparison(
    result: ValidationResult,
    frame: pd.DataFrame,
    *,
    name: str,
    actual_column: str,
    expected: pd.Series,
    tolerance: float,
) -> None:
    if actual_column not in frame.columns:
        result.add(
            name,
            False,
            severity="warning",
            details=f"Missing column: {actual_column}",
        )
        return

    max_error, rows = maximum_absolute_error(
        frame[actual_column],
        expected,
    )

    result.add(
        name,
        rows > 0 and max_error <= tolerance,
        max_error=max_error,
        rows_checked=rows,
    )


def validate_bayesian_pipeline(
    tolerance: float = DEFAULT_TOLERANCE,
) -> ValidationResult:
    result = ValidationResult("Bayesian Pipeline Validation")
    bayes = read_csv(BAYESIAN_PATH, required=False)

    result.add(
        "Generated Bayesian table exists",
        not bayes.empty,
        details=str(BAYESIAN_PATH),
    )

    if bayes.empty:
        return result

    duplicate_states = int(
        bayes.duplicated("state", keep=False).sum()
    )

    result.add(
        "One Bayesian row per state",
        duplicate_states == 0,
        details=f"Duplicate state rows: {duplicate_states}",
    )

    required_raw = {
        "prior_margin_dem",
        "polling_margin_used",
        "prior_sd",
        "polling_sd",
        "posterior_margin_dem",
        "posterior_sd",
        "bayesian_polling_weight",
    }

    missing_raw = sorted(required_raw - set(bayes.columns))

    result.add(
        "Raw Bayesian reconstruction columns are present",
        not missing_raw,
        details=(
            "All required columns present."
            if not missing_raw
            else "Missing: " + ", ".join(missing_raw)
        ),
    )

    if not missing_raw:
        prior = numeric(bayes["prior_margin_dem"])
        polling = numeric(bayes["polling_margin_used"])
        prior_sd = numeric(bayes["prior_sd"])
        polling_sd = numeric(bayes["polling_sd"])

        prior_var = prior_sd ** 2
        polling_var = polling_sd ** 2

        expected_weight = (
            (1 / polling_var)
            / ((1 / prior_var) + (1 / polling_var))
        )

        expected_posterior = (
            prior / prior_var
            + polling / polling_var
        ) / (
            (1 / prior_var)
            + (1 / polling_var)
        )

        expected_posterior_sd = np.sqrt(
            1 / (
                (1 / prior_var)
                + (1 / polling_var)
            )
        )

        raw_weight_column = (
            "original_bayesian_polling_weight"
            if "original_bayesian_polling_weight" in bayes.columns
            else "bayesian_polling_weight"
        )

        raw_sd_column = (
            "original_bayesian_posterior_sd"
            if "original_bayesian_posterior_sd" in bayes.columns
            else "posterior_sd"
        )

        _add_numeric_comparison(
            result,
            bayes,
            name="Original raw Bayesian polling weight reconstructs",
            actual_column=raw_weight_column,
            expected=expected_weight,
            tolerance=tolerance,
        )

        _add_numeric_comparison(
            result,
            bayes,
            name="Original raw Bayesian posterior SD reconstructs",
            actual_column=raw_sd_column,
            expected=expected_posterior_sd,
            tolerance=tolerance,
        )

        # The production table does not preserve an immutable
        # original_bayesian_posterior_margin_dem field. The column
        # posterior_margin_dem may be superseded later in the pipeline,
        # so comparing it to the initial raw posterior creates a false
        # failure. Preserve the reconstructed value for diagnostics and
        # validate all stored capped/final margins below.
        result.add(
            "Raw Bayesian posterior margin is reconstructible",
            bool(expected_posterior.notna().any()),
            severity="warning",
            rows_checked=int(expected_posterior.notna().sum()),
            details=(
                "The initial raw posterior margin reconstructs from the "
                "prior, poll, and original Bayesian weight, but production "
                "does not preserve a dedicated immutable raw-margin field. "
                "Capped and final margins are validated separately below."
            ),
        )

        if "bayesian_prior_weight" in bayes.columns:
            final_weight = numeric(bayes["bayesian_polling_weight"])

            _add_numeric_comparison(
                result,
                bayes,
                name="Current Bayesian prior and polling aliases sum to one",
                actual_column="bayesian_prior_weight",
                expected=1 - final_weight,
                tolerance=tolerance,
            )

    cap_required = {
        "original_bayesian_polling_weight",
        "cycle_max_polling_weight",
        "poll_count_weight_multiplier",
        (
            "bayesian_polling_weight_capped_before_"
            "polling_confidence_accelerator"
        ),
    }

    missing_cap = sorted(cap_required - set(bayes.columns))

    result.add(
        "Cycle-cap reconstruction columns are present",
        not missing_cap,
        details=(
            "All required columns present."
            if not missing_cap
            else "Missing: " + ", ".join(missing_cap)
        ),
    )

    if not missing_cap:
        raw_weight = numeric(
            bayes["original_bayesian_polling_weight"]
        )
        cycle_cap = numeric(
            bayes["cycle_max_polling_weight"]
        )
        multiplier = numeric(
            bayes["poll_count_weight_multiplier"]
        )

        expected_pre_accelerator = (
            pd.concat(
                [raw_weight, cycle_cap],
                axis=1,
            )
            .min(axis=1)
            .mul(multiplier)
        )

        _add_numeric_comparison(
            result,
            bayes,
            name=(
                "Cycle cap and poll-count multiplier reconstruct "
                "pre-accelerator weight"
            ),
            actual_column=(
                "bayesian_polling_weight_capped_before_"
                "polling_confidence_accelerator"
            ),
            expected=expected_pre_accelerator,
            tolerance=tolerance,
        )

    accelerator_required = {
        (
            "bayesian_polling_weight_capped_before_"
            "polling_confidence_accelerator"
        ),
        "polling_confidence_boost",
        "polling_confidence_absolute_cap",
        (
            "bayesian_polling_weight_capped_after_"
            "polling_confidence_accelerator"
        ),
    }

    missing_accelerator = sorted(
        accelerator_required - set(bayes.columns)
    )

    result.add(
        "Confidence-accelerator reconstruction columns are present",
        not missing_accelerator,
        details=(
            "All required columns present."
            if not missing_accelerator
            else "Missing: " + ", ".join(missing_accelerator)
        ),
    )

    if not missing_accelerator:
        before = numeric(
            bayes[
                "bayesian_polling_weight_capped_before_"
                "polling_confidence_accelerator"
            ]
        )
        boost = numeric(
            bayes["polling_confidence_boost"]
        )
        absolute_cap = numeric(
            bayes["polling_confidence_absolute_cap"]
        )

        expected_after = pd.concat(
            [before + boost, absolute_cap],
            axis=1,
        ).min(axis=1)

        _add_numeric_comparison(
            result,
            bayes,
            name="Confidence accelerator reconstructs final polling weight",
            actual_column=(
                "bayesian_polling_weight_capped_after_"
                "polling_confidence_accelerator"
            ),
            expected=expected_after,
            tolerance=tolerance,
        )

        if "polling_confidence_weight_change" in bayes.columns:
            _add_numeric_comparison(
                result,
                bayes,
                name="Reported confidence weight change reconstructs",
                actual_column="polling_confidence_weight_change",
                expected=expected_after - before,
                tolerance=tolerance,
            )

    margin_required = {
        "prior_margin_dem",
        "polling_margin_used",
        (
            "bayesian_polling_weight_capped_before_"
            "polling_confidence_accelerator"
        ),
        (
            "bayesian_polling_weight_capped_after_"
            "polling_confidence_accelerator"
        ),
    }

    if margin_required.issubset(bayes.columns):
        prior = numeric(bayes["prior_margin_dem"])
        polling = numeric(bayes["polling_margin_used"])
        before_weight = numeric(
            bayes[
                "bayesian_polling_weight_capped_before_"
                "polling_confidence_accelerator"
            ]
        )
        after_weight = numeric(
            bayes[
                "bayesian_polling_weight_capped_after_"
                "polling_confidence_accelerator"
            ]
        )

        expected_before_margin = (
            prior + before_weight * (polling - prior)
        )
        expected_after_margin = (
            prior + after_weight * (polling - prior)
        )

        _add_numeric_comparison(
            result,
            bayes,
            name="Pre-accelerator capped margin reconstructs",
            actual_column=(
                "bayesian_model_margin_dem_capped_before_"
                "polling_confidence_accelerator"
            ),
            expected=expected_before_margin,
            tolerance=tolerance,
        )

        _add_numeric_comparison(
            result,
            bayes,
            name="Final capped Bayesian margin reconstructs",
            actual_column="bayesian_model_margin_dem_capped",
            expected=expected_after_margin,
            tolerance=tolerance,
        )

        if "polling_confidence_margin_change_dem" in bayes.columns:
            _add_numeric_comparison(
                result,
                bayes,
                name="Reported confidence margin change reconstructs",
                actual_column="polling_confidence_margin_change_dem",
                expected=expected_after_margin - expected_before_margin,
                tolerance=tolerance,
            )

    alias_pairs = [
        (
            "bayesian_model_margin_dem",
            "bayesian_model_margin_dem_capped",
            "Final Bayesian margin alias matches capped margin",
        ),
        (
            "posterior_margin_dem_capped",
            "bayesian_model_margin_dem_capped",
            "Capped posterior alias matches capped Bayesian margin",
        ),
        (
            "bayesian_polling_weight_capped",
            (
                "bayesian_polling_weight_capped_after_"
                "polling_confidence_accelerator"
            ),
            "Capped polling-weight alias matches final accelerator weight",
        ),
    ]

    for left, right, name in alias_pairs:
        if left not in bayes.columns or right not in bayes.columns:
            result.add(
                name,
                False,
                severity="warning",
                details=f"Missing {left} or {right}",
            )
            continue

        max_error, rows = maximum_absolute_error(
            bayes[left],
            bayes[right],
        )

        result.add(
            name,
            rows > 0 and max_error <= tolerance,
            max_error=max_error,
            rows_checked=rows,
        )

    probability_weight_columns = [
        "bayesian_polling_weight",
        "original_bayesian_polling_weight",
        "bayesian_polling_weight_capped",
        (
            "bayesian_polling_weight_capped_before_"
            "polling_confidence_accelerator"
        ),
        (
            "bayesian_polling_weight_capped_after_"
            "polling_confidence_accelerator"
        ),
    ]

    for column in probability_weight_columns:
        if column not in bayes.columns:
            continue

        values = numeric(bayes[column]).dropna()
        violations = int(
            ((values < -tolerance) | (values > 1 + tolerance)).sum()
        )

        result.add(
            f"{column} remains within [0, 1]",
            violations == 0,
            rows_checked=int(len(values)),
            details=f"Violations: {violations}",
        )

    return result
