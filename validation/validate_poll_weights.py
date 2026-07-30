from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .models import ValidationResult
from .utils import (
    DEFAULT_TOLERANCE,
    available_column,
    duplicate_key_count,
    maximum_absolute_error,
    numeric,
    read_csv,
)


AUDIT_PATH = Path("outputs/senate_poll_weighting_live_audit.csv")
AVERAGES_PATH = Path("inputs/polling_averages_generated.csv")


def validate_poll_weights(
    tolerance: float = DEFAULT_TOLERANCE,
) -> ValidationResult:
    result = ValidationResult("Poll Weight Validation")

    audit = read_csv(AUDIT_PATH, required=False)
    averages = read_csv(AVERAGES_PATH, required=False)

    result.add(
        "Poll-level production audit exists",
        not audit.empty,
        details=str(AUDIT_PATH),
    )

    result.add(
        "Generated polling averages exist",
        not averages.empty,
        details=str(AVERAGES_PATH),
    )

    if audit.empty or averages.empty:
        return result

    result.add(
        "Audit contains state identifiers",
        "state" in audit.columns,
    )
    result.add(
        "Polling averages contain state identifiers",
        "state" in averages.columns,
    )

    if "state" not in audit.columns or "state" not in averages.columns:
        return result

    state_duplicates = duplicate_key_count(averages, ["state"])
    result.add(
        "One polling-average row per state",
        state_duplicates == 0,
        details=f"Duplicate state rows: {state_duplicates}",
    )

    exact_key_candidates = [
        ["state", "pollster", "start_date", "end_date", "dem_pct", "rep_pct"],
        ["state", "pollster", "end_date", "dem_pct", "rep_pct"],
    ]

    exact_key = next(
        (
            columns
            for columns in exact_key_candidates
            if set(columns).issubset(audit.columns)
        ),
        None,
    )

    if exact_key is not None:
        duplicate_rows = duplicate_key_count(audit, exact_key)
        result.add(
            "No exact duplicate polls remain in production audit",
            duplicate_rows == 0,
            details=(
                f"Duplicate rows: {duplicate_rows}; "
                f"key={', '.join(exact_key)}"
            ),
        )
    else:
        result.add(
            "Exact duplicate-poll audit",
            False,
            severity="warning",
            details="Required poll identity columns were not all available.",
        )

    normalized_weight_col = available_column(
        audit,
        [
            "normalized_weight",
            "final_normalized_weight",
            "poll_weight_normalized",
        ],
    )

    if normalized_weight_col:
        sums = (
            audit.assign(
                _validation_weight=numeric(audit[normalized_weight_col])
            )
            .groupby("state")["_validation_weight"]
            .sum(min_count=1)
        )
        errors = (sums - 1.0).abs().dropna()
        max_error = float(errors.max()) if not errors.empty else 0.0

        result.add(
            "Normalized poll weights sum to one within each state",
            max_error <= tolerance,
            max_error=max_error,
            rows_checked=int(len(sums)),
        )
    else:
        result.add(
            "Normalized poll-weight sum audit",
            False,
            severity="warning",
            details="No recognized normalized-weight column was found.",
        )

    adjusted_margin_col = available_column(
        audit,
        [
            "adjusted_margin_dem",
            "adjusted_poll_margin_dem",
            "polling_margin_dem_adjusted",
            "margin_dem_adjusted",
            "poll_margin_dem",
            "margin_dem",
        ],
    )

    final_weight_col = available_column(
        audit,
        [
            "normalized_weight",
            "final_normalized_weight",
            "poll_weight_normalized",
            "final_weight",
        ],
    )

    production_margin_col = available_column(
        averages,
        ["polling_margin_dem"],
    )

    if adjusted_margin_col and final_weight_col and production_margin_col:
        reconstruction = audit.copy()
        reconstruction["_margin"] = numeric(
            reconstruction[adjusted_margin_col]
        )
        reconstruction["_weight"] = numeric(
            reconstruction[final_weight_col]
        )

        if final_weight_col in {
            "normalized_weight",
            "final_normalized_weight",
            "poll_weight_normalized",
        }:
            reconstructed = (
                reconstruction
                .assign(_contribution=lambda frame: (
                    frame["_margin"] * frame["_weight"]
                ))
                .groupby("state", as_index=False)["_contribution"]
                .sum(min_count=1)
                .rename(
                    columns={
                        "_contribution":
                        "reconstructed_polling_margin_dem"
                    }
                )
            )
        else:
            grouped = reconstruction.groupby("state", as_index=False).agg(
                weighted_sum=(
                    "_margin",
                    lambda values: 0.0,
                )
            )

            records = []
            for state, group in reconstruction.groupby("state"):
                valid = group["_margin"].notna() & group["_weight"].notna()
                denominator = group.loc[valid, "_weight"].sum()
                numerator = (
                    group.loc[valid, "_margin"]
                    * group.loc[valid, "_weight"]
                ).sum()

                records.append(
                    {
                        "state": state,
                        "reconstructed_polling_margin_dem": (
                            numerator / denominator
                            if denominator > 0
                            else np.nan
                        ),
                    }
                )

            reconstructed = pd.DataFrame(records)

        comparison = averages[
            ["state", production_margin_col]
        ].merge(
            reconstructed,
            on="state",
            how="left",
            validate="one_to_one",
        )

        max_error, rows = maximum_absolute_error(
            comparison[production_margin_col],
            comparison["reconstructed_polling_margin_dem"],
        )

        result.add(
            "Polling averages reconstruct from poll-level audit",
            rows > 0 and max_error <= tolerance,
            max_error=max_error,
            rows_checked=rows,
            details=(
                f"margin={adjusted_margin_col}; "
                f"weight={final_weight_col}"
            ),
        )
    else:
        missing = []
        if not adjusted_margin_col:
            missing.append("adjusted margin")
        if not final_weight_col:
            missing.append("final weight")
        if not production_margin_col:
            missing.append("production polling margin")

        result.add(
            "Polling-average reconstruction",
            False,
            severity="warning",
            details="Missing recognized columns: " + ", ".join(missing),
        )

    if {
        "effective_poll_count",
        "poll_count",
    }.issubset(averages.columns):
        effective = numeric(averages["effective_poll_count"])
        poll_count = numeric(averages["poll_count"])
        valid = effective.notna() & poll_count.notna()
        violations = int(
            (effective[valid] > poll_count[valid] + tolerance).sum()
        )

        result.add(
            "Effective poll count does not exceed poll count",
            violations == 0,
            rows_checked=int(valid.sum()),
            details=f"Violations: {violations}",
        )

    if "largest_pollster_weight_share" in averages.columns:
        shares = numeric(
            averages["largest_pollster_weight_share"]
        ).dropna()
        violations = int(
            ((shares < -tolerance) | (shares > 1 + tolerance)).sum()
        )

        result.add(
            "Largest pollster share remains within [0, 1]",
            violations == 0,
            rows_checked=int(len(shares)),
            details=f"Violations: {violations}",
        )

    return result
