"""
M3 Bayesian-style updater for Senate model margins.

Idempotent version for the N dashboard pipeline.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


BAYES_COLUMNS = [
    "pre_bayes_model_margin_dem",
    "bayesian_model_margin_dem",
    "bayesian_polling_weight",
    "bayesian_posterior_sd",
]


def _interp(calibration: pd.DataFrame, days_out: float, column: str, default: float) -> float:
    if "days_out" not in calibration.columns or column not in calibration.columns:
        return default
    curve = calibration[["days_out", column]].dropna().sort_values("days_out")
    if curve.empty:
        return default
    return float(np.interp(float(days_out), curve["days_out"], curve[column]))



def _refresh_poll_metadata_from_generated(merged):
    """
    After merging races and polling averages, force poll_count / weight / age
    to come from inputs/polling_averages_generated.csv when available.

    This avoids stale race_inputs.csv poll_count values suppressing polling weight.
    """
    from pathlib import Path
    import pandas as pd

    polling_path = Path("inputs/polling_averages_generated.csv")

    if not polling_path.exists():
        return merged

    polling = pd.read_csv(polling_path)

    if polling.empty or "state" not in polling.columns:
        return merged

    polling["state"] = polling["state"].astype(str).str.strip().str.upper()
    merged = merged.copy()
    merged["state"] = merged["state"].astype(str).str.strip().str.upper()

    refresh_cols = [
        "poll_count",
        "total_poll_weight",
        "avg_poll_age_days",
        "polling_margin_dem",
    ]

    keep = ["state"] + [c for c in refresh_cols if c in polling.columns]
    polling = polling[keep].copy()

    merged = merged.merge(
        polling,
        on="state",
        how="left",
        suffixes=("", "_generated_truth"),
    )

    for col in refresh_cols:
        truth_col = f"{col}_generated_truth"
        if truth_col in merged.columns:
            merged[col] = merged[truth_col].combine_first(merged.get(col))
            merged = merged.drop(columns=[truth_col])

    return merged


def run_bayesian_update(
    input_dir: str | Path = "inputs",
    as_of_days_out: int | None = None,
    prior_sd_multiplier: float = 1.15,
    min_polling_sd: float = 2.0,
    sparse_poll_penalty: float = 1.25,
) -> pd.DataFrame:
    input_dir = Path(input_dir)
    races = pd.read_csv(input_dir / "race_inputs.csv")
    calibration = pd.read_csv(input_dir / "calibration_parameters.csv")

    races = races.drop(columns=[c for c in BAYES_COLUMNS if c in races.columns], errors="ignore")

    polling_avg_path = input_dir / "polling_averages_generated.csv"
    if polling_avg_path.exists():
        polling = pd.read_csv(polling_avg_path)
    else:
        polling = pd.DataFrame(columns=[
            "state",
            "polling_margin_dem",
            "poll_count",
            "latest_poll_end_date",
            "avg_poll_age_days",
            "total_poll_weight",
        ])

    days_out = int(as_of_days_out) if as_of_days_out is not None else 165
    total_error_sd = _interp(calibration, days_out, "total_error_sd", 6.0)
    prior_sd = total_error_sd * prior_sd_multiplier

    merged = races.merge(
        polling[["state", "polling_margin_dem", "poll_count", "avg_poll_age_days", "total_poll_weight"]],
        on="state",
        how="left",
        suffixes=("", "_poll_generated"),
    )

    merged["polling_margin_used"] = merged["polling_margin_dem_poll_generated"].fillna(
        merged["polling_margin_dem"]
    )
    merged = _refresh_poll_metadata_from_generated(merged)

    # Force generated polling metadata to numeric dtype.
    # Empty/no-poll CSVs can otherwise leave these columns as object dtype,
    # which breaks downstream NumPy calculations.
    merged["poll_count"] = pd.to_numeric(
        merged["poll_count"],
        errors="coerce",
    ).fillna(0.0)

    merged["total_poll_weight"] = pd.to_numeric(
        merged["total_poll_weight"],
        errors="coerce",
    ).fillna(0.0)

    merged["avg_poll_age_days"] = pd.to_numeric(
        merged["avg_poll_age_days"],
        errors="coerce",
    ).fillna(999.0)

    effective_weight = merged["total_poll_weight"].clip(lower=0.0)
    polling_sd = total_error_sd / np.sqrt(1 + effective_weight)
    polling_sd = np.where(merged["poll_count"] < 2, polling_sd * sparse_poll_penalty, polling_sd)
    polling_sd = np.maximum(polling_sd, min_polling_sd)

    merged["prior_margin_dem"] = merged["fundamentals_margin_dem"].astype(float)
    merged["prior_sd"] = prior_sd
    merged["polling_sd"] = polling_sd

    prior_var = merged["prior_sd"] ** 2
    poll_var = merged["polling_sd"] ** 2

    merged["posterior_margin_dem"] = (
        merged["prior_margin_dem"] / prior_var
        + merged["polling_margin_used"] / poll_var
    ) / ((1 / prior_var) + (1 / poll_var))

    merged["posterior_sd"] = np.sqrt(1 / ((1 / prior_var) + (1 / poll_var)))
    merged["bayesian_polling_weight"] = (1 / poll_var) / ((1 / prior_var) + (1 / poll_var))
    merged["bayesian_prior_weight"] = 1 - merged["bayesian_polling_weight"]

    out_cols = [
        "state",
        "prior_margin_dem",
        "polling_margin_used",
        "posterior_margin_dem",
        "prior_sd",
        "polling_sd",
        "posterior_sd",
        "bayesian_prior_weight",
        "bayesian_polling_weight",
        "poll_count",
        "total_poll_weight",
        "avg_poll_age_days",
    ]
    output = merged[out_cols].copy()
    output.to_csv(input_dir / "bayesian_update_generated.csv", index=False)

    races_updated = races.merge(
        output[["state", "posterior_margin_dem", "bayesian_polling_weight", "posterior_sd"]],
        on="state",
        how="left",
    )

    races_updated["pre_bayes_model_margin_dem"] = races_updated["fundamentals_margin_dem"]
    # Robust posterior alias handling.
    # Some downstream scripts use posterior_margin_dem, while the Bayesian updater itself
    # may only have bayesian_model_margin_dem / posterior_margin_dem_capped depending on
    # pipeline order. Ensure the aliases exist before reading them.
    if "posterior_margin_dem" not in races_updated.columns:
        if "bayesian_model_margin_dem" in races_updated.columns:
            races_updated["posterior_margin_dem"] = races_updated["bayesian_model_margin_dem"]
        elif "posterior_margin_dem_capped" in races_updated.columns:
            races_updated["posterior_margin_dem"] = races_updated["posterior_margin_dem_capped"]
        else:
            races_updated["posterior_margin_dem"] = races_updated["fundamentals_margin_dem"]

    races_updated["bayesian_model_margin_dem"] = races_updated["posterior_margin_dem"].fillna(
        races_updated["fundamentals_margin_dem"]
    )
    races_updated["bayesian_polling_weight"] = races_updated["bayesian_polling_weight"].fillna(0)
    races_updated["bayesian_posterior_sd"] = races_updated["posterior_sd"].fillna(prior_sd)

    races_updated["fundamentals_margin_dem"] = races_updated["bayesian_model_margin_dem"]
    races_updated = races_updated.drop(columns=["posterior_margin_dem", "posterior_sd"], errors="ignore")
    # Do not overwrite race_inputs.csv; keep it as a human-edited input file.
    return output
