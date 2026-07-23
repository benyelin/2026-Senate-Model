
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import json
import numpy as np
import pandas as pd


@dataclass
class ModelConfig:
    election_date: str = "2026-11-03"
    today: str | None = None
    n_sims: int = 20000
    dem_baseline_seats: int = 34
    control_threshold: int = 51
    seed: int = 20260522
    default_total_error_sd: float = 6.0
    default_correlation: float = 0.78
    probability_scale: float = 5.5


def load_inputs(input_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    input_dir = Path(input_dir)
    races = pd.read_csv(input_dir / "race_inputs.csv")
    national = pd.read_csv(input_dir / "national_environment.csv")
    calibration = pd.read_csv(input_dir / "calibration_parameters.csv")

    required = {
        "state",
        "dem_candidate",
        "gop_candidate",
        "current_holder",
        "race_tier",
        "polling_margin_dem",
        "fundamentals_margin_dem",
        "elasticity",
        "tier_error_multiplier",
        "dem_win_counts_for_seat_change",
    }
    missing = required - set(races.columns)
    if missing:
        raise ValueError(f"race_inputs.csv missing columns: {sorted(missing)}")
    return races, national, calibration


def _get_scalar(df: pd.DataFrame, key: str, default=None):
    if {"parameter", "value"}.issubset(df.columns):
        rows = df.loc[df["parameter"].astype(str).str.lower() == key.lower(), "value"]
        if not rows.empty:
            val = rows.iloc[0]
            try:
                return float(val)
            except Exception:
                return val
    return default


def _interp(calibration: pd.DataFrame, days_out: float, column: str, default: float) -> float:
    if "days_out" not in calibration.columns or column not in calibration.columns:
        return default
    curve = calibration[["days_out", column]].dropna().sort_values("days_out")
    if curve.empty:
        return default
    x = curve["days_out"].astype(float).to_numpy()
    y = curve[column].astype(float).to_numpy()
    return float(np.interp(float(days_out), x, y))


def compute_days_out(config: ModelConfig) -> int:
    election = pd.Timestamp(config.election_date)
    today = pd.Timestamp(config.today) if config.today else pd.Timestamp.today().normalize()
    return max(0, int((election - today).days))


def national_environment_margin(national: pd.DataFrame) -> float:
    gb = float(_get_scalar(national, "generic_ballot_dem_margin", 0.0))
    approval = float(_get_scalar(national, "presidential_approval", 50.0))
    approval_slope = float(_get_scalar(national, "approval_slope", 0.25))
    midterm = float(_get_scalar(national, "midterm_effect_dem", 0.0))
    manual = float(_get_scalar(national, "manual_adjustment", 0.0))
    return gb + ((50.0 - approval) * approval_slope) + midterm + manual


def prepare_race_table(
    races: pd.DataFrame,
    national: pd.DataFrame,
    calibration: pd.DataFrame,
    config: ModelConfig,
) -> pd.DataFrame:
    days_out = compute_days_out(config)

    polling_weight = _interp(calibration, days_out, "polling_weight", default=0.50)
    polling_weight = max(0.0, min(1.0, polling_weight))
    fundamentals_weight = 1.0 - polling_weight
    national_env = national_environment_margin(national)

    out = races.copy()
    out["days_out"] = days_out
    out["polling_weight"] = polling_weight
    out["fundamentals_weight"] = fundamentals_weight
    out["national_environment_margin"] = national_env
    out["state_environment_adjustment"] = out["elasticity"].astype(float) * national_env
    out["adjusted_fundamentals_margin_dem"] = (
        out["fundamentals_margin_dem"].astype(float) + out["state_environment_adjustment"]
    )
    out["model_margin_dem"] = (
        out["polling_margin_dem"].astype(float) * polling_weight
        + out["adjusted_fundamentals_margin_dem"].astype(float) * fundamentals_weight
    )
    out["pre_sim_dem_win_prob"] = 1 / (1 + np.exp(-out["model_margin_dem"] / config.probability_scale))
    return out


def run_forecast(
    input_dir: str | Path = "inputs",
    output_dir: str | Path = "outputs",
    config: ModelConfig | None = None,
) -> Dict[str, pd.DataFrame | dict]:
    config = config or ModelConfig()
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    races, national, calibration = load_inputs(input_dir)
    race_table = prepare_race_table(races, national, calibration, config)

    days_out = compute_days_out(config)
    total_error_sd = _interp(calibration, days_out, "total_error_sd", default=config.default_total_error_sd)
    corr = _interp(calibration, days_out, "correlation", default=config.default_correlation)
    corr = max(0.0, min(0.98, corr))

    national_sd = total_error_sd * np.sqrt(corr)
    race_sd = total_error_sd * np.sqrt(1.0 - corr)

    rng = np.random.default_rng(config.seed)
    n_races = len(race_table)
    tier_multiplier = race_table['tier_error_multiplier'].astype(float).to_numpy() if 'tier_error_multiplier' in race_table.columns else np.ones(n_races)

    national_error = rng.normal(0.0, national_sd, size=(config.n_sims, 1))
    race_error = rng.normal(0.0, race_sd * tier_multiplier.reshape(1, n_races), size=(config.n_sims, n_races))

    base_margins = race_table["model_margin_dem"].to_numpy(dtype=float).reshape(1, n_races)
    simulated_margins = base_margins + national_error + race_error
    dem_wins = simulated_margins > 0

    seat_value = race_table["dem_win_counts_for_seat_change"].to_numpy(dtype=float).reshape(1, n_races)
    dem_seats = config.dem_baseline_seats + (dem_wins * seat_value).sum(axis=1)
    control = dem_seats >= config.control_threshold

    summary = {
        "n_sims": config.n_sims,
        "days_out": days_out,
        "expected_dem_seats": float(np.mean(dem_seats)),
        "median_dem_seats": float(np.median(dem_seats)),
        "dem_control_probability": float(np.mean(control)),
        "dem_control_threshold": config.control_threshold,
        "dem_baseline_seats": config.dem_baseline_seats,
        "total_error_sd": float(total_error_sd),
        "national_error_sd": float(national_sd),
        "race_error_sd": float(race_sd),
        "implied_correlation": float(corr),
        "national_environment_margin": float(race_table["national_environment_margin"].iloc[0]),
        "polling_weight": float(race_table["polling_weight"].iloc[0]),
        "fundamentals_weight": float(race_table["fundamentals_weight"].iloc[0]),
    }

    race_stats = race_table[[
        "state",
        "dem_candidate",
        "gop_candidate",
        "current_holder",
        "race_tier",
        "polling_margin_dem",
        "fundamentals_margin_dem",
        "adjusted_fundamentals_margin_dem",
        "model_margin_dem",
        "pre_sim_dem_win_prob",
        "elasticity",
        "tier_error_multiplier",
        "dem_win_counts_for_seat_change",
    ]].copy()

    race_stats["simulated_dem_win_prob"] = dem_wins.mean(axis=0)
    race_stats["avg_simulated_margin_dem"] = simulated_margins.mean(axis=0)

    # Tipping-ish diagnostic: among control simulations, identify the closest positive
    # Democratic pickup margin. Safely skip simulations with no positive pickup.
    pickup_mask = race_table["dem_win_counts_for_seat_change"].to_numpy(dtype=float) > 0
    tipping_counts = np.zeros(n_races, dtype=int)

    if pickup_mask.any() and control.any():
        pickup_margins = simulated_margins[:, pickup_mask]
        pickup_indices = np.flatnonzero(pickup_mask)

        for sim_idx in np.flatnonzero(control):
            margins = pickup_margins[sim_idx]
            positive = margins > 0
            if not positive.any():
                continue
            local_idx = np.argmin(np.where(positive, margins, np.inf))
            race_idx = pickup_indices[local_idx]
            tipping_counts[race_idx] += 1

    race_stats["tipping_count"] = tipping_counts
    race_stats["tipping_share_of_control_sims"] = (
        race_stats["tipping_count"] / max(int(control.sum()), 1)
    )

    seat_distribution = (
        pd.Series(dem_seats, name="dem_seats")
        .value_counts(normalize=True)
        .sort_index()
        .rename_axis("dem_seats")
        .reset_index(name="probability")
    )
    seat_distribution["count"] = (seat_distribution["probability"] * config.n_sims).round().astype(int)

    simulation_draws = pd.DataFrame({
        "simulation": np.arange(1, config.n_sims + 1),
        "dem_seats": dem_seats,
        "dem_control": control.astype(int),
        "national_error": national_error[:, 0],
    })

    race_stats.to_csv(output_dir / "race_stats.csv", index=False)
    seat_distribution.to_csv(output_dir / "seat_distribution.csv", index=False)
    simulation_draws.to_csv(output_dir / "simulation_draws.csv", index=False)
    pd.DataFrame([summary]).to_csv(output_dir / "forecast_summary.csv", index=False)

    with open(output_dir / "forecast_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {
        "summary": summary,
        "race_stats": race_stats,
        "seat_distribution": seat_distribution,
        "simulation_draws": simulation_draws,
    }
