from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parent
RACE_INPUTS = BASE / "inputs" / "race_inputs.csv"
OVERRIDES = BASE / "inputs" / "candidate_status_overrides.csv"

def truthy(value) -> bool:
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}

def clean_str(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return text

def main():
    if not RACE_INPUTS.exists():
        raise FileNotFoundError(f"Missing {RACE_INPUTS}")

    if not OVERRIDES.exists():
        print(f"No candidate status overrides found at {OVERRIDES}; skipping.")
        return

    races = pd.read_csv(RACE_INPUTS)
    overrides = pd.read_csv(OVERRIDES)

    if "state" not in races.columns:
        raise ValueError("race_inputs.csv must contain a 'state' column.")

    if "state" not in overrides.columns:
        raise ValueError("candidate_status_overrides.csv must contain a 'state' column.")

    races["state"] = races["state"].astype(str).str.strip().str.upper()
    overrides["state"] = overrides["state"].astype(str).str.strip().str.upper()

    required_defaults = {
        "dem_candidate": "",
        "gop_candidate": "",
        "polling_margin_dem": np.nan,
        "polling_margin_used": np.nan,
        "poll_count": 0,
        "polling_active": False,
        "candidate_status": "",
        "candidate_uncertainty_penalty": 0.0,
        "total_poll_weight": 0.0,
        "bayesian_polling_weight": 0.0,
    }

    for col, default in required_defaults.items():
        if col not in races.columns:
            races[col] = default

    changed_states = []

    for _, override in overrides.iterrows():
        state = override["state"]
        mask = races["state"].eq(state)

        if not mask.any():
            print(f"Warning: override state {state} not found in race_inputs.csv; skipping.")
            continue

        status = clean_str(override.get("candidate_status", ""))
        status_lower = status.lower()

        exclude_polling = truthy(override.get("exclude_current_polling", False))

        try:
            penalty = float(override.get("uncertainty_penalty", 0))
        except Exception:
            penalty = 0.0

        dem_override = clean_str(override.get("dem_candidate_override", ""))
        gop_override = clean_str(override.get("gop_candidate_override", ""))

        if status:
            races.loc[mask, "candidate_status"] = status

        races.loc[mask, "candidate_uncertainty_penalty"] = penalty

        # Explicit override columns win if present.
        if dem_override:
            races.loc[mask, "dem_candidate"] = dem_override

        if gop_override:
            races.loc[mask, "gop_candidate"] = gop_override

        # Backward-compatible defaults.
        if status_lower in {"replacement_pending", "dem_replacement_pending", "democratic_replacement_pending"}:
            races.loc[mask, "dem_candidate"] = "Democratic nominee TBD"

        if status_lower in {"gop_replacement_pending", "republican_replacement_pending"}:
            races.loc[mask, "gop_candidate"] = "Republican nominee TBD"

        replacement_pending = status_lower in {
            "replacement_pending",
            "dem_replacement_pending",
            "democratic_replacement_pending",
            "gop_replacement_pending",
            "republican_replacement_pending",
        }

        if exclude_polling or replacement_pending:
            # Core polling fields.
            races.loc[mask, "polling_margin_dem"] = np.nan
            races.loc[mask, "polling_margin_used"] = np.nan
            races.loc[mask, "poll_count"] = 0
            races.loc[mask, "polling_active"] = False
            races.loc[mask, "total_poll_weight"] = 0.0

            # Clear stale generated/cached polling and Bayesian polling columns.
            for col in races.columns:
                lower = col.lower()

                should_zero = (
                    "polling_weight" in lower
                    or "poll_weight" in lower
                    or "bayesian_polling_weight" in lower
                    or lower == "total_poll_weight"
                )

                should_nan = (
                    "polling_margin" in lower
                    or "polling_contribution" in lower
                    or ("model_margin_dem" in lower and "polling" in lower)
                )

                if col in {"polling_margin_dem", "polling_margin_used"}:
                    continue

                if should_zero:
                    races.loc[mask, col] = 0.0

                if should_nan:
                    races.loc[mask, col] = np.nan

        changed_states.append(state)

    races.to_csv(RACE_INPUTS, index=False)

    if changed_states:
        print("Applied candidate status overrides for:", ", ".join(sorted(set(changed_states))))
    else:
        print("No candidate status overrides applied.")

if __name__ == "__main__":
    main()
