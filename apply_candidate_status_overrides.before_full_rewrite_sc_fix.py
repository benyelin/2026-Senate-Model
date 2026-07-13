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

    # Normalize state abbreviations.
    races["state"] = races["state"].astype(str).str.strip().str.upper()
    overrides["state"] = overrides["state"].astype(str).str.strip().str.upper()

    # Ensure columns exist before writing to them.
    for col in ["polling_margin_dem", "polling_margin_used"]:
        if col not in races.columns:
            races[col] = np.nan

    if "poll_count" not in races.columns:
        races["poll_count"] = 0

    if "polling_active" not in races.columns:
        races["polling_active"] = False

    if "candidate_status" not in races.columns:
        races["candidate_status"] = ""

    if "candidate_uncertainty_penalty" not in races.columns:
        races["candidate_uncertainty_penalty"] = 0.0

    if "dem_candidate" not in races.columns:
        races["dem_candidate"] = ""

    changed_states = []

    for _, override in overrides.iterrows():
        state = override["state"]
        mask = races["state"].eq(state)

        if not mask.any():
            print(f"Warning: override state {state} not found in race_inputs.csv; skipping.")
            continue

        status = str(override.get("candidate_status", "")).strip()
        exclude_polling = truthy(override.get("exclude_current_polling", False))
        penalty = override.get("uncertainty_penalty", 0)

        try:
            penalty = float(penalty)
        except Exception:
            penalty = 0.0

        if status:
            races.loc[mask, "candidate_status"] = status

        if penalty:
            races.loc[mask, "candidate_uncertainty_penalty"] = penalty

        # Special handling for replacement-pending races.
        if status.lower() == "replacement_pending":
            races.loc[mask, "dem_candidate"] = "Democratic nominee TBD"

        # This is the key stale-polling cleanup.
        if exclude_polling or status.lower() == "replacement_pending":
            races.loc[mask, "polling_margin_dem"] = np.nan
            races.loc[mask, "polling_margin_used"] = np.nan
            races.loc[mask, "poll_count"] = 0
            races.loc[mask, "polling_active"] = False

        changed_states.append(state)

    races.to_csv(RACE_INPUTS, index=False)

    if changed_states:
        print("Applied candidate status overrides for:", ", ".join(sorted(set(changed_states))))
    else:
        print("No candidate status overrides applied.")

if __name__ == "__main__":
    main()
