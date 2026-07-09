from pathlib import Path
import pandas as pd

INPUTS = Path("inputs")
OVERRIDES = INPUTS / "candidate_status_overrides.csv"
RACE_INPUTS = INPUTS / "race_inputs.csv"

if not OVERRIDES.exists():
    print("No candidate_status_overrides.csv found. Nothing to do.")
    raise SystemExit(0)

if not RACE_INPUTS.exists():
    raise SystemExit("Missing inputs/race_inputs.csv")

overrides = pd.read_csv(OVERRIDES)
races = pd.read_csv(RACE_INPUTS)

overrides["state"] = overrides["state"].astype(str).str.strip().str.upper()
races["state"] = races["state"].astype(str).str.strip().str.upper()

exclude_states = set(
    overrides[
        overrides["exclude_current_polling"]
        .astype(str)
        .str.strip()
        .str.upper()
        .isin(["TRUE", "1", "YES", "Y"])
    ]["state"]
)

if not exclude_states:
    print("No states marked for polling exclusion.")
    raise SystemExit(0)

mask = races["state"].isin(exclude_states)

na_cols = [
    "polling_margin_dem",
    "polling_margin_used",
    "latest_poll_end_date",
    "avg_poll_age_days",
    "largest_pollster_weight_share",
    "only_partisan_or_internal_polls",
    "recent_poll_count_45d",
    "most_recent_poll_end_date",
    "polling_confidence_weight_change",
    "polling_confidence_margin_change_dem",
]

zero_cols = [
    "poll_count",
    "effective_poll_count",
    "total_poll_weight",
    "polling_confidence_boost",
    "bayesian_polling_weight",
    "bayesian_polling_weight_capped",
    "bayesian_polling_weight_capped_before_polling_confidence_accelerator",
    "bayesian_polling_weight_capped_after_polling_confidence_accelerator",
]

for col in na_cols:
    if col in races.columns:
        races.loc[mask, col] = pd.NA

for col in zero_cols:
    if col in races.columns:
        races.loc[mask, col] = 0

if "polling_active" in races.columns:
    races.loc[mask, "polling_active"] = False

if "race_notes" in races.columns:
    races.loc[mask, "race_notes"] = (
        races.loc[mask, "race_notes"]
        .fillna("")
        .astype(str)
        + " Candidate-status override: current polling excluded due to replacement-pending event."
    )

if "dem_candidate" in races.columns:
    races.loc[mask, "dem_candidate"] = "Democratic nominee TBD"

races.to_csv(RACE_INPUTS, index=False)

print("Applied candidate-status overrides.")
print("Excluded current polling for:", ", ".join(sorted(exclude_states)))
