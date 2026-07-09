from datetime import date
from pathlib import Path

import pandas as pd

from poll_weighting_engine import weight_polls


ELECTION_DATE = pd.Timestamp("2026-11-03")


def update_race_inputs_from_polls(
    input_dir="inputs",
    output_dir="outputs",
    as_of=None,
    as_of_date=None,
    half_life_days=None,
):
    """
    Create Senate polling averages using the shared poll-weighting engine.

    Reads:
        outputs/manual_polls_clean.csv

    Writes:
        inputs/polling_averages_generated.csv
        outputs/senate_poll_weighting_live_audit.csv

    The engine applies:
        - recency weighting
        - pollster-grade weighting
        - sample-size weighting
        - population weighting
        - partisan-sponsor adjustment and downweighting
        - exact-poll deduplication
        - same-pollster concentration limits

    Methodology is intentionally not used as a weighting factor.
    Environment translation remains disabled until a dated national-
    environment history has been built.
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    manual_path = output_dir / "manual_polls_clean.csv"
    out_path = input_dir / "polling_averages_generated.csv"
    audit_path = output_dir / "senate_poll_weighting_live_audit.csv"

    output_columns = [
        "state",
        "polling_margin_dem",
        "poll_count",
        "effective_poll_count",
        "latest_poll_end_date",
        "avg_poll_age_days",
        "total_poll_weight",
        "largest_pollster_weight_share",
        "only_partisan_or_internal_polls",
        "max_absolute_environment_translation",
    ]

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not manual_path.exists():
        empty = pd.DataFrame(columns=output_columns)
        empty.to_csv(out_path, index=False)
        print("No manual polls found. Wrote empty polling averages.")
        return empty

    polls = pd.read_csv(manual_path)

    if polls.empty:
        empty = pd.DataFrame(columns=output_columns)
        empty.to_csv(out_path, index=False)
        print("Manual polls file is empty. Wrote empty polling averages.")
        return empty

    required = ["state", "pollster", "dem_pct", "rep_pct", "end_date"]

    missing = [column for column in required if column not in polls.columns]
    if missing:
        raise ValueError(f"Manual polls file missing required columns: {missing}")

    polls["state"] = polls["state"].astype(str).str.strip().str.upper()
    polls["end_date"] = pd.to_datetime(polls["end_date"], errors="coerce")

    resolved_as_of = as_of_date if as_of_date is not None else as_of

    if resolved_as_of is None:
        as_of_timestamp = pd.Timestamp(date.today())
    else:
        as_of_timestamp = pd.Timestamp(resolved_as_of)

    days_out = max(0, int((ELECTION_DATE - as_of_timestamp).days))

    # Prefer validator-computed age when present, but recompute when absent.
    if "days_old" in polls.columns:
        polls["poll_age_days"] = pd.to_numeric(
            polls["days_old"],
            errors="coerce",
        )
    else:
        polls["poll_age_days"] = pd.NA

    calculated_age = (as_of_timestamp - polls["end_date"]).dt.days.clip(lower=0)

    polls["poll_age_days"] = polls["poll_age_days"].fillna(
        calculated_age
    ).clip(lower=0)

    # Environment translation is deliberately inactive for now.
    polls["current_national_environment_dem"] = pd.NA
    polls["national_environment_at_poll_date_dem"] = pd.NA
    polls["race_environment_sensitivity"] = 1.0

    weighted_polls, race_averages = weight_polls(
        polls,
        race_col="state",
        pollster_col="pollster",
        days_out=days_out,
    )

    weighted_polls.to_csv(audit_path, index=False)

    latest_dates = (
        weighted_polls
        .groupby("state", as_index=False)["end_date"]
        .max()
        .rename(columns={"end_date": "latest_poll_end_date"})
    )

    race_averages = race_averages.merge(
        latest_dates,
        on="state",
        how="left",
    )

    race_averages = race_averages.rename(
        columns={
            "weighted_avg_poll_age_days": "avg_poll_age_days",
        }
    )

    race_averages = race_averages[
        [column for column in output_columns if column in race_averages.columns]
    ].sort_values("state")

    race_averages.to_csv(out_path, index=False)

    print(f"Wrote new-engine polling averages to {out_path}")
    print(f"Wrote detailed poll audit to {audit_path}")
    print(f"Days out used for recency weighting: {days_out}")

    return race_averages
