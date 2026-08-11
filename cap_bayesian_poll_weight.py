from pathlib import Path
from datetime import date
import argparse
import pandas as pd
import numpy as np

INPUT_DIR = Path("inputs")
ELECTION_DAY = date(2026, 11, 3)


def election_day_for_year(year):
    """Federal general election: first Tuesday after first Monday in November."""
    november_first = pd.Timestamp(year=int(year), month=11, day=1)
    first_monday_offset = (7 - november_first.weekday()) % 7
    first_monday = november_first + pd.Timedelta(days=first_monday_offset)
    return (first_monday + pd.Timedelta(days=1)).normalize()


def labor_day_for_year(year):
    """First Monday in September."""
    september_first = pd.Timestamp(year=int(year), month=9, day=1)
    offset = (7 - september_first.weekday()) % 7
    return (september_first + pd.Timedelta(days=offset)).normalize()


def resolve_replay_dates(days_out=None, as_of=None):
    """
    Resolve the effective model date and days-out value.

    Live production behavior is unchanged when neither argument is supplied.
    Historical replay can supply --as-of and/or --days-out explicitly.
    """
    if as_of is not None:
        effective_today = pd.Timestamp(as_of).normalize()
        election_day = election_day_for_year(effective_today.year)
    else:
        effective_today = pd.Timestamp(date.today()).normalize()
        election_day = pd.Timestamp(ELECTION_DAY).normalize()

    if days_out is None:
        resolved_days_out = max(
            0,
            int((election_day - effective_today).days),
        )
    else:
        resolved_days_out = max(0, float(days_out))

    return effective_today, election_day, resolved_days_out


def max_polling_weight_for_days_out(days_out):
    """
    Smoothly increase the maximum polling weight as Election Day approaches.

    Anchor points preserve the model's existing intended milestones:
      180 days out: 12%
      120 days out: 18%
       60 days out: 35%
       30 days out: 50%
        0 days out: 70%
    """
    try:
        days_out = float(days_out)
    except Exception:
        days_out = 180.0

    days_out = max(0.0, days_out)

    anchors = [
        (180.0, 0.12),
        (120.0, 0.18),
        (60.0, 0.35),
        (30.0, 0.50),
        (0.0, 0.70),
    ]

    if days_out >= anchors[0][0]:
        return anchors[0][1]

    for (older_day, older_weight), (newer_day, newer_weight) in zip(
        anchors[:-1],
        anchors[1:],
    ):
        if newer_day <= days_out <= older_day:
            progress = (older_day - days_out) / (older_day - newer_day)
            return older_weight + progress * (newer_weight - older_weight)

    return anchors[-1][1]


def poll_count_multiplier(poll_count):
    try:
        poll_count = float(poll_count)
    except Exception:
        poll_count = 0

    if poll_count <= 0:
        return 0.0
    if poll_count == 1:
        return 0.30
    if poll_count == 2:
        return 0.55
    if poll_count == 3:
        return 0.75
    return 1.0


def refresh_polling_metadata_from_generated(df):
    """
    Use inputs/polling_averages_generated.csv as the source of truth for poll_count,
    total_poll_weight, and avg_poll_age_days.

    This prevents stale values in race_inputs.csv from reducing polling weight after
    the poll ingestion step has correctly counted multiple manual polls.
    """
    from pathlib import Path
    import pandas as pd

    polling_path = Path("inputs/polling_averages_generated.csv")

    if not polling_path.exists():
        print("WARNING: inputs/polling_averages_generated.csv not found; using existing poll metadata.")
        return df

    polling = pd.read_csv(polling_path)

    if polling.empty:
        print("No generated polling averages found; using existing poll metadata.")
        return df

    required = ["state", "poll_count", "total_poll_weight", "avg_poll_age_days"]

    missing = [c for c in required if c not in polling.columns]

    if missing:
        print(f"WARNING: polling_averages_generated.csv missing columns: {missing}; using existing poll metadata.")
        return df

    polling = polling[required].copy()
    polling["state"] = polling["state"].astype(str).str.strip().str.upper()

    df = df.copy()
    df["state"] = df["state"].astype(str).str.strip().str.upper()

    df = df.merge(
        polling,
        on="state",
        how="left",
        suffixes=("", "_from_generated"),
    )

    for col in ["poll_count", "total_poll_weight", "avg_poll_age_days"]:
        generated_col = f"{col}_from_generated"

        if generated_col in df.columns:
            df[col] = df[generated_col].combine_first(df[col])
            df = df.drop(columns=[generated_col])

    return df


def minimum_uncertainty_for_days_out(days_out):
    if days_out > 180:
        return 7.5
    if days_out > 120:
        return 8.0
    if days_out > 60:
        return 5.5
    if days_out > 30:
        return 4.5
    return 3.5


def load_race_inputs():
    race_path = INPUT_DIR / "race_inputs.csv"

    if not race_path.exists():
        raise FileNotFoundError("Could not find inputs/race_inputs.csv")

    races = pd.read_csv(race_path)

    if "state" not in races.columns:
        raise ValueError("race_inputs.csv missing state column")

    races["state"] = races["state"].astype(str).str.strip().str.upper()

    return races


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--days-out",
        type=float,
        default=None,
        help="Override days until Election Day.",
    )
    parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        help="Historical/current model date in YYYY-MM-DD format.",
    )
    args = parser.parse_args()

    effective_today, election_day, days_out = resolve_replay_dates(
        days_out=args.days_out,
        as_of=args.as_of,
    )

    max_cycle_cap = max_polling_weight_for_days_out(days_out)
    min_sd = minimum_uncertainty_for_days_out(days_out)

    bayes_path = INPUT_DIR / "bayesian_update_generated.csv"

    if not bayes_path.exists():
        raise FileNotFoundError("Could not find inputs/bayesian_update_generated.csv")

    df = pd.read_csv(bayes_path)
    races = load_race_inputs()

    # Repair missing state column using race_inputs row order.
    if "state" not in df.columns:
        if len(df) != len(races):
            raise ValueError(
                "bayesian_update_generated.csv has no state column, and row count "
                f"does not match race_inputs.csv. bayes rows={len(df)}, race rows={len(races)}"
            )

        df.insert(0, "state", races["state"].values)
        print("Reconstructed missing state column from race_inputs.csv row order.")

    df["state"] = df["state"].astype(str).str.strip().str.upper()

    required_cols = [
        "prior_margin_dem",
        "polling_margin_used",
        "posterior_margin_dem",
        "posterior_sd",
        "bayesian_prior_weight",
        "bayesian_polling_weight",
        "poll_count",
    ]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"bayesian_update_generated.csv missing required columns: {missing}")

    print("\nCalibration settings:")
    print(f"days_out: {days_out}")
    print(f"cycle max polling cap: {max_cycle_cap}")
    print(f"minimum posterior SD: {min_sd}")

    df["prior_margin_dem"] = pd.to_numeric(
        df["prior_margin_dem"],
        errors="coerce"
    )

    df["polling_margin_used"] = pd.to_numeric(
        df["polling_margin_used"],
        errors="coerce"
    )

    df["posterior_margin_dem"] = pd.to_numeric(
        df["posterior_margin_dem"],
        errors="coerce"
    )

    df["posterior_sd"] = pd.to_numeric(
        df["posterior_sd"],
        errors="coerce"
    )

    df["bayesian_polling_weight"] = pd.to_numeric(
        df["bayesian_polling_weight"],
        errors="coerce"
    ).fillna(0.0)

    df["poll_count"] = pd.to_numeric(
        df["poll_count"],
        errors="coerce"
    ).fillna(0.0)

    df["original_bayesian_polling_weight"] = df["bayesian_polling_weight"]
    df["original_bayesian_posterior_sd"] = df["posterior_sd"]

    df["poll_count_weight_multiplier"] = df["poll_count"].apply(
        poll_count_multiplier
    )

    df["cycle_max_polling_weight"] = max_cycle_cap

    df["bayesian_polling_weight_capped"] = np.minimum(
        df["original_bayesian_polling_weight"],
        df["cycle_max_polling_weight"] * df["poll_count_weight_multiplier"]
    )

    df["bayesian_polling_weight_capped"] = (
        df["bayesian_polling_weight_capped"]
        .clip(lower=0.0, upper=1.0)
    )

    # If polling is missing or there are zero polls, polling should not affect the race.
    df.loc[
        df["polling_margin_used"].isna() | (df["poll_count"] <= 0),
        "bayesian_polling_weight_capped"
    ] = 0.0

    df["bayesian_fundamentals_weight_capped"] = (
        1.0 - df["bayesian_polling_weight_capped"]
    )

    df["bayesian_model_margin_dem_capped"] = (
        df["prior_margin_dem"] * df["bayesian_fundamentals_weight_capped"]
        + df["polling_margin_used"] * df["bayesian_polling_weight_capped"]
    )

    # Calibrated uncertainty floor.
    df["bayesian_posterior_sd_calibrated"] = (
        df["posterior_sd"]
        .fillna(0.0)
        .clip(lower=min_sd)
    )

    # Add a little extra uncertainty for RCV/top-four races using race_inputs mapping.
    if "election_system" in races.columns:
        election_map = dict(
            zip(
                races["state"],
                races["election_system"].fillna("plurality").astype(str).str.lower()
            )
        )

        df["election_system"] = df["state"].map(election_map).fillna("plurality")

        is_rcv = df["election_system"].str.contains("rcv", na=False)

        df.loc[
            is_rcv,
            "bayesian_posterior_sd_calibrated"
        ] = df.loc[
            is_rcv,
            "bayesian_posterior_sd_calibrated"
        ] + 0.75
    else:
        df["election_system"] = "plurality"

    # Replace operational columns.
    df["bayesian_polling_weight"] = df["bayesian_polling_weight_capped"]
    df["bayesian_prior_weight"] = df["bayesian_fundamentals_weight_capped"]
    df["posterior_margin_dem"] = df["bayesian_model_margin_dem_capped"]
    df["posterior_sd"] = df["bayesian_posterior_sd_calibrated"]

    df.to_csv(bayes_path, index=False)

    print(f"\nApplied early-cycle calibration to {bayes_path}")

    # Patch generated Bayesian fields into race_inputs without touching core fundamentals.
    margin_map = dict(zip(df["state"], df["posterior_margin_dem"]))
    weight_map = dict(zip(df["state"], df["bayesian_polling_weight"]))
    sd_map = dict(zip(df["state"], df["posterior_sd"]))

    races["bayesian_model_margin_dem"] = races["state"].map(margin_map)
    races["bayesian_polling_weight"] = races["state"].map(weight_map)
    races["bayesian_posterior_sd"] = races["state"].map(sd_map)

    race_path = INPUT_DIR / "race_inputs.csv"
    races.to_csv(race_path, index=False)

    print(f"Updated calibrated Bayesian fields in {race_path}")


if __name__ == "__main__":
    main()


# --- BEGIN POLLING CONFIDENCE ACCELERATOR FINALIZER ---
# Runs after the ordinary Bayesian cap step. This makes polling-confidence acceleration
# part of the official model-driving Bayesian fields, rather than a separate audit-only step.
if __name__ == "__main__":
    try:
        from pathlib import Path
        from datetime import datetime
        import pandas as pd

        INPUTS = Path("inputs")
        OUTPUTS = Path("outputs")
        OUTPUTS.mkdir(exist_ok=True)

        BAYES = INPUTS / "bayesian_update_generated.csv"
        RACES = INPUTS / "race_inputs.csv"
        POLL_FILES = [OUTPUTS / "senate_poll_weighting_live_audit.csv"]
        AUDIT = OUTPUTS / "senate_polling_confidence_accelerator_audit.csv"

        RECENT_DAYS = 45
        LABOR_DAY_2026 = pd.Timestamp("2026-09-07")
        PRE_LABOR_DAY_ABSOLUTE_CAP = 0.30
        POST_LABOR_DAY_ABSOLUTE_CAP = 0.45

        def safe_num(s, default=0.0):
            return pd.to_numeric(s, errors="coerce").fillna(default)

        def norm_state(x):
            if pd.isna(x):
                return ""
            return str(x).strip().upper()

        def state_from_race(x):
            if pd.isna(x):
                return ""
            txt = str(x).strip().upper()
            if len(txt) == 2 and txt.isalpha():
                return txt
            parts = txt.replace("_", "-").replace("/", "-").replace(" ", "-").split("-")
            for p in parts:
                p = p.strip()
                if len(p) == 2 and p.isalpha():
                    return p
            return ""

        def pick_col(df, names):
            for n in names:
                if n in df.columns:
                    return n
            return None

        def first_existing(paths):
            for p in paths:
                if p.exists():
                    return p
            return None

        poll_path = first_existing(POLL_FILES)

        if BAYES.exists() and RACES.exists() and poll_path is not None:
            bayes = pd.read_csv(BAYES)
            races = pd.read_csv(RACES)
            polls = pd.read_csv(poll_path)

            # Build recent poll counts.
            state_col = pick_col(polls, ["state", "State"])
            race_col = pick_col(polls, ["race", "Race", "contest", "Contest"])
            end_col = pick_col(polls, ["end_date", "poll_end_date", "field_end", "date", "poll_date"])
            sample_col = pick_col(polls, ["sample_size", "n", "Sample Size", "sample"])
            pollster_col = pick_col(polls, ["pollster", "Pollster"])

            if state_col:
                polls["state_norm"] = polls[state_col].apply(norm_state)
            elif race_col:
                polls["state_norm"] = polls[race_col].apply(state_from_race)
            else:
                polls["state_norm"] = ""

            if end_col:
                polls["poll_end_dt"] = pd.to_datetime(polls[end_col], errors="coerce")
            else:
                polls["poll_end_dt"] = pd.NaT

            if sample_col:
                polls["sample_size_num"] = safe_num(polls[sample_col], 0.0)
            else:
                polls["sample_size_num"] = 1.0

            if pollster_col:
                polls["has_named_pollster"] = (
                    polls[pollster_col]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .ne("")
                )
            else:
                polls["has_named_pollster"] = False

            usable = polls[
                polls["state_norm"].astype(str).str.len().eq(2)
                & polls["poll_end_dt"].notna()
                & polls["sample_size_num"].gt(0)
                & polls["has_named_pollster"]
            ].copy()

            today = effective_today
            cutoff = today - pd.Timedelta(days=RECENT_DAYS)

            recent = usable[usable["poll_end_dt"] >= cutoff].copy()

            if recent.empty:
                counts = pd.DataFrame(columns=[
                    "state",
                    "recent_poll_count_45d",
                    "most_recent_poll_end_date",
                    "polling_confidence_boost",
                ])
            else:
                counts = (
                    recent.groupby("state_norm")
                    .agg(
                        recent_poll_count_45d=("state_norm", "size"),
                        most_recent_poll_end_date=("poll_end_dt", "max"),
                    )
                    .reset_index()
                    .rename(columns={"state_norm": "state"})
                )

                def boost(n):
                    n = int(n)
                    if n >= 4:
                        return 0.09
                    if n == 3:
                        return 0.06
                    if n == 2:
                        return 0.03
                    return 0.0

                counts["polling_confidence_boost"] = counts["recent_poll_count_45d"].apply(boost)
                counts["most_recent_poll_end_date"] = counts["most_recent_poll_end_date"].dt.strftime("%Y-%m-%d")

            # Clean stale accelerator columns before merge.
            stale = [
                c for c in bayes.columns
                if (
                    "polling_confidence" in c
                    or "recent_poll_count_45d" in c
                    or "most_recent_poll_end_date" in c
                    or "after_polling_confidence_accelerator" in c
                    or "before_polling_confidence_accelerator" in c
                )
            ]
            bayes = bayes.drop(columns=stale, errors="ignore")

            bayes["state_norm"] = bayes["state"].apply(norm_state)

            merged = bayes.merge(
                counts,
                left_on="state_norm",
                right_on="state",
                how="left",
                suffixes=("", "_poll_counts"),
            )

            for col in ["recent_poll_count_45d", "polling_confidence_boost"]:
                if col not in merged.columns:
                    merged[col] = 0

            merged["recent_poll_count_45d"] = safe_num(merged["recent_poll_count_45d"], 0).astype(int)
            merged["polling_confidence_boost"] = safe_num(merged["polling_confidence_boost"], 0.0)

            historical_labor_day = labor_day_for_year(today.year)

            accelerator_floor_cap = (
                PRE_LABOR_DAY_ABSOLUTE_CAP
                if today < historical_labor_day
                else POST_LABOR_DAY_ABSOLUTE_CAP
            )

            accelerator_days_out = days_out
            accelerator_cycle_cap = max_polling_weight_for_days_out(
                accelerator_days_out
            )

            # The confidence accelerator may raise the early-cycle ceiling,
            # but it must never suppress the ordinary cycle cap later in the
            # campaign.
            absolute_cap = max(
                accelerator_floor_cap,
                accelerator_cycle_cap,
            )
            merged["polling_confidence_absolute_cap"] = absolute_cap

            if "bayesian_polling_weight_capped" not in merged.columns:
                merged["bayesian_polling_weight_capped"] = safe_num(merged.get("bayesian_polling_weight", 0.0), 0.0)

            base_weight = safe_num(merged["bayesian_polling_weight_capped"], 0.0).clip(lower=0.0, upper=1.0)
            merged["bayesian_polling_weight_capped_before_polling_confidence_accelerator"] = base_weight

            accelerated_weight = (base_weight + merged["polling_confidence_boost"]).clip(
                lower=0.0,
                upper=absolute_cap,
            )

            prior = safe_num(merged["prior_margin_dem"], 0.0)
            poll_margin = safe_num(merged["polling_margin_used"], 0.0)

            if "bayesian_model_margin_dem_capped" not in merged.columns:
                merged["bayesian_model_margin_dem_capped"] = prior * (1.0 - base_weight) + poll_margin * base_weight

            merged["bayesian_model_margin_dem_capped_before_polling_confidence_accelerator"] = safe_num(
                merged["bayesian_model_margin_dem_capped"], 0.0
            )

            accelerated_margin = prior * (1.0 - accelerated_weight) + poll_margin * accelerated_weight

            # Authoritative Bayesian fields.
            merged["bayesian_polling_weight"] = accelerated_weight
            merged["bayesian_polling_weight_capped"] = accelerated_weight
            merged["bayesian_prior_weight"] = 1.0 - accelerated_weight
            merged["bayesian_fundamentals_weight_capped"] = 1.0 - accelerated_weight

            merged["bayesian_model_margin_dem"] = accelerated_margin
            merged["bayesian_model_margin_dem_capped"] = accelerated_margin
            merged["posterior_margin_dem"] = accelerated_margin
            merged["posterior_margin_dem_capped"] = accelerated_margin

            merged["bayesian_polling_weight_capped_after_polling_confidence_accelerator"] = accelerated_weight
            merged["polling_confidence_weight_change"] = (
                merged["bayesian_polling_weight_capped_after_polling_confidence_accelerator"]
                - merged["bayesian_polling_weight_capped_before_polling_confidence_accelerator"]
            )
            merged["polling_confidence_margin_change_dem"] = (
                merged["bayesian_model_margin_dem_capped"]
                - merged["bayesian_model_margin_dem_capped_before_polling_confidence_accelerator"]
            )

            # Save Bayesian update file.
            bayes_out = merged.drop(columns=["state_norm", "state_poll_counts"], errors="ignore")
            bayes_out.to_csv(BAYES, index=False)

            # Sync race_inputs canonical fields.
            races["state_norm"] = races["state"].apply(norm_state)

            sync_cols = [
                "state_norm",
                "bayesian_polling_weight",
                "bayesian_polling_weight_capped",
                "bayesian_prior_weight",
                "bayesian_fundamentals_weight_capped",
                "bayesian_model_margin_dem",
                "bayesian_model_margin_dem_capped",
                "posterior_margin_dem",
                "posterior_margin_dem_capped",
                "polling_confidence_boost",
                "polling_confidence_absolute_cap",
                "polling_confidence_weight_change",
                "polling_confidence_margin_change_dem",
                "recent_poll_count_45d",
                "most_recent_poll_end_date",
                "bayesian_polling_weight_capped_before_polling_confidence_accelerator",
                "bayesian_polling_weight_capped_after_polling_confidence_accelerator",
                "bayesian_model_margin_dem_capped_before_polling_confidence_accelerator",
            ]
            sync_cols = [c for c in sync_cols if c in merged.columns]

            sync = merged[sync_cols].drop_duplicates("state_norm")

            # Drop stale fields in races so merge is clean.
            stale_races = [c for c in races.columns if c in sync_cols and c != "state_norm"]
            races = races.drop(columns=stale_races, errors="ignore")

            races = races.merge(sync, on="state_norm", how="left")
            races = races.drop(columns=["state_norm"], errors="ignore")
            races.to_csv(RACES, index=False)

            audit_cols = [
                "state_norm",
                "recent_poll_count_45d",
                "most_recent_poll_end_date",
                "polling_confidence_boost",
                "polling_confidence_absolute_cap",
                "bayesian_polling_weight_capped_before_polling_confidence_accelerator",
                "bayesian_polling_weight_capped_after_polling_confidence_accelerator",
                "polling_confidence_weight_change",
                "polling_confidence_margin_change_dem",
                "bayesian_model_margin_dem_capped_before_polling_confidence_accelerator",
                "bayesian_model_margin_dem_capped",
            ]
            audit_cols = [c for c in audit_cols if c in merged.columns]

            audit = (
                merged[audit_cols]
                .drop_duplicates("state_norm")
                .sort_values(["polling_confidence_weight_change", "recent_poll_count_45d"], ascending=[False, False])
            )
            audit.to_csv(AUDIT, index=False)

            print("Applied polling-confidence accelerator inside Bayesian cap step.")
            me = audit[audit["state_norm"].eq("ME")]
            if not me.empty:
                print("Maine polling-confidence accelerator:")
                print(me.to_string(index=False))

    except Exception as e:
        print(f"WARNING: polling-confidence accelerator finalizer failed: {e}")
# --- END POLLING CONFIDENCE ACCELERATOR FINALIZER ---

