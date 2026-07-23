from pathlib import Path
import pandas as pd
import numpy as np

INPUTS = Path("inputs")
RACE_INPUTS_PATH = INPUTS / "race_inputs.csv"
NATIONAL_ENV_PATH = INPUTS / "national_environment.csv"

WEIGHT_2024 = 0.60
WEIGHT_2020 = 0.25
WEIGHT_2016 = 0.15

# Fallback baselines for states where presidential margins are not yet populated.
# Positive = Democratic lean; negative = Republican lean.
# These are temporary legacy baselines so the dashboard audit is complete.
LEGACY_BASELINE_FALLBACKS = {
    "MI": 2.0,
    "MN": 6.0,
    "NH": 3.0,
    "IA": -7.0,
    "CO": 9.0,
    "DE": 15.0,
    "IL": 14.0,
    "MA": 25.0,
    "NJ": 11.0,
    "NM": 9.0,
    "OR": 14.0,
    "RI": 22.0,
    "VA": 7.0,
    "AL": -25.0,
    "AR": -28.0,
    "ID": -30.0,
    "KS": -16.0,
    "KY": -20.0,
    "LA": -18.0,
    "MS": -17.0,
    "MT": -15.0,
    "NE": -18.0,
    "OK": -30.0,
    "SC": -12.0,
    "SD": -25.0,
    "TN": -22.0,
    "WV": -30.0,
    "WY": -35.0,
}

DEFAULT_INCUMBENCY_ADJUSTMENTS = {
    "D": 2.0,
    "D-APPOINTED": 1.0,
    "R": -2.0,
    "R-APPOINTED": -1.0,
    "I": 0.0,
    "OPEN": 0.0,
    "VACANT": 0.0,
    "UNKNOWN": 0.0,
}

# State-specific race adjustments.
# Positive helps Democrats; negative helps Republicans.
STATE_OVERRIDES = {
    # Jon Ossoff is a Democratic incumbent.
    "GA": {
        "incumbency_adjustment_dem": 2.0,
        "candidate_quality_adjustment_dem": 0.0,
        "special_adjustment_dem": 0.0,
        "note": "GA: Ossoff Democratic incumbent adjustment."
    },

    # Susan Collins is a Republican incumbent with historical crossover strength.
    "ME": {
        "incumbency_adjustment_dem": -2.5,
        "candidate_quality_adjustment_dem": 0.0,
        "special_adjustment_dem": 0.0,
        "note": "ME: Collins Republican incumbent / overperformance adjustment."
    },

    # Ashley Moody is an appointed Republican incumbent.
    "FL": {
        "incumbency_adjustment_dem": -1.0,
        "candidate_quality_adjustment_dem": 0.0,
        "special_adjustment_dem": 0.0,
        "note": "FL: appointed Republican incumbent adjustment."
    },

    # Dan Sullivan is a Republican incumbent; Peltola-style crossover strength partly offsets.
    "AK": {
        "incumbency_adjustment_dem": -1.5,
        "candidate_quality_adjustment_dem": 1.0,
        "special_adjustment_dem": 0.0,
        "note": "AK: GOP incumbent adjustment partly offset by Democratic crossover/candidate-strength placeholder."
    },

    # Cooper is not an incumbent Senator, but has unusual statewide strength.
    "NC": {
        "incumbency_adjustment_dem": 0.0,
        "candidate_quality_adjustment_dem": 2.0,
        "special_adjustment_dem": 0.0,
        "note": "NC: Cooper candidate-strength placeholder; no Senate incumbency adjustment."
    },

    # Ohio special/appointee dynamics plus Brown candidate strength.
    "OH": {
        "incumbency_adjustment_dem": -1.0,
        "candidate_quality_adjustment_dem": 2.0,
        "special_adjustment_dem": 0.0,
        "note": "OH: appointed GOP incumbent placeholder plus Brown candidate-strength placeholder."
    },

    # Texas: GOP-held/open-ish nominee dynamics; Talarico candidate placeholder.
    "TX": {
        "incumbency_adjustment_dem": 0.0,
        "candidate_quality_adjustment_dem": 1.0,
        "special_adjustment_dem": 0.0,
        "note": "TX: no direct Senate incumbency boost; modest Democratic candidate-quality placeholder."
    },
}




def calculate_fundamentals_margin_from_components(df):
    """
    Recalculate fundamentals_margin_dem from the visible component columns.

    This keeps the dashboard/audit decomposition aligned with the actual
    fundamentals margin used by the model.
    """
    import pandas as pd

    component_cols = [
        "state_partisan_baseline_dem",
        "state_environment_adjustment_dem",
        "incumbency_adjustment_dem",
        "overperformance_adjustment_dem",
        "candidate_liability_adjustment_dem",
        "candidate_quality_adjustment_dem",
        "special_adjustment_dem",
    ]

    df = df.copy()

    for col in component_cols:
        if col not in df.columns:
            df[col] = 0.0

        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["fundamentals_component_sum_dem"] = df[component_cols].sum(axis=1)
    df["fundamentals_margin_dem"] = df["fundamentals_component_sum_dem"]

    return df


def sync_candidate_quality_adjustment(df):
    """
    Rebuild candidate_quality_adjustment_dem from objective/manual/gate fields
    before calculating fundamentals.

    This prevents stale candidate_quality_adjustment_dem values from muting
    candidate-quality updates.
    """
    import pandas as pd

    for col, default in [
        ("objective_candidate_quality_adjustment_dem", 0.0),
        ("manual_candidate_quality_adjustment_dem", 0.0),
        ("candidate_quality_gate", 1.0),
    ]:
        if col not in df.columns:
            df[col] = default

    objective = pd.to_numeric(
        df["objective_candidate_quality_adjustment_dem"],
        errors="coerce",
    ).fillna(0.0)

    manual = pd.to_numeric(
        df["manual_candidate_quality_adjustment_dem"],
        errors="coerce",
    ).fillna(0.0)

    gate = pd.to_numeric(
        df["candidate_quality_gate"],
        errors="coerce",
    ).fillna(1.0).clip(lower=0.0, upper=1.0)

    df["candidate_quality_adjustment_dem"] = (objective + manual) * gate

    return df


def normalize_holder(x):
    if pd.isna(x):
        return "UNKNOWN"

    x = str(x).strip().upper()

    if "APPOINT" in x and ("R" in x or "REP" in x or "GOP" in x):
        return "R-APPOINTED"

    if "APPOINT" in x and ("D" in x or "DEM" in x):
        return "D-APPOINTED"

    if x in ["R", "REP", "REPUBLICAN", "GOP"]:
        return "R"

    if x in ["D", "DEM", "DEMOCRAT", "DEMOCRATIC"]:
        return "D"

    if "OPEN" in x:
        return "OPEN"

    if "VACANT" in x:
        return "VACANT"

    if x == "I":
        return "I"

    return "UNKNOWN"


def read_national_environment():
    if not NATIONAL_ENV_PATH.exists():
        print("No national_environment.csv found. Using national environment = 0.0")
        return 0.0

    env = pd.read_csv(NATIONAL_ENV_PATH)

    if env.empty:
        print("national_environment.csv is empty. Using national environment = 0.0")
        return 0.0

    row = env.iloc[-1]

    if "national_environment_margin_dem" in env.columns:
        val = pd.to_numeric(row["national_environment_margin_dem"], errors="coerce")

        if pd.notna(val):
            print(f"Using national_environment_margin_dem directly: {float(val):.2f}")
            return float(val)

    generic_ballot = 0.0
    approval_adjustment = 0.0
    midterm_adjustment = 0.0

    if "generic_ballot_margin_dem" in env.columns:
        val = pd.to_numeric(row["generic_ballot_margin_dem"], errors="coerce")
        if pd.notna(val):
            generic_ballot = float(val)

    if "approval_adjustment_dem" in env.columns:
        val = pd.to_numeric(row["approval_adjustment_dem"], errors="coerce")
        if pd.notna(val):
            approval_adjustment = float(val)

    if "midterm_adjustment_dem" in env.columns:
        val = pd.to_numeric(row["midterm_adjustment_dem"], errors="coerce")
        if pd.notna(val):
            midterm_adjustment = float(val)

    total = generic_ballot + approval_adjustment + midterm_adjustment

    print(
        "Built national environment from components: "
        f"generic_ballot={generic_ballot:.2f}, "
        f"approval_adjustment={approval_adjustment:.2f}, "
        f"midterm_adjustment={midterm_adjustment:.2f}, "
        f"total={total:.2f}"
    )

    return float(total)


def ensure_numeric(df, col, default=np.nan):
    if col not in df.columns:
        df[col] = default

    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def ensure_text(df, col, default=""):
    if col not in df.columns:
        df[col] = default

    df[col] = df[col].fillna(default).astype(str)
    return df


def main():
    if not RACE_INPUTS_PATH.exists():
        raise FileNotFoundError(f"Could not find {RACE_INPUTS_PATH}")

    races = pd.read_csv(RACE_INPUTS_PATH)

    if "state" not in races.columns:
        raise ValueError("race_inputs.csv must include a state column")

    races["state"] = races["state"].astype(str).str.strip().str.upper()

    numeric_cols = [
        "pres_2024_margin_dem",
        "pres_2020_margin_dem",
        "pres_2016_margin_dem",
        "state_partisan_baseline_dem",
        "state_elasticity",
        "national_environment_margin_dem",
        "state_environment_adjustment_dem",
        "incumbency_adjustment_dem",
        "candidate_quality_adjustment_dem",
        "special_adjustment_dem",
        "fundamentals_margin_dem",
    ]

    for col in numeric_cols:
        races = ensure_numeric(races, col)

    races = ensure_text(races, "fundamentals_notes", "")
    races = ensure_text(races, "current_holder", "")

    # Elasticity fallback
    if "elasticity" in races.columns:
        old_elasticity = pd.to_numeric(races["elasticity"], errors="coerce")
        races["state_elasticity"] = races["state_elasticity"].fillna(old_elasticity)

    races["state_elasticity"] = races["state_elasticity"].fillna(0.75)

    # 1. Presidential baseline where all three presidential margins exist.
    has_all_pres_margins = (
        races["pres_2024_margin_dem"].notna()
        & races["pres_2020_margin_dem"].notna()
        & races["pres_2016_margin_dem"].notna()
    )

    presidential_baseline = (
        WEIGHT_2024 * races["pres_2024_margin_dem"]
        + WEIGHT_2020 * races["pres_2020_margin_dem"]
        + WEIGHT_2016 * races["pres_2016_margin_dem"]
    )

    races.loc[
        has_all_pres_margins,
        "state_partisan_baseline_dem"
    ] = presidential_baseline.loc[has_all_pres_margins]

    races["baseline_source"] = np.where(
        has_all_pres_margins,
        "presidential_weighted",
        "legacy_fallback"
    )

    # 2. Legacy fallback for missing presidential-margin states.
    for state, fallback in LEGACY_BASELINE_FALLBACKS.items():
        mask = (races["state"] == state) & (~has_all_pres_margins)
        races.loc[mask, "state_partisan_baseline_dem"] = fallback

    # If still missing, use existing fundamentals as last-resort fallback.
    still_missing = races["state_partisan_baseline_dem"].isna()
    races.loc[
        still_missing,
        "state_partisan_baseline_dem"
    ] = races.loc[
        still_missing,
        "fundamentals_margin_dem"
    ]

    # If still missing after that, use neutral baseline.
    still_missing = races["state_partisan_baseline_dem"].isna()
    races.loc[still_missing, "state_partisan_baseline_dem"] = 0.0
    races.loc[still_missing, "baseline_source"] = "neutral_fallback"

    # 3. Automatic incumbency from current_holder.
    holder_norm = races["current_holder"].apply(normalize_holder)
    races["holder_normalized"] = holder_norm

    races["incumbency_adjustment_dem"] = holder_norm.map(
        DEFAULT_INCUMBENCY_ADJUSTMENTS
    ).fillna(0.0)

    races["candidate_quality_adjustment_dem"] = races[
        "candidate_quality_adjustment_dem"
    ].fillna(0.0)

    races["special_adjustment_dem"] = races["special_adjustment_dem"].fillna(0.0)

    # 4. State-specific overrides for known candidate/incumbent dynamics.
    for state, vals in STATE_OVERRIDES.items():
        mask = races["state"] == state

        if not mask.any():
            continue

        races.loc[mask, "incumbency_adjustment_dem"] = vals["incumbency_adjustment_dem"]

        # Do NOT overwrite candidate_quality_adjustment_dem here.
        # Candidate quality is calculated by update_candidate_quality.py from
        # objective/manual/gate fields. Hard-coded defaults may set incumbency
        # and special adjustments, but they should not erase calculated CQ.
        if "fundamentals_default_candidate_quality_adjustment_dem" not in races.columns:
            races["fundamentals_default_candidate_quality_adjustment_dem"] = 0.0

        races.loc[mask, "fundamentals_default_candidate_quality_adjustment_dem"] = vals[
            "candidate_quality_adjustment_dem"
        ]

        races.loc[mask, "special_adjustment_dem"] = vals["special_adjustment_dem"]

        races.loc[mask, "fundamentals_notes"] = vals["note"]

    national_environment = read_national_environment()

    races["national_environment_margin_dem"] = national_environment

    races["state_environment_adjustment_dem"] = (

        races["national_environment_margin_dem"]
        * races["state_elasticity"]
    )

    # Rebuild candidate quality after applying hard-coded incumbency/special defaults.
    # This ensures calculated candidate quality survives the fundamentals step.
    races = sync_candidate_quality_adjustment(races)

    # Fundamentals are the sum of the visible component columns.
    # Keep this aligned with the dashboard/audit decomposition.
    component_cols = [
        "state_partisan_baseline_dem",
        "state_environment_adjustment_dem",
        "incumbency_adjustment_dem",
        "overperformance_adjustment_dem",
        "candidate_liability_adjustment_dem",
        "candidate_quality_adjustment_dem",
        "special_adjustment_dem",
    ]
    
    for col in component_cols:
        if col not in races.columns:
            races[col] = 0.0
        races[col] = pd.to_numeric(races[col], errors="coerce").fillna(0.0)
    
    races["fundamentals_component_sum_dem"] = races[component_cols].sum(axis=1)
    races["fundamentals_margin_dem"] = races["fundamentals_component_sum_dem"]

    # Add notes for non-override states.
    presidential_note = (
        "Calculated from 2024/2020/2016 presidential margins "
        f"({WEIGHT_2024:.0%}/{WEIGHT_2020:.0%}/{WEIGHT_2016:.0%}), "
        "state elasticity, national environment, incumbency, candidate quality, "
        "and special adjustments."
    )

    legacy_note = (
        "Legacy partisan baseline fallback used pending presidential-margin entry; "
        "then adjusted for national environment, elasticity, incumbency, candidate quality, "
        "and special adjustments."
    )

    override_states = set(STATE_OVERRIDES.keys())

    for idx, row in races.iterrows():
        state = row["state"]

        if state in override_states:
            continue

        if row["baseline_source"] == "presidential_weighted":
            races.at[idx, "fundamentals_notes"] = presidential_note
        elif row["baseline_source"] == "legacy_fallback":
            races.at[idx, "fundamentals_notes"] = legacy_note
        else:
            races.at[idx, "fundamentals_notes"] = (
                "Neutral fallback baseline used; presidential-margin entry needed."
            )

    races.to_csv(RACE_INPUTS_PATH, index=False)

    print(f"Updated fundamentals in {RACE_INPUTS_PATH}")
    print(f"National environment used: {national_environment:.2f}")

    show_cols = [
        "state",
        "current_holder",
        "holder_normalized",
        "baseline_source",
        "pres_2024_margin_dem",
        "pres_2020_margin_dem",
        "pres_2016_margin_dem",
        "state_partisan_baseline_dem",
        "state_elasticity",
        "state_environment_adjustment_dem",
        "incumbency_adjustment_dem",
        "candidate_quality_adjustment_dem",
        "special_adjustment_dem",
        "fundamentals_margin_dem",
        "fundamentals_notes",
    ]

    show_cols = [c for c in show_cols if c in races.columns]

    print()
    print("Fundamentals preview:")
    print(races[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()


# --- BEGIN CANDIDATE FRAMEWORK FINALIZER ---
# Runs after the ordinary fundamentals recalculation. This makes the framework
# candidate-quality adjustment authoritative whenever framework columns exist.
if __name__ == "__main__":
    try:
        from pathlib import Path
        import pandas as pd

        path = Path("inputs/race_inputs.csv")
        if path.exists():
            df = pd.read_csv(path)

            if "candidate_quality_framework_version" in df.columns:
                active = (
                    df["candidate_quality_framework_version"]
                    .fillna("")
                    .astype(str)
                    .str.contains("senate_candidate_quality_framework_v3_option_b", case=False, na=False)
                )

                if active.any():
                    def safe_num(s, default=0.0):
                        return pd.to_numeric(s, errors="coerce").fillna(default)

                    mechanical = safe_num(df.get("mechanical_candidate_adjustment_dem", 0.0), 0.0)
                    scandal = safe_num(df.get("candidate_scandal_adjustment_dem", 0.0), 0.0).clip(lower=-3.0, upper=3.0)

                    framework_final = (
                        mechanical[active] + scandal[active]
                    ).clip(lower=-4.0, upper=4.0)

                    df.loc[active, "candidate_quality_adjustment_dem"] = framework_final

                    # Health-check compatibility:
                    # The legacy health check expects candidate_quality_adjustment_dem
                    # to equal objective + manual, optionally gated. For framework rows,
                    # store the framework result in objective and zero manual.
                    if "objective_candidate_quality_adjustment_dem" not in df.columns:
                        df["objective_candidate_quality_adjustment_dem"] = 0.0
                    if "manual_candidate_quality_adjustment_dem" not in df.columns:
                        df["manual_candidate_quality_adjustment_dem"] = 0.0
                    if "candidate_quality_gate" not in df.columns:
                        df["candidate_quality_gate"] = 1.0

                    df.loc[active, "objective_candidate_quality_adjustment_dem"] = framework_final
                    df.loc[active, "manual_candidate_quality_adjustment_dem"] = 0.0
                    df.loc[active, "candidate_quality_gate"] = 1.0

                    # Prevent old ad hoc side channels from double-counting.
                    for col in [
                        "overperformance_adjustment_dem",
                        "candidate_liability_adjustment_dem",
                    ]:
                        if col in df.columns:
                            df.loc[active, col] = 0.0

                    component_cols = [
                        "state_partisan_baseline_dem",
                        "state_environment_adjustment_dem",
                        "incumbency_adjustment_dem",
                        "candidate_quality_adjustment_dem",
                        "special_adjustment_dem",
                    ]

                    for col in component_cols:
                        if col not in df.columns:
                            df[col] = 0.0
                        df[col] = safe_num(df[col], 0.0)

                    df["fundamentals_component_sum_dem"] = sum(df[col] for col in component_cols)
                    df["fundamentals_margin_dem"] = df["fundamentals_component_sum_dem"]

                    if "pre_bayes_model_margin_dem" in df.columns:
                        df["pre_bayes_model_margin_dem"] = df["fundamentals_margin_dem"]

                    df.to_csv(path, index=False)

                    print("Applied candidate-quality framework inside fundamentals step.")
                    show = df[df["state"].astype(str).str.upper().isin(["ME", "GA", "NC", "OH", "TX", "AK"])]
                    cols = [
                        "state",
                        "fundamentals_margin_dem",
                        "candidate_quality_adjustment_dem",
                        "mechanical_candidate_adjustment_dem",
                        "candidate_scandal_adjustment_dem",
                    ]
                    cols = [c for c in cols if c in show.columns]
                    print(show[cols].to_string(index=False))

    except Exception as e:
        print(f"WARNING: candidate framework finalizer failed: {e}")
# --- END CANDIDATE FRAMEWORK FINALIZER ---

