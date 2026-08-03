from pathlib import Path
import pandas as pd
import numpy as np

INPUTS = Path("inputs")
OUTPUTS = Path("outputs")
OUTPUTS.mkdir(exist_ok=True)

RACE_INPUTS = INPUTS / "race_inputs.csv"
WAR_AUDIT = OUTPUTS / "senate_candidate_war_audit.csv"
OUT = OUTPUTS / "senate_candidate_quality_framework_audit.csv"

# Framework v3 / Option B settings.
WAR_APPLICATION_MULTIPLIER = 0.50
MECHANICAL_CAP = 1.50
SCANDAL_CAP = 3.00
FINAL_CAP = 4.00
STATEWIDE_WIN_BONUS = 0.75


def safe_num(series, default=0.0):
    return pd.to_numeric(series, errors="coerce").fillna(default)


def clean_text(series):
    return series.fillna("").astype(str).replace({"nan": "", "None": "", "NaN": ""})


def safe_bool(series, default=False):
    return (
        series.fillna(default)
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes", "y"])
    )


def pick_col(df, candidates, default=None):
    for col in candidates:
        if col in df.columns:
            return col
    return default


def current_model_candidate_adjustment_from_races(races):
    legacy_cols = [
        "overperformance_adjustment_dem",
        "candidate_liability_adjustment_dem",
        "objective_candidate_quality_adjustment_dem",
        "manual_candidate_quality_adjustment_dem",
        "candidate_quality_adjustment_dem",
        "special_adjustment_dem",
    ]

    total = pd.Series(0.0, index=races.index)

    for col in legacy_cols:
        if col in races.columns:
            total += safe_num(races[col], 0.0)

    return total


def classify_recommendation(row):
    gap = row["framework_minus_current_adjustment_dem"]
    match_status = str(row.get("war_match_status", ""))

    if abs(gap) < 0.50:
        return "Keep current adjustment; framework difference is small."

    if match_status == "Neither matched" and row["statewide_win_bonus_dem"] == 0:
        return "No WAR match and no statewide winner bonus; framework removes old subjective adjustment."

    if gap < -0.50:
        return "Framework reduces Dem candidate-side adjustment."

    if gap > 0.50:
        return "Framework increases Dem candidate-side adjustment."

    return "Review."


def main():
    if not RACE_INPUTS.exists():
        raise FileNotFoundError(f"Missing {RACE_INPUTS}")

    if not WAR_AUDIT.exists():
        raise FileNotFoundError(
            f"Missing {WAR_AUDIT}. Run: python3 build_senate_candidate_war_audit.py"
        )

    races = pd.read_csv(RACE_INPUTS)
    war = pd.read_csv(WAR_AUDIT)

    for df in [races, war]:
        if "state" not in df.columns:
            raise ValueError("Both race_inputs.csv and senate_candidate_war_audit.csv must contain state.")
        df["state"] = df["state"].astype(str).str.upper().str.strip()

    if "proposed_war_adjustment_dem" not in war.columns:
        raise ValueError("senate_candidate_war_audit.csv must contain proposed_war_adjustment_dem.")

    races["current_model_candidate_adjustment_dem"] = current_model_candidate_adjustment_from_races(races)

    keep_war_cols = [
        "state",
        "dem_candidate",
        "gop_candidate",
        "war_match_status",
        "proposed_war_adjustment_dem",
        "war_review_priority",
        "dem_war_name",
        "gop_war_name",
        "dem_candidate_war",
        "gop_candidate_war",
        "candidate_war_net_dem",
        "dem_war_cycles",
        "gop_war_cycles",
    ]
    keep_war_cols = [c for c in keep_war_cols if c in war.columns]

    out = war[keep_war_cols].copy()

    out = out.merge(
        races[["state", "current_model_candidate_adjustment_dem"]].drop_duplicates("state"),
        on="state",
        how="left",
    )

    out["current_model_candidate_adjustment_dem"] = safe_num(
        out["current_model_candidate_adjustment_dem"], 0.0
    )
    out["proposed_war_adjustment_dem"] = safe_num(out["proposed_war_adjustment_dem"], 0.0)

    # WAR component.
    out["candidate_war_adjustment_dem_raw"] = out["proposed_war_adjustment_dem"]
    out["candidate_war_application_multiplier"] = WAR_APPLICATION_MULTIPLIER
    out["candidate_war_adjustment_dem"] = (
        out["candidate_war_adjustment_dem_raw"] * WAR_APPLICATION_MULTIPLIER
    )

    # Statewide winner and scandal fields from race_inputs.csv.
    framework_cols = [
        "state",
        "dem_previous_statewide_winner",
        "gop_previous_statewide_winner",
        "previous_statewide_winner_notes",
        "candidate_scandal_adjustment_dem",
        "candidate_scandal_rationale",
    ]
    framework_cols = [c for c in framework_cols if c in races.columns]

    if len(framework_cols) > 1:
        out = out.merge(
            races[framework_cols].drop_duplicates("state"),
            on="state",
            how="left",
        )

    if "dem_previous_statewide_winner" not in out.columns:
        out["dem_previous_statewide_winner"] = False
    if "gop_previous_statewide_winner" not in out.columns:
        out["gop_previous_statewide_winner"] = False

    out["dem_previous_statewide_winner"] = safe_bool(out["dem_previous_statewide_winner"], False)
    out["gop_previous_statewide_winner"] = safe_bool(out["gop_previous_statewide_winner"], False)

    out["statewide_win_bonus_dem"] = 0.0
    out.loc[
        out["dem_previous_statewide_winner"] & ~out["gop_previous_statewide_winner"],
        "statewide_win_bonus_dem",
    ] = STATEWIDE_WIN_BONUS
    out.loc[
        out["gop_previous_statewide_winner"] & ~out["dem_previous_statewide_winner"],
        "statewide_win_bonus_dem",
    ] = -STATEWIDE_WIN_BONUS

    if "candidate_scandal_adjustment_dem" in out.columns:
        out["candidate_scandal_adjustment_dem"] = safe_num(
            out["candidate_scandal_adjustment_dem"], 0.0
        ).clip(lower=-SCANDAL_CAP, upper=SCANDAL_CAP)
    else:
        out["candidate_scandal_adjustment_dem"] = 0.0

    if "candidate_scandal_rationale" not in out.columns:
        out["candidate_scandal_rationale"] = ""
    if "previous_statewide_winner_notes" not in out.columns:
        out["previous_statewide_winner_notes"] = ""

    out["mechanical_candidate_adjustment_dem"] = (
        out["candidate_war_adjustment_dem"] + out["statewide_win_bonus_dem"]
    ).clip(lower=-MECHANICAL_CAP, upper=MECHANICAL_CAP)

    out["framework_candidate_quality_adjustment_dem"] = (
        out["mechanical_candidate_adjustment_dem"]
        + out["candidate_scandal_adjustment_dem"]
    ).clip(lower=-FINAL_CAP, upper=FINAL_CAP)

    out["framework_minus_current_adjustment_dem"] = (
        out["framework_candidate_quality_adjustment_dem"]
        - out["current_model_candidate_adjustment_dem"]
    )
    out["abs_framework_gap"] = out["framework_minus_current_adjustment_dem"].abs()

    out["framework_candidate_quality_method"] = (
        "Framework v3 / Option B: mechanical_candidate_adjustment_dem = "
        "50% of Senate WAR-audit proposed adjustment + +/-0.75 previous statewide winner bonus, "
        "capped at +/-1.5. candidate_event_adjustment_dem comes from the shared registry and is capped at +/-3.0. "
        "Final candidate_quality_adjustment_dem = mechanical + candidate event, capped at +/-4.0."
    )

    out["framework_review_recommendation"] = out.apply(classify_recommendation, axis=1)

    rationale_cols = [
        "state",
        "candidate_quality_rationale",
        "human_review_status",
        "last_human_review_date",
    ]
    rationale_cols = [c for c in rationale_cols if c in races.columns]

    if len(rationale_cols) > 1:
        out = out.merge(
            races[rationale_cols].drop_duplicates("state"),
            on="state",
            how="left",
        )

    for col in [
        "dem_candidate",
        "gop_candidate",
        "war_match_status",
        "war_review_priority",
        "dem_war_name",
        "gop_war_name",
        "dem_war_cycles",
        "gop_war_cycles",
        "candidate_quality_rationale",
        "candidate_scandal_rationale",
        "previous_statewide_winner_notes",
        "human_review_status",
        "last_human_review_date",
    ]:
        if col in out.columns:
            out[col] = clean_text(out[col])

    priority_order = {"High": 3, "Medium": 2, "Low": 1}
    if "war_review_priority" in out.columns:
        out["war_review_priority_rank"] = out["war_review_priority"].map(priority_order).fillna(0)
        out = out.sort_values(["war_review_priority_rank", "abs_framework_gap"], ascending=[False, False])
    else:
        out = out.sort_values("abs_framework_gap", ascending=False)

    output_cols = [
        "state",
        "dem_candidate",
        "gop_candidate",
        "war_match_status",
        "war_review_priority",
        "current_model_candidate_adjustment_dem",
        "proposed_war_adjustment_dem",
        "candidate_war_adjustment_dem",
        "dem_previous_statewide_winner",
        "gop_previous_statewide_winner",
        "statewide_win_bonus_dem",
        "mechanical_candidate_adjustment_dem",
        "candidate_scandal_adjustment_dem",
        "candidate_scandal_rationale",
        "framework_candidate_quality_adjustment_dem",
        "framework_minus_current_adjustment_dem",
        "abs_framework_gap",
        "framework_review_recommendation",
        "dem_candidate_war",
        "gop_candidate_war",
        "candidate_war_net_dem",
        "dem_war_name",
        "gop_war_name",
        "dem_war_cycles",
        "gop_war_cycles",
        "previous_statewide_winner_notes",
        "candidate_quality_rationale",
        "human_review_status",
        "last_human_review_date",
        "framework_candidate_quality_method",
    ]
    output_cols = [c for c in output_cols if c in out.columns]

    out[output_cols].to_csv(OUT, index=False)

    print(f"Wrote {OUT}")
    print(f"Rows: {len(out)}")
    print()
    print("Largest framework/current gaps:")
    print(out[output_cols].head(20).to_string(index=False))

    print()
    print("Georgia:")
    ga = out[out["state"].eq("GA")]
    print(ga[output_cols].to_string(index=False) if not ga.empty else "No GA row found.")

    print()
    print("Maine:")
    me = out[out["state"].eq("ME")]
    print(me[output_cols].to_string(index=False) if not me.empty else "No ME row found.")


if __name__ == "__main__":
    main()
