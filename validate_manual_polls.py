from pathlib import Path
import pandas as pd

from pollster_registry import apply_pollster_registry
import numpy as np

RAW_PATH = Path("inputs/manual_polls_adjusted.csv") if Path("inputs/manual_polls_adjusted.csv").exists() else Path("inputs/manual_polls.csv")
OUT_PATH = Path("outputs/manual_polls_clean.csv")

REQUIRED_COLUMNS = [
    "race",
    "state",
    "chamber",
    "pollster",
    "pollster_grade",
    "start_date",
    "end_date",
    "sample_size",
    "sample_type",
    "dem_candidate",
    "rep_candidate",
    "dem_pct",
    "rep_pct",
    "ind_pct",
    "other_pct",
    "undecided_pct",
]

PCT_COLUMNS = [
    "dem_pct",
    "rep_pct",
    "ind_pct",
    "other_pct",
    "undecided_pct",
]

POLLSTER_GRADE_WEIGHTS = {
    "A+": 1.15,
    "A": 1.10,
    "A-": 1.05,
    "B+": 1.00,
    "B": 0.95,
    "B-": 0.90,
    "C+": 0.80,
    "C": 0.70,
    "C-": 0.60,
    "D": 0.45,
    "Unknown": 0.75,
}


def pct_to_number(x):
    if pd.isna(x) or x == "":
        return 0.0
    return float(x)


def normalize_grade(x):
    if pd.isna(x) or str(x).strip() == "":
        return "Unknown"

    x = str(x).strip()

    if x not in POLLSTER_GRADE_WEIGHTS:
        return "Unknown"

    return x


def validate_columns(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def normalize_manual_polls():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Could not find {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)

    # Apply the canonical pollster registry before margin
    # calculations. The registry supplies the default pollster-wide
    # house effect for every matching poll.
    df = apply_pollster_registry(df)

    # Optional poll-level override. A blank value means "use the
    # pollster registry value." A numeric value replaces the registry
    # value for this individual poll only.
    if "manual_house_effect_adjustment_dem" in df.columns:
        poll_level_override = pd.to_numeric(
            df["manual_house_effect_adjustment_dem"],
            errors="coerce",
        )
    elif "house_effect_dem" in df.columns:
        # Backward compatibility for older manual-poll files.
        poll_level_override = pd.to_numeric(
            df["house_effect_dem"],
            errors="coerce",
        )
    else:
        poll_level_override = pd.Series(
            float("nan"),
            index=df.index,
            dtype=float,
        )

    registry_house_effect = pd.to_numeric(
        df["pollster_house_effect_dem"],
        errors="coerce",
    ).fillna(0.0)

    df["manual_house_effect_override_dem"] = (
        poll_level_override
    )

    df["effective_house_effect_dem"] = (
        poll_level_override.where(
            poll_level_override.notna(),
            registry_house_effect,
        )
    )

    df["house_effect_source"] = "pollster_registry"
    df.loc[
        poll_level_override.notna(),
        "house_effect_source",
    ] = "poll_level_override"

    # Preserve the established production alias consumed by the
    # weighting engine and historical downstream code.
    df["pollster_house_effect_dem"] = pd.to_numeric(
        df["effective_house_effect_dem"],
        errors="coerce",
    ).fillna(0.0)

    # Preserve this older alias temporarily for compatibility.
    df["house_effect_dem"] = df[
        "pollster_house_effect_dem"
    ]

    validate_columns(df)

    errors = []
    warnings = []

    for col in ["ind_candidate", "other_candidate", "notes"]:
        if col not in df.columns:
            df[col] = ""

    text_cols = [
        "race",
        "state",
        "chamber",
        "pollster",
        "sample_type",
        "dem_candidate",
        "rep_candidate",
        "ind_candidate",
        "other_candidate",
        "notes",
    ]

    for col in text_cols:
        df[col] = df[col].fillna("").astype(str).str.strip()

    df["state"] = df["state"].str.upper()
    df["pollster_grade"] = df["pollster_grade"].apply(normalize_grade)

    df["pollster_house_effect_dem"] = pd.to_numeric(
        df["pollster_house_effect_dem"],
        errors="coerce",
    ).fillna(0.0)

    df["house_effect_dem"] = df[
        "pollster_house_effect_dem"
    ]

    for col in PCT_COLUMNS:
        df[col] = df[col].apply(pct_to_number)

    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    df["sample_size"] = pd.to_numeric(df["sample_size"], errors="coerce")

    for idx, row in df.iterrows():
        row_num = idx + 2

        if pd.isna(row["start_date"]):
            errors.append(f"Row {row_num}: invalid start_date")

        if pd.isna(row["end_date"]):
            errors.append(f"Row {row_num}: invalid end_date")

        if pd.notna(row["start_date"]) and pd.notna(row["end_date"]):
            if row["end_date"] < row["start_date"]:
                errors.append(f"Row {row_num}: end_date before start_date")

        if pd.isna(row["sample_size"]) or row["sample_size"] <= 0:
            errors.append(f"Row {row_num}: invalid sample_size")

        pct_sum = sum(row[c] for c in PCT_COLUMNS)

        if pct_sum < 95 or pct_sum > 105:
            warnings.append(
                f"Row {row_num}: percentages sum to {pct_sum:.1f} "
                f"(race={row.get('race')}, state={row.get('state')}, pollster={row.get('pollster')}; "
                f"dem={row.get('dem_pct')}, rep={row.get('rep_pct')}, ind={row.get('ind_pct')}, "
                f"other={row.get('other_pct')}, undecided={row.get('undecided_pct')})"
            )

    duplicate_cols = [
        "race",
        "state",
        "pollster",
        "start_date",
        "end_date",
        "dem_candidate",
        "rep_candidate",
        "dem_pct",
        "rep_pct",
        "ind_pct",
        "other_pct",
        "undecided_pct",
    ]

    duplicate_mask = df.duplicated(subset=duplicate_cols, keep=False)

    if duplicate_mask.any():
        duplicate_rows = [str(i + 2) for i in df.index[duplicate_mask].tolist()]
        warnings.append("Possible duplicate polls on CSV rows: " + ", ".join(duplicate_rows))

    if errors:
        raise ValueError("Validation failed:\n" + "\n".join(errors))

    today = pd.Timestamp.today().normalize()

    df["mid_date"] = df["start_date"] + (df["end_date"] - df["start_date"]) / 2
    df["days_old"] = (today - df["mid_date"]).dt.days

    vote_cols = {
        "Dem": "dem_pct",
        "Rep": "rep_pct",
        "Ind": "ind_pct",
        "Other": "other_pct",
    }

    leaders = []
    leader_pcts = []
    second_pcts = []
    leader_margins = []

    for _, row in df.iterrows():
        vals = {label: row[col] for label, col in vote_cols.items()}
        sorted_vals = sorted(vals.items(), key=lambda x: x[1], reverse=True)

        leaders.append(sorted_vals[0][0])
        leader_pcts.append(sorted_vals[0][1])
        second_pcts.append(sorted_vals[1][1])
        leader_margins.append(sorted_vals[0][1] - sorted_vals[1][1])

    df["leader"] = leaders
    df["leader_pct"] = leader_pcts
    df["second_place_pct"] = second_pcts
    df["leader_margin"] = leader_margins

    df["dem_margin"] = df["dem_pct"] - df["rep_pct"]

    df["allocated_share"] = (
        df["dem_pct"]
        + df["rep_pct"]
        + df["ind_pct"]
        + df["other_pct"]
    )

    df["undecided_share"] = df["undecided_pct"]

    df["undecided_discount"] = (
        df["allocated_share"] / 100
    ).clip(lower=0.50, upper=1.00)

    # Preserve the poll's observed Democratic margin. A large undecided
    # share should affect uncertainty or poll weight, not mechanically
    # shrink the reported margin toward zero.
    df["undecided_adjusted_dem_margin"] = df["dem_margin"]

    df["reported_margin_dem"] = df["dem_margin"]

    df["house_effect_adjusted_dem_margin"] = (
        df["reported_margin_dem"]
        - df["pollster_house_effect_dem"]
    )

    df["final_poll_margin_dem"] = (
        df["house_effect_adjusted_dem_margin"]
    )

    df["dem_margin_vs_top_opponent"] = (
        df["dem_pct"]
        - df[["rep_pct", "ind_pct", "other_pct"]].max(axis=1)
    )

    df["rep_margin_vs_top_opponent"] = (
        df["rep_pct"]
        - df[["dem_pct", "ind_pct", "other_pct"]].max(axis=1)
    )

    df["ind_margin_vs_top_opponent"] = (
        df["ind_pct"]
        - df[["dem_pct", "rep_pct", "other_pct"]].max(axis=1)
    )

    df["is_three_way_race"] = df["ind_pct"] >= 10

    df["base_poll_weight"] = (
        np.sqrt(df["sample_size"])
        / (1 + (df["days_old"] / 30))
    )

    df["pollster_quality_weight"] = (
        df["pollster_grade"]
        .map(POLLSTER_GRADE_WEIGHTS)
        .fillna(POLLSTER_GRADE_WEIGHTS["Unknown"])
    )

    df["poll_weight"] = df["base_poll_weight"] * df["pollster_quality_weight"]

    clean_cols = [
        "race",
        "state",
        "chamber",
        "pollster",
        "pollster_grade",
        "sponsor",
        "poll_sponsor_type",
        "partisan_sponsor_party",
        "is_internal_poll",
        "pollster_partisan_affiliation",
        "partisan_pollster_review_notes",
        "start_date",
        "end_date",
        "mid_date",
        "days_old",
        "sample_size",
        "sample_type",
        "dem_candidate",
        "rep_candidate",
        "ind_candidate",
        "other_candidate",
        "dem_pct",
        "rep_pct",
        "ind_pct",
        "other_pct",
        "undecided_pct",
        "leader",
        "leader_pct",
        "second_place_pct",
        "leader_margin",
        "dem_margin",
        "allocated_share",
        "undecided_share",
        "undecided_discount",
        "undecided_adjusted_dem_margin",
        "pollster_raw",
        "pollster_normalized_key",
        "canonical_pollster",
        "pollster_match_method",
        "reported_margin_dem",
        "pollster_house_effect_dem",
        "manual_house_effect_override_dem",
        "effective_house_effect_dem",
        "house_effect_source",
        "house_effect_confidence",
        "house_effect_notes",
        "house_effect_adjusted_dem_margin",
        "final_poll_margin_dem",
        "dem_margin_vs_top_opponent",
        "rep_margin_vs_top_opponent",
        "ind_margin_vs_top_opponent",
        "is_three_way_race",
        "base_poll_weight",
        "pollster_quality_weight",
        "poll_weight",
        "notes",
    ]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df[clean_cols].to_csv(OUT_PATH, index=False)

    print(f"Saved clean manual polls to {OUT_PATH}")

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"- {w}")
    else:
        print("No validation warnings.")


if __name__ == "__main__":
    normalize_manual_polls()
