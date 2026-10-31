from pathlib import Path
import pandas as pd
import numpy as np

INPUT = Path("inputs/manual_polls.csv")
OUTPUT = Path("inputs/manual_polls_adjusted.csv")

SETTINGS_CANDIDATES = [
    Path("inputs/senate_calibration_settings.csv"),
    Path("inputs/calibration_settings.csv"),
]


def read_settings():
    settings_path = next((p for p in SETTINGS_CANDIDATES if p.exists()), None)

    if settings_path is None:
        return {}

    s = pd.read_csv(settings_path)
    out = {}

    if "setting" not in s.columns or "value" not in s.columns:
        return out

    for _, row in s.iterrows():
        key = str(row.get("setting", "")).strip()
        try:
            out[key] = float(row.get("value"))
        except Exception:
            pass

    return out


def setting(settings, key, default):
    return float(settings.get(key, default))


def norm(x):
    return str(x).strip().lower()


def boolish(x):
    return norm(x) in ["1", "true", "yes", "y", "internal", "campaign", "campaign internal"]


def infer_party(row):
    for col in ["partisan_sponsor_party", "sponsor_party", "pollster_partisan_affiliation"]:
        if col in row.index:
            v = norm(row.get(col, ""))
            if v in ["d", "dem", "democratic", "democrat"]:
                return "D"
            if v in ["r", "rep", "republican", "gop"]:
                return "R"
    return ""


def infer_internal(row):
    if "is_internal_poll" in row.index and boolish(row.get("is_internal_poll")):
        return True

    sponsor_type = norm(row.get("poll_sponsor_type", ""))
    return sponsor_type in [
        "internal",
        "campaign",
        "campaign internal",
        "candidate",
        "party",
        "party committee",
    ]


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"{INPUT} not found.")

    settings = read_settings()

    use_adjustments = setting(settings, "use_partisan_pollster_adjustments", 1) >= 0.5
    default_adj = setting(settings, "partisan_pollster_default_adjustment", 2.0)
    internal_adj = setting(settings, "partisan_pollster_internal_adjustment", 2.5)
    max_adj = setting(settings, "partisan_pollster_max_adjustment", 3.0)
    weight_multiplier = setting(settings, "partisan_pollster_weight_multiplier", 0.50)

    df = pd.read_csv(INPUT)

    # Fail closed if an excluded pollster reappears.
    if "pollster" not in df.columns:
        raise ValueError(
            "Senate polling input lacks pollster column."
        )

    excluded_mask = (
        df["pollster"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"wedgewood", "wedgewood polls"})
    )

    if excluded_mask.any():
        raise RuntimeError(
            "Excluded Wedgewood pollster detected in Senate "
            "raw polling input. Quarantine before adjustment."
        )

    required_metadata = {
        "poll_sponsor_type": "",
        "pollster_partisan_affiliation": "",
        "partisan_sponsor_party": "",
        "is_internal_poll": False,
        "partisan_pollster_review_notes": "",
    }

    for col, default in required_metadata.items():
        if col not in df.columns:
            df[col] = default

    if "dem_pct" not in df.columns or "rep_pct" not in df.columns:
        raise ValueError("Senate manual polls must contain dem_pct and rep_pct.")

    df["dem_pct_original"] = pd.to_numeric(df["dem_pct"], errors="coerce")
    df["rep_pct_original"] = pd.to_numeric(df["rep_pct"], errors="coerce")
    df["polling_margin_dem_original"] = df["dem_pct_original"] - df["rep_pct_original"]

    df["partisan_pollster_adjustment_dem"] = 0.0
    df["partisan_pollster_weight_multiplier"] = 1.0
    df["partisan_pollster_adjusted"] = False
    df["partisan_pollster_notes"] = "No partisan pollster adjustment"

    if use_adjustments:
        for idx, row in df.iterrows():
            party = infer_party(row)
            internal = infer_internal(row)

            if party not in ["D", "R"]:
                continue

            raw_adj = internal_adj if internal else default_adj
            raw_adj = min(abs(raw_adj), max_adj)

            # Adjust against the sponsoring party.
            # D sponsor: reduce Dem margin.
            # R sponsor: increase Dem margin.
            adj = -raw_adj if party == "D" else raw_adj

            old_dem = pd.to_numeric(row.get("dem_pct"), errors="coerce")
            old_rep = pd.to_numeric(row.get("rep_pct"), errors="coerce")

            if pd.isna(old_dem) or pd.isna(old_rep):
                continue

            # Preserve two-candidate total while changing margin:
            # D_new - R_new = old_margin + adj
            # D_new + R_new = old_dem + old_rep
            total_major = old_dem + old_rep
            new_margin = (old_dem - old_rep) + adj
            new_dem = (total_major + new_margin) / 2.0
            new_rep = (total_major - new_margin) / 2.0

            df.loc[idx, "dem_pct"] = new_dem
            df.loc[idx, "rep_pct"] = new_rep
            df.loc[idx, "partisan_pollster_adjustment_dem"] = adj
            df.loc[idx, "partisan_pollster_weight_multiplier"] = weight_multiplier
            df.loc[idx, "partisan_pollster_adjusted"] = True
            df.loc[idx, "partisan_pollster_notes"] = (
                f"{party}-sponsored"
                f"{' internal/campaign' if internal else ''} poll: "
                f"applied {adj:+.1f} point Dem-margin adjustment and "
                f"{weight_multiplier:.2f} weight multiplier"
            )

    df["polling_margin_dem_adjusted"] = (
        pd.to_numeric(df["dem_pct"], errors="coerce")
        - pd.to_numeric(df["rep_pct"], errors="coerce")
    )

    # If a weight field exists in raw polls, reduce it. validate_manual_polls may
    # also compute its own weight later, so we preserve an explicit multiplier column.
    for wcol in ["poll_weight", "weight", "manual_poll_weight"]:
        if wcol in df.columns:
            df[f"{wcol}_original"] = pd.to_numeric(df[wcol], errors="coerce").fillna(1.0)
            df[wcol] = df[f"{wcol}_original"] * df["partisan_pollster_weight_multiplier"]

    df.to_csv(OUTPUT, index=False)

    print(f"Wrote {OUTPUT}")
    print()
    print("Partisan pollster adjustment summary")
    print("------------------------------------")
    print(df["partisan_pollster_adjusted"].value_counts(dropna=False).to_string())
    print()
    show_cols = [
        "race",
        "state",
        "pollster",
        "sponsor",
        "poll_sponsor_type",
        "partisan_sponsor_party",
        "is_internal_poll",
        "dem_pct_original",
        "rep_pct_original",
        "dem_pct",
        "rep_pct",
        "polling_margin_dem_original",
        "polling_margin_dem_adjusted",
        "partisan_pollster_adjustment_dem",
        "partisan_pollster_weight_multiplier",
        "partisan_pollster_notes",
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    print(df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
